# Lecture 2 — Serving the Two Tiers, the Eval Suite, and OTel Tracing

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can stand up a vLLM continuous-batching cluster for the local tier, put LiteLLM in front as an OpenAI-compatible router with a vendor fallback to `claude-opus-4-8` and a cost-tracked easy-vs-hard routing layer, assemble a Ragas + calibrated-LLM-as-judge eval suite over a 100-question gold set that gates the build, and instrument every agent step / tool call / model request with OpenTelemetry Gen-AI semantic conventions exported to Langfuse and Arize Phoenix.

Lecture 1 wired the brain (supervisor) and the hands (MCP tools). This lecture wires the metabolism (serving) and the senses (eval + tracing). The mantra carries over from week 21:

> **The cheapest token is the one you do not generate. The second cheapest is the one a 7B handles instead of a 70B.**

And the one this lecture exists to enforce:

> **If you cannot measure it, you cannot ship it.** An agentic system without a gate is a vibe with a deploy URL.

---

## 1. The two-tier serving stack

The capstone serves from two tiers:

- **Local tier** — a 7B/13B model on a **vLLM** cluster you run. Cheap per token, no per-call vendor charge, full control. Handles the *easy* routes: simple lookups, routing decisions, the critique pass.
- **Vendor tier** — `claude-opus-4-8` via the Anthropic API. Expensive per token, but state-of-the-art on the *hard* routes: synthesizing a nuanced grounded answer, reasoning over a tangle of retrieved clauses.

Both tiers sit behind **LiteLLM**, which presents one OpenAI-compatible surface to the agents. The agents don't know or care which tier served a turn — they call `litellm/chat/completions` with a model name, and LiteLLM routes.

```
 Agents ──▶ LiteLLM proxy ──┬──▶ vLLM cluster (Qwen2.5-7B/14B)   [easy routes]
 (OpenAI-compatible)        └──▶ Anthropic (claude-opus-4-8)      [hard routes]
                                  with fallback + cost tracking
```

### 1.1 vLLM — the local tier

vLLM's headline feature is **continuous batching**: instead of waiting for a batch to fill, the scheduler interleaves requests at the token level, so a new request joins the in-flight batch immediately and a finished one frees its slot at once. That's the throughput multiplier that makes self-hosting cheaper than per-call vendor pricing once you have steady traffic.

Bring it up with the OpenAI-compatible server:

```bash
# On a rented L4/A10 (7B) or H100 (14B). ~$0.50/h for the 7B tier.
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching        # reuse the shared system-prompt prefix
```

`--enable-prefix-caching` is the serving-side analog of prompt caching: the retrieval-agent and critique-agent share a big stable system prompt, and prefix caching means vLLM computes it once and reuses the KV cache across requests. Free throughput on a workload with shared prefixes — which an agentic system is.

> **No GPU?** Run the local tier on Ollama instead: `ollama serve` + `ollama pull qwen2.5:7b`. LiteLLM fronts Ollama with the same OpenAI-compatible surface, so the wiring is identical — only the throughput differs. The integration still runs end-to-end; the chaos drill's "kill a replica" scenario is just less dramatic.

### 1.2 LiteLLM — the router with fallback and cost tracking

LiteLLM's config lists both tiers, sets up a fallback (so a hard route degrades to a different model rather than failing), and tracks per-key cost. The fallback is what makes the local tier *and* the vendor tier resilient — and it's the exact mechanism the week-24 chaos drill exercises when you kill a vLLM replica.

```yaml
# litellm_config.yaml
model_list:
  - model_name: local-fast            # easy routes land here
    litellm_params:
      model: openai/Qwen/Qwen2.5-7B-Instruct
      api_base: http://localhost:8001/v1
      api_key: "not-needed"
  - model_name: vendor-hard           # hard routes land here
    litellm_params:
      model: anthropic/claude-opus-4-8
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  # If the local tier is down, a request for local-fast fails over to vendor-hard.
  # If the vendor is rate-limited, it can fail back to local-fast.
  fallbacks:
    - local-fast: ["vendor-hard"]
    - vendor-hard: ["local-fast"]
  # Per-request usage is tracked so the cost report (capstone deliverable 5) is real.
  success_callback: ["langfuse"]      # spans flow to Langfuse too
```

```bash
litellm --config litellm_config.yaml --port 4000
```

Now every agent calls `http://localhost:4000` with `model="local-fast"` or `model="vendor-hard"`, and LiteLLM handles routing, fallback, retries, and cost accounting. The cost accounting is not optional — the capstone's deliverable 5 is a cost report (median / p95 / p99 per query, broken down by route), and LiteLLM's per-request usage is where those numbers come from.

### 1.3 The easy-vs-hard classifier

The decision of *which* tier a route uses is itself a small model call (or a heuristic). The cheap version: a `claude-haiku-4-5` or local-7B classifier that labels a query `easy` or `hard`; the easy ones go to `local-fast`, the hard ones to `vendor-hard`. You built the pattern in week 21; here it's wired into the supervisor so the *writing* route in particular can escalate to the vendor when the question is gnarly.

```python
def route_model(query: str) -> str:
    """Return the LiteLLM model name for this query's difficulty."""
    label = classify_difficulty(query)      # 'easy' | 'hard' (week-21 classifier)
    return "vendor-hard" if label == "hard" else "local-fast"
```

The win you measure: cost reduction versus a vendor-only baseline, plus the cache-hit rate of the semantic cache (week 21) if you wire one. Exercise 3 builds this classifier and the cost accounting.

---

## 2. The eval suite — Ragas + a calibrated judge

A capstone without an eval is a demo. The suite has two halves: **Ragas** (automatic, retrieval-and-answer metrics) and a **calibrated LLM-as-judge** (a model scoring answers, anchored to human labels).

### 2.1 Ragas — the four metrics

Over the 100-question gold set, Ragas computes four numbers. Each tells you about a *different* layer, which is what makes them useful for triage:

- **Faithfulness** — is the answer grounded in the retrieved context? Low faithfulness = the writing-agent is confabulating. This is the headline gate metric.
- **Answer relevancy** — does the answer actually address the question? Low = the writing-agent wandered.
- **Context precision** — of the chunks retrieved, how many were relevant? Low = the retrieval-agent's `k` is too high or the reranker is weak.
- **Context recall** — of the relevant chunks, how many were retrieved? Low = the retrieval is *missing* material; the answer can't be grounded in what was never fetched.

The split between the two context metrics is the diagnostic. Low precision but high recall → you're retrieving the right stuff plus noise (tighten `k`, lean on the reranker). Low recall → you're missing material (the retriever or chunking is the problem, not the generator). This is how you turn a red gate into a specific fix.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall,
)
from datasets import Dataset

# Each row: {question, answer, contexts, ground_truth}
# question/ground_truth from the gold set; answer/contexts from a full system run.
ds = Dataset.from_list(rows)

result = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)   # {'faithfulness': 0.91, 'answer_relevancy': 0.90,
                #  'context_precision': 0.84, 'context_recall': 0.88}
```

Ragas uses an LLM under the hood to compute these; point it at `claude-opus-4-8` (or a local model for a cheaper, noisier run). The 100 questions cost real tokens — run it deliberately, not on every save.

### 2.2 The calibrated LLM-as-judge

Ragas is automatic but coarse. For a sharper "is this a *good* answer?" signal you add a **calibrated** LLM-as-judge: a model that scores each answer 1–5, where the scale is *anchored to human labels* so the model's "4" means what a human's "4" means.

Calibration is the part beginners skip and seniors insist on. Without it, an LLM-judge is a vibe wearing a number. The recipe:

1. Hand-label 10 answers yourself (the calibration set), 1–5, with a one-line reason each.
2. Put those 10 labeled examples *in the judge's prompt* as the rubric anchor.
3. Run the judge over the other 50; its scores are now interpretable against your labels.
4. Spot-check: re-score a few of the 10 with the judge and confirm it agrees with your labels. If it doesn't, your rubric is ambiguous — fix the rubric, not the judge.

```python
from anthropic import Anthropic

client = Anthropic()

JUDGE_SYSTEM = """You score research-assistant answers 1-5 for groundedness and
helpfulness, using EXACTLY this rubric, anchored to the calibration examples below.
5 = fully grounded in context, directly answers, no unsupported claims.
3 = mostly grounded, answers the question, one minor unsupported aside.
1 = ungrounded or off-topic.

Calibration examples (human-labeled — match this standard):
{calibration_block}

Output only a JSON object {{"score": <1-5>, "reason": "<one line>"}}."""


def judge(question, answer, contexts, calibration_block) -> dict:
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "medium",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        system=JUDGE_SYSTEM.format(calibration_block=calibration_block),
        messages=[{"role": "user", "content":
                   f"Q: {question}\nContext: {contexts}\nAnswer: {answer}"}],
    )
    import json
    return json.loads(next(b.text for b in resp.content if b.type == "text"))
```

Same Anthropic surface notes as the supervisor: `claude-opus-4-8`, adaptive thinking, `effort` for depth, structured output for the score — never parse "I'd give this a 4 out of 5" out of prose.

### 2.3 The gate

The suite *gates* the build. A CI step runs the full eval and fails if any metric is below threshold:

```python
def gate(ragas_scores: dict, judge_mean: float) -> bool:
    thresholds = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.80,
    }
    failures = [m for m, t in thresholds.items() if ragas_scores[m] < t]
    if judge_mean < 0.80:           # judge mean normalized to [0,1]
        failures.append("judge")
    if failures:
        print(f"GATE: FAIL — below threshold: {', '.join(failures)}")
        return False
    print("GATE: PASS")
    return True
```

The gate is what makes "green" mean something. Wire it so the capstone's CI runs it; a red gate blocks the ship, and the failing metric tells you *which layer* to fix. This is the difference between the rubric's "meets" and a fail-vibes-only submission.

---

## 3. OpenTelemetry Gen-AI tracing

> **An agentic system without traces is a closed-box. You will eventually re-open it the hard way.** (Week 18; the capstone is where the trace is the only thing standing between you and a print-statement bisection at 2 AM.)

Every agent step, tool call, and model request becomes a **span** following the **OpenTelemetry Gen-AI semantic conventions** — the cross-vendor standard so that Langfuse *and* Arize Phoenix both understand the same trace. The conventions name the attributes: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, plus tool-call attributes. You add a few capstone-specific ones (`gen_ai.route`, `gen_ai.abort_reason`) so a misroute or a budget abort is queryable.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
# Export to BOTH backends — Langfuse and Phoenix both speak OTLP.
provider.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://localhost:3000/api/public/otel/v1/traces")))  # Langfuse
provider.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")))                  # Phoenix
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("capstone")


def writing_agent(state):
    with tracer.start_as_current_span("writing_agent") as span:
        span.set_attribute("gen_ai.system", "anthropic")
        span.set_attribute("gen_ai.route", state["next"])
        model = route_model(state["query"])
        span.set_attribute("gen_ai.request.model", model)
        resp = call_model(model, build_prompt(state))
        span.set_attribute("gen_ai.usage.input_tokens", resp.usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", resp.usage.output_tokens)
        state["draft"] = resp.text
        return state
```

With this in place, the promise-format line from the README is *real* — `trace: langfuse/trace/3f9c... phoenix/span/aa12...` are clickable links to a span tree showing the supervisor's routing reason, each subordinate's tool calls, the model that served each turn, and the token cost. When the eval gate goes red, you open the trace for a failing question and *see* the bad route or the missing retrieval. That's the half-hour-to-thirty-seconds difference the corollary promised.

Two backends, not one, because they have different strengths: **Langfuse** is your prompt-management + trace + cost home; **Arize Phoenix** is the one you lean on next week for *eval-in-prod* (running evals over production traces, shadow traffic). Export to both now so next week's chaos drill has the data it needs.

A practical note on span hygiene: keep the span tree *shallow and meaningful*. A common beginner mistake is wrapping every tiny function in a span, producing a trace with three hundred nested spans that nobody can read. The right granularity is one span per *meaningful unit of work* — the supervisor decision, each subordinate agent run, each MCP tool call, each model request. That's a span tree a human can scan in seconds and find the slow step or the failed tool. The trace is only useful if it's *readable*; over-instrumenting defeats the purpose as surely as under-instrumenting. Name the spans after the work (`writing_agent`, `corpus.search`, `model_request`), not the function, so the trace reads like the system's story rather than its call stack.

---

## 4. Putting it together — the gated end-to-end run

The thin slice from Lecture 1, now fully instrumented and served, produces the promise:

```
$ python -m capstone.eval run --gold gold/eval_100.jsonl --gate
[supervisor] q07 "What indemnity cap applies to data-breach claims?"
  route=retrieval -> retrieval_agent -> corpus.search (mcp/stdio) -> 4 chunks
  route=write     -> writing_agent   -> draft (claude-opus-4-8, hard route)
  route=critique  -> critique_agent  -> grounded ✓
  trace: langfuse/trace/3f9c...  phoenix/span/aa12...
ragas: faithfulness=0.91 context_recall=0.88 context_precision=0.84 answer_relevancy=0.90
judge (calibrated): 0.87 on 50-q subset
GATE: PASS (faithfulness 0.91 >= 0.85, judge 0.87 >= 0.80)
```

Each piece of that output is a thing this lecture built: the route is the supervisor's decision (L1), the `(claude-opus-4-8, hard route)` is the LiteLLM router's choice, the trace links are the OTel exporters, the ragas line is the Ragas suite, the judge line is the calibrated judge, and `GATE: PASS` is the threshold check. When it prints `FAIL`, the trace link is how you find the culprit.

### 4.1 The cost report — turning LiteLLM's accounting into a deliverable

The capstone spec asks for a cost report: total cost per query at the median, p95, and p99, broken down by route, with cache-hit accounting. That report is not a separate measurement system — it's a view over the per-request usage LiteLLM already tracks. Every served request carries `input_tokens` and `output_tokens`; multiply by the per-model rate (local-fast's amortized self-host rate vs the vendor's published rate), and you have per-query cost. Aggregate across the 100-question run and you have the distribution.

```python
def cost_report(per_query_costs: list[tuple[str, float]]) -> dict:
    """per_query_costs: [(route, dollars)]. Returns median/p95/p99 + split."""
    costs = sorted(c for _, c in per_query_costs)
    n = len(costs)
    local = [c for route, c in per_query_costs if route == "local-fast"]
    return {
        "median": costs[n // 2],
        "p95": costs[int(0.95 * (n - 1))],
        "p99": costs[int(0.99 * (n - 1))],
        "local_fraction": len(local) / n,
        "total": sum(costs),
        "vendor_only_baseline": sum(vendor_cost(q) for q, _ in per_query_costs),
    }
```

The number that makes the report *mean* something is the savings versus a vendor-only baseline — the dollars you saved by routing easy queries local. If that number is small (or negative), your classifier is mis-labeling everything hard, or the corpus questions are genuinely all hard; either way the report tells you. The p99 matters because a single hard query that escalates to the vendor with a long context can dominate the tail; reporting only the median hides that. A senior cost report shows the *distribution*, not just the average, because the average is what you budget and the p99 is what surprises you.

### 4.2 Offline gate now, eval-in-prod next week

A word on what this eval suite *is* and *isn't*. The gate you build this week is an **offline** eval: it scores the system against a fixed 100-question gold set. That's the right tool for "don't ship an obviously broken build" — it's reproducible, it's cheap to re-run, and it gates CI. But it is *not* the same as knowing the system is good on *real* traffic, because the gold set is static and production isn't. Next week (eval-in-prod) you'll close that gap — replaying production traces through candidate versions, shadow traffic, an online judge on live answers. For now, internalize the distinction: a green offline gate is necessary but not sufficient. It means "good on the gold set," which is a real and useful claim, but it is a *weaker* claim than "good in production." Build the offline gate well this week; you'll layer the online evals on top of it next week, and the two together are what a serious agentic product runs.

This is also why you export traces to Phoenix specifically, not just Langfuse: Phoenix is the backend you'll point the eval-in-prod machinery at next week. Wiring it now means next week's chaos drill and eval-in-prod work have the trace data they need from day one, rather than starting from an empty backend.

---

## 5. What you can do now

You can:

- Stand up a vLLM continuous-batching cluster for the local tier (with prefix caching) and explain why continuous batching makes self-hosting economical.
- Put LiteLLM in front as an OpenAI-compatible router with a fallback that degrades a dead local tier to the vendor — the exact mechanism next week's chaos drill exercises.
- Route easy queries local and hard queries to `claude-opus-4-8`, and account for per-request cost so the cost report is real numbers.
- Run a Ragas suite over the 100-question gold set and use the precision/recall split to diagnose whether a failure is retrieval or generation.
- Calibrate an LLM-as-judge against 10 human labels and explain why an uncalibrated judge is a vibe with a number.
- Gate the build on the eval and explain why a red gate blocks the ship.
- Instrument every step with OTel Gen-AI spans exported to both Langfuse and Phoenix, and read a trace to find a misroute or budget abort in thirty seconds.

That is the whole of Sprint B: a supervisor routing to four agents (L1), an MCP tool surface (L1), a two-tier served backend, a gated eval, and full tracing (L2). The mini-project this week is to *ship* it — a live URL or a `docker compose up` image. And next week, in a controlled four-hour window, you find out whether the budgets, the fallbacks, the tool defenses, and the eval-on-traces you built actually hold when you break the system on purpose.

A final scoping reminder, tying back to Lecture 1's title:

- **Get the thin slice green first.** One query through every layer, scored and traced, before you deepen anything.
- **Deepen by what the eval flags**, not by what feels unfinished. Low context recall → the retriever; low faithfulness → the writing prompt; a disagreeing judge → the calibration.
- **Write the cut list.** What you shipped, what you dropped, and why — the senior move made legible.
- **Build it to be broken.** Every fallback, budget, tool defense, and trace is a thing next week's chaos drill will test. The system you ship is the system you'll attack.

Ship the measured, served, observable system — not the pile of brilliant parts. The last 10% is 90% of the engineering, and the way you get there is by triaging with the eval gate and the trace, one route at a time.

---

*If you find errors in this material, please open an issue or send a PR.*
