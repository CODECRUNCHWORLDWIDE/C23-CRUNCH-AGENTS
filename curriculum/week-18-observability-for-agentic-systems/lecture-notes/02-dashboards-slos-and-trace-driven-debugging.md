# Lecture 2 — Dashboards, SLOs, and Trace-Driven Debugging

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can set latency budgets and SLOs for an agent's steps (p50/p95/p99, error budgets), build the three lab dashboards (token usage per route, p95 latency per agent step, retrieval-precision over time) in Langfuse and Phoenix, walk a broken trace to the failing step with a decision tree in under five minutes, replay a production trace through a new prompt version (eval-on-traces), and choose honestly between self-hosted Langfuse, self-hosted Phoenix, and hosted LangSmith.

Lecture 1 was *how to produce traces and what to put on them*. This lecture is *what to do with them* — because a trace you never look at is a log file with extra steps. The mantra still holds, sharpened into a promise:

> **An agentic system without traces is a closed-box. You will eventually re-open it the hard way.** The traces from Lecture 1 are how you re-open it the *easy* way — from a dashboard, in under five minutes, while the user is still waiting.

That five-minutes-from-the-dashboard target is the week's recurring marker, the observability cousin of week 8's "the answer survived the chunking":

> **Found it in the trace in under 5 minutes.**

Everything in this lecture exists to earn that line.

---

## 1. Latency budgets and SLOs

### 1.1 Why averages lie — p50, p95, p99

The first instinct is to track the *average* latency of a step. Don't. Averages hide the pain. If 99 requests take 200ms and one takes 20 seconds, the average is 398ms — a number no real request ever experienced, and which completely hides the 20-second user who is furious. Latency is not normally distributed; it has a long tail, and the tail is where users churn. So you track **percentiles**:

- **p50 (the median)** — half of requests are faster than this. The *typical* experience.
- **p95** — 95% of requests are faster than this. The experience of your *unlucky* users, and the number you usually set SLOs against. One in twenty requests is at least this slow.
- **p99** — 99% are faster. The *tail* — the worst 1%. Where retries, cold caches, and a slow downstream model show up.

Read them together. If p50 is 300ms and p99 is 9 seconds, most calls are snappy but 1% are catastrophic — a tail problem (a flaky dependency, a retry storm), not a "the whole thing is slow" problem. If p50 *and* p99 are both 4 seconds, everything is uniformly slow — a different bug entirely. **The shape of the p50/p95/p99 spread tells you which kind of slow you have**, the same way week 8's Recall@5 curve shape told you which kind of chunking failure you had.

Computing percentiles from span durations is a sort-and-index:

```python
import numpy as np


def percentiles(durations_ms: list[float]) -> dict[str, float]:
    """p50/p95/p99 from a list of span durations (milliseconds)."""
    if not durations_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
    arr = np.asarray(durations_ms, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n": len(arr),
    }
```

You compute these **per agent step**, by grouping spans on `gen_ai.operation.name` (or your `crunch.route` / agent name), exactly as you grouped tokens in Lecture 1 §6. The retriever step has its own p95; the writer's `chat` call has its own p95. A single end-to-end p95 hides *which step* is slow; per-step p95 names it.

### 1.2 SLOs and error budgets

A **latency budget** is the per-step target you commit to: "the retrieve step's p95 must stay under 500ms," "end-to-end p95 under 6 seconds." An **SLO (Service Level Objective)** wraps that in a reliability target over a window: "99% of requests succeed *and* land under the p95 budget, measured over 30 days."

The SLO's twin is the **error budget**: `100% − SLO`. A 99.9% success SLO means a 0.1% error budget — over a month of, say, 1,000,000 requests, you are *allowed* 1,000 failures. This reframes reliability from "zero errors" (impossible, and paralysing) to "spend the budget wisely." While there's budget left, you ship features. When you've burned it, you stop shipping and fix reliability. The budget makes the trade-off explicit and unemotional.

```python
def slo_report(durations_ms, errors: int, total: int, p95_budget_ms: float,
               success_slo: float = 0.99) -> dict:
    """Did this window meet its latency budget and stay inside the error budget?"""
    pcts = percentiles(durations_ms)
    success_rate = (total - errors) / total if total else 1.0
    error_budget = 1.0 - success_slo
    budget_used = (1.0 - success_rate) / error_budget if error_budget else 0.0
    return {
        "p95_ms": pcts["p95"],
        "p95_budget_ms": p95_budget_ms,
        "p95_ok": pcts["p95"] <= p95_budget_ms,
        "success_rate": success_rate,
        "error_budget_used_pct": round(100 * budget_used, 1),
        "slo_met": pcts["p95"] <= p95_budget_ms and success_rate >= success_slo,
    }
```

> **The discipline:** set per-step latency budgets *before* you optimise, and alert on the p95 budget, not the average. "The retrieve step's p95 budget is 500ms" is a commitment you can monitor and page on. "It feels slow sometimes" is not. And track the **burn rate** of your error budget — if you've used 80% of a month's budget on day 3, something regressed; the budget turns a vague "errors are up" into a quantified "we will run out by Thursday at this rate."

---

## 2. The three dashboards

The lab builds exactly three dashboards. Each answers a question you'll actually be asked, and each is a group-by over the spans you instrumented in Lecture 1. Build them in *both* backends (Langfuse and Phoenix) so you see how each renders the same underlying span data.

### 2.1 Dashboard 1 — token usage per route

**The question:** which route costs us the most, and is any route's cost drifting up?

**The data:** every `chat` span carries `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.request.model`, and your `crunch.route`. Group by `crunch.route`, sum the tokens, price them with the §1-Lecture-1 table, plot cost over time.

- **In Langfuse:** Langfuse natively ingests `gen_ai.usage.*` and applies a **model-pricing table** to compute cost per trace. Add `route` as a trace attribute (Langfuse calls these *metadata* / *tags*), then group the Dashboard view by it. Langfuse gives you cost-per-trace and cost-by-metadata out of the box — you're configuring a chart, not computing tokens.
- **In Phoenix:** Phoenix aggregates token usage from the OpenInference span attributes; filter the spans view by your `crunch.route` attribute and read the summed `llm.token_count.*` / `gen_ai.usage.*` totals. Phoenix surfaces token counts per span and lets you aggregate across a filtered set.

The reusable building block is `rollup(spans, "crunch.route")` from Lecture 1 §6 — the dashboards are that rollup, rendered and refreshed.

### 2.2 Dashboard 2 — p95 latency per agent step

**The question:** which step is the slow one, and is its p95 within budget?

**The data:** every span has a duration and a `gen_ai.operation.name` / agent name. Group by step, compute p95 (§1.1), plot p95 per step over time with the budget line drawn on it.

- **In Langfuse:** Langfuse tracks latency per observation (span) and per trace; build a Dashboard chart of **p95 latency grouped by span name** (your agent steps). The budget is a threshold you eyeball or alert on.
- **In Phoenix:** Phoenix shows latency distributions per span kind/name in its traces view; filter to a step and read the p95 of its duration.

This is the dashboard you stare at when "the app feels slow." Per-step p95 turns that into "the writer's `chat` p95 jumped from 2.1s to 7.8s at 14:05" — a fact, with a time, you can act on.

### 2.3 Dashboard 3 — retrieval-precision over time

**The question:** is retrieval quality (week 8's metric, now in production) degrading?

This one is different: precision is not a raw span attribute — it's an **eval result you attach to retrieval spans**. The retriever's span records what it retrieved (the chunk ids, as a span attribute or event); a scorer compares those against ground truth (or an LLM-judge for relevance) and writes a `crunch.retrieval_precision` score onto the span. Then you plot that score's average over time.

- **In Langfuse:** Langfuse has a **Scores** API — attach a numeric score (`retrieval_precision`) to a trace/observation, then chart the score over time in a Dashboard. This is the canonical "online eval metric over time" pattern.
- **In Phoenix:** Phoenix has **evaluations** you attach to spans (`px.Client().log_evaluations(...)` / the evals API); log a `retrieval_precision` eval per retrieval span and chart it.

Why it matters: retrieval quality drifts *silently*. The corpus changes, the index ages, a model swaps — and precision slides without a single error or latency blip. The only way you catch it is a metric on a chart trending down. This dashboard is your week-8 work, kept honest in production. It's also the dashboard the synthetic failure in the challenge can target: inject a retrieval-precision drop, and this is the chart that catches it.

> **All three dashboards are the same move:** a group-by over instrumented spans, rendered. Tokens-per-route groups by `crunch.route` and sums `gen_ai.usage.*`. p95-per-step groups by step and percentiles the duration. Precision-over-time groups by time and averages an attached eval score. Build them once and you have a control panel for cost, latency, and quality — the three things that break in production.

---

## 3. Trace-driven debugging — walk the trace to the failing step

A user reports: "the agent gave a wrong/empty/slow answer." You have the trace. Here is the decision tree to find the failing step — the observability analogue of week 8's extraction-to-retrieval tree, and the procedure that earns "found it in the trace in under 5 minutes."

```
A run produced a bad result (wrong answer, error, or too slow). Open its trace.
│
├─ Look at the ROOT span status. Is the whole trace marked ERROR?
│   ├─ Yes → find the FIRST span with status=ERROR (errors propagate up).
│   │        Read its events: there's an exception event with the stack. That
│   │        span IS the failing step. Done. (§3.1)
│   └─ No  ↓   (no exception — the failure is silent: wrong/empty/slow)
│
├─ Is the complaint SLOW? Read the tree as a flame graph: which span's
│   duration dominates the root's duration?
│   ├─ One span dominates → THAT step is the latency culprit. Check its
│   │   gen_ai.* attrs: huge input_tokens? a retry (two model spans)? a slow
│   │   tool span? Fix the dominant span. (§3.2)
│   └─ Time spread across many → systemic (cold cache, overloaded model);
│       check the per-step p95 dashboard for a regression.
│
├─ Is the complaint WRONG or EMPTY answer? Walk DOWN the tree to the data:
│   ├─ Find the retrieval / tool span. Open its events / output attribute.
│   │   Did it return EMPTY or an error payload the agent swallowed?
│   │   ├─ Empty/garbage retrieval → the generation had nothing to work with.
│   │   │   RETRIEVAL is the failing step (week-8 territory, now in prod).
│   │   └─ Tool returned an error string the agent treated as data →
│   │       the TOOL span is the failing step; the agent didn't check is_error.
│   └─ Retrieval/tool looks fine ↓
│
├─ Open the generation (chat) span's input event (the actual prompt) and
│   output event (the actual completion). Does the prompt contain the right
│   context but the completion ignores it?
│   ├─ Yes → a GENERATION/prompt problem. This is the span to replay through a
│   │        new prompt version (eval-on-traces, §4).
│   └─ No (the prompt is missing context that retrieval DID return) →
│        a WIRING bug: the supervisor didn't pass retrieval output into the
│        writer's prompt. The handoff is the failing step.
│
└─ Still unclear → check context propagation: are tool spans ORPHANED into a
     separate trace? If the tree is missing branches, you're debugging an
     incomplete trace — fix instrumentation first (Lecture 1 §5.3), then re-run.
```

### 3.1 The error case

The fast case. OTel records exceptions as span events with a stack trace, and a span's status goes `ERROR`. A backend draws errored spans in red. So: open the trace, find the *first* red span (errors bubble up, so the root being red just means *something* below it failed), read its exception event. That span is the failing step, and the stack tells you why. Thirty seconds, not five minutes.

### 3.2 The silent cases

The hard cases — no exception, just a bad outcome — are why the tree above walks the *data*, not just the status. A wrong answer with everything green means the failure is in *what the steps did*, not *whether they ran*. The retrieval span returned empty (green — it "succeeded" at returning nothing). The tool returned `{"error": "rate limited"}` as a 200 and the agent fed it to the model as context. The supervisor routed to the wrong agent. None of these throws. All of them are visible the moment you open the span's input/output events and *read what flowed through*. **This is why capturing prompts/completions as span events (Lecture 1 §3.4) matters: the silent bugs are only debuggable if the trace recorded the data, not just the timing.**

> **The five-minute method, distilled:** (1) is it red? → first error span, done. (2) is it slow? → biggest span in the flame graph. (3) is it wrong? → walk down to the retrieval/tool data, then read the generation's actual prompt and completion. Three questions, each one glance at the trace. The trace makes "the agent is broken" answerable; the lack of one makes it a guessing game.

---

## 4. Eval-on-traces — replay a production trace through a new prompt version

Here's a capability that traces unlock and logs never could. A production trace captured the *exact inputs* to a step — the real prompt, the real retrieved context, the real user message (they're in the span events). That means you can **replay that exact input through a *new* prompt version, offline, and compare the new output to the old** — without touching production, and on *real* traffic instead of a synthetic test set.

This is the bridge from week 8's offline eval to production. Week 8 evaluated against a fixed gold set. Eval-on-traces evaluates against *what actually happened* — the prompts users really sent, the contexts retrieval really returned. The replay harness:

```python
import anthropic

client = anthropic.Anthropic()


def replay_generation(span, new_system_prompt: str) -> dict:
    """Replay ONE captured generation span through a new system prompt.

    The span's events hold the real inputs that happened in production:
    the retrieved context and the user message. We swap ONLY the system
    prompt (the variable under test) and re-run — everything else is fixed,
    exactly the one-variable-at-a-time discipline from week 8.
    """
    inputs = extract_messages_from_span(span)   # the real user+context messages
    old_output = extract_completion_from_span(span)

    resp = client.messages.create(
        model=span.attributes["gen_ai.request.model"],  # same model as prod
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=new_system_prompt,                # the ONE thing we changed
        messages=inputs,                         # the REAL production inputs
    )
    new_output = next(b.text for b in resp.content if b.type == "text")
    return {
        "trace_id": span.context.trace_id,
        "old": old_output,
        "new": new_output,
        "changed": old_output.strip() != new_output.strip(),
    }


def replay_suite(spans, new_system_prompt: str) -> dict:
    """Replay a batch of production traces; summarise how the new prompt differs."""
    results = [replay_generation(s, new_system_prompt) for s in spans
               if s.attributes.get("gen_ai.operation.name") == "chat"]
    changed = sum(1 for r in results if r["changed"])
    return {"replayed": len(results), "changed": changed,
            "unchanged": len(results) - changed, "results": results}
```

The workflow: sample N recent production traces, replay each generation span through the candidate prompt, and diff. Now you can answer "does prompt v2 change the answer on the last 200 real requests, and for the better?" — judged by an LLM judge or by the same retrieval/answer metrics from week 8, on *real* inputs. **That is the difference between testing your prompt on examples you made up and testing it on what your users actually do.** Langfuse formalises this as **Datasets** (curate traces into a dataset, run experiments against prompt versions); Phoenix has **experiments** over datasets built from spans. The harness above is the mechanism; the backends give it a UI.

> **The one-variable rule, again.** Replay swaps *only* the prompt and holds the model and inputs fixed — same discipline as week 8's chunking A/B (fix the embedding and store, vary the chunker). Change the prompt *and* the model and you can't attribute the delta. The whole validity of eval-on-traces is that the production inputs are frozen and exactly one thing — the prompt version — moves.

### 4.1 The offline replay harness, end to end

The snippet above is the *call*. A harness wraps it into a repeatable, no-production-impact loop: pull a recorded trace from a file (a JSON export, or spans you captured with an `InMemorySpanExporter` in a test), re-run its generation step through a candidate prompt, and diff both the *output text* and the *metrics* you care about (tokens, latency, an LLM-judge score). Nothing about this touches the live system — that's the point. You replay yesterday's real traffic against tomorrow's prompt, on your laptop.

```python
"""replay_harness.py — offline eval-on-traces: a recorded trace -> a new prompt.

Reads spans from a recorded trace (loaded from disk), re-runs each generation
through a candidate system prompt, and diffs old vs new output AND metrics. Runs
entirely offline against a frozen recording — production never sees it.
"""
import json
import time

import anthropic

client = anthropic.Anthropic()


def load_recorded_spans(path: str) -> list[dict]:
    """Load a recorded trace export. Each span is a dict with 'attributes',
    'events', and a recorded 'output_text' — the shape an OTLP/JSON export gives
    you (or what you dumped from an InMemorySpanExporter in a test)."""
    with open(path) as f:
        return json.load(f)["spans"]


def messages_from_span(span: dict) -> list[dict]:
    """Reconstruct the real production input messages from the span's events.
    The user message and retrieved context were captured as gen_ai.* events
    (Lecture 1 §3.4); we rebuild the messages list exactly as it was sent."""
    msgs = []
    for ev in span.get("events", []):
        if ev["name"] == "gen_ai.system.message":
            continue  # system prompt is the variable under test; we replace it
        if ev["name"] == "gen_ai.user.message":
            msgs.append({"role": "user", "content": ev["attributes"]["content"]})
    return msgs


def replay_one(span: dict, new_system_prompt: str) -> dict:
    """Replay ONE recorded generation span through a candidate system prompt and
    diff both the text and the metrics against what production actually did."""
    attrs = span["attributes"]
    old_text = span["output_text"]
    old_out_tokens = int(attrs.get("gen_ai.usage.output_tokens", 0))

    t0 = time.perf_counter()
    resp = client.messages.create(
        model=attrs["gen_ai.request.model"],     # SAME model as production
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=new_system_prompt,                # the ONE thing we changed
        messages=messages_from_span(span),       # the REAL production inputs
    )
    new_latency_ms = (time.perf_counter() - t0) * 1000
    new_text = "".join(b.text for b in resp.content if b.type == "text")

    return {
        "trace_id": span["trace_id"],
        "text_changed": old_text.strip() != new_text.strip(),
        "old_text": old_text,
        "new_text": new_text,
        # metric diffs — the second half of a real replay, not just the text
        "old_out_tokens": old_out_tokens,
        "new_out_tokens": resp.usage.output_tokens,
        "out_token_delta": resp.usage.output_tokens - old_out_tokens,
        "new_latency_ms": round(new_latency_ms, 1),
    }


def replay_recording(path: str, new_system_prompt: str) -> dict:
    """Replay every generation span in a recorded trace file and summarise."""
    spans = [s for s in load_recorded_spans(path)
             if s["attributes"].get("gen_ai.operation.name") == "chat"]
    results = [replay_one(s, new_system_prompt) for s in spans]
    changed = sum(1 for r in results if r["text_changed"])
    token_delta = sum(r["out_token_delta"] for r in results)
    return {
        "replayed": len(results),
        "text_changed": changed,
        "text_unchanged": len(results) - changed,
        "total_output_token_delta": token_delta,  # did v2 get more verbose?
        "results": results,
    }
```

Two things make this a *harness* and not just the bare call. First, it **diffs metrics, not only text**: `out_token_delta` catches a prompt rewrite that quietly doubled output length (and therefore cost) even when the answer looks fine — a regression invisible to an eyeball diff. Second, it runs against a **frozen recording on disk**, so the same N traces feed every candidate prompt; prompt v2 and prompt v3 are judged on *identical* inputs, which is the only way the comparison is fair. Wire an LLM judge (`claude-opus-4-8`) over `old_text`/`new_text` and you get a quality verdict per trace on top of the token/latency deltas — eval-on-traces, complete.

> **Why a recording, not live replay.** If you replayed against live retrieval, the context would differ between runs (the index changed, a doc was added) and you'd be back to two moving variables. Replaying a *recording* freezes the retrieved context exactly as it was, so the only thing that moves is the prompt. The recording is the time machine that makes the one-variable rule enforceable offline.

---

## 4.5 A worked p50/p95/p99 and error-budget computation

The percentile and SLO code in §1 is abstract until you run real numbers through it. Here is a worked example you can reproduce by hand — exactly the kind of arithmetic the homework asks for, so do it once here slowly.

Suppose you pull the `writer`'s `chat` span durations for a window — **20 requests**, in milliseconds, sorted:

```
180, 190, 195, 200, 205, 210, 215, 220, 230, 240,
250, 260, 270, 285, 300, 320, 360, 410, 600, 9200
```

Nineteen of these are a tight, healthy cluster from 180–600ms. The twentieth — `9200` — is the tail: one request that hit a retry or a cold model and took 9.2 seconds. Now compute the percentiles (NumPy's default linear interpolation, `np.percentile`):

- **p50 (median):** with 20 points, the median is the average of the 10th and 11th sorted values: `(240 + 250) / 2 = 245ms`. The *typical* writer call is a snappy quarter-second.
- **p95:** the 95th-percentile index is `0.95 × (20 − 1) = 18.05`, so interpolate between the 19th value (`600`) and the 20th (`9200`): `600 + 0.05 × (9200 − 600) = 600 + 430 = 1030ms`. One in twenty writer calls is at least ~1 second.
- **p99:** index `0.99 × 19 = 18.81`, between `600` and `9200`: `600 + 0.81 × 8600 = 600 + 6966 = 7566ms`. The worst 1% are catastrophic — 7.5 seconds.

Compare those to the **mean**: `(sum) / 20 ≈ 1146ms`. The mean (1146ms) is *higher than p95* (1030ms) — dragged up entirely by the single 9.2s outlier — and describes *no actual request*. Nineteen of twenty users saw ≤600ms; one saw 9.2s; "average 1.1s" hides both truths. **This is the whole case for percentiles in one table:**

| Statistic | Value | What it tells you |
|---|---:|---|
| mean | 1146ms | A fiction — no request was near it; the outlier poisoned it. |
| p50 | 245ms | The typical experience — fast. |
| p95 | 1030ms | The unlucky-user experience; what you SLO against. |
| p99 | 7566ms | The tail — one bad dependency call, the 1% who suffer. |

The p50/p99 spread (245ms vs 7566ms, a ~31× ratio) screams **tail problem**, not systemic slowness: the system is fast for almost everyone, with a rare catastrophic stall to hunt down (per §3.2, open one of those 9.2s traces and find the dominant span — likely a retry).

Now the **error budget**. Say your SLO is **99.0% success over the window** and your p95 budget is **1500ms**. Take a 30-day window with **1,000,000 requests** and **7,400 failures**:

- **Success rate:** `(1,000,000 − 7,400) / 1,000,000 = 0.9926` → **99.26%**. Above the 99.0% SLO. ✓
- **Error budget (the allowance):** `1 − 0.99 = 0.01` → **10,000 failures** permitted in the window.
- **Budget consumed:** `7,400 / 10,000 = 0.74` → **74% of the error budget burned.**
- **p95 vs budget:** `1030ms ≤ 1500ms` → latency SLO **met.** ✓

So `slo_report(...)` returns `slo_met=True` — but `error_budget_used_pct=74.0` is the number that should make you nervous. You met the SLO *this* window, yet you've spent three-quarters of the month's failure allowance. Plot the **burn rate**: if those 7,400 failures landed in the first 10 days, you're on pace for ~22,000 by day 30 — you'll blow the budget by day ~14. The SLO says "fine today"; the burn rate says "freeze feature work and fix reliability now." That gap — *passing the check while burning the budget too fast* — is exactly what the error-budget framing exists to surface, and why §1.2 insists you watch the burn rate, not just the pass/fail.

> **The reproducible drill:** feed the 20 durations above into `percentiles()` and the failure counts into `slo_report(..., p95_budget_ms=1500, success_slo=0.99)` and confirm you get p50≈245, p95≈1030, p99≈7566, `error_budget_used_pct≈74.0`, `slo_met=True`. If your p95 comes back as the mean (1146) or as a raw max (9200), you computed the wrong statistic — percentiles are an interpolated sort-and-index, not an average and not the worst case.

---

## 5. Self-hosted Langfuse vs Phoenix vs hosted LangSmith — the honest comparison

You'll meet four tools by name. Here is the straight version, oriented to the Phase III milestone (which requires **self-hosted Langfuse + Phoenix**).

| Tool | Hosting | Open source | Best at | The catch |
|---|---|---|---|---|
| **Langfuse** | self-host (Docker) or cloud | yes (MIT-ish core) | product analytics: cost-per-trace, datasets/experiments, prompt management, scores, OTLP ingest | self-hosting is a real service (Postgres + ClickHouse + worker) |
| **Arize Phoenix** | self-host (pip / Docker) or cloud | yes (open) | dev-loop tracing + evals, OpenInference, zero-friction local (`px.launch_app()`) | leaner on long-term product analytics than Langfuse |
| **LangSmith** | hosted (LangChain) | no | deep LangChain/LangGraph integration, polished UI, eval tooling | proprietary + hosted: your traces live on their servers; not self-hostable |
| **Helicone** | self-host or cloud | yes (open) | dead-simple proxy-based logging (point your base_url at it), cost/usage | proxy model is less granular than full span instrumentation |

The decision rules:

- **For this course and the capstone: self-hosted Langfuse + self-hosted Phoenix.** The milestone says so, and the reasons are real — your traces (and any captured prompts) stay inside your trust boundary, there's no per-trace bill, and OTel Gen-AI conventions mean your instrumentation is portable across both. Phoenix is the zero-friction dev loop (`px.launch_app()`, no key); Langfuse is the durable product-analytics layer (cost, datasets, scores).
- **Phoenix for the inner loop, Langfuse for the outer loop.** Phoenix's `px.launch_app()` / `phoenix serve` runs locally with no API key — you see traces *while you code*. Langfuse is where the cost dashboards, the score-over-time charts, and the replay datasets live for the team. They are complements, not competitors; the dual-export trick (Lecture 1 §4) sends to both.
- **LangSmith is excellent — and proprietary and hosted.** If you're all-in on LangChain and fine with your traces on a vendor's servers, it's the smoothest experience. But it's not open and not self-hostable, which is exactly why the milestone (and any data-sovereignty requirement) routes around it. Know it; we don't depend on it.
- **Helicone when you want logging in one line.** Its proxy model — repoint your model `base_url` and it logs everything — is the lowest-effort way to get cost/usage visibility. The trade-off is granularity: a proxy sees requests, not your in-app span tree (the supervisor's routing decision, your `crunch.route`). Great for a quick cost view; not a substitute for OTel instrumentation when you need the *agent* structure.

> **Why OTel is the hedge.** Because you instrument with the *OTel Gen-AI conventions* and not a vendor SDK, your traces flow to Langfuse and Phoenix today and to whatever you adopt tomorrow with a config change, not a rewrite. The conventions are the spine precisely so the tool choice above is reversible. Lock into one vendor's bespoke SDK and it isn't.

---

## 6. Recap

You should now be able to:

- Set **latency budgets and SLOs**: track p50/p95/p99 per agent step (never the average), read the spread to tell tail problems from systemic slowness, and frame reliability as an SLO with an error budget you spend deliberately and watch the burn rate of.
- Build the **three dashboards** — token usage per route (group by `crunch.route`, sum `gen_ai.usage.*`, price it), p95 latency per agent step (group by step, percentile the duration, draw the budget line), and retrieval-precision over time (attach an eval score to retrieval spans, chart it) — in both Langfuse (Dashboards, Scores, model-pricing) and Phoenix (span aggregation, evaluations).
- Run **trace-driven debugging** with the decision tree: red span → first error; slow → biggest span in the flame graph; wrong/empty → walk down to the retrieval/tool data, then read the generation's real prompt and completion — and recognise an orphaned-tool-span trace as an instrumentation bug to fix first. The method that earns "found it in the trace in under 5 minutes."
- Do **eval-on-traces**: replay a captured production trace's exact inputs through a new prompt version, holding model and inputs fixed (one-variable discipline), and diff old vs new on *real* traffic — Langfuse Datasets / Phoenix experiments give it a UI.
- Choose honestly among **Langfuse (self-host, product analytics), Phoenix (self-host, dev-loop + evals, zero-friction local), LangSmith (hosted, proprietary, deep LangChain), and Helicone (open, proxy, one-line)** — and explain why the milestone uses self-hosted Langfuse + Phoenix and why OTel conventions make the choice reversible.

Next: the exercises put this on a real instrumented run — stand up a backend and read a trace by eye, emit correct `gen_ai.*` spans for a simulated agent run, and compute token + latency rollups from recorded spans. Continue to [the exercises](../exercises/README.md).

---

## References

- *OpenTelemetry Gen-AI semantic conventions* (attributes, operations, events): <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
- *Langfuse documentation* (Dashboards, Scores, Datasets/Experiments, OTLP, model pricing, self-hosting): <https://langfuse.com/docs>
- *Arize Phoenix documentation* (tracing, evaluations, `px.launch_app`, experiments): <https://docs.arize.com/phoenix>
- *LangSmith* (hosted tracing + evals for LangChain/LangGraph): <https://docs.smith.langchain.com/>
- *Helicone* (open-source LLM observability via proxy): <https://www.helicone.ai/>
- *Google SRE Book — Service Level Objectives & Error Budgets*: <https://sre.google/sre-book/service-level-objectives/>
- *OpenTelemetry — recording exceptions on spans*: <https://opentelemetry.io/docs/languages/python/instrumentation/#recording-exceptions>
