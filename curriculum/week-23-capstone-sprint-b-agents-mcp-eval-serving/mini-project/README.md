# Mini-Project — Ship the Production Agentic Research Assistant (Sprint B)

> Finish the capstone. Take the retrieval and memory you built in Sprint A and make it a *system*: a LangGraph supervisor routing to a retrieval-agent, code-agent, writing-agent, and critique-agent; an MCP tool surface live over stdio and streamable HTTP; a vLLM cluster serving the local tier behind a LiteLLM router with a vendor fallback to `claude-opus-4-8`; a full eval suite (Ragas + calibrated LLM-as-judge) green on a 100-question gold set; and OpenTelemetry Gen-AI tracing flowing to Langfuse and Phoenix. The deliverable is a live URL or a `docker compose up`-runnable image.

This is the culmination of C23. Eighteen weeks of components — tokenizers, prompts, the agent loop, local inference, embeddings, chunking, reranking, vector stores, memory, multimodal eval, graphs, MCP, fine-tuning, safety, observability, vLLM, NeMo, cost routing — converge here into one runnable, measured, traced system. Sprint A (week 22) landed the foundation; Sprint B ships the assistant. Next week (the chaos drill) you break it on purpose and write the postmortem. This week you build the thing that has to survive that.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** The system you ship here is the *exact* system week 24 attacks. The chaos drill kills a vLLM replica (your LiteLLM fallback had better work), injects a prompt-injection attack through a retrieved document (your MCP tool defenses had better hold), and corrupts 5% of the vector store (your Ragas-faithfulness measurement had better catch it, and your backup had better restore it). Every defense, budget, fallback, and trace you build this week is a thing you'll test under fire next week. Build it to be broken — because it will be.

---

## What you will build

The Production Agentic Research Assistant, as specified in the capstone spec, with seven things wired together:

1. **The supervisor graph** (`capstone/graph.py`) — a LangGraph state machine where the supervisor *routes* a research turn to one of four subordinate agents, with conditional edges, a SQLite checkpointer for resumability, and per-route step/token/time/cost budgets.
2. **The four subordinate agents** (`capstone/agents/`) — retrieval (calls the corpus MCP tool + Sprint A hybrid retrieval), code (calls the calculator MCP tool), writing (synthesizes a grounded answer via the routed model), critique (checks the draft is grounded before it ships).
3. **The MCP tool surface** (`capstone/mcp/`) — four servers (filesystem, calculator, web-fetch, corpus-search) over stdio and streamable HTTP, each hardened (argument validation, path-traversal defense, rate limiting).
4. **The serving stack** (`capstone/serving/`) — vLLM (or Ollama) for the local 7B/13B tier, LiteLLM in front with the vendor fallback to `claude-opus-4-8`, and the easy-vs-hard classifier for cost-tracked routing.
5. **The eval suite** (`capstone/eval/`) — Ragas (four metrics) over the 100-question gold set, a calibrated LLM-as-judge on a 50-question subset, and the gate.
6. **The tracing** (`capstone/telemetry.py`) — OTel Gen-AI spans on every step, exported to Langfuse and Phoenix.
7. **The deploy** (`docker-compose.yml` + README) — one command brings the whole stack up; one sample query runs end-to-end.

By the end you have a public repo that a reviewer can `docker compose up` and a `5-minute video` can walk through (the video and postmortem are week-24/final deliverables; the *system* ships now).

---

## Why one system and not seven scripts

You could run each piece as a standalone script for the demo. Don't — not as the artifact. A single composed system gives you:

- **One trace per query.** A reviewer opens one trace and sees the supervisor route, the tool calls, the serving model, and the cost — the whole story of one query in one span tree. Seven scripts give you seven disconnected logs.
- **One gate.** `capstone.eval run --gate` runs the *whole* system over the gold set and prints `PASS`/`FAIL`. That's the measurement the rubric grades. Seven scripts can't be gated as a system.
- **One deploy.** `docker compose up` is the runnable artifact the spec requires. Seven scripts are not a deploy.

The senior-shop convention in 2026 is a composed, traced, gated system with a one-command deploy. That's what you ship.

---

## Package layout

```
capstone/
├── docker-compose.yml          # supervisor + MCP servers + LiteLLM + vLLM/Ollama + Langfuse + Phoenix
├── litellm_config.yaml         # the two-tier router with vendor fallback
├── README.md                   # the one command, env vars, sample query, architecture diagram
├── gold/
│   └── eval_100.jsonl          # 100 questions: {"query","answer","relevant_doc_ids"}
├── capstone/
│   ├── __init__.py
│   ├── graph.py                # the supervisor state graph + checkpointer + budgets
│   ├── supervisor.py           # the router decision (structured, claude-opus-4-8)
│   ├── telemetry.py            # OTel Gen-AI spans -> Langfuse + Phoenix
│   ├── agents/
│   │   ├── retrieval.py        # calls corpus MCP + Sprint A hybrid_search
│   │   ├── code.py             # calls the calculator MCP tool
│   │   ├── writing.py          # synthesizes a grounded answer via the routed model
│   │   └── critique.py         # checks the draft is grounded
│   ├── mcp/
│   │   ├── corpus_server.py    # the custom corpus-search server (stdio + HTTP)
│   │   ├── fs_server.py        # filesystem-read, path-traversal-hardened
│   │   ├── calc_server.py      # calculator / Python
│   │   └── web_server.py       # web-fetch, rate-limited
│   ├── serving/
│   │   └── router.py           # easy-vs-hard classifier + cost accounting
│   └── eval/
│       ├── run.py              # full-system run over the gold set + the gate
│       ├── ragas_suite.py      # the four Ragas metrics
│       └── judge.py            # the calibrated LLM-as-judge
└── tests/
    ├── test_supervisor.py      # routing decisions are valid enum values
    ├── test_mcp_security.py    # path traversal + bad args are rejected
    └── test_gate.py            # the gate fails on a synthetic below-threshold run
```

Your Sprint A package is a dependency; the retrieval pipeline and memory tiers are imported **unchanged**.

---

## Deliverable 1 — the supervisor graph (the control structure)

The heart of the system. The supervisor routes; the subordinates do. Reuse Exercise 1's router and budget; wire the four real subordinates.

```python
# capstone/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from capstone.supervisor import supervisor_node          # router + budget check
from capstone.agents import retrieval, code, writing, critique


def build_app(checkpoint_path: str = "capstone_checkpoints.sqlite"):
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("retrieval", retrieval.run)
    g.add_node("code", code.run)
    g.add_node("writing", writing.run)
    g.add_node("critique", critique.run)

    g.add_edge(START, "supervisor")
    for agent in ("retrieval", "code", "writing", "critique"):
        g.add_edge(agent, "supervisor")           # all return to the supervisor
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {"retrieval": "retrieval", "code": "code", "writing": "writing",
         "critique": "critique", "finish": END},
    )
    saver = SqliteSaver.from_conn_string(checkpoint_path).__enter__()
    return g.compile(checkpointer=saver)
```

The non-negotiables `graph.py` enforces:

- **The supervisor routes, it does not do.** Its prompt forbids retrieval/computation/writing.
- **Every route is budgeted.** The budget check runs at the top of `supervisor_node`; a runaway aborts cleanly.
- **The graph resumes.** The SQLite checkpointer is wired; a `thread_id` per research run survives a crash.

---

## Deliverable 2 — the MCP tool surface (the hands)

Four servers, each hardened. The corpus server wraps Sprint A's `hybrid_search`; the filesystem server defends against path traversal; the web-fetch server rate-limits; every tool validates its arguments before acting.

```python
# capstone/agents/retrieval.py — the retrieval-agent calls the corpus MCP tool
from capstone.telemetry import tracer
from capstone.mcp.client import call_corpus_search       # MCP stdio client


def run(state):
    with tracer.start_as_current_span("retrieval_agent") as span:
        span.set_attribute("gen_ai.route", "retrieval")
        chunks = call_corpus_search(state["query"], k=5)   # over MCP, not a direct call
        span.set_attribute("retrieval.k", len(chunks))
        state["contexts"] = chunks
        return state
```

The retrieval-agent does **not** call `hybrid_search` directly — it calls it *through* the corpus MCP server. That indirection is the point: the same tool surface works from the agent, from Claude Desktop, and from Cursor, and it's the surface week-24's prompt-injection drill attacks.

---

## Deliverable 3 — the serving stack (the metabolism)

vLLM (or Ollama) for the local tier, LiteLLM with the vendor fallback, the easy-vs-hard classifier. The writing-agent's model is chosen by the classifier; the fallback is what makes the local tier resilient (and is the exact mechanism week-24 tests).

```yaml
# litellm_config.yaml
model_list:
  - model_name: local-fast
    litellm_params:
      model: openai/Qwen/Qwen2.5-7B-Instruct
      api_base: http://vllm:8001/v1
  - model_name: vendor-hard
    litellm_params:
      model: anthropic/claude-opus-4-8
      api_key: os.environ/ANTHROPIC_API_KEY
litellm_settings:
  fallbacks:
    - local-fast: ["vendor-hard"]      # dead local tier -> vendor (week-24 failover)
    - vendor-hard: ["local-fast"]
  success_callback: ["langfuse"]
```

---

## Deliverable 4 — the eval suite and the gate (the senses)

Ragas over the 100-question gold set, a calibrated judge on a 50-question subset, the gate. This is the measurement the rubric grades; a red gate blocks the ship.

```python
# capstone/eval/run.py — the full-system run + the gate
def main(gold_path: str, gate: bool):
    rows = []
    app = build_app()
    for q in load_gold(gold_path):                  # 100 questions
        result = app.invoke({"query": q["query"], "budget": Budget()},
                            {"configurable": {"thread_id": q["id"]}})
        rows.append({"question": q["query"], "answer": result["answer"],
                     "contexts": result["contexts"], "ground_truth": q["answer"]})
    ragas_scores = run_ragas(rows)                  # four metrics
    judge_mean = run_calibrated_judge(rows[:50])    # 50-question subset
    print_scores(ragas_scores, judge_mean)
    if gate and not gate_check(ragas_scores, judge_mean):
        raise SystemExit(1)                         # red gate fails the build
```

---

## Rules

- **You may** reuse everything from weeks 1–22 and the exercises: the supervisor router, the corpus MCP server, the cost-tracked router, the Sprint A retrieval and memory.
- **You must not** let the supervisor do the work — it routes; the subordinates do. A supervisor that retrieves and writes has collapsed the pattern.
- **You must not** ship behind a red gate. The eval gate is the measurement; `PASS` on the 100-question set is the bar.
- **You must not** route everything to the vendor. The local tier must serve some easy routes — this is the cost-engineered system the syllabus specifies, and the cost report proves it.
- **You must** harden every MCP tool (validation, path-traversal defense, rate limiting) — week 24 attacks exactly these.
- Python 3.12, `langgraph`, `mcp`, `litellm`, `vllm` (or Ollama), `ragas`, `anthropic`, `opentelemetry-sdk`, plus `pytest`.
- The vendor tier and judge use `claude-opus-4-8` with `thinking={"type":"adaptive"}` and `output_config={"effort":...}` — never `budget_tokens` or `temperature` (both 400).

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-capstone-<yourhandle>` (or your Sprint A repo, extended).
- [ ] `docker compose up` brings up the whole stack: supervisor, MCP servers, LiteLLM, vLLM/Ollama, Langfuse, Phoenix.
- [ ] `python -m capstone.eval run --gold gold/eval_100.jsonl --gate` runs all 100 questions and prints `GATE: PASS`.
- [ ] The supervisor routes to four subordinate agents; the trace shows the routing chain per query.
- [ ] The retrieval-agent calls the corpus MCP server (over a transport, not a direct function call); the corpus server is also reachable over streamable HTTP.
- [ ] An easy query is served by the local tier and a hard query by `claude-opus-4-8`, visible in the trace.
- [ ] Every MCP tool is hardened; `pytest tests/test_mcp_security.py` proves path traversal and bad args are rejected.
- [ ] The judge is calibrated (10 human labels) and spot-checked.
- [ ] One query's trace is readable in both Langfuse and Phoenix.
- [ ] `README.md` documents the one command, the env vars, the architecture (Mermaid diagram from Sprint A, kept in sync), and a sample query.
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Supervisor & agents** | 20 | Supervisor routes (doesn't do); four subordinate agents with real handoff; budgets abort runaways; checkpointer resumes after a crash. |
| **MCP tool surface** | 20 | Four servers; corpus server wraps Sprint A retrieval; stdio + streamable HTTP; every tool hardened (validation, traversal defense, rate limit); security tests pass. |
| **Two-tier serving** | 15 | vLLM/Ollama local tier; LiteLLM with working vendor fallback; easy-vs-hard routing; cost report with real numbers and the savings vs vendor-only. |
| **Eval & gate** | 20 | Ragas four metrics on 100 questions; calibrated judge on 50; the gate fails the build on red; the calibration is real (10 labels, spot-checked). |
| **Tracing** | 15 | OTel Gen-AI spans on every step; exported to Langfuse + Phoenix; one query's trace is readable end-to-end in both. |
| **Deploy & docs** | 10 | `docker compose up` runs the stack; README documents the command, env vars, architecture diagram, and a sample query; no secrets committed. |

**90+** is portfolio-grade and ready for the week-24 chaos drill. **70–89** ships but has a soft spot (an unhardened tool, an uncalibrated judge, a vendor-only route). **Below 70** means the system isn't measured, served, or observable enough to survive next week — fix that first, because week 24 *will* find the soft spot.

---

## Stretch goals

- **Adaptive-thinking routing.** Run the supervisor's routing decision with `claude-opus-4-8` adaptive thinking and compare route accuracy on a 25-query routing gold set against a no-thinking baseline. Does the thinking pay for itself?
- **Semantic cache on the vendor route.** Add a pgvector semantic cache (week 21) in front of the vendor tier with a 0.92 cosine threshold; chart the cache-hit rate over the 100-question run and the extra cost savings.
- **Prompt caching on the writing-agent.** Cache the stable grounded-answer system prompt; verify `cache_read_input_tokens > 0` on the second query and chart the per-query cost delta.
- **Cross-client corpus tool.** Expose the corpus MCP server over streamable HTTP behind LiteLLM auth and consume it from Claude Desktop or Cursor — proof the surface is genuinely cross-client.

---

## How this connects to the rest of C23

- **Sprint A (week 22)** gave you the retrieval pipeline and the memory tiers; this sprint imports them unchanged and wires the agents that *use* them.
- **Weeks 13 / 15 / 18 / 19 / 21 / 12** gave you LangGraph, MCP, OTel tracing, vLLM/LiteLLM, cost routing, and Ragas; this sprint makes all of them load-bearing in one system.
- **Week 24 (the chaos drill)** attacks *this exact system*: it kills a vLLM replica (tests your LiteLLM fallback), injects a prompt-injection through a retrieved document (tests your MCP tool defenses), and corrupts the vector store (tests your Ragas measurement and your backup). The final capstone deliverable is this system plus the chaos-drill postmortem plus the 5-minute video.

When you've finished, push the repo, confirm `docker compose up` works on a clean checkout, and take the [quiz](../quiz.md). Then get ready to break it.
