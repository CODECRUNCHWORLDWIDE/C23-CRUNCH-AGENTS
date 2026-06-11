# Week 18 — Observability for Agentic Systems

Welcome to the week that decides whether you can *operate* the system you spent Phase III building. Through weeks 13, 15, and 17 you assembled a multi-agent stack: a LangGraph supervisor routing to worker agents over an MCP tool surface, with safety rails. It works on your laptop. This week you make it **observable** — so that when it misbehaves in front of a user, you can see exactly which step failed, how many tokens it burned, and how long it took, without adding a single `print`. This is a **Phase III milestone week**: the headline lab is the milestone, and the capstone (weeks 22–24) assumes traces are already flowing.

By Friday you will be able to take any agent run and answer three operational questions from a trace alone — *what did it cost, how slow was it, and where did it break* — and you will treat **a trace as the only artifact that survives the run**. You instrument with the vendor-neutral **OpenTelemetry Gen-AI semantic conventions** so the same spans render in Langfuse, Phoenix, or a Grafana dashboard without a rewrite.

The one sentence to internalize before you read another line:

> **An agentic system without traces is a closed-box. You will eventually re-open it the hard way.**

The whole week is in service of re-opening it the *easy* way — from a dashboard, while the user is still waiting. That target is the week's recurring marker (see "the promise" below): **found it in the trace in under 5 minutes.**

## Learning objectives

By the end of this week, you will be able to:

- **Explain** why an agent run is a *tree* of spans, not a line of logs — and why the interesting failures live *between* steps, where only a trace can see them.
- **Name** the OpenTelemetry **Gen-AI semantic conventions** verbatim — the `gen_ai.*` attributes (`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`/`output_tokens`), the operation types (`chat` / `embeddings` / `execute_tool` / `invoke_agent`), span kinds, and prompts/completions captured as span *events*.
- **Wire** the OTel SDK + an OTLP exporter into an agent, both via **auto-instrumentation** (OpenInference / OpenLLMetry) and by hand — including the **dual-export** trick (one `TracerProvider`, two OTLP processors → Langfuse *and* Phoenix).
- **Roll up** token usage and cost **per route / per user / per model** as a group-by over span attributes, priced with real 2026 Anthropic rates.
- **Set** latency budgets and SLOs: p50/p95/p99 per agent step (never the average), error budgets, and burn rate — and read the percentile spread to tell a tail problem from systemic slowness.
- **Build** the three lab dashboards (token usage per route, p95 latency per agent step, retrieval-precision over time) in both Langfuse and Phoenix.
- **Debug** a broken run with the trace decision tree — red span → first error; slow → biggest span; wrong/empty → walk the data — and earn "found it in the trace in under 5 minutes."
- **Replay** a recorded production trace through a new prompt version (**eval-on-traces**), holding model and inputs fixed, and diff output and metrics on *real* traffic.

## Prerequisites

This week assumes you have completed **C23 Phase III** (or have equivalent fluency). Specifically:

- **Week 13 (LangGraph supervisor)** — you have a supervisor graph routing to worker agents. **This week instruments that system**; if it's broken, fix it first. The trace tree you read all week *is* your week-13 graph's execution.
- **Week 15 (MCP)** — your tools live behind MCP, often in a separate process. Section 5.3 of Lecture 1 propagates trace context across that boundary so tool spans nest in the agent trace instead of orphaning.
- **Week 17 (safety)** — the prompt/response content-capture switch is the observability cousin of the safety surface; you decide deliberately whether message bodies belong in traces.
- **A tracing backend, zero-friction by default.** **Arize Phoenix** is the no-friction default: `pip install arize-phoenix`, then `import phoenix as px; px.launch_app()` gives you a local UI + OTLP collector with **zero API key**. **Self-hosted Langfuse** (via `docker compose`) is the durable product-analytics layer for cost dashboards, scores, and replay datasets. The milestone uses both; every exercise runs on Phoenix-local (or even just `ConsoleSpanExporter`) so you are never blocked on infrastructure.

You do **not** need a GPU. You do **not** need any paid API key for the exercises — Exercise 2 runs on `ConsoleSpanExporter` + `InMemorySpanExporter` with only `opentelemetry-sdk` installed, and Exercise 3 is stdlib + numpy.

## Topics covered

- **Why trace an agent:** the run-is-a-tree argument, failures-live-between-steps, and "you cannot bill, budget, or debug what you didn't record."
- **Spans and traces:** the span/trace/trace-id model, parent/child links, context propagation, and reading a multi-agent run as a flame graph.
- **The OTel Gen-AI conventions:** the `gen_ai.*` attribute namespace, operation types, span kinds, and capturing prompts/completions as span events behind the content-capture switch.
- **Wiring the SDK:** `TracerProvider` + `Resource`, `BatchSpanProcessor`/`SimpleSpanProcessor`, `ConsoleSpanExporter`/`InMemorySpanExporter`/`OTLPSpanExporter`, and the dual-export trick.
- **Auto vs manual instrumentation:** OpenInference / OpenLLMetry for the model/tool calls; manual spans for the application layer (`crunch.route`, `crunch.user_id`); MCP context propagation.
- **Token & cost accounting:** rollups per route / user / model from `gen_ai.usage.*`, priced with 2026 Anthropic rates.
- **Latency SLOs:** p50/p95/p99 per step, error budgets, burn rate, and why averages hide the tail.
- **Trace-driven debugging:** the decision tree from a bad outcome to the failing step in under five minutes.
- **Eval-on-traces:** offline replay of recorded production traces through a new prompt version, diffing output and metrics.
- **Tool landscape:** self-hosted Langfuse vs Phoenix vs hosted LangSmith vs Helicone, and why OTel conventions make the choice reversible.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                       | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Why trace; spans/traces; the OTel Gen-AI conventions        |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Wiring the SDK; auto vs manual; reading a trace             |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Token/cost accounting; SLOs (p50/p95/p99); error budgets    |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | The three dashboards; trace-driven debugging; eval-on-traces|    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The milestone lab: instrument + find the failure            |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work (`crunchobs`)                        |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                   |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                             | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The OTel Gen-AI spec, Langfuse/Phoenix docs, instrumentation libraries, papers, and the glossary cheat-sheet |
| [lecture-notes/01-tracing-and-otel-genai-conventions.md](./lecture-notes/01-tracing-and-otel-genai-conventions.md) | Why trace, the span/trace tree, the `gen_ai.*` conventions, wiring the SDK, instrumenting LangGraph, token accounting |
| [lecture-notes/02-dashboards-slos-and-trace-driven-debugging.md](./lecture-notes/02-dashboards-slos-and-trace-driven-debugging.md) | p50/p95/p99 and error budgets, the three dashboards, the debugging decision tree, eval-on-traces replay, and the tool comparison |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-read-a-trace.md](./exercises/exercise-01-read-a-trace.md) | Stand up Phoenix (or Langfuse), send traced agent calls, and find the slow span by eye |
| [exercises/exercise-02-otel-genai-spans.py](./exercises/exercise-02-otel-genai-spans.py) | Build a nested OTel span tree for a 3-step agent with correct `gen_ai.*` attributes and assert its shape (zero infra) |
| [exercises/exercise-03-token-accounting.py](./exercises/exercise-03-token-accounting.py) | Roll up tokens/cost per route/user/model and compute p50/p95/p99 per step from recorded spans |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge (the Phase III milestone in lab form) |
| [challenges/challenge-01-instrument-and-find-the-failure.md](./challenges/challenge-01-instrument-and-find-the-failure.md) | Instrument the Phase III stack end-to-end, dual-export, build three dashboards, inject one failure, find it in under 5 minutes |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the one-page observability memo |
| [mini-project/README.md](./mini-project/README.md) | The `crunchobs` reusable observability package — tracing wrapper, dual export, accounting, SLO checker, replay tool |

## The "found it in the trace in under 5 minutes" promise

C23 uses a recurring marker for every exercise that ends in observability *actually working* — a real failure located from a dashboard or a trace, fast, while the user is still waiting. It is the observability cousin of week 8's "the answer survived the chunking":

```
$ python find_the_failure.py --trace a1b2c3
opening trace a1b2c3 (root: agent.invoke, 9.4s, status=OK) ...
  flame graph (duration dominates):
    agent.invoke              9.4s
    └─ agent retriever        8.9s   <-- dominates the root
        └─ vector_search      8.7s   gen_ai.operation.name=... db.system=pgvector
  retrieval span returned 0 chunks; generation had nothing to ground on.
  FAILING STEP: retrieval (empty result, swallowed) — found in 00:51.
```

If you cannot point at *which* span, *why*, and *how long it took* from the trace, the system is still a closed box — no matter how green the final answer looked. The point of week 18 is to make the failing step *visible*, and to prove it by finding an injected failure from the dashboard in under five minutes, not by re-running with `print`.

## Stretch goals

If you finish the regular work early and want to push further:

- **Span-link the replay to its source trace.** When eval-on-traces replays a recorded generation, attach an OTel **span link** from the replay span back to the original production span so the two are navigable in the backend. Now "this experiment came from *that* real request" is one click.
- **Sampling.** Add a **tail-based sampler** that keeps 100% of errored/slow traces and 5% of healthy ones, so you keep the interesting traces without paying to store every healthy one. Measure the storage drop.
- **Alert on burn rate, not threshold.** Wire a simple alert that fires when the error-budget *burn rate* (not the raw error count) projects budget exhaustion before the window ends — the multi-window burn-rate alert from the Google SRE book.
- **Cardinality audit.** Grep your spans for high-cardinality attributes accidentally promoted to `gen_ai.*` or used as a group-by (raw prompt text, per-request ids). Each one bloats the index; move them to events. Report what you found.

## Up next

Week 19 takes the observability literacy you built here into **vLLM in Production** — serving open-weight models yourself at scale. The traces don't stop at the API boundary: you instrument the *serving* layer too (queue time, KV-cache hits, batch sizes) and watch the same p95/error-budget discipline applied to a model server you own. Push your `crunchobs` mini-project before you start it; week 19 (and the capstone) assume your agent's spans already flow to Langfuse and Phoenix.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
