# Week 18 Homework

Six problems that revisit the week's topics and force observability literacy into your fingers. The full set should take about **5 hours**. Work in your Week 18 Git repository (the same workspace as the exercises and the `crunchobs` mini-project) so every problem produces at least one commit you can point to at the Phase III milestone review.

The headline deliverable is **Problem 6 — the one-page observability memo**. Treat it as the artifact an on-call engineer reads at 2 AM, not a journal entry.

Have your **Phase III multi-agent system** (the week-13 LangGraph supervisor + week-15 MCP tools) importable, and a trace backend reachable: Phoenix is the no-friction default (`pip install arize-phoenix`; `import phoenix as px; px.launch_app()` — no API key), with self-hosted Langfuse via `docker compose` as the alternative. If the agent is broken, fix it first — this week instruments it.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Instrument a run and read its trace

**Problem statement.** Take one agent run (your supervisor answering one question through at least two sub-steps). Instrument it with OpenTelemetry — auto-instrument the framework with OpenInference *and* add one manual span for a domain step (e.g. a retrieval-precision check). Export to Phoenix (or Langfuse). Produce `notes/week-18/first-trace.md` containing: a screenshot or text dump of the span tree, and one paragraph naming each span, its `gen_ai.*` attributes where present, and which span took the longest.

**Acceptance criteria.**

- The run produces a single trace with a nested span tree (parent supervisor span, child LLM/tool/retrieval spans).
- At least one manual span exists for a domain step the framework doesn't emit on its own.
- The LLM spans carry `gen_ai.request.model` and `gen_ai.usage.input_tokens`/`output_tokens`.
- One paragraph identifying the longest span. Committed.

**Hint.** Don't instrument only `invoke()` — that gives you one span and no per-step signal (the trap from Lecture 1 §5). Use the OpenInference instrumentor for your framework, then wrap the domain step in `tracer.start_as_current_span(...)`.

**Estimated time.** 40 minutes.

---

## Problem 2 — Per-route / per-user / per-model token accounting

**Problem statement.** Run your agent over at least 10 requests spanning two "routes" (e.g. a cheap-FAQ route and an expensive-research route) and two models (e.g. `claude-haiku-4-5` and `claude-opus-4-8`). From the recorded spans' `gen_ai.usage.*` attributes, compute total tokens and cost grouped by route, by user, and by model. Produce `notes/week-18/token-accounting.md` with the three rollup tables and the per-request median cost.

**Acceptance criteria.**

- Three rollup tables (by route, by user, by model) of input tokens, output tokens, and cost.
- Costs use real 2026 prices (opus $5/$25, sonnet $3/$15, haiku $1/$5 per million in/out; $0 for a self-hosted open model).
- The numbers come from span attributes, not a vendor invoice. Committed.

**Hint.** Reuse `exercise-03-token-accounting.py` — feed it your real recorded spans instead of the synthetic ones. Record `route`/`user`/`model` as span attributes at emit time so the grouping is a one-liner.

**Estimated time.** 45 minutes.

---

## Problem 3 — Define an SLO and check it against measured latency

**Problem statement.** Define one latency SLO for your supervisor (e.g. "95% of runs under 8 s") and one error-budget window (e.g. per 1,000 runs). Collect the per-run durations from your traces (≥30 runs), compute p50/p95/p99, and write an SLO check that reports PASS/FAIL and the remaining error budget. Produce `notes/week-18/slo.md` with the SLO statement, the percentile table, and the verdict.

**Acceptance criteria.**

- A written SLO (threshold + window) and an error-budget number.
- p50/p95/p99 computed from real durations (≥30 runs).
- A PASS/FAIL verdict with the budget spent vs available. Committed.

**Hint.** p95 is the (0.95 × N)-th sorted duration — see Lecture 2 §4.5. If you're failing, look at the *tail* trace (the slowest run), not the average — Problem 4 is exactly that.

**Estimated time.** 45 minutes.

---

## Problem 4 — Trace-driven debug of an injected failure

**Problem statement.** Deliberately break one step (e.g. point a tool at a dead endpoint so it times out, or corrupt a retrieval so it returns empty). Run the agent, open the resulting trace, and find the failing/slow span *from the trace alone* — time yourself. Produce `notes/week-18/debug.md` with the injected failure, the span you found (with its duration / error status), and the wall-clock time it took you to find it.

**Acceptance criteria.**

- A named injected failure (what you broke, where).
- The exact span identified as the culprit, with its duration or error/exception status from the trace.
- Your time-to-find recorded (aim for under 5 minutes — the week's promise). Committed.

**Hint.** Sort spans by duration, or filter to error status — the OTel span carries `status` and `exception` events. The dashboard's p95-by-step chart points you at the *step*; the individual trace points you at the *run* (Lecture 2 §3).

**Estimated time.** 45 minutes.

---

## Problem 5 — Eval-on-traces: replay old vs new prompt

**Problem statement.** Pick a prompt your agent uses, write a v2 of it, and replay at least 10 recorded production traces through both v1 and v2 — same recorded inputs, new prompt. Diff the outputs and at least one metric (e.g. faithfulness, answer length, or token cost). Produce `notes/week-18/replay.md` with the replay set size, the per-version metric, and a sentence on whether v2 is better, worse, or inside the noise.

**Acceptance criteria.**

- A replay over ≥10 recorded traces (real inputs from your spans), both prompt versions.
- A metric diff (v1 vs v2) plus output diffs on at least two traces.
- An honest conclusion that accounts for the sample size (don't over-claim a tiny delta). Committed.

**Hint.** The recorded *input* is on the span (the prompt/user turn captured as a span event); replay re-runs that input through the new prompt. A 0.01 delta on 10 traces is noise (Lecture 2 §4) — say so if that's what you see.

**Estimated time.** 1 hour.

---

## Problem 6 — The one-page observability memo (headline deliverable)

**Problem statement.** Write a **one-page** memo at `notes/week-18/observability-memo.md` against this template:

1. **The system** — one sentence: what agent you instrumented and which backends its spans flow to (Langfuse + Phoenix).
2. **The three dashboards** — token usage per route, p95 latency per agent step, retrieval-precision over time: one line each on what they show *for your system* with a real number.
3. **The SLO** — your latency SLO, its current status, and the error budget remaining.
4. **The cost picture** — per-request median cost and the most expensive route, from your token accounting.
5. **The failure you found** — the injected failure from Problem 4 and the time-to-find, in the promise format: `synthetic tool timeout -> span "mcp.web_fetch" (12.4 s, rank 1 by duration) ✓ found in 3m10s`.
6. **The one change you'd make** — the single highest-leverage instrumentation or SLO change you'd ship next, with the reason.

**Acceptance criteria.**

- `notes/week-18/observability-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- Every claim is backed by a number from your own traces, not a vibe.
- At least one per-query/per-failure trace line in the promise format.
- Committed.

**Hint.** This is the Phase III milestone artifact in miniature — the reviewer will ask "how do you know your agent is healthy, and how fast can you find a problem?" The memo *is* that answer, with numbers.

**Estimated time.** 45 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Instrument a run + read its trace | 40 min |
| 2 — Per-route/user/model token accounting | 45 min |
| 3 — Define + check an SLO | 45 min |
| 4 — Trace-driven debug of an injected failure | 45 min |
| 5 — Eval-on-traces replay (v1 vs v2) | 1 h 0 min |
| 6 — Observability memo (headline) | 45 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchobs` [mini-project](./mini-project/README.md) is in the same workspace — Week 19 (and the capstone) assume your agent's spans already flow to Langfuse and Phoenix. Then take the [quiz](./quiz.md) with your notes closed.
