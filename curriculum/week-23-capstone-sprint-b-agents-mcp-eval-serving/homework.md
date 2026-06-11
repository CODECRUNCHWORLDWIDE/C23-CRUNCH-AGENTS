# Week 23 Homework

Six problems that drive Sprint B into your fingers and produce the artifacts the final-capstone rubric grades. The full set should take about **5 hours**. Work in your Sprint A capstone repo so every problem produces a commit you can point to at the week-24 chaos drill and the final review.

The headline deliverable is **Problem 4 — the Sprint B cut-list memo**: the written record of what you scoped *in*, what you dropped, and why — the evidence that you engineered like a senior rather than a completionist.

Have your **Sprint A retrieval + memory** importable, **vLLM** (or Ollama) serving the local tier, **LiteLLM** routing, and `ANTHROPIC_API_KEY` set. If Sprint A is broken, fix it first — this week depends on it.

Each problem includes a **problem statement**, **acceptance criteria**, a **hint**, and an **estimated time**.

---

## Problem 1 — The four-agent supervisor graph, traced

**Problem statement.** Assemble the supervisor graph with all four subordinate agents (retrieval, code, writing, critique), real handoff, the SQLite checkpointer, and the per-route budget. Run three queries through it and capture the routing chain for each from the trace.

**Acceptance criteria.**

- The graph compiles and runs three queries end-to-end with real subordinates.
- The supervisor routes (it doesn't do); the trace shows the chain per query.
- The SQLite checkpointer is wired; killing the process mid-run and re-invoking with the same `thread_id` resumes.
- The budget aborts a forced runaway cleanly (no Ctrl-C).
- `notes/week-23/supervisor.md` records one query's full routing chain read from the trace.
- Committed.

**Hint.** Reuse Exercise 1's router and budget. Make the critique-agent fail once deliberately and confirm the trace shows `write → critique → write` and then a clean budget abort with `gen_ai.abort_reason`.

**Estimated time.** 55 minutes.

---

## Problem 2 — The MCP tool surface, hardened

**Problem statement.** Stand up the four MCP servers (filesystem, calculator, web-fetch, corpus-search) with the `mcp` SDK. Wire the corpus-search server to your Sprint A `hybrid_search`. Expose at least the corpus server over *both* stdio and streamable HTTP. Harden every tool: argument validation, path-traversal defense on the filesystem tool, rate limiting on the web-fetch tool.

**Acceptance criteria.**

- Four MCP servers run; the corpus server wraps the real Sprint A pipeline.
- The corpus server runs over stdio *and* streamable HTTP (consume it from the LangGraph agent over stdio; verify HTTP with `curl` or an MCP client).
- A test proves the filesystem tool rejects `../../etc/passwd` and the corpus tool rejects out-of-range `k`.
- The web-fetch tool rate-limits (a token bucket per run).
- `notes/week-23/tool-surface.md` documents each tool, its schema, and its defenses.
- Committed.

**Hint.** Port Exercise 2's `safe_path` and validation. The tool-surface doc is the input to the week-17-style threat model you'll re-test under attack in week 24 — write it for that audience.

**Estimated time.** 1 hour.

---

## Problem 3 — Two-tier serving with cost-tracked routing

**Problem statement.** Bring up vLLM (or Ollama) for the local 7B/13B, put LiteLLM in front with the vendor fallback to `claude-opus-4-8`, and wire the easy-vs-hard classifier so the writing route escalates to the vendor on hard queries. Run the 100-question gold set and produce the cost report skeleton: per-query cost, the local-vs-vendor split, and the savings versus a vendor-only baseline.

**Acceptance criteria.**

- vLLM/Ollama serves the local tier; LiteLLM routes with a working vendor fallback.
- The classifier routes easy → local, hard → vendor (visible in the trace / LiteLLM logs).
- `notes/week-23/cost-report.md` reports median / p95 / p99 cost per query, the route split, and the savings versus vendor-only.
- Committed.

**Hint.** Reuse Exercise 3's classifier and cost arithmetic, but swap `serve_mock` for the real LiteLLM call. The savings number is the capstone deliverable-5 headline — if the split is 100% vendor, your classifier is mis-labeling everything hard; check it.

**Estimated time.** 1 hour.

---

## Problem 4 — The Sprint B cut-list memo (headline deliverable)

**Problem statement.** Write a **one-page** memo at `notes/week-23/cut-list.md` recording the scoping decisions of Sprint B against this template:

1. **Shipped** — the components that are live and in the eval path (supervisor, which agents, which MCP tools, which serving tiers, the eval suite, the tracing).
2. **The cut list** — what you dropped or deferred, each with one sentence of *why* (e.g. "dropped the web-fetch server: no gold question needed it"; "deferred HITL approval on the code-agent: eval showed generated expressions 98% correct").
3. **The thin-slice timeline** — when the slice first ran end-to-end, and what you deepened after, driven by which eval metric.
4. **Eval state** — the four Ragas numbers, the judge mean, and `PASS`/`FAIL`.
5. **The one risk you're carrying into week 24** — the component you're least confident survives a chaos drill, and why.

**Acceptance criteria.**

- `notes/week-23/cut-list.md` exists, fits roughly one page (350–550 words), and hits all five headings.
- Every cut is justified by a *reason* (cost, no gold coverage, measured low value), not "ran out of time" alone.
- The eval state cites real numbers from your gate run, not aspirations.
- The week-24 risk is specific (names a component and a failure mode).
- Committed.

**Hint.** The cut list is not an apology — it's the senior move made legible. A capstone with a clear cut list and a green gate beats one with every feature half-wired and a red gate. Graders read this to see whether you scoped by measurement.

**Estimated time.** 50 minutes.

---

## Problem 5 — The eval gate in CI

**Problem statement.** Wire the eval suite (Ragas + calibrated judge) into a CI step that runs the gate and fails the build if any metric is below threshold. Calibrate the judge with 10 human-labeled examples and spot-check that it agrees with them.

**Acceptance criteria.**

- A CI workflow (GitHub Actions or equivalent) runs `capstone.eval run --gate` and fails on a red gate.
- The judge prompt embeds the 10 calibration examples; a spot-check note shows the judge agrees with your labels on those 10.
- `notes/week-23/calibration.md` records the 10 labels, the judge's scores on them, and the agreement.
- Committed, with a green CI check on the current build.

**Hint.** Keep the gold set in the repo; the CI job needs `ANTHROPIC_API_KEY` as a secret for the judge and the vendor route. If CI spend is a concern, gate on a 20-question subset in CI and the full 100 locally — document the split.

**Estimated time.** 50 minutes.

---

## Problem 6 — Ship a runnable artifact

**Problem statement.** Produce the `docker compose up`-runnable artifact (or a live deploy URL): one command brings up the supervisor service, the MCP servers, LiteLLM (fronting vLLM/Ollama and the vendor), and the tracing backends (Langfuse + Phoenix). A README documents the one command and a sample query.

**Acceptance criteria.**

- `docker compose up` (or a documented deploy) brings up the whole stack.
- A sample query runs end-to-end and produces a traced, grounded answer.
- The README documents the command, the env vars (`ANTHROPIC_API_KEY`), and the no-GPU Ollama fallback.
- Committed.

**Hint.** This is the mini-project's deliverable in skeleton — getting it `compose`-runnable now means the mini-project is polish, not panic. The trace links in the README are the proof the system is observable, which the rubric grades.

**Estimated time.** 45 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Four-agent supervisor graph, traced | 55 min |
| 2 — MCP tool surface, hardened | 1 h 0 min |
| 3 — Two-tier serving + cost-tracked routing | 1 h 0 min |
| 4 — Sprint B cut-list memo (headline) | 50 min |
| 5 — Eval gate in CI | 50 min |
| 6 — Ship a runnable artifact | 45 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the [mini-project](./mini-project/README.md) artifact is `compose`-runnable — week 24 attacks exactly this system. Then take the [quiz](./quiz.md) with your notes closed.
