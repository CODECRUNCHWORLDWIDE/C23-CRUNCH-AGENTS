# Week 13 Homework

Six problems that revisit the week's topics and force the graph pattern into your fingers. The full set should take about **5 hours**. Work in your Week 13 Git repository (the same workspace as the exercises and the `crunchagent_graph` mini-project) so every problem produces at least one commit you can point to at the Phase III capstone.

The headline deliverable is **Problem 4 — the LangGraph re-implementation plus the LoC/observability/resumability comparison memo**, the syllabus lab. Treat it as the artifact a reviewer reads, not a journal entry.

Have your **week-5 ReAct agent** and its **25-task benchmark** importable (this week re-implements that agent and runs that benchmark), and your **weeks 7–12 retrieval** importable (the retrieve node calls it unchanged). Install the stack: `pip install langgraph langchain-core langgraph-checkpoint-sqlite anthropic`. If week 5 or the benchmark is missing, reconstruct it first — this week depends on it.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Build a minimal state graph and watch the conditional edge loop

**Problem statement.** Build a three-node `StateGraph` (plan → act → critique) with one conditional edge that loops back to `plan` when the critique fails and a step budget that caps the loop (Exercise 1's shape). Use a deterministic stub for the node "work" so it runs with no API key. Stream the run and capture the trace showing the critique fail once, the edge loop back, and the second pass terminate.

**Acceptance criteria.**

- A runnable `graph_min.py` with a `TypedDict` state (at least one `operator.add` reducer key), three nodes, normal edges, and a conditional edge with a step budget in the router.
- The captured trace (in `notes/week-13/min-graph.md`) shows: critique fails → loops back to `plan` → second pass passes → graph ends.
- You can explain in one sentence why each node returns a *partial* dict.
- Committed.

**Hint.** Make the stub critique fail on the first attempt (`"answer-v1" → fail`) and pass on the second so you *see* the loop. The budget in the router (`attempts >= 3 → stop`) is what guarantees termination even if it never passed — never omit it.

**Estimated time.** 40 minutes.

---

## Problem 2 — Re-implement your week-5 ReAct agent as the four-node graph

**Problem statement.** Take your week-5 ReAct agent and re-implement it as a LangGraph `StateGraph` with four nodes — plan, retrieve, execute, critique — and a conditional edge that re-plans on critique failure. The retrieve node must call your **weeks 7–12 retriever** (not a re-implementation). The nodes call `claude-sonnet-4-6` (plan/retrieve/execute) and `claude-opus-4-8` (critique), or a local open model. Run it on three benchmark tasks and print the node trace.

**Acceptance criteria.**

- A runnable `react_graph.py` with the four nodes wired (`add_node` / `add_edge` / `add_conditional_edges`), compiled and invoked on at least three tasks.
- The retrieve node calls your weeks 7–12 retriever; retrieval is not re-implemented.
- The node trace (via `stream(stream_mode="updates")`) is printed/captured, showing each node fire.
- Claude calls use adaptive thinking and no `temperature`/`budget_tokens` (or a documented local-model call).
- Committed.

**Hint.** Port the node bodies from Lecture 1 §4. The retrieve node puts retrieved *text* in `docs` (a reducer key), never the retriever object — keep the state serializable for Problem 3. If a node 400s, check you removed `temperature`/`top_p`/`budget_tokens` (they're rejected on Sonnet 4.6 / Opus 4.8).

**Estimated time.** 55 minutes.

---

## Problem 3 — Survive a process kill

**Problem statement.** Attach a `SqliteSaver` to your Problem 2 graph. Run a task partway, simulate a process kill *after* the retrieve node, then resume in a fresh app instance from the same SQLite file and `thread_id`. Prove the agent resumed — the retrieve node ran **once**, not twice — by a side-effect counter or by inspecting the recovered state.

**Acceptance criteria.**

- `survive.py` runs the graph with a `SqliteSaver`, crashes after retrieve, then resumes via `invoke(None, config={"configurable": {"thread_id": ...}})`.
- The resume continues from the checkpoint: execute and critique run, retrieve does **not** re-run.
- A `PASS` line proves retrieve ran exactly once across the kill (the agent survived the kill).
- `notes/week-13/resume.md` records the before/after and the PASS line.
- Committed.

**Hint.** Put the crash in a node *after* retrieve so retrieve's state is checkpointed before the kill. On resume, pass `None` as the input — that's the signal to continue, not restart. If retrieve runs twice, the resume restarted from scratch; check the `thread_id` matches and the SQLite file is the same. Exercise 3 is the template.

**Estimated time.** 50 minutes.

---

## Problem 4 — The LangGraph re-implementation + comparison memo (headline deliverable)

**Problem statement.** This is the syllabus lab. Run the **full 25-task benchmark** through both your week-5 loop and your week-13 graph (with `SqliteSaver` + step/token/time budgets). Write a **one-page** memo at `notes/week-13/comparison-memo.md` against this template:

1. **Pass rate** — the 25-task pass rate for the loop and for the graph (they should be roughly equal — it's a refactor, not a smarter agent; if they diverge a lot, something's wrong).
2. **Lines of code** — count both, and *categorize*: how much of the loop was accidental control-flow complexity (tangled `if`s) vs essential work, and how much of the graph is LangGraph boilerplate vs node logic. The honest claim is "the remaining lines are clearer," not just "fewer lines."
3. **Observability** — the loop's opacity ("the agent is in the loop") vs `app.stream` showing every node fire. Include a sample streamed trace.
4. **Resumability** — loop = restart from zero on a crash; graph = resume from checkpoint. Include the Problem 3 kill-and-resume proof.
5. **The budget** — confirm all three (step/token/time) are enforced in the graph's router, and what happens on an unsatisfiable task (it hits the cap and ends).
6. **One promise trace** — the agent killed mid-run and resumed, in the format `retrieve ran once, not twice`.

**Acceptance criteria.**

- `notes/week-13/comparison-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- The 25-task pass rate is reported for both implementations from the same benchmark.
- The LoC comparison categorizes lines (accidental vs essential), not just totals.
- The resumability section includes the actual kill-and-resume proof.
- The budget section confirms all three caps are enforced.
- Committed.

**Hint.** Use the *same* models and the *same* tools in both implementations so the comparison isn't confounded by a model change — the only thing that varies is loop vs graph (+ persistence). The graph harness is in Challenge 1; reuse it. Generate the streamed trace with `stream(stream_mode="updates")`.

**Estimated time.** 1 hour 5 minutes.

---

## Problem 5 — Add a supervisor over two sub-agents

**Problem statement.** Build a supervisor graph: a router node that delegates to two sub-agents — a "retrieval" sub-agent (calls your weeks 7–12 retriever) and a "compute" sub-agent (does arithmetic, via the model or a calculator tool) — and routes back to the supervisor after each, ending when the task is done. Run it on two tasks: one that needs retrieval, one that needs computation. Confirm the supervisor routes each to the right sub-agent.

**Acceptance criteria.**

- A runnable `supervisor.py` with a supervisor node, two sub-agent nodes, conditional edges from the supervisor to each sub-agent, and edges from each sub-agent back to the supervisor.
- The supervisor uses a cheap model (`claude-haiku-4-5`) for the routing decision; sub-agents use stronger models where needed.
- A budget caps the supervisor loop.
- The trace shows correct routing for both a retrieval task and a compute task.
- Committed (`notes/week-13/supervisor.md` records the traces).

**Hint.** The supervisor returns a `next` key ("retrieval" / "compute" / "FINISH"); a routing function maps it to the sub-agent node or `END`. Sub-agents edge *back* to the supervisor (`add_edge("retrieval", "supervisor")`). This is Lecture 2 §3.1 — and the weeks 22–23 capstone skeleton, so build it cleanly.

**Estimated time.** 50 minutes.

---

## Problem 6 — When a state machine beats an agent

**Problem statement.** Take one of your 25 benchmark tasks whose control flow is actually *fixed* (e.g. "extract field X, validate it, format the output" — a known three-step pipeline). Implement it twice: once as an *agent* (the model decides routing via a conditional edge) and once as a *deterministic state machine* (the same `StateGraph` but the conditional edges are plain `if state[...]`, no LLM routing decision). Compare them on tokens, latency, and reproducibility. Write a one-paragraph conclusion on which was the honest choice for *this* task.

**Acceptance criteria.**

- Two implementations of the same fixed-flow task: an agent (LLM-routed) and a state machine (deterministically routed), both as `StateGraph`s.
- A comparison (in `notes/week-13/state-machine.md`) of tokens, latency, and reproducibility across a few runs each.
- A one-paragraph conclusion naming which was the honest choice and why (the tell: could you draw the control flow before running it?).
- Committed.

**Hint.** The state machine's router is `return "validate" if state["extracted"] else "extract"` — a plain `if`, no model call. The agent's router asks the model "what next?". On a fixed-flow task the state machine should win on tokens (no routing LLM call), latency (no round-trip), and reproducibility (deterministic) — which is the point of Lecture 2 §5: not everything that looks like an agent should be one.

**Estimated time.** 40 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Minimal state graph + conditional-edge loop | 40 min |
| 2 — Re-implement the week-5 ReAct agent as a graph | 55 min |
| 3 — Survive a process kill (SqliteSaver) | 50 min |
| 4 — Re-implementation + LoC/observability/resumability memo (headline) | 1 h 5 min |
| 5 — Supervisor over two sub-agents | 50 min |
| 6 — When a state machine beats an agent | 40 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchagent_graph` [mini-project](./mini-project/README.md) is in the same workspace — the weeks 22–23 capstone imports it and builds its supervisor on top of it. Then take the [quiz](./quiz.md) with your notes closed.
