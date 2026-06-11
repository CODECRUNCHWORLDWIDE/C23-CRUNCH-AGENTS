# Week 13 — Exercises

Three focused drills that take you from "a loop with too many `if`s" to "a checkpointed agent that survives a kill." Each takes 30–60 minutes. Do them in order — exercise 2 builds the four-node graph that exercise 3 checkpoints, and exercise 1 builds the minimal graph that proves you understand nodes, edges, and conditional edges first.

## Index

1. **[Exercise 1 — Your first state graph](exercise-01-first-state-graph.md)** — build a minimal 3-node `StateGraph` (plan → act → critique) with one conditional edge that loops back to `plan` when the critique fails, and inspect the state at each node. (~45 min, guided, with real embedded Python)
2. **[Exercise 2 — ReAct as a graph](exercise-02-react-as-a-graph.py)** — implement the plan/retrieve/execute/critique graph with a conditional edge that re-plans on critique failure, run it on a couple of tasks, and print the node trace. (~50 min, runnable)
3. **[Exercise 3 — Survive the kill](exercise-03-survive-the-kill.py)** — attach a SqliteSaver checkpointer with a `thread_id`, run the graph partway, simulate a process kill, then resume from the checkpoint and prove the agent continued from where it died. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install the deps: `pip install langgraph langchain-core langgraph-checkpoint-sqlite anthropic`. (The `langchain-anthropic` wrapper is optional; the exercises call the raw SDK.)
- **Set `ANTHROPIC_API_KEY`** if you want the nodes to call Claude. If you don't set it, every runnable exercise has a **deterministic stub "LLM"** so the file *still runs* and the graph/edge/checkpoint lessons are identical — only the node *content* is stubbed.
- **The graph, not the model, is the lesson.** Exercises 2 and 3 work the same whether the node calls `claude-sonnet-4-6`, a local Ollama model, or the deterministic stub. If you want to swap in a local open model, change the one `call_model` function and nothing else — that's the point.
- **Watch the trace.** The whole observability win over the week-5 loop is that you can *see* each node fire. Every runnable exercise prints a node-by-node trace; read it.
- When the graph loops or won't terminate, look at the **conditional edge router and the step budget** first — an unbounded re-plan loop is the classic bug, and the budget in the router is the fix.
- Each runnable exercise (`.py`) ends with an **Expected output (shape)** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone. They run with or without `langgraph` installed and with or without an API key:

```bash
# With everything installed and a key set — the real path:
export ANTHROPIC_API_KEY=sk-ant-...
python3 exercise-02-react-as-a-graph.py
python3 exercise-03-survive-the-kill.py
```

If `langgraph` is **not** installed, both files fall back to a tiny hand-rolled state-machine engine that demonstrates the *identical* node/edge/conditional-edge (exercise 2) and checkpoint/resume (exercise 3) concepts — so you can read and run the lesson today and install LangGraph when you're ready. The header of each file documents the fallback and prints which path is active.

## A note on determinism

The graph engine is deterministic: same state and same edges produce the same path every run. The only non-determinism is the *model's* output inside a node — and with the deterministic stub (no API key), even that is fixed, so the whole exercise is reproducible. Exercise 3's "did the agent resume?" check is fully deterministic regardless of the model: it asserts that `retrieve` ran *once* across the kill, which is a property of the checkpointer, not the model. If you can't reproduce the resume, the checkpointer or the `thread_id` is wrong, and that's worth finding.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-13` to compare.
