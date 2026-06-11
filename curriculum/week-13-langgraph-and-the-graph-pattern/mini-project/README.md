# Mini-Project — `crunchagent_graph`: A Reusable Agent-Graph Package

> Build a reusable Python package that wraps the plan/retrieve/execute/critique state graph + SQLite persistence + a step/token/time budget + a benchmark runner, so any project can `from crunchagent_graph import build_agent` and get a checkpointed, budgeted, observable agent that reuses your weeks 7–12 retrieval as the retrieve node — and runs the 25-task benchmark with a single command.

This is the artifact that turns this week's lectures and exercises into a thing you *own* and *reuse*. After this week, building an agent is `build_agent(retriever=my_retriever)` and running the benchmark is `python -m crunchagent_graph bench` — not re-deriving the graph, the checkpointer, and the budgets every time. The package is corpus-agnostic, model-agnostic (swap the node's model call), and persistence-backed by default. It reuses week 5's tasks and weeks 7–12's retrieval **unchanged**, and it is the direct ancestor of the weeks 22–23 capstone supervisor.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This package is imported directly by the **weeks 22–23 multi-agent capstone**. The capstone's supervisor is a `StateGraph` that routes across multiple sub-agents — and each sub-agent is an instance of *this* graph. The budgets you enforce here, the checkpointer you wire here, and the `build_agent` factory you write here are reused at capstone scale. Build it well now; the capstone leans on it.

---

## What you will build

A small Python package `crunchagent_graph` with five deliverables:

1. **`crunchagent_graph/state.py`** — the `AgentState` `TypedDict` and the reducer choices (which keys append, which overwrite). The single source of truth for "what flows through the graph."
2. **`crunchagent_graph/nodes.py`** — the four node functions (plan, retrieve, execute, critique), each calling the model through one swappable `call_model` function and the retrieve node calling an injected retriever. The per-node logic, and nothing else.
3. **`crunchagent_graph/graph.py`** — the `build_agent(...)` factory: wires the nodes, the conditional edge, the budget router, and the checkpointer into a compiled, runnable graph. The thing callers import.
4. **`crunchagent_graph/budget.py`** — the step/token/time budget logic, enforced in the conditional-edge router. A loop without this is a bill.
5. **`crunchagent_graph/cli.py`** — a `bench` command that runs the 25-task benchmark through the graph (with persistence) and prints the pass rate, and a `run` command for a single task.

By the end you have a public repo of ~400–500 lines of Python that any future agent project can import and stop re-deriving the graph.

---

## Why a package and not a script

You could do all of this in one `agent.py`. Don't — not as the artifact. A package gives you:

- **Reuse.** The capstone imports `build_agent` and the budgets. A script gets copy-pasted, drifts, and rots.
- **A fixed contract.** The benchmark, the budgets, and the "everything is checkpointed" discipline live in code, version-controlled. "Did this change help?" is answered by re-running `bench`, not by eyeballing a new cell.
- **Model- and retriever-injection.** `build_agent(retriever=..., model=...)` makes the graph reusable across corpora and models. A hard-coded script is none of those.

The thing you ship and depend on is a package. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchagent_graph/
├── pyproject.toml
├── README.md                   # the benchmark results + the loop-vs-graph memo
├── corpus/
│   └── tasks.jsonl             # the 25 benchmark tasks: {"task": "...", "answer": "..."}
├── crunchagent_graph/
│   ├── __init__.py             # exports build_agent
│   ├── state.py                # AgentState TypedDict + reducers
│   ├── nodes.py                # plan / retrieve / execute / critique
│   ├── budget.py               # step/token/time budget router
│   ├── graph.py                # build_agent(...) factory (wires + compiles)
│   └── cli.py                  # `bench` and `run` commands
└── tests/
    ├── test_graph.py           # the graph runs a task end to end; budget caps the loop
    └── test_resume.py          # kill-and-resume: retrieve runs once across a kill
```

Your week-5 benchmark tasks and your weeks 7–12 retrieval package are dependencies (installed editable or vendored); the graph imports them **unchanged**.

---

## Deliverable 1 — `state.py` (the state and its reducers)

The state is the spine. Define it once, choose the reducer per key, and never touch a `while` loop's scattered locals again.

```python
"""crunchagent_graph.state — the typed state that flows through the graph."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    task: str                                   # set once at start
    plan: str                                   # overwrite (latest plan)
    docs: Annotated[list[str], operator.add]    # APPEND (re-plan accumulates docs)
    answer: str                                 # overwrite (latest answer)
    critique: str                               # overwrite (latest verdict)
    steps: int                                  # step counter (budget)
    tokens: Annotated[int, operator.add]        # APPEND (cumulative token count)
    t0: float                                   # start time (time budget)


def initial_state(task: str) -> AgentState:
    import time
    # TODO 1: return a fully-seeded initial state. Every key must be present
    #   (the graph seeds, then nodes overwrite/append). Set t0 = time.monotonic().
    ...
```

> **The rule the package enforces:** the state holds **plain serializable data only** — strings, lists, ints. Never a database connection, a file handle, a client object, or a lambda. The checkpointer serializes the state to disk; a non-serializable value breaks resume (the challenge's Trap 2). The retrieve node puts retrieved *text* in `docs`, never the retriever object.

---

## Deliverable 2 — `nodes.py` (the four nodes + one swappable model call)

Every node calls the model through *one* function, so swapping Claude for a local model is a one-line change.

```python
"""crunchagent_graph.nodes — the plan/retrieve/execute/critique node functions."""
from __future__ import annotations

import os

_client = None
if os.environ.get("ANTHROPIC_API_KEY"):
    import anthropic
    _client = anthropic.Anthropic()


def call_model(prompt: str, *, hard: bool = False) -> tuple[str, int]:
    """The ONE place a model is called. Returns (text, tokens_used).

    Swap THIS to point at Ollama/vLLM and the whole graph is model-agnostic.
    """
    # TODO 2: if _client is set, call client.messages.create with
    #   model="claude-opus-4-8" if hard else "claude-sonnet-4-6",
    #   thinking={"type": "adaptive"}, (and output_config={"effort":"high"} if hard),
    #   max_tokens=1024. Return (text, resp.usage.input_tokens + output_tokens).
    #   Else return a deterministic stub + a fixed token count so it runs offline.
    ...


def plan_node(state):
    text, tok = call_model(f"Task: {state['task']}\n\nWrite a one-paragraph plan.")
    return {"plan": text, "steps": state["steps"] + 1, "tokens": tok}


def make_retrieve_node(retriever):
    """Closure: inject YOUR weeks 7-12 retriever. The graph never hard-codes it."""
    def retrieve_node(state):
        # TODO 3: call retriever(state["task"]) -> list[str] of doc texts.
        #   retriever is the weeks 7-12 pipeline (store.knn + embed_query), passed
        #   in by build_agent. Return {"docs": [...], "steps": state["steps"] + 1}.
        ...
    return retrieve_node


def execute_node(state):
    ctx = "\n---\n".join(state["docs"]) or "(no docs)"
    text, tok = call_model(
        f"Task: {state['task']}\nContext:\n{ctx}\n\nAnswer using the context."
    )
    return {"answer": text, "steps": state["steps"] + 1, "tokens": tok}


def critique_node(state):
    text, tok = call_model(
        f"Task: {state['task']}\nAnswer: {state['answer']}\n"
        "Reply 'pass' or 'fail: <reason>'.", hard=True,
    )
    return {"critique": text.strip().lower(), "steps": state["steps"] + 1, "tokens": tok}
```

The retrieve node is a **closure** so the package never hard-codes your retriever — `build_agent(retriever=...)` injects it. That's how this graph reuses weeks 7–12 unchanged: your retrieval *is* the retrieve node's tool.

---

## Deliverable 3 — `budget.py` (the budget router)

The conditional edge is the only place the loop can close, so the budget lives here. All three caps, no exceptions.

```python
"""crunchagent_graph.budget — step/token/time caps enforced in the router."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Budget:
    steps: int = 12
    tokens: int = 60_000
    seconds: float = 120.0


def make_router(budget: Budget):
    """Return the conditional-edge routing function for critique."""
    def route_after_critique(state) -> str:
        # TODO 4: return "end" if ANY budget is exceeded (steps/tokens/time),
        #   OR if state["critique"] starts with "pass". Otherwise return "replan".
        #   This is the ONLY place the re-plan loop terminates — get it right.
        ...
    return route_after_critique
```

> **Non-negotiable:** the router must end the graph when *any* budget is hit, regardless of the critique. An agent whose loop only ends on "pass" can loop forever on a task it never satisfies. The grading rubric weights this heavily, and `test_graph.py` asserts that a deliberately-unsatisfiable task hits the cap and terminates.

---

## Deliverable 4 — `graph.py` (the `build_agent` factory)

The function callers import. It wires everything and compiles with a checkpointer.

```python
"""crunchagent_graph.graph — build_agent: the compiled, checkpointed agent."""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .budget import Budget, make_router
from .nodes import plan_node, make_retrieve_node, execute_node, critique_node


def build_agent(retriever, checkpointer, budget: Budget | None = None):
    """Wire the four-node graph, the budget router, and the checkpointer.

    retriever: a callable(task: str) -> list[str], YOUR weeks 7-12 pipeline.
    checkpointer: a SqliteSaver (or MemorySaver for tests).
    """
    budget = budget or Budget()
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", make_retrieve_node(retriever))
    g.add_node("execute", execute_node)
    g.add_node("critique", critique_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "execute")
    g.add_edge("execute", "critique")
    # TODO 5: add the conditional edge from "critique" using make_router(budget),
    #   with path map {"replan": "plan", "end": END}.
    ...
    return g.compile(checkpointer=checkpointer)
```

Note that `build_agent` takes the **checkpointer** as an argument rather than constructing it, so callers control persistence: a `SqliteSaver` for real runs, a `MemorySaver` for tests. The factory does not bypass the checkpointer — there is no "run without persistence" path, because resumability is the point.

---

## Deliverable 5 — `cli.py` (the `bench` and `run` commands)

```bash
# Run the 25-task benchmark through the graph, with SQLite persistence:
python -m crunchagent_graph bench --tasks corpus/tasks.jsonl --db bench.sqlite

# Run a single task and stream the node trace:
python -m crunchagent_graph run --task "What is the confidentiality duration?"
```

`bench` should give each task its own `thread_id` (so a crash on one task resumes that task), invoke the graph, score the answer against the expected answer, and print:

```
TASK                                           PASS   STEPS   TOKENS
01  confidentiality duration                    ✓       4      2,310
02  termination notice period                   ✓       4      2,180
...
25  multi-step compute then lookup              ✓       8      6,940
--------------------------------------------------------------------
25-task benchmark: 23/25 passed   median steps: 4   total tokens: 71,420
(loop vs graph comparison: see README memo)
```

The `run` command should `stream` the node trace so you watch the agent move plan → retrieve → execute → critique — the observability win, made visible on demand.

---

## Rules

- **You may** read the LangGraph docs, the lecture notes, your week-5 agent, and your weeks 7–12 retrieval code.
- **You must not** bypass the checkpointer — there is no "run without persistence" mode. Every run is checkpointed; resumability is the deliverable, not an option.
- **You must** enforce all three budgets (step, token, time) in the router. A loop without a budget fails the rubric.
- **You must** reuse your weeks 7–12 retrieval as the retrieve node's tool (injected via `build_agent(retriever=...)`), not re-implement retrieval.
- **You must not** stuff non-serializable objects (clients, connections, lambdas) into the state — the checkpointer must be able to serialize it, or resume breaks.
- **You must not** bury the model call in LangChain chains. The node calls the model through one swappable `call_model` function; the framework orchestrates, it does not abstract away the model call.
- Python 3.12, `langgraph`, `langchain-core`, `langgraph-checkpoint-sqlite`, `anthropic`, plus `pytest`. The model behind the nodes may be Claude or a local open model (Ollama/vLLM) — the graph is model-agnostic by design.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-13-crunchagent-graph-<yourhandle>`.
- [ ] `build_agent(retriever, checkpointer, budget)` returns a compiled four-node graph with the conditional re-plan edge and the budget router.
- [ ] The retrieve node uses **your weeks 7–12 retriever**, injected — not re-implemented.
- [ ] All three budgets (step, token, time) are enforced in the router; an unsatisfiable task hits the cap and terminates (tested).
- [ ] `python -m crunchagent_graph bench` runs the **25-task benchmark** with SQLite persistence and prints the pass rate.
- [ ] A **resumability test** (`test_resume.py`): a task is run partway, the process/app is killed, a fresh app resumes the same `thread_id`, and a node's side-effect ran exactly **once** across the kill (proving resume, not re-run).
- [ ] `pytest` passes, with at least `test_graph.py` (end-to-end run + budget cap) and `test_resume.py` (kill-and-resume).
- [ ] A `README.md` with the benchmark table and a **loop-vs-graph memo**: lines of code, observability, and resumability, loop versus graph, with the 25-task pass rate for each.
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Graph correctness** | 25 | Four nodes wired correctly; conditional re-plan edge present; nodes return partial state; reducers chosen right (`docs`/`tokens` append, others overwrite); the graph runs the 25 tasks to a pass rate matching the week-5 loop within noise. |
| **Persistence & resume** | 25 | `SqliteSaver` wired; runs are checkpointed; the kill-and-resume test proves a node ran once across a kill; no non-serializable state; checkpointer is never bypassed. |
| **Budgets** | 20 | All three (step/token/time) enforced in the router; an unsatisfiable task hits the cap and terminates; the loop can never run forever; tested. |
| **Retriever reuse** | 15 | Weeks 7–12 retrieval is injected and used as the retrieve node's tool, unchanged — not re-implemented; the graph is corpus-agnostic via the injection. |
| **CLI & comparison** | 10 | `bench` and `run` work; `run` streams the node trace; the README memo compares loop vs graph on LoC/observability/resumability with the pass rate. |
| **Docs & hygiene** | 5 | Clear README + memo, no secrets committed, sensible commits, no `__pycache__`/`.venv`/`*.sqlite` checked in. |

**90+** is portfolio-grade and ready to drop into the capstone as a sub-agent. **70–89** works but has a soft budget, an unlabeled model call, or a resume that re-runs work. **Below 70** means the agent isn't actually resumable or bounded — fix that first, because the capstone builds directly on this package.

---

## Stretch goals

- **Supervisor wrapper.** Add `crunchagent_graph/supervisor.py` with a `build_supervisor(sub_agents)` that routes across multiple `build_agent` instances (Lecture 2 §3.1). This is the literal weeks 22–23 capstone skeleton — building it here is a capstone head start.
- **Time-travel CLI.** Add `python -m crunchagent_graph history --thread task-17` that lists the checkpoint history and `... --resume-from <checkpoint_id>` that replays from an earlier checkpoint with a corrected input. Debugging-by-replay as a command.
- **Postgres backend.** Swap `SqliteSaver` for `PostgresSaver` (`langgraph-checkpoint-postgres`) behind a `--db postgres://...` flag. Confirm the graph, nodes, and budgets are unchanged — only the checkpointer backend differs. That's the production path.
- **Ragas on the answers.** Pipe the benchmark's retrieved docs + answers into your week-12 Ragas suite and report faithfulness/answer-relevancy alongside the pass rate. Your retrieval *and* your agent now share one eval surface.
- **CI.** A GitHub Actions workflow that runs `pytest` (with `MemorySaver` for the resume test, no external services) on every push. Green check or it didn't happen.

---

## How this connects to the rest of C23

- **Week 5 (the hand-rolled ReAct loop)** gave you the agent this package re-implements as a graph, and the 25-task benchmark `bench` runs. The loop → this graph is the syllabus lab; this package is that lab, packaged for reuse.
- **Weeks 7–12 (retrieval + Ragas)** give you the retriever this graph injects as its retrieve node, and the eval surface the stretch goal reuses. The graph *orchestrates* your RAG; it doesn't replace it.
- **Week 14 (Mastra / Inngest, TypeScript)** is the contrast: the same ideas — explicit steps, durable state, resumability — in a different stack, where the "checkpoint" is a workflow step a job runner retries. You'll be able to say precisely what `SqliteSaver` and Inngest step memoization have in common after building this.
- **Weeks 22–23 (the multi-agent capstone)** import `build_agent` and the budgets directly: the capstone supervisor routes across sub-agents, each one an instance of *this* graph, checkpointed and bounded exactly as you built it here.

When you've finished, push the repo and take the [quiz](../quiz.md).
