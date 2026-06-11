# Challenge 1 — From ReAct Loop to State Graph

**Time estimate:** ~150 minutes.

## Problem statement

You have a working agent: the hand-rolled ReAct loop from week 5, scored on a 25-task benchmark. It works. It also has the three problems Lecture 1 named — you can't see where it is, you can't pause and resume it, and the control flow is a `while` loop you have to simulate in your head. You are going to re-implement it as a **LangGraph state graph** with explicit nodes (plan, retrieve, execute, critique), add a **SQLite checkpointer** so it survives a process kill, run the **same 25-task benchmark** through both implementations, and write down the comparison the syllabus asks for: **lines of code, observability, and resumability** — loop versus graph, with numbers.

This is the syllabus deliverable in lab form. The output is a re-implementation *plus* a measured comparison: not "I rewrote it in LangGraph," but "here is the loop, here is the graph, here is the 25-task pass rate for each (it should be roughly equal — the graph is a *refactor*, not a smarter agent), here is the LoC delta, here is the observability delta (I can stream every node), and here is the resumability delta (I killed it mid-run and it continued)."

## What's fixed, what varies

**Fixed (do not let these change between the loop and the graph):**

- **The 25 tasks** — the same benchmark from week 5, unchanged. Same tasks, same scoring. A task passes if the agent's final answer is correct by the same check you used in week 5.
- **The tools** — the same tool surface the week-5 agent had (your retriever from weeks 7–12 as the `retrieve` step, plus whatever calculator/lookup tools the benchmark needs).
- **The model(s)** — use `claude-sonnet-4-6` for plan/retrieve/execute and `claude-opus-4-8` for the critique (or your local open model via Ollama/vLLM — the graph is model-agnostic). Use the *same* models in both implementations so the comparison isn't confounded by a model change.
- **The budget** — both implementations enforce the same step/token/time budget. (Your week-5 loop had *some* termination condition; make the graph's match it.)

**Varies (this is the whole point):**

- The **implementation**: a `while` loop with conditionals (week 5) versus a `StateGraph` with nodes and edges (week 13).
- The **persistence**: none (the loop) versus a `SqliteSaver` checkpointer (the graph).

## The harness approach

The graph is the four-node ReAct graph from Lecture 1, compiled with a checkpointer:

```python
from typing import TypedDict, Annotated
import operator, time
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import anthropic

client = anthropic.Anthropic()


class AgentState(TypedDict):
    task: str
    plan: str
    docs: Annotated[list[str], operator.add]
    answer: str
    critique: str
    steps: int
    tokens: Annotated[int, operator.add]     # token budget accumulator
    t0: float                                # start time (for the time budget)


def plan_node(state):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": f"Task: {state['task']}\nWrite a plan."}],
    )
    used = resp.usage.input_tokens + resp.usage.output_tokens
    return {"plan": next(b.text for b in resp.content if b.type == "text"),
            "steps": state["steps"] + 1, "tokens": used}


def retrieve_node(state):
    from crunchrag_embed import store, embedders     # weeks 7-12, UNCHANGED
    bge = embedders.load("bge")
    hits = store.knn("clauses", bge.embed_query(state["task"]), k=3)
    return {"docs": [h.text for h in hits], "steps": state["steps"] + 1}


def execute_node(state):
    ctx = "\n---\n".join(state["docs"]) or "(no docs)"
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{"role": "user",
                   "content": f"Task: {state['task']}\nContext:\n{ctx}\nAnswer."}],
    )
    used = resp.usage.input_tokens + resp.usage.output_tokens
    return {"answer": next(b.text for b in resp.content if b.type == "text"),
            "steps": state["steps"] + 1, "tokens": used}


def critique_node(state):
    resp = client.messages.create(
        model="claude-opus-4-8", max_tokens=512,
        thinking={"type": "adaptive"}, output_config={"effort": "high"},
        messages=[{"role": "user",
                   "content": f"Task: {state['task']}\nAnswer: {state['answer']}\n"
                              "Reply 'pass' or 'fail: <reason>'."}],
    )
    used = resp.usage.input_tokens + resp.usage.output_tokens
    return {"critique": next(b.text for b in resp.content if b.type == "text").lower(),
            "steps": state["steps"] + 1, "tokens": used}


STEP_BUDGET, TOKEN_BUDGET, TIME_BUDGET = 12, 60_000, 120.0


def route_after_critique(state):
    if state["steps"] >= STEP_BUDGET:                 return "end"
    if state["tokens"] >= TOKEN_BUDGET:               return "end"
    if time.monotonic() - state["t0"] >= TIME_BUDGET: return "end"
    return "end" if state["critique"].startswith("pass") else "replan"


def build():
    g = StateGraph(AgentState)
    for name, fn in [("plan", plan_node), ("retrieve", retrieve_node),
                     ("execute", execute_node), ("critique", critique_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "execute")
    g.add_edge("execute", "critique")
    g.add_conditional_edges("critique", route_after_critique,
                            {"replan": "plan", "end": END})
    return g


def run_benchmark(tasks):
    passed = 0
    with SqliteSaver.from_conn_string("bench.sqlite") as cp:
        app = build().compile(checkpointer=cp)
        for i, (task, check) in enumerate(tasks):
            config = {"configurable": {"thread_id": f"task-{i}"}}
            init = {"task": task, "plan": "", "docs": [], "answer": "",
                    "critique": "", "steps": 0, "tokens": 0, "t0": time.monotonic()}
            final = app.invoke(init, config=config)
            passed += int(check(final["answer"]))
    return passed, len(tasks)
```

The resumability test is the kill-and-resume from Exercise 3, applied to the benchmark: run a task partway, kill the process, restart, and confirm the task completes from the checkpoint without re-running the nodes that already ran. Each task gets its own `thread_id`, so a crash on task 17 resumes task 17, not the whole benchmark.

## Acceptance criteria

- [ ] A `challenge-01/` directory with a runnable graph implementation of the plan/retrieve/execute/critique agent, compiled with a `SqliteSaver`.
- [ ] The **same 25-task benchmark** from week 5 runs through the graph and prints a pass rate. The pass rate is roughly equal to the week-5 loop's (within noise) — this is a refactor, not a smarter agent.
- [ ] A **budget** (step + token + time) is enforced in the conditional-edge router; an unbounded task hits the cap and ends, it does not loop forever.
- [ ] A **resumability demo**: a task is run partway, the process is killed (a real `kill`/`SystemExit`), a fresh process resumes the *same* `thread_id` from `bench.sqlite`, and the task completes **without re-running already-completed nodes** (proven by a side-effect counter or by inspecting the recovered state).
- [ ] A `comparison-memo.md` (one page) with a table: **loop vs graph** across **lines of code** (count both), **observability** (the loop's opacity vs `app.stream` node-by-node), and **resumability** (loop = restart from zero; graph = resume from checkpoint), plus the 25-task pass rate for each.
- [ ] At least one **promise trace** in the format: the agent killed mid-run and resumed, showing a node ran once across the kill (`retrieve ran once, not twice`).

## The trap (read after a first attempt)

Three traps catch people on this lab.

**Trap 1 — the unbounded loop.** The conditional edge `critique → plan` is a loop. If the critique *never* passes (a hard task, a strict judge, a model having a bad day) and you forgot the budget, the graph runs forever, burning tokens until you kill it. The fix is non-negotiable and lives in the router: `if state["steps"] >= STEP_BUDGET: return "end"`. **A re-plan loop without a budget is not an agent, it's a bill.** Every loop in every graph you ever ship gets a budget. The challenge requires all three (step, token, time) precisely so this reflex sticks.

**Trap 2 — non-deterministic state that breaks resume.** The checkpointer serializes your state to disk. If your state holds something that *can't* round-trip — a live database connection, an open file handle, a lambda, a non-serializable object — the resume fails or silently loses data. Keep the state to plain serializable data (strings, lists, ints, dicts). The `retrieve` node should put the *retrieved text* in the state, not the retriever object. If you find yourself stuffing a client or a connection into `AgentState`, that's the trap: the state is data, not handles.

**Trap 3 — conflating LangGraph's value with LangChain's chains.** LangGraph is a *graph engine over your own functions*. It is **not** LangChain, and the win is **not** "I replaced my code with LangChain chains." If your re-implementation buries the model call inside a stack of `LCEL` chains and prompt-template abstractions, you've added the indirection the course warns against and lost the legibility that was the whole point. The graph's value is explicit *state and edges*; inside a node, call the model the plain way (the raw SDK, as in the harness above). Measuring "LoC" only counts as a win if the lines you removed were *accidental complexity* (the loop's tangled conditionals), not *essential clarity* (the visible model call) traded for hidden chains.

## Stretch goals

- **Supervisor over sub-agents.** Split the agent into two sub-agent graphs (a "retrieval" sub-agent and a "compute" sub-agent) and add a supervisor node that routes between them (Lecture 2 §3.1). Run the 25 tasks through the supervisor. This is the exact shape of the weeks 22–23 capstone — building it now is a head start on the capstone.
- **Time-travel replay.** For a task that failed, list its checkpoint history (`get_state_history`), find the checkpoint just before the bad decision, and resume from there with a corrected plan injected. Show the same task now passes. This is debugging-by-replay, the agent equivalent of `git checkout`.
- **Loop vs graph LoC, honestly counted.** Don't just count total lines — categorize them. How many lines of the *loop* were accidental control-flow complexity (the tangled `if`s) versus essential work? How many lines of the *graph* are LangGraph boilerplate (state def, edges) versus the actual node logic? The honest comparison isn't "fewer lines," it's "the lines that remain are clearer." Make that case with the breakdown.
- **Open-model parity.** Run the same 25-task benchmark with a local Ollama model behind every node (change only `call_model`) and report the pass-rate delta. The point isn't that the open model matches Claude — it's that the *graph* didn't change at all.

## Why this matters

In **weeks 22–23**, the Phase III capstone is a multi-agent system, and at its center is a **supervisor** — a `StateGraph` that routes across sub-agents, checkpointed so a long-running multi-agent run survives a crash. *This graph is that supervisor's skeleton.* The plan/retrieve/execute/critique loop you make explicit here, the budget you enforce in the router, the checkpointer that survives the kill — you build all of it again in the capstone, at scale, with multiple agents. The engineer who did this lab has built the capstone's core once already and can reason about it; the engineer who skipped it is meeting `StateGraph` and `SqliteSaver` for the first time under capstone pressure. The agent survived the kill, you have the comparison memo, and you can defend the architecture — that's the week, and it's the foundation of the phase.
