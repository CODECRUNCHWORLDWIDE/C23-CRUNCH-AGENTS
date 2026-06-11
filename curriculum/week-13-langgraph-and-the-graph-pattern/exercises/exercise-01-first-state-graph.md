# Exercise 1 — Your First State Graph

**Goal:** Build a minimal LangGraph `StateGraph` from scratch — three nodes (`plan` → `act` → `critique`) and **one conditional edge** that loops back to `plan` when the critique fails — and *inspect the state at every node* so you see the graph's defining property: explicit, printable state flowing through named nodes. By the end you will have written the four LangGraph primitives (state, node, edge, conditional edge) with your own hands, and you'll understand why a node returns a *partial* dict.

**Estimated time:** 45 minutes. Guided.

---

## Setup

```bash
pip install langgraph langchain-core
# Optional — only if you want the nodes to call a real model in the stretch:
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

This exercise uses a **deterministic stub** for the node "work" so it runs with zero API key and zero cost — the lesson is the *graph*, not the model. The stretch swaps in a real Claude call.

---

## Step 1 — Define the state

The state is a `TypedDict`. Every node reads it and returns updates to it. For a plan/act/critique agent:

```python
from typing import TypedDict, Annotated
import operator


class State(TypedDict):
    task: str                                  # set once at the start
    plan: str                                  # set by the plan node (overwrite)
    result: str                                # set by the act node (overwrite)
    critique: str                              # set by the critique node (overwrite)
    attempts: int                              # incremented each loop (the budget counter)
    log: Annotated[list[str], operator.add]    # APPENDED by every node (reducer!)
```

Two reducer behaviours to notice, because this is the concept the whole week rests on:

- `plan`, `result`, `critique`, `attempts` use the **default** reducer: a node returning `{"plan": "..."}` *overwrites* the old plan.
- `log` is `Annotated[list[str], operator.add]`: a node returning `{"log": ["..."]}` *appends* to the existing log. This is how we accumulate a trace across nodes without clobbering it.

---

## Step 2 — Write the three nodes

A node is a function: state in, **partial** dict out. It returns only the keys it changes.

```python
# A deterministic stub so the exercise runs with no API key. (Stretch swaps in Claude.)
def stub_plan(task: str, attempt: int) -> str:
    return f"attempt {attempt}: break the task into steps and answer"


def stub_act(plan: str) -> str:
    return "answer-v1" if "attempt 1" in plan else "answer-v2 (improved)"


def stub_critique(result: str) -> str:
    # First attempt "fails" so we can SEE the conditional edge loop back; second passes.
    return "fail: first answer was too vague" if result == "answer-v1" else "pass"


def plan_node(state: State) -> dict:
    attempt = state["attempts"] + 1
    plan = stub_plan(state["task"], attempt)
    return {"plan": plan, "attempts": attempt, "log": [f"plan(attempt={attempt})"]}


def act_node(state: State) -> dict:
    result = stub_act(state["plan"])
    return {"result": result, "log": [f"act -> {result}"]}


def critique_node(state: State) -> dict:
    verdict = stub_critique(state["result"])
    return {"critique": verdict, "log": [f"critique -> {verdict}"]}
```

Read what each `return` does: `plan_node` overwrites `plan`, overwrites `attempts` (to the incremented value), and *appends* one line to `log`. It says nothing about `result` or `critique` — those keep their values. That's the partial-state model: return your delta, let LangGraph merge it per the reducer.

---

## Step 3 — Wire the graph (normal edges + the conditional edge)

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)
graph.add_node("plan", plan_node)
graph.add_node("act", act_node)
graph.add_node("critique", critique_node)

graph.add_edge(START, "plan")        # begin at plan
graph.add_edge("plan", "act")        # plan -> act (unconditional)
graph.add_edge("act", "critique")    # act -> critique (unconditional)
```

Now the conditional edge — the one that makes this an agent and not a straight pipeline. After `critique`, a *routing function* reads the state and decides where to go:

```python
def route(state: State) -> str:
    if state["attempts"] >= 3:                  # BUDGET: never loop forever
        return "stop"
    if state["critique"].startswith("pass"):
        return "stop"
    return "retry"                              # critique failed -> back to plan


graph.add_conditional_edges(
    "critique",
    route,
    {"retry": "plan", "stop": END},             # map the router's key -> destination
)

app = graph.compile()
```

The path map `{"retry": "plan", "stop": END}` turns the router's return value into the next node. `"retry"` → `plan` (the loop-back), `"stop"` → `END` (terminate). The `attempts >= 3` check is the **step budget** — without it, a critique that never passed would loop forever.

---

## Step 4 — Run it and watch the state at every node

Use `stream(stream_mode="values")` to see the full state after each node fires:

```python
initial = {"task": "summarize the contract", "plan": "", "result": "",
           "critique": "", "attempts": 0, "log": []}

for step in app.stream(initial, stream_mode="values"):
    last = step["log"][-1] if step["log"] else "(start)"
    print(f"attempts={step['attempts']}  last={last:28s}  critique={step['critique']!r}")

final = app.invoke(initial)
print("\nFINAL log:")
for line in final["log"]:
    print("  ", line)
print("attempts taken:", final["attempts"])
```

Read the output. You should see the graph run `plan → act → critique`, the critique **fail** on the first attempt, the conditional edge route **back to `plan`** (attempt 2), and the second critique **pass**, ending the graph. The `log` (appended by the reducer) is your full node-by-node trace — the observability the week-5 loop couldn't give you for free.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] You defined a `State` `TypedDict` with at least one **reducer key** (`log` with `operator.add`) and at least one default-overwrite key.
- [ ] Three nodes are wired with `add_node`, normal edges with `add_edge`, and a **conditional edge** with `add_conditional_edges(source, router, path_map)`.
- [ ] The router contains a **step budget** (`attempts >= N → stop`) so the loop can't run forever.
- [ ] Running it shows the critique **fail once**, the conditional edge **loop back to `plan`**, and the second pass **terminate** — visible in the streamed trace.
- [ ] You can explain, in one sentence, why each node returns a *partial* dict and not the whole state (LangGraph merges the delta per the reducer; you don't carry everything forward by hand).

---

## Stretch

- **Swap the stub for Claude.** Replace `stub_plan`/`stub_act`/`stub_critique` with real `client.messages.create(model="claude-sonnet-4-6", thinking={"type": "adaptive"}, ...)` calls (and `claude-opus-4-8` for the critique). The graph, edges, and budget do not change — only the node bodies. That model-agnosticism is the lesson.
- **Add a token budget.** Add a `tokens: Annotated[int, operator.add]` key, accumulate `resp.usage` into it in each node, and add `if state["tokens"] > BUDGET: return "stop"` to the router. Now the loop is bounded by *two* budgets.
- **Draw the graph.** Call `app.get_graph().draw_ascii()` (or `.draw_mermaid()`) and confirm the picture matches what you wired: `START → plan → act → critique`, with `critique` branching back to `plan` or to `END`. The diagram *is* the control flow — the thing the loop never gave you.
- **Make it a state machine.** Replace the LLM-style critique with a deterministic check (`return "pass" if len(state["result"]) > 5 else "fail"`). Notice the graph still works identically — and ask yourself whether *this* version even needs an agent, or whether it's a state machine wearing agent clothes (Lecture 2 §5).

---

When this feels comfortable, move to [Exercise 2 — ReAct as a graph](exercise-02-react-as-a-graph.py).
