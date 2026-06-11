# Lecture 1 — When the Loop Becomes a Graph: LangGraph Mechanics and the ReAct Re-Implementation

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can state *when* a hand-rolled agent loop should become a graph (the "fourth `if`" heuristic), build a LangGraph `StateGraph` over a `TypedDict` state with reducers, nodes that return partial state, and normal + conditional edges, and re-implement your week-5 plan→retrieve→execute→critique ReAct agent as four explicit nodes that call the Anthropic SDK (or a local model) directly.

If you remember one sentence from this entire week, remember this one:

> **When the agent gets a fourth tool, you graduate from a loop to a graph. Reach for LangGraph before the loop gets a fourth `if`.**

There's a corollary you should tape next to it:

> **The graph's value is not "it's a framework." The graph's value is explicit state.** A `while` loop hides its state in the program counter and a pile of locals — you can't print it, log it, checkpoint it, or resume it. A `StateGraph` makes the state a typed object you can do all four to. Everything good about this week falls out of that one difference.

In week 5 you wrote a ReAct agent from scratch. You should be proud of it — writing the loop by hand is how you learned what an agent *is* underneath every framework's marketing. But you also felt where it broke down: the loop grew a second tool, a retry branch, a "re-plan if the critique fails" branch, and by the fourth conditional it became a thing you could no longer read at a glance or reason about with confidence. This lecture is about *that exact moment* — the moment the loop wants to become a graph — and the mechanics of making the move with LangGraph.

---

## 1. Why a graph, and why now — the fourth `if`

Start with the agent you already have. Stripped to its skeleton, the week-5 ReAct loop looks like this:

```python
def react_agent(task: str) -> str:
    messages = [{"role": "user", "content": task}]
    while True:
        response = call_model(messages)
        if response.stop_reason == "end_turn":
            return final_text(response)
        if response.stop_reason == "tool_use":
            tool_results = run_tools(response)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            continue
        # ... and what about refusals? max_tokens? a step budget? a retry?
```

This is a fine agent for one or two tools and a single control path. Now watch it grow, the way real agents grow:

- The task needs a **plan** before acting, so you add a planning step at the top.
- The plan needs **retrieval** (your weeks 7–12 pipeline), so you add a branch: "if the plan needs facts, retrieve first."
- After acting, you want a **critique** — did this actually answer the task? — so you add a check, and a branch back to re-plan if it failed.
- You need a **step budget** so it can't loop forever, so you add a counter and an `if step > MAX: break`.

Each of those is an `if`. By the fourth one, the loop has *four* control paths tangled into one function, and three problems have appeared that no amount of cleanup fixes:

1. **You can't see where it is.** "The agent is in the loop" is all you can say. Which phase? Why did it branch back? The state lives in local variables and the program counter, and neither is inspectable from outside.
2. **You can't pause and resume it.** If the process dies after retrieval but before execution, you start over from the top — re-planning, re-retrieving, re-paying for tokens you already spent. There's nothing to resume *from*, because "where I was" was never written down.
3. **You can't reason about the control flow.** A reader has to simulate the loop in their head to know which branches are reachable. There's no diagram, because the structure is implicit.

A **state graph** fixes all three by making the implicit explicit. The phases become **nodes** you can name and watch. The control flow becomes **edges** (and conditional edges) you can draw. And the state becomes a **typed object** you can print, log, checkpoint, and resume. The heuristic — *reach for the graph before the loop gets a fourth `if`* — is really "reach for explicit state before implicit state becomes a liability," and four branches is about where it does.

> **The honest caveat (we'll return to it in Lecture 2):** a graph is not free. It's a dependency, a new vocabulary, and indirection. If your "agent" has *one* fixed path — plan, then answer, no branching — a graph is overkill and a plain function is the honest answer. The graph earns its keep exactly when the control flow *branches and loops*, which is exactly when the loop got its fourth `if`.

---

## 2. The LangGraph mental model in one paragraph

LangGraph gives you four primitives and one rule. The primitives: a **state** (a `TypedDict` describing the data that flows through the run), **nodes** (functions that take the state and return a *partial* update to it), **edges** (which node runs after which), and a **checkpointer** (Lecture 2 — the thing that persists the state). The rule: **a node returns a dict of just the keys it changed, and LangGraph merges that dict into the running state.** That's it. Everything else — conditional edges, reducers, streaming, persistence — is elaboration on those four primitives and that one merge rule. Hold that paragraph in your head and the API stops being a pile of imports and becomes obvious.

```python
pip install langgraph langchain-core
```

`langgraph` is the engine; `langchain-core` is a light dependency it leans on. You do **not** need the heavy LangChain package, and you do **not** need LangChain's LLM wrappers — we call the Anthropic SDK directly inside nodes, which is the clean default this course prefers (more on that in §7).

---

## 3. The state: a `TypedDict` and the reducer

The state is the spine of the graph. It's a `TypedDict` — a plain dict with a type for each key — that every node reads and updates. Define it for the ReAct agent:

```python
from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    task: str                                   # the user's task (set once, at start)
    plan: str                                   # the current plan (set by the plan node)
    docs: Annotated[list[str], operator.add]    # retrieved docs (APPENDED across nodes)
    answer: str                                 # the current answer (set by execute)
    critique: str                               # the critique verdict ("pass"/"fail: ...")
    steps: int                                  # step counter (for the budget)
```

Two keys behave differently, and the difference is the single most important LangGraph concept after "node returns partial state":

- **Default merge = overwrite.** When the `plan` node returns `{"plan": "..."}`, that *replaces* the old `plan`. Most keys want this: the latest plan, the latest answer, the latest critique.
- **Reducer merge = combine.** `docs` is `Annotated[list[str], operator.add]`. The annotation says "when a node returns a `docs` value, **add** it to the existing list, don't overwrite." So if the `retrieve` node runs twice (re-plan loop) and returns three docs each time, the state ends up with six docs, not three. The reducer is `operator.add`, which for lists is concatenation.

This is why nodes return *partial* dicts. A node that only changes the plan returns `{"plan": "..."}` and says nothing about `docs` or `answer` — those keys keep their values. The reducer system means you never write the "carry everything forward" boilerplate that a manual loop forces on you; you return only your delta, and LangGraph merges it correctly per key.

The built-in reducer you'll meet most is `add_messages`, for accumulating a list of chat messages:

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`add_messages` is `operator.add` plus two niceties: it *appends* new messages and *dedupes/updates* by message id (so re-emitting a message with the same id replaces it rather than duplicating). When your nodes pass around LangChain message objects, use `add_messages`; when they pass around plain strings or your own structures, `operator.add` (or a custom reducer function) is right. Choosing the wrong reducer is the classic LangGraph state bug: use overwrite where you meant append and your retrieved docs vanish on the second pass; use append where you meant overwrite and your `plan` becomes a growing list of every plan you ever made.

---

## 4. Nodes: functions that return partial state

A node is a function. It takes the state, does work, and returns a partial state dict. Here's the `plan` node for the ReAct agent, calling Claude directly:

```python
import anthropic

client = anthropic.Anthropic()
NODE_MODEL = "claude-sonnet-4-6"          # default node model
CRITIQUE_MODEL = "claude-opus-4-8"        # harder reasoning for the critique node


def plan_node(state: AgentState) -> dict:
    """Read the task; produce a short plan. Returns ONLY the keys it changes."""
    prompt = (
        f"Task: {state['task']}\n\n"
        "Write a one-paragraph plan for answering this. If it needs facts you "
        "don't have, say 'RETRIEVE' and what to look up."
    )
    resp = client.messages.create(
        model=NODE_MODEL,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    plan = next(b.text for b in resp.content if b.type == "text")
    return {"plan": plan, "steps": state["steps"] + 1}
```

Notes that matter:

- **It returns a partial dict.** `{"plan": ..., "steps": ...}` — not the whole state. LangGraph merges these two keys; everything else is untouched.
- **It calls the SDK directly.** No LangChain `ChatAnthropic` wrapper, no chain. Inside the node it's just the Anthropic SDK you've used since week 5 — `client.messages.create(...)`, iterate `resp.content`, take the text block. This is the thin path the course prefers.
- **Claude facts, applied:** `model="claude-sonnet-4-6"`, `thinking={"type": "adaptive"}`, no `temperature`/`top_p`/`top_k` (they 400 on Sonnet 4.6 / Opus 4.8), no `budget_tokens` (removed; use `thinking` + `output_config={"effort": ...}` instead). These are not optional stylistic choices — passing the removed params raises a 400.

The `retrieve` node calls *your* retriever from weeks 7–12 — the graph orchestrates your RAG, it doesn't replace it:

```python
def retrieve_node(state: AgentState) -> dict:
    """Call the weeks 7-12 retriever. Returns docs to be APPENDED (operator.add)."""
    from crunchrag_embed import store, embedders     # your prior-weeks package
    bge = embedders.load("bge")
    hits = store.knn("clauses", bge.embed_query(state["task"]), k=3)
    docs = [h.text for h in hits]
    return {"docs": docs, "steps": state["steps"] + 1}   # docs is appended, not replaced
```

The `execute` node answers from the retrieved docs, and the `critique` node judges the answer (using the stronger Opus model, because judging is the hard reasoning):

```python
def execute_node(state: AgentState) -> dict:
    context = "\n---\n".join(state["docs"]) or "(no documents retrieved)"
    prompt = (
        f"Task: {state['task']}\nPlan: {state['plan']}\n\n"
        f"Context:\n{context}\n\nAnswer the task using the context. Be specific."
    )
    resp = client.messages.create(
        model=NODE_MODEL, max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    answer = next(b.text for b in resp.content if b.type == "text")
    return {"answer": answer, "steps": state["steps"] + 1}


def critique_node(state: AgentState) -> dict:
    prompt = (
        f"Task: {state['task']}\nAnswer: {state['answer']}\n\n"
        "Does the answer fully and correctly address the task? "
        "Reply 'pass' or 'fail: <one-line reason>'."
    )
    resp = client.messages.create(
        model=CRITIQUE_MODEL, max_tokens=512,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = next(b.text for b in resp.content if b.type == "text").strip().lower()
    return {"critique": verdict, "steps": state["steps"] + 1}
```

Four nodes, four functions, each reading the state and returning its delta. There is no loop variable, no `messages` list threaded by hand, no "carry everything forward." Each node's job is small and local, and that locality is exactly what makes the graph readable where the loop wasn't.

---

## 5. Edges: normal and conditional

Nodes are the *what*; edges are the *when*. You wire them on a `StateGraph`:

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("execute", execute_node)
graph.add_node("critique", critique_node)

graph.add_edge(START, "plan")          # the graph starts at plan
graph.add_edge("plan", "retrieve")     # plan -> retrieve (unconditional)
graph.add_edge("retrieve", "execute")  # retrieve -> execute (unconditional)
graph.add_edge("execute", "critique")  # execute -> critique (unconditional)
```

`START` and `END` are sentinels. `add_edge(START, "plan")` says "begin here." A **normal edge** (`add_edge("plan", "retrieve")`) is unconditional: after `plan`, always go to `retrieve`. The interesting one is the **conditional edge** after `critique` — the branch that makes this a real agent and not a fixed pipeline:

```python
def route_after_critique(state: AgentState) -> str:
    """Decide what runs after critique. This is the 'fourth if', made a function."""
    if state["steps"] >= 8:                 # BUDGET: never loop forever (Lecture 2)
        return "end"
    if state["critique"].startswith("pass"):
        return "end"
    return "replan"                         # critique failed -> go back to plan


graph.add_conditional_edges(
    "critique",
    route_after_critique,
    {"replan": "plan", "end": END},        # map the function's return -> a node
)

app = graph.compile()
```

Read `add_conditional_edges` carefully, because it's the heart of the whole pattern:

- **First arg:** the source node (`"critique"`).
- **Second arg:** a *routing function* that takes the state and returns a string key.
- **Third arg:** a *path map* `{key: destination}` that turns the returned key into the next node.

So after `critique`, LangGraph calls `route_after_critique(state)`. If it returns `"end"`, the graph goes to `END` (stops). If it returns `"replan"`, the graph goes back to `"plan"` — and the whole plan→retrieve→execute→critique cycle runs again, this time with the previous attempt's state still present (the appended `docs`, the incremented `steps`). That loop-back edge is the week-5 `while` loop's "re-plan on failure" branch — except now it's a *named function returning a destination*, not an `if` buried in a control-flow tangle. You can read it, test it in isolation, and draw it.

Notice the budget check sitting in the router. `if state["steps"] >= 8: return "end"` is the step budget — the guard that makes the graph terminate even if the critique never passes. **This is mandatory.** An agent with a re-plan loop and no budget is an agent that can run forever and bankrupt you; the conditional edge is exactly the right place to enforce the cap, because it's the only place the loop can close. We'll deepen budgets (token, time) in Lecture 2, but the step budget lives here, in the router, from day one.

---

## 6. Compile, invoke, stream

`graph.compile()` turns the definition into a runnable app. Then you run it. The simplest call is `invoke`, which runs the whole graph and returns the final state:

```python
final_state = app.invoke({
    "task": "What is the confidentiality duration after termination?",
    "plan": "", "docs": [], "answer": "", "critique": "", "steps": 0,
})
print(final_state["answer"])
print("steps taken:", final_state["steps"])
```

You pass the *initial* state (every key seeded, even the ones nodes will fill), and you get back the *final* state after the graph reaches `END`. Note that `docs` started as `[]` and ended populated — the reducer did its job.

But `invoke` is the opaque option, and opacity is the thing we left the loop to escape. The observability win is `stream`, which yields each node's output as it fires:

```python
for event in app.stream(
    {"task": "...", "plan": "", "docs": [], "answer": "", "critique": "", "steps": 0},
    stream_mode="updates",      # yield each node's partial-state update as it runs
):
    for node_name, update in event.items():
        print(f"node={node_name:9s} -> {list(update.keys())}")
# node=plan      -> ['plan', 'steps']
# node=retrieve  -> ['docs', 'steps']
# node=execute   -> ['answer', 'steps']
# node=critique  -> ['critique', 'steps']
# (if critique failed: the cycle repeats from plan)
```

`stream_mode="updates"` yields *what each node changed*; `stream_mode="values"` yields the *full state after each node*; `stream_mode="debug"` yields the most detail (useful when you're chasing why a conditional edge routed the way it did). This is the observability the `while` loop couldn't give you: you watch the agent move node by node, see exactly which keys each node wrote, and — when the critique fails — watch it loop back to `plan` *visibly*, instead of guessing from a debugger why the loop went around again. In the challenge you'll measure this concretely: "observability" is one of the three axes (alongside lines of code and resumability) on which the graph beats the loop, and `stream` is why.

---

## 7. Direct SDK in a node vs the `ChatAnthropic` wrapper

You'll see two ways to call the model inside a node. The course default is the one above — the **raw Anthropic SDK**:

```python
def plan_node(state):
    resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                                  thinking={"type": "adaptive"},
                                  messages=[{"role": "user", "content": prompt}])
    return {"plan": next(b.text for b in resp.content if b.type == "text")}
```

The alternative is LangChain's wrapper, `ChatAnthropic`:

```python
from langchain_anthropic import ChatAnthropic            # pip install langchain-anthropic
llm = ChatAnthropic(model="claude-sonnet-4-6", thinking={"type": "adaptive"})

def plan_node(state):
    msg = llm.invoke([("user", prompt)])
    return {"plan": msg.content}
```

Both work. `ChatAnthropic` is convenient if you're already deep in LangChain's ecosystem (it integrates with LangChain tools, output parsers, and streaming abstractions). But this course **prefers the raw SDK inside the node**, for a reason that is a through-line of the whole week: **the framework should orchestrate, not abstract away the model call.** LangGraph's value is the *graph* — the explicit state, the edges, the checkpointer. It does not need to also hide your model call behind a wrapper, and when it does, you've added a layer of indirection over the one part of the system you most need to see clearly (the exact prompt, the exact params, the exact response). The lecture's position — and the course's — is: use LangGraph for what it's uniquely good at (state + edges + persistence), and call the model the way you've called it since week 5 (the SDK). That keeps the abstraction *thin* and the model call *legible*. We'll sharpen this argument in Lecture 2 when we contrast LangGraph's explicitness with CrewAI's leaky role-play abstraction — but the principle starts here: a node is just a function, and inside it, the model call should be the plain one.

---

## 8. The week-5 loop and the graph, side by side

Make the correspondence exact, because it's the whole point of the week. Here is the loop's structure and the graph's structure, mapped phase for phase:

| Week-5 loop | Week-13 graph |
|---|---|
| `while True:` top | `START` → `plan` |
| "decide what to do" | `plan_node` (a node) |
| "if I need facts, retrieve" branch | `plan` → `retrieve` edge |
| "call the tool" | `execute_node` (a node) |
| "did it work?" check | `critique_node` (a node) |
| `if not ok: continue` (re-plan) | conditional edge `critique` → `plan` |
| `if step > MAX: break` | budget check in the router → `END` |
| local variables (`messages`, `step`, `docs`) | the `AgentState` `TypedDict` |
| nothing (state lives in the call stack) | the checkpointer (Lecture 2) |

The last row is the kicker. In the loop, "where I am and what I've done" lives in the call stack — ephemeral, un-inspectable, gone if the process dies. In the graph, it lives in the `AgentState`, which the checkpointer can write to disk after every node. *That* is the difference between an agent that restarts from zero on a crash and one that survives the kill — and it's the subject of Lecture 2.

Everything else on this list is a readability and observability win: each phase is a named node you can test alone, each branch is an edge you can draw, and the control flow is a graph you can look at instead of a loop you have to simulate in your head. The graph didn't make the agent smarter. It made the agent *legible, resumable, and bounded* — which, for anything past a demo, is what "production" means.

---

## 9. Recap

You should now be able to:

- State **when a loop becomes a graph** — the fourth `if` heuristic — and explain that the real win is *explicit state* (printable, loggable, checkpointable, resumable) replacing the loop's implicit state in the call stack.
- Build a **`StateGraph` over a `TypedDict`**: define the state, choose the right **reducer** per key (`operator.add` / `add_messages` to append, default overwrite otherwise), and know that a node returns a **partial** dict that LangGraph merges.
- Write **nodes as functions** that call the Anthropic SDK directly (`claude-sonnet-4-6` default, `claude-opus-4-8` for the critique, adaptive thinking, no `temperature`/`budget_tokens`) and reuse your weeks 7–12 retriever inside the `retrieve` node.
- Wire **normal edges** and **conditional edges** (`add_conditional_edges(source, router_fn, path_map)`), and put the **step budget** in the router so the re-plan loop always terminates.
- **Compile, invoke, and stream** the graph, using `stream(stream_mode="updates")` to watch each node fire — the observability the `while` loop couldn't give you.
- Argue for the **thin path**: LangGraph orchestrates; call the model with the raw SDK inside the node rather than hiding it behind a wrapper.

Next: how the state *survives a process kill* — checkpointers (`MemorySaver` vs `SqliteSaver`), `thread_id`, resume-after-crash, and time-travel — plus the multi-agent patterns (supervisor / swarm / hierarchical) and the honest comparison against AutoGen and CrewAI. Continue to [Lecture 2 — Persistence, Patterns, and Framework Honesty](./02-persistence-patterns-and-framework-honesty.md).

---

## References

- *LangGraph — low-level concepts (StateGraph, state, nodes, edges, reducers)*: <https://langchain-ai.github.io/langgraph/concepts/low_level/>
- *LangGraph — Graph API how-tos (`add_node`, `add_edge`, `add_conditional_edges`)*: <https://langchain-ai.github.io/langgraph/how-tos/>
- *LangGraph — streaming (`stream_mode`)*: <https://langchain-ai.github.io/langgraph/how-tos/streaming/>
- *ReAct: Synergizing Reasoning and Acting in Language Models* — Yao et al., 2022: <https://arxiv.org/abs/2210.03629>
- *Reflexion: Language Agents with Verbal Reinforcement Learning* — Shinn et al., 2023 (the critique-and-retry idea): <https://arxiv.org/abs/2303.11366>
- *Anthropic — tool use overview* (the contract you call inside a node): <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>
- *Anthropic — adaptive thinking & effort* (Sonnet 4.6 / Opus 4.8 params): <https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking>
- *`langchain-anthropic` — `ChatAnthropic`* (the wrapper alternative we mention but don't default to): <https://python.langchain.com/docs/integrations/chat/anthropic/>
