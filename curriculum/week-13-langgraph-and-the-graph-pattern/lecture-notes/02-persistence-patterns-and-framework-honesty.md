# Lecture 2 — Persistence, Multi-Agent Patterns, and Framework Honesty

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can attach a `SqliteSaver` checkpointer with a `thread_id` so a LangGraph agent survives a process kill and resumes from its last checkpoint, list and replay the checkpoint history (time-travel), build the three multi-agent patterns (supervisor, swarm, hierarchical) in real LangGraph code, and compare LangGraph against AutoGen and CrewAI honestly — including *why CrewAI's role-play abstraction leaks* and *when a deterministic state machine beats an agent entirely*.

Lecture 1 made the agent a graph: explicit nodes, explicit edges, explicit state. This lecture makes the state **durable**, the graph **multi-agent**, and your framework choices **honest**.

> **Persistence is the line between a demo and production.** A graph that runs in one process and forgets everything on a crash is a notebook toy. A graph that writes its state after every node and resumes from where it died is a system. The `SqliteSaver` is a four-line change that crosses that line — and "the agent survived the kill" is how we prove we crossed it.

---

## Part 1 — Checkpointers: how state survives a kill

### 1.1 The checkpointer in one sentence

A **checkpointer** writes the graph's state to durable storage **after every node**, keyed by a **`thread_id`**, so that invoking the graph again with the *same* `thread_id` resumes from the last saved state instead of starting over. That's the whole idea. Everything below is mechanics.

In Lecture 1, `app = graph.compile()` gave you a graph with no memory: each `invoke` started from the initial state you passed and forgot everything when it returned. To add persistence, you pass a checkpointer to `compile`:

```python
app = graph.compile(checkpointer=checkpointer)
```

and you pass a `thread_id` when you invoke:

```python
config = {"configurable": {"thread_id": "task-42"}}
app.invoke(initial_state, config=config)
```

The `thread_id` is the key. Two invokes with the same `thread_id` share state; two with different `thread_id`s are independent runs. The checkpointer stores, under each `thread_id`, the full state after each node — so the second invoke can pick up exactly where the first left off.

### 1.2 `MemorySaver` vs `SqliteSaver` — RAM dies, disk survives

LangGraph ships two checkpointers you'll use, and the difference is the whole point of this week's promise:

```python
# In-memory: state lives in a dict in THIS process. Dies when the process dies.
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
```

```python
# On-disk SQLite: state lives in a file. SURVIVES a process kill.
# pip install langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    app.invoke(initial_state, config={"configurable": {"thread_id": "task-42"}})
```

`MemorySaver` is perfect for tests and for human-in-the-loop *within one process* (you can pause and resume across `invoke` calls in the same run). But it lives in a Python dict — kill the process and the state is gone. `SqliteSaver` writes to a file on disk. Kill the process, start a fresh one, open the *same* SQLite file, invoke with the *same* `thread_id`, and the graph resumes from the last checkpoint. That is the difference between an agent that restarts from zero on a crash and one that survives the kill.

Note the `with` block: `SqliteSaver.from_conn_string(...)` is a context manager that opens (and closes) the SQLite connection. For production you'd use `langgraph-checkpoint-postgres`'s `PostgresSaver` — same interface (`BaseCheckpointSaver`), different backend — so a Postgres-backed agent survives not just a process kill but a host failure, and multiple workers can resume the same threads. The lab default is SQLite because it's zero-infrastructure (a file), and the *interface* is identical, so what you learn on SQLite transfers to Postgres unchanged.

### 1.3 Surviving the kill, concretely

Here's the demo that *is* this week's promise. Run a graph partway, simulate a crash, then resume in a fresh process:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

THREAD = {"configurable": {"thread_id": "task-42"}}


def run_once(initial_state: dict | None) -> None:
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
        app = graph.compile(checkpointer=cp)
        if initial_state is not None:
            # First run: start fresh. (A simulated crash node raises after retrieve.)
            app.invoke(initial_state, config=THREAD)
        else:
            # Resume run: pass None as the input to CONTINUE from the last checkpoint.
            app.invoke(None, config=THREAD)
        snapshot = app.get_state(THREAD)
        print("recovered state keys:", {k: bool(v) for k, v in snapshot.values.items()})
```

The mechanics that make this work:

- **First process:** `invoke(initial_state, config=THREAD)` runs `plan`, then `retrieve` — and the checkpointer writes the state to `checkpoints.sqlite` after *each* node. If the process then dies (a `SystemExit`, a `kill -9`, a server restart), the file on disk already holds the state as of the last completed node.
- **Second process (fresh Python, same file, same `thread_id`):** `invoke(None, config=THREAD)` — note the `None`. Passing `None` as the input tells LangGraph "don't start over; **resume** from the checkpoint for this thread." The graph picks up at the node *after* the last one that completed, with `plan` and `docs` already in state. `retrieve` does **not** run again — it already ran, its result is in the checkpoint — so you don't re-pay for those tokens.
- **`app.get_state(config)`** returns a `StateSnapshot` whose `.values` is the recovered state. That's how you inspect what survived.

The exercise (`exercise-03-survive-the-kill.py`) does exactly this and prints a `PASS` line proving `retrieve` ran *once*, not twice — the agent survived the kill. If this had been a week-5 `while` loop, the second process would start from the top: re-plan, re-retrieve, re-pay. The checkpointer is the four-line change that makes resumability real.

### 1.4 Time-travel: replaying the checkpoint history

Because the checkpointer stores state after *every* node, you don't just have the latest state — you have the *history*. LangGraph exposes it:

```python
with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
    app = graph.compile(checkpointer=cp)
    for snap in app.get_state_history(THREAD):
        print(snap.config["configurable"]["checkpoint_id"], "->",
              snap.next)   # the node(s) that would run next from this checkpoint
```

This is **time-travel**: you can list every checkpoint, pick one from *before* a bad decision, and resume the graph from *there* with a corrected input — rather than restarting the whole run. It's the agent equivalent of `git log` + `git checkout`: every node boundary is a commit you can return to. For debugging a misbehaving agent ("it went wrong after `retrieve` on the third loop — let me resume from the checkpoint just before that and feed it a better plan"), time-travel turns a re-run-and-pray cycle into a targeted replay. You won't need it for the basic labs, but it's a stretch goal and a genuine production tool, and it exists *only because the state is explicit and checkpointed* — the loop had no such history to travel through.

### 1.5 What you can and can't put in the state

The checkpointer serializes the state to disk, and that constraint is load-bearing: **the state must hold plain, serializable data — and nothing else.** Strings, ints, floats, lists, dicts of those: fine. A live database connection, an open file handle, a model client object, a lambda, a thread, a socket: *not* fine. The instant one of those lands in the state, one of two things happens — the checkpointer raises trying to pickle it, or (worse) it silently writes something that can't be reconstructed, and your resume in a fresh process comes back with a dead handle.

The rule that keeps you out of trouble: **the state is data; the machinery is not.** The `retrieve` node calls your retriever and puts the *retrieved text* in `docs` — it does not put the *retriever object* in the state. The model client is a module-level singleton the nodes close over, not a state key. If you find yourself wanting to stash a connection "so the next node can reuse it," that's the tell that you're about to break resume: reconstruct the machinery in each node (or close over it), and keep the state to the facts the run has accumulated. This is the challenge's Trap 2, and it's the single most common reason a resume that "should work" doesn't.

### 1.6 Checkpoints are per-thread, and that's the unit of resumability

One more property to internalize: checkpoints are namespaced by `thread_id`. In the benchmark runner, **each task gets its own `thread_id`** — `task-0`, `task-1`, …, `task-24`. That's deliberate: if the process dies on task 17, you resume *task 17* (its own thread, its own checkpoint history), not the whole benchmark. The other 16 completed tasks aren't re-run; task 17 picks up from its last node. The `thread_id` is therefore your unit of resumability — choose it to match the unit of work you want to be able to resume. One thread per conversation, per task, per job: whatever granularity you'd want to restart at on a crash.

---

## Part 2 — Budgets: the non-negotiable guard

Before patterns, a hard rule the whole course enforces: **an agent with a loop must have a budget.** Lecture 1 put a *step* budget in the conditional-edge router. Production needs three:

- **Step budget** — cap the number of node executions (or re-plan loops). The router check `if state["steps"] >= MAX: return "end"`. Stops the "critique never passes, so it loops forever" failure mode.
- **Token budget** — cap cumulative tokens across all node LLM calls. Accumulate `resp.usage.input_tokens + resp.usage.output_tokens` into the state with an `operator.add` reducer, and end the graph when it crosses a ceiling. Stops a runaway agent from bankrupting you.
- **Time budget** — cap wall-clock. Record a start time in the initial state; in the router, `if time.monotonic() - state["t0"] > MAX_SECONDS: return "end"`. Stops an agent that's stuck calling a slow tool from blocking a queue forever.

```python
def route_after_critique(state: AgentState) -> str:
    if state["steps"] >= STEP_BUDGET:        return "end"   # step cap
    if state["tokens"] >= TOKEN_BUDGET:      return "end"   # token cap
    if time.monotonic() - state["t0"] > TIME_BUDGET: return "end"  # time cap
    return "end" if state["critique"].startswith("pass") else "replan"
```

The conditional edge is the right home for all three, because it's the one place the loop can close. The through-line: **the graph gives you the structure to enforce budgets cleanly; use it.** A graph without a budget is just a `while True` with extra steps — and a more expensive way to loop forever. The mini-project makes budgets a graded requirement, and the challenge's "trap" section calls out the unbounded loop as the classic failure.

---

## Part 3 — Multi-agent patterns: supervisor, swarm, hierarchical

One graph with four nodes is a single agent. Real systems compose *multiple* agents. There are three patterns, and the difference between them is **who decides what runs next**.

### 3.1 Supervisor — a central router delegates to sub-agents

A **supervisor** is a router node that decides which specialized sub-agent to call, calls it, collects the result, and decides again — until done. The sub-agents don't know about each other; the supervisor is the only one with the big picture. This is the most common pattern and the one your capstone uses.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator


class TeamState(TypedDict):
    task: str
    messages: Annotated[list[str], operator.add]
    next: str                       # the supervisor's routing decision


def supervisor_node(state: TeamState) -> dict:
    """Decide which sub-agent handles the task next. A cheap model is fine here."""
    prompt = (
        f"Task: {state['task']}\nProgress: {state['messages'][-3:]}\n\n"
        "Reply with ONE of: 'researcher', 'mathematician', 'FINISH'."
    )
    resp = client.messages.create(
        model="claude-haiku-4-5", max_tokens=32,        # routing is a cheap decision
        messages=[{"role": "user", "content": prompt}],
    )
    choice = next(b.text for b in resp.content if b.type == "text").strip().lower()
    return {"next": "FINISH" if "finish" in choice else choice}


def researcher_node(state: TeamState) -> dict:
    # ... calls the weeks 7-12 retriever + Claude, returns a finding ...
    return {"messages": ["[researcher] found: ..."]}


def mathematician_node(state: TeamState) -> dict:
    # ... calls Claude (or a calculator tool) for the arithmetic ...
    return {"messages": ["[mathematician] computed: ..."]}


def route_from_supervisor(state: TeamState) -> str:
    return END if state["next"] == "FINISH" else state["next"]


g = StateGraph(TeamState)
g.add_node("supervisor", supervisor_node)
g.add_node("researcher", researcher_node)
g.add_node("mathematician", mathematician_node)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route_from_supervisor,
                        {"researcher": "researcher", "mathematician": "mathematician", END: END})
g.add_edge("researcher", "supervisor")     # sub-agents report BACK to the supervisor
g.add_edge("mathematician", "supervisor")
team = g.compile(checkpointer=checkpointer)
```

The shape: `supervisor` routes out to a sub-agent, the sub-agent does its work and reports *back* to `supervisor`, and the supervisor decides again — a star topology with the supervisor at the center. It's just a `StateGraph` with conditional edges, exactly the primitives from Lecture 1, scaled up. Note the cheap model (`claude-haiku-4-5`) on the routing decision and the stronger models inside the sub-agents — you pay for intelligence where it's needed and keep the orchestration layer cheap. **This graph is the skeleton of the weeks 22–23 capstone supervisor.**

### 3.2 Swarm — peers hand off to each other

A **swarm** has no central router. Each agent decides, *itself*, when to hand control to a peer. Agent A works until it hits something B is better at, then hands off to B directly; B works, maybe hands to C or back to A. Control is a baton the agents pass; there's no supervisor holding the map.

In LangGraph, a swarm is modeled as nodes whose conditional edges point at *each other*, with each node's routing function deciding "do I keep this, or hand off?":

```python
def agent_a(state) -> dict:
    # ... does work; if it needs B's specialty, signals a handoff ...
    return {"messages": ["..."], "handoff": "agent_b" if needs_b else ""}

def route_a(state) -> str:
    return state["handoff"] or END

g.add_conditional_edges("agent_a", route_a, {"agent_b": "agent_b", END: END})
g.add_conditional_edges("agent_b", route_b, {"agent_a": "agent_a", END: END})
```

Swarms shine when the agents are genuine *peers* with overlapping context and the handoff logic is local ("I, agent A, know when to call B") rather than central ("the supervisor decides"). They're harder to reason about than a supervisor — control can ping-pong — which is exactly why the budget matters even more. Use a swarm when there's no natural hierarchy and the agents collaborate as equals; use a supervisor when one coordinator *should* hold the plan.

### 3.3 Hierarchical — supervisors of supervisors

When a single supervisor's roster of sub-agents grows too big (too many tools, too many specialists for one router to choose well), you go **hierarchical**: a top supervisor that routes to *mid-level supervisors*, each of which routes to its own team of sub-agents. It's a tree. A top-level "research-org" supervisor might delegate to a "web-research" supervisor (which has its own search/scrape/summarize sub-agents) and a "data-analysis" supervisor (which has its own query/compute/chart sub-agents).

In LangGraph, each supervisor is a compiled graph, and a parent graph treats a child graph as a *node* (a compiled graph is itself a runnable that can be added with `add_node`). The same `StateGraph` primitives nest. Hierarchical is the pattern for genuinely large systems; for everything in this course up to the capstone, a single supervisor is enough — but you should know the escalation path: **one agent → supervisor over a few sub-agents → hierarchy of supervisors**, reached only when the level below it stops being able to choose well.

The picking rule, plainly:

| Pattern | Who routes | Use when |
|---|---|---|
| **Supervisor** | one central node | one coordinator should hold the plan and delegate to specialists (the default; the capstone shape) |
| **Swarm** | each agent, locally | agents are peers with no natural hierarchy and handoff logic is local |
| **Hierarchical** | a tree of supervisors | one supervisor's roster is too big to route well; you need supervisors of supervisors |

---

## Part 4 — Framework honesty: LangGraph vs AutoGen vs CrewAI

The syllabus asks for an *honest* comparison, so here it is, opinions included. All three are open source; all three can build a working multi-agent system. They differ in *what they make explicit* — and that difference is everything once you have to debug, checkpoint, and resume a real system.

### 4.1 The three, in one line each

- **LangGraph** — an explicit **state graph**. You define the state, the nodes, and the edges. Control flow is a graph you draw; state is a typed object you checkpoint. Verbose for simple cases, unbeatable for systems you have to inspect and resume.
- **AutoGen** (Microsoft) — **conversational multi-agent**. Agents talk to each other in a chat; the conversation *is* the control flow. Great for open-ended, exploratory collaboration ("a coder agent and a critic agent hash it out"). The control flow is emergent from the chat, which is its strength (flexible) and its weakness (hard to make deterministic or resumable).
- **CrewAI** — a **role-play abstraction**: you define `Agent`s (a "researcher", a "writer"), `Task`s, and a `Crew` that runs them. It reads like assembling a team of humans. Fast to a demo; the abstraction is the appeal.

### 4.2 Why CrewAI's role-play abstraction leaks

CrewAI's pitch is seductive: describe your agents as *roles* — "You are a senior researcher", "You are an editor" — give them tasks, put them in a crew, and run. For a linear, demo-shaped workflow (research → write → edit), it's genuinely quick and the code reads like English.

The abstraction **leaks** the moment you need real control flow, and here is exactly where:

- **Branching and loops are second-class.** The role-play model is built around a roster of tasks run in sequence (or a simple manager delegating). The instant you need "if the editor rejects it, loop back to the writer *with the editor's notes*, but only twice, and if it still fails escalate to a human" — the conditional-edge logic that LangGraph makes a one-line router — you're fighting the abstraction, threading control through agent *prose* instead of code. The role metaphor has no clean place to put a `if critique fails and attempts < 2: re-plan` edge, so you end up encoding control flow in prompts, which is exactly the un-inspectable, un-resumable place control flow should not live.
- **State is implicit again.** The whole win of Lecture 1 was *explicit state you can checkpoint*. CrewAI's state is largely the agents' conversation and task outputs — back to implicit. Asking "what is the exact state, and can I serialize it and resume after a crash?" is awkward; the abstraction wasn't built around a typed, durable state object.
- **The role is not the structure.** "Researcher", "Writer", "Editor" are *labels*, not control flow. The labels make the demo readable and the *system* opaque: the actual sequencing, branching, and termination conditions are hidden inside the framework's task-running logic and the agents' prompts, not visible as a graph you can draw and reason about.

This is not "CrewAI is bad." It's "CrewAI optimizes for *time-to-demo on a linear workflow*, and that optimization leaks the moment the workflow branches, loops, needs a hard budget, or needs to survive a crash." Those are exactly the properties this week is about. The honest summary: **CrewAI's role abstraction is appealing because it hides the graph — and that's also why it leaks, because the graph is the thing you need when the system gets real.** LangGraph makes you write the graph; that verbosity *is* the feature.

Make the contrast concrete. The same "research then write, loop if the editor rejects" workflow, sketched in each:

```python
# CrewAI — reads like assembling a team. Great until the loop.
from crewai import Agent, Task, Crew
researcher = Agent(role="Senior Researcher", goal="find the facts", backstory="...")
writer     = Agent(role="Editor",            goal="write the brief",  backstory="...")
crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()
# Now add: "if the editor rejects it, loop back to the writer WITH the editor's
# notes, at most twice, else escalate to a human." There is no clean, inspectable
# place for that conditional-with-budget-and-loop. You end up steering it through
# the agents' prose and task descriptions — control flow living in prompts.
```

```python
# LangGraph — you write the graph. The reject-loop-with-budget is one router.
def route_after_edit(state) -> str:
    if state["attempts"] >= 2:               return "escalate"   # budget
    return "end" if state["approved"] else "rewrite"             # the loop
g.add_conditional_edges("edit", route_after_edit,
                        {"rewrite": "write", "escalate": "human", "end": END})
```

The LangGraph version is more lines up front and *radically* clearer about the one thing that matters: where the loop is, what bounds it, and what happens at the boundary. That's the trade the whole week argues for — explicitness over a friendly surface, once the control flow is real.

### 4.3 AutoGen, fairly

AutoGen deserves a fairer hearing, because its model is genuinely different rather than a leakier version of LangGraph's. AutoGen's control flow *is* the conversation — agents send each other messages, and the "program" is which agent speaks when. For **open-ended, exploratory** tasks where you *want* emergent collaboration ("two agents debate a design until they converge"), the conversational model fits the problem: you don't *want* to specify the control flow in advance, because the whole point is that it emerges. Where it struggles is the same place CrewAI struggles and LangGraph shines: **determinism and resumability.** A conversation is hard to checkpoint-and-resume node-by-node, hard to bound with a clean budget, and hard to draw as a graph. So: reach for AutoGen when the task is genuinely open-ended chat-style collaboration and you don't need deterministic, resumable control flow; reach for LangGraph when you do — which, for anything production, is most of the time.

### 4.4 The course's position

LangGraph wins *for this course's goals* — building agents you have to **debug, checkpoint, resume, and bound** — because it makes the four things you most need to control (state, edges, budgets, persistence) **explicit**. AutoGen and CrewAI hide some of those behind a friendlier surface, and the hiding is great for a demo and a liability for a system. You learned the loop by hand in week 5 precisely so you'd recognize what a framework is doing *for* you and *to* you; LangGraph does the least *to* you (it's a thin graph engine over your own functions) while giving the most structure, which is why it's the course default. Use the right tool — AutoGen for open-ended chat collaboration, a plain function for a fixed pipeline — but when control flow branches and the system has to survive contact with production, the explicit graph is the honest choice.

---

## Part 5 — When a state machine beats an agent

The last and most important judgment of the week: **not everything that looks like an agent should be one.** An "agent" — an LLM deciding its own control flow at runtime — is the right tool when the path through the problem is *genuinely unknown in advance* and depends on what the model discovers as it goes. But a large fraction of "agent" workflows have a *fixed, known* control flow that someone dressed up as an agent because agents are fashionable.

When the transitions are fixed and known, a **deterministic state machine** — a graph whose edges are plain `if`s, not LLM decisions — beats an agent on every axis that matters:

- **Cheaper** — no LLM call to *decide the routing*, only LLM calls where you actually need generation. A router that calls a model to pick between two known branches is paying for a coin flip.
- **Faster** — no round-trip to decide what's already decided.
- **Can't loop forever** — fixed transitions terminate by construction; there's no "the model keeps choosing re-plan" failure mode.
- **Debuggable** — the control flow is your code, fully deterministic, reproducible.

The tell: **if you can draw the control-flow diagram before running it, and it doesn't change run to run, you have a state machine, not an agent — so build a state machine.** "Extract these five fields, validate them, then format the output" is a pipeline (three nodes, two fixed edges) — making it an "agent" that *decides* to extract, then *decides* to validate, is paying LLM latency and cost for decisions that were never in doubt, and adding a loop-forever risk for nothing. LangGraph builds *both*: a `StateGraph` with conditional edges driven by `if state[...]` (deterministic) is a state machine; the same engine with conditional edges driven by an LLM's output is an agent. Use the model to *decide routing* only where the routing is genuinely open-ended. Where it isn't, the `if` is cheaper, faster, safer, and more honest. The skill this week is partly fluency in the graph and partly the judgment to know that the best agent is often the one you didn't build — because a state machine was the right answer.

---

## Part 6 — Recap

You should now be able to:

- **Attach a checkpointer**: `MemorySaver` (RAM, dies with the process) vs `SqliteSaver` (disk, survives a kill), pass it to `compile(checkpointer=...)`, and use a `thread_id` in `config={"configurable": {"thread_id": ...}}` so the same thread resumes.
- **Survive a process kill**: run partway (state written after each node), crash, then in a fresh process `invoke(None, config=THREAD)` to **resume** from the last checkpoint — without re-running completed nodes or re-paying their tokens.
- **Time-travel** over `get_state_history(...)` to list checkpoints and replay from before a bad decision.
- **Enforce budgets** (step, token, time) in the conditional-edge router — the mandatory guard against the unbounded loop.
- **Build the three patterns**: supervisor (central router → sub-agents → back), swarm (peers hand off to each other), hierarchical (supervisors of supervisors), and pick the right one by *who routes*.
- **Compare frameworks honestly**: LangGraph's explicit state graph vs AutoGen's conversational multi-agent vs CrewAI's role-play abstraction — and explain *why CrewAI's role abstraction leaks* (branching/loops second-class, state implicit, role ≠ structure) and where AutoGen genuinely fits (open-ended chat collaboration).
- **Decide when a state machine beats an agent**: if you can draw the control flow in advance and it doesn't change run to run, build the deterministic state machine, not the agent.

Next: the exercises put this on your own code — build a minimal `StateGraph`, re-implement the week-5 ReAct agent as the four-node graph, and kill-and-resume it with a SQLite checkpointer. Continue to [the exercises](../exercises/README.md).

---

## References

- *LangGraph — persistence / checkpointers*: <https://langchain-ai.github.io/langgraph/concepts/persistence/>
- *LangGraph — checkpointer reference (`BaseCheckpointSaver`, `MemorySaver`)*: <https://langchain-ai.github.io/langgraph/reference/checkpoints/>
- *`langgraph-checkpoint-sqlite`* (the `SqliteSaver`): <https://pypi.org/project/langgraph-checkpoint-sqlite/>
- *`langgraph-checkpoint-postgres`* (the production saver): <https://pypi.org/project/langgraph-checkpoint-postgres/>
- *LangGraph — time travel*: <https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/>
- *LangGraph — multi-agent (supervisor / swarm / hierarchical)*: <https://langchain-ai.github.io/langgraph/concepts/multi_agent/>
- *AutoGen (Microsoft) — docs*: <https://microsoft.github.io/autogen/>
- *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation* — Wu et al., 2023: <https://arxiv.org/abs/2308.08155>
- *CrewAI — docs*: <https://docs.crewai.com/>
- *Anthropic — adaptive thinking & effort* (node model params): <https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking>
