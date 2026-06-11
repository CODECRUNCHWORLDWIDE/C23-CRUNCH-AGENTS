# Week 13 — LangGraph and the Graph Pattern

Welcome to **Phase III — Agents & Orchestration**, and to the week where your hand-rolled agent stops being a loop and becomes a graph. Back in week 5 you wrote a ReAct agent from scratch: a `while` loop that called the model, parsed a tool request, ran the tool, fed the result back, and went around again. It worked. It taught you what an agent *is* underneath the marketing. And then it grew a second tool, and a third, and a retry branch, and a "if the critique fails, re-plan" branch — and somewhere around the fourth `if` you stopped being able to read it. This week you graduate that loop to a **LangGraph state graph**: explicit nodes (plan, retrieve, execute, critique), explicit edges, a typed state object, and — the part that makes it production rather than demo — a **SQLite checkpointer** so the agent survives a process kill and resumes from exactly where it died.

The one sentence to internalize before you read another line:

> **When the agent gets a fourth tool, you graduate from a loop to a graph. Reach for LangGraph before the loop gets a fourth `if`.**

Here's why that's not a framework advertisement. A `while` loop with control flow tangled into conditionals is *implicit* state: the "where am I" lives in the program counter and a pile of local variables, and you cannot inspect it, serialize it, or resume it. A state graph makes the state *explicit* — a `TypedDict` you can print, log, checkpoint, and replay. The moment your agent has four tools and a re-plan branch, the loop's implicit state is a liability: you can't tell why it looped, you can't pause it, and if the process dies mid-run you start over from zero. The graph fixes all three at once, and the SQLite checkpointer is what turns "my agent ran" into "my agent ran, crashed, and picked up where it left off." That last property — **resumability** — is the difference between a notebook toy and something you'd put behind a queue.

There's a corollary worth taping next to it:

> **Not everything that looks like an agent needs to be one.** When the control flow is fixed and known in advance, a deterministic state machine beats an agent — it's cheaper, faster, and can't loop forever. The skill this week is partly knowing *when to reach for the graph* and partly knowing *when a plain state machine (or even a function) is the honest answer*.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** when a loop becomes a graph — the "fourth `if`" heuristic — and articulate why explicit state beats implicit state once control flow branches.
- **Build** a LangGraph `StateGraph` over a `TypedDict` state: define nodes as functions that return partial state, wire normal edges and conditional edges, set `START`/`END`, compile, and invoke.
- **Re-implement** your week-5 ReAct agent as a four-node graph (plan → retrieve → execute → critique) with a conditional edge that loops back to `plan` when the critique fails — calling the Anthropic SDK (or a local model) directly inside the node functions.
- **Add persistence** with a `SqliteSaver` checkpointer and a `thread_id`, so state is written after every node and the agent **survives a process kill** and resumes from the last checkpoint.
- **Use** reducers (`Annotated[list, operator.add]`, `add_messages`) to accumulate state across nodes correctly, and understand why a node returning a partial dict is the right mental model.
- **Compare** LangGraph against AutoGen and CrewAI honestly — why CrewAI's role-play "Agent/Task/Crew" abstraction is appealing but leaks, why AutoGen's conversational multi-agent is a different shape, and why LangGraph's explicit-state-graph clarity wins for systems you have to debug and resume.
- **Reason** about supervisor, swarm, and hierarchical multi-agent patterns and know which one a problem wants.
- **Enforce** a step/token/time budget in the agent so an unbounded graph can't loop forever — a non-negotiable for anything you ship.

## Prerequisites

This week assumes you have completed **C23 weeks 1–12**, or have equivalent fluency. Specifically:

- You finished **week 5** and have a working **hand-rolled ReAct agent** — a loop that calls the model, parses a tool call, executes it, and feeds the result back. This week re-implements *that exact agent* as a graph, and the headline lab runs *that exact 25-task benchmark* to compare the two. If your week-5 agent is broken, fix it first.
- You finished **weeks 7–12** and have a **retrieval pipeline** ending in a Ragas eval suite. This week's `retrieve` node calls *your* retriever — the graph doesn't replace your RAG, it orchestrates it. The mini-project reuses your week-7–12 retrieval as the retrieve node's tool, unchanged.
- Python 3.12 on Linux, macOS, or WSL2; a virtualenv you can `pip install` into.
- You can read and write the Anthropic Python SDK at the level of weeks 5–6 (`client.messages.create`, tool use, iterating `response.content`). We call it directly inside nodes.

You do **not** need a GPU. The LLM behind each node can be **Anthropic Claude** (the frontier path — `claude-sonnet-4-6` for most nodes, `claude-opus-4-8` for the critique node) **or** a local open model via Ollama/vLLM (the open-source path). LangGraph itself is fully open source; nothing this week is paywalled.

## Topics covered

- **Why frameworks now:** when the loop becomes a graph, the "fourth `if`" heuristic, and the cost of implicit state in a branching agent.
- **LangGraph mechanics:** `StateGraph` over a `TypedDict`, nodes as functions returning partial state, reducers (`operator.add`, `add_messages`), normal vs conditional edges, `START`/`END`, `compile()`, `invoke()`, and `stream()`.
- **The four-node ReAct graph:** plan → retrieve → execute → critique, with a conditional edge that re-plans on critique failure — the week-5 loop, made explicit.
- **Persistence and checkpointers:** `MemorySaver` vs `SqliteSaver`, the `thread_id`, how state survives a process kill and **resumes**, and time-travel / replay over the checkpoint history.
- **Multi-agent patterns:** supervisor (a router that delegates to sub-agents), swarm (peers that hand off to each other), and hierarchical (supervisors of supervisors) — with real LangGraph code.
- **The honest framework comparison:** AutoGen vs CrewAI vs LangGraph — what each is good at, why CrewAI's role-play abstraction leaks, and when a deterministic state machine beats an agent entirely.
- **Budgets:** step, token, and time budgets as a mandatory guard against the unbounded-loop failure mode.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                       | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Why a graph; the fourth `if`; StateGraph + TypedDict + edges |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | The four-node ReAct graph; conditional edges; the exercises |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Checkpointers, thread_id, surviving a kill; patterns         |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Supervisor/swarm/hierarchical; framework critique; harness   |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The 25-task re-implementation + LoC/observability memo       |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                       |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                   |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                             | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | LangGraph docs, the checkpointer package, the ReAct paper, AutoGen + CrewAI docs (for the critique), Anthropic tool-use docs, and the glossary cheat-sheet |
| [lecture-notes/01-when-the-loop-becomes-a-graph.md](./lecture-notes/01-when-the-loop-becomes-a-graph.md) | Why a graph, the fourth `if`, StateGraph + TypedDict + reducers + edges, and re-implementing the week-5 ReAct agent as four explicit nodes |
| [lecture-notes/02-persistence-patterns-and-framework-honesty.md](./lecture-notes/02-persistence-patterns-and-framework-honesty.md) | Checkpointers, surviving a process kill and resuming, time-travel; supervisor/swarm/hierarchical patterns; the honest AutoGen vs CrewAI vs LangGraph comparison; when a state machine beats an agent |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-first-state-graph.md](./exercises/exercise-01-first-state-graph.md) | Build a minimal 3-node `StateGraph` with one conditional edge that loops back, and inspect the state at each node |
| [exercises/exercise-02-react-as-a-graph.py](./exercises/exercise-02-react-as-a-graph.py) | Implement the plan/retrieve/execute/critique graph and print the node trace on a couple of tasks (offline fallback so it always runs) |
| [exercises/exercise-03-survive-the-kill.py](./exercises/exercise-03-survive-the-kill.py) | Attach a SqliteSaver, run partway, simulate a process kill, then resume from the checkpoint — the agent survived the kill, made measurable |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-react-loop-to-graph.md](./challenges/challenge-01-react-loop-to-graph.md) | Re-implement the week-5 ReAct agent as a LangGraph state graph + SQLite checkpointer, run the same 25-task benchmark, compare lines of code, observability, and resumability |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the LangGraph re-implementation + the LoC/observability/resumability memo |
| [mini-project/README.md](./mini-project/README.md) | The `crunchagent_graph` package — the plan/retrieve/execute/critique graph + persistence + budgets + a benchmark runner |

## The "the agent survived the kill" promise

C23 uses a recurring marker for every exercise that ends in something *production* actually working. Week 8's was "the answer survived the chunking." This week's is **"the agent survived the kill"** — a SQLite-checkpointed agent that you kill mid-run and resume, continuing from exactly where it died, with no re-work of the steps it had already completed:

```
$ python survive_the_kill.py --thread task-42
[run 1] thread=task-42
  node=plan      -> plan set ("retrieve the termination clause, then answer")
  node=retrieve  -> 3 docs fetched, written to checkpoint
  *** SIMULATED PROCESS KILL after retrieve (raise SystemExit) ***

$ python survive_the_kill.py --thread task-42        # fresh process, same thread
[run 2] thread=task-42  (resuming from checkpoint)
  RESUMED — state recovered: plan=set, docs=3 already in state
  node=execute   -> answered from the 3 recovered docs (retrieve NOT re-run)
  node=critique  -> pass
  PASS: agent resumed after kill; retrieve ran once, not twice.
```

If that agent had been a `while` loop, run 2 would start from zero — re-planning, re-retrieving, re-paying for every token it already spent before the crash. The checkpointer is what makes the second process pick up the first one's state. The point of week 13 is to make the agent *resumable* — and to prove it by killing it and watching it continue, not by trusting that it would.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **ReAct paper** (Yao et al., 2022, arXiv 2210.03629) again with graph eyes: every "Thought → Action → Observation" cycle is a loop iteration, and the graph makes each phase a *node*. Map the paper's loop onto your four nodes until the correspondence is exact: <https://arxiv.org/abs/2210.03629>.
- Add a **supervisor** over two sub-agent graphs (a "research" sub-agent and a "math" sub-agent) and a router node that picks which to call. This is the multi-agent pattern from Lecture 2, and it's the skeleton of the Phase III capstone.
- Implement **time-travel**: list the checkpoint history for a thread, pick a checkpoint *before* a bad decision, and resume the graph from there with a corrected input. LangGraph exposes the checkpoint history; use it.
- Stand up the **same graph behind a local open model** (Ollama running `llama3.1` or `qwen2.5`) by swapping only the node's model call. Confirm the graph, the checkpointer, and the budgets are model-agnostic — only the inference call changed.

## Up next

Phase III continues. **Week 14** takes the orchestration literacy you built here and moves it into the **TypeScript world — Mastra and Inngest** — durable, event-driven agent execution where the "checkpoint" is a workflow step that a job runner can retry and resume. You'll see the same ideas (explicit steps, durable state, resumability) wearing different clothes, and you'll be able to say precisely what LangGraph's `SqliteSaver` and Inngest's step memoization have in common and where they differ. Push your `crunchagent_graph` mini-project before you start it; the contrast lands harder when you've built the LangGraph version first.

This graph — plan/retrieve/execute/critique, checkpointed, budgeted — is not a throwaway. In **weeks 22–23** it becomes the **supervisor at the heart of the capstone**: the same `StateGraph` shape, scaled up to route across multiple sub-agents. Build it well now.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
