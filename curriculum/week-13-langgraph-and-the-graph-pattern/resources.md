# Week 13 — Resources

Every resource here is **free** or has a free tier. **LangGraph** and **langgraph-checkpoint-sqlite** are open source (MIT). The **ReAct paper** is on arXiv. **AutoGen** and **CrewAI** are open source — you read their docs this week to *critique* them honestly, not to adopt them. The Anthropic SDK and tool-use docs are free to read; running Claude costs tokens, but every lab has an **open-model path** (Ollama/vLLM) and a **deterministic-stub path** so nothing here is paywalled.

Library names and APIs move every cohort — the *concepts* (nodes, edges, state, reducers, checkpointers, `thread_id`, supervisor/swarm/hierarchical) are stable. When a specific page 404s, search the project's docs for the symbol name.

This week sits on top of **week 5** (the hand-rolled ReAct loop) and **weeks 7–12** (the retrieval pipeline). The 25-task benchmark and the `retrieve` tool come from there; the resources below assume you have both.

## Required reading (work it into your week)

- **LangGraph — overview and core concepts** — the canonical reference for `StateGraph`, state, nodes, and edges. Read the "Low Level Concepts" / "Graph API" pages until the `TypedDict` + reducer + node-returns-partial-state model is second nature:
  <https://langchain-ai.github.io/langgraph/concepts/low_level/>
- **LangGraph — persistence (checkpointers)** — the page that makes this week's promise real: how the checkpointer writes state after every node, what a `thread_id` is, and how a graph resumes. Read it twice:
  <https://langchain-ai.github.io/langgraph/concepts/persistence/>
- **ReAct: Synergizing Reasoning and Acting in Language Models** — Yao et al., 2022. The paper your week-5 loop implemented; this week you turn its Thought→Action→Observation cycle into explicit graph nodes:
  <https://arxiv.org/abs/2210.03629>
- **Anthropic — tool use overview** — the tool-use contract you call *inside* a node: `tools=[{name, description, input_schema}]`, `stop_reason == "tool_use"`, returning `tool_result` blocks:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>

## The LangGraph references

- **LangGraph — Graph API how-tos** — `add_node`, `add_edge`, `add_conditional_edges`, `START`, `END`, `compile`. The mechanics, with runnable snippets:
  <https://langchain-ai.github.io/langgraph/how-tos/>
- **LangGraph — conditional edges** — how a routing function returns the name of the next node (or a list), and how the path map `{...}` works:
  <https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges>
- **LangGraph — `add_messages` reducer** — the built-in reducer for accumulating a message list (append + dedupe by id) without clobbering prior turns:
  <https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers>
- **LangGraph — streaming** — `app.stream(...)` with `stream_mode` ("values", "updates", "debug") so you can watch each node fire — the observability win over a `while` loop:
  <https://langchain-ai.github.io/langgraph/how-tos/streaming/>
- **LangGraph — time travel** — listing the checkpoint history (`get_state_history`) and resuming from an earlier checkpoint; the replay/debug surface:
  <https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/>

## Persistence / checkpointing

- **`langgraph-checkpoint-sqlite`** — the SQLite checkpointer used in the labs. `pip install langgraph-checkpoint-sqlite`; `from langgraph.checkpoint.sqlite import SqliteSaver`. This is what survives the kill:
  <https://pypi.org/project/langgraph-checkpoint-sqlite/>
- **LangGraph — checkpointer interface** — `MemorySaver` (in-memory, dies with the process) vs `SqliteSaver` (on-disk, survives) vs the Postgres saver for production; the `BaseCheckpointSaver` contract they share:
  <https://langchain-ai.github.io/langgraph/reference/checkpoints/>
- **`langgraph-checkpoint-postgres`** — the production-grade saver (mentioned for contrast; SQLite is the lab default):
  <https://pypi.org/project/langgraph-checkpoint-postgres/>

## AutoGen and CrewAI (for the critique — read, don't adopt)

You read these to be able to *compare* them honestly against LangGraph, per the syllabus. The course's position: LangGraph's explicit state graph wins for systems you have to debug, checkpoint, and resume; CrewAI's role-play abstraction is appealing in a demo and leaks the moment you need real control flow; AutoGen's conversational multi-agent is a different (and sometimes better) shape for open-ended chat-style collaboration.

- **AutoGen (Microsoft) — docs** — conversational multi-agent: agents that talk to each other (`AssistantAgent`, `UserProxyAgent`, group chat). Read the "core concepts" to see the conversation-as-control-flow model:
  <https://microsoft.github.io/autogen/>
- **CrewAI — docs** — the "Agent / Task / Crew" role-play abstraction (a "researcher" agent, a "writer" agent, a "crew" that runs them). Read it, then read Lecture 2 §"Why CrewAI leaks" — the abstraction is the appeal *and* the problem:
  <https://docs.crewai.com/>
- **LangChain — agent/framework landscape** — context on where LangGraph sits relative to the broader ecosystem (LangChain chains vs LangGraph graphs — and why conflating the two is a mistake the lecture calls out):
  <https://python.langchain.com/docs/concepts/>

## Anthropic / model docs (the LLM behind each node)

- **Anthropic Python SDK** — `pip install anthropic`; `client.messages.create(...)`. The call you make *inside* each node. The course default node model is `claude-sonnet-4-6`; the critique node uses `claude-opus-4-8`:
  <https://platform.claude.com/docs/en/api/client-sdks>
- **Anthropic — adaptive thinking & effort** — Opus 4.8 / Sonnet 4.6 use `thinking={"type":"adaptive"}` and `output_config={"effort":"high"}`. No `budget_tokens`, no `temperature`/`top_p`/`top_k` (they 400). Read this before you write a node that calls Claude:
  <https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking>
- **Anthropic — tool use (Python)** — the exact tool-use loop you collapse into the `execute` node: declare tools, check `stop_reason == "tool_use"`, return `tool_result` blocks in a user turn:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use>
- **Ollama** — the open-model path: run `llama3.1` or `qwen2.5` locally and call it from a node instead of Claude. The graph and checkpointer don't change; only the node's inference call does:
  <https://github.com/ollama/ollama>
- **vLLM** — the higher-throughput open-model server if you have a GPU; OpenAI-compatible endpoint you call from a node:
  <https://github.com/vllm-project/vllm>

## Papers worth your time (free on arXiv)

- **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022) — the loop your graph makes explicit. Thought → Action → Observation is exactly plan → execute → (observe), iterated:
  <https://arxiv.org/abs/2210.03629>
- **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al., 2023) — the "critique and re-try" idea your `critique → plan` conditional edge implements; an agent that reflects on its own failures and re-plans:
  <https://arxiv.org/abs/2303.11366>
- **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** (Wu et al., 2023) — the paper behind AutoGen's conversational model; read it to critique it fairly:
  <https://arxiv.org/abs/2308.08155>

## Models you'll use this week

- **`claude-sonnet-4-6`** — the default model behind most nodes (plan, execute). Fast, strong, adaptive thinking. Exact model ID, no date suffix.
- **`claude-opus-4-8`** — the model behind the **critique** node, where harder reasoning earns its cost. Adaptive thinking, `effort` up to `high`/`max`.
- **`claude-haiku-4-5`** — the cheap option for a router/supervisor node where the decision is simple (pick a sub-agent), to keep the orchestration layer cheap.
- **A local open model** (`llama3.1`, `qwen2.5` via Ollama, or any vLLM-served model) — the open-source path. The graph is model-agnostic; only the node's call changes.

## Tools you'll use this week

- **`langgraph`** — `pip install langgraph langchain-core`. `from langgraph.graph import StateGraph, START, END`. The graph engine.
- **`langgraph-checkpoint-sqlite`** — `pip install langgraph-checkpoint-sqlite`. `from langgraph.checkpoint.sqlite import SqliteSaver`. The on-disk checkpointer that survives the kill. (In-memory `MemorySaver` ships inside `langgraph` itself.)
- **`anthropic`** — `pip install anthropic`. `client = anthropic.Anthropic()`; the SDK you call inside a node.
- **`langchain-anthropic`** *(optional)* — `pip install langchain-anthropic`. Provides `ChatAnthropic`, the LangChain wrapper. We show it as an *alternative*, but the course default is calling the raw SDK in a node — the lecture explains why we prefer the thin path.
- **Your week-5 ReAct agent** and your **week-7–12 retrieval package** — imported, not rewritten. The graph orchestrates them.

## A note on the benchmark

The labs and mini-project run against the **same 25-task agent benchmark you built in week 5** — the suite of small, gradable tasks ("find the termination clause and state the notice period", "compute X then look up Y", etc.) that your hand-rolled ReAct loop was scored on. This week you point the *same* benchmark at the *same* tasks through the *new* graph implementation, so the comparison is apples-to-apples: same tasks, same scoring, two implementations. The headline number isn't just "did it pass" — it's the **delta in lines of code, observability, and resumability** between the loop and the graph. If your week-5 benchmark harness is missing, reconstruct the 25 tasks from your week-5 repo before Friday; the challenge depends on it.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Node** | A function in the graph. Takes the state, does work (often an LLM call), returns a *partial* state dict to merge back in. |
| **Edge** | A directed connection: "after node A, go to node B." Unconditional. |
| **Conditional edge** | An edge whose target is chosen at runtime by a routing function that reads the state and returns the next node's name (e.g. `critique` → `plan` if the critique failed, else `END`). |
| **State** | The single typed object (a `TypedDict`) that flows through the graph. Every node reads it and returns updates to it. The explicit alternative to a `while` loop's scattered local variables. |
| **Reducer** | The rule for *merging* a node's returned value into the state for a given key. Default = overwrite; `operator.add` / `add_messages` = append. Declared with `Annotated[type, reducer]`. |
| **`START` / `END`** | Sentinel nodes: where the graph begins and where a path terminates. `add_edge(START, "plan")`; route to `END` to stop. |
| **`compile()`** | Turns the `StateGraph` definition into a runnable app. Pass `checkpointer=...` here to add persistence. |
| **Checkpointer** | The thing that writes the state to durable storage after every node, keyed by `thread_id`, so the graph can resume. `MemorySaver` (RAM, dies) vs `SqliteSaver` (disk, survives a kill). |
| **`thread_id`** | The conversation/run identifier passed in `config={"configurable": {"thread_id": "..."}}`. The key under which the checkpointer stores this run's state; resuming = invoking with the *same* `thread_id`. |
| **Supervisor** | A multi-agent pattern: a central router node that delegates to specialized sub-agents and collects results. The capstone shape (weeks 22–23). |
| **Swarm** | A multi-agent pattern: peer agents that *hand off* control to each other directly (no central router), each deciding when to pass the baton. |
| **Hierarchical** | A multi-agent pattern: supervisors of supervisors — a tree of routers, for problems too big for one supervisor's tool/agent budget. |
| **State machine (vs agent)** | A graph whose transitions are *fixed and known in advance* (no LLM decides the routing). Cheaper, faster, can't loop forever — the right answer when the control flow isn't actually open-ended. |
| **Budget** | A hard cap on steps, tokens, or wall-clock time, enforced in the graph so an agent that would loop forever stops. Mandatory for production. |

---

*If a link 404s, please open an issue so we can replace it.*
