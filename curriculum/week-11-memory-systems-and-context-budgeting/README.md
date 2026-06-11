# Week 11 — Memory Systems and Context Budgeting

Welcome to the week your agent stops having amnesia. For ten weeks your models have been *stateless* — every call starts from nothing, remembers nothing, and the only "memory" was whatever you stuffed into the prompt by hand. That works for one-shot questions and falls apart the moment a conversation has a past: the user told you their project's name in turn 3, and in turn 38 the agent has no idea what they're talking about. This week you build the machinery that fixes it — **memory tiers** — and you learn the discipline that makes them affordable: **budgeting the context window like the expensive cache it is.**

This is week 5 of **Phase II — RAG & Memory Systems**, and it's the week that completes the phase's promise. Weeks 7–10 gave the model *knowledge it didn't have at training time* (retrieval). This week gives the agent *a memory of its own interactions* — what was said, what it learned about the user, what tools it has run — and the two are different things. Retrieval pulls from a fixed corpus; memory accumulates from the conversation. By Friday you'll have a chat agent with three memory tiers that still remembers the user's project name in turn 38, measured against a no-memory baseline that doesn't.

The one sentence to internalize before you read another line:

> **Context is the most expensive cache on the planet. Spend it like one.** Every token in the window costs money and latency, the window is finite, and — worse than finite — the model gets measurably *worse* at using the middle of a long context. You are not "adding memory"; you are *managing a cache* with eviction, summarization, and a hit-or-miss budget.

Here's why that's not hyperbole. The naive "memory" is to keep appending every turn to the prompt. That works until the conversation is long, and then three things break at once: the prompt blows past the context window (hard failure), the cost-per-turn climbs linearly with conversation length (you pay to re-send the whole history every turn), and the model's accuracy on facts buried in the middle of that long context *drops* — the "lost in the middle" effect (Liu et al., 2023). So you can't just remember everything; you have to decide *what* to remember, *where* (which tier), and *what to evict* when the budget is tight. That decision — memory architecture plus context budgeting — is the engineering of this week.

There's a corollary worth taping next to last week's mantra:

> **More context is not more memory.** A 200k-token window you fill with raw history is *worse* than a 4k summary that holds the three facts that matter — slower, costlier, and less accurate on the buried facts. Memory is about *what you keep*, not *how much you can hold*.

## Learning objectives

By the end of this week, you will be able to:

- **Distinguish** the three memory tiers — **episodic** (the turn-by-turn conversation history), **semantic** (facts about the user/world, in a vector + knowledge-graph store), and **procedural** (tool histories, learned behaviors, how-to) — and say what belongs in each and why.
- **Summarize** conversation history with the right strategy — **rolling** (running summary updated each turn), **hierarchical** (summaries of summaries), and **map-reduce** (summarize chunks, then combine) — and pick by the conversation's shape.
- **Budget** the context window like a cache: allocate token slices to system prompt, retrieved memory, recent turns, and the user query; measure the budget; and refuse to exceed it.
- **Design** an eviction policy — **LRU** (drop the least-recently-used), **salience-weighted** (drop the least-important, where importance is learned or scored), and TTL — and reason about which memories to keep when the budget is full.
- **Build** the three tiers concretely: a rolling summary (episodic), a vector store of user facts (semantic — using the week-10 store adapter), and a tool-history log (procedural).
- **Place** the memory frameworks — **MemGPT / Letta**, **Zep**, **Mem0** — by what they do, and know when to reach for one versus the self-built pgvector + KG path (the preferred teaching path).
- **Explain** the **"lost in the middle"** effect (long-context models degrade on facts buried mid-context) and why it makes a *tight, well-budgeted* context beat a *full* one.
- **Measure** memory with a regression test: a multi-turn conversation benchmark that asks "does the agent still remember the user's project name in turn 38?" — and compare a memory-equipped agent against a no-memory baseline with a number.

## Prerequisites

This week assumes you have completed **C23 weeks 1–10**, or have equivalent fluency. Specifically:

- You finished **week 10** and have the `crunchstore` vector-store adapter: one interface (`create`, `upsert`, `search`, `search_filtered`) over pgvector/Qdrant/Weaviate. **This week uses that adapter as the semantic memory tier** — if it's broken, fix it first.
- You're comfortable from **week 2** with tokens and the context window, and from **weeks 7–9** with embeddings and retrieval — the semantic memory tier is retrieval pointed at *accumulated user facts* instead of a fixed corpus.
- You have an **agent loop** from week 5 (hand-rolled ReAct) you can wire memory into; the procedural tier logs *its* tool calls.
- Python 3.12 on Linux, macOS, or WSL2; a virtualenv; Docker for the vector store. An API key for a chat model (Anthropic `claude-sonnet-4-6` is the reference; a local 7B from week 6 is an acceptable open path for the summarization and judging).

You do **not** need a GPU for the memory work (it's orchestration + retrieval + summarization calls). You do **not** need prior agent-memory experience — we build the three tiers from scratch and use the frameworks only to place them.

## Topics covered

- **The three tiers:** episodic (turn history), semantic (vector + KG facts about the user/world), procedural (tool histories, learned behaviors) — what goes where and why the split matters.
- **Summarization strategies:** rolling (running summary), hierarchical (summaries of summaries), map-reduce (summarize-then-combine); the cost/fidelity trade of each.
- **Context budgeting:** the window as a cache; allocating token slices (system / memory / recent / query); measuring and enforcing the budget; the cost of a cache miss.
- **Eviction:** LRU, salience-weighted, TTL; what to drop when the budget is full; why eviction policy is a design decision, not a default.
- **The frameworks, placed:** MemGPT/Letta (tiered memory with paging/eviction), Zep (conversation memory store), Mem0 (long-term memory layer) — and the self-built pgvector + Postgres-KG path (preferred).
- **Hybrid vector + knowledge-graph memory:** when the *relationships* between remembered facts (the user works at X, X uses Y) are the retrieval signal, not just similarity.
- **Lost in the middle:** the long-context degradation effect (Liu et al., 2023) and why it makes context budgeting a quality lever, not just a cost lever.
- **Memory as evaluation:** the multi-turn regression test — does the agent remember turn-3 facts in turn 38? — measured against a no-memory baseline.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                            | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | The three tiers; episodic/semantic/procedural; summarization     |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Context budgeting + eviction; the budget exercise                |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Summarization strategies; lost-in-the-middle; the summarizer     |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Frameworks (Letta/Zep/Mem0) + KG memory; the harness             |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The 40-turn memory benchmark + memo; eviction tuning             |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                            |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                         |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                                  | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The memory-tier references, the summarization patterns, the Letta/Zep/Mem0 docs, the "lost in the middle" paper |
| [lecture-notes/01-memory-tiers-and-summarization.md](./lecture-notes/01-memory-tiers-and-summarization.md) | The three tiers, what goes where, and the summarization strategies (rolling/hierarchical/map-reduce) |
| [lecture-notes/02-context-budgeting-eviction-and-evaluation.md](./lecture-notes/02-context-budgeting-eviction-and-evaluation.md) | The window as a cache, budgeting, eviction policies, lost-in-the-middle, frameworks, and the memory regression test |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-budget-the-window.md](./exercises/exercise-01-budget-the-window.md) | Allocate and enforce a token budget across system/memory/recent/query and watch a naive append blow the window |
| [exercises/exercise-02-rolling-summary.py](./exercises/exercise-02-rolling-summary.py) | Build a rolling-summary episodic memory and measure tokens-saved vs facts-retained |
| [exercises/exercise-03-memory-regression.py](./exercises/exercise-03-memory-regression.py) | The turn-38 test: does the agent still remember the turn-3 fact? Memory vs no-memory baseline |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-three-tier-memory.md](./challenges/challenge-01-three-tier-memory.md) | The full three-tier agent + 40-turn benchmark + eviction policy, measured against a no-memory baseline |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the memory-architecture memo and the eviction-policy comparison |
| [mini-project/README.md](./mini-project/README.md) | The `crunchmem` three-tier memory system + the 40-turn regression harness |

## The "it remembered in turn 38" promise

C23 uses a recurring marker for every exercise that ends in memory actually *working* across a long conversation, with a number that proves it:

```
$ python memory_bench.py --memory three-tier --turns 40
memory=three-tier  turns=40  context_budget=4096 tokens
  turn 03: user says "my project is called Helios"   -> stored (semantic)
  turn 38: "what's my project called?"  -> "Helios" ✓  (recalled from semantic tier)
  facts retained: 18/20   context used: 3,710/4,096 tok (90%)   no window overflow
  vs no-memory baseline: 2/20 facts retained — it forgot turn 3 by turn 9
```

That `turn 38 -> "Helios" ✓` line is the whole point of the week. The no-memory baseline forgot the project name nine turns after hearing it; the three-tier agent recalls it 35 turns later *and* stays inside a 4k budget. The point of week 11 is to make memory a *measured* capability — proven with a turn-38 recall and a tokens-saved number, not a vibe about the agent "feeling" like it remembers.

## Stretch goals

If you finish the regular work early and want to push further:

- Read **"Lost in the Middle"** (Liu et al., 2023, arXiv 2307.03172) until you can explain the U-shaped accuracy curve — models recall facts at the *start* and *end* of a long context far better than the middle: <https://arxiv.org/abs/2307.03172>. Then plant a fact at the start, middle, and end of a long context and *measure* the recall difference yourself.
- Read the **MemGPT paper** (Packer et al., 2023, arXiv 2310.08560) until you can explain its OS-inspired paging of memory between a small "main context" and a large external store, and how that maps to your three tiers: <https://arxiv.org/abs/2310.08560>.
- Add a **knowledge-graph memory tier**: store user facts as `(subject, relation, object)` triples in Postgres and answer a multi-hop question ("what does the company my user works at use for CI?") that flat vector memory can't. This is the week-10 GraphRAG idea, pointed at memory.
- Implement **salience-weighted eviction**: score each memory by recency × importance (importance from an LLM or a heuristic) and evict the lowest; compare its turn-38 recall to plain LRU on the 40-turn benchmark.

## Up next

Week 11 closes the memory chapter of Phase II. **Week 12 — Multimodal RAG and Evaluation** caps the phase: you'll add vision-language models and ASR/TTS, and — the headline — build a **Ragas** evaluation suite for your whole Phase II pipeline (chunking, embeddings, reranking, store, *and* memory) with a calibrated LLM-as-judge. The memory regression test you build this week is a first taste of that eval discipline; week 12 generalizes it. Push your `crunchmem` system before you start; the Phase II milestone (week 12) requires three memory tiers wired and measured — which is exactly what you build here.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
