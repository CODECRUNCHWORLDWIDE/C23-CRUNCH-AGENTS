# Week 11 — Resources

Every resource here is **free** or has a free tier. The memory frameworks (Letta, Zep, Mem0) are open source and self-hostable. The papers (Lost in the Middle, MemGPT) are on arXiv. The vector store is your week-10 `crunchstore` (pgvector/Qdrant), already running. The only paid path is a chat model for summarization and judging — Anthropic `claude-sonnet-4-6` is the reference, and a local 7B from week 6 is an acceptable open fallback for every lab.

Framework names and APIs move every cohort — the *concepts* (the three tiers, summarization, the context-as-cache budget, eviction policies, the turn-38 regression test) are stable. When a specific client method 404s, search the project's docs for the concept name.

This week sits on top of week 10. The semantic memory tier *is* a vector store — your `crunchstore` adapter — and the eval discipline comes from the same "measure, don't vibe" stance as weeks 7–10.

## Required reading (work it into your week)

- **Lost in the Middle** — Liu et al., *How Language Models Use Long Contexts* (2023). The U-shaped recall curve that makes context budgeting a *quality* lever, not just a cost one. Read §3–4 until you can draw the curve:
  <https://arxiv.org/abs/2307.03172>
- **MemGPT paper** — Packer et al., *Towards LLMs as Operating Systems* (2023). The OS-inspired paging of memory between a small main context and a large external store — the pattern Letta implements:
  <https://arxiv.org/abs/2310.08560>
- **Anthropic — token counting.** Count context with the *model's* tokenizer (`count_tokens`), never `tiktoken` (a different tokenizer that undercounts Claude). The budget is measured in real tokens:
  <https://docs.claude.com/en/docs/build-with-claude/token-counting>
- **Anthropic — context windows.** The finite budget you're spending; what fits, what it costs, how to think about long context:
  <https://docs.claude.com/en/docs/build-with-claude/context-windows>

## The memory frameworks (placed, not required)

- **Letta (formerly MemGPT)** — tiered memory with paging and eviction between main context and an external store; the most direct realization of "context is a cache you page." Self-hostable:
  <https://github.com/letta-ai/letta>
- **Zep** — a conversation memory store: ingests history, builds a summary and a fact/entity graph, serves memory back to the agent. The "managed episodic + semantic" option:
  <https://github.com/getzep/zep>
- **Mem0** — a long-term memory layer that extracts and stores salient facts across sessions and retrieves them. The "managed semantic memory" option:
  <https://github.com/mem0ai/mem0>

## Summarization patterns

- **LangChain — conversation summary memory** — the canonical reference implementations of rolling-summary and buffer-plus-summary memory; read for the pattern, not necessarily to adopt the library:
  <https://python.langchain.com/docs/concepts/memory/>
- **Map-reduce summarization** — the summarize-chunks-then-combine pattern (same shape as GraphRAG's community summaries from week 10); LangChain's `map_reduce` chain documents it clearly:
  <https://python.langchain.com/docs/how_to/summarize_map_reduce/>
- **Anthropic — long-context tips** — practical guidance on structuring long prompts (put the important content at the edges — the lost-in-the-middle mitigation):
  <https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/long-context-tips>

## The semantic tier (reused from week 10)

- **Your `crunchstore` adapter** — the week-10 vector-store abstraction *is* the semantic memory tier. You upsert user facts and retrieve them by similarity, exactly as you retrieved corpus chunks — only now the "corpus" accumulates from the conversation.
- **pgvector** — the default store behind the semantic tier; `vector_cosine_ops`, `<=>`, the metadata columns hold the fact's source turn and salience:
  <https://github.com/pgvector/pgvector>
- **Postgres KG schema** — for the hybrid vector + knowledge-graph memory tier: a `(subject, relation, object)` triple table answers the multi-hop memory question ("what does the company my user works at use?") that flat vector memory can't. The self-built, preferred path.

## The Claude API for memory (the stateless core)

- **Anthropic Messages API** — the API is *stateless*: you send the full conversation history every call, and the model remembers nothing on its own. Memory is the machinery *you* build around that. `claude-sonnet-4-6` is the reference summarizer/judge:
  <https://docs.claude.com/en/api/messages>
- **`anthropic` Python SDK** — `client.messages.create(...)` for the agent and summarizer; `client.messages.count_tokens(...)` for the budget. Model id `claude-sonnet-4-6` (cheap, capable) for summarization; `claude-opus-4-8` if you want the strongest judge:
  <https://github.com/anthropics/anthropic-sdk-python>
- **Local fallback** — any chat model you stood up in week 6 (Qwen2.5-7B via Ollama/vLLM) works as the summarizer and judge; the open-only path the course always documents.

## Papers worth your time (free on arXiv)

- **Lost in the Middle** (Liu et al., 2023) — the U-shaped context-recall curve: <https://arxiv.org/abs/2307.03172>
- **MemGPT** (Packer et al., 2023) — memory paging as an OS: <https://arxiv.org/abs/2310.08560>
- **MemoryBank / generative-agent memory lineage** — the salience + recency + retrieval scoring that informs salience-weighted eviction; search arXiv for the current treatment. The concept (score memories, evict the lowest) is durable.

## Tools you'll use this week

- **`crunchstore`** — your week-10 vector-store adapter; the semantic memory tier.
- **`anthropic`** — the chat model (summarizer, agent, judge) and `count_tokens` for the budget.
- **`psycopg[binary]`** — Postgres/pgvector client for the semantic tier and the KG schema.
- **A local model (optional)** — the open summarizer/judge path from week 6.
- **`week-5` agent** — your ReAct loop; the procedural tier logs *its* tool calls, and the memory loop wraps it.

## A note on the benchmark

The memory regression test runs a **40-turn conversation** that plants durable facts early (turn 3: "my project is called Helios"; turn 7: "I prefer Python") and asks about them late (turn 38: "what's my project called?"), with distractor turns filling the gap so a fact must *survive* many turns to be recalled. The metric is **recall rate** (facts recalled / facts asked), measured for the three-tier agent *and* a no-memory baseline. The delta — e.g. 18/20 vs 2/20 — is the number that justifies the memory system, exactly as Recall@5 justified a chunking strategy in week 8. Build the benchmark once; reuse it to compare eviction policies (LRU vs salience) and to prove each tier earns its place.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Episodic memory** | The turn-by-turn conversation history — rolling summary of older turns + recent verbatim. "What was said." |
| **Semantic memory** | Durable facts about the user/world, in a vector store (+ optional KG). "What's true." Retrieved by similarity. |
| **Procedural memory** | Tool-call histories and outcomes, in a log. "What I did." Retrieved by tool / recency. |
| **Rolling summary** | A running summary updated each turn; the default episodic-compression strategy. |
| **Hierarchical summary** | Summaries of summaries at multiple levels; for long conversations needing resolution. |
| **Map-reduce summary** | Summarize chunks independently, then combine; for large batches, parallelizable. |
| **Context budget** | A deliberate token allocation across prompt slices (system/semantic/episodic/recent/query), measured and enforced. |
| **Eviction** | Dropping the lowest-value content when the budget is full — by LRU, salience, or TTL. |
| **LRU** | Least-recently-used eviction: drop the oldest. Recency-as-relevance. |
| **Salience-weighted** | Score memories by importance; evict the lowest-salience even if recent. Keeps the important old fact. |
| **TTL** | Time-to-live: expire transient facts that stop being true. |
| **Lost in the middle** | Long-context models recall edge content far better than middle content (U-shaped curve). |
| **Memory regression test** | Plant facts early, ask late, score recall vs a no-memory baseline. The turn-38 test. |

---

*If a link 404s, please open an issue so we can replace it.*
