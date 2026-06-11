# Week 22 — Resources

Every resource here is **free** or has a free tier, and most are *your own prior work* — Sprint A integrates the harnesses you built in weeks 8–11, so the most important "resource" this week is your own week-8 chunker, week-9 hybrid retriever, and week-11 memory tiers. The new external references are the architecture-document and Mermaid guides, plus the capstone spec and rubric you're building against.

This is a capstone sprint, not a topic week — the "library names move" caveat is replaced by a sharper one: **the interfaces you design this week are contracts the rest of the capstone depends on, so design them to be stable.** When you pick a vector store, a chunker, or a memory schema, you're committing the capstone to it; change it later and you re-architect under deadline.

## Read FIRST (the spec you're building against)

- **The capstone specification** — the syllabus's Capstone section: the one-paragraph spec, the eight required deliverables, the default architecture diagram, the chaos drill. Re-read it this week with Sprint A's foundation in mind:
  `C23-CRUNCH-AGENTS/SYLLABUS.md` (Capstone specification section)
- **The capstone rubric** — the public rubric the sealed-review panel grades against. Sprint A is graded on the Phase II milestone criteria (hybrid retrieval, chunking A/B'd, three memory tiers, the architecture memo):
  `capstone/RUBRIC.md`
- **The Phase II milestone (end of week 12)** — the syllabus's deeper-detail list: hybrid retrieval over a real corpus, chunking A/B'd, three memory tiers, a Ragas report, a 6-page architecture memo. Sprint A lands most of this:
  `C23-CRUNCH-AGENTS/SYLLABUS.md` (Phase capstone milestones)

## The architecture-document references

- **Mermaid documentation** — the diagram-as-code syntax for the architecture diagram the capstone requires (`flowchart`, `graph`); kept in the repo and rendered in the README so it stays in sync with reality:
  <https://mermaid.js.org/>
- **arc42 / C4 model** — two well-known templates for architecture documentation; read for the *shape* of a good architecture doc (context, components, decisions), not as a mandate — the capstone wants a 6-page doc, not a 60-page one:
  <https://arc42.org/> and <https://c4model.com/>
- **ADR (Architecture Decision Records)** — the lightweight "we chose X over Y because Z" format; the capstone doc's decisions-and-alternatives section is a handful of these:
  <https://adr.github.io/>

## Your own prior work (the real integration points)

- **Week 8 — `crunchrag_chunk`** — your chunking A/B harness and winning chunker. The capstone ingest uses `chunkers.load(<your-winner>)`. If it's broken, fix it before Sprint A.
- **Week 9 — the hybrid retriever** — BM25 + dense + reranker + RRF fusion and `evaluate()`. The capstone's `retrieve()` interface wraps this.
- **Week 11 — the three memory tiers** — episodic / semantic / procedural and the memory regression test. The capstone wires exactly these behind agent-facing interfaces.
- **Week 10 — vector-store ops** — your pgvector/Qdrant operational notes; you pick the store for the 10 GB corpus and own its backup/rebuild story.

## The retrieval and store references

- **pgvector README** — the default store; `vector_cosine_ops`, HNSW, filtered ANN. At 10 GB you care about index build time and the rebuild-after-schema-change cost:
  <https://github.com/pgvector/pgvector>
- **Qdrant** — the Rust store with first-class filtered search; an alternative if the corpus needs heavy metadata filtering at scale:
  <https://qdrant.tech/documentation/>
- **Tantivy / BM25** — the lexical leg of hybrid retrieval (or Elasticsearch); the sparse signal that complements dense vectors:
  <https://github.com/quickwit-oss/tantivy>
- **bge-reranker-v2** — the cross-encoder reranker on top of the fused candidates; the cheapest meaningful win in RAG (week 9):
  <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- **Reciprocal Rank Fusion (RRF)** — the standard fusion of dense and sparse rankings; a few lines of code, no tuning:
  <https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking>

## The memory references

- **Letta (formerly MemGPT)** — the tiered-memory-with-eviction reference; read for the memory-tier *patterns* even if you build your own (the preferred path) over pgvector:
  <https://github.com/letta-ai/letta>
- **The "Lost in the Middle" paper** — Liu et al., 2023; why context budgeting and memory summarization matter (long-context models degrade when the needle is buried), the motivation for the episodic tier's summarization:
  <https://arxiv.org/abs/2307.03172>
- **pgvector + a KG schema** — the self-built semantic memory tier: facts as (subject, relation, object) rows plus their embeddings, queried by both graph traversal and vector similarity:
  <https://github.com/pgvector/pgvector>

## Models you'll use this week

- **Embedding:** **`BAAI/bge-large-en-v1.5`** (week 7/8) for the dense leg and the semantic-memory facts — same model, same vector space.
- **Reranker:** **`BAAI/bge-reranker-v2-m3`** (week 9) on the fused candidates.
- **The supervisor / summarizer LLM:** **`claude-sonnet-4-6`** for the supervisor draft and the episodic-memory rolling summary; **`claude-haiku-4-5`** for the cheap summarization the memory tier runs frequently. (Routing between these is week 21's job, accounted for in the architecture doc.)
- **The serving tier:** your week-19 **vLLM** Qwen2.5-7B/13B is the local tier in the architecture doc; you don't deploy it this week, but the doc shows where it plugs in.

## Tools you'll use this week

- **Mermaid** — diagram-as-code; render in the repo README or a Markdown preview. No install; it's text.
- **`psycopg[binary]` + pgvector** — the corpus index and the semantic-memory store.
- **`tantivy` / `rank-bm25`** — the lexical leg.
- **`sentence-transformers` / `transformers`** — the BGE embedder and reranker.
- **Your week-8/9/11 packages** — imported, not rebuilt. The whole point of Sprint A is integration.

## A note on the corpus

The capstone corpus is a **10 GB private corpus** of your choosing — a domain you care about (a company's docs, a legal/scientific/technical archive, a personal knowledge base). It must be ≥10 documents and ≥100 pages (the Phase II milestone bar), and big enough that the *operational* story matters: ingest takes real time, the index takes real disk, a rebuild is a real cost. If 10 GB is impractical on your hardware, a **representative subset** (1–2 GB) is acceptable for Sprint A as long as you *report* the scaled-down numbers and the architecture doc reasons about the full-scale story (week-23's serving and week-24's chaos drill assume the real scale). The gold set is a **100-question** set over the corpus (the capstone's eval bar), built however you build gold sets — hand-authored, or LLM-generated and human-verified.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Architecture document** | The 6-page doc: components, interfaces, data flow, decisions-and-alternatives, build sequence. |
| **Mermaid diagram** | Diagram-as-code, committed to the repo, kept in sync with the running system. |
| **Retrieval interface** | The single `retrieve(query) -> ranked_chunks` contract every agent calls; hides the hybrid internals. |
| **Hybrid retrieval** | BM25 (lexical) + dense (vector) + reranker, fused with RRF; the capstone's retrieval substrate. |
| **RRF** | Reciprocal Rank Fusion — combine two rankings by `sum(1/(k+rank))`; the standard dense+sparse fuse. |
| **Episodic memory** | A rolling/hierarchical summary of the conversation's turns; "what have we said?" |
| **Semantic memory** | Vector + KG facts about the user/task; "what do we know?" |
| **Procedural memory** | A log of tool calls and actions taken; "what have we done?" |
| **Memory regression test** | A check that a load-bearing fact (the user's project name) survives to a late turn. |
| **Supervisor agent** | The LangGraph state machine that delegates to sub-agents; drafted this week, built in week 23. |
| **Sub-agent contract** | The interface (inputs/outputs) of a retrieval/code/writing/critique agent — designed before it's built. |
| **Build sequence** | The dependency-ordered plan: what's done (Sprint A), next (Sprint B), deferred (chaos drill). |

---

*If a link 404s, please open an issue so we can replace it.*
