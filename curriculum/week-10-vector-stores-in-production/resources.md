# Week 10 — Resources

Every resource here is **free** or has a free tier. pgvector, Qdrant, Weaviate, Milvus, and Chroma are all open source and self-hostable; each has a managed cloud tier you do *not* need for this week. GraphRAG is open source (MIT). The papers are on arXiv. Everything runs in Docker on a laptop.

Store names, APIs, and config flags move every cohort — the *concepts* (filtered ANN, metadata indexes, backup/restore, time-to-recover, GraphRAG community summaries, agentic retriever choice) are stable. When a specific client method 404s or a config key is renamed, search the store's docs for the concept name.

This week sits on top of weeks 7–9. The retrieval metrics (`Recall@5`, `MRR`) and the hybrid pipeline (dense + BM25 + reranker) come from there; the resources below assume you have that pipeline and only swap the store.

## Required reading (work it into your week)

- **pgvector README** — your baseline store. Re-read the HNSW index section, the `<=>` operator, and the filtering notes (`WHERE` + vector search is exactly the filtered-ANN problem):
  <https://github.com/pgvector/pgvector>
- **Qdrant documentation — filtering and indexing.** Qdrant's headline is *fast filtered search*; read the "Filtering" and "Indexing" (payload index) pages until you understand how it filters *during* ANN traversal, not after:
  <https://qdrant.tech/documentation/concepts/filtering/>
- **Weaviate documentation — concepts.** The graph-leaning, generative-search store; read the data-schema, the filtered search, and the `nearText`/generative pages:
  <https://weaviate.io/developers/weaviate>
- **GraphRAG paper** — Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* (Microsoft, 2024). The community-summary mechanism that answers global questions flat retrieval can't. Read §2–3:
  <https://arxiv.org/abs/2404.16130>

## The stores (have these open all week)

- **pgvector** — `docker run -d -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17`. Postgres + a `vector` type + HNSW/IVFFlat indexes. The default; you already know it:
  <https://github.com/pgvector/pgvector>
- **Qdrant** — `docker run -d -p 6333:6333 qdrant/qdrant`. Rust, purpose-built for filtered vector search; payload indexes; snapshots for backup:
  <https://github.com/qdrant/qdrant>
- **Weaviate** — `docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate`. Schema-first, generative search, graph-leaning cross-references:
  <https://github.com/weaviate/weaviate>
- **Milvus** — `milvus` via its docker-compose (etcd + MinIO + Milvus); the billion-scale store with separated query/data/index nodes. Heavier to stand up; the stretch store:
  <https://github.com/milvus-io/milvus>
- **Chroma** — `pip install chromadb`; `chromadb.PersistentClient(...)`. The developer-ergonomics store: zero-config, great for prototyping, not for billion-scale:
  <https://github.com/chroma-core/chroma>

## Filtered ANN and metadata

- **Qdrant — filtering strategy** (the filterable HNSW that combines the filter with the graph traversal). The clearest production treatment of the filtered-ANN problem:
  <https://qdrant.tech/articles/filtrable-hnsw/>
- **pgvector — filtering with HNSW** (iterative scans, `WHERE` + `ORDER BY embedding <=> ...`); the trade-offs of filtering a Postgres HNSW index:
  <https://github.com/pgvector/pgvector#filtering>
- **Weaviate — filtered search** (how a `where` filter interacts with the vector index):
  <https://weaviate.io/developers/weaviate/search/filters>
- **"Filtered vector search" survey background** — the pre-filter vs post-filter vs in-filter taxonomy; search the store docs and the Pinecone/Qdrant blogs for the current treatment. The concept (selectivity decides which strategy wins) is durable.

## Operations: backup, restore, replication

- **Qdrant — snapshots.** Create and restore collection snapshots; the cleanest backup story of the three. This is what makes the recovery drill fast:
  <https://qdrant.tech/documentation/concepts/snapshots/>
- **PostgreSQL backup & restore** — `pg_dump` / `pg_restore` / base backups. pgvector's backup story *is* Postgres's, which is its big operational advantage (your team already knows it):
  <https://www.postgresql.org/docs/current/backup.html>
- **Weaviate — backup.** The backup module (filesystem / S3 / GCS) and restore:
  <https://weaviate.io/developers/weaviate/configuration/backups>
- **Qdrant — distributed deployment** (sharding + replication for HA):
  <https://qdrant.tech/documentation/guides/distributed_deployment/>

## GraphRAG and agentic RAG

- **Microsoft GraphRAG (the implementation)** — the open-source library that builds the entity graph + community summaries from a corpus:
  <https://github.com/microsoft/graphrag>
- **GraphRAG paper** (community-summary retrieval for global questions): <https://arxiv.org/abs/2404.16130>
- **LlamaIndex — agentic / router retrieval** — the `RouterRetriever` and agent-over-retrievers patterns; the cleanest reference for "the agent chooses the retriever":
  <https://docs.llamaindex.ai/en/stable/module_guides/querying/router/>
- **Anthropic — "Contextual Retrieval"** — the engineering write-up on improving retrieval with context-augmented chunks; useful framing for when retrieval architecture (not the store) is the lever:
  <https://www.anthropic.com/news/contextual-retrieval>

## Models and tools you'll use this week

- **`BAAI/bge-large-en-v1.5`** — the fixed embedding (1024-dim, normalized), same as weeks 7–9, so the *store* is the only variable:
  <https://huggingface.co/BAAI/bge-large-en-v1.5>
- **`BAAI/bge-reranker-v2-m3`** — your week-9 reranker, carried over unchanged:
  <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- **`qdrant-client`** — `pip install qdrant-client`. The Python client for Qdrant (incl. snapshots).
- **`weaviate-client`** — `pip install weaviate-client`. The v4 Python client for Weaviate.
- **`psycopg[binary]`** — the Postgres/pgvector client from weeks 7–9, unchanged.
- **`docker compose`** — you bring up three database servers; a `docker-compose.yml` per store keeps it reproducible.

## A note on the corpus

The exercises and mini-project run against the **legal corpus** you've used since week 7 — a synthetic services agreement of ~50 clauses plus a 40-question gold set — chunked with your week-8 winner and retrieved with your week-9 hybrid pipeline. This week the *corpus content doesn't change*; what changes is the *store* it lives in, and we add **metadata** to each chunk (a `tenant`, a `clause_type`, a `version`) so the filtered-ANN exercises have something real to filter on. The gold set and `evaluate()` carry over from week 7 unchanged, so Recall@5/MRR mean the same thing across all three stores — the whole point of the bakeoff.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Vector store** | A database whose primary index is an ANN index over embeddings — but with backups, replicas, and migrations like any database. |
| **ANN index** | Approximate nearest-neighbor index (HNSW, IVF) — finds *close* vectors fast without scanning all of them. |
| **Filtered ANN** | Vector search combined with a metadata filter (`where tenant='acme'`). Harder than it looks; where stores differ most. |
| **Pre-filter** | Apply the metadata filter *first*, then ANN over the survivors. Exact but can be slow if the filter is broad. |
| **Post-filter** | ANN first, then drop results failing the filter. Fast but can return too few (recall collapse on selective filters). |
| **Native/in-filter** | The store filters *during* ANN traversal (Qdrant's filterable HNSW); the production answer. |
| **Metadata / payload index** | An index on the filterable fields, so filters are fast — as important as the vector index for multi-tenant retrieval. |
| **Snapshot / backup** | A point-in-time copy you restore from. The recovery-time metric depends on it. |
| **Time-to-recover** | How long from "index gone" to "Recall@5 back to baseline." The 2 AM number; a first-class selection criterion. |
| **Replication / HA** | Copies of the data on other nodes so a node loss doesn't lose the index. |
| **GraphRAG** | Build a knowledge graph + community summaries from the corpus; retrieve over it for multi-hop and global questions (Microsoft, 2024). |
| **Agentic RAG** | The agent chooses the retriever/store/strategy per query (or skips retrieval). Lift over a fixed pipeline, at a complexity cost. |

---

*If a link 404s, please open an issue so we can replace it.*
