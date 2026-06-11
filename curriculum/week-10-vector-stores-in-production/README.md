# Week 10 — Vector Stores in Production

Welcome to the week the demo grows up. For three weeks you've measured retrieval — embeddings (week 7), chunking (week 8), reranking and hybrid search (week 9) — against a vector store you mostly took for granted: pgvector in a Docker container, `<=>`, an HNSW index, done. This week the store stops being a given and becomes a *decision*. You'll run the same hybrid pipeline against **pgvector**, **Qdrant**, and **Weaviate**, and measure not just query latency but the things that actually decide a production choice: ingest throughput, filtered-search performance, and — the one nobody benchmarks until it bites them — *how long it takes to recover when the index is gone at 2 AM*.

This is week 4 of **Phase II — RAG & Memory Systems**, and it sits on top of weeks 7–9. Everything here assumes you can embed a corpus, chunk it, run hybrid (dense + BM25 + reranker) retrieval, and *measure* it with the `evaluate()` harness you've carried since week 7. This week you point that same discipline at a new variable: not *what* you retrieve or *how* you rank it, but *what system stores the vectors* — and what that system costs you in operations, not just in milliseconds.

The one sentence to internalize before you read another line:

> **Pick the vector store with the operational story you can live with at 2 AM, not the one with the best benchmark.** The benchmark is run on a good day by the vendor. The 2 AM story — backup, restore, replication, the rebuild after a schema change — is run by you, on a bad day, with a customer on the phone.

Here's why that's not hyperbole. Every vector store on the market will do approximate nearest-neighbor search fast enough for most workloads; the query-latency differences between a well-tuned pgvector and a well-tuned Qdrant are real but rarely decisive. What *is* decisive is what happens around the query: Can you back it up with a tool your team already knows? Can you restore it in minutes or does it take a four-hour re-embed? Does it survive a node loss? Can you change the metadata schema without rebuilding the whole index? Those questions — not the leaderboard QPS — are what you'll be living with. The store you pick is the store you operate, and this week is where you learn to choose for the operating, not the benchmarking.

There's a corollary worth taping next to last week's mantra:

> **A vector store is a database first and a vector index second.** It has backups, replicas, migrations, and failure modes like any database. The "vector" part is one index type; the "store" part is everything that keeps your data alive. Treat it like the database it is.

## Learning objectives

By the end of this week, you will be able to:

- **Map** the 2026 vector-store landscape — **pgvector** (Postgres-native, the default), **Qdrant** (Rust, fast filtered search), **Weaviate** (graph-leaning, generative), **Milvus** (massive scale), **Chroma** (developer ergonomics, smaller scale) — and place each by the workload and operational story it fits.
- **Run** the same hybrid-retrieval pipeline against pgvector, Qdrant, and Weaviate and measure ingest throughput, query latency (p50/p95), and **filtered-ANN** performance on the same corpus and gold set.
- **Reason** about **filtered ANN** — why combining a metadata filter with vector search is harder than it looks (pre-filter vs post-filter vs the store's native filtered index) and why it's where stores differ most.
- **Operate** a vector store like a database: take a backup, restore it, reason about replication and high availability, and **rebuild the index after a schema change** without re-embedding from scratch where the store allows it.
- **Measure** operational complexity honestly — lines of config, time-to-first-query, and **time-to-recover from a simulated index loss** — and treat that recovery number as a first-class selection criterion.
- **Explain** **GraphRAG** (Microsoft's knowledge-graph-augmented retrieval pattern) and knowledge-graph hybrids — when a graph over your corpus answers questions a flat vector index can't (multi-hop, "what connects X and Y").
- **Describe** **Agentic RAG** — the pattern where the *agent* chooses which retriever (or which store, or whether to retrieve at all) to use — and where it earns its complexity over a fixed pipeline.
- **Write** a vector-store architecture memo: the store you'd ship, the numbers (ingest, p95, recovery time, config complexity) that justify it, and the operational trade-off you accepted.

## Prerequisites

This week assumes you have completed **C23 weeks 1–9**, or have equivalent fluency. Specifically:

- You finished **week 9** and have a hybrid-retrieval pipeline: dense (your chunking-A/B winner from week 8, embedded with your week-7 model) + BM25 + RRF fusion + a bge-reranker, with `evaluate()` returning Recall@5 / MRR. **This week swaps only the store underneath it** — if the pipeline is broken, fix it first.
- You're comfortable with **pgvector** from weeks 7–9: `vector_cosine_ops`, the `<=>` operator, building an HNSW index, `ef_search`. This week pgvector is the *baseline* the other stores must beat on the metric *you* care about.
- Python 3.12 on Linux, macOS, or WSL2; **Docker and `docker compose`** are non-negotiable this week — you'll bring up three different database servers in containers. A virtualenv you can `pip install` into.
- You can read the week-7/8/9 retrieval metrics. We reuse Recall@5 and MRR all week and add **ingest throughput**, **p95 query latency**, and **time-to-recover** as the new, operational signals.

You do **not** need a GPU for the store work (it's I/O and CPU-bound database work). Embedding the corpus needs the same compute as weeks 7–9 (a local embedding model on CPU is fine for the small corpus). You do **not** need prior database-ops experience — we cover backup/restore/recovery from the RAG engineer's angle, not the DBA's.

## Topics covered

- **The landscape, placed:** pgvector (Postgres-native default), Qdrant (Rust, filtered ANN), Weaviate (generative/graph-leaning), Milvus (billion-scale), Chroma (ergonomics, prototyping) — chosen by workload and operations, not by leaderboard.
- **Filtered ANN:** the hard problem of "vector search *and* `where tenant='acme'`" — pre-filter vs post-filter vs native filtered indexes — and why it's where stores actually differ.
- **Metadata indexes:** payload/metadata indexing for fast filters; why a metadata index is as important as the vector index for multi-tenant and faceted retrieval.
- **Operational realities:** backup and restore; replication and high availability; the rebuild-after-schema-change problem; eviction and TTL; the "your index is gone, how fast is it back" drill.
- **GraphRAG:** Microsoft's pattern — build a knowledge graph from the corpus, retrieve over graph communities for multi-hop and global questions a flat index can't answer; the cost and when it's worth it.
- **Knowledge-graph hybrids:** vector + KG stores; when the relationships between entities are the retrieval signal.
- **Agentic RAG:** the agent picks the retriever/store/strategy (or skips retrieval); the lift over a fixed pipeline and the complexity tax.
- **Store selection as engineering:** the same hybrid pipeline, three stores, measured on ingest / latency / filtered-search / recovery-time — and a memo that picks one with reasons.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                            | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | The landscape; pgvector/Qdrant/Weaviate; filtered ANN            |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Filtered search + metadata indexes; the multi-store exercise     |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Operational realities: backup/restore/replication/recovery       |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | GraphRAG + agentic RAG; the bakeoff harness                      |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The three-store bakeoff + recovery drill + memo                  |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                            |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                         |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                                  | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The store docs (pgvector/Qdrant/Weaviate/Milvus/Chroma), the GraphRAG paper, filtered-ANN references, backup/restore docs |
| [lecture-notes/01-the-vector-store-landscape.md](./lecture-notes/01-the-vector-store-landscape.md) | The five stores placed by workload, filtered ANN, metadata indexes, and the selection criteria that actually matter |
| [lecture-notes/02-operations-graphrag-and-agentic-rag.md](./lecture-notes/02-operations-graphrag-and-agentic-rag.md) | Backup/restore/replication/recovery, the index-loss drill, GraphRAG, KG hybrids, and agentic RAG |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-three-stores-same-pipeline.md](./exercises/exercise-01-three-stores-same-pipeline.md) | Bring up pgvector, Qdrant, and Weaviate and run the same retrieval against all three |
| [exercises/exercise-02-filtered-ann.py](./exercises/exercise-02-filtered-ann.py) | Measure pre-filter vs post-filter vs native filtered search and see where the recall/latency trade lives |
| [exercises/exercise-03-recovery-drill.py](./exercises/exercise-03-recovery-drill.py) | Simulate an index loss, restore from backup, and measure time-to-recover |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-store-bakeoff.md](./challenges/challenge-01-store-bakeoff.md) | The full bakeoff: three stores, one pipeline, ingest/latency/filter/recovery, pick a store with a memo |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the architecture memo and the recovery-drill writeup |
| [mini-project/README.md](./mini-project/README.md) | The `crunchstore` adapter + bakeoff harness — one interface, pluggable stores, an operational scorecard |

## The "it survived the index loss" promise

C23 uses a recurring marker for every exercise that ends in a store actually proving itself *operationally*, with a number that matters at 2 AM:

```
$ python recovery_drill.py --store qdrant
store=qdrant
  ingest: 12,400 vec/s   query p95: 14ms   filtered query p95: 19ms
  *** simulating index loss (drop collection) ***
  restore from snapshot...  Recall@5 back to 0.88 in 47s
  RECOVERED in 47s — the index came back before the customer hung up
```

That `RECOVERED in 47s` line is the whole point of the week. A store that queries 2 ms faster but takes four hours to restore from a re-embed is the *worse* production choice, and the only way you know that is by *running the drill*. The point of week 10 is to make the store choice an *operational* decision — proven with a recovery number, not a vibe about which store "feels" production-grade.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **GraphRAG paper** (Edge et al., Microsoft, 2024, arXiv 2404.16130) until you can explain why community-summary retrieval answers "global" questions ("what are the main themes across the corpus?") that flat vector search can't: <https://arxiv.org/abs/2404.16130>. Then build a tiny GraphRAG over your legal corpus and find one question it answers that hybrid retrieval misses.
- Add **Milvus** as a fourth store and measure where its scale architecture (separate query/data/index nodes) helps and where it's overkill for a 50-clause corpus. The lesson is matching the store's scale to the corpus's.
- Build an **agentic-RAG router**: a small classifier/LLM that decides per query whether to hit the vector store, the BM25 index, or skip retrieval entirely. Measure the lift (and the added latency/cost) over the fixed hybrid pipeline.
- Do a **filtered-ANN deep dive**: on Qdrant, compare its native filtered HNSW against a naive post-filter on a high-selectivity filter (e.g. `tenant='rare'`). Watch post-filter's recall collapse and native filtering hold — the §3 lesson, measured.

## Up next

Week 11 takes the store you can now operate and asks a new question: not "where do the vectors live" but "what should the agent *remember*, and for how long?" You'll build **memory systems** — episodic, semantic, procedural tiers — and budget the context window like a cache, using a vector store (the one you chose this week) as the semantic-memory tier. Push your `crunchstore` bakeoff before you start; week 11's semantic memory plugs straight into the store adapter you build here.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
