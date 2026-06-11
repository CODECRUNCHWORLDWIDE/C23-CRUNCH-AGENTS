# Week 10 Homework

Six problems that revisit the week's topics and force vector-store-operations literacy into your fingers. The full set should take about **5 hours**. Work in your Week 10 Git repository (the same workspace as the exercises and the `crunchstore` mini-project) so every problem produces at least one commit you can point to at the Week 12 architecture review.

The headline deliverable is **Problem 4 — the store architecture memo**, the one a reviewer reads at week 12. The most *operationally* important is **Problem 5 — the recovery drill writeup**, the rehearsal for the week-24 chaos drill.

Have your **week-9 hybrid-retrieval pipeline** and **week-7 `crunchrag_embed`** (`evaluate()` reused unchanged) importable, and the three store containers runnable (`docker compose up` with pgvector/Qdrant/Weaviate, or at least Qdrant). If week 9 is broken, fix it first — this week swaps only the store underneath it.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Two stores behind one adapter

**Problem statement.** Implement the `VectorStore` adapter for **two** stores (pgvector + Qdrant is the natural pair) and run the *same* `evaluate()` with the *same* embedding and gold set against both. Produce `notes/week-10/two-stores.md` showing the two unfiltered Recall@5/MRR numbers and confirming they agree.

**Acceptance criteria.**

- Two adapters implement the same interface; one `evaluate()` runs against both via a swapped adapter.
- Unfiltered Recall@5 agrees (within ANN noise); any large gap is debugged to an adapter bug.
- Each adapter indexes its filterable metadata fields, not just the vector.
- Committed.

**Hint.** Port the Exercise 1 adapters. If pgvector and Qdrant disagree on unfiltered recall, check the distance metric (both cosine) and that you normalized the vectors. The stores *should* agree on basic ANN quality — that agreement is what lets you compare them on operations.

**Estimated time.** 45 minutes.

---

## Problem 2 — Measure filtered recall at a selective filter

**Problem statement.** Add a selective metadata filter (a rare `tenant`, matching a small fraction of chunks) and measure filtered Recall@5 two ways on one store: with the store's **native** filtered search, and with a **naive Python post-filter** (fetch top-K, drop non-matches). Produce `notes/week-10/filtered-recall.md` showing the post-filter recall collapse and the native recall holding.

**Acceptance criteria.**

- Filtered Recall@5 reported for native filtering vs naive post-filter, at a *selective* filter.
- The post-filter version measurably under-returns (recall collapse) where native holds.
- A one-sentence explanation grounded in Lecture 1 §3.
- Committed.

**Hint.** Make the tenant genuinely rare (e.g. 1–2% of chunks) so the collapse shows — at a broad filter both look fine (Exercise 2's lesson). Native filtering is `query_filter` (Qdrant) or `WHERE` on an indexed column (pgvector); the naive post-filter is "search k=20, then drop in Python," which silently returns too few for the rare tenant.

**Estimated time.** 45 minutes.

---

## Problem 3 — Ingest throughput and config complexity

**Problem statement.** For the two (or three) stores, measure **ingest throughput** (vectors/second to load the corpus + build the index) and count **lines of config** (docker-compose + adapter setup) to get each to first query. Produce `notes/week-10/ops.md` with a small table.

**Acceptance criteria.**

- Ingest vectors/sec measured for each store on the same rows.
- Lines-of-config (or time-to-first-query) recorded per store.
- A one-sentence note on which store was *operationally* lightest to stand up and why.
- Committed.

**Hint.** Time `create` + `upsert` + index build together — that's what bounds your recovery time too (Lecture 2 §2). Config complexity is a real selection criterion (Lecture 1 §5): a store you stand up in ten lines beats one needing a distributed-systems setup, unless you're at the scale that justifies it.

**Estimated time.** 40 minutes.

---

## Problem 4 — The store architecture memo (headline deliverable)

**Problem statement.** Run the bakeoff from Challenge 1 (three stores if you can, two minimum) and write a **one-page** memo at `notes/week-10/store-memo.md` against this template:

1. **Decision** — one sentence: which store you'd ship, and the headline operational reason (familiarity? filtered-recall? recovery time?).
2. **The scorecard** — the stores with Recall@5, MRR, ingest, query p95, filtered-recall at a selective filter, and time-to-recover.
3. **Why this winner, for this workload** — the operational mechanism (e.g. "we already run Postgres, so pgvector's pg_restore is a recovery our DBA has done a hundred times; the filtering is light enough that pgvector's filtered scan holds recall"), not a general claim.
4. **The trade-off accepted** — what you gave up (e.g. Qdrant's faster snapshots, in exchange for not running a second system).
5. **The recovery story** — the measured time-to-recover and how it was achieved (snapshot vs pg_dump vs re-embed), explicitly.
6. **The "unfiltered recall agrees" note** — state that the stores have comparable ANN quality, so the decision is operational, not about basic recall.

**Acceptance criteria.**

- `notes/week-10/store-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- The pipeline, corpus, embedding, and gold set are demonstrably identical across stores; only the store varies.
- The winner is justified by a **specific operational** mechanism, not "it felt production-grade."
- A measured time-to-recover is stated.
- Committed.

**Hint.** Don't pick on query latency alone (the trap). Lead with recovery time and filtered-recall, which actually decide production. If you ran only two stores, say so and reason about the third from the docs + a short test.

**Estimated time.** 1 hour.

---

## Problem 5 — The recovery drill writeup (operational headline)

**Problem statement.** Run a **real** recovery drill on at least one store: baseline → backup (snapshot/`pg_dump`) → drop the index → restore (timed) → confirm Recall@5 back to baseline. Then run the *disaster* version (recover by re-embedding from source) and time it. Produce `notes/week-10/recovery.md` comparing the two recovery times.

**Acceptance criteria.**

- A real backup-and-restore (not just re-ingest) with a measured time-to-recover and Recall@5 confirmed back to baseline.
- The re-embed-from-source recovery time, for contrast (the disaster path).
- A one-sentence conclusion: by how many times is snapshot-restore faster, and why that gap makes "we'll re-embed" an outage.
- Committed.

**Hint.** Qdrant snapshots (`create_snapshot`/`recover_snapshot`) make this clean; pgvector uses `pg_dump`/`pg_restore`. The re-embed path is "regenerate every vector from the source text, then rebuild" — time it honestly; on a real corpus it's hours. This is the week-24 chaos-drill rehearsal (Lecture 2 §2).

**Estimated time.** 50 minutes.

---

## Problem 6 — One question GraphRAG answers that flat retrieval misses

**Problem statement.** Build a *tiny* GraphRAG over the legal corpus (Microsoft's library, or a hand-rolled entity/relationship graph + community summaries), and find **one** multi-hop or global question it answers correctly that your flat hybrid pipeline gets wrong. Produce `notes/week-10/graphrag.md` with the question, both answers, and why GraphRAG won.

**Acceptance criteria.**

- A working (even minimal) GraphRAG over the corpus, and the flat hybrid pipeline as the baseline.
- One concrete question where GraphRAG answers correctly and flat retrieval misses (e.g. a multi-hop dependency or a corpus-wide theme).
- A one-sentence explanation of *why* flat retrieval structurally can't answer it (no single chunk contains the answer / it requires following relationships).
- Committed.

**Hint.** Pick a question that's *inherently* multi-hop or global — "which clauses does termination depend on?" (follow references) or "what are the corpus's main obligations?" (combine summaries). Flat retrieval finds *one similar chunk*; these need *relationships* or *synthesis* (Lecture 2 §3). One good example is the whole justification for the pattern.

**Estimated time.** 50 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Two stores, one adapter | 45 min |
| 2 — Filtered recall at a selective filter | 45 min |
| 3 — Ingest throughput + config complexity | 40 min |
| 4 — Store architecture memo (headline) | 1 h 0 min |
| 5 — Recovery drill writeup (operational headline) | 50 min |
| 6 — One question GraphRAG answers | 50 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchstore` [mini-project](./mini-project/README.md) is in the same workspace — Week 11 imports its adapter as the semantic memory tier, and the capstone's retrieval runs on the store this bakeoff picked. Then take the [quiz](./quiz.md) with your notes closed.
