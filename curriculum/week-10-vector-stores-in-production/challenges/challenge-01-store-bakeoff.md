# Challenge 1 — The Vector-Store Bakeoff

**Time estimate:** ~150 minutes.

## Problem statement

You have a corpus, a working retrieval pipeline, and a store decision to make. Three reasonable people on your team each swear by a different vector store. You are going to end the argument the only way it can honestly end: run the *same* pipeline against each store, measure the *operational* axes that actually decide a production choice, and let the numbers — including the one nobody benchmarks until it bites them — pick the store. Then you write the architecture memo, because at the week-12 review a reviewer will point at your index and ask "why *that* store?"

This is the syllabus hands-on lab in committed form. The output is a decision: one vector store, with ingest throughput, query p50/p95, filtered-search recall at a selective filter, config complexity, and **time-to-recover** — and a paragraph of reasons grounded in *your* workload and operations.

## The three stores

Run exactly these three, same pipeline against each:

1. **pgvector** — the Postgres-native default; your baseline from weeks 7–9. The one your team likely already operates.
2. **Qdrant** — the filtered-ANN specialist (Rust, filterable HNSW, snapshots). The one to beat on filtered search and recovery.
3. **Weaviate** — the generative/graph-leaning, schema-first store.

## What is fixed (do not let these vary)

- **Pipeline:** your week-9 hybrid retrieval — dense (week-7 BGE embedding, week-8 chunking-A/B winner) + BM25 + RRF fusion + bge-reranker. The *same* pipeline against every store.
- **Corpus + metadata:** the legal corpus, chunked your week-8 way, with `tenant` / `clause_type` / `version` metadata attached so the filtered query is real.
- **Gold set:** the 40-query legal gold set from week 7, unchanged. Gold is in *clause ids*; you retrieve *chunks*; you map chunk hits back to clause ids before scoring (the week-8 discipline).
- **Metric suite:** Recall@5 and MRR (the spine, via `evaluate()` unchanged) plus the operational metrics: ingest vectors/sec, query p50/p95, filtered-query recall at a selective `tenant` filter, lines-of-config, and time-to-recover.

## The harness approach

The whole bakeoff reduces to: implement the `VectorStore` adapter (Exercise 1) for each store, then run the *same* measurement against each adapter.

```python
from crunchrag_embed.eval import evaluate     # week 7, UNCHANGED

STORES = {"pgvector": PgvectorStore(), "qdrant": QdrantStore(), "weaviate": WeaviateStore()}

for name, store in STORES.items():
    t0 = time.perf_counter(); store.create("clauses", dim=1024)
    store.upsert("clauses", rows); ingest_s = time.perf_counter() - t0   # ingest throughput

    rfn = build_retrieve_fn(store, "clauses")           # same pipeline, this store
    metrics = evaluate(gold, rfn, k=5)                  # Recall@5 / MRR (the spine)
    p50, p95 = latency_percentiles(rfn, gold)           # query latency
    filt_recall = filtered_recall(store, selective_tenant)   # the §3 number
    ttr = recovery_drill(store, "clauses", gold, rfn)["time_to_recover_s"]  # 2 AM number

    record(name, ingest_s, metrics, p50, p95, filt_recall, ttr)
```

That identical measurement per store is the whole point: you changed only the store. The scorecard is the result.

## Acceptance criteria

- [ ] A `challenge-01/` directory with a runnable `bakeoff.py` (built on the Exercise 1 adapters) that runs all three stores through the same pipeline and prints a comparison scorecard.
- [ ] The scorecard reports, per store: **Recall@5, MRR, ingest vectors/sec, query p50/p95, filtered-query recall at a selective filter, lines-of-config, and time-to-recover.**
- [ ] The pipeline, corpus, embedding, and gold set are demonstrably **identical** across stores — only the store varies.
- [ ] Unfiltered **Recall@5 agrees** across stores (they have the same ANN quality on this corpus); the stores are differentiated on the *operational* axes, not basic recall.
- [ ] A **recovery drill** is run against at least one store (Qdrant snapshot, or pgvector `pg_dump`/`pg_restore`), with a measured time-to-recover and a confirmation that Recall@5 returns to baseline.
- [ ] A one-page `store-memo.md` that names the **store you'd ship**, gives its numbers, and explains in a paragraph **why it won for this workload** (multi-tenant filtering? Postgres familiarity? recovery time?) — not in general.
- [ ] At least one **promise-format line**, e.g. `store=qdrant -> filtered Recall@5 0.86, p95 19ms, RECOVERED in 47s`, plus a counter-example (a store whose filtered recall sagged, or whose recovery was slow).

## The trap (read after a first attempt)

The trap is **picking the store on query latency alone.** Every well-tuned store will query fast enough; the p50/p95 differences are real but rarely decisive, and if you rank the stores on that number you'll miss the ones that actually decide production: **recovery time** and **filtered-search recall at your selectivity.** A store that queries 2 ms faster but restores via a 4-hour re-embed, or silently returns nothing for your rare tenants (post-filter recall collapse, Lecture 1 §3), is the *worse* choice — and you'll only see that if you measure recovery and filtered-recall, not just latency. **Measure the operational axes, and weight them above the query speed everyone over-weights.**

A second, subtler trap: **measuring filtered recall only at a broad filter.** A filter that matches 40% of the corpus won't expose the post-filter collapse — you need a *selective* filter (a rare tenant) to see which stores hold recall and which don't. Run the filtered query at a high-selectivity filter, or your filtered-recall number is measuring nothing (the exact lesson of Exercise 2).

## Stretch goals

- **Add Milvus or Chroma as a fourth store.** Milvus exposes where billion-scale architecture is overkill for 50 clauses; Chroma exposes how much less config a prototyping store needs (Lecture 1 §2). Either teaches the scale-matching lesson.
- **Real recovery on all three.** Run the drill against pgvector (`pg_dump`/`pg_restore`) *and* Qdrant (snapshots) *and* Weaviate (backup module) and compare the three recovery times directly. That comparison is the heart of the memo.
- **GraphRAG question.** Build a tiny GraphRAG over the corpus and find one multi-hop or global question it answers that your flat hybrid pipeline misses (Lecture 2 §3). That one question justifies the pattern.
- **Agentic router.** Add a small per-query router (vector / filter / skip) and measure the lift over the fixed pipeline on a heterogeneous query set (Lecture 2 §4). Ship it only if it earns its added latency.

## Why this matters

In Week 12 you defend your whole retrieval pipeline at the architecture review, and the capstone serves a hybrid-retrieval system over a 10 GB corpus that has to survive the week-24 chaos drill's "retrieval index corruption" scenario. The reviewer will not ask you to recite the five stores — they'll point at your index and ask "why *that* store, and how fast does it come back when it's gone?" This challenge *is* that conversation, rehearsed: you ran the alternatives, you have the scorecard, you can name the store and the recovery number that justifies it. Every RAG system you ship after this stores its vectors *somewhere* whether you chose it deliberately or not — the engineer who *chose* it, with a measured recovery time and filtered-recall number behind the choice, is the one whose retrieval doesn't quietly fail for the smallest tenant at the worst hour. The index survived the loss, and you can prove how fast.
