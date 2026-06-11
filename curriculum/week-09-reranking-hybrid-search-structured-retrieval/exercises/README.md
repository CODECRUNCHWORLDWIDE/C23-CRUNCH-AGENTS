# Week 9 — Exercises

Three focused drills that take you from "BM25 wins where dense lost" to "the reranker pulled the right doc to rank 1." Each takes 30–60 minutes. Do them in order — exercise 2 fuses the kind of lists exercise 1 produces, and exercise 3 reranks the kind of candidate set exercise 2 fuses.

## Index

1. **[Exercise 1 — BM25 baseline](exercise-01-bm25-baseline.md)** — build a `rank-bm25` index over the legal corpus, run the 40-query gold set, and prove BM25 wins exactly the queries dense missed (an id, "Delaware", "$1,000,000"). (~45 min, guided)
2. **[Exercise 2 — RRF fusion](exercise-02-rrf-fusion.py)** — implement Reciprocal Rank Fusion over a dense ranked list and a BM25 ranked list, and prove the fused order beats either alone on the queries each one fumbles. (~40 min, runnable)
3. **[Exercise 3 — Reranker lift](exercise-03-reranker-lift.py)** — score a candidate set with `BAAI/bge-reranker-v2-m3` and watch the cross-encoder pull the right document from rank 4 to rank 1. (~45 min, runnable)

## How to work the exercises

- Reuse the **week-7 venv** if you have it (you already have `sentence-transformers`, `psycopg`, `numpy`). Add this week's two deps: `pip install rank-bm25` and — for exercise 3 — confirm `sentence-transformers` is current.
- Exercise 1 needs **no model and no DB** for the BM25 part — `rank-bm25` is pure Python. The dense comparison reuses your week-7 pgvector pipeline if you have it; if not, the exercise documents a model-only fallback.
- Exercise 3 **downloads `BAAI/bge-reranker-v2-m3`** (~600 MB) on first run. Do it on Monday on good wifi, not five minutes before a deadline.
- Everything runs on **CPU**. The reranker is slower on CPU than GPU but never blocked — we rerank tiny candidate sets, so it's fast enough.
- **Read the lift before you trust the layer.** Every exercise ends in a comparison you can check: BM25 vs dense, fused vs either, reranked vs first-stage. A layer that doesn't move a number doesn't belong in your pipeline.

## Running the Python exercises

The two `.py` files are standalone:

```bash
# exercise 2 needs nothing but rank-bm25 (and it's self-contained with hardcoded lists)
python3 exercise-02-rrf-fusion.py

# exercise 3 downloads the reranker on first run
python3 exercise-03-reranker-lift.py
```

Each runnable exercise ends with an **expected output** block. If your output doesn't match the *shape* (exact scores vary by model version and machine), you're not done.

## A note on determinism

`rank-bm25` is fully deterministic — same corpus and query, same scores every run. The reranker is deterministic too (same model, same pair, same score). What *will* vary is the absolute score magnitudes across model versions and machines; the *ordering* is what's invariant and what you're proving. When exercise 3 says "the right doc moves from rank 4 to rank 1," that rank movement is the lesson, not the exact reranker logit.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-09` to compare.
