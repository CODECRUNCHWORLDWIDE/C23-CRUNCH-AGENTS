# Week 7 — Exercises

Three focused drills that take you from "what is a vector" to "I measured my retrieval." Each takes 30–60 minutes. Do them in order — exercise 3 reuses the index you build in exercise 2, which reuses the embedding intuition from exercise 1.

## Index

1. **[Exercise 1 — Embed and inspect](exercise-01-embed-and-inspect.md)** — embed text with three open models, inspect the raw vectors, prove cosine ranking by hand, and reproduce the query/document-prefix bug on purpose. (~45 min, guided)
2. **[Exercise 2 — pgvector k-NN](exercise-02-pgvector-knn.py)** — stand up Postgres + pgvector in Docker, embed a small corpus, build an HNSW index, and run k-NN queries that actually return the right clause. (~45 min, runnable)
3. **[Exercise 3 — Recall vs `ef_search`](exercise-03-recall-vs-efsearch.py)** — sweep `ef_search` against a brute-force ground truth and chart the recall/latency curve, so the iron triangle stops being abstract. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install the deps once: `pip install sentence-transformers "psycopg[binary]" numpy`.
- The first `SentenceTransformer(...)` call **downloads the model** (a few hundred MB to ~1.3 GB). Do it on Monday on good wifi, not five minutes before a deadline.
- Everything runs on **CPU**. It's slower than GPU but never blocked. The first embedding call is slow (model load); subsequent calls are fast.
- **Read the score before you trust the result.** A retrieval that returns *something* is not a retrieval that returns the *right* thing. Every exercise ends in a number you can check.
- When retrieval is bad, run the §4 decision tree from Lecture 2 before you touch code. Prefix first, normalization second, index-op third, `ef_search` fourth.

## Running the Python exercises

The two `.py` files are standalone. Exercise 2 and 3 need Postgres + pgvector running. The fastest way:

```bash
docker run -d --name crunch-pg \
  -e POSTGRES_PASSWORD=crunch \
  -p 5432:5432 \
  pgvector/pgvector:pg17

# then, with the venv active:
python3 exercise-02-pgvector-knn.py
```

Each runnable exercise ends with an **expected output** block. If your output doesn't match the *shape* (exact scores vary by model version and machine), you're not done.

## A note on determinism

Embedding models are deterministic — the same text gives the same vector every run. But ANN search is *approximate*, so exercise 3's recall numbers will wobble slightly run to run and machine to machine. That's expected. The *shape* of the curve (recall rises with `ef_search`, latency rises with it too) is what's invariant and what you're proving.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-07` to compare.
