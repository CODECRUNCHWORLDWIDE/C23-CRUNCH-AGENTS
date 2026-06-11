# Week 10 — Exercises

Three focused drills that take you from "pgvector is where the vectors live" to "I chose a store on an operational number I measured." Each takes 30–60 minutes. Do them in order — exercise 3 (the recovery drill) uses the store adapter you build in exercise 1, and the filtered-ANN intuition from exercise 2 is the technical reason one store beats another.

## Index

1. **[Exercise 1 — Three stores, same pipeline](exercise-01-three-stores-same-pipeline.md)** — bring up pgvector, Qdrant, and Weaviate in Docker and run the *same* retrieval against all three behind one adapter interface. (~50 min, guided)
2. **[Exercise 2 — Filtered ANN](exercise-02-filtered-ann.py)** — measure pre-filter vs post-filter vs native filtered search on a selective filter and *see* post-filter's recall collapse where native filtering holds. (~50 min, runnable)
3. **[Exercise 3 — The recovery drill](exercise-03-recovery-drill.py)** — baseline → backup → destroy the index → restore (timed) → record time-to-recover, the 2 AM number that reorders the stores. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install the deps as each exercise needs them: `pip install qdrant-client weaviate-client "psycopg[binary]" numpy sentence-transformers`. Each store also needs its **Docker container** (commands below).
- **Hold the pipeline fixed; vary only the store.** Same embedding (BGE-large), same chunking-A/B winner from week 8, same `evaluate()` from week 7. Changing the store *and* the embedding is how you get numbers you can't compare.
- **Measure recall at a *filter*, not just unfiltered latency.** Exercise 2's whole point is that the interesting (and dangerous) behavior is at a *selective* filter, where post-filter silently returns too few results.
- **Time the restore, not just the query.** Exercise 3's headline is time-to-recover, the operational number a benchmark never gives you.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the stores in Docker

```bash
# pgvector (you know this one from weeks 7-9):
docker run -d --name crunch-pg -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17

# Qdrant:
docker run -d --name crunch-qdrant -p 6333:6333 qdrant/qdrant

# Weaviate:
docker run -d --name crunch-weaviate -p 8080:8080 -p 50051:50051 \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  cr.weaviate.io/semitechnologies/weaviate:1.27.0
```

Exercises 2 and 3 are written so they run against **Qdrant alone** if you only stand up one store — the filtered-ANN and recovery lessons are identical, and the headers document the single-store fallback. The full three-store comparison is the challenge.

The first `SentenceTransformer("BAAI/bge-large-en-v1.5")` call downloads ~1.3 GB. Do it on good wifi, not five minutes before a deadline.

## A note on determinism

ANN search is approximate (you learned this in week 7), so Recall@5 can wobble by a hair across runs — tiny on a corpus this size. Ingest throughput and restore time depend on disk and machine load, so run each **3 times and report the median**, exactly as the drills do. The *shape* — post-filter recall collapsing on a selective filter, snapshot-restore recovering in seconds — is what's reproducible and what you're proving; if you can't reproduce the shape, something changed (the filter selectivity, the corpus, the store version).

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-10` to compare.
