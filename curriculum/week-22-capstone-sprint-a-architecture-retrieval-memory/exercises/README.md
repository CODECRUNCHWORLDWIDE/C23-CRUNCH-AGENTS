# Week 22 — Exercises

Three focused drills that take you from "I have components" to "I have a foundation with tested interfaces." Each takes 30–60 minutes. Do them in order — exercise 1 draws the architecture the other two implement, exercise 2 builds the retrieval interface, exercise 3 builds the memory interfaces.

## Index

1. **[Exercise 1 — The architecture diagram](exercise-01-architecture-diagram.md)** — draw the capstone's Mermaid architecture diagram and write the component/interface table, at the right level of detail. (~45 min, guided)
2. **[Exercise 2 — The retrieval interface](exercise-02-retrieval-interface.py)** — define and test the single `retrieve()` interface that fuses BM25 + dense + reranker behind one clean contract. (~50 min, runnable)
3. **[Exercise 3 — The memory tiers](exercise-03-memory-tiers.py)** — implement the three memory tiers behind their interfaces and pass the turn-38 regression test. (~50 min, runnable)

## How to work the exercises

- **Integrate, don't rebuild.** Sprint A's whole point is that you *connect* your week-8 chunker, week-9 hybrid retriever, and week-11 memory tiers — you don't write them from scratch. The exercises here build the *interfaces* that wrap your existing work. If your week-8/9/11 packages are broken, fix them first; this week depends on them.
- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install: `pip install sentence-transformers "psycopg[binary]" rank-bm25 numpy`.
- **Design the interface before the implementation.** Exercise 2's lesson is that the agents call `retrieve(query)` and don't know there's a BM25 leg behind it — so write the *signature* first, then fill in the hybrid internals. The interface is the contract; the implementation hides behind it.
- **Run the regression test continuously.** Exercise 3's turn-38 test exercises every memory tier; run a small version as you build each tier, not once at the end, so a failure points at the tier you just added (Lecture 2 §3d).
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone with fallbacks. Exercise 2 prefers a real corpus + pgvector but ships a tiny in-memory corpus so it runs anywhere. Exercise 3 is pure Python (no DB needed for the regression test — the semantic tier uses an in-memory vector list).

```bash
docker run -d --name crunch-pg -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17

# then, with the venv active:
python3 exercise-02-retrieval-interface.py
python3 exercise-03-memory-tiers.py
```

The first `SentenceTransformer("BAAI/bge-large-en-v1.5")` call downloads ~1.3 GB (cached if you have it from earlier weeks).

## A note on this being a capstone sprint

This is not a topic week with isolated drills — it's the first sprint of the capstone, and the exercises are *components of your actual capstone repo*, not throwaway practice. The `retrieve()` interface you build in Exercise 2 is the one your week-23 agents will call. The memory tiers in Exercise 3 are the ones the supervisor will read. Treat the code as production code headed for a graded deliverable, because it is. The "expected output" blocks show the *shape* of a working foundation; your real numbers come from your real corpus and gold set.

There are no solutions checked in — the capstone is *your* system. After you finish, the architecture document and the foundation are yours to carry into Sprint B.
