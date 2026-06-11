# Week 21 — Exercises

Three focused drills that take you from "I have a model bill" to "I cut it 80% and proved the answers held." Each takes 30–60 minutes. Do them in order — exercise 1's accounting tells you where the money is, exercise 2 caches the repeats, exercise 3 routes the rest.

## Index

1. **[Exercise 1 — Token accounting](exercise-01-token-accounting.md)** — meter a real multi-turn conversation, attribute cost per route from `usage`, and build the cost table that tells you which lever to pull. (~40 min, guided)
2. **[Exercise 2 — The semantic cache](exercise-02-semantic-cache.py)** — build a semantic cache over pgvector and sweep the cosine threshold against the cost-vs-correctness trade-off. (~50 min, runnable)
3. **[Exercise 3 — The router and cascade](exercise-03-router.py)** — build an easy/hard classifier router and a cascade with a verifier; measure cost reduction and the quality delta against an all-frontier baseline. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install per exercise: `pip install anthropic openai sentence-transformers "psycopg[binary]" numpy`.
- **Read tokens from `usage`, never estimate.** Every cost number in this week starts from `response.usage` (`input_tokens`, `output_tokens`, `cache_read_input_tokens`). `len(text.split())` is off by 20–40% and corrupts every downstream number. This is the week-21 analog of week 8's "count tokens in the model's units."
- **Measure the quality delta, not just the cost.** Exercises 2 and 3 both have a way to save money that silently returns worse answers (a loose cache threshold, an over-aggressive router). The acceptance criteria require you to measure *both* the saving and the quality impact — a saving without a quality check is the trap.
- **Use the right SDK params.** Any Claude call uses `client.messages.create(...)` with `thinking={"type": "adaptive"}` and `output_config={"effort": ...}` — never `budget_tokens` or `temperature` (those 400 on current models).
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone. Exercise 2 prefers Postgres + pgvector (same container as week 7) but ships an in-memory fallback so it runs anywhere. Exercise 3 needs a cheap and a frontier model; it ships a `--mock` path that simulates the two tiers' cost and quality so the routing *logic* and the *measurement* run with no API key.

```bash
docker run -d --name crunch-pg -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17

# then, with the venv active:
export ANTHROPIC_API_KEY=sk-ant-...     # for the real-model path (optional; --mock works without)
python3 exercise-02-semantic-cache.py --threshold-sweep
python3 exercise-03-router.py --mock
```

The first `SentenceTransformer("BAAI/bge-large-en-v1.5")` call downloads ~1.3 GB (the same model as week 7 — if you have it cached, it's instant). Do it on good wifi.

## A note on the labeled workload

Cost savings only mean something against a *labeled* workload — one where you know which queries are paraphrase-duplicates (so the cache should hit) and which are hard (so the router should escalate). The exercises ship a small labeled set; the mini-project and challenge use the full 500-query set. Without labels you can measure the cost drop but not the *quality delta*, and the quality delta is the half of the result that keeps you honest. The same discipline as every measurement week: a number without a method (and here, without a quality check) doesn't count.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-21` to compare.
