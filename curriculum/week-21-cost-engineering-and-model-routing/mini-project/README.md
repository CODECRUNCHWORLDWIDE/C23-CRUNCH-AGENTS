# Mini-Project — `crunchroute`: The Cost-Engineering Harness

> Build a reusable cost-engineering module that any LLM product can import to put a semantic cache, a difficulty router, and a cascade in front of its models, run a fixed labeled workload through it, and emit a **cost-reduction report** — total cost vs baseline, per-route breakdown, cache-hit rate over time, and the quality delta — so "how much did we save and did quality hold?" becomes a command, not an argument.

This is the artifact that turns cost engineering from folklore into a measurement. After this week, cutting an LLM bill is `python -m crunchroute run --workload workload.jsonl` and reading a report — not guessing which lever is worth it. The harness is workload-agnostic, lever-pluggable, and quality-honest, and it produces the exact cost report the capstone is graded on.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This harness is the seed of the **capstone's cost-and-routing layer (weeks 22–24)**. The capstone's required cost report — "cost per query at the median, p95, p99, broken down by route, with cache-hit accounting" — is exactly what this harness emits. Its cost-tracked routing between local 7B/13B and a vendor frontier model is exactly the router you build here, executed through the week-19 LiteLLM proxy. Build it well now; in week 22 you point it at the capstone's real query distribution and it produces the deliverable.

---

## What you will build

A small Python package `crunchroute` with four deliverables:

1. **`crunchroute/cache.py`** — a semantic cache (embed the query, pgvector lookup, return above a cosine threshold) plus an exact-match front layer, with an invalidation hook (TTL / version key). The single source of truth for "have we answered this (or a paraphrase) before?"
2. **`crunchroute/router.py`** — a difficulty classifier (heuristic by default, pluggable to few-shot / trained) and a cascade (try cheap, verify, escalate). The single source of truth for "which model handles this query, and when do we escalate?"
3. **`crunchroute/cost.py`** — token accounting from `usage` (input/output/cache-read pricing), per-route cost attribution, and the cost-per-query distribution (median/p95/p99). The single source of truth for "what did this cost, and where?"
4. **`crunchroute/cli.py`** — a `run` command that ties cache + router + cascade + cost together over a workload and emits the cost-reduction report (against an all-frontier baseline), with a quality delta.

By the end you have a public repo of ~400–500 lines of Python (plus the workload) that any future LLM project can `from crunchroute import pipeline` and stop overpaying.

---

## Why a module and not a notebook

You could do all of this in a Jupyter notebook. Don't — not as the artifact. A module gives you:

- **Reuse.** The capstone imports your `pipeline` and your `cost.py` to build its cost-and-routing layer and its cost report. A notebook gets copy-pasted, drifts, and rots.
- **A fixed measurement.** The workload, the quality metric, and the "tokens from `usage`, quality delta required" discipline live in code, version-controlled. "Did this threshold change help?" is answered by re-running the *same* `run`, not by eyeballing a new notebook cell.
- **A CLI.** `crunchroute run --workload x.jsonl --threshold 0.92` is greppable, scriptable, and CI-able. A notebook cell is none of those.

Notebooks are great for *exploring* a single threshold by eye. The thing you ship and depend on is a module. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchroute/
├── pyproject.toml
├── docker-compose.yml          # pgvector/pgvector:pg17 on localhost:5432 (from week 7)
├── README.md                   # the cost-reduction report + the cost memo
├── workload/
│   ├── queries.jsonl           # 500 labeled queries: {"query": "...", "difficulty": "easy|hard", "dup_of": "..."}
│   └── gold.jsonl              # reference answers / labels for the quality delta
├── crunchroute/
│   ├── __init__.py
│   ├── cache.py                # exact-match + semantic cache (pgvector) + invalidation
│   ├── router.py               # difficulty classifier + cascade (verify + escalate)
│   ├── cost.py                 # usage-based cost accounting + per-route + p50/p95/p99
│   ├── judge.py                # the quality-delta scorer (LLM-judge / labeled accuracy)
│   └── cli.py                  # the `run` command
└── tests/
    ├── test_cache.py           # threshold behavior; exact-before-semantic; invalidation
    ├── test_cost.py            # the cost math from usage is correct
    └── test_router.py          # classifier + cascade routing + expected-cost math
```

The semantic cache prefers Postgres + pgvector (week 7's container) but ships an in-memory fallback so the harness develops anywhere. The real-model path needs an API key; a `--mock` path simulates the tiers' cost and quality so the logic and measurement run keyless.

---

## Deliverable 1 — `cache.py` (exact-match + semantic + invalidation)

The cache is the biggest single lever on a repetitive workload, and it's where quality leaks if the threshold is loose.

```python
"""crunchroute.cache — exact-match front layer + semantic cache + invalidation.

Exact-match is checked first (free); semantic second (catches paraphrases). The
cosine threshold is the cost-vs-correctness knob (Lecture 1 §3); insertions carry
a TTL / version so stale answers can be invalidated.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np


def exact_key(query: str, **params) -> str:
    return hashlib.sha256(
        json.dumps({"q": query, **params}, sort_keys=True).encode()).hexdigest()


class Cache:
    def __init__(self, embed_fn, threshold: float = 0.92, ttl_s: float = 86400.0):
        self.embed = embed_fn
        self.threshold = threshold
        self.ttl = ttl_s
        self._exact: dict[str, tuple[str, float]] = {}          # key -> (answer, ts)
        self._semantic: list[tuple[np.ndarray, str, str, float]] = []  # (vec, query, answer, ts)

    def lookup(self, query: str, **params):
        now = time.time()
        # 1. exact-match (cheapest check)
        k = exact_key(query, **params)
        if k in self._exact:
            answer, ts = self._exact[k]
            if now - ts <= self.ttl:                            # not stale
                return answer
        # 2. semantic (catch paraphrases)
        # TODO 1: embed query, find the nearest non-stale entry by cosine
        #   (pgvector <=> in production), and return its answer iff
        #   similarity >= self.threshold. Return None on miss.
        ...

    def insert(self, query: str, answer: str, **params):
        # TODO 2: store in BOTH the exact-match map (by exact_key) and the
        #   semantic list (with the query embedding + timestamp).
        ...

    def invalidate(self, predicate):
        # TODO 3: drop entries matching `predicate(query)` -- the version-key /
        #   event-driven purge (Lecture 1 §3b). Stale answers are a correctness
        #   bug, not just a cost issue.
        ...
```

> **The rule the project enforces:** the semantic threshold ships *tuned* against the labeled set, reported with both its hit rate and its wrong-answer rate (Exercise 2). A threshold with no wrong-answer-rate number next to it is the trap — it might be cheap *and* wrong.

---

## Deliverable 2 — `router.py` (classifier + cascade)

The router decides the tier; the cascade catches the cheap model's failures.

```python
HARD_KEYWORDS = {"compare", "explain", "analyze", "derive", "evaluate", "why"}


def classify(query: str, n_docs: int = 0) -> str:
    """Heuristic difficulty classifier (Lecture 2 §4d option 1). Biased toward
    'hard' because false-easy errors cost quality and false-hard only cost money."""
    q = query.lower()
    if len(q.split()) > 9 or any(k in q for k in HARD_KEYWORDS) or n_docs > 3:
        return "hard"
    return "easy"


def cascade(query, cheap_complete, frontier_complete, verify):
    """Try cheap, escalate on verifier failure. Returns (answer, route, cost)."""
    cheap_answer, cheap_cost = cheap_complete(query)
    # TODO 4: run verify(query, cheap_answer); if it passes, return the cheap
    #   answer + route "cheap". If it fails, call frontier_complete, and return
    #   the frontier answer + route "escalated" + (cheap_cost + verify_cost +
    #   frontier_cost). Track P(escalate) for the expected-cost report.
    ...
```

The non-negotiable: **the classifier is biased conservative** (toward `hard`), because a false-easy error degrades a user's answer (a quality failure you won't see in the cost dashboard) while a false-hard error just overpays (a cost failure that's harmless to quality). The quality delta is measured on the *routed-to-cheap* set, where false-easy errors live.

---

## Deliverable 3 — `cost.py` (usage accounting + the distribution)

The module that turns `usage` blocks into the report.

```python
PRICES = {  # 2026 Claude prices, $/1M tokens in/out
    "claude-haiku-4-5":  {"in": 1.0, "out": 5.0},
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
    "claude-opus-4-8":   {"in": 5.0, "out": 25.0},
}


def cost_of(usage, model: str) -> float:
    p = PRICES[model]
    return (usage.input_tokens  * p["in"]  / 1e6
          + usage.output_tokens * p["out"] / 1e6
          + getattr(usage, "cache_read_input_tokens", 0) * p["in"] * 0.1 / 1e6)


def distribution(per_query_costs: list[float]) -> dict:
    """The capstone's required shape: median / p95 / p99 cost per query."""
    import numpy as np
    a = np.array(per_query_costs)
    return {"median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99))}
```

`cost.py` is pure arithmetic over `usage` — no network — so `test_cost.py` pins it down exactly (known `usage` → known cost and distribution).

---

## Deliverable 4 — `cli.py` (the `run` command and the report)

```bash
python -m crunchroute run \
    --workload workload/queries.jsonl \
    --threshold 0.92 \
    --cheap claude-haiku-4-5 --frontier claude-sonnet-4-6 \
    --tolerance 0.03
```

It runs the pipeline and the baseline over the workload and prints:

```
COST REDUCTION REPORT (500-query workload)
  baseline (all-frontier)   : $7.20
  engineered pipeline       : $0.84     -> 88% reduction
  per-route:
    cache hits   (41%)      : $0.00
    local        (44%)      : $0.31
    frontier     (8%)       : $0.42
    escalated    (7%)       : $0.11
  cache-hit rate over time  : 0% -> 41% (warmed over the run)
  cost/query   median=$0.0003  p95=$0.02  p99=$0.06
  quality delta vs baseline : -0.01   (tolerance -0.03)   ✓ WITHIN TOLERANCE
```

The report is the syllabus deliverable and the capstone's cost report in miniature — and it carries the quality delta *next to* the cost, because a cost number without it is half the story.

---

## Rules

- **You may** read the pricing docs, the lecture notes, and your Exercise 1/2/3 code.
- **You must** read tokens from `usage`, never estimate them — an estimated denominator corrupts the whole report.
- **You must** measure the quality delta (LLM-judge or labeled accuracy) on the routed-to-cheap and cache-hit sets, and report it next to the cost. A cost reduction without a quality delta is rejected.
- **You must not** ship a semantic-cache or router threshold you didn't tune against the labeled set.
- Any Claude call (cheap tier, frontier tier, or the judge) uses `client.messages.create(...)` with `thinking={"type": "adaptive"}` and `output_config={"effort": ...}` — never `budget_tokens` or `temperature`.
- Python 3.12, `anthropic`, `openai`, `sentence-transformers`, `psycopg[binary]`, `numpy`, plus `pytest`. The pgvector + real-model path needs services; the `--mock` + cost path runs anywhere.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-21-crunchroute-<yourhandle>`.
- [ ] `docker compose up -d` brings up pgvector; the semantic cache runs against it (or the in-memory fallback).
- [ ] `cache.py` implements exact-match + semantic + invalidation; the threshold is tuned and reported with its wrong-answer rate.
- [ ] `router.py` implements the conservative classifier and the cascade with the expected-cost math.
- [ ] `cost.py` computes per-route cost and the median/p95/p99 distribution from `usage`; `test_cost.py` proves the math.
- [ ] `python -m crunchroute run ...` prints a cost-reduction report with the reduction %, per-route breakdown, cache-hit rate over time, the cost/query distribution, *and* the quality delta vs baseline.
- [ ] `pytest` passes (`test_cache.py` threshold + invalidation, `test_cost.py` math, `test_router.py` routing + expected cost).
- [ ] A `README.md` with the report and the one-page cost memo (config shipped, reduction + quality delta, the tolerance, the trap avoided).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Cost accounting** | 20 | Tokens from `usage`, never estimated; per-route attribution correct; median/p95/p99 distribution computed; cache hits = $0. |
| **Caching** | 20 | Exact-match before semantic; threshold tuned and reported with its wrong-answer rate; an invalidation hook exists (stale answers are a correctness bug). |
| **Routing & cascade** | 20 | Conservative classifier; cascade with verify+escalate; the expected-cost math reported; the right lever for the workload. |
| **The quality delta** | 20 | Quality measured (judge/accuracy) on the routed-to-cheap + cache-hit sets, reported next to the cost; a config out of tolerance is rejected, not shipped. |
| **Tests** | 10 | `pytest` green; cache threshold + invalidation, cost math, routing + expected cost all covered. |
| **Docs & hygiene** | 10 | Clear README + cost memo; no secrets committed; sensible commits; no `__pycache__`/`.venv`/`models/`. |

**90+** is portfolio-grade and drops straight into the capstone's cost-and-routing layer. **70–89** works but has a soft mapping — an untuned threshold, a quality delta over the whole set instead of the touched set, or a missing invalidation story. **Below 70** means the harness saves money without proving quality held — fix that first, because the capstone's cost report is graded on exactly this honesty.

---

## Stretch goals

- **Calibrated cascade.** Replace the hard verifier with a cheap LLM-judge at a tuned pass threshold; plot the cost-vs-quality frontier (Lecture 2 §6c).
- **Prompt compression leg.** Add LLMLingua compression on the longest prompts and measure the extra saving and the quality delta; find the ratio where quality drops (Lecture 1 §5).
- **Batch leg.** Route the latency-tolerant slice through the Anthropic Batch API for the 50% cut and report the additional saving (Lecture 2 §4).
- **Live-drift monitoring.** Add the three drift signals (cache-hit rate down, escalation up, frontier fraction up) as metrics, the down-payment on the capstone's cost dashboard and week 24's cost-spike alarm (Lecture 2 §6b).

---

## How this connects to the rest of C23

- **Week 7 (embeddings)** gave you pgvector and the BGE embedder; the semantic cache is that machinery in a new use — a cache keyed by query meaning.
- **Week 19 (vLLM + LiteLLM)** gave you the cheap local tier and the router that *executes* the routing decision; this week decides *which* tier each query goes to. Routing (by choice) here + fallback (by failure) from week 19 = the full cost-and-resilience layer.
- **Capstone (weeks 22–24)** runs cost-tracked routing between local 7B/13B and a vendor frontier model and requires a cost report — both of which *are* this harness. You point `crunchroute` at the capstone's real query distribution in week 22 and it produces the graded deliverable.

When you've finished, push the repo and take the [quiz](../quiz.md).
