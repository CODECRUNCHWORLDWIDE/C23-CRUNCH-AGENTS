# Week 21 Homework

Six problems that revisit the week's topics and force cost-engineering literacy into your fingers. The full set should take about **5 hours**. Work in your Week 21 Git repository (the same workspace as the exercises and the `crunchroute` mini-project) so every problem produces at least one commit you can point to when the capstone's cost-and-routing layer gets built.

The headline deliverable is **Problem 6 — the one-page cost-reduction memo**, called out explicitly in the syllabus. Treat it as the artifact a reviewer reads to decide whether your cost engineering held quality, not a journal entry.

Have an `ANTHROPIC_API_KEY` for the real-model path (or use the `--mock` paths each `.py` ships) and pgvector running for the semantic cache (`docker run -d -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17`). This week builds on **week 7** (the pgvector + BGE machinery the cache reuses) and **week 19** (the local cheap tier you route to). If those are broken, fix them first.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Build a per-route cost table and find the leak

**Problem statement.** Meter a small workload (real calls or representative `usage` records) across at least three features at different models, including one deliberately wasteful one (a trivial task on a frontier model). Produce `notes/week-21/cost-table.md` with a per-feature, per-model cost table from `usage`, and a paragraph naming the leak and the lever that fixes it.

**Acceptance criteria.**

- A per-feature, per-model cost table built from `usage` (input/output, and cache-read where applicable).
- The biggest *justified* spend and the biggest *wasteful* spend identified.
- The lever for the leak named (route / cache / compress) with an estimated saving.
- Committed.

**Hint.** Scan for volume × model-price: a high-volume trivial task on Opus (50K classify calls/day) often outspends a few hundred chat calls. Output tokens cost 4–5× input, so check `output_tokens`, not just prompt length, when hunting the expensive feature.

**Estimated time.** 40 minutes.

---

## Problem 2 — Tune a semantic-cache threshold

**Problem statement.** Using Exercise 2's harness (or your `crunchroute/cache.py`), sweep the semantic-cache cosine threshold against the labeled set and produce `notes/week-21/cache-sweep.md` with the hit-rate-and-wrong-answer-rate table and a sentence naming the threshold you'd ship and *why* (the loosest one inside the wrong-answer tolerance).

**Acceptance criteria.**

- A sweep table reporting, per threshold, both the hit rate and the wrong-answer rate.
- The chosen threshold is the loosest one whose wrong-answer rate is under a stated tolerance — *not* the highest hit rate.
- One sentence justifying the choice with both numbers.
- Committed.

**Hint.** If every threshold is either all-wrong or all-miss, your labeled look-alikes are too close for the embedder — use BGE (not the hashing fallback), or report that finding honestly. The sweet spot is where the hit rate is still meaningful but the look-alike queries ("refund window" vs "return window") haven't started matching.

**Estimated time.** 45 minutes.

---

## Problem 3 — Add prompt caching and measure the discount

**Problem statement.** Take a feature with a repeated large prefix (a long system prompt). Add `cache_control: {"type": "ephemeral"}` to the prefix, call it several times, and measure the cost with and without the cache read from `cache_read_input_tokens`. Then *deliberately break it* by injecting a per-request timestamp at the front of the prefix and show the cache stops hitting. Produce `notes/week-21/prompt-cache.md` with both results.

**Acceptance criteria.**

- A before/after cost comparison showing `cache_read_input_tokens > 0` and the measured discount on the repeated-prefix calls.
- The broken case: a timestamp in the prefix drives `cache_read_input_tokens` to zero, with the cost rising back to full.
- One sentence on why prefix byte-stability is the whole game (same lesson as week 8 / week 19).
- Committed.

**Hint.** The cache key is the prefix bytes; any change at the front invalidates everything after it. Put the timestamp *after* the stable prefix (or drop it) and the cache hits again. Verify with `cache_read_input_tokens`, not by guessing.

**Estimated time.** 40 minutes.

---

## Problem 4 — Build a router and measure the quality delta

**Problem statement.** Using Exercise 3's harness (or your `crunchroute/router.py`), run a difficulty router over the labeled set, sending easy→cheap and hard→frontier. Produce `notes/week-21/router.md` with the cost reduction vs an all-frontier baseline *and* the quality delta measured *on the routed-to-cheap queries*. State whether the delta is inside tolerance, and if not, what you'd change in the classifier.

**Acceptance criteria.**

- Cost reduction vs the all-frontier baseline, with the per-route breakdown.
- The quality delta measured specifically on the *routed-to-cheap* set (where false-easy errors live), not the whole workload.
- A statement of whether the delta is inside tolerance and, if not, the classifier change (bias more conservative) to fix it.
- Committed.

**Hint.** Measuring quality over all queries dilutes the routing damage — the frontier-handled hard queries are fine and hide the cheap-tier failures. Measure on the touched set. If the delta is bad, the fix is almost always a more conservative classifier (more false-hard, fewer false-easy), per Lecture 2 §1b.

**Estimated time.** 50 minutes.

---

## Problem 5 — Build a cascade and do the expected-cost math

**Problem statement.** Add a cascade (try cheap, verify, escalate) to the same workload. Produce `notes/week-21/cascade.md` with the measured `P(escalate)`, the expected-cost formula (`cost_cheap + cost_verify + P(escalate) × cost_frontier`) computed with your numbers, the cost vs the router and the baseline, and the quality delta. State whether the cascade beats the router on this workload and why.

**Acceptance criteria.**

- The measured escalation rate `P(escalate)` and the expected-cost formula computed with your actual numbers.
- Cost compared against both the router (Problem 4) and the all-frontier baseline.
- The quality delta, showing the cascade holds quality (the verifier catches cheap-model failures) where the router didn't.
- A sentence on whether the cascade or the router is the right pick for this workload.
- Committed.

**Hint.** The cascade pays when `P(escalate)` is low and the cheap model is much cheaper (the break-even escalation rate is ~90% for a 10× cost gap). Its advantage over the router is that it doesn't need the classifier to be right up front — it verifies *after* generating, so it catches the cheap model's actual failures rather than predicting them.

**Estimated time.** 45 minutes.

---

## Problem 6 — The one-page cost-reduction memo (headline deliverable)

**Problem statement.** This is the syllabus deliverable. Combining the semantic cache (Problem 2), the router (Problem 4), and the cascade (Problem 5) into one pipeline over the 500-query workload, write a **one-page** cost-reduction memo at `notes/week-21/cost-memo.md` against this template:

1. **Decision** — one sentence: the pipeline config you ship (thresholds + which levers), and the headline reduction.
2. **The cost reduction** — total cost vs the all-frontier baseline (the reduction %), with the per-route breakdown (cache / local / frontier / escalated).
3. **The quality delta** — measured on the routed-to-cheap + cache-hit sets (where the damage hides), vs baseline, with the tolerance it stays inside.
4. **The cost-per-query distribution** — median, p95, p99 (the capstone's required shape): cheap by default, expensive only in the tail.
5. **The trap avoided** — a cheaper config you *rejected* because its quality delta was out of tolerance (the "make it cheaper at all costs" trap).
6. **One per-query trace** — in the promise format: `q137 ("what's our refund window?") -> cache HIT (0.97) $0.00 ✓` for a win, plus one query where a too-aggressive config gave a worse answer.

**Acceptance criteria.**

- `notes/week-21/cost-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- The reduction is justified by **measured** cost and the quality delta is measured on the **touched** set — not "it felt cheaper."
- The cost-per-query median/p95/p99 distribution is reported.
- The rejected-config trap is named honestly.
- At least one promise-format per-query trace.
- Committed.

**Hint.** The whole memo turns on reporting the quality delta *next to* the cost reduction. A reviewer doesn't trust an 88% cut without seeing the quality held; pair them in every claim. Measure the delta on the queries the levers *touched* (routed-to-cheap, cache-hit), where regressions hide — the all-500 average will lie to you.

**Estimated time.** 1 hour.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Per-route cost table + find the leak | 40 min |
| 2 — Tune a semantic-cache threshold | 45 min |
| 3 — Prompt caching + measure the discount | 40 min |
| 4 — Router + quality delta | 50 min |
| 5 — Cascade + expected-cost math | 45 min |
| 6 — Cost-reduction memo (headline) | 1 h 0 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchroute` [mini-project](./mini-project/README.md) is in the same workspace — the capstone's cost-and-routing layer and its required cost report both start from this harness. Then take the [quiz](./quiz.md) with your notes closed.

Up next: **Week 22 — Capstone Sprint A**, where you start building the production agentic research assistant for real, and this cost-engineered routing layer becomes its serving-and-cost layer.
