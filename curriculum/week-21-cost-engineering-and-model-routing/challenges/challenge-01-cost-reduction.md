# Challenge 1 — The Cost-Reduction Lab

**Time estimate:** ~150 minutes.

## Problem statement

Your LLM product's bill is growing linearly with usage, and someone has to bring it down without anyone noticing in the answers. You are going to do it the only honest way: build a cost-engineered pipeline (semantic cache + difficulty router + cascade), run it against a *fixed, labeled* 500-query workload, measure the cost reduction *and* the quality delta against the naive all-frontier baseline, and write a memo that commits to a configuration with both numbers behind it.

This is the syllabus cost-engineering deliverable in lab form. The output is a **decision** — ship this pipeline config — justified by a measured cost cut and a measured quality delta, not by a vibe about which lever "feels" worth it.

## What's fixed (do not let these vary)

- **The workload:** the fixed 500-query labeled set (resources.md) — a known difficulty mix (labeled easy/hard) plus a set of near-duplicate paraphrases so the cache has something to hit. Fixing it is what makes the cost reduction and the quality delta comparable across configs. (No labeled set handy? The `--mock` path ships one and simulates the tiers.)
- **The baseline:** all 500 queries to the frontier model. This is the cost and quality you measure *against*. The reduction is relative to it; the quality delta is relative to it.
- **The quality metric:** an LLM-judge (or labeled-answer accuracy) scoring whether each pipeline answer matches the baseline's quality, reported as a delta. The tolerance is stated up front (e.g. -0.03).
- **The cost source:** `usage` blocks, never estimated tokens. Cache hits cost $0; routed-to-cheap costs the cheap-tier price; escalations cost both.

## The harness approach

The pipeline is a stack of short-circuiting layers (Lecture 1 §6); the measurement runs the same 500 queries through it and through the baseline.

```python
from crunchroute import semantic_cache, router, cascade, cost_of, judge

def pipeline(query, true_difficulty):
    # 1. semantic cache (Exercise 2)
    hit = semantic_cache.lookup(query, threshold=0.92)
    if hit:
        return hit.answer, "cache-hit", 0.0

    # 2. route by difficulty (Exercise 3)
    if router.classify(query) == "easy":
        answer, cost = cheap_complete(query)          # local 7B / Haiku
        route = "local"
    else:
        # 3. cascade on the hard ones: try cheap, escalate if verify fails
        answer, route, cost = cascade.run(query, cheap_complete, frontier_complete, verify)

    semantic_cache.insert(query, answer)              # warm the cache
    return answer, route, cost

# measure: run all 500 through pipeline() and through the baseline, compare
results = [pipeline(q, d) for q, d in WORKLOAD]
baseline = [frontier_complete(q) for q, _ in WORKLOAD]
```

Then aggregate: total cost vs baseline, per-route breakdown (cache / local / frontier / escalated), cache-hit rate over the run, and the quality delta (judge each pipeline answer against the baseline).

## Acceptance criteria

- [ ] A `challenge-01/` directory with a runnable `cost_lab.py` that runs the full pipeline and the baseline over the 500-query workload and prints the cost-reduction report.
- [ ] The report shows **total cost vs baseline (the reduction %), the per-route breakdown (cache / local / frontier / escalated), the cache-hit rate over time, and the quality delta** vs baseline.
- [ ] Tokens come from `usage` (cache hits = $0); the quality delta is measured (LLM-judge or labeled accuracy), not assumed.
- [ ] The semantic-cache threshold and the router/verifier thresholds are the ones you *tuned* (Exercises 2–3), not guesses.
- [ ] A one-page `cost-memo.md` that names the **config you ship** (thresholds + which levers), gives the **cost reduction and the quality delta**, states the **tolerance** the delta must stay inside, and names the **trap** you avoided (a cheaper config you rejected because its quality delta was out of tolerance).
- [ ] At least one **per-query trace** in the promise format: `q137 ("what's our refund window?") -> cache HIT (0.97) $0.00 ✓`, plus one query where a *too-aggressive* config gave a wrong/worse answer.

## The trap (read after a first attempt)

The trap is **maximizing the cost reduction and ignoring the quality delta.** It is trivially easy to cut cost 95% — route everything to the 7B, set the cache threshold to 0.80, and your bill collapses. And so does your hard-question accuracy, your cache returns the refund answer to return questions, and the product is measurably worse — but the *cost dashboard looks fantastic*, so nobody notices until the users do. **A cost reduction without a quality delta next to it is not a result; it's the first half of a quality incident.** Every config you report must carry both numbers, and a config that wins on cost but loses on quality past tolerance is *rejected*, no matter how good the cost number looks.

A second, subtler trap: **measuring quality on the wrong set.** The quality damage from routing lives specifically on the *routed-to-cheap* queries (where a hard query got a cheap answer) and the *cache-hit* queries (where a paraphrase got a slightly-off cached answer). If you average quality over *all* 500 queries, the frontier-handled hard queries (which are fine) dilute the damage and hide it. Measure the quality delta on the queries the cost-saving levers *touched* — that's where the regression hides, and where the all-500 average will lie to you.

## Stretch goals

- **Tune the cascade verifier.** Replace the hard verifier with a calibrated one (a cheap LLM-judge with a tuned pass threshold), and plot the cost-vs-quality frontier as you move the threshold. Find the point that cuts the most cost while keeping the quality delta inside tolerance (Lecture 2 §6c).
- **Add prompt compression.** Compress the longest prompts with LLMLingua before sending to the frontier model, and measure the additional saving *and* the quality delta. Find the compression ratio where quality starts to drop (Lecture 1 §5).
- **Batch the patient slice.** Identify the queries in the workload that don't need a real-time answer, run them through the Anthropic Batch API for the 50% discount, and report the additional saving (Lecture 2 §4).
- **The cost-per-query distribution.** Report cost per query at the median, p95, and p99 (Lecture 1 §6c) — the exact shape the capstone's cost report requires. Show the cheap default (low median) and the expensive tail (frontier escalations at p99).

## Why this matters

The capstone (weeks 22–24) requires a **cost report** — "cost per query at the median, p95, p99, broken down by route, with cache-hit accounting" — and runs **cost-tracked routing between local 7B/13B and a vendor frontier model.** This challenge *is* that report and that routing layer, built and measured a sprint early. A reviewer will point at the capstone's serving cost and ask "how do you know this is cheap *and* still good?", and this lab is that conversation rehearsed with numbers. The bill fell, the quality held, and you can prove both — which is the only kind of cost saving that survives contact with production.
