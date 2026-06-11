# Week 11 Homework

Six problems that revisit the week's topics and force memory-systems literacy into your fingers. The full set should take about **5 hours**. Work in your Week 11 Git repository (the same workspace as the exercises and the `crunchmem` mini-project) so every problem produces at least one commit you can point to at the Week 12 architecture review.

The headline deliverable is **Problem 4 — the memory-architecture memo**, the one a reviewer reads. The most *measurement*-important is **Problem 5 — the eviction-policy comparison**, where you choose LRU vs salience with a number.

Have your **week-10 `crunchstore`** importable (the semantic tier reuses its adapter) and pgvector running (`docker run -d -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17`). For the summarizer/judge, set `ANTHROPIC_API_KEY` (`claude-sonnet-4-6` is the reference) or point at a local model from week 6. If `crunchstore` is broken, fix it first — this week's semantic tier depends on it.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Budget a window and prove it stays bounded

**Problem statement.** Build a context budget with named slices (system / semantic / episodic / recent / query), measure a 40-turn conversation two ways — naive append-everything vs budgeted — and produce `notes/week-11/budget.md` showing the naive prompt's linear token growth and the budgeted prompt's bounded footprint, with the turn-3 fact still present in the budgeted version.

**Acceptance criteria.**

- Token counts for naive (growing) vs budgeted (bounded) over 40 turns, measured with the *model's tokenizer*.
- The budgeted prompt is a fraction of the naive one and still contains the turn-3 fact (promoted to the semantic slice).
- A one-sentence statement of the naive approach's failure mode (overflow / cost / lost-in-the-middle).
- Committed.

**Hint.** Use `client.messages.count_tokens(...)` for Claude (never `tiktoken` — wrong tokenizer). The budgeted prompt stays bounded because old turns collapse into the rolling summary and durable facts live in the semantic slice — the transcript never grows unbounded.

**Estimated time.** 40 minutes.

---

## Problem 2 — Implement and test the rolling-summary tier

**Problem statement.** In your `crunchmem` package, implement `EpisodicMemory` with a rolling summary + recent window (Exercise 2's logic, real summarizer optional). Write `tests/test_tiers.py` proving (a) the token footprint stays bounded as turns grow, and (b) a fact the summary drops is *lost* unless also stored in semantic memory.

**Acceptance criteria.**

- `EpisodicMemory` implemented with a bounded rolling summary + recent window.
- A test showing the footprint stays bounded over many turns.
- A test showing a non-durable fact is lost from the summary (motivating the semantic tier).
- Committed.

**Hint.** The stub summarizer from Exercise 2 (keep durable-keyword sentences) is deterministic and perfect for testing. Assert the rolling footprint stays under a cap as you add 40 turns, and assert a fact with no durable keyword disappears from the summary — that's the lossiness, in a test.

**Estimated time.** 45 minutes.

---

## Problem 3 — Wire the semantic tier to crunchstore

**Problem statement.** Implement `SemanticMemory` backed by your week-10 `crunchstore` adapter: extract durable facts from turns, upsert them, retrieve by similarity to a query. Produce `notes/week-11/semantic.md` showing a planted fact ("project: Helios") stored and then retrieved by a paraphrased query ("what's my project name?").

**Acceptance criteria.**

- `SemanticMemory` uses the `crunchstore` adapter (or the toy embedder from Exercise 3 if the store is down, noted).
- A planted fact is upserted and retrieved by a *paraphrased* query (similarity, not exact match).
- The retrieval returns the fact even when the query wording differs from the stored fact.
- Committed.

**Hint.** This is week 10's retrieval, pointed at accumulated facts instead of a corpus — same `embed` + `search` shape. The test of a real semantic tier is that "what's my project name?" retrieves "My project is called Helios" despite different wording; that's what makes it robust to how the user phrases the turn-38 question.

**Estimated time.** 45 minutes.

---

## Problem 4 — The memory-architecture memo (headline deliverable)

**Problem statement.** Run the three-tier agent and the no-memory baseline through the 40-turn benchmark (Challenge 1 / the mini-project) and write a **one-page** memo at `notes/week-11/memory-memo.md` against this template:

1. **Decision** — one sentence: your memory architecture (three tiers + budget + eviction policy) and the headline recall delta (e.g. "three-tier 0.90 vs baseline 0.10").
2. **The tiers** — what each tier holds, its storage, and which tier carried the recall (usually semantic for old facts).
3. **The budget** — your token allocation across slices, and that it's measured with the model's tokenizer.
4. **The eviction policy** — LRU or salience, *chosen with the number* from a comparison (Problem 5).
5. **The recall measurement** — the three-tier-vs-baseline recall rates and the delta; one per-query trace in the promise format.
6. **The trap avoided** — that durable facts go to the semantic tier (not just the lossy summary), and that you asked *late* (after the recent window).

**Acceptance criteria.**

- `notes/week-11/memory-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- The recall delta is reported, three-tier vs no-memory baseline.
- The winning tier and eviction policy are justified by *numbers*, not "it felt better."
- At least one per-query trace in the promise format.
- Committed.

**Hint.** The recall delta is the headline — it's what justifies the whole system, exactly as the Recall@5 delta justified a chunking strategy in week 8. Make sure you asked about each planted fact *after* it scrolled out of the recent window, or the baseline trivially passes and the delta vanishes.

**Estimated time.** 1 hour.

---

## Problem 5 — Eviction-policy comparison (measurement headline)

**Problem statement.** Implement both LRU and salience-weighted eviction, run the 40-turn benchmark under each, and produce `notes/week-11/eviction.md` comparing their recall rates. State which you'd ship and the number that justifies it.

**Acceptance criteria.**

- Both LRU and salience eviction implemented and pluggable.
- The benchmark run under each, with recall rates reported.
- A one-sentence decision: which policy, and by how much it beat the other on old-but-important facts.
- The embedding/store/budget are held fixed; only the eviction policy varies (one variable).
- Committed.

**Hint.** Salience should win where the benchmark plants *important old* facts (the project name) among *recent trivial* ones — salience keeps the project name and drops the recent "thanks!", while LRU drops the old project name. Predict the direction first, then measure. If they tie, your benchmark may not have enough old-important-vs-recent-trivial tension; add some.

**Estimated time.** 50 minutes.

---

## Problem 6 — Measure lost-in-the-middle yourself

**Problem statement.** Build a long context with a planted fact at the *start*, the *middle*, and the *end*, ask a real model to recall it from each placement, and produce `notes/week-11/lost-in-the-middle.md` with the recall result per placement. Confirm (or challenge) the U-shaped curve.

**Acceptance criteria.**

- A long context (e.g. 8k+ tokens of filler) with the same fact planted at start / middle / end across three runs.
- The model's recall result for each placement.
- A one-sentence conclusion: did middle placement recall worse than the edges (the lost-in-the-middle effect), and what that implies for where you place retrieved facts in your budget.
- Committed.

**Hint.** Use a real model (`claude-sonnet-4-6` or a local long-context model) — the toy harness can't show this. Keep everything identical except the fact's *position*. The expected finding is the U: start and end recall well, middle recalls worse — which is *why* the budgeter places semantic facts and the query at the edges (Lecture 2 §3).

**Estimated time.** 50 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Budget a window, prove it's bounded | 40 min |
| 2 — Rolling-summary tier + tests | 45 min |
| 3 — Semantic tier wired to crunchstore | 45 min |
| 4 — Memory-architecture memo (headline) | 1 h 0 min |
| 5 — Eviction-policy comparison (measurement) | 50 min |
| 6 — Measure lost-in-the-middle | 50 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchmem` [mini-project](./mini-project/README.md) is in the same workspace — the Phase II milestone requires three tiers wired and measured, and the capstone imports your `MemoryAgent`. Then take the [quiz](./quiz.md) with your notes closed.
