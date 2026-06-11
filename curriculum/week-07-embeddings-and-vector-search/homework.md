# Week 7 Homework

Six problems that revisit the week's topics and force the retrieval literacy into your fingers. The full set should take about **5 hours**. Work in your Week 7 Git repository (the same workspace as the exercises and the `crunchrag_embed` mini-project) so every problem produces at least one commit you can point to at the Phase II architecture review in Week 12.

The headline deliverable is **Problem 4 — the one-page embedding-choice memo**, called out explicitly in the syllabus. Treat it as the artifact a reviewer reads, not a journal entry.

Each problem includes a short **problem statement**, **acceptance criteria** so you know when you're done, a **hint** if you get stuck, and an **estimated time**.

Have pgvector running (`docker run -d --name crunch-pg -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17`) and your venv active with `sentence-transformers`, `psycopg[binary]`, and `numpy` installed. Problems 1–4 run against the legal corpus from the mini-project; if you haven't pulled it, the exercise-02 8-clause corpus is a fallback — say so in your writeup.

---

## Problem 1 — The vector-inspection table

**Problem statement.** Embed the same five sentences with `bge-large-en-v1.5`, `gte-large-en-v1.5`, and `nomic-embed-text-v1.5` (each with its correct convention). For each model record, in a markdown table at `notes/week-07/vector-inspection.md`: the dimension, whether the output is normalized (compute the L2 norm), the model's pooling type (from the card), and the cosine similarity between the first two sentences.

| Model | Dim | L2 norm of output | Pooling | cos(s0, s1) |
|---|---:|---:|---|---:|

**Acceptance criteria.**

- `notes/week-07/vector-inspection.md` exists with one row per model (three rows).
- Every number comes from your own `model.encode(...)` output, not the card.
- You note, in one line, why the cosine values differ across models even for the *same* two sentences (different spaces — vectors aren't comparable across models).
- Committed.

**Hint.** `np.linalg.norm(vec)` for the norm; `a @ b` for cosine on normalized vectors. The pooling type is in the model card or `model[1]` (the pooling module) in `sentence-transformers`.

**Estimated time.** 35 minutes.

---

## Problem 2 — Build and tune an index, prove the curve

**Problem statement.** Load the legal corpus into pgvector with `bge-large`, build an HNSW index, and run a 10-query subset of the gold set. Measure Recall@5 and median latency at `ef_search` ∈ {10, 40, 100, 200}. Build a table and identify the elbow.

**Acceptance criteria.**

- A `notes/week-07/efsearch-curve.md` with a four-row table: `ef_search | Recall@5 | median ms`.
- Recall rises (or holds) and latency rises as `ef_search` increases.
- You name the elbow `ef_search` (where recall has plateaued but latency is still low) in one sentence and justify it from your numbers.
- Committed.

**Hint.** Compute a brute-force ground truth top-5 for the 10 queries (full dot product over the corpus), then measure overlap with the ANN result at each `ef_search`. Exercise 3 is the template; here you use *real* embedded clauses instead of synthetic vectors.

**Estimated time.** 50 minutes.

---

## Problem 3 — Reproduce the normalization bug

**Problem statement.** Deliberately corrupt your pipeline: embed the *documents* with `normalize_embeddings=True` but the *queries* with `normalize_embeddings=False` (un-normalized). Run the 10-query subset and measure Recall@5. Then fix it (normalize both) and re-measure. Document the before/after.

**Acceptance criteria.**

- `notes/week-07/normalization-bug.md` records the Recall@5 *with* the bug and *without* it, with the actual numbers.
- You correctly explain *why* mixing normalized and un-normalized vectors corrupts the ranking (the magnitude of un-normalized vectors swamps the direction the metric should compare).
- You state, in one sentence, why this produced **no error** — just bad results.
- Committed.

**Hint.** With one side un-normalized, the dot products are on a wildly different scale and ranking by `<=>` becomes meaningless. The fix is to normalize everywhere, in one place. This is the homework version of the silent-bug lesson.

**Estimated time.** 40 minutes.

---

## Problem 4 — The one-page embedding-choice memo (headline deliverable)

**Problem statement.** This is the syllabus deliverable. Run the bakeoff (your mini-project, or the challenge harness) over the legal corpus with at least three models on the 40-query gold set. Then write a one-page memo at `notes/week-07/embedding-choice-memo.md` against this template:

1. **Recommendation** — one sentence: which embedding for this corpus, and the headline reason.
2. **The numbers** — the per-model top-1 / Recall@5 / MRR table, plus one line on the spread (big or small? what does that imply?).
3. **The trade-off** — why this model over the runner-up, naming the deciding axis (recall, dimension/storage, license, per-query cost, operational simplicity).
4. **The MTEB reality** — one honest sentence: did your measured winner match the leaderboard's favorite? If not, name the gap.
5. **Prevention / process** — one concrete process change so the *next* embedding decision is also measured (e.g. "the bakeoff harness runs in CI on every corpus change; we never pick from MTEB alone").

**Acceptance criteria.**

- `notes/week-07/embedding-choice-memo.md` exists, fits on roughly one page (350–550 words), and hits all five headings.
- Every metric in the table is **computed**, not estimated, from a real `evaluate()` run on a fixed gold set.
- The trade-off names a **specific** deciding axis, not "it had the best numbers."
- The MTEB sentence is honest about whether the leaderboard predicted your result.
- Committed.

**Hint.** The strongest memos pick a model that *isn't* the highest single number, because a 768-dim model within a point of a 1024-dim model wins on storage and latency. If your memo's reason is "highest Recall@5," push harder — name the trade-off you *accepted* to get there. That's the senior move.

**Estimated time.** 1 hour.

---

## Problem 5 — Matryoshka truncation, measured

**Problem statement.** Embed the corpus with `nomic-embed-text-v1.5` at full 768 dims and again truncated to 256 dims (take `v[:256]`, re-normalize). Run the 40-query gold set on each and report Recall@5 and MRR for both. Compute the storage saving (256/768 = one third).

**Acceptance criteria.**

- `notes/week-07/matryoshka.md` shows Recall@5 and MRR at 768 and 256 dims, plus the storage ratio.
- You state how much retrieval quality you traded for the 3x storage saving (usually very little).
- You note, in one sentence, when you'd take the truncation in production (large corpus, storage/latency-bound) and when you wouldn't (small corpus, quality-bound).
- Committed.

**Hint.** Truncating is `v_trunc = v[:256]; v_trunc /= np.linalg.norm(v_trunc)`. You re-normalize because the truncated vector is no longer unit length. The whole point of a Matryoshka model is that this *works* — a non-Matryoshka model would lose far more quality.

**Estimated time.** 45 minutes.

---

## Problem 6 — Negation probe

**Problem statement.** Construct five pairs of sentences that are *opposites* via a single negation (e.g. "the fee is refundable" / "the fee is not refundable"). Embed each pair with `bge-large` and compute the cosine similarity. Tabulate. Then add a true *paraphrase* pair (same meaning, different words) and compare its similarity.

**Acceptance criteria.**

- `notes/week-07/negation-probe.md` has the five negation pairs with their cosine similarities, plus the paraphrase pair.
- You observe that the negation pairs are *highly similar* (often > 0.9) despite being opposites, and the paraphrase pair is also highly similar.
- You explain, in two sentences, why this is a fundamental limit of dense retrieval and which week-9 technique (lexical/hybrid search) helps — and why even hybrid doesn't fully "solve" negation.
- Committed.

**Hint.** This is the homework that motivates week 9. Dense embeddings can't tell "refundable" from "not refundable" reliably, because the negation token is a tiny fraction of the meaning. Lexical search at least matches the exact word "not" — but the real fix for negation is downstream (a reranker or the LLM reading the actual text), which is the honest answer.

**Estimated time.** 35 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Vector-inspection table | 35 min |
| 2 — Index tuning curve | 50 min |
| 3 — Normalization bug | 40 min |
| 4 — Embedding-choice memo (headline) | 1 h 0 min |
| 5 — Matryoshka truncation | 45 min |
| 6 — Negation probe | 35 min |
| **Total** | **~4 h 45 min** |

When you've finished all six, push your repo and make sure the `crunchrag_embed` [mini-project](./mini-project/README.md) is in the same workspace — Week 8 imports it. Then take the [quiz](./quiz.md) with your notes closed.

---

## Grading rubric (homework)

Homework is graded on the same four axes as the weekly mini-project, scaled to the problem set.

| Axis | Weight | What "meets" looks like |
|---|---:|---|
| **Correctness** | 30% | The pipelines run; the metrics are computed correctly; the normalization and Matryoshka experiments produce the expected directional results. |
| **Engineering quality** | 25% | Readable scripts, the gold set fixed across comparisons, no convention leaks, sensible commits. |
| **Measurement** | 25% | Every claim is backed by a number from a real run — the memo, the curve, the truncation trade-off. Vibes do not count. |
| **Write-up** | 20% | The memo (Problem 4) hits all five sections, names a specific trade-off, and is honest about MTEB. The other notes are clear and reproducible. |

Graders are instructed to **fail vibes-only submissions** — a bakeoff with no measured metrics, or a memo that says "it felt best," is not a "meets." The whole week is about replacing intuition with numbers; the homework is graded the same way.
