# Week 12 Homework

Six problems that revisit the week's topics and force evaluation literacy into your fingers. The full set should take about **5 hours**. Work in your Week 12 Git repository (the same workspace as the exercises and the `crunchrag_eval` mini-project) so every problem produces at least one commit you can point to at the Phase II architecture review.

The headline deliverable is **Problem 4 — the Ragas evaluation report memo**, called out explicitly in the syllabus as the **Phase II milestone**. Treat it as the artifact a reviewer reads, not a journal entry.

Have your **weeks-7–11 `crunchrag_embed` package** importable (`evaluate()` and `store.py` are reused unchanged) and pgvector running (`docker run -d -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17`). If your Phase II pipeline is broken, fix it first — this week measures it.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Ask a VLM and Claude the same figure question

**Problem statement.** Render a PDF page that has a figure or table to an image (week-8 `get_pixmap`). Ask the **same question** over the picture two ways — an open VLM (Qwen2.5-VL local, or its Ollama/documented fallback) *and* Claude vision (`claude-opus-4-8`) — and run the week-8 *text* extraction on the same page. Produce `notes/week-12/multimodal.md` with all three outputs and a one-paragraph comparison.

**Acceptance criteria.**

- `notes/week-12/multimodal.md` exists with the VLM answer, the Claude-vision answer, and the text-extraction output for the same page.
- A named, verifiable fact (e.g. an amount in the figure) is shown present in the VLM/Claude answers and stated whether the text pipeline had a clean chunk for it.
- One paragraph: which path you'd use for this page and why (reads vs. retrieves).
- Committed.

**Hint.** Use the authoritative Claude image-block shape — `{"type":"image","source":{"type":"base64","media_type":"image/png","data":<b64>}}` then a text block — and no `temperature` on `claude-opus-4-8`. If you have no GPU, the Claude leg alone plus the documented expected VLM behavior is acceptable. (Lecture 1 §1–3; Exercise 1.)

**Estimated time.** 40 minutes.

---

## Problem 2 — Implement and unit-test one Ragas metric from scratch

**Problem statement.** In your `crunchrag_eval` package (or a standalone module), implement **faithfulness** from scratch following Exercise 2: decompose the answer into claims and check each against the context with a pluggable judge (the deterministic stub is fine). Write `tests/test_metrics.py` proving that (a) a fully-grounded answer scores 1.0, and (b) an answer with one hallucinated claim scores strictly less.

**Acceptance criteria.**

- `faithfulness(answer, contexts, judge)` is implemented and importable.
- `pytest tests/test_metrics.py` passes with at least two assertions: grounded answer = 1.0; one-hallucination answer < 1.0.
- The judge is pluggable (the metric calls only `judge.claim_supported`), so swapping in a real LLM changes nothing in the metric.
- Committed.

**Hint.** Port Exercise 2's `decompose_claims` + `faithfulness`. For the hallucination test, take a grounded answer and append a claim that's absent from the context (e.g. a fabricated penalty amount); assert the score drops to (supported / total). (Lecture 2 Part 1.1; Exercise 2.)

**Estimated time.** 45 minutes.

---

## Problem 3 — Calibrate the judge and report Cohen's kappa

**Problem statement.** Take 10 (question, answer, context) examples and **label them yourself** (faithful = 1, not = 0). Run an LLM-as-judge (the Exercise 3 stub, or a real `claude-opus-4-8`) on the same 10, sweep the threshold τ, compute **Cohen's kappa** at each, and produce `notes/week-12/calibration.md` with the sweep table and the calibrated `(τ, kappa)`.

**Acceptance criteria.**

- A table of raw agreement and Cohen's kappa for at least five thresholds against your 10 labels.
- The calibrated τ (kappa-maximizing) is identified, with its kappa and band (poor/fair/moderate/substantial/almost-perfect).
- You note at least one threshold where raw agreement is high but kappa is low (the chance-agreement trap), and state why you report kappa, not raw agreement.
- Committed.

**Hint.** Reuse Exercise 3's `cohen_kappa` and threshold sweep. If you use a real judge, run it 2–3 times and note the variance — that variance is the judge's measurement noise. Cross-check your kappa against `sklearn.metrics.cohen_kappa_score`. (Lecture 2 Part 4; Exercise 3.)

**Estimated time.** 50 minutes.

---

## Problem 4 — The Ragas evaluation report memo (headline deliverable / Phase II milestone)

**Problem statement.** This is the syllabus milestone. Run the four Ragas metrics (faithfulness, context_recall, context_precision, answer_relevancy) across **three variants** of your Phase II pipeline — baseline dense, +reranker, +hybrid (weeks 8–10) — on the 40-question gold set, behind your **calibrated** judge from Problem 3. Plot the four metrics × three variants. Write a **one-page** memo at `notes/week-12/ragas-report-memo.md` against this template:

1. **Decision** — one sentence: which variant you ship, and its headline numbers.
2. **The table** — the three variants × four metrics, with the calibrated `(τ, kappa)` and the generator named in the header.
3. **Which metric moved most for which change** — the finding: e.g. "the reranker moved context_precision +0.11 by demoting irrelevant chunks; hybrid moved context_recall +0.07 by catching lexical matches dense missed" — with the *mechanism*, not just the delta.
4. **The trade-off accepted** — what each improvement cost (latency, complexity).
5. **The judge note** — your generator is a *different model* from the judge (no self-preference), the judge is calibrated (state the kappa), and you trust the numbers only at the calibrated threshold.
6. **One per-query trace** — in the promise format: `q12 ("five-year confidentiality") faithful=1.0 grounded in ctx_09 ✓` for the shipped variant, plus one query where a losing variant hallucinated or missed.

**Acceptance criteria.**

- `notes/week-12/ragas-report-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- The four-metrics × three-variants plot (`ragas_report.png`) is committed.
- The generator is held constant across variants and is different from the judge; only the retriever varies.
- The judge is calibrated and its kappa is stated; metrics are reported at the calibrated threshold.
- At least one per-query trace in the promise format.
- Committed.

**Hint.** Hold the generator fixed across variants or you've changed two things at once and the finding is unanswerable (the challenge's trap). Run faithfulness/relevancy with the calibrated judge; if cost is a concern, calibrate and run the sweep with `claude-haiku-4-5`, reserve `claude-opus-4-8` for the final numbers. (Challenge 1; Lecture 2 Parts 2–4.)

**Estimated time.** 1 hour.

---

## Problem 5 — Probe the judge for verbosity bias

**Problem statement.** Take 5 correct, concise answers from your gold set. Make a padded copy of each by appending correct-but-redundant filler (restating the answer, adding a true-but-irrelevant sentence). Run your judge on both versions and check whether the **padded answers score higher** on faithfulness/relevancy. Record the result in `notes/week-12/verbosity-bias.md`.

**Acceptance criteria.**

- A before/after table: 5 answers × (concise score, padded score) for at least one metric.
- A one-sentence conclusion: does your judge reward length, and by how much?
- If a bias is present, a note on how you'd correct it (instruct the judge to ignore length, or normalize).
- Committed.

**Hint.** Keep the *facts* identical between concise and padded — only the length differs — so any score difference is the bias, not a content change. A judge that scores the padded version higher has a verbosity bias you must defend against in the report. (Lecture 2 Part 5.)

**Estimated time.** 45 minutes.

---

## Problem 6 — Text-anchored vs. vision-anchored on one figure question

**Problem statement.** Take one gold question whose answer lives in a *figure*. Build both multimodal architectures from Lecture 1 §3: (A) use Claude vision to *describe* the figure into a text chunk and retrieve it with BGE; (B) embed the page image with CLIP and retrieve it for a VLM to answer. Compare faithfulness of the two answers and record which architecture grounded the figure answer in `notes/week-12/text-vs-vision.md`.

**Acceptance criteria.**

- Both architectures (A: extract+describe; B: embed-page-image) implemented for one figure question.
- The generated answer and its faithfulness (calibrated judge or stub) for each.
- A one-sentence conclusion: which architecture answered the figure question and *why* (was the number in the description, or only in the picture?).
- Committed.

**Hint.** Option A's success hinges on whether your VLM description captured the *number*, not just "a chart"; Option B's hinges on whether CLIP retrieved the right page. Different failure surfaces — the eval tells you which one your corpus suffers from. (Lecture 1 §3; Q11 in the quiz.)

**Estimated time.** 40 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — VLM vs. Claude over a figure | 40 min |
| 2 — One Ragas metric from scratch + tests | 45 min |
| 3 — Calibrate the judge, report Cohen's kappa | 50 min |
| 4 — Ragas evaluation report memo (headline / milestone) | 1 h 0 min |
| 5 — Probe the judge for verbosity bias | 45 min |
| 6 — Text-anchored vs. vision-anchored on a figure | 40 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchrag_eval` [mini-project](./mini-project/README.md) is in the same workspace — Phase III (week 13, LangGraph) imports it to grade agent trajectories, and your Ragas report is the Phase II milestone you defend at the architecture review. Then take the [quiz](./quiz.md) with your notes closed.
