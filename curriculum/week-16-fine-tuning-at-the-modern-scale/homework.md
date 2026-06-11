# Week 16 Homework

Six problems that revisit the week's topics and put fine-tuning judgment into your fingers. The full set should take about **5 hours** (the GPU training in Problem 3 is ~15–30 minutes of that; budget ~$3 of rented compute, or use the CPU-mechanics path to do everything but the real run). Work in your Week 16 Git repository (the same workspace as the exercises and the `crunchtune` mini-project) so every problem produces at least one commit you can point to for the Phase III milestone.

The headline deliverable is **Problem 4 — the one-page fine-tune-or-not decision memo**, which is exactly the artifact the Phase III milestone (end of week 18) requires for the capstone's domain.

Have your **week-2 tokenizer intuition** fresh (the chat template is central) and a rentable GPU available for Problem 3. If you can't rent one, the CPU-mechanics path (a tiny 0.5B model, a few steps) lets you complete the *mechanics* of every problem; note in your write-up that the real run is pending.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Run the ladder on a real decision

**Problem statement.** Take a "should we fine-tune?" decision — one of Exercise 1's scenarios, a capstone-domain candidate, or one from your own work — and write `notes/week-16/decision.md` walking it through the prompt→retrieve→fine-tune ladder: the symptom, what each rung would do, the verdict, and (if fine-tune) the legitimate reason and the objective.

**Acceptance criteria.**

- `notes/week-16/decision.md` exists with the full ladder reasoning.
- The verdict is justified by *which rung* the problem belongs to, not by gut feel.
- If "fine-tune", a named legitimate reason (style / vocabulary / latency-cost) and objective (SFT vs preference) are stated, with the eval contract (a held-out baseline comparison).
- If "don't fine-tune", the cheaper rung (prompt or retrieval) is named with why it suffices.
- Committed.

**Hint.** The most common honest verdict is "fix the prompt" or "use retrieval." If you reach "fine-tune," double-check it's a style/vocabulary/latency problem with a *measured* prompt ceiling, not a knowledge gap (retrieval) or a lazy prompt (rung 1).

**Estimated time.** 40 minutes.

---

## Problem 2 — Build and validate an SFT dataset

**Problem statement.** In your `crunchtune` package, implement `data.prepare()`: validate every target (parses + canonical form), apply the chat template via `apply_chat_template`, de-duplicate, and split train/test with a fixed seed and a no-leakage assertion. Write `tests/test_data.py` proving (a) invalid/non-canonical targets are rejected, and (b) no test example appears in train.

**Acceptance criteria.**

- `data.prepare()` validates, templates, de-dups, and splits with a fixed seed.
- `pytest tests/test_data.py` passes with at least: an invalid target rejected, a non-canonical target rejected, and the no-leakage assertion holding.
- The chat template is applied via the **tokenizer**, not hand-concatenation (demonstrated in the test or a print).
- Committed.

**Hint.** Port Exercise 2's validation and split. For the leakage test, deliberately *plant* a duplicate across the split in a fixture and assert the leakage check catches it — proving the firewall works, not just that it didn't fire by luck.

**Estimated time.** 50 minutes.

---

## Problem 3 — Train a LoRA adapter and read the loss curve

**Problem statement.** Train a QLoRA + LoRA adapter on Qwen2.5-7B (or, CPU path, a 0.5B for the mechanics) on your `train.jsonl`. Capture the loss curve and produce `notes/week-16/training.md` with: the trainable-param count (and its fraction of total), the hyperparameters (rank/alpha/LR/epochs) with one sentence of reasoning each, and the loss-curve shape (learning / memorizing / diverging).

**Acceptance criteria.**

- The training run uses **QLoRA** (4-bit base) + **LoRA**; the adapter is <1% of params (shown).
- The loss curve is captured (the per-step log) and its shape is named with evidence.
- Each hyperparameter (rank, alpha, learning rate, epochs) has a one-sentence justification.
- The adapter is saved to disk (the small adapter file, not the full model).
- Committed.

**Hint.** `FastLanguageModel.get_peft_model(...)` reports trainable params on the first call. For the loss curve, `trainer.state.log_history` has the per-step losses. If your loss diverges (spikes), lower the LR; if it dives to ~0 instantly on a tiny set, you're memorizing — that's expected on the CPU/0.5B mechanics path and a real warning on the 7B run.

**Estimated time.** 50 minutes (plus ~15–30 min GPU training).

---

## Problem 4 — The fine-tune-or-not decision memo (headline deliverable)

**Problem statement.** This is the Phase III milestone artifact. Evaluate your fine-tune against the base model (best prompt) on the held-out `test.jsonl`, then write a **one-page** memo at `notes/week-16/fine-tune-memo.md` against this template:

1. **Decision** — one sentence: do you ship the fine-tune, and the headline number.
2. **The table** — base (best prompt) vs fine-tune: exact-match, valid-DSL, latency.
3. **Why this result, on this task** — the mechanism (e.g. "the prompt couldn't reliably hit the canonical DSL form; demonstrations taught it"), not a general claim.
4. **The cost** — training $ + time, plus the new artifact to version and serve.
5. **The verdict** — worth it / not worth it, against a stated threshold. An honest negative is a valid answer.
6. **One held-out trace** — in the promise format: `q: "contracts after 2024 in Delaware" -> base: invalid / ft: SELECT * FROM contracts WHERE signed_year > 2024 AND state = 'DE'; ✓`.

**Acceptance criteria.**

- `notes/week-16/fine-tune-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- The evaluation is on the **held-out** set (never training data), with the base prompted *as well as you can*.
- The verdict is rule-based (a stated threshold), and the memo is willing to render an honest negative.
- The mechanism is specific to this task, not "fine-tuning helps."
- At least one held-out per-query trace.
- Committed.

**Hint.** The memo is read by a reviewer deciding whether to ship the artifact — make every claim falsifiable. Don't write "the fine-tune is better," write "fine-tune exact-match 0.86 vs base 0.42 on 50 held-out examples — see `eval.py` output." If your delta is small, *say so* and ship the prompt; the negative memo is just as much a pass.

**Estimated time.** 1 hour.

---

## Problem 5 — Break the eval to feel the trap

**Problem statement.** Deliberately fall into both eval traps and document what happens. (a) Evaluate your fine-tune on its *training* data and record the inflated score. (b) Compare your fine-tune to a *lazy one-line prompt* baseline and record the inflated delta. Then fix both and show the honest numbers. Produce `notes/week-16/eval-traps.md`.

**Acceptance criteria.**

- `notes/week-16/eval-traps.md` shows the train-data score (inflated) vs the held-out score (honest), with the gap.
- It shows the weak-prompt baseline (inflated delta) vs the strong-prompt baseline (honest delta), with the gap.
- A one-sentence conclusion on *why* each trap inflates the number and how the fix removes the inflation.
- Committed.

**Hint.** The train-data score will look suspiciously high (memorization). The weak-prompt comparison will make the fine-tune look like a hero. Both are lies. The honest numbers — held-out set, strong-prompt baseline — are the only ones you'd defend. Feeling the gap is the point: it's how you learn to distrust a too-good result.

**Estimated time.** 40 minutes.

---

## Problem 6 — The regression spot-check

**Problem statement.** Run a handful (5–10) of *general* queries — unrelated to the DSL task — through your base model and your fine-tuned model. Did the fine-tune degrade general capability (catastrophic forgetting)? Record the before/after in `notes/week-16/regression.md` with a verdict: did the fine-tune *trade* general ability for the narrow win, and is that trade acceptable?

**Acceptance criteria.**

- `notes/week-16/regression.md` has 5–10 general queries run through both models with the outputs.
- A verdict on whether the fine-tune regressed general capability, and by how much (qualitatively or with a simple judge).
- A statement of whether the trade (if any) is acceptable for the intended use.
- Committed.

**Hint.** Fine-tuning on a narrow task can make the model worse at everything else — sometimes dramatically. If your fine-tuned model can now write perfect DSL but can no longer answer "what's the capital of France?", that's a real cost. It's *fine* if the model only ever does the narrow task — but you have to *notice* the trade to make that call. The held-out eval measured the win; this measures the price.

**Estimated time.** 40 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Run the ladder on a real decision | 40 min |
| 2 — Build + validate an SFT dataset | 50 min |
| 3 — Train a LoRA + read the loss curve | 50 min |
| 4 — Fine-tune-or-not memo (headline) | 1 h 0 min |
| 5 — Break the eval to feel the trap | 40 min |
| 6 — The regression spot-check | 40 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchtune` [mini-project](./mini-project/README.md) is in the same workspace — the Phase III milestone wants the fine-tune-or-not decision document this pipeline produces. Then take the [quiz](./quiz.md) with your notes closed.
