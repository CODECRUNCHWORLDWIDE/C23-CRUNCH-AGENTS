# Exercise 1 — Decide to Fine-Tune (Usually: Don't)

**Goal:** Train the single most important fine-tuning judgment: **climb the prompt→retrieve→fine-tune ladder before reaching for weights.** You will take three real scenarios, reason each through the ladder, and write down the decision *and the legitimate reason* (or the reason not to). The point is that fine-tuning is the *last* rung — most "let's fine-tune" instincts are prompt or retrieval problems wearing a fine-tuning costume.

**Estimated time:** 45 minutes. Guided. No code — this is a reasoning exercise, and reasoning is the part engineers skip.

---

## The framework (from Lecture 1)

For each scenario, run the sequence:

1. **State the symptom** precisely.
2. **Rung 1 — prompt:** would better instructions / few-shot / a structured-output format plausibly fix it? (Cheap, reversible, seconds to test.)
3. **Rung 2 — retrieve:** is this a *knowledge* gap that giving the model the right context at inference would close? (Updatable, auditable, no retraining.)
4. **Rung 3 — fine-tune:** only if 1 and 2 hit a measured ceiling. If so, name the **legitimate reason** — (a) output style/format, (b) domain vocabulary, (c) latency/cost via a small specialized model — and the **objective** (SFT vs preference).

Write the verdict for each in `notes/week-16/decisions.md`.

---

## Scenario A — The support bot that doesn't know the new product

**The situation.** Your customer-support assistant answers questions about your products well — except it's confidently wrong about the product you launched last month, because that product didn't exist at the model's training cutoff. Someone proposes fine-tuning the model on the new product's docs.

**Your task.** Run the ladder. Which rung is this? Write the verdict and the reasoning.

> **Checkpoint (don't read until you've decided):** This is a **knowledge gap** — the model doesn't *know* a fact. The answer is **Rung 2, retrieval**, not fine-tuning. Fine-tuning facts into weights is slow, lossy, and stale the moment the product changes again; retrieval is updatable (re-index the new docs) and auditable (cite the source). Fine-tuning here would be the classic mistake: using weights to solve a retrieval problem. Verdict: **do not fine-tune; add the new product docs to the RAG index.**

---

## Scenario B — The assistant whose tone is "off-brand" 1 in 5 times

**The situation.** Your assistant mostly writes in your company's terse, no-emoji, declarative house voice — but about 20% of responses drift into a chatty, hedge-heavy, emoji-sprinkled style you've explicitly told it (in the system prompt) to avoid. You've iterated the system prompt five times; it's better than it was, but the 20% drift persists. You have 600 examples of on-brand responses.

**Your task.** Run the ladder. Which rung? Name the legitimate reason and the objective.

> **Checkpoint:** You climbed Rung 1 (five prompt iterations) and hit a **measured ceiling** — prompting got you most of the way but not to consistency. This is **legitimate reason #1: output style/format the model can't reliably hit by prompt.** Style is learnable from demonstrations in a way a prompt can't fully pin down. Verdict: **fine-tune is justified**, **SFT** on the 600 on-brand demonstrations (it's a style-demonstration task, not a preference-pair task). The *contract*: a held-out set of inputs, judged for on-brand-ness (an LLM-as-judge, calibrated), base vs fine-tune. If the fine-tune doesn't beat the prompt ceiling on held-out data, you don't ship it.

---

## Scenario C — The frontier model that's too expensive at volume

**The situation.** A frontier model (`claude-opus-4-8`) does your narrow classification task — routing support tickets into 12 categories — beautifully, but at your volume (millions/month) it's too expensive and a touch too slow. A prompted local 7B gets it ~75% right, which isn't good enough. You can generate thousands of correctly-labeled examples (you have the frontier model's outputs and human review).

**Your task.** Run the ladder. Which rung, which reason, which objective — and is there a subtlety about the data?

> **Checkpoint:** Rung 1 (prompt the 7B) hit a ceiling at 75%; retrieval doesn't help a *classification* task (no knowledge gap). This is **legitimate reason #3: latency/cost — get a small model to do a big model's narrow job** (distillation). Verdict: **fine-tune the 7B** via **SFT** on the labeled examples, then serve the cheap fast 7B. The subtlety: **quality over quantity still applies** — clean, correctly-labeled, format-consistent examples beat a noisy dump of frontier outputs; label noise in training becomes confident misclassification at inference. And the contract is the same: held-out accuracy, fine-tuned 7B vs prompted 7B *and* vs the frontier baseline you're trying to replace — if the fine-tuned 7B doesn't get close enough to the frontier model's accuracy to justify the swap, you keep paying for the frontier model.

---

## Step — Write the decisions

In `notes/week-16/decisions.md`, for each scenario write:

| Scenario | Symptom | Rung 1 (prompt)? | Rung 2 (retrieve)? | Verdict | If fine-tune: reason + objective |
|---|---|---|---|---|---|
| A | doesn't know new product | no (it's a fact) | **YES — retrieval** | don't fine-tune | — |
| B | tone drifts 20% | tried, ceiling | no (not knowledge) | **fine-tune** | reason #1 style; SFT |
| C | frontier too costly | 7B ceiling 75% | no (classification) | **fine-tune** | reason #3 latency/cost; SFT |

Then add one paragraph: **the common thread.** (It's that fine-tuning is for *style/vocabulary/cost* problems with a *measured prompt ceiling*, never for *knowledge* problems — those are retrieval — and never as a *first* move.)

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] `notes/week-16/decisions.md` has a verdict for all three scenarios with the ladder reasoning for each.
- [ ] Scenario A is correctly resolved as **retrieval, not fine-tuning** (a knowledge gap).
- [ ] Scenarios B and C are resolved as **fine-tune**, each with a named legitimate reason (#1 style, #3 latency/cost) and the objective (**SFT** for both).
- [ ] Each fine-tune decision names its **eval contract** — a held-out comparison against the relevant baseline.
- [ ] The closing paragraph states the common thread: fine-tune for style/vocab/cost with a measured prompt ceiling, never for knowledge, never first.
- [ ] You can state, in one sentence, *why* fine-tuning facts into a model is worse than retrieving them (slow, lossy, stale, unauditable; retrieval is updatable and citable).

---

## Stretch

- Add a fourth scenario from your *own* work or a project you know: a place where you (or a teammate) reached for fine-tuning. Run the ladder honestly. Was it actually a prompt or retrieval problem?
- For Scenario B, sketch the held-out eval: how would you *measure* "on-brand-ness" without a human reading every output? (Hint: a calibrated LLM-as-judge from week 12 — score against a few human labels first.)
- For Scenario C, work out the break-even: at what fine-tuned-7B accuracy and what volume does the swap from the frontier model actually save money, given the training cost and the serving cost? This is the week-21 cost-engineering muscle, previewed.

---

When this feels comfortable, move to [Exercise 2 — Build an SFT dataset](exercise-02-build-an-sft-dataset.py).
