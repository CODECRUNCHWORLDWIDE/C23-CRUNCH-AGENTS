# Week 16 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 17. Answer key is at the bottom — don't peek.

---

**Q1.** What is the correct ordering of options when a model isn't doing what you need?

- A) Fine-tune first, then try a prompt if that fails.
- B) Climb the ladder cheapest-and-most-reversible first: fix the **prompt**, then add **retrieval**, then **fine-tune** only if the first two demonstrably hit a measured ceiling.
- C) Always fine-tune; prompts are unreliable.
- D) Retrieval, then fine-tune, then prompt.

---

**Q2.** Your model is confidently wrong about a product launched after its training cutoff. What's the right fix?

- A) Fine-tune the model on the new product's docs.
- B) **Retrieval** — give the model the new product's docs at inference time. It's a *knowledge* gap; fine-tuning facts into weights is slow, lossy, stale, and unauditable, while retrieval is updatable and citable.
- C) A bigger model.
- D) DPO on preference pairs.

---

**Q3.** Which is NOT a legitimate reason to fine-tune?

- A) Output style/format you can't reliably hit by prompting.
- B) Domain vocabulary the base model handles poorly.
- C) Latency/cost — getting a small model to do a big model's narrow job.
- D) **To make the model generally smarter** — fine-tuning *narrows and specializes* a model, often trading away general capability; it doesn't broaden it.

---

**Q4.** Why is full fine-tuning of a 7B prohibitive on a 24 GB GPU?

- A) The weights are too large to load.
- B) Full fine-tuning needs the weights **plus** the Adam optimizer state (two extra tensors per parameter) plus gradients plus activations — roughly 60–80 GB for a 7B. The optimizer state is the killer.
- C) 7B models can't be fine-tuned at all.
- D) GPUs can't do backpropagation.

---

**Q5.** What does LoRA do?

- A) It retrains every weight in the model at low precision.
- B) It **freezes the base weights** and learns small **low-rank update matrices** (`A`, `B`) whose product is the update — training <1% of the parameters. Rank `r` and `alpha` are the knobs.
- C) It deletes layers to save memory.
- D) It quantizes the model to 4-bit and trains it directly.

---

**Q6.** What does QLoRA add to LoRA, and why does it barely hurt quality?

- A) It trains the base in 8-bit.
- B) It **quantizes the frozen base to 4-bit** (so a 7B base is ~5 GB instead of ~14), then trains the LoRA adapter on top — and quality barely drops because the base is *frozen* (never trained), so its precision loss doesn't matter, while the higher-precision adapter learns around it. This is what fits a 7B in 24 GB.
- C) It uses two GPUs.
- D) It removes the optimizer state.

---

**Q7.** Which post-training objective do you want for a task with a single correct answer (like NL→DSL)?

- A) DPO, because preference methods are always better.
- B) **SFT** (supervised fine-tuning on input→output demonstrations) — you have correct answers, so you train the model to reproduce them. DPO/ORPO/KTO are for *preference* tasks and need preference-pair data SFT doesn't.
- C) RLHF, because that's how frontier models are trained.
- D) ORPO, because it's newest.

---

**Q8.** You hand-format your training data as `"instruction\noutput"` strings. Your fine-tune "trains fine" (loss drops) but produces garbage at inference. What's the most likely bug?

- A) The learning rate was too high.
- B) You didn't apply the model's **chat template** — you trained on a format the model never sees at inference, so it learned the wrong distribution. Always format with `apply_chat_template` using the target model's tokenizer.
- C) The dataset was too small.
- D) LoRA rank was too low.

---

**Q9.** For an SFT dataset, which is true about size?

- A) More data is always better — scrape as much as possible.
- B) **Quality over quantity** — a few hundred clean, correct, format-consistent examples beat thousands of noisy ones, because the model learns the distribution of your demonstrations *including their errors*. Label noise becomes confident wrongness.
- C) Exactly 10,000 examples is optimal.
- D) Size doesn't matter at all.

---

**Q10.** Why must you evaluate a fine-tune on a held-out test set?

- A) It's faster than evaluating on training data.
- B) Evaluating on training data measures **memorization** (which any fine-tune can do) and tells you nothing about whether it learned the *task*. The held-out set — data the model never trained on — is the only honest measure of generalization. Leakage into training inflates the score into a lie.
- C) Held-out sets are required by the SFTTrainer API.
- D) You don't — train-set accuracy is fine.

---

**Q11.** During training, the loss keeps dropping smoothly toward zero. Training loss looks great. What might be wrong, and how would you catch it?

- A) Nothing — a loss near zero is always a win.
- B) It may be **memorizing/overfitting**: training loss → 0 while held-out performance plateaus or declines. You only catch it with a **held-out eval** — training loss alone looks great while the model overfits. Fix: fewer epochs, more data, or lower rank.
- C) The learning rate is too low.
- D) The chat template is wrong.

---

**Q12.** When you compare the fine-tune to the base model, what must the base baseline be?

- A) The base model with a lazy one-line prompt.
- B) The base model with your **best prompt** (few-shot, format instructions, the works) — because that's the alternative you'd actually ship if you *didn't* fine-tune. Beating a strawman prompt proves nothing; you must beat the strong baseline.
- C) A different model entirely.
- D) The fine-tuned model evaluated twice.

---

**Q13.** Your fine-tune scores +0.03 exact-match over the best-prompt baseline, for $3 of compute and a new adapter to version and serve. What's the honest verdict?

- A) WORTH IT — any improvement justifies a fine-tune.
- B) **NOT WORTH IT** — a 0.03 gain doesn't justify the cost plus a new artifact to version and serve; the prompt ceiling was already near the fine-tune. Ship the prompt. A measured negative result is a real, valid engineering deliverable.
- C) Re-run it until the number goes up.
- D) WORTH IT, because you already paid the $3.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — The prompt → retrieve → fine-tune ladder: cheapest/most-reversible first; fine-tune only on a measured ceiling. (Lecture 1 §1.)
2. **B** — Knowledge gap → retrieval, not fine-tuning. Facts in weights are slow, lossy, stale, unauditable. (Lecture 1 §1, §2; Exercise 1 Scenario A.)
3. **D** — The three legitimate reasons are style/format, vocabulary, latency/cost. "Make it smarter" isn't one — fine-tuning narrows and can trade away general ability. (Lecture 1 §2.)
4. **B** — Weights + Adam optimizer state + gradients + activations ≈ 60–80 GB for a 7B. The optimizer state dominates. (Lecture 1 §4.1; Exercise 3 Part A.)
5. **B** — Freeze the base, learn low-rank update matrices; <1% of params trained; rank + alpha are the knobs. (Lecture 1 §4.2.)
6. **B** — 4-bit frozen base fits the 7B in 24 GB; quality holds because the frozen base isn't trained and the adapter learns around the quantization. (Lecture 1 §4.3; Exercise 3 Part A.)
7. **B** — SFT for correctness/demonstration tasks; preference methods need preference data and suit preference tasks. (Lecture 1 §5.1, §5.2.)
8. **B** — The chat-template bug: hand-formatting trains on a distribution the model never sees at inference. Always use `apply_chat_template`. (Lecture 2 §1.2.)
9. **B** — Quality over quantity: clean examples beat noisy volume; label noise becomes confident error. (Lecture 2 §1.3.)
10. **B** — Held-out is the firewall: training-data eval measures memorization, not learning; leakage inflates the score. (Lecture 2 §1.4, §3.1; the challenge trap.)
11. **B** — A loss diving to zero on a small set is a memorization warning; only a held-out eval catches overfitting. (Lecture 2 §2.1.)
12. **B** — The base baseline must be your *best* prompt — the real alternative. Beating a strawman proves nothing. (Lecture 2 §3.1; the challenge's second trap.)
13. **B** — A 0.03 gain for real cost + a new artifact is NOT WORTH IT; ship the prompt. The honest negative is a valid deliverable. (Lecture 2 §3.2.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
