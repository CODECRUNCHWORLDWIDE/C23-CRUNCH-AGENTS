# Week 1 — Quiz

Fourteen questions. Take it with your lecture notes closed. Aim for 11/14 before moving to Week 2. Answer key is at the bottom — don't peek.

---

**Q1.** At the function level, what does a large language model take as input and produce as output?

- A) Text in, text out.
- B) A sequence of token IDs in; a vector of logits (one score per vocabulary entry) for the next token, out.
- C) A prompt in; a fully-formed answer out, in one step.
- D) An embedding in; a classification label out.

---

**Q2.** "The conversation history is in the model's memory." Why is this statement wrong at the function level?

- A) The model only remembers the last token.
- B) The model is stateless — it's a pure function of its input sequence; history exists only because your code re-sends it every turn.
- C) Memory is stored in the KV cache permanently.
- D) The model forgets history after the training cutoff.

---

**Q3.** You see "creativity" / "randomness" in a model's output. Which stage of the five-stage pipeline owns that behavior?

- A) Stage 1, the tokenizer.
- B) Stage 3, the forward pass.
- C) Stage 4, sampling.
- D) Stage 5, the detokenizer.

---

**Q4.** Why can a model "fail" to count the letters in "strawberry" or spell a word backwards?

- A) The model is too small.
- B) Its training cutoff was before that word existed.
- C) It never saw individual letters — the tokenizer presents text as sub-word tokens, not characters.
- D) Sampling temperature was too high.

---

**Q5.** Attention in a decoder-only transformer is `O(n²)` in sequence length. What is the main systems consequence?

- A) Output is always slow regardless of prompt size.
- B) Long context is expensive *beyond* the linear token cost, because doubling the prompt roughly quadruples the attention work.
- C) The model can't generate more than n tokens.
- D) The tokenizer slows down on long inputs.

---

**Q6.** Prefill and decode are the two phases of generation. Which is compute-bound and which is memory-bandwidth-bound?

- A) Prefill is bandwidth-bound; decode is compute-bound.
- B) Both are compute-bound.
- C) Prefill is compute-bound (sets TTFT); decode is memory-bandwidth-bound (sets TPOT).
- D) Both are bandwidth-bound.

---

**Q7.** A 100,000-token prompt "feels slow to start but streams normally once it gets going." Why?

- A) The tokenizer is slow on long input.
- B) Prefill scales with prompt length (slow first token / high TTFT), but decode is per-token and roughly prompt-length-independent (normal TPOT).
- C) The model re-reads the whole prompt for every output token.
- D) The KV cache is disabled for long prompts.

---

**Q8.** What does the KV cache buy you?

- A) It lets late subscribers receive old tokens.
- B) It stores prior tokens' keys/values so each new decode step doesn't reprocess the whole sequence — making the second token much faster than the first.
- C) It caches the final text output for reuse.
- D) It increases the context window size.

---

**Q9.** A model has a training cutoff of early 2025. A user asks about an event from last week (2026). The model gets it wrong. Which stage / property is responsible, and what's the right fix?

- A) Sampling; lower the temperature.
- B) The frozen MLP weights (parametric knowledge ends at the cutoff); fix by injecting the information via context/retrieval, not by re-prompting harder.
- C) The tokenizer; switch tokenizers.
- D) The context window; make it larger.

---

**Q10.** Two axes organize the 2026 model landscape. What are they?

- A) Fast vs slow, and big vs small.
- B) Open-weights vs closed-weights (control vs convenience), and capability vs cost.
- C) English vs multilingual, and text vs vision.
- D) Free vs paid, and old vs new.

---

**Q11.** Which is the *first* fact you should extract from any model card, and why?

- A) The benchmark score, because it tells you if the model is good.
- B) The license / commercial terms, because if you legally can't ship on it, nothing else on the card matters.
- C) The parameter count, because it determines cost.
- D) The release date, because newer is better.

---

**Q12.** A model tops a leaderboard. Why is that not sufficient to pick it for your specific product?

- A) Leaderboards are always fraudulent.
- B) A leaderboard measures average performance on a fixed benchmark distribution; your product is one specific point, and the rank ignores your cost, latency, and license constraints.
- C) Leaderboards only test open models.
- D) The top model is always too expensive.

---

**Q13.** Llama 4 ships under a "community license." Which statement is correct?

- A) It is OSI-approved open source identical to Apache-2.0.
- B) It is source-available (not OSI-open): commercial use is permitted with conditions, including an acceptable-use policy and a separate-license requirement above 700M monthly active users.
- C) It forbids all commercial use.
- D) It requires you to open-source your entire application.

---

**Q14.** For the job "classify 100k support tickets/day into 8 categories, PII-sensitive, sub-second, pennies budget," why is a frontier model like `claude-opus-4-8` usually the *wrong* pick?

- A) It can't do classification.
- B) It's the most *capable*, but the binding constraints here are cost, latency, and PII — and a frontier model is ~10× over budget, slower, and ships PII to a vendor, all to add ~zero accuracy on an easy task a 7B handles.
- C) Frontier models don't support 8 categories.
- D) It has no context window.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Token IDs in, next-token logits out. The model emits a distribution; your sampler picks the token. (Lecture 1 §1.)
2. **B** — The model is a pure, stateless function of its input sequence; conversation persists only because your wrapper re-sends the history. (Lecture 1 §1.)
3. **C** — Sampling (Stage 4) owns determinism/randomness. The forward pass emits the same logits; the sampler decides what to do with them. (Lecture 1 §2.4.)
4. **C** — Sub-word tokenization: the model never sees individual characters, so letter-level tasks are hard. (Lecture 1 §2.1.)
5. **B** — Quadratic attention makes long context expensive beyond the linear token cost; this is why retrieval (curating *which* tokens go in the window) matters. (Lecture 1 §3.)
6. **C** — Prefill is compute-bound and sets TTFT; decode is memory-bandwidth-bound and sets TPOT. (Lecture 1 §4.)
7. **B** — Prefill scales with prompt length (slow start), decode is per-token and roughly length-independent (normal streaming rate). (Lecture 1 §4.)
8. **B** — The KV cache stores prior keys/values so decode doesn't reprocess the whole sequence; that's why the second token is fast. (Lecture 1 §4.)
9. **B** — Parametric knowledge is frozen at the training cutoff in the MLP weights; the fix is injecting the info via context/retrieval, which is what Phase II is for. (Lecture 1 §3, §5.)
10. **B** — Open vs closed (control vs convenience) and capability vs cost. The dots move every quarter; the axes don't. (Lecture 2 §1.)
11. **B** — License first, always: it can veto everything else. (Lecture 2 §3.)
12. **B** — A rank is an average over a benchmark distribution and ignores your point-task, cost, latency, and license; measure on your own data. (Lecture 2 §3.)
13. **B** — Source-available, not OSI-open: commercial use with conditions, an acceptable-use policy, and the 700M-MAU clause. (Lecture 2 §5.)
14. **B** — Capability is the wrong axis here; the binding constraints (cost, latency, PII) pick a local 7B, not the frontier tier. (Lecture 2 §6.)

</details>

---

If you scored under 10, re-read the lecture sections cited in the answers you missed. If you scored 12 or higher, you're ready for the [homework](./homework.md).
