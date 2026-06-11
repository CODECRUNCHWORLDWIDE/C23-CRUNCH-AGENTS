# Week 2 — Quiz

Fourteen questions. Take it with your lecture notes closed. Aim for 11/14 before moving to Week 3. Answer key is at the bottom — don't peek.

---

**Q1.** What is a sub-word tokenizer, fundamentally?

- A) A word splitter that breaks text on spaces.
- B) A character splitter that emits one token per character.
- C) A learned compression scheme: frequent strings get short codes (single tokens), rare strings decompose into several sub-word pieces, and nothing is out-of-vocabulary (byte fallback).
- D) A spell-checker that normalizes text before the model sees it.

---

**Q2.** In BPE *encoding* (applying a trained tokenizer), how is a string turned into tokens?

- A) Look the whole word up in a dictionary; if missing, drop it.
- B) Start from bytes/characters and repeatedly apply the highest-priority adjacent merge from the merge list until no adjacent pair is mergeable.
- C) Split on whitespace and assign each word a random ID.
- D) Run the text through the model once to predict its own tokens.

---

**Q3.** Why does the same English-with-code-and-Chinese paragraph produce *different* token counts on the Llama, Qwen, and `tiktoken` tokenizers?

- A) The tokenizers are buggy and disagree by accident.
- B) They were trained on different corpora, so their merge rules and vocabularies differ — and text the corpus saw rarely (code, non-Latin scripts) gets little merge help and falls toward bytes.
- C) Only the vocabulary size matters; counts scale linearly with it.
- D) The counts are actually identical; any difference is rounding.

---

**Q4.** You need to estimate the token cost of a prompt for `claude-opus-4-8`. Which is correct?

- A) Use `tiktoken` — it's the standard tokenizer for all models.
- B) Count the words and multiply by 1.3.
- C) Use `client.messages.count_tokens(model="claude-opus-4-8", messages=[...])` — the model's own tokenizer, via the SDK.
- D) Count characters and divide by 4.

---

**Q5.** Which is the more expensive side of a request, per token, and what follows?

- A) Input is more expensive; long documents are the main cost.
- B) Output is typically 3–5× the price of input, so a job that *writes* a lot (verbose generation) costs more than one that *reads* a lot (extraction) — favor compact outputs to cut cost.
- C) They cost the same; only the total token count matters.
- D) Neither is billed; you pay per request, not per token.

---

**Q6.** The context window is best thought of as:

- A) A free bucket you fill until it's full.
- B) A budget you spend, with three costs: linear token cost, super-linear (`O(n²)`) attention cost paid as TTFT, and a position-dependent quality cost.
- C) A cache the vendor manages for you at no charge.
- D) A hard limit that has no effect on cost as long as you stay under it.

---

**Q7.** The "lost in the middle" effect says:

- A) Tokens in the middle of the vocabulary are sampled less often.
- B) Models attend unevenly across long context: a relevant fact placed in the *middle* is used less reliably than the same fact at the beginning or end (a U-shaped accuracy curve).
- C) The middle layers of the transformer are skipped for long inputs.
- D) The KV cache drops middle tokens to save memory.

---

**Q8.** Apply softmax to logits `z`. Why subtract `max(z)` first?

- A) It changes the result to favor the top token.
- B) It's a numerical-stability trick — mathematically a no-op, but it keeps the exponentials in a safe range so you don't overflow.
- C) It converts logits to log-probabilities.
- D) It's required to make the probabilities sum to 1; without it they don't.

---

**Q9.** What does temperature do, precisely?

- A) It adds creativity by giving the model new ideas.
- B) It scales the logits before softmax: `T<1` sharpens the distribution toward the top token, `T>1` flattens it, `T→0` approaches greedy. It does not invent options — it reshapes a fixed distribution.
- C) It increases the context window so the model can consider more.
- D) It penalizes repeated tokens.

---

**Q10.** Top-p (nucleus) and top-k differ how?

- A) They're identical; top-p is just a rename of top-k.
- B) Top-k keeps a *fixed count* of the highest-logit tokens; top-p keeps the *smallest set whose cumulative probability ≥ p*, so its cutoff adapts to how confident the model is.
- C) Top-p keeps a fixed count; top-k keeps a cumulative-mass set.
- D) Top-p operates on logits; top-k operates on raw text.

---

**Q11.** Min-p keeps which tokens?

- A) The `min_p` lowest-probability tokens.
- B) Tokens whose probability is at least `min_p × (probability of the top token)` — so the threshold scales with the model's confidence at that position.
- C) Exactly `min_p` tokens, as a count.
- D) Tokens whose logit is above `min_p`.

---

**Q12.** Your JSON generator emits valid output "99% of the time" and you retry on the 1% that breaks. What is the correct diagnosis?

- A) You're done — 99% with a retry is the best achievable.
- B) Retry-on-broken-JSON is the symptom of asking the model nicely instead of constraining the sampler; constrain decoding to the schema so invalid output is structurally impossible.
- C) Raise the temperature so the model is more careful.
- D) Switch to a bigger model; capability is the only fix.

---

**Q13.** How does grammar-constrained decoding (e.g. `outlines`) guarantee schema-valid output?

- A) It validates the output after generation and retries until it passes.
- B) At each decode step it masks every token that would violate the grammar to `-inf`, so the sampler can only choose tokens that keep the output on a valid path — invalid output is unreachable.
- C) It fine-tunes the model on the schema first.
- D) It asks the model to double-check its own JSON.

---

**Q14.** Why does almost nobody use beam search for open-ended LLM generation in 2026?

- A) It's too simple to be useful.
- B) It optimizes for the *most probable sequence*, which is the right target for closed-ended tasks (translation) but produces bland, repetitive text for open-ended generation — and it's more expensive than sampling, which is simply better here.
- C) It can't generate more than `b` tokens.
- D) It requires a GPU that no one has.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **C** — Sub-word tokenization is a learned compression scheme; frequent strings get single tokens, rare ones decompose, byte fallback means nothing is out-of-vocabulary. (Lecture 1 §1.)
2. **B** — Encoding is greedy application of the highest-priority adjacent merge until none apply, starting from bytes/characters. (Lecture 1 §2.2.)
3. **B** — Different training corpora → different merges/vocab → different compression; code and non-Latin scripts diverge most because they got little merge help and fall toward bytes. (Lecture 1 §2.3, §3.2.)
4. **C** — Count with the model's own tokenizer via `count_tokens`; never `tiktoken` for a non-OpenAI model, never characters/words. (Lecture 1 §4.1.)
5. **B** — Output is the expensive side (3–5× input price), so verbose generation costs more than heavy reading; favor compact outputs. (Lecture 1 §4.3.)
6. **B** — The window is a budget with three costs: linear tokens, super-linear attention (paid as TTFT), and position-dependent quality. (Lecture 1 §5.)
7. **B** — Lost-in-the-middle: a U-shaped accuracy curve where middle-position facts are used less reliably than edge-position ones. (Lecture 1 §5.3.)
8. **B** — Subtracting the max is a numerical-stability trick; it's mathematically a no-op but prevents overflow. (Lecture 2 §1.)
9. **B** — Temperature scales logits before softmax: `<1` sharpens, `>1` flattens, `→0` greedy. It reshapes a fixed distribution; it is not "creativity." (Lecture 2 §2.1.)
10. **B** — Top-k is a fixed count of highest-logit tokens; top-p keeps the smallest set with cumulative prob ≥ p, an adaptive cutoff. (Lecture 2 §2.2–2.3.)
11. **B** — Min-p keeps tokens with prob ≥ `min_p × p_max`, scaling with the model's confidence at that position. (Lecture 2 §2.4.)
12. **B** — Retry-on-broken-JSON is the symptom of asking nicely; constrain the sampler so invalid output is impossible. (Lecture 2 §4, week README promise.)
13. **B** — It masks grammar-violating tokens to `-inf` each decode step, making invalid output unreachable by construction. (Lecture 2 §4.2.)
14. **B** — Beam search targets the most probable sequence — right for translation, bland for open text — and costs more than sampling, which wins for open-ended generation. (Lecture 2 §5.)

</details>

---

If you scored under 10, re-read the lecture sections cited in the answers you missed. If you scored 12 or higher, you're ready for the [homework](./homework.md).
