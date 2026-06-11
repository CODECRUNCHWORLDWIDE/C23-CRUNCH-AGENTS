# Week 2 Homework

Six problems that revisit the week's topics and force the "tokens are money, sampling is a distribution transform, structure is a constraint" thinking into your fingers. The full set should take about **5 hours**. Work in your Week 2 Git repository (the same workspace as the exercises and the `toklab` mini-project) so every problem produces at least one commit you can point to at the Phase I capstone milestone.

The headline deliverable is **Problem 4 — the token-budget cost report**, the per-request accounting artifact a budget owner actually reads. Treat it as a report, not a journal entry.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

Have your environment ready in every terminal: venv active with `transformers`, `numpy`, `outlines`, `jsonschema`, and `anthropic`; Ollama running with `qwen2.5:7b` pulled; and `ANTHROPIC_API_KEY` exported (or accept the local-only fallback and say so in your writeup).

---

## Problem 1 — BPE by hand, then in code

**Problem statement.** Given this tiny merge list (priority = rank, lower merges first): `("l","o")→0`, `("lo","w")→1`, `("e","r")→2`, `("low","er")→3`, encode the string `"lower lower"` **by hand** (each space-separated chunk independently), writing each merge step. Then implement the ~20-line greedy BPE encoder from Lecture 1 §2.2 and confirm it produces the same tokenization. Put both in `notes/week-02/bpe-by-hand.md` (the trace) and `bpe_encode.py` (the code).

**Acceptance criteria.**

- A written step-by-step trace for at least one `"lower"` chunk showing each merge applied in priority order.
- A runnable `bpe_encode.py` whose output matches your hand trace.
- One sentence on why the *order* of the merge list (not just its contents) is the algorithm.
- Committed.

**Hint.** Encode each whitespace-separated chunk independently. At each step, scan all adjacent pairs, pick the one with the lowest rank present in the list, merge it, repeat until no adjacent pair is in the list. `"lower"` should collapse to a single `"lower"` token via the chain `lo → low → lower → ... `; trace it carefully.

**Estimated time.** 35 minutes.

---

## Problem 2 — Three tokenizers, three counts, one ratio

**Problem statement.** Reusing `exercise-01`, take the five probe strings (or your own English / code / CJK / emoji / whitespace set) and produce the counts table across the three open tokenizers, with the max/min ratio per probe. In `notes/week-02/tokenizer-counts.md`, record the table and answer: which probe had the largest disagreement, and by what factor would your cost estimate be wrong if you used the cheapest-counting tokenizer to budget the most-expensive-counting one's job?

**Acceptance criteria.**

- A 5×3 counts table with a per-probe max/min ratio column.
- The Chinese (or other non-Latin) probe shows the largest ratio.
- A two-sentence explanation tying the disagreement to different corpora / byte fallback, and a stated cost-error factor with direction.
- Committed.

**Hint.** Use `len(tok.encode(text, add_special_tokens=False))` so you compare the content, not each tokenizer's special-token wrapper. Never reach for `tiktoken` to count a non-OpenAI model — that mistake *is* the subject of this problem.

**Estimated time.** 40 minutes.

---

## Problem 3 — The temperature sweep, on a real distribution

**Problem statement.** Pull the next-token logits from a local model for one position (via the `transformers` API: run the model on a short prompt, take `logits[0, -1, :]`), then run your `exercise-02` sampler at temperature `0.1, 0.7, 1.0, 1.5`. For each temperature, draw 2,000 samples and report the entropy of the empirical chosen-token distribution (or the top-5 tokens and their frequencies). Put the table in `notes/week-02/temperature-sweep.md` and state, in two sentences, what happens to the distribution as `T` rises and why.

**Acceptance criteria.**

- A table of temperature → entropy (or top-5 frequencies) over ≥4 temperatures, from real model logits.
- Entropy (spread) clearly **increases** with temperature.
- A two-sentence interpretation correctly attributing the spread to temperature flattening the distribution before softmax.
- Committed.

**Hint.** Load a small model with `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")`, tokenize a prompt, forward it, and take the last position's logits as a NumPy array (`.detach().numpy()`). Then feed them to your sampler. If you can't run the model, the toy logits from `exercise-02` show the same effect — but real logits make it land.

**Estimated time.** 50 minutes.

---

## Problem 4 — The token-budget cost report (headline deliverable)

**Problem statement.** This is the syllabus deliverable. Pick one real task (summarize an incident report / classify a ticket / extract fields from a document). Build a small instrument that, for **one representative call**, records `tokens_in` (broken down: system prompt vs user content vs chat-template overhead), `tokens_out`, and `cost`, using **real token counts** (`count_tokens` for the input, `msg.usage` for actuals — never an estimate). Then project the **monthly cost** at a stated volume (e.g. 20,000 calls/day × 30), and compute the same projection for two *other* model tiers from the price table. Write `notes/week-02/cost-report.md` with the per-call breakdown, the monthly projection per tier, and a one-sentence recommendation of the cheapest tier that you'd trust for the task.

**Acceptance criteria.**

- `notes/week-02/cost-report.md` exists with a per-call token breakdown (system / content / overhead / output) from **real counts**, not estimates.
- A monthly-cost projection at a stated volume, computed for **three** model tiers using the per-MTok price table and the `in + out` formula.
- The report notes that **output tokens are priced 3–5× input** and reflects that in the recommendation.
- A one-line recommendation naming the cheapest tier you'd trust, with the deciding number.
- Committed.

**Hint.** Cost per call = `tokens_in * price_in/1e6 + tokens_out * price_out/1e6`. The breakdown matters: a 1,500-token system prompt re-sent every call is often the hidden cost — flag it and note that prompt caching would discount it. Reuse your `toklab` token-accounting instrument; that's exactly what it's for.

**Estimated time.** 1 hour 15 minutes.

---

## Problem 5 — Make broken JSON impossible

**Problem statement.** Take a tiny extraction task (e.g. "extract `{name, age, city}` from a sentence"). First, run it **unconstrained** on a small local model with only a prompt asking for JSON, across 10 varied inputs, and record how many produced valid, schema-matching JSON. Then run the **same** task **constrained** with `outlines` (`generate.json`) and assert with `jsonschema` that all 10 are valid. Put the before/after counts in `notes/week-02/constrained-json.md` and explain the difference in one sentence.

**Acceptance criteria.**

- An unconstrained validity count (X/10) and a constrained validity count (10/10) for the same task and inputs.
- The constrained run is asserted to be **100%** (the assertion is in your code, not just claimed in prose).
- A one-sentence explanation: the guarantee comes from masking the sampler to the grammar, not from a better prompt or a bigger model.
- Committed.

**Hint.** Reuse `exercise-03`'s structure. Use a deliberately small model (`Qwen/Qwen2.5-0.5B-Instruct`) so the unconstrained run *visibly* fails — that contrast is the lesson. Constrain *structure*, not content: enforce the shape, let the model fill the values.

**Estimated time.** 45 minutes.

---

## Problem 6 — The sampling-knob diagnosis drill

**Problem statement.** Write `notes/week-02/sampling-diagnoses.md` with **five** plausible generation symptoms and, for each: which knob (temperature / top-k / top-p / min-p / repetition penalty) or sampling property is responsible and the concrete fix. At least two must be about *structure or determinism* (e.g. "I need the same output every run" → the draw is the only randomness; set `T=0`/greedy or seed the RNG; "I keep getting broken JSON" → constrain the sampler, don't lower temperature). This drills localizing output behavior to the sampler.

**Acceptance criteria.**

- Five symptom → knob/property → fix entries.
- At least **two** concern determinism or structured output, with the correct fix (seed/greedy for determinism; constraint — not a knob — for valid JSON).
- Each fix is concrete and names the specific knob or technique, not "use a better model."
- Committed.

**Hint.** Good symptoms: "output is repetitive/looping" (raise repetition penalty, or temperature too low) ; "output is incoherent/random" (temperature or top-p too high — tighten truncation) ; "same prompt gives different answers" (the draw; greedy or seed it) ; "JSON breaks sometimes" (constrain the sampler — a structure problem, not a temperature problem) ; "it ignores low-probability but correct tokens" (truncation too aggressive — loosen top-p/min-p). The determinism and JSON ones are the load-bearing ones; they're the week's thesis.

**Estimated time.** 30 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — BPE by hand, then in code | 35 min |
| 2 — Three tokenizers, three counts | 40 min |
| 3 — Temperature sweep on real logits | 50 min |
| 4 — Token-budget cost report (headline) | 1 h 15 min |
| 5 — Make broken JSON impossible | 45 min |
| 6 — Sampling-knob diagnosis drill | 30 min |
| **Total** | **~4 h 55 min** |

When you've finished all six, push your repo and make sure the `toklab` [mini-project](./mini-project/README.md) is in the same workspace — Week 3 (prompt engineering as engineering) builds on the token-accounting instrument you wrote here. Then take the [quiz](./quiz.md) with your notes closed.
