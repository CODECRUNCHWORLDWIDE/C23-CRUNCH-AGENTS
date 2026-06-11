# Exercise 1 — Read Three Model Cards

**Goal:** Read three real model cards — an open-weights flagship (Llama 4), an open-weights challenger (Qwen 3), and a closed frontier model (the Anthropic Claude 4 class) — and extract the six load-bearing facts from each into one comparison table. Then write a one-line "can we ship a commercial product on this?" verdict per model, backed by the actual license clause. You will train the single most important card-reading habit: going straight to the license, the context window, and the cutoff before you ever look at a benchmark.

**Estimated time:** 45 minutes. Guided.

---

## Setup

You need only a browser and a text editor. Open these three sources (also in `resources.md`):

- **Llama 4 model card:** <https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md> and its `LICENSE`.
- **Qwen 3 card/blog:** <https://qwenlm.github.io/blog/qwen3/> — and the license on the specific Hugging Face checkpoint you'd use (linked from the blog).
- **Claude 4 class (closed):** <https://docs.claude.com/en/docs/about-claude/models/overview> for the model facts, plus Anthropic's usage/data terms for the "license analog."

Create a file `notes/week-01/card-comparison.md` in your week-1 repo for your answers.

---

## Step 1 — Find the license first, every time

For each model, **before reading anything else**, locate and record the license (or, for the closed model, the commercial-use terms). Write the exact license name. This is deliberate: you are training yourself to check the constraint that can veto everything else, first.

You should end with something like:

- Llama 4 → "Llama 4 Community License" (source-available, not OSI-open).
- Qwen 3 (your chosen checkpoint) → e.g. "Apache-2.0" (verify on the actual checkpoint — it varies).
- Claude 4 class → no weight license (closed); governed by Anthropic's commercial terms of service.

---

## Step 2 — Extract the six facts into a table

For each model, fill this table in `card-comparison.md`. Use real values from the cards, not memory.

| Fact | Llama 4 | Qwen 3 (your checkpoint) | Claude 4 class |
|------|---------|--------------------------|----------------|
| **License / commercial terms** | | | |
| **Context window (tokens)** | | | |
| **Training cutoff date** | | | |
| **Intended use (1 phrase)** | | | |
| **Out-of-scope / prohibited use (1 phrase)** | | | |
| **Modalities + sizes available** | | | |

Notes while you fill it:

- For **context window**, record the number, not "large." For the closed model, the overview page gives it (e.g., 1M tokens for the Opus tier).
- For **cutoff**, if a card gives a month, record the month. If it's vague, note that — vagueness is itself information.
- For **out-of-scope/prohibited**, the open cards have explicit sections; the closed model's prohibited uses live in the usage policy. Find them.

---

## Step 3 — The "can we ship?" verdict

For each model, write **one sentence** answering: *"Can a commercial startup ship a paid product on this, and what's the catch?"* Back it with the specific clause.

Examples of the shape we want (yours will be your own words, from the real text):

- **Llama 4:** "Yes — the community license permits commercial use, with the catch that a separate Meta license is required above 700M monthly active users, and the acceptable-use policy applies."
- **Qwen 3 (Apache checkpoint):** "Yes, cleanly — Apache-2.0 permits commercial use, modification, and redistribution with only an attribution/notice obligation."
- **Claude 4 class:** "Yes — outputs are usable commercially under Anthropic's terms; the catch is vendor dependency and per-token cost, and you must confirm the data-handling tier matches your data-residency needs."

The point is the *catch*. A verdict with no catch named means you didn't read the license.

---

## Step 4 — The benchmark trap (write two sentences)

Find one place in any of the three cards (or the linked benchmark tables) where a headline benchmark number could mislead you. Write two sentences: what the number claims, and why it might not predict performance on a *specific* task (your own customer-support classification, say). This cements Lecture 2 §3.

Example shape: "Qwen 3 reports a high score on a broad reasoning benchmark, but that's an average over a benchmark distribution that doesn't include my eight-way ticket classification; I'd have to measure on my own held-out tickets before trusting it for that job."

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] `notes/week-01/card-comparison.md` contains the filled six-fact table for all three models, with real values from the cards (not "large," not "recent").
- [ ] Each model has a one-sentence "can we ship?" verdict that **names the catch** (the MAU clause, the attribution obligation, the vendor dependency, etc.).
- [ ] You checked the **license first** for each — and you can say which of the three buckets (true-open / source-available / closed-vendor-terms) each falls into.
- [ ] The two-sentence benchmark-trap note is present and specific to a concrete task.
- [ ] Committed.

---

## Stretch

- Open the **actual Llama 4 LICENSE file** and find the literal text of the 700M-MAU clause. Quote it in your notes. Knowing where this clause physically lives is the kind of thing a founding engineer is asked in an interview.
- Pick a **Qwen 3 checkpoint that is *not* Apache-2.0** (they exist) and contrast its terms with the Apache one. Write one line on how the same "family" can carry two very different shipping stories — the trap of saying "Qwen is open."
- For the closed model, find Anthropic's statement on **whether API inputs are used to train models** on the standard commercial tier, and record it. This is the closed-weights analog of the "derivative works" clause.

---

When this feels comfortable, move to [Exercise 2 — The uniform client](exercise-02-uniform-client.py).
