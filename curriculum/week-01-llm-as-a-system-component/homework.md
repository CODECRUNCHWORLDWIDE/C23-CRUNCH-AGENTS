# Week 1 Homework

Six problems that revisit the week's topics and force the "model as a component" thinking into your fingers. The full set should take about **5 hours**. Work in your Week 1 Git repository (the same workspace as the exercises and the `llmpick` mini-project) so every problem produces at least one commit you can point to at the Phase I capstone milestone.

The headline deliverable is **Problem 4 — the model-selection memo**, the same one called out in the challenge. Treat it as the artifact a reviewer reads, not a journal entry.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

Have your environment ready in every terminal: venv active with `anthropic` and `httpx`, Ollama running with `qwen2.5:7b` pulled, and `ANTHROPIC_API_KEY` exported (or accept the local-only fallback and say so in your writeup).

---

## Problem 1 — The five-stage trace

**Problem statement.** Take one prompt — "Write a haiku about a KV cache." — and write a markdown trace in `notes/week-01/five-stage-trace.md` that walks it through all five stages of the interface (tokenizer → context window → forward pass → sampling → detokenizer). For each stage, write 1–2 sentences: what happens to the prompt at that stage, and one concrete observable the stage owns (a token count, the window ceiling, TTFT, the temperature/determinism, the output text). Use real numbers where you can get them: run `client.messages.count_tokens` (or Ollama's `prompt_eval_count`) for the actual tokenizer output.

**Acceptance criteria.**

- `notes/week-01/five-stage-trace.md` exists with one section per stage (five sections).
- At least one stage cites a **real measured number** (e.g., the actual token count of the prompt from a real tokenizer).
- Each stage names the **observable it owns** (cost, ceiling, latency-phase, determinism, or text).
- Committed.

**Hint.** The tokenizer section is the easiest place to put a real number. The forward-pass section is where prefill/decode lives — name which sets TTFT. The sampling section is where you note determinism is *your* choice, not the model's.

**Estimated time.** 35 minutes.

---

## Problem 2 — Same prompt, three tokenizers, three counts

**Problem statement.** Take one fixed paragraph of ~120 words (English; include a couple of code tokens and a non-English word to make it interesting). Count its tokens three ways: (a) the Anthropic tokenizer via `count_tokens`, (b) the Ollama model's tokenizer via `prompt_eval_count`, and (c) a local Hugging Face tokenizer you load with `AutoTokenizer.from_pretrained(...)` for one open model. Record all three counts in `notes/week-01/tokenizer-counts.md` and explain, in two sentences, why estimating one model's cost with another's tokenizer is a real error.

**Acceptance criteria.**

- Three token counts for the **same** paragraph, from three different tokenizers, recorded with the model/tokenizer name next to each.
- The three counts **differ** (they will — that's the lesson).
- A two-sentence explanation of why cross-tokenizer estimation is wrong, tied to the cost consequence.
- Committed.

**Hint.** `pip install transformers tokenizers`. To get the Ollama count without generating, send a 1-token generation and read `prompt_eval_count`, or use `/api/embeddings` if your model supports it. Never use `tiktoken` for a non-OpenAI model — that's the exact mistake this problem is about.

**Estimated time.** 45 minutes.

---

## Problem 3 — TTFT and TPOT under load

**Problem statement.** Using your `exercise-03` measurement (or a fresh script), measure TTFT and TPOT for `qwen2.5:7b` under three prompt sizes: ~10 tokens, ~500 tokens, ~3000 tokens. Build a small table in `notes/week-01/ttft-tpot.md` with prompt-tokens, TTFT, and TPOT for each. Then write two sentences: which number grew with prompt size and which stayed flat, and what that tells you about prefill vs decode.

**Acceptance criteria.**

- `notes/week-01/ttft-tpot.md` has a 3-row table (small / medium / large prompt) with measured TTFT and TPOT.
- TTFT clearly **grows** with prompt size; TPOT stays **roughly flat** across the three.
- The two-sentence interpretation correctly attributes the growth to prefill and the flatness to decode.
- Committed.

**Hint.** Run each size ≥3 times and take the median — a single run is noisy, especially on a laptop doing other things. If your TPOT looks noisy, close other apps; thermal throttling on a laptop can muddy the pattern. The *shape* (TTFT up, TPOT flat) is what matters.

**Estimated time.** 45 minutes.

---

## Problem 4 — The model-selection memo (headline deliverable)

**Problem statement.** This is the syllabus deliverable. Pick **one** of the three jobs from the challenge (ticket classifier / document summarizer / hard reasoning) and write a one-page memo at `notes/week-01/model-selection-memo.md` that selects a model and defends it. Use the five-part structure: (1) the binding constraint, (2) candidates considered and the paper cut, (3) the measurement (measured p50 latency from ≥5 runs and measured cost/call from real token counts, plus a 1–3 sentence quality note on 3 self-authored test inputs), (4) the decision (pick + runner-up, each with the deciding number), (5) the license/data note.

**Acceptance criteria.**

- `notes/week-01/model-selection-memo.md` exists, fits on roughly one page (350–550 words), and hits all five parts.
- The **binding constraint** is named first and the decision flows from it.
- Latency is a **median of ≥5 measured runs**; cost is from **real token counts** — not estimates, not "it felt fast."
- The pick and runner-up each cite the **specific number** that decided it.
- The license/data note is present and relevant to the job (PII for the classifier; faithfulness for the summarizer; etc.).
- Committed.

**Hint.** Reuse your `llmpick` tool or `exercise-02` client to gather the numbers — that's exactly what they're for. The strongest memos *don't* pick the most capable model; they pick the cheapest/fastest model that clears the bar, and say so. If your pick is the frontier model and the job is the classifier, re-read the challenge's "trap" section.

**Estimated time.** 1 hour 15 minutes.

---

## Problem 5 — Read one license end to end

**Problem statement.** Open the **full LICENSE file** of one open-weights model you'd actually consider (Llama 4 community license, or a specific Qwen/Gemma checkpoint's license). Read it with the Ctrl-F checklist from Lecture 2 §5. Write `notes/week-01/license-readthrough.md` answering each checklist question (commercial? MAU/revenue threshold? derivative works? distribution obligations? prohibited uses? trademark/naming?) with the **literal clause or section number** that answers it. End with a one-sentence "can we ship?" verdict and which of the three buckets (true-open / source-available / research-only) it falls in.

**Acceptance criteria.**

- `notes/week-01/license-readthrough.md` answers all six checklist questions, each citing the clause/section that answers it (quote or section number, not paraphrase from memory).
- A one-sentence "can we ship?" verdict and the bucket classification.
- Committed.

**Hint.** For Llama 4, the MAU clause is in the body of the community license — find the literal "700 million" text. The acceptable-use policy is usually a separate linked document; note that it's incorporated by reference (so it binds you even though it's a separate file). For an Apache-2.0 model, the readthrough is short and that's the point: a clean license is a feature.

**Estimated time.** 40 minutes.

---

## Problem 6 — The "wrapper bug" diagnosis drill

**Problem statement.** Using the diagnosis table from Lecture 1 §5, write `notes/week-01/wrapper-bugs.md` with **five** plausible production symptoms (you can use ones from the table or invent your own), and for each: the stage/property responsible and the fix. At least two must be bugs in the *wrapper* (your code) rather than the model — e.g. "it forgot what I said three turns ago" (your history-assembly dropped it) or "it costs 3× what I budgeted" (you counted words not tokens). This drills the habit of localizing LLM behavior to a stage.

**Acceptance criteria.**

- Five symptom → stage → fix entries.
- At least **two** are wrapper bugs (your code), not model bugs — and you say so explicitly.
- Each fix is concrete and actionable, not "use a better model."
- Committed.

**Hint.** The two most instructive wrapper bugs are the stateless one ("it forgot earlier turns" → your code didn't re-send the history) and the tokenizer-cost one ("it's more expensive than expected" → you estimated with the wrong tokenizer or counted characters). Both are the wrapper, not the model — and that distinction is the whole week.

**Estimated time.** 30 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Five-stage trace | 35 min |
| 2 — Three tokenizers, three counts | 45 min |
| 3 — TTFT and TPOT under load | 45 min |
| 4 — Model-selection memo (headline) | 1 h 15 min |
| 5 — Read one license end to end | 40 min |
| 6 — Wrapper-bug diagnosis drill | 30 min |
| **Total** | **~4 h 50 min** |

When you've finished all six, push your repo and make sure the `llmpick` [mini-project](./mini-project/README.md) is in the same workspace — week 5 and week 21 both build on it. Then take the [quiz](./quiz.md) with your notes closed.
