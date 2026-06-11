# Challenge 1 — The Token-Budget Audit

**Time estimate:** ~90 minutes.

## Problem statement

You inherit a document-processing pipeline. It works — it summarizes incident reports into structured triage records — but the monthly LLM bill is higher than anyone expected, and finance is asking why. Your job is to **audit the token budget, find where the tokens (and dollars) go, cut the waste, and prove on real measurements that quality held.** Not "rewrite it from scratch." Not "switch to a cheaper model and hope." Measure, cut, prove.

This is the real cost-engineering loop, rehearsed. In production nobody asks "is your prompt good?" — they ask "this costs $4,200/month, can you halve it without breaking it, and how do you know?" That conversation is this challenge.

## The pipeline you're auditing

A per-document pipeline that, for each incident report, makes one model call shaped like this:

```
system prompt  : a 1,400-token instruction block (role, 12 few-shot examples,
                 a verbose style guide, a full copy of the JSON schema in prose)
user message   : the full incident report (avg ~2,000 tokens) PLUS the entire
                 "runbook appendix" (~3,500 tokens) pasted in every call "for context"
output         : a ~400-token free-text summary the caller then regex-parses into fields
model          : claude-opus-4-8 (chosen because "it's the best")
volume         : 40,000 documents/day
```

Per-MTok prices (use these): `claude-opus-4-8` in $5.00 / out $25.00; `claude-sonnet-4-6` in $3.00 / out $15.00; `claude-haiku-4-5` in $1.00 / out $5.00.

If you don't have a corpus, generate one: write or synthesize **10 representative incident reports** (~2,000 tokens each) and reuse the same 1,400-token system prompt and 3,500-token runbook so your before/after numbers are apples-to-apples.

## Step 1 — Instrument and establish the baseline

Build (or reuse your Week-2 `toklab` token-accounting instrument) to record, per call: `tokens_in`, `tokens_out`, and `cost`. Use **real token counts** — `client.messages.count_tokens` for the input and `msg.usage` for actuals, never an estimate, never `tiktoken`.

Produce a **baseline table**: for your 10 documents, the mean input tokens (broken down: system prompt / report / runbook / chat-template overhead), mean output tokens, mean cost/call, and the projected **monthly cost** at 40,000/day × 30. This is the "before" number finance sees.

## Step 2 — Find the waste (the four usual suspects)

Audit where the tokens go and identify cuts. The four classic sources of token waste, at least three of which are present here:

1. **The bloated system prompt.** 1,400 tokens re-sent on *every* call. Do you need 12 few-shot examples, or do 3 do as well? Is the schema-in-prose redundant if you constrain the output (Step 3)? Could a fixed system prompt be **prompt-cached** so you stop paying full price for it on every call?
2. **Redundant context.** The 3,500-token runbook pasted into every call — is it actually used per document, or is it "just in case" padding that also risks burying the real content in the *middle* of the context (lost-in-the-middle, Lecture 1 §5.3)? Could it be retrieved on demand, or summarized once into a 300-token cheat-sheet?
3. **The wrong tier.** `claude-opus-4-8` for a structured-extraction task. Is the frontier tier buying you anything here, or is this an easy job a cheaper tier (or a local model) handles at a fraction of the per-token price? (This is the Week-1 binding-constraint lesson, now with the token numbers behind it.)
4. **Output verbosity.** A 400-token free-text summary that gets regex-parsed. Output is the *expensive* side (3–5× input price). Constraining the output to a compact schema-valid JSON record (Week 2's structured-output lesson) can cut output tokens **and** delete the fragile regex parser at the same time.

## Step 3 — Cut, and measure the "after"

Apply your cuts and re-measure on the **same 10 documents**:

- Trim the system prompt (fewer few-shot examples; drop redundant schema prose).
- Replace the pasted runbook with a compact summary or on-demand retrieval (or justify keeping it).
- Right-size the model tier for the task.
- Switch the output to **schema-constrained JSON** (compact, parseable, no regex), shrinking output tokens.

Re-run the instrument. Produce the **after table** (same columns as baseline) and the new projected monthly cost.

## Step 4 — Prove quality held

A cost cut that breaks quality is not a cut; it's a regression you'll pay for later. Define a **quality bar** before you cut (e.g. "the triage category and priority match a human-labeled gold answer on all 10 docs; no invented facts in the summary"). Run your before and after pipelines against the 10 gold-labeled docs and report the quality score for each. The cut is only valid if quality is **>= baseline**.

## Deliverables

Commit to `notes/week-02/token-budget-audit/`:

1. `baseline.md` — the before table (token breakdown, cost/call, monthly projection).
2. `after.md` — the after table and the same monthly projection.
3. `audit.md` — a one-page memo: which of the four waste sources you found, what you cut, the before/after monthly numbers, the **percent saved**, and the quality result (before vs after) proving you didn't break it.
4. The instrument script you used to gather the numbers (or a pointer to your `toklab` instrument).

## Rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Real measurement** | 25 | Token counts from real tokenizers (`count_tokens` / `usage`), not estimates; baseline broken down by component. |
| **Finding the waste** | 25 | At least three of the four waste sources identified with the token cost of each named. |
| **The cuts** | 20 | Cuts applied and re-measured on the same docs; includes the output-constraint cut and a tier or caching decision. |
| **Quality preserved** | 20 | A pre-defined quality bar, scored before AND after on gold labels, showing quality >= baseline. |
| **The memo** | 10 | One page, leads with the percent saved and the monthly dollar delta, states the quality result. A budget owner could act on it. |

**90+** is a memo you could send to finance unedited. **70–89** finds the waste but estimates somewhere it should measure, or cuts cost without proving quality held. **Below 70** cuts cost on vibes — the exact failure this challenge teaches you to avoid.

## The trap (read after a first attempt)

The trap is **cutting the most visible thing instead of the most expensive thing.** The system prompt is right there at the top, so people trim it first — but at 1,400 tokens × $5/MTok input it may be a smaller line item than the 400-token output at $25/MTok output, or than the 3,500-token runbook re-sent every call. **Measure the cost of each component before you cut.** Output tokens are 5× the price of input on Opus; the single highest-leverage cut is often "stop generating 400 tokens of prose and emit a 60-token JSON record instead." If your audit trims the prompt and ignores the output, you optimized the visible thing, not the expensive thing.

## Stretch

- **Add prompt caching to the math.** Model the fixed system prompt as a cached prefix (Lecture 1 §6) and recompute the monthly cost with the cached-prefix discount. For a fixed 1,400-token system prompt at 40k calls/day, the savings are large — quantify them.
- **Plot cost vs quality across tiers.** Run the job on Opus, Sonnet, Haiku, and a local `qwen2.5:7b`, plotting cost/call against your quality score. The cheapest point that clears the quality bar is your answer — and the plot is the most defensible artifact you can put in a review.
- **Find the lost-in-the-middle effect.** Put a fact the summary needs at the start, middle, and end of the (still-pasted) runbook and check whether the model uses it. If middle-position degrades quality, you have an empirical reason to cut the runbook beyond just cost.
