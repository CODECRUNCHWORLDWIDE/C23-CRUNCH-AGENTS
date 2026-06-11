# Challenge 1 — The Model-Selection Memo

**Time estimate:** ~90 minutes.

## Problem statement

You're the AI engineer on a small team. Product hands you three jobs and asks "which model should we use for each?" There is no single right answer — each job has different constraints, and the model that wins one loses another. Your task is to **pick one model per job from a fixed shortlist and defend each pick in a one-page memo backed by measured numbers**, not by a leaderboard rank.

This mirrors the real skill. In an architecture review nobody asks "what's the best model?" — they ask "why *this* model for *this* job, and how do you know it holds at our budget and latency?" This challenge is that conversation, rehearsed.

## The three jobs

| Job | Description | Hard constraints |
|---|---|---|
| **A — Ticket classifier** | Classify a support ticket into one of 8 categories. | 100k calls/day. p50 latency < 1.0s. Budget < $0.0003/call. Input is customer PII (data-handling matters). |
| **B — Document summarizer** | Summarize an 800–1500 word incident report into 3 bullets. | p50 latency < 3.0s. Budget < $0.003/call. Quality must be "faithful" (no invented facts). |
| **C — Hard reasoning** | Given a tricky multi-step word problem, produce a correct worked answer. | No hard latency cap (async). Budget < $0.05/call. Correctness is the only thing that matters. |

## The shortlist (use exactly these candidates)

- `claude-opus-4-8` (hosted frontier)
- `claude-sonnet-4-6` (hosted balanced)
- `claude-haiku-4-5` (hosted fast/cheap)
- `qwen2.5:7b` via Ollama (local open-weights)

You may add **one** more local model of your choice (e.g. `llama3.2:3b`, `gemma3:4b`) if you want a second local data point.

## Your task

For **each of the three jobs**, produce a one-page memo (`notes/week-01/memo-job-{A,B,C}.md`) with these five parts:

1. **The constraint that dominates.** State which constraint (cost, latency, correctness, or data-handling) is the *binding* one for this job. Every selection is really about the binding constraint; name it first.
2. **Candidates considered and the cut.** List which shortlist models you tested and which you eliminated on paper before measuring (and why — e.g. "Opus eliminated for Job A: ~10× over budget for an easy task").
3. **The measurement.** For the surviving candidates, run the *actual job's prompt* (write one representative prompt per job) and report **measured p50 latency** (≥5 runs, take the median) and **measured cost/call** (real token counts × price table) and a **quality note** (1–3 sentences on whether the output met the job's quality bar on 3 self-authored test inputs).
4. **The decision.** State your pick and the runner-up, each in one sentence with the number that decided it.
5. **License / data note.** One line on the license (for local) or data-handling terms (for hosted) relevant to this job — especially Job A's PII.

You may reuse your `exercise-02` uniform client to gather the numbers. That's the point — the uniform client is the measurement harness.

## Acceptance criteria

- [ ] Three memo files exist, one per job, each fitting on roughly one page (300–500 words).
- [ ] Each memo names the **binding constraint** first.
- [ ] Each memo reports **measured p50 latency** (median of ≥5 runs) and **measured cost/call** from real token counts — not estimates, not "it felt fast."
- [ ] Each memo states a pick **and** a runner-up, each with the deciding number.
- [ ] Job A's memo explicitly addresses **PII / data-handling** in the license/data note.
- [ ] At least one job's "right" answer is a **local** model and at least one is a **hosted** model — if all three picks are the same model, you've mis-measured or mis-reasoned; re-check.
- [ ] Committed to your Week 1 repo under `notes/week-01/`.

## The trap (read after a first attempt)

The trap is **picking the most capable model for every job because it's "the best."** It isn't the best — it's the most *capable*, which is a different axis (Lecture 2 §1). For Job A, the frontier model is the *wrong* choice: it's ~10× over budget, slower, and sends PII to a vendor, all to add zero accuracy on an easy 8-way classification a 7B handles fine. For Job B, the balanced or fast tier is usually right. Only Job C, where correctness is the sole constraint and the budget is generous, justifies reaching for the frontier tier. If your three memos all pick `claude-opus-4-8`, you fell in the trap: you optimized capability and ignored the binding constraints. The binding constraint picks the model, not the leaderboard.

## Stretch

- **Add a cost-at-scale line to Job A.** Multiply your measured cost/call by 100,000 calls/day × 30 days. The monthly number makes the local-vs-hosted decision visceral in a way the per-call number hides — this is exactly the calculation that justifies the week-21 routing lab.
- **Quantify the quality gap on Job C.** Write 5 hard reasoning problems with known answers, run them against your top-2 candidates, and report how many each got right. Now your Job C pick has a correctness *number*, not a vibe — the strongest possible defense.
- **Re-run Job A under both `qwen2.5:7b` and a smaller local model** (`llama3.2:3b`). Does the smaller, faster model still classify accurately enough? If so, you've found the cheapest viable option — and that hunt is the whole spirit of cost engineering (Phase IV).

## Why this matters

In the Phase I capstone milestone you ship a working agent on a local 7B and must justify the model behind it. The reviewer will point at your choice and ask "why this model, and how would you know if it were wrong?" This challenge *is* that question, three times over. Every AI engineering role you'll interview for asks some version of "design a model-routing layer" or "how would you pick a model for X" — and the answer they're listening for is exactly this: name the binding constraint, shortlist, measure on the real task, decide with numbers, note the license. Not "I'd use the best one."
