# Challenge 1 — Hand-Rolled vs the SDK

**Time estimate:** ~90 minutes.

## Problem statement

You have built the agent loop by hand. Now your tech lead asks the question every team eventually faces: *"Should we keep maintaining our own loop, or move to the SDK?"* The wrong way to answer is taste ("the SDK feels cleaner") or fear ("our loop works, don't touch it"). The right way is to run *the same agent two ways* against *the same task set* and report four numbers — pass rate, cost, code surface, and what the SDK hides — then make a recommendation you can defend.

This challenge is that exercise, scaled down to a 10-task set so it fits in 90 minutes. (The mini-project does the full 25.)

## What you build

Two implementations of the *same* agent over your **Week 4 tool registry**:

1. **`agent_handrolled.py`** — your loop from Exercise 3, with the four budgets and a trace.
2. **`agent_sdk.py`** — the same tools and system prompt, but the loop driven by the **in-SDK tool runner** (`client.beta.messages.tool_runner` with `@beta_tool`), or the `claude-agent-sdk`. No hand-written loop.

Both run `claude-opus-4-8`. Both face the same 10 tasks. Both record `usage` so you can total the cost.

### The task set

Write 10 tasks spanning the three categories the syllabus calls out — math, web lookup, and code/text execution — with a deterministic grader for each (a substring the correct answer must contain, or an exact value). Example:

```python
TASKS = [
    {"task": "What is (1234 * 7) + 19?", "expect": "8657"},
    {"task": "How many words are in 'the quick brown fox jumps'?", "expect": "5"},
    {"task": "Fetch https://example.com and report the page title.", "expect": "Example Domain"},
    # ... 7 more across math / lookup / text
]
```

A deterministic grader keeps "pass rate" honest — no LLM-as-judge, no vibes. The answer either contains the expected string or it does not.

## Your task

For **both** implementations, run all 10 tasks and collect:

1. **Pass rate** — fraction graded correct. Same tasks, same grader, both ways.
2. **Cost** — total dollars across the 10 tasks (sum `usage.input_tokens` × $5/M + `usage.output_tokens` × $25/M).
3. **Code surface** — lines of agent code *you* wrote and maintain (exclude tool implementations, which are shared; count the loop/runner plumbing).
4. **What the SDK hides** — a short list of things you can no longer see or control in the SDK version (the exact trace format, where budgets go, the retry behavior, the interleaving).

Then write the recommendation.

## Deliverables

- [ ] `agent_handrolled.py` and `agent_sdk.py`, both runnable, both over the same Week 4 tools.
- [ ] `bench.py` (or a notebook) that runs all 10 tasks through both and prints a comparison table.
- [ ] `challenge-01-report.md` containing:
  - The comparison table: pass rate, cost, code surface (LOC), for both implementations.
  - The "what the SDK hides" list — at least three concrete items.
  - A **recommendation** (keep hand-rolled / move to SDK) with the *reason*, tied to the numbers. "Move to the SDK; pass rate and cost were within noise and it's 60 fewer lines to maintain" is a defensible answer. "The SDK feels nicer" is not.
  - One **failure mode** you observed in either implementation, named from the Lecture 2 §3 catalog, with the trace excerpt.
- [ ] Committed to your Week 5 repo under `challenges/challenge-01/`.

## Acceptance criteria

- [ ] Both implementations use the *same* tools, system prompt, and model — the only difference is who drives the loop.
- [ ] The grader is deterministic; pass rate is a real fraction, not an impression.
- [ ] Cost is computed from real `usage` numbers, not estimated.
- [ ] The recommendation is tied to at least two of the four measured numbers.
- [ ] The named failure mode matches the trace you quote.

## The trap (read after a first attempt)

The tempting shortcut is to declare the SDK the winner on code surface and stop. Resist it. The four numbers can disagree: the SDK might add scaffolding tokens that push cost up a few percent, or its default retry behavior might change the pass rate on a flaky web-fetch task. **Report what you measured, including the parts that don't favor your gut.** A report that says "the SDK was cheaper *and* fewer lines *and* equal pass rate" is suspicious — re-check your cost accounting. The honest answer usually has a trade-off in it, and naming the trade-off is the whole point.

## Stretch

- Add a **reflection pass** to the hand-rolled version (Lecture 1 §4.2) and measure the pass-rate change *and* the cost change on the same 10 tasks. Did the critique earn its tokens? This is the measured version of the week's central pattern-selection question.
- Run the hand-rolled agent against **Qwen 7B via Ollama** on the same 10 tasks. Now you have a 2×2: {hand-rolled, SDK} × {frontier, local} — except the SDK path is Claude-only, so really {hand-rolled-claude, hand-rolled-qwen, sdk-claude}. Chart pass rate and cost for all three. The frontier/local gap is the headline; the build/buy gap is the subplot.
- Instrument both implementations to emit the same **termination summary** line. Confirm that across 10 tasks, *no* run hangs — every run ends on `end_turn` or a named budget. That invariant is the deliverable of the whole budgets topic.

## Why this matters

In Week 13 you re-implement this exact agent a *third* way — as a LangGraph state graph — and the question returns: what did the graph buy you over the loop and over the SDK? You will answer it the same way you answer it here: with measured pass rate, cost, and code surface, plus an honest account of what each abstraction hides. Every framework decision in your career is this conversation. This challenge is the rehearsal — and the engineer who can run it is the one whose architecture reviews go well.
