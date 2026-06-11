# Week 11 — Exercises

Three focused drills that take you from "the model forgets everything between calls" to "I measured that my agent remembers a turn-3 fact in turn 38." Each takes 30–60 minutes. Do them in order — exercise 3 (the regression test) uses the rolling-summary memory you build in exercise 2, and the budget intuition from exercise 1 is why the memory has to be compressed at all.

## Index

1. **[Exercise 1 — Budget the window](exercise-01-budget-the-window.md)** — allocate a token budget across system/semantic/episodic/recent/query, enforce it, and watch a naive append-everything approach blow the window. (~45 min, guided)
2. **[Exercise 2 — Rolling-summary episodic memory](exercise-02-rolling-summary.py)** — build a rolling-summary episodic memory and measure tokens-saved versus facts-retained as the conversation grows. (~50 min, runnable)
3. **[Exercise 3 — The memory regression test](exercise-03-memory-regression.py)** — the turn-38 test: plant a fact early, ask about it late, and score recall for a three-tier agent vs a no-memory baseline. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install the deps as each exercise needs them: `pip install anthropic numpy sentence-transformers`. The semantic tier reuses your week-10 `crunchstore` (pgvector in Docker). For the summarizer/judge, set `ANTHROPIC_API_KEY`, or point at a local model from week 6.
- **Count tokens with the model's tokenizer, not characters.** Every budget is in real tokens (`client.messages.count_tokens(...)` for Claude, the model's tokenizer for a local model). Counting characters — or reaching for `tiktoken` on a Claude model — is the classic budget-that-lies bug (week 2's lesson).
- **Measure recall, don't eyeball it.** Exercise 3's whole point is that "it feels like it remembers" is a vibe; the recall rate against a no-memory baseline is the number.
- **Promote durable facts to semantic memory.** The reason an old fact survives to turn 38 is that it was extracted into the semantic tier, not because the rolling summary happened to keep it. If recall is bad, check the semantic tier first.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

Exercises 2 and 3 are standalone and written to run **without an API key** by default — they use a deterministic stub summarizer/judge so the *mechanics* (budgeting, rolling summary, recall scoring) are exercisable today, and document how to swap in a real model (`claude-sonnet-4-6` or a local 7B) for the real thing.

```bash
# Exercise 1 is a markdown walkthrough you run interactively.
# Exercises 2 and 3 run standalone:
python3 exercise-02-rolling-summary.py
python3 exercise-03-memory-regression.py

# To use a real model for summarization/judging, set:
export ANTHROPIC_API_KEY=sk-ant-...
# (or point the script's --base-url at a local Ollama/vLLM endpoint from week 6)
```

## A note on determinism

Rolling-summary content is *not* deterministic when a real LLM does the summarizing — the same turns can produce slightly different summaries run-to-run. The *recall rate* is what's reproducible: a three-tier agent should recall the turn-3 fact in turn 38 essentially every run, and a no-memory baseline should fail essentially every run. If your three-tier recall swings wildly, the semantic tier isn't reliably storing/retrieving the fact — find that before you trust the number. The stub-summarizer path *is* deterministic, so use it to debug the harness before swapping in a real model.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-11` to compare.
