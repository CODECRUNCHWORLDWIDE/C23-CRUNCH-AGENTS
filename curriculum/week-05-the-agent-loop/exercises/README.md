# Week 5 — Exercises

Three drills that turn the agent loop from a diagram into muscle memory. Do them in order — exercise 3 reuses the loop and budgets you build a feel for in 1 and 2. Everything runs against your **Week 4 tool registry** (`crunch_tools` or equivalent) and either `claude-opus-4-8` via the Anthropic SDK or `qwen2.5:7b-instruct` via Ollama.

## Index

1. **[Exercise 1 — Trace a run and find the failure](exercise-01-trace-a-run.md)** — run a hand-rolled agent on three tasks, annotate every step of one trace, and name the failure mode. The diagnostic habit of the week. (~45 min, guided)
2. **[Exercise 2 — Budget guards](exercise-02-budget-guards.py)** — add step/token/time/cost budgets to a loop and force *each one* to fire, proving the agent always terminates. (~45 min, runnable)
3. **[Exercise 3 — A complete ReAct loop](exercise-03-react-loop.py)** — a full ~150-line ReAct agent over the Week 4 registry, runnable against Claude and Qwen with the same tool schema. (~50 min, runnable)

## How to work the exercises

- Have your **Week 4 tool registry** importable (`from crunch_tools.registry import REGISTRY`) with at least a calculator, a sandboxed file-read, and an SSRF-guarded web-fetch. If yours is shaped differently, adapt the two adapter functions at the top of each file; the loop is unchanged.
- Export `ANTHROPIC_API_KEY` for the Claude path. Have Ollama serving `qwen2.5:7b-instruct` (`ollama list` shows it) for the local path. Each runnable exercise works on either.
- **Read the trace before you touch code.** When an agent misbehaves, run the failure-mode decision tree from Lecture 2 §3 before editing — terminate? which budget? which failure mode? The trace tells you.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output's *shape* doesn't match (it terminates cleanly; the budget that should fire fires), you're not done — exact token counts and timings vary by run.

## Running the Python exercises

```bash
pip install anthropic openai          # openai client is used only for the Ollama path
export ANTHROPIC_API_KEY=sk-ant-...
python3 exercise-02-budget-guards.py
```

The local path additionally needs Ollama running: `ollama serve` and `ollama pull qwen2.5:7b-instruct`. Each file's header says which models it touches and how to switch.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-05` to compare.
