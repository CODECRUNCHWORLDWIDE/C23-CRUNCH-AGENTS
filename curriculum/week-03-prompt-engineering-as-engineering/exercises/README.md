# Week 3 — Exercises

Three focused drills that turn the lectures into muscle memory. Each takes 30–60 minutes. Do them in order — exercise 2 builds the regression harness the challenge and mini-project both grow from, and exercise 3 gives you the CoT/self-consistency numbers you'll cite when deciding whether a prompt needs them.

## Index

1. **[Exercise 1 — Spec then implement](exercise-01-spec-then-implement.md)** — write a prompt spec *first*, implement it as a versioned file, diff two iterations, and review the change against a structured prompt-review checklist. The discipline that makes every later prompt change defensible. (~50 min, guided)
2. **[Exercise 2 — The promptfoo-style harness](exercise-02-promptfoo-harness.py)** — build a minimal regression harness in Python: load golden examples, run a prompt version against a model, score the pass rate, and compare two versions to catch a regression. (~50 min, runnable)
3. **[Exercise 3 — CoT vs direct vs self-consistency](exercise-03-cot-self-consistency.py)** — measure chain-of-thought against direct prompting, and self-consistency (majority vote over N samples), on a small reasoning set; report the accuracy delta *and* the cost multiple so the trade-off is numbers, not slogans. (~50 min, runnable)

## How to work the exercises

- **Have your environment ready before you start.** A venv with `anthropic` installed, `ANTHROPIC_API_KEY` exported (or accept the local-only Ollama fallback), Node 18+ for `npx promptfoo` if you want to run the real tool alongside exercise 2, and `git` for committing versions:

  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install anthropic
  ollama pull qwen2.5:7b      # or llama3.2:3b on a ≤16GB machine (fallback path)
  node --version              # 18+ for `npx promptfoo eval` (exercise 1 / challenge)
  ```

- **Commit every prompt version.** This week the git history *is* the deliverable. When you write `v2` of a prompt, commit it with the pass rate in the message. Treat the log as the artifact a reviewer reads.
- Each runnable exercise (`.py`) ends with an **expected output** block. Your exact numbers will differ (your model, your network, sampling nondeterminism) but the *shape* — a regression caught, a pass-rate delta, an accuracy/cost trade — must match. If it doesn't, you're not done.
- **No API key? No problem.** Exercises 2 and 3 both run end-to-end against Ollama alone. The Anthropic path degrades gracefully to an "unavailable" line; the local path carries the lesson.

## Running the Python exercises

The two `.py` files are standalone — no package, no framework. Activate your venv and run them directly:

```bash
source .venv/bin/activate
python3 exercise-02-promptfoo-harness.py
python3 exercise-03-cot-self-consistency.py
```

Each file's header documents how to use it, the acceptance criteria, and the expected output shape. Read the header before you run.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-03` to compare.
