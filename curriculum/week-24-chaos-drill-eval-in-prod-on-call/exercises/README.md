# Week 24 — Exercises

Three focused drills that take you from "design a chaos drill" to "I ran one and gated a deploy on eval-in-prod." Each takes 45–60 minutes. Do them in order — exercise 2 runs the drill you design in exercise 1, and exercise 3 is the eval-in-prod gate that decides whether a candidate ships after a drill exposes a fix.

## Index

1. **[Exercise 1 — Design the drill](exercise-01-design-the-drill.md)** — write the steady-state hypothesis, blast radius, controlled window, tested revert, and measurement plan for one of the three required drills. (~45 min, guided)
2. **[Exercise 2 — The chaos-drill runner](exercise-02-chaos-drill-runner.py)** — a runner that injects a fault, probes the system every second, measures impact + recovery time, reverts cleanly, and emits a postmortem skeleton. (~55 min, runnable)
3. **[Exercise 3 — Eval on traces](exercise-03-eval-on-traces.py)** — replay production traces through a candidate version and gate the deploy on the replayed eval, catching a regression before any user sees it. (~50 min, runnable)

## How to work the exercises

- These run against your **Sprint B capstone** (the week-23 system). The chaos drill attacks it; the eval-on-traces gates a candidate change to it. If Sprint B isn't runnable, you have nothing to break — fix it first.
- **Test the revert before you inject.** Exercise 2's whole discipline is that the revert runs in a `finally` and is confirmed working *before* the fault. For the index drill, you restore a backup onto a copy first. A drill with no tested revert is an outage you caused.
- **Measure a number, not a vibe.** Every drill starts with a steady-state hypothesis that is a *number* (error_rate=0%, p95<2.5s, faithfulness>=0.85). You compare the fault state to that baseline; "it seemed fine" is not a measurement.
- Set `ANTHROPIC_API_KEY` — the vendor fallback (drill 1), the prompt-injection probe (drill 2), and the eval-on-traces judge (exercise 3) all may call `claude-opus-4-8` with `thinking={"type":"adaptive"}` and `output_config={"effort":...}` (never `budget_tokens`/`temperature`).
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone and ship a small simulated capstone so they run with no GPU, no live cluster, and no corpus — the simulation lets you see the *shape* of a drill and a gate before you point them at your real Sprint B system (which the mini-project does). Exercise 3 may call the judge; it mocks the candidate's answers so the gate logic is testable cheaply.

```bash
# with ANTHROPIC_API_KEY set (optional for the simulated paths), venv active:
python3 exercise-02-chaos-drill-runner.py     # runs a simulated GPU-node-loss drill
python3 exercise-03-eval-on-traces.py         # replays traces, gates a candidate
```

## A note on determinism

The simulated drills in exercise 2 are deterministic — the same fault produces the same timeline shape every run, so you can develop the runner without flakiness. Against the *real* Sprint B system, the recovery time will vary a little (network, model latency); that variance is itself data — record the range, not a single number. The eval-on-traces gate in exercise 3 is deterministic given fixed candidate answers; the point is the *gate logic* (does a regression block the deploy?), which must be reproducible.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-24` to compare.
