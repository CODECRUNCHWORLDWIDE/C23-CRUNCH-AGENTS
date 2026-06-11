# Mini-Project — `llmpick`: A Model Recommender That Shows Its Work

> Build a CLI tool `llmpick` that takes a `--prompt`, a `--budget` (max USD per call), and a `--latency-target` (max p50 seconds), queries N candidate models **in parallel**, measures each one's latency and cost on the actual prompt, and recommends one model with reasons you can defend in a review.

This is the artifact that turns Week 1's whole thesis — *a model is a component you select against constraints, with numbers, not vibes* — into a tool you'll actually reach for. By the end you have a command that answers "which model should I use for this prompt, under this budget and this latency target?" with a recommendation, a runner-up, the rejected candidates, and the measured number behind every line.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** The parallel-query-and-measure core you build here is the seed of the **week-21 model-routing layer** (route easy queries to a cheap model, hard ones to a frontier model). The uniform `complete()` interface is the same one every later lab swaps models through. Build it well now; you'll extend it in twenty weeks.

---

## What you will build

A small Python package `llmpick` with three deliverables:

1. **`llmpick/backends.py`** — the uniform client. A `complete(prompt, model) -> Completion` interface over at least two transports: the Anthropic SDK (hosted) and the Ollama HTTP API (local). Every backend returns the same `Completion` dataclass (text, tokens_in, tokens_out, latency_s, cost). This is `exercise-02` promoted to a real module.
2. **`llmpick/recommend.py`** — the decision engine. Given a prompt, a budget, a latency target, and a list of candidate models, it queries them **in parallel**, runs each ≥3 times to get a stable p50 latency, computes cost from real token counts, and returns a ranked recommendation with a reason per model (chosen / runner-up / rejected-and-why).
3. **`llmpick/cli.py`** — the command-line entry point. `llmpick --prompt ... --budget ... --latency-target ...` prints the recommendation block.

By the end you have a public repo of ~250–400 lines of Python (excluding tests) that you can run against any prompt and get a defensible answer.

---

## Why parallel, and why measure-don't-estimate

Two design rules are non-negotiable, because they're the lessons of the week:

- **Query candidates in parallel, not in series.** A serial loop over five models takes the sum of their latencies — minutes. In parallel it takes the slowest single model. Use `asyncio.gather` (async backends) or a thread pool. This also models reality: a routing layer in production fans out, it doesn't wait in line.
- **Measure on the actual prompt; never estimate.** The whole point is that cost and latency depend on *this* prompt's tokens and *this* model's tokenizer and speed. Estimating from a different model's tokenizer or a static price-per-call is exactly the mistake Lecture 1 and 2 warned against. `llmpick` exists to replace the estimate with a measurement.

---

## Package layout

```
llmpick/
├── pyproject.toml
├── llmpick/
│   ├── __init__.py
│   ├── backends.py        # the uniform complete() interface + Completion
│   ├── recommend.py       # parallel query, p50, cost, the ranking logic
│   └── cli.py             # argparse entry point -> recommendation block
└── tests/
    ├── test_recommend.py  # unit tests: ranking logic, constraint filtering
    └── test_backends.py   # unit tests: cost computation, Completion shape
```

---

## Deliverable 1 — `backends.py` (the uniform client)

Promote `exercise-02` into a module. It must:

- Define the `Completion` dataclass (`backend`, `model`, `text`, `tokens_in`, `tokens_out`, `latency_s`, `cost_usd()`).
- Provide `complete(prompt: str, model: str) -> Completion` that dispatches to the right transport based on the model name (a `claude-*` name → Anthropic SDK; an Ollama tag → Ollama HTTP).
- Compute cost from real token counts and a price table (per-MTok), with the local model's marginal token cost at 0.
- **Degrade gracefully:** if a backend is unavailable (no key, Ollama down), raise a clean exception the recommender can catch and report as "unavailable," not crash the whole run.

Here is the spine to start from; fill in the rest yourself:

```python
"""llmpick.backends — one complete() interface over hosted and local models."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

PRICES = {
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-opus-4-8": {"in": 5.00, "out": 25.00},
}
LOCAL_PRICE = {"in": 0.0, "out": 0.0}


@dataclass
class Completion:
    model: str
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float

    def cost_usd(self) -> float:
        price = PRICES.get(self.model, LOCAL_PRICE)
        return (self.tokens_in * price["in"] + self.tokens_out * price["out"]) / 1_000_000


def complete(prompt: str, model: str) -> Completion:
    """Dispatch to the right transport. Raise on unavailability; don't crash."""
    if model.startswith("claude-"):
        return _complete_anthropic(prompt, model)
    return _complete_ollama(prompt, model)


def _complete_anthropic(prompt: str, model: str) -> Completion:
    if anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(f"{model}: anthropic SDK / key unavailable")
    client = anthropic.Anthropic()
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model, max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return Completion(model, text, msg.usage.input_tokens,
                      msg.usage.output_tokens, time.perf_counter() - t0)


def _complete_ollama(prompt: str, model: str) -> Completion:
    t0 = time.perf_counter()
    r = httpx.post("http://localhost:11434/api/generate",
                   json={"model": model, "prompt": prompt, "stream": False},
                   timeout=300.0)
    r.raise_for_status()
    d = r.json()
    return Completion(model, d["response"], d["prompt_eval_count"],
                      d["eval_count"], time.perf_counter() - t0)
```

---

## Deliverable 2 — `recommend.py` (the decision engine)

This is the heart of the project. Given a prompt, a budget, a latency target, and a candidate list, it must:

1. **Query all candidates in parallel**, each run **≥3 times**, and take the **median latency** (p50) per model. (Median, not mean — one slow run shouldn't dominate.)
2. **Compute cost/call** from the real token counts (use the first successful run's counts; they're stable for the same prompt).
3. **Filter against the constraints**: a model is *eligible* only if its p50 latency ≤ `latency_target` AND its cost ≤ `budget`.
4. **Rank the eligible models** and pick a winner. The ranking policy is yours to design and *document* — a sensible default is "cheapest eligible model wins, ties broken by latency," because among models that all clear the bar, cost is usually what you optimize. State your policy in the repo README.
5. **Return a structured result** with: the chosen model + reason, the runner-up + reason, and every rejected model + the specific reason it was rejected (over budget / over latency / unavailable).

A model that's *unavailable* (backend down) is reported as rejected-with-reason, not silently dropped — you want to know a candidate couldn't even be measured.

The ranking logic must be **unit-testable without calling any model** — separate the "given these measured Completions and constraints, what's the verdict?" pure function from the "go measure the models" I/O. `test_recommend.py` tests the pure function with hand-built `Completion` objects.

Sketch of the pure ranking core:

```python
@dataclass
class Verdict:
    chosen: str | None
    reason: str
    runner_up: str | None
    rejected: dict[str, str]   # model -> reason


def rank(results: dict[str, Completion], budget: float,
         latency_target: float) -> Verdict:
    """Pure: given measured results + constraints, decide. No I/O here."""
    eligible = {}
    rejected = {}
    for model, c in results.items():
        if c.latency_s > latency_target:
            rejected[model] = f"over latency target ({c.latency_s:.2f}s > {latency_target}s)"
        elif c.cost_usd() > budget:
            rejected[model] = f"over budget (${c.cost_usd():.6f} > ${budget})"
        else:
            eligible[model] = c
    if not eligible:
        return Verdict(None, "no candidate met both constraints", None, rejected)
    # Default policy: cheapest eligible wins, ties broken by latency. Document this.
    ordered = sorted(eligible.items(), key=lambda kv: (kv[1].cost_usd(), kv[1].latency_s))
    chosen, c = ordered[0]
    reason = (f"met latency ({c.latency_s:.2f}s ≤ {latency_target}s) and budget "
              f"(${c.cost_usd():.6f} ≤ ${budget}); cheapest eligible.")
    runner_up = ordered[1][0] if len(ordered) > 1 else None
    return Verdict(chosen, reason, runner_up, rejected)
```

---

## Deliverable 3 — `cli.py` (the command)

An `argparse` entry point that wires it together. Required flags: `--prompt`, `--budget`, `--latency-target`. Optional: `--models` (comma-separated candidate list, with a sensible default shortlist). It prints the recommendation block — the "I can defend this number" format from the week README:

```
$ llmpick --prompt "Summarize this 800-word incident report in 3 bullets." \
          --budget 0.002 --latency-target 2.0

RECOMMENDATION: claude-haiku-4-5
  reason: met the 2.0s latency target (measured 1.30s p50) and the $0.002
          budget (measured $0.00041/call).
  runner-up: qwen2.5:7b (local, $0.00) — 0.90s but eliminated on a quality
             spot-check / or: kept as runner-up if quality acceptable.
  rejected:
    claude-opus-4-8  -> over budget ($0.0073 > $0.002)
    claude-sonnet-4-6 -> over budget ($0.0029 > $0.002)
```

---

## Rules

- **You may** read the lecture notes, the SDK docs, and your own exercise-02.
- **You must** query candidates in parallel (`asyncio.gather` or a thread pool) — a serial loop is an automatic fail on the "engineering quality" axis.
- **You must** measure cost and latency on the actual prompt; no static estimates, no other model's tokenizer.
- **You must not** crash when a backend is unavailable — report it as a rejected candidate with reason "unavailable."
- Python 3.12; dependencies limited to `anthropic`, `httpx`, and the standard library (plus `pytest` for tests).
- The ranking policy must be **documented** in the repo README — graders check that your tie-breaking is stated, not implicit.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-01-llmpick-<yourhandle>`.
- [ ] `pip install -e .` succeeds; `llmpick --help` prints usage.
- [ ] `llmpick --prompt "..." --budget X --latency-target Y` prints a recommendation block with: a chosen model + measured reason, a runner-up, and every rejected model + a specific reason.
- [ ] Candidates are queried **in parallel** (demonstrably — the total runtime is ~the slowest model, not the sum).
- [ ] Cost and latency are **measured** from real token counts and `perf_counter`, not estimated.
- [ ] A backend being unavailable produces a "rejected: unavailable" line, not a crash.
- [ ] `pytest` passes, with at least:
  - `test_recommend.py`: the pure `rank()` function tested with hand-built `Completion`s, covering the chosen / runner-up / all-rejected / over-budget / over-latency cases.
  - `test_backends.py`: `cost_usd()` computed correctly for a hosted and a local model.
- [ ] A `README.md` in the repo root with the run commands, the **documented ranking policy**, and one paragraph on a result that surprised you (e.g. "the local 7B beat the hosted fast tier on cost-per-call by ∞ but lost on a quality check").
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Uniform interface** | 20 | One `complete()` over ≥2 transports; identical `Completion` out; graceful unavailability. |
| **Parallelism** | 15 | Candidates queried concurrently; total runtime ~slowest model, demonstrated. |
| **Measurement discipline** | 25 | p50 from ≥3 real runs; cost from real token counts and the price table; no estimation, no cross-tokenizer counting. |
| **Ranking logic** | 20 | Constraint filtering correct; chosen/runner-up/rejected with specific reasons; tie-break policy documented. |
| **Tests** | 15 | Pure `rank()` tested without I/O; cost computation tested; `pytest` green. |
| **Docs & hygiene** | 5 | Clear README, documented ranking policy, no secrets committed, sensible commits. |

**90+** is portfolio-grade and ready to grow into the week-21 router. **70–89** works but estimates somewhere it should measure, or serializes a query that should parallelize. **Below 70** means the tool recommends without measuring — fix that first; a recommender that doesn't measure is the exact thing this week argues against.

---

## Stretch goals

- **Add a quality gate.** Take a tiny held-out check (e.g. "the summary must contain ≤ 3 bullets and mention the incident's root cause") and run it as a cheap programmatic check on each model's output. Now eligibility is latency AND budget AND a quality floor — the full three-way decision a real router makes.
- **Add a third backend.** Wire in a second local model (`llama3.2:3b`) so the shortlist spans frontier / balanced / fast-hosted / two-local. Watch how the recommendation shifts as the budget tightens.
- **Cost-at-scale flag.** Add `--calls-per-day N` and print the projected monthly cost of the chosen model. This is the number that wins (or loses) the local-vs-hosted argument in a real review.
- **JSON output.** Add `--json` so `llmpick` can be a subcommand inside a larger tool. The week-21 router will want exactly this machine-readable verdict.

---

## How this connects to the rest of C23

- **Week 2 (tokens & sampling)** makes your cost numbers more precise — you'll understand exactly why two models' token counts differ for the same text.
- **Week 5 (the agent loop)** uses your uniform `complete()` as the model interface its ReAct loop calls.
- **Week 21 (cost engineering & routing)** is `llmpick` grown up: instead of recommending one model for one prompt, it routes a *stream* of prompts to the cheapest model that clears a quality bar, with a semantic cache in front. The parallel-measure core you build now is that router's beating heart.

When you've finished, push the repo and take the [quiz](../quiz.md).
