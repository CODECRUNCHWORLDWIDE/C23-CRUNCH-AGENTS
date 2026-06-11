# Mini-Project — `crunch_agent`: A Benchmarked ReAct Agent, Frontier vs Local

> Build a reusable ReAct agent over your Week 4 tool registry, with the four budgets and a legible trace, then run it against a **25-task benchmark** (math, web lookup, code/text execution) on both `claude-opus-4-8` and a local `qwen2.5:7b-instruct`. Report pass rate, cost, and median latency per provider — with a deterministic grader, not vibes — and write the one-page analysis your Phase I milestone review will read.

This is the artifact that proves you can do the thing the whole phase has been building toward: a working ReAct agent on a local 7B with a measurable benchmark score, plus the honest frontier-vs-local comparison. It is the Phase I capstone milestone in miniature, due this week so Week 6 can point the same loop at local inference you stood up yourself.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This agent becomes the spine of **Week 6** (the same loop, served entirely on local inference you bring up), **Week 13** (re-implemented as a LangGraph graph — you will diff against this), and the **capstone** (the supervisor's subordinate agents are this loop with a tool surface). Build it well now; you will import it three more times.

---

## What you will build

A small Python package `crunch_agent` with three deliverables:

1. **`crunch_agent/loop.py`** — the reusable ReAct loop. Provider-agnostic at the seams: one `run(task, provider)` entry point, two thin adapters (`_step_claude`, `_step_qwen`) that differ only in the API plumbing (`tool_use`/`tool_result` vs `tool_calls`/`role:"tool"`). The four budgets are first-class state; every run terminates on `end_turn` or a named budget and prints a termination summary.
2. **`crunch_agent/bench.py`** — the benchmark harness. Loads a 25-task set with deterministic graders, runs every task through a chosen provider, and reports per-provider pass rate, total cost, and p50/p95 latency. Writes a results table.
3. **A 25-task benchmark** (`crunch_agent/tasks.py`) — math, web lookup, and code/text-execution tasks, each with a deterministic grader (an expected substring or exact value). This is what makes "pass rate" a number, not an impression.

By the end you have a public repo of ~300–400 lines of Python (excluding tests) that any future `crunch` package can `from crunch_agent.loop import run` and get a budgeted, traceable, provider-portable agent.

---

## Why a hand-rolled loop and not a framework

You could `pip install` an agent framework and call `.run()`. Don't — not this week. A hand-rolled loop gives you:

- **Trace control.** You see every reason/act/observe/budget tick in a format your eyes already parse. A framework's trace is its format, not yours.
- **Budget placement you chose.** You decide that budgets are checked before each model call, that the cost budget uses real `usage` numbers, that the time budget can wrap a slow tool. A framework decides for you.
- **Debuggability.** When Qwen hallucinates a tool name, you read *your* loop and see exactly where; you do not file an issue against someone else's abstraction.

Week 13 introduces LangGraph where it earns its keep (state graphs with persistence). This week you build the thing it wraps, so that in Week 13 you can say what the graph bought you. That is the contract: the loop is the engineering, not the import.

---

## Package layout

```
crunch_agent/
├── pyproject.toml
├── crunch_agent/
│   ├── __init__.py
│   ├── loop.py          # the ReAct loop + budgets + trace (provider-agnostic)
│   ├── budgets.py       # the Budgets dataclass (step/token/time/cost)
│   ├── adapters.py      # _step_claude, _step_qwen — the only provider-specific code
│   ├── bench.py         # the benchmark harness + reporting
│   └── tasks.py         # the 25-task set with deterministic graders
└── test/
    ├── test_budgets.py  # each budget fires when it should (no API calls)
    └── test_loop.py     # the loop terminates; tool errors don't crash it (fake model)
```

---

## Deliverable 1 — `loop.py` + `budgets.py` (the agent)

The heart of the project. It must:

- Expose `run(task: str, *, provider: str, budgets: Budgets, registry) -> AgentResult` where `provider` is `"claude"` or `"qwen"` and `AgentResult` carries the final answer, the full trace, the termination reason, and the final budget counters.
- Implement the loop exactly once. Provider differences live behind `_step_claude` / `_step_qwen` adapters that take the current `messages` and tools and return `(assistant_turn, tool_calls, usage, stop)` in a normalized shape — so the loop body never branches on provider.
- Enforce the four budgets (step, token, time, cost) as first-class state, checked at the top of each iteration. Use real `usage` from each response for the token and cost budgets. Use `claude-opus-4-8` pricing ($5/M in, $25/M out) for the Claude path; the Qwen path's dollar cost is effectively zero (note this in the trace).
- Never crash on a tool error or a hallucinated tool name — both become `is_error` results the model can recover from (Lecture 1 §2, Lecture 2 §3).
- Print a structured trace (`step / reason / act / observe / final / budget`) and always end with a termination summary line.

The spine to start from (`loop.py`) — fill in the adapter wiring and the registry dispatch:

```python
"""crunch_agent.loop — the reusable ReAct loop. One loop, two providers."""
from __future__ import annotations

from dataclasses import dataclass

from .budgets import Budgets
from .adapters import step_claude, step_qwen   # normalized provider adapters


@dataclass
class AgentResult:
    answer: str
    trace: list[str]
    reason: str           # "end_turn" or "<name> budget exceeded"
    budgets: Budgets


def run(task: str, *, provider: str, budgets: Budgets, registry, system: str) -> AgentResult:
    step = {"claude": step_claude, "qwen": step_qwen}[provider]
    trace: list[str] = []
    messages = _initial_messages(provider, system, task)

    while True:
        breached = budgets.exceeded()
        if breached:
            line = budgets.summary(breached + " exceeded")
            trace.append(line)
            return AgentResult("Stopped: budget exceeded.", trace, breached + " budget exceeded", budgets)

        # The adapter does the one provider-specific thing: call the model and
        # normalize the response. Everything below is provider-agnostic.
        assistant_turn, tool_calls, usage, is_final, final_text = step(
            messages, registry
        )
        budgets.steps += 1
        budgets.record(usage["in"], usage["out"])
        messages.append(assistant_turn)

        if is_final:
            trace.append(f"step {budgets.steps:<2} final   {final_text[:90]}")
            trace.append(budgets.summary("end_turn"))
            return AgentResult(final_text, trace, "end_turn", budgets)

        # TODO 1: for each normalized tool_call (name, args, id), dispatch through
        #   registry (catching errors into is_error results), append reason/act/
        #   observe lines to `trace`, and append the tool results back to `messages`
        #   in the provider's expected shape (the adapter exposes a result-builder).
        ...


# TODO 2: _initial_messages(provider, system, task) — Claude puts system as a
#   top-level kwarg (handled in the adapter); Qwen puts it as a system message.
#   Return the starting messages list for the provider.
```

> **Design rule the tests enforce:** the loop body must not contain the strings `"claude"` or `"qwen"` except in the one dispatch dict at the top. If `grep -n 'claude\|qwen' crunch_agent/loop.py` finds a provider branch *inside* the loop, you have leaked provider logic into the loop and broken the project's reason to exist. Provider differences live in `adapters.py`, full stop.

---

## Deliverable 2 — `tasks.py` (the 25-task benchmark)

A list of 25 tasks across the three syllabus categories, each with a deterministic grader. Aim for roughly:

- **~10 math tasks** — multi-step arithmetic the agent must route to the calculator (and the occasional one it should *not*, to test under/over-triggering).
- **~8 web-lookup tasks** — facts behind a `web_fetch` to a stable URL (use a few fixed pages so the benchmark is reproducible; flaky live pages make a flaky benchmark).
- **~7 code/text tasks** — word counts, string transforms, small computations the agent routes to a sandbox or a text tool.

Each task is `{"id", "task", "category", "grade"}` where `grade(answer: str) -> bool` is deterministic — substring containment or exact value. No LLM-as-judge here; that is Week 12. A deterministic grader is what makes "pass rate" defensible at the milestone review.

```python
TASKS = [
    {"id": "math-01", "category": "math",
     "task": "What is (1234 * 7) + 19?",
     "grade": lambda a: "8657" in a},
    {"id": "lookup-01", "category": "lookup",
     "task": "Fetch https://example.com and report the page's main heading.",
     "grade": lambda a: "Example Domain" in a},
    {"id": "text-01", "category": "text",
     "task": "How many words are in 'the agent loop is a while loop with budgets'?",
     "grade": lambda a: "8" in a},
    # ... 22 more across the three categories
]
```

---

## Deliverable 3 — `bench.py` (the harness + report)

Runs every task through a chosen provider and reports:

1. **Pass rate** per provider — fraction graded correct.
2. **Cost** per provider — total dollars (Claude) / ~$0 (Qwen, noted as electricity not API).
3. **Latency** per provider — p50 and p95 wall-clock per task.
4. A per-category breakdown so you can see *where* the local model loses to the frontier (usually lookup and multi-step math).

Expected shape of the report:

```
PROVIDER   PASS    COST       p50      p95     | math    lookup   text
claude     23/25   $0.31      2.1s     5.8s    | 10/10   7/8      6/7
qwen       17/25   ~$0.00     1.4s     4.2s    | 8/10    4/8      5/7
--------------------------------------------------------------------------
gap: frontier solves +6 tasks; local is free and ~30% faster per call.
```

```bash
python -m crunch_agent.bench --provider claude
python -m crunch_agent.bench --provider qwen
python -m crunch_agent.bench --compare        # both, side by side
```

---

## Rules

- **You may** read the Anthropic docs, the `claude-agent-sdk`, the Ollama docs, and your own Week 4 code.
- **You must not** put provider-specific logic inside the loop body. Adapters only. The `grep` check above is the rule.
- **You must not** use an agent framework for the loop (LangGraph is Week 13). The in-SDK tool runner is allowed *only* in the optional SDK-comparison stretch, clearly separated.
- **You must** compute cost from real `usage` numbers and pass rate from a deterministic grader. Vibes do not count — the rubric fails vibes-only submissions.
- Python 3.12, `anthropic` for Claude, `openai` client pointed at Ollama for Qwen.
- Every run must terminate on `end_turn` or a named budget. No run may hang.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-05-crunch-agent-<yourhandle>`.
- [ ] `crunch_agent.loop.run(...)` works for both `provider="claude"` and `provider="qwen"` over your Week 4 registry.
- [ ] `grep -n 'claude\|qwen' crunch_agent/loop.py` finds provider names **only** in the one dispatch dict — never as a branch inside the loop body.
- [ ] All four budgets are enforced; every run prints a termination summary; no run hangs.
- [ ] `tasks.py` has 25 tasks across math / lookup / text, each with a deterministic grader.
- [ ] `python -m crunch_agent.bench --compare` prints a pass-rate / cost / latency table for both providers with a per-category breakdown.
- [ ] `pytest` passes, with at least:
  - `test_budgets.py`: each of the four budgets fires when it should (no API calls; use a fake model).
  - `test_loop.py`: the loop terminates; a tool that raises becomes an `is_error` result and does not crash the loop.
- [ ] A `README.md` with the results table and a one-page analysis: which provider for what, where the local model loses, and one failure mode you observed (named from the Lecture 2 §3 catalog, with a trace excerpt).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Loop correctness** | 25 | The loop runs both providers, appends turns verbatim, returns one result per tool_use_id, terminates on end_turn or a budget; no provider logic leaks into the loop body. |
| **Budgets** | 20 | All four enforced as first-class state, checked before each call, cost from real `usage`; termination summary always prints; no run hangs. |
| **Benchmark rigor** | 25 | 25 tasks across three categories; deterministic graders; pass rate, cost (real usage), and p50/p95 latency reported per provider with a category breakdown. |
| **Frontier-vs-local analysis** | 15 | The README states which provider for what, *where* local loses (with the category numbers), and names one failure mode from the trace — not vibes. |
| **Tests** | 10 | Budget tests cover all four; loop test proves termination and tool-error resilience; `pytest` green. |
| **Docs & hygiene** | 5 | Clear README with the table, no API keys committed, sensible commits, no `build/`/`__pycache__/` checked in. |

**90+** is portfolio-grade and ready to import into Week 6. **70–89** works but has a soft benchmark or leaked provider logic. **Below 70** means the agent isn't actually portable or the pass rate isn't actually measured — fix that first.

---

## Stretch goals

- **The SDK comparison.** Add `crunch_agent/sdk_loop.py` that drives the same tools with the `claude-agent-sdk` / in-SDK tool runner, run it on the same 25 tasks, and add an SDK row to the report. Now your README answers build-vs-buy with numbers (this is Challenge 1 at full scale).
- **The reflection pass.** Add an optional `--reflect` flag that runs a critique step before accepting the answer (Lecture 1 §4.2). Measure pass-rate change *and* cost change on the 25 tasks. Did it earn its tokens? Report the number.
- **A CI gate.** A GitHub Actions workflow that runs `pytest` and a 3-task smoke benchmark against Qwen in a headless container (no API key needed for the local path). Green check on every push.
- **Trace export.** Emit each run's trace as JSON lines so a future Week-18 observability stack can ingest it. One trace, one file, replayable.

---

## How this connects to the rest of C23

- **Week 6 (local inference)** points this exact loop at Ollama / llama.cpp / vLLM servers you stand up yourself, and benchmarks tokens/sec and quantization trade-offs. Your `provider="qwen"` path is already most of the way there.
- **Week 13 (LangGraph)** re-implements this agent as a state graph with persistence; you will diff lines-of-code, observability, and resumability against this loop.
- **The capstone** uses this loop as the subordinate agents under a supervisor. Build it clean now, keep the repo, import it later.

When you've finished, push the repo and take the [quiz](../quiz.md).
