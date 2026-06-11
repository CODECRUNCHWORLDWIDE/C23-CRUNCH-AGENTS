# Mini-Project — `crunchobs`: The Reusable Observability Package

> Build a reusable observability package that any C23 agent imports to get OTel Gen-AI tracing, dual export to Langfuse *and* Phoenix, token/latency accounting, a small SLO checker, and an eval-on-traces replay tool — so "make this agent observable" becomes `from crunchobs import trace_agent`, not a week of bespoke instrumentation per project.

This is the artifact that turns observability from a per-project chore into a one-import default. After this week, instrumenting an agent is `crunchobs.init()` at startup and a decorator on each node — not copy-pasting `TracerProvider` boilerplate into every repo and getting the `gen_ai.*` names subtly wrong each time. The package is agent-agnostic, backend-dual (Langfuse + Phoenix), and convention-correct, and it is the thing the **capstone (weeks 22–24)** imports to make its traces flow.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This module is imported directly by your **capstone**. The capstone rubric assumes traces flow to self-hosted Langfuse + Phoenix and that you can produce the three dashboards and an eval-on-traces replay on demand — `crunchobs` is *how*. It also feeds **week 19 (vLLM in production)**, where you instrument a model server you own with the same package. Build it well now; you will lean on it for the rest of the course.

---

## What you will build

A small Python package `crunchobs` with six deliverables:

1. **`crunchobs/tracing.py`** — the instrumentation wrapper: a `trace_agent` decorator / context manager that emits a span with the correct `gen_ai.*` attributes from an LLM-call result, so callers stamp tokens, model, and operation in one line instead of six `set_attribute` calls they'll get wrong.
2. **`crunchobs/exporters.py`** — the dual-exporter config: one `TracerProvider` with two OTLP processors (Phoenix + Langfuse), plus a `ConsoleSpanExporter` dev mode and an `InMemorySpanExporter` test mode — the whole "export to both backends" milestone, behind one `init()`.
3. **`crunchobs/accounting.py`** — the token/cost/latency layer: `rollup(spans, key)` for per-route/user/model cost (2026 Anthropic prices), and `percentiles()` / `latency_per_step()` for p50/p95/p99 per step.
4. **`crunchobs/slo.py`** — the SLO checker: given durations + error/total counts + a p95 budget + a success SLO, report p95-vs-budget, success rate, error-budget consumed, burn rate, and a pass/fail.
5. **`crunchobs/replay.py`** — the eval-on-traces replay tool: load a recorded trace, re-run a generation span through a new prompt version (same model, same inputs), and diff output + metrics.
6. **`crunchobs/cli.py`** — a CLI tying it together: `crunchobs accounting`, `crunchobs slo`, `crunchobs replay`.

By the end you have a public repo of ~500–700 lines of Python that any future C23 project can `from crunchobs import init, trace_agent` and stop hand-rolling observability.

---

## Why a module and not a notebook

You could instrument an agent inline in a notebook. Don't — not as the artifact. A module gives you:

- **Reuse.** The capstone and week 19 import this package. A notebook gets copy-pasted, drifts, and the `gen_ai.*` names rot into `tokens_in`.
- **Correctness once.** The convention attribute names, the dual-export wiring, the rollup-only-counts-chat-spans rule — these are easy to get subtly wrong. Encode them *once*, test them, and every caller inherits correctness.
- **A CLI and tests.** `crunchobs slo --budget 1500` is greppable, scriptable, and CI-able; the rollup and percentile logic is unit-testable against fixed recorded spans. A notebook cell is none of those.

Notebooks are great for *exploring* a single trace by eye (Exercise 1 territory). The thing you ship and depend on is a module. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchobs/
├── pyproject.toml
├── docker-compose.langfuse.yml   # self-hosted Langfuse (Postgres + ClickHouse + web + worker)
├── README.md                     # the accounting/SLO/replay results + how to wire it
├── recordings/
│   └── sample_trace.json         # a recorded trace for replay + accounting fixtures
├── crunchobs/
│   ├── __init__.py               # exports: init, trace_agent, rollup, slo_report, replay_recording
│   ├── tracing.py                # the trace_agent wrapper (gen_ai.* in one line)
│   ├── exporters.py              # dual export (Phoenix + Langfuse) + console/memory modes
│   ├── accounting.py             # rollup() + percentiles() + latency_per_step()
│   ├── slo.py                    # slo_report() with burn rate
│   ├── replay.py                 # eval-on-traces replay
│   └── cli.py                    # the `accounting` / `slo` / `replay` commands
└── tests/
    ├── test_accounting.py        # rollup sums + percentile correctness on fixed spans
    ├── test_slo.py               # error-budget + burn-rate math on known inputs
    └── test_tracing.py           # trace_agent emits the real gen_ai.* names (InMemory)
```

Your Phase III agent is the *consumer* of this package, not part of it: it does `import crunchobs; crunchobs.init(...)` and decorates its nodes.

---

## Deliverable 1 — `tracing.py` (the instrumentation wrapper)

The heart of the developer experience. Instead of six `set_attribute` calls per LLM span (and the chance to fmisspell one), callers stamp a span from a result object in one line.

```python
"""crunchobs.tracing — emit conventions-correct gen_ai.* spans in one call.

The convention attribute NAMES live HERE and nowhere else, so callers can never
spell gen_ai.usage.input_tokens wrong. (Lecture 1 §3)
"""
from __future__ import annotations

from contextlib import contextmanager

from opentelemetry import trace

# The convention names — exact. The single source of truth for the whole course.
SYSTEM = "gen_ai.system"
OPERATION = "gen_ai.operation.name"
REQUEST_MODEL = "gen_ai.request.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"


@contextmanager
def trace_agent(name: str, operation: str, *, route: str | None = None,
                user_id: str | None = None):
    """Open a span for an agent step. `operation` is a real convention value
    (chat / embeddings / execute_tool / invoke_agent). Business dimensions go
    under crunch.*, never under gen_ai.*."""
    tracer = trace.get_tracer("crunchobs")
    with tracer.start_as_current_span(name) as span:
        span.set_attribute(OPERATION, operation)
        if route is not None:
            span.set_attribute("crunch.route", route)
        if user_id is not None:
            span.set_attribute("crunch.user_id", user_id)
        yield span


def record_llm_usage(span, *, system: str, model: str,
                     input_tokens: int, output_tokens: int) -> None:
    """Stamp a chat span with the gen_ai.* usage attributes from a response.
    For the Anthropic SDK: input_tokens=resp.usage.input_tokens, etc."""
    span.set_attribute(SYSTEM, system)
    span.set_attribute(REQUEST_MODEL, model)
    span.set_attribute(INPUT_TOKENS, int(input_tokens))
    span.set_attribute(OUTPUT_TOKENS, int(output_tokens))


# TODO 1: add a `traced_messages_create(client, **kwargs)` helper that wraps an
#   Anthropic client.messages.create call: open a chat span, make the call with
#   thinking={"type": "adaptive"} (NEVER budget_tokens/temperature), then call
#   record_llm_usage(span, ...) from resp.usage. Return resp. One call, correctly
#   traced, every time.
```

> **The rule the package enforces:** no caller writes `gen_ai.*` strings by hand. They call `record_llm_usage(...)` and the names are correct by construction. If `grep -rn 'tokens_in\|input_token\b\|model_name' crunchobs` (outside `tracing.py`) hits anything, someone reintroduced the bespoke-name bug the conventions exist to kill.

---

## Deliverable 2 — `exporters.py` (dual export + dev/test modes)

The "export to both backends" milestone, behind one `init()`. One `TracerProvider`, two OTLP processors.

```python
"""crunchobs.exporters — one provider, fan out to Phoenix + Langfuse (+ dev/test)."""
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor, SimpleSpanProcessor, ConsoleSpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def init(service_name: str = "crunch-agents", *, mode: str = "dual"):
    """mode='dual' -> Phoenix + Langfuse; 'console' -> stdout; 'memory' -> test buffer."""
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    memory = None
    if mode == "dual":
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")))   # Phoenix
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint="http://localhost:3000/api/public/otel/v1/traces",
                headers={"Authorization": f"Basic {os.environ['LANGFUSE_OTLP_B64']}"})))
    elif mode == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif mode == "memory":
        memory = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory))
    else:
        raise ValueError(f"unknown mode: {mode}")

    trace.set_tracer_provider(provider)
    # TODO 2: when openinference/openllmetry is installed, auto-instrument the
    #   framework here (LangChainInstrumentor().instrument(tracer_provider=provider))
    #   so the caller's model/tool calls are traced with no per-call code.
    return provider, memory
```

The `memory` mode is what `test_tracing.py` uses to assert the emitted names with zero infrastructure; the `console` mode is the dev loop; `dual` is the milestone. One function, three audiences.

---

## Deliverable 3 — `accounting.py` (tokens, cost, latency)

The rollup + percentile layer from Exercise 3, packaged and tested.

```python
from collections import defaultdict
import numpy as np

PRICES = {  # 2026 Anthropic, USD/token (input, output); self-hosted open = $0
    "claude-opus-4-8":   (5.00 / 1e6, 25.00 / 1e6),
    "claude-sonnet-4-6": (3.00 / 1e6, 15.00 / 1e6),
    "claude-haiku-4-5":  (1.00 / 1e6,  5.00 / 1e6),
    "self-hosted-open":  (0.0,         0.0),
}


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    p_in, p_out = PRICES[model]
    return in_tok * p_in + out_tok * p_out


def rollup(spans, key: str) -> dict:
    """Group chat spans by `key`; sum input/output tokens + USD. Only chat spans
    have usage — skip the rest (an embeddings/tool span has no tokens to bill)."""
    out = defaultdict(lambda: {"in": 0, "out": 0, "usd": 0.0})
    for s in spans:
        if s.get("gen_ai.operation.name") != "chat":
            continue
        # TODO 3: read input/output tokens, accumulate into out[str(s[key])],
        #   and add cost_usd(s["gen_ai.request.model"], ...). (Port from Exercise 3.)
        ...
    return dict(out)


def percentiles(durations_ms) -> dict:
    if not durations_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
    arr = np.asarray(durations_ms, dtype=float)
    return {"p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)), "n": len(arr)}


def latency_per_step(spans) -> dict:
    """Group durations by step name and percentile each group."""
    by_step = defaultdict(list)
    for s in spans:
        by_step[s["name"]].append(float(s["duration_ms"]))
    return {step: percentiles(d) for step, d in by_step.items()}
```

---

## Deliverable 4 — `slo.py` (the SLO checker)

Wraps the percentiles into a budget + burn-rate verdict (Lecture 2 §1.2, §4.5).

```python
from crunchobs.accounting import percentiles


def slo_report(durations_ms, errors: int, total: int, *, p95_budget_ms: float,
               success_slo: float = 0.99, window_fraction: float = 1.0) -> dict:
    """Did the window meet its latency budget and stay inside the error budget?
    window_fraction: how far through the SLO window we are (for burn-rate)."""
    pcts = percentiles(durations_ms)
    success_rate = (total - errors) / total if total else 1.0
    error_budget = 1.0 - success_slo
    budget_used = (1.0 - success_rate) / error_budget if error_budget else 0.0
    # TODO 4: compute burn rate = budget_used / window_fraction, and a
    #   projected_exhaustion flag = burn_rate > 1.0 (on pace to blow the budget).
    return {
        "p95_ms": pcts["p95"],
        "p95_budget_ms": p95_budget_ms,
        "p95_ok": pcts["p95"] <= p95_budget_ms,
        "success_rate": success_rate,
        "error_budget_used_pct": round(100 * budget_used, 1),
        "slo_met": pcts["p95"] <= p95_budget_ms and success_rate >= success_slo,
    }
```

The non-negotiable: it alerts on **p95 vs budget and burn rate**, never on the mean. A worked check (`p95≈1030`, `error_budget_used_pct≈74.0`, `slo_met=True`) is in Lecture 2 §4.5 — `test_slo.py` should reproduce exactly those numbers.

---

## Deliverable 5 — `replay.py` (eval-on-traces)

Replay a recorded production trace through a new prompt version, holding model + inputs fixed (Lecture 2 §4, §4.1).

```python
import json, time
import anthropic

client = anthropic.Anthropic()


def load_recorded_spans(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["spans"]


def replay_one(span: dict, new_system_prompt: str) -> dict:
    attrs = span["attributes"]
    old_text = span["output_text"]
    # TODO 5: rebuild the input messages from the span's gen_ai.user.message
    #   events, then call client.messages.create with the SAME model
    #   (attrs["gen_ai.request.model"]), thinking={"type": "adaptive"}, and the
    #   new system prompt. Diff old vs new text AND output-token count.
    ...


def replay_recording(path: str, new_system_prompt: str) -> dict:
    spans = [s for s in load_recorded_spans(path)
             if s["attributes"].get("gen_ai.operation.name") == "chat"]
    results = [replay_one(s, new_system_prompt) for s in spans]
    changed = sum(1 for r in results if r["text_changed"])
    return {"replayed": len(results), "text_changed": changed,
            "text_unchanged": len(results) - changed, "results": results}
```

The validity rule: replay swaps *only* the prompt, against a *frozen recording* — same one-variable discipline as week 8's chunking A/B.

---

## Deliverable 6 — `cli.py` (the commands)

```bash
crunchobs accounting --recording recordings/sample_trace.json --by route
crunchobs slo        --recording recordings/sample_trace.json --p95-budget 1500 --slo 0.99
crunchobs replay     --recording recordings/sample_trace.json --prompt prompts/v2.txt
```

`accounting` prints the per-route/user/model cost table + per-step p50/p95/p99; `slo` prints the budget/burn-rate verdict; `replay` prints the old-vs-new diff summary. Each is a thin wrapper over Deliverables 3–5.

---

## Rules

- **You may** read the OTel docs, the Langfuse/Phoenix docs, the lecture notes, and your Exercise 2/3 code.
- **You must not** invent `gen_ai.*` attribute names — use the convention names verbatim (they live in `tracing.py` as constants; import them, don't retype them).
- **You must not** roll up non-chat spans into cost (an embeddings/tool span has no tokens to bill); `rollup` skips them.
- **You must not** alert on the mean latency — the SLO checker uses p95 and burn rate.
- **You must** keep `init(mode="memory")` working for tests so the package is verifiable with zero infrastructure.
- Any LLM call (replay, judge): model id `claude-opus-4-8` (or sonnet/haiku), Anthropic SDK `client.messages.create(...)`, `thinking={"type": "adaptive"}`, never `budget_tokens`/`temperature`; structured output via `output_config={"format": {...}}`, no assistant prefills.
- Python 3.12, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`, `numpy`, `anthropic`, plus `pytest`. Langfuse/Phoenix are runtime backends, not import-time hard deps.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-18-crunchobs-<yourhandle>`.
- [ ] `crunchobs.init(mode="dual")` brings up dual export; the *same* trace id appears in both Langfuse and Phoenix (verify in both UIs).
- [ ] `tracing.py` emits the real `gen_ai.*` attribute names (proven by `test_tracing.py` against an `InMemorySpanExporter`).
- [ ] `accounting.py` produces per-route/user/model cost tables (2026 prices) and per-step p50/p95/p99; non-chat spans are excluded from cost.
- [ ] `slo.py` reproduces the Lecture 2 §4.5 worked example exactly (`p95≈1030`, `error_budget_used_pct≈74.0`, `slo_met=True`) and reports burn rate.
- [ ] `replay.py` replays a recorded trace's generation through a new prompt, same model + inputs, diffing text and output tokens.
- [ ] `pytest` passes, with at least: `test_accounting` (rollup sums + percentile values on fixed spans), `test_slo` (error-budget + burn-rate math), `test_tracing` (the emitted names).
- [ ] `crunchobs accounting|slo|replay` all run against `recordings/sample_trace.json`.
- [ ] A `README.md` with the wiring instructions, the accounting/SLO output, and a one-paragraph note on how the capstone imports the package.
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Tracing & conventions** | 25 | `trace_agent`/`record_llm_usage` emit the real `gen_ai.*` names verbatim; operations are valid convention values; business dims under `crunch.*`; proven against InMemory. |
| **Dual export** | 20 | One `TracerProvider`, two OTLP processors → the same trace lands in both Langfuse and Phoenix; console/memory modes work for dev/test. |
| **Accounting** | 20 | Per-route/user/model cost (2026 prices, $0 for self-hosted); per-step p50/p95/p99; non-chat spans excluded from cost; correct on fixed spans. |
| **SLO checker** | 15 | p95-vs-budget, error budget, and burn rate; reproduces the §4.5 worked numbers; never alerts on the mean. |
| **Replay tool** | 15 | Eval-on-traces against a frozen recording; same model + inputs, only the prompt varies; diffs text AND output tokens. |
| **Docs & hygiene** | 5 | Clear README, no secrets committed, sensible commits, no `__pycache__`/`.venv` checked in. |

**90+** is portfolio-grade and drops straight into the capstone. **70–89** works but has a soft attribute name or a mean-based SLO. **Below 70** means the package isn't a safe, reusable observability layer — fix that first, because the capstone imports it.

---

## Stretch goals

- **Span links on replay.** Attach an OTel span link from each replay span back to its source production span, so an experiment is navigable from the real request it came from.
- **Tail-based sampling.** Add a sampler that keeps 100% of errored/slow traces and 5% of healthy ones; expose it through `init()`. Measure the storage drop on a recorded batch.
- **A `dashboards.py` helper.** Emit the three dashboards' group-bys as a JSON spec (route→cost, step→p95, time→precision) that you can POST to Langfuse's dashboard API — so the dashboards are version-controlled, not click-configured.
- **CI.** A GitHub Actions workflow that runs `pytest` (memory mode, no backend) and a headless `crunchobs accounting` on the sample recording. Green check on every push.

---

## How this connects to the rest of C23

- **Weeks 13 / 15 / 17 (Phase III)** built the system this package instruments — the supervisor, the MCP tools, the safety rails. `crunchobs` is what makes that system observable without editing its logic; you `init()` and decorate.
- **Week 19 (vLLM in production)** uses the *same* package to instrument a model server you own — the spans extend past the API boundary into queue time and cache hits, but the tracing/accounting/SLO layer is this one.
- **The capstone (weeks 22–24)** *requires* traces flowing to self-hosted Langfuse + Phoenix and the three dashboards; the rubric assumes `crunchobs` (or equivalent) is wired. The capstone doesn't re-teach observability — it depends on the package you build here.

When you've finished, push the repo and take the [quiz](../quiz.md).
