# Challenge 1 — Instrument the Stack and Find the Failure

**Time estimate:** ~150 minutes.

## Problem statement

You have a working multi-agent system from Phase III — a LangGraph supervisor routing to worker agents (`retriever`, `writer`) over an MCP tool surface, with safety rails. It runs. But right now it is a closed box: when it misbehaves you have nothing but the final answer. This challenge re-opens the box.

You will instrument the whole stack with the OTel Gen-AI conventions, **dual-export every span to BOTH self-hosted Langfuse AND Arize Phoenix at once**, build the three dashboards, then inject *one* synthetic failure into a run and **find the failing step from a dashboard in under five minutes** — without adding a single `print`. The deliverable is the milestone: an instrumented system, two backends receiving the same traces, three live dashboards, and a recorded "found it" walkthrough.

This is the Phase III observability milestone in lab form. The output is operational capability: traces flowing, dashboards live, and a demonstrated five-minute path from "something's wrong" to "*this* span, *this* reason."

## What is fixed (do not let these vary)

- **The system-under-test:** your real week-13 supervisor + week-15 MCP tools + week-17 safety rails. You instrument it; you do not rewrite it.
- **The conventions:** the OTel Gen-AI `gen_ai.*` attribute names, verbatim (Lecture 1 §3). Business dimensions go under your own `crunch.*` namespace, never under `gen_ai.*`.
- **Both backends:** self-hosted Langfuse (Docker) *and* self-hosted Phoenix (`px.launch_app()` / `phoenix serve`). The dual export is one `TracerProvider` with two OTLP processors — not two instrumented apps.
- **The three dashboards:** token usage per route, p95 latency per agent step, retrieval-precision over time. Build each in both backends.

## The harness approach

The whole thing is: instrument once, fan out to both backends, then read.

```python
# obs_setup.py — dual export: every span -> Langfuse AND Phoenix, one provider.
import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from openinference.instrumentation.langchain import LangChainInstrumentor


def init_dual_export(service_name: str = "crunch-agents") -> trace.Tracer:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))

    # Backend 1 — Phoenix (local OTLP collector, no key).
    provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")))

    # Backend 2 — self-hosted Langfuse (OTLP endpoint + basic-auth header).
    provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="http://localhost:3000/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {os.environ['LANGFUSE_OTLP_B64']}"})))

    trace.set_tracer_provider(provider)
    # Auto-instrument the framework calls; manual crunch.* spans wrap the nodes.
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return trace.get_tracer(service_name)
```

Then enrich the application layer the auto-instrumentor can't see — the slicing dimensions your dashboards group by (Lecture 1 §5.2):

```python
tracer = init_dual_export()

def supervisor_node(state: dict) -> dict:
    with tracer.start_as_current_span("supervisor.route") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "supervisor")
        span.set_attribute("crunch.route", state["route"])       # dashboard #1 group-by
        span.set_attribute("crunch.user_id", state["user_id"])
        return {**state, "next": decide_route(state)}
```

For **Dashboard 3** (retrieval precision), attach an eval score to each retrieval span: have the retriever record its returned chunk ids, score them against ground truth (or an LLM-judge with `claude-opus-4-8`), and write `crunch.retrieval_precision` onto the span — then chart it via Langfuse **Scores** / Phoenix **evaluations** (Lecture 2 §2.3).

## Acceptance criteria

- [ ] A `challenge-01/` directory with a runnable harness that instruments the Phase III stack and **dual-exports to both Langfuse and Phoenix** (one provider, two OTLP processors — verify the *same* trace id appears in both UIs).
- [ ] Every LLM-call span carries the real `gen_ai.*` attributes (`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`/`output_tokens`); every agent/tool span carries the right operation name; business dimensions live under `crunch.*`.
- [ ] **Dashboard 1 — token usage per route:** grouped by `crunch.route`, summing `gen_ai.usage.*`, priced with 2026 Anthropic rates, in both backends.
- [ ] **Dashboard 2 — p95 latency per agent step:** grouped by step, the **p95** (not the mean), with the budget line, in both backends.
- [ ] **Dashboard 3 — retrieval-precision over time:** a `crunch.retrieval_precision` score attached per retrieval span, charted over time, in both backends.
- [ ] You inject **exactly one** synthetic failure, then produce a recorded walkthrough (a short note or screen recording) that finds the failing step **from a dashboard in under five minutes**, naming the span and the reason, in the "found it in the trace in under 5 minutes" promise format.

## The trap (read after a first attempt)

The trap is **instrumenting only the top-level call**, so the failing sub-span is invisible. If you trace just `agent.invoke` and not the nested `retriever` / `vector_search` / `writer` spans, the trace is a single bar — it tells you the *request* was slow or wrong but not *which step*, and you're back to guessing. The whole value of the tree is that the failing step is a *child* span you can point at. Instrument every node and every tool call (and propagate context across the MCP boundary, Lecture 1 §5.3) so the tree has the branches that contain the failure.

A second, subtler trap: **charting the mean latency instead of p95**, so the tail spike hides. If Dashboard 2 plots the *average* per step, a step that's normally 200ms with an occasional 9-second stall shows an unremarkable ~700ms average — the tail that's actually hurting users is averaged into invisibility (Lecture 2 §1.1). The injected failure can be a tail-latency spike *specifically* to punish this mistake: a mean chart won't catch it, a p95 chart will jump. Plot the percentile, not the average, or the failure you injected will hide from the dashboard meant to catch it.

## Stretch goals

- **Span-link the failure to its replay.** When you find the failing generation, replay it through a fixed prompt (eval-on-traces, Lecture 2 §4) and attach an OTel **span link** from the replay back to the original — so "the fix" is navigable from "the failure."
- **Tail-based sampling.** Add a sampler that keeps 100% of errored/slow traces and 5% of healthy ones. Confirm the injected failure still lands (it's slow/errored, so it's kept) while storage drops.
- **Burn-rate alert.** Wire a simple alert on the error-budget *burn rate* for the failing route, not a raw threshold — fire when the projected exhaustion is before the window ends (Lecture 2 §1.2).
- **Three failure classes.** Inject one of each — an exception (red span), a tail-latency spike (slow span), and a silent empty-retrieval (wrong answer, all green) — and time yourself finding each with the §3 decision tree. The silent one is the hard one; that's the point.

## Why this matters

This is the Phase III milestone, and it is load-bearing for the capstone. In weeks 22–24 you build and defend a production agent, and the rubric assumes its traces already flow to self-hosted Langfuse and Phoenix with these exact dashboards — the capstone doesn't re-teach observability, it *depends* on it. More immediately: every agent you ship after this will break in production, and the only question that matters at 3am is "can you find the failing step before the user gives up?" This challenge is that moment, rehearsed: you instrumented the tree, you have the dashboards, and you found the injected failure in under five minutes. Do it once here under no pressure so you can do it for real under all of it. You re-opened the box the easy way — and you can prove it.
