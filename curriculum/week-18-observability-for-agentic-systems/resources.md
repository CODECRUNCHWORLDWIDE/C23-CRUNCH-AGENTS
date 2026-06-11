# Week 18 — Resources

Every resource here is **free** or has a free tier. The whole observability stack this week — OpenTelemetry, self-hosted Langfuse, Arize Phoenix, the OpenInference and OpenLLMetry instrumentors — is open source and runs locally with no paid key. Phoenix's `px.launch_app()` needs *no* API key at all; self-hosted Langfuse runs from a `docker compose` file. The OTel Gen-AI semantic conventions are a public spec.

Tool names and SDK surfaces move every cohort — the *concepts* (spans, traces, the `gen_ai.*` conventions, dual export, p95/error budgets, eval-on-traces) are stable. When a specific page 404s, search the project's docs for the function or attribute name.

This week sits on top of Phase III (weeks 13/15/17). The system you instrument is your **LangGraph supervisor + MCP tools + safety rails**; the resources below assume you have that stack runnable.

## Required reading (work it into your week)

- **OpenTelemetry Gen-AI semantic conventions** — the canonical spec for `gen_ai.*` attributes, the operation types (`chat` / `embeddings` / `execute_tool` / `invoke_agent`), span kinds, and prompts/completions-as-events. Read it until the attribute names are muscle memory; emitting them verbatim is the whole portability story:
  <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
- **Langfuse documentation** — Dashboards, Scores, Datasets/Experiments, OTLP ingestion, the model-pricing tables, and self-hosting via Docker. Read the OTLP-ingest and Scores pages for the dashboards/replay legs of the lab:
  <https://langfuse.com/docs>
- **Arize Phoenix documentation** — local tracing (`px.launch_app()`), `phoenix.otel.register`, evaluations, and experiments. This is the zero-friction default for every exercise:
  <https://docs.arize.com/phoenix>
- **OpenInference instrumentation** — Arize's OTel-compatible instrumentors (`openinference-instrumentation-langchain`, `-anthropic`) that emit conventions-following spans with one call at startup. The native auto-instrumentation path for Phoenix:
  <https://github.com/Arize-ai/openinference>
- **OpenLLMetry (Traceloop)** — the other auto-instrumentation ecosystem: `traceloop-sdk` and the `opentelemetry-instrumentation-*` Gen-AI instrumentors. Same idea, Traceloop flavour:
  <https://github.com/traceloop/openllmetry>

## Tracing-tool references

- **OpenTelemetry Python SDK** — `TracerProvider`, `Resource`, `BatchSpanProcessor`/`SimpleSpanProcessor`, `ConsoleSpanExporter`, `OTLPSpanExporter`, and `start_as_current_span`. The plumbing every lab uses:
  <https://opentelemetry.io/docs/languages/python/>
- **OpenTelemetry traces data model** — spans, trace id, parent/child links, context propagation. Read this to understand *why* a trace reassembles into a tree from out-of-order spans:
  <https://opentelemetry.io/docs/concepts/signals/traces/>
- **LangSmith** — LangChain's hosted tracing + eval platform; deep LangGraph integration, polished UI. Proprietary and hosted (not self-hostable) — know it, but the milestone routes around it for data sovereignty:
  <https://docs.smith.langchain.com/>
- **Helicone** — open-source proxy-based LLM observability: repoint your model `base_url` and it logs cost/usage in one line. Lower effort, lower granularity (a proxy sees requests, not your in-app span tree):
  <https://www.helicone.ai/>

## Dashboards, metrics, and SLOs

- **Google SRE Book — Service Level Objectives** — the canonical treatment of SLOs and error budgets. Read this before you set a single latency budget:
  <https://sre.google/sre-book/service-level-objectives/>
- **Google SRE Workbook — Alerting on SLOs (burn rate)** — the multi-window burn-rate alerting pattern; the stretch goal builds a small version of this:
  <https://sre.google/workbook/alerting-on-slos/>
- **OpenTelemetry — recording exceptions on spans** — how OTel records a stack trace as a span event and sets `status=ERROR`; the basis of the "find the first red span" debugging step:
  <https://opentelemetry.io/docs/languages/python/instrumentation/#recording-exceptions>
- **Grafana — histograms and percentiles** — if you fan your OTLP metrics out to Prometheus/Grafana, this is how p95 panels are built; useful background even when you read p95 in Langfuse/Phoenix:
  <https://grafana.com/docs/grafana/latest/panels-visualizations/visualizations/histogram/>

## Papers and specs worth your time

- **OpenTelemetry Gen-AI semantic conventions (the spec itself)** — re-listed because it *is* the paper for this week; the attribute table is the thing you cite in the quiz answer key:
  <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
- **Dapper, a Large-Scale Distributed Systems Tracing Infrastructure** (Sigelman et al., Google, 2010) — the origin of the span/trace model every modern tracer (including OTel) descends from. Read it once to see where "trace id + parent span id" came from:
  <https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/>
- **The Tail at Scale** (Dean & Barroso, CACM 2013) — *the* paper on why the tail (p99) dominates user experience in fan-out systems. An agent run is a fan-out; this is why you SLO against p95/p99, not the mean:
  <https://research.google/pubs/the-tail-at-scale/>

## Models you'll use this week

Observability is mostly *not* an LLM task — you instrument, aggregate, and chart, none of which calls a model. The exceptions are the two judge-style calls, kept minimal:

- **`claude-opus-4-8`** — used only as an **LLM-as-judge** in (a) the retrieval-precision eval that feeds Dashboard 3, and (b) the quality verdict in the eval-on-traces replay. Call it via the Anthropic SDK `client.messages.create(...)` with `thinking={"type": "adaptive"}`; never `budget_tokens` or `temperature`. Structured judgments use `output_config={"format": {...}}`. The exercises avoid even this where possible (Exercise 2 and 3 make zero LLM calls):
  <https://docs.claude.com/en/api/>

## Tools you'll use this week

- **`opentelemetry-sdk`** — `pip install opentelemetry-sdk`. `TracerProvider`, span processors, `ConsoleSpanExporter`, `InMemorySpanExporter`. The only hard dependency of Exercise 2.
- **`opentelemetry-exporter-otlp-proto-http`** — `pip install opentelemetry-exporter-otlp-proto-http`. The `OTLPSpanExporter` that ships spans to Langfuse and Phoenix over OTLP/HTTP.
- **`arize-phoenix`** — `pip install arize-phoenix`. `import phoenix as px; px.launch_app()` for a local UI + collector with **zero API key**; `phoenix.otel.register()` to wire OTel → Phoenix.
- **`langfuse`** — `pip install langfuse`. The SDK + OTLP endpoint for the self-hosted product-analytics backend; Scores and Datasets for Dashboard 3 and replay.
- **`openinference-instrumentation-langchain`** — `pip install openinference-instrumentation-langchain`. Auto-instruments your LangGraph/LangChain calls into conventions-following spans (the Phoenix-native path).
- **`traceloop-sdk` (OpenLLMetry)** — `pip install traceloop-sdk`. The alternative auto-instrumentation path; the same spans, Traceloop flavour.
- **`numpy`** — percentiles for the SLO math (`np.percentile`). The only dependency of Exercise 3 beyond stdlib.

## A note on the system-under-test

The exercises and mini-project instrument the **Phase III multi-agent stack** you built in weeks 13/15/17 — the LangGraph supervisor routing to worker agents (`retriever`, `writer`) over an MCP tool surface, with safety rails. Exercises 2 and 3 *simulate* that stack (a synthetic plan → retrieve → generate run, and a list of recorded spans) so they run with zero infrastructure and no key; the **challenge and the mini-project instrument the real thing**. The trace tree you read all week is your week-13 graph's execution, made visible.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Trace** | The whole tree of spans for one request, tied together by a shared trace id. The only artifact that survives the run. |
| **Span** | One unit of work — start time, end time, name, parent, attributes, events, status. "The writer's model call" is a span. |
| **Span kind** | `CLIENT` (calling out), `SERVER` (handling an inbound call), `INTERNAL` (in-process). Tells "I called the model" from "I answered a tool call." |
| **`gen_ai.*` attributes** | The OTel Gen-AI convention names for LLM/agent spans (`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`/`output_tokens`). Emit verbatim. |
| **Operation type** | `gen_ai.operation.name`: `chat` / `embeddings` / `execute_tool` / `invoke_agent`. Lets a backend colour the tree by what each span did. |
| **OTLP** | OpenTelemetry Protocol — the wire format (over HTTP/gRPC) both Langfuse and Phoenix ingest; one exporter config fans out to both. |
| **Exporter** | The object that ships finished spans somewhere: `ConsoleSpanExporter` (stdout), `InMemorySpanExporter` (a test buffer), `OTLPSpanExporter` (a backend). |
| **Context propagation** | Carrying the active span across calls/processes so child spans nest under their parent. Get it wrong and tool spans orphan into a separate trace. |
| **p95** | The latency 95% of requests beat — the unlucky-user experience, and the number you SLO against. Not the mean (which the tail poisons). |
| **SLO** | Service Level Objective — a reliability target over a window ("99% succeed and land under the p95 budget over 30 days"). |
| **Error budget** | `100% − SLO` — the failures you're *allowed*. Reframes reliability from "zero errors" to "spend the budget; watch the burn rate." |
| **Eval-on-traces** | Replaying a recorded production trace's exact inputs through a new prompt version (offline), diffing output and metrics — testing on real traffic, not a synthetic set. |
| **Dual export** | One `TracerProvider` with two OTLP processors → every span goes to Langfuse *and* Phoenix at once. The milestone's "export to both" is this, not two apps. |

---

*If a link 404s, please open an issue so we can replace it.*
