# Lecture 1 — Tracing an Agent: Spans, Traces, and the OpenTelemetry Gen-AI Conventions

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can explain why an agent without traces is a closed box, read a multi-agent run as a tree of spans, name the OpenTelemetry Gen-AI semantic-convention attributes (`gen_ai.*`) and the operation types they describe, wire the OTel SDK + an OTLP exporter into a LangGraph supervisor (both via auto-instrumentation and by hand), and roll up token usage per user / per route / per model straight from span attributes.

If you remember one sentence from this entire week, remember this one:

> **An agentic system without traces is a closed box. You will eventually re-open it the hard way.**

There's a corollary worth taping next to it:

> **A trace is the only artifact that survives the run.** The LLM call is gone, the tool result is gone, the supervisor's routing decision is gone — unless a span recorded it. When production breaks at 3am, the trace is what you have, or you have nothing.

Through Phase III you built a multi-agent system: a LangGraph supervisor (week 13) routing to worker agents over an MCP tool surface (week 15), with safety rails (week 17). It works on your laptop. This week you make it *observable* — so that when it misbehaves in front of a user, you can see exactly which step failed, how many tokens it burned, and how long it took, without adding a single `print`. Everything that follows is in service of one measurable question: **when a multi-agent run goes wrong, can you find the failing step from a trace in under five minutes?**

---

## 1. Why trace an agent at all

You don't *have* to trace. You could log. You could `print(state)` between nodes. Three hard realities say that isn't enough for an agentic system.

**Reality 1 — an agent run is a tree, not a line.** A single user request to a supervisor graph fans out: the supervisor calls the model to decide a route, routes to a `retriever` agent, which calls an embedding model and a vector store, then routes to a `writer` agent, which calls the model again, maybe twice, maybe with a tool call in between. That's a *tree* of operations with timing, parent/child structure, and causality. A flat log file flattens the tree into interleaved lines and throws the structure away. You cannot reconstruct "the retrieve step inside the second supervisor turn took 4 seconds" from `grep`. A trace keeps the tree.

**Reality 2 — the interesting failures are *between* steps.** Single-LLM bugs are easy: bad output, you see it. Agentic bugs hide in the handoffs — the supervisor routed to the wrong agent, the tool returned an error the agent silently swallowed, retrieval came back empty so generation hallucinated, a step that's normally 200ms spiked to 9 seconds and blew the latency budget. None of these is visible in the *final answer*. They're visible only if every step recorded what it did, what it received, and how long it took. That recording is a span.

**Reality 3 — you cannot bill, budget, or debug what you don't measure.** Token usage per user (for cost allocation), latency per agent step (for SLOs), error rate per tool (for reliability) — these are aggregate questions you answer by querying many traces. If the data isn't on the spans, the questions are unanswerable. "How much does the `summarize` route cost per call?" has an exact answer the moment your spans carry `gen_ai.usage.input_tokens` and a `route` attribute; without them it's a guess.

So you trace. And the vocabulary for tracing — the thing that makes a trace from your LangGraph app readable by Langfuse *and* Phoenix *and* a Grafana dashboard — is OpenTelemetry.

> **The spine of the week:** OpenTelemetry (OTel) is the vendor-neutral standard for traces, metrics, and logs. The **Gen-AI semantic conventions** are OTel's agreed-upon names for the attributes an LLM/agent span should carry. Emit those names, and every observability tool understands your traces for free. Invent your own names, and you're locked into one vendor and one dashboard forever.

---

## 2. Spans and traces — the trace tree of a multi-agent run

Three nouns, and they nest.

- A **span** is one unit of work with a start time, an end time, a name, a parent, and a bag of attributes. "The supervisor's model call" is a span. "The vector-store query" is a span. A span can also carry **events** (timestamped points inside it — "prompt sent", "first token") and a **status** (`OK` or `ERROR`).
- A **trace** is the whole tree of spans for one request, tied together by a shared **trace id**. Every span also has a **span id** and a **parent span id**; the root span has no parent. The parent/child links *are* the tree.
- **Context propagation** is how a child span learns its parent: the active span is carried in a context object, and when you start a new span inside it, OTel wires the parent link automatically. Get propagation wrong and your tree falls apart into a flat list of orphan spans — the single most common instrumentation bug.

Here is one run of a supervisor graph, drawn as the trace tree you want to see in your dashboard:

```
agent.invoke  (trace_id=a1b2…, 5.8s)                          ← root span
├─ chat supervisor  (gen_ai.operation.name=chat, 0.9s)        ← route decision
├─ agent retriever  (gen_ai.operation.name=invoke_agent, 1.4s)
│   ├─ embeddings query  (gen_ai.operation.name=embeddings, 0.2s)
│   └─ vector_search  (db.system=pgvector, 0.3s)              ← tool / store span
├─ agent writer  (gen_ai.operation.name=invoke_agent, 3.1s)
│   ├─ execute_tool get_policy  (gen_ai.operation.name=execute_tool, 0.4s)
│   └─ chat writer  (gen_ai.operation.name=chat, 2.5s)        ← the slow span
└─ (root ends)
```

Read it like a flame graph. Indentation is parent/child. The number on each line is duration. Your eye goes straight to `chat writer` at 2.5s — that's where the time went. If this run were *slow*, that span is your first suspect, and you didn't read a single log line to find it. **That is the whole value of a trace tree: it turns "the request was slow" into "the writer's model call was slow" in one glance.** Lecture 2 turns that glance into a decision tree you can run under five minutes.

Two structural facts to internalize now:

- **Every span shares the trace id; each has its own span id and a parent span id.** That's what lets a backend reassemble the tree from spans that arrive out of order, over the network, from different processes.
- **Spans nest across process boundaries** if you propagate context. The supervisor calls a tool that lives in a separate MCP server (week 15); inject the trace context into that call and the tool's spans appear *under* the agent's span, in one trace. Don't, and the tool gets its own orphan trace and the connection is lost.

---

## 3. The OpenTelemetry Gen-AI semantic conventions, in depth

A span is just a name plus attributes. The convention is *which* attribute names to use so the data is portable. The OTel **Gen-AI semantic conventions** (`https://opentelemetry.io/docs/specs/semconv/gen-ai/`) define exactly that for LLM and agent spans. They are the spine of this week, and they are worth memorizing.

### 3.1 The core `gen_ai.*` attributes

Every LLM-call span should carry these. Names are exact — emit them verbatim.

| Attribute | Meaning | Example |
|---|---|---|
| `gen_ai.system` | The provider/system the request goes to | `anthropic`, `openai`, `vllm` |
| `gen_ai.operation.name` | What kind of operation this span is | `chat`, `embeddings`, `execute_tool`, `invoke_agent` |
| `gen_ai.request.model` | The model you *asked* for | `claude-opus-4-8` |
| `gen_ai.response.model` | The model that actually answered (may differ) | `claude-opus-4-8` |
| `gen_ai.usage.input_tokens` | Prompt tokens billed | `1843` |
| `gen_ai.usage.output_tokens` | Completion tokens billed | `412` |
| `gen_ai.response.finish_reasons` | Why generation stopped | `["end_turn"]` |
| `gen_ai.request.max_tokens` | The output cap you set | `16000` |

The two you will use the most are the token-usage pair. **`gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` are the foundation of all cost and budget accounting in this week.** They come straight off the provider's usage object — for the Anthropic SDK, `response.usage.input_tokens` and `response.usage.output_tokens` — and you copy them onto the span. Every "tokens per route / per user / per model" number is a sum over these two attributes, filtered by the metadata you also put on the span (§6).

> **The naming rule that saves you:** the conventions namespace everything under `gen_ai.`. Sub-namespaces group by concern: `gen_ai.request.*` is what you sent, `gen_ai.response.*` is what came back, `gen_ai.usage.*` is the billing counters. If you find yourself inventing `tokens_in` or `model_name`, stop — there's a `gen_ai.*` name for it, and using yours breaks every dashboard that expects the standard.

### 3.2 The operation types — `gen_ai.operation.name`

The convention enumerates the kinds of Gen-AI operations, and `gen_ai.operation.name` tells the backend which one a span is. The ones that matter for an agentic system:

- **`chat`** — a chat-completion call to a model. The supervisor's routing call, the writer's generation call.
- **`embeddings`** — an embedding call (the retriever embedding a query).
- **`execute_tool`** — a tool/function execution (an MCP tool call). Pairs with `gen_ai.tool.name`.
- **`invoke_agent`** — an agent invocation: a whole worker agent doing its job. Pairs with `gen_ai.agent.name`.

These four let a backend *colour the tree by operation*: all `chat` spans one way, all `execute_tool` spans another, agents as containers. It's why Langfuse and Phoenix can render a LangGraph run as a readable agent trace and not a wall of generic spans — they read `gen_ai.operation.name` and lay it out accordingly. The agent- and tool-specific attributes:

| Attribute | On which operation | Meaning |
|---|---|---|
| `gen_ai.agent.name` | `invoke_agent` | The worker agent's name (`retriever`, `writer`) |
| `gen_ai.agent.id` | `invoke_agent` | A stable agent identifier |
| `gen_ai.tool.name` | `execute_tool` | The tool being called (`get_policy`) |
| `gen_ai.tool.call.id` | `execute_tool` | The tool-call id, to correlate request and result |

### 3.3 Span kinds

OTel also tags each span with a **span kind**, orthogonal to the gen_ai operation. The kinds you'll set:

- **`CLIENT`** — your process is calling *out* to something (the model API, the vector store). LLM-call spans are `CLIENT`.
- **`INTERNAL`** — in-process work with no remote call (an agent step that's pure orchestration). `invoke_agent` containers are often `INTERNAL`.
- **`SERVER`** — your process is *handling* an inbound request (an MCP tool server receiving a call). The tool-side span is `SERVER`; the agent-side `execute_tool` span is `CLIENT`. Those two, linked by propagated context, are the two halves of one tool call across a process boundary.

You usually don't agonize over span kind — the auto-instrumentors set it. But knowing it exists explains why a backend can tell "I called the model" from "I answered a tool call."

### 3.4 Capturing prompts and completions as span events

Attributes are for small, queryable, low-cardinality values (a model name, a token count). The **prompt and the completion are large free text** — you don't want them as attributes (they'd bloat every span and tank query performance). The convention's answer: record them as **span events**, timestamped log records attached to the span.

The relevant event names:

- **`gen_ai.user.message`** / **`gen_ai.system.message`** — input messages.
- **`gen_ai.assistant.message`** — the model's output message.
- **`gen_ai.choice`** — the generated choice (the completion).

Each event carries the message content (and role) as event attributes. This is exactly what lets Langfuse show you the *actual prompt and response* for a span when you click into it — the events are the message bodies, the attributes are the metadata.

> **The privacy lever you must know about.** Capturing message content means PII flows into your tracing backend. The conventions and every instrumentor gate this behind a switch — typically the env var `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`. **Off by default for a reason.** In week 17 you learned the safety surface; here's its observability cousin: decide deliberately whether prompt/response bodies belong in your traces, and if they do, make sure your backend (self-hosted Langfuse, self-hosted Phoenix) is inside your trust boundary. We default it *on* in the local labs (the data is synthetic) and call out where you'd turn it *off* in production.

---

## 4. Wiring the OTel SDK and an OTLP exporter

Now the plumbing. To produce traces you need three objects, set up once at process start:

1. A **`TracerProvider`** — the factory that makes tracers and owns the export pipeline. It also holds a **`Resource`**: process-level attributes (`service.name`, version) stamped on every span so you can tell *which service* a trace came from.
2. A **span processor + exporter** — the processor batches finished spans and hands them to the exporter; the exporter ships them somewhere. The universal exporter is **OTLP** (OpenTelemetry Protocol) over HTTP or gRPC. Both Langfuse and Phoenix speak OTLP, which is why one exporter config can fan out to both.
3. A **tracer** — what your code calls to start spans.

Here is the canonical setup. It exports to the console (zero infrastructure, so it runs anywhere) *and* shows where the OTLP endpoint slots in:

```python
"""otel_setup.py — stand up an OTel tracer that exports to console + OTLP."""
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)


def init_tracing(service_name: str = "crunch-agents") -> trace.Tracer:
    # The Resource stamps every span with which service emitted it.
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # 1) Console exporter — prints spans; needs no backend, great for dev.
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # 2) OTLP exporter — ships spans to Langfuse / Phoenix over OTLP/HTTP.
    #    endpoint + headers come from env (OTEL_EXPORTER_OTLP_ENDPOINT, etc.)
    #    or are passed explicitly; this is the ONE line that fans out to a backend.
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter())  # reads OTEL_EXPORTER_OTLP_* env
    )

    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)
```

Pointing that OTLP exporter at a backend is pure configuration — environment variables, no code change:

```bash
# Self-hosted Phoenix (default OTLP collector on 6006):
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:6006/v1/traces"

# Self-hosted Langfuse exposes an OTLP endpoint; auth via a header.
# (Langfuse's OTLP path + the base64 public:secret key — see resources.md.)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:3000/api/public/otel/v1/traces"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64 pk:sk>"
```

> **The "dual export" trick the challenge depends on.** A `TracerProvider` can hold *multiple* span processors. Add one `BatchSpanProcessor(OTLPSpanExporter(endpoint=phoenix))` **and** one `BatchSpanProcessor(OTLPSpanExporter(endpoint=langfuse))` and every span goes to *both* backends at once. That's how the milestone — "export to self-hosted Langfuse AND Arize Phoenix" — is one provider with two processors, not two instrumented apps.

Manually creating spans uses the tracer as a context manager. The pattern you'll repeat all week:

```python
tracer = init_tracing()

with tracer.start_as_current_span("chat supervisor") as span:
    span.set_attribute("gen_ai.system", "anthropic")
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.request.model", "claude-opus-4-8")
    # ... make the model call ...
    span.set_attribute("gen_ai.usage.input_tokens", resp.usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", resp.usage.output_tokens)
```

`start_as_current_span` does two crucial things: it makes the new span the *current* one (so any span started inside it becomes its child — automatic tree-building), and it ends the span when the `with` block exits (recording the duration). Nest these and you get the tree from §2 for free.

---

## 5. Instrumenting a LangGraph supervisor

You have two paths, and a mature setup uses both: **auto-instrumentation** for the LLM/framework calls you didn't write, and **manual spans** for the orchestration logic you did.

### 5.1 Auto-instrumentation (openinference / OpenLLMetry)

You do not hand-write spans for every Anthropic call. Two ecosystems wrap the SDKs and emit gen_ai spans automatically:

- **OpenInference** (Arize) — instrumentors like `openinference-instrumentation-langchain` and `openinference-instrumentation-anthropic` that patch the libraries to emit spans following the conventions. This is the native path for Phoenix.
- **OpenLLMetry** (Traceloop) — `traceloop-sdk` / the `opentelemetry-instrumentation-*` Gen-AI instrumentors, the same idea with a Traceloop flavour.

The point of both: **one call at startup, and every model/embedding/tool call your framework makes is traced with the right `gen_ai.*` attributes, no per-call code.** With Phoenix and OpenInference it's about this small:

```python
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

px.launch_app()                       # local Phoenix UI + OTLP collector, no key
tracer_provider = register()          # wires OTel → Phoenix
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# From here, every LangGraph/LangChain model & tool call emits gen_ai spans
# into Phoenix automatically. You wrote zero span code.
```

`register()` returns a configured provider; you can pass it to OpenLLMetry instrumentors too, or add your own OTLP processors to it for the dual-export to Langfuse. The framework instrumentor sees the supervisor graph run and produces the `chat` / `invoke_agent` / `execute_tool` spans of §2.

### 5.2 Manual spans for the parts the framework can't see

Auto-instrumentation traces the *calls*. It does not know your *business* structure — which user, which route, which tenant, the supervisor's routing *decision*. Those you add with manual spans (or by enriching the current span). Wrap each supervisor node:

```python
from opentelemetry import trace

tracer = trace.get_tracer("crunch-agents")


def supervisor_node(state: dict) -> dict:
    # A manual span around the routing decision, as an agent operation.
    with tracer.start_as_current_span("supervisor.route") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "supervisor")
        # Business metadata the auto-instrumentor can't know — the keys you
        # will GROUP BY in the dashboards (route, user, tenant).
        span.set_attribute("crunch.route", state["route"])
        span.set_attribute("crunch.user_id", state["user_id"])

        decision = decide_route(state)        # your routing logic (may call the LLM)
        span.set_attribute("crunch.routed_to", decision)
        return {**state, "next": decision}
```

Notice the `crunch.*` namespace: **business attributes you add live under your own namespace, never under `gen_ai.*`.** `gen_ai.*` is reserved for the convention; `crunch.route` is yours. Keep them separate and your dashboards stay clean — `gen_ai.*` for the model metrics, `crunch.*` for the slicing dimensions.

> **The two-layer mental model:** auto-instrumentation gives you the *model* layer of the trace (every LLM/tool call, correctly attributed) for free; manual spans give you the *application* layer (which user, which route, your routing decisions) that no library can infer. The good traces are the union — a `chat` span from the instrumentor, sitting under a `supervisor.route` span you added, carrying your `crunch.route` so the dashboard can slice by it.

### 5.3 Propagating context to the MCP tool surface

Week 15's tools live behind MCP, often in a separate process. To keep their spans *inside* the agent's trace, inject the trace context into the outbound call and extract it on the tool side:

```python
from opentelemetry.propagate import inject, extract

# Agent side (CLIENT): inject the current trace context into the tool request.
carrier = {}
inject(carrier)                       # writes traceparent into `carrier`
tool_request_headers.update(carrier)  # ride along with the MCP call

# Tool side (SERVER): extract it so the tool's span links to the agent's span.
ctx = extract(incoming_headers)
with tracer.start_as_current_span("execute_tool get_policy", context=ctx) as span:
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", "get_policy")
```

Get this right and the tool's `SERVER` span nests under the agent's `CLIENT` span in one trace. Get it wrong and the tool spawns a separate orphan trace — the classic "my tool calls don't show up in the agent trace" bug, and it's always a propagation problem.

---

## 6. Token accounting from span attributes — per user, per route, per model

Here is the payoff of putting `gen_ai.usage.*` and your `crunch.*` metadata on every span: **all token and cost accounting is a group-by over your spans.** You don't build a separate billing pipeline; you query the traces you already have.

The mental model is a table with one row per LLM-call span:

| span | crunch.user_id | crunch.route | gen_ai.request.model | input_tokens | output_tokens |
|---|---|---|---|---:|---:|
| chat supervisor | u_17 | summarize | claude-haiku-4-5 | 612 | 28 |
| chat writer | u_17 | summarize | claude-opus-4-8 | 1843 | 412 |
| chat supervisor | u_42 | search | claude-haiku-4-5 | 590 | 25 |

Roll it up however you need:

- **Per route:** `GROUP BY crunch.route` → sum tokens → the cost of each route.
- **Per user:** `GROUP BY crunch.user_id` → who's expensive (and who to bill).
- **Per model:** `GROUP BY gen_ai.request.model` → spend split across Opus / Sonnet / Haiku.

Cost is tokens times the model's price. Using real 2026 Anthropic prices (per million tokens, input / output):

| Model | Input $/M | Output $/M |
|---|---:|---:|
| `claude-opus-4-8` | $5.00 | $25.00 |
| `claude-sonnet-4-6` | $3.00 | $15.00 |
| `claude-haiku-4-5` | $1.00 | $5.00 |

```python
PRICES = {  # USD per token (input, output) — 2026 Anthropic pricing / 1e6
    "claude-opus-4-8":   (5.00 / 1e6, 25.00 / 1e6),
    "claude-sonnet-4-6": (3.00 / 1e6, 15.00 / 1e6),
    "claude-haiku-4-5":  (1.00 / 1e6,  5.00 / 1e6),
}


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    p_in, p_out = PRICES[model]
    return in_tok * p_in + out_tok * p_out


def rollup(spans, key: str) -> dict[str, dict]:
    """Group LLM-call spans by an attribute key and sum tokens + cost."""
    out: dict[str, dict] = {}
    for s in spans:
        a = s.attributes
        if a.get("gen_ai.operation.name") != "chat":
            continue                                   # only count model calls
        bucket = out.setdefault(str(a[key]), {"in": 0, "out": 0, "usd": 0.0})
        in_tok = int(a.get("gen_ai.usage.input_tokens", 0))
        out_tok = int(a.get("gen_ai.usage.output_tokens", 0))
        bucket["in"] += in_tok
        bucket["out"] += out_tok
        bucket["usd"] += cost_usd(str(a["gen_ai.request.model"]), in_tok, out_tok)
    return out
```

`rollup(spans, "crunch.route")` gives you token-usage-per-route — **dashboard #1 of the lab**. `rollup(spans, "crunch.user_id")` gives per-user. `rollup(spans, "gen_ai.request.model")` gives per-model. Same function, different group-by key, because the dimensions were on the span all along.

> **The discipline:** put the slicing dimension on the span *at emit time*. You cannot group by a `route` you never recorded. The moment a span lacks `crunch.route`, that call is invisible to your per-route dashboard — and you only discover the gap when the number looks too low. Decide the dimensions you'll need (user, route, model, tenant) and stamp them on every relevant span from day one.

For self-hosted Langfuse and Phoenix, you usually don't even write this loop in production — the backends compute token and cost rollups *for you* from the `gen_ai.usage.*` attributes (Langfuse has model-pricing tables; Phoenix aggregates usage). The loop above is the *mechanism* they implement, and Exercise 3 has you build it by hand so you understand exactly what the dashboard is doing — the same way week 8 had you build the metric before trusting the harness.

---

## 7. Recap

You should now be able to:

- Make the **closed-box argument**: an agent run is a tree, the interesting failures live between steps, and you cannot bill/budget/debug what you didn't record — so you trace, and the trace is the only artifact that survives the run.
- Read a **trace tree**: spans nest by parent/child under a shared trace id, indentation-plus-duration reads like a flame graph, and your eye goes to the slow or errored span without touching a log.
- Name the **OTel Gen-AI conventions**: the `gen_ai.*` attributes (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`/`output_tokens`, `gen_ai.operation.name`), the operation types (`chat` / `embeddings` / `execute_tool` / `invoke_agent`), span kinds (`CLIENT`/`INTERNAL`/`SERVER`), and capturing prompts/completions as span *events* (`gen_ai.user.message`, `gen_ai.choice`) behind the content-capture switch.
- Wire the **OTel SDK + OTLP exporter**: a `TracerProvider` with a `Resource`, batch processors, and the dual-export trick (one provider, two OTLP processors → Langfuse *and* Phoenix).
- Instrument a **LangGraph supervisor** both ways: auto-instrumentation (openinference / OpenLLMetry) for the model/tool calls, manual spans for the application layer (`crunch.route`, `crunch.user_id`), and context propagation so MCP tool spans nest in the agent trace instead of orphaning.
- Roll up **token usage per user / route / model** as a group-by over span attributes, priced with real 2026 Anthropic rates.

Next: how to turn that span data into latency SLOs and three live dashboards, and how to walk a broken trace to the failing step in under five minutes. Continue to [Lecture 2 — Dashboards, SLOs, and Trace-Driven Debugging](./02-dashboards-slos-and-trace-driven-debugging.md).

---

## References

- *OpenTelemetry Gen-AI semantic conventions* (the `gen_ai.*` attribute spec, operation types, events): <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
- *OpenTelemetry traces data model* (spans, trace id, context propagation): <https://opentelemetry.io/docs/concepts/signals/traces/>
- *OpenTelemetry Python SDK* (`TracerProvider`, span processors, OTLP exporter): <https://opentelemetry.io/docs/languages/python/>
- *Arize Phoenix — OpenInference instrumentation* (`phoenix.otel.register`, `px.launch_app`): <https://docs.arize.com/phoenix>
- *OpenLLMetry / Traceloop* (Gen-AI auto-instrumentation): <https://github.com/traceloop/openllmetry>
- *Langfuse — OpenTelemetry / OTLP ingestion*: <https://langfuse.com/docs>
- *Anthropic SDK — usage object* (`response.usage.input_tokens` / `output_tokens`): <https://docs.claude.com/en/api/>
