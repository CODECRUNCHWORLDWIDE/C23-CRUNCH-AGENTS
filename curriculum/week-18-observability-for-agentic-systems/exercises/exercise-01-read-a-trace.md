# Exercise 1 — Read a Trace

**Goal:** Stand up a tracing backend with zero friction, send a handful of traced agent calls into it, and then *read the resulting trace tree by eye* until you can point at the slow span without touching a log file. You will train the single most important habit of observability: **the failing (or slow) step is visible in the tree — open the trace and look at it.** A trace you never look at is a log file with extra steps.

**Estimated time:** 50 minutes. Guided.

---

## Setup

You have two backends to choose from. **Phoenix is the no-friction default — use it unless you specifically want the Langfuse product-analytics view.**

### Option A — Arize Phoenix (default, zero API key)

```bash
pip install arize-phoenix openinference-instrumentation-langchain opentelemetry-sdk
```

Phoenix runs a local UI *and* an OTLP collector in-process. No account, no key:

```python
# launch_phoenix.py — a local UI + OTLP collector, no API key.
import phoenix as px

session = px.launch_app()          # opens http://localhost:6006
print("Phoenix UI:", session.url)
input("Phoenix is up. Press Enter to shut down...")
```

### Option B — Self-hosted Langfuse (Docker)

If you want the durable, team-facing backend (cost dashboards, scores, replay datasets), bring up Langfuse with its official compose file:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d              # Postgres + ClickHouse + web + worker
# open http://localhost:3000 , create a project, copy the public/secret keys
```

Then point an OTLP exporter at Langfuse's OTLP endpoint (the base64 is `public_key:secret_key`):

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:3000/api/public/otel/v1/traces"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic $(printf 'pk-lf-...:sk-lf-...' | base64)"
```

> If you are blocked on infrastructure for *any* reason, you can still do the whole exercise with `ConsoleSpanExporter` — the spans print to your terminal and you read the tree there. But the visual flame graph in Phoenix is the lesson, so prefer Phoenix.

---

## Step 1 — Wire OTel to your backend

Register the tracer and (Phoenix path) auto-instrument LangChain/LangGraph so your framework calls emit `gen_ai.*` spans with no per-call code:

```python
# wire_tracing.py
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register()                 # wires OTel -> Phoenix
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
print("tracing wired: LangChain/LangGraph calls now emit gen_ai spans")
```

If you are not using LangChain, you will build the span tree by hand in Step 2 — that path is below.

---

## Step 2 — Send a few traced agent calls

You want a trace that *looks like an agent run* — a parent span with nested children for plan → retrieve → generate. Two ways:

**Path A — your real Phase III graph.** If your week-13 supervisor is importable, just run it under the instrumentation from Step 1. Send three or four different user requests so you get a few traces to browse.

**Path B — a hand-built span tree** (works with no framework, and teaches you the structure):

```python
# manual_trace.py — a nested agent run, built by hand, exported to your backend.
import time
from opentelemetry import trace

tracer = trace.get_tracer("crunch-agents")   # provider already set by register()


def fake_step(name: str, op: str, seconds: float, **attrs):
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("gen_ai.operation.name", op)
        for k, v in attrs.items():
            span.set_attribute(k, v)
        time.sleep(seconds)                  # simulate the step's real latency


def run_once(user_id: str):
    with tracer.start_as_current_span("agent.invoke") as root:
        root.set_attribute("crunch.user_id", user_id)
        # plan: a quick supervisor routing call
        fake_step("chat supervisor", "chat", 0.3,
                  **{"gen_ai.system": "anthropic",
                     "gen_ai.request.model": "claude-haiku-4-5",
                     "gen_ai.usage.input_tokens": 612,
                     "gen_ai.usage.output_tokens": 28})
        # retrieve: an embedding + a vector search (the SLOW one on purpose)
        with tracer.start_as_current_span("agent retriever") as r:
            r.set_attribute("gen_ai.operation.name", "invoke_agent")
            r.set_attribute("gen_ai.agent.name", "retriever")
            fake_step("embeddings query", "embeddings", 0.2)
            fake_step("vector_search", "execute_tool", 2.7,        # <-- the slow span
                      **{"gen_ai.tool.name": "vector_search",
                         "db.system": "pgvector"})
        # generate: the writer's model call
        fake_step("chat writer", "chat", 1.1,
                  **{"gen_ai.system": "anthropic",
                     "gen_ai.request.model": "claude-opus-4-8",
                     "gen_ai.usage.input_tokens": 1843,
                     "gen_ai.usage.output_tokens": 412})


for uid in ("u_17", "u_42", "u_88"):
    run_once(uid)
print("sent 3 traces — open the backend UI and look at them")
```

Run it (`python3 manual_trace.py`) and open the UI. You now have three traces to read.

---

## Step 3 — Read the trace tree by eye

Open one trace in the UI. You will see a flame graph: indentation is parent/child, bar length is duration. Read it the way Lecture 1 §2 taught:

```
agent.invoke              (4.3s)                       <-- root
├─ chat supervisor        (0.3s)  gen_ai.operation.name=chat
├─ agent retriever        (2.9s)  gen_ai.operation.name=invoke_agent
│   ├─ embeddings query   (0.2s)  gen_ai.operation.name=embeddings
│   └─ vector_search      (2.7s)  gen_ai.operation.name=execute_tool   <-- dominates
└─ chat writer            (1.1s)  gen_ai.operation.name=chat
```

Do three things with your eyes only:

1. **Find the slow span.** Whose duration dominates the root's? Here it's `vector_search` at 2.7s inside `agent retriever`. Your eye should land there in one glance — *that* is the value of a trace tree.
2. **Read its attributes.** Click the slow span. Confirm it carries `gen_ai.operation.name=execute_tool`, `db.system=pgvector`. In a real run this is where you'd see "huge input_tokens" or "a retry (two model spans)" — the *why* behind the slow.
3. **Check the token attributes.** Click `chat writer`; confirm `gen_ai.usage.input_tokens=1843` and `gen_ai.usage.output_tokens=412` are present. These are what every cost dashboard sums — if they're missing on a span, that call is invisible to your per-route cost (Lecture 1 §6).

---

## Step 4 — Confirm context propagation (no orphans)

Verify all four child spans sit *under* the same `agent.invoke` root — one tree, not four separate traces. If a span shows up as its own root, context propagation broke (Lecture 1 §5.3) and you'd be debugging an incomplete trace. On the hand-built tree this is automatic (each `start_as_current_span` nests inside the active one); on a real MCP tool call across a process boundary it's the propagation `inject`/`extract` you must get right.

> Write down, in `notes/week-18/read-a-trace.md`: which span was slowest, its duration, its `gen_ai.operation.name`, and whether the tree was intact (one root) or orphaned. That note is the "found it in the trace" habit, recorded.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] A backend is up (Phoenix local via `px.launch_app()`, or self-hosted Langfuse via Docker) and at least three traces landed in it.
- [ ] You opened a trace and identified the **slow span by eye** (the one whose duration dominates the root), naming it and its duration.
- [ ] You confirmed the slow span's `gen_ai.operation.name` and at least one other attribute, and confirmed a `chat` span carries `gen_ai.usage.input_tokens` / `output_tokens`.
- [ ] The trace is **one tree under one root** (no orphaned spans) — you checked context propagation held.
- [ ] `notes/week-18/read-a-trace.md` records the slow span, its duration, its operation name, and whether the tree was intact.

---

## Stretch

- **Dual export.** Add an `OTLPSpanExporter` pointed at Langfuse *alongside* the Phoenix one (one `TracerProvider`, two `BatchSpanProcessor`s) and confirm the *same* trace appears in both UIs. That is the milestone's "export to both" trick (Lecture 1 §4), proven.
- **Inject a real error.** Make `vector_search` raise an exception inside its span. Confirm OTel records the stack as a span event and the span turns red in the UI — then practice the "find the first red span" debugging step (Lecture 2 §3.1) on it.
- **Capture the prompt.** Set `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` and run a real model call; confirm the prompt and completion show up as span *events* (`gen_ai.user.message`, `gen_ai.choice`) when you click into the `chat` span — the data that makes the *silent* bugs debuggable (Lecture 2 §3.2).

---

When this feels comfortable — when you can open any trace and find the slow span in seconds — move to [Exercise 2 — OTel Gen-AI spans](exercise-02-otel-genai-spans.py).
