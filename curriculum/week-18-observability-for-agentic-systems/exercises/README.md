# Week 18 — Exercises

Three focused drills that take you from "I stood up a backend and read a trace" to "I computed tokens, cost, and p95 from recorded spans." Each takes 30–60 minutes. Do them in order — exercise 3 reasons about the same `gen_ai.*` attributes that exercise 2 teaches you to emit, which assumes the trace-reading intuition from exercise 1.

## Index

1. **[Exercise 1 — Read a trace](exercise-01-read-a-trace.md)** — stand up Phoenix locally (`px.launch_app()`, zero API key) or self-hosted Langfuse via Docker, send a few traced agent calls, read the resulting trace tree by eye, and find the slow span. (~50 min, guided)
2. **[Exercise 2 — OTel Gen-AI spans](exercise-02-otel-genai-spans.py)** — build a nested OTel span tree for a simulated 3-step agent (plan → retrieve → generate), emit the correct `gen_ai.*` attributes, and assert the tree shape + token attributes with an `InMemorySpanExporter`. Runs with only `opentelemetry-sdk`, no key, no backend. (~50 min, runnable)
3. **[Exercise 3 — Token accounting](exercise-03-token-accounting.py)** — given a list of recorded spans, roll up tokens + cost per route / user / model and compute p50/p95/p99 latency per agent step. stdlib + numpy only. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install deps as each exercise needs them: `pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http arize-phoenix numpy`. Exercise 1 also wants `openinference-instrumentation-langchain` (or a manual span tree); Exercise 2 needs *only* `opentelemetry-sdk`; Exercise 3 needs *only* numpy.
- **The no-infra defaults are deliberate.** Exercise 2 exports to `ConsoleSpanExporter` (prints spans) and captures with `InMemorySpanExporter` (asserts on them) — zero backend, zero key. Exercise 1's default backend is **Phoenix local** (`px.launch_app()`), which also needs no key. Reach for self-hosted Langfuse (Docker) only when you want the durable product-analytics view.
- **Read the trace before you trust the metric.** Exercise 1's whole point is that the failing step is *visible* in the tree — open the trace and look at it with your eyes before you reach for any aggregate.
- **Emit the convention names verbatim.** Every span carries `gen_ai.*` attributes spelled exactly as the spec writes them (`gen_ai.usage.input_tokens`, not `tokens_in`). Inventing your own names is the classic mistake that breaks every dashboard; Exercise 2 asserts the real names.
- When a run looks wrong, walk the §3 decision tree from Lecture 2 *before* you guess: is it red (first error span), slow (biggest span), or wrong (walk the data)?
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

Both `.py` files are standalone and need **no network, no key, no backend**.

```bash
# Exercise 2 needs the OTel SDK; it exports to console + an in-memory buffer it asserts on.
pip install opentelemetry-sdk
python3 exercise-02-otel-genai-spans.py

# Exercise 3 is stdlib + numpy only — recorded spans are in the file.
pip install numpy
python3 exercise-03-token-accounting.py
```

If `opentelemetry-sdk` is missing, Exercise 2 prints a one-line install hint and exits cleanly rather than crashing — so the file always *runs*, it just asks for its one dependency.

## A note on determinism

The span-tree shape (parent/child structure, operation names, token attributes) is **fully deterministic** — Exercise 2 builds the same tree and asserts the same attributes every run, with no LLM call and no randomness. Exercise 3's rollups and percentiles are deterministic too: the recorded spans are fixed in the file, so the per-route cost table and the p50/p95/p99 numbers reproduce exactly. The only thing that would move is real wall-clock latency if you instrumented a *live* run — which is why the exercises use *recorded* durations: so the lesson (how to compute the statistic) is separable from the noise of a real machine.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-18` to compare.
