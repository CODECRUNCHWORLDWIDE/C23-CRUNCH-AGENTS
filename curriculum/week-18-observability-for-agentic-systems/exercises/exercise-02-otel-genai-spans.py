#!/usr/bin/env python3
# Exercise 2 — OTel Gen-AI spans for a simulated agent run (build the tree, assert it)
#
# Goal: Build a CORRECT OpenTelemetry span tree for a simulated 3-step agent run
#       (plan -> retrieve -> generate), nested under a parent agent span, emitting
#       the REAL gen_ai.* semantic-convention attributes from Lecture 1 §3. Export
#       the spans to the console so you SEE them, AND capture them in memory so you
#       can ASSERT the tree shape (parent/child links) and the token attributes.
#       The lesson is mechanical: this is exactly what an auto-instrumentor emits,
#       built by hand once so you know what the dashboard is reading.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   Standalone. NO backend, NO network, NO API key required. The ONLY dependency
#   is the OpenTelemetry SDK:
#
#       pip install opentelemetry-sdk
#       python3 exercise-02-otel-genai-spans.py
#
#   It builds a parent "agent.invoke" span with three nested child spans
#   (chat plan -> invoke_agent retrieve -> chat generate), stamps each with the
#   correct gen_ai.* attributes, exports them to the ConsoleSpanExporter (so they
#   print), captures the SAME spans with an InMemorySpanExporter, and then walks
#   the captured spans to ASSERT:
#     * the tree shape (every child's parent is the root; the root has no parent),
#     * the operation names are the real convention values,
#     * the token attributes are present with the real attribute NAMES.
#
#   If opentelemetry-sdk is NOT installed, the file prints a one-line install hint
#   and exits cleanly (code 0) — it always RUNS; it just asks for its one dep.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The script prints the three child spans + the root span (Console export).
#   [ ] The in-memory assertions PASS: 4 spans total, 3 children all parented to
#       the root, root has parent=None.
#   [ ] gen_ai.operation.name is one of {chat, invoke_agent, embeddings,
#       execute_tool} on every gen_ai span (the REAL convention values).
#   [ ] The two chat spans carry gen_ai.usage.input_tokens AND
#       gen_ai.usage.output_tokens (the REAL attribute names, not tokens_in).
#   [ ] You can explain why nesting with start_as_current_span builds the tree
#       for free (the active span becomes the parent of the next one).
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import sys

# --- Import the OTel SDK, or exit cleanly with a hint so the file always runs ---
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
except ImportError:
    print(
        "This exercise needs the OpenTelemetry SDK. Install it and re-run:\n"
        "    pip install opentelemetry-sdk\n"
        "(No backend, no API key, no network are required — only the SDK.)"
    )
    raise SystemExit(0)


# --- The gen_ai.* attribute NAMES — exact, from the OTel Gen-AI conventions -----
# Emit these verbatim. Inventing your own (tokens_in, model_name) breaks every
# dashboard that expects the standard. (Lecture 1 §3.1)
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

# The real, enumerated operation values (Lecture 1 §3.2).
OPS = {"chat", "embeddings", "execute_tool", "invoke_agent"}


def init_tracing() -> tuple[trace.Tracer, InMemorySpanExporter]:
    """Stand up a TracerProvider that exports to BOTH the console (so you see the
    spans) and an in-memory buffer (so we can assert on them). We use
    SimpleSpanProcessor — it exports synchronously on span end, so by the time we
    inspect the buffer the spans are guaranteed to be there (BatchSpanProcessor
    would need an explicit force_flush). (Lecture 1 §4)"""
    resource = Resource.create({"service.name": "crunch-agents-ex02"})
    provider = TracerProvider(resource=resource)

    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    provider.add_span_processor(SimpleSpanProcessor(memory))

    trace.set_tracer_provider(provider)
    return trace.get_tracer("crunch-agents"), memory


def run_simulated_agent(tracer: trace.Tracer) -> None:
    """Build the plan -> retrieve -> generate tree by hand. Each
    start_as_current_span makes its span the ACTIVE one, so any span started
    inside the block becomes its child automatically — that is how the tree is
    built for free (Lecture 1 §4)."""
    with tracer.start_as_current_span("agent.invoke") as root:
        root.set_attribute("crunch.route", "answer_question")
        root.set_attribute("crunch.user_id", "u_17")

        # --- Step 1: plan — a quick supervisor routing call (a chat op) ---------
        with tracer.start_as_current_span("chat plan") as plan:
            plan.set_attribute(GEN_AI_SYSTEM, "anthropic")
            plan.set_attribute(GEN_AI_OPERATION, "chat")
            plan.set_attribute(GEN_AI_REQUEST_MODEL, "claude-haiku-4-5")
            plan.set_attribute(GEN_AI_INPUT_TOKENS, 612)
            plan.set_attribute(GEN_AI_OUTPUT_TOKENS, 28)

        # --- Step 2: retrieve — an agent invocation wrapping a tool call --------
        with tracer.start_as_current_span("invoke_agent retrieve") as retr:
            retr.set_attribute(GEN_AI_OPERATION, "invoke_agent")
            retr.set_attribute(GEN_AI_AGENT_NAME, "retriever")
            # NOTE: a real retriever would nest an embeddings + a vector_search
            # span here; we keep it one span so the asserted tree is exactly the
            # 4 spans the expected-output block documents. The stretch adds them.
            retr.set_attribute(GEN_AI_TOOL_NAME, "vector_search")
            retr.set_attribute("db.system", "pgvector")
            retr.set_attribute("crunch.retrieved_chunks", 5)

        # --- Step 3: generate — the writer's model call (a chat op) -------------
        with tracer.start_as_current_span("chat generate") as gen:
            gen.set_attribute(GEN_AI_SYSTEM, "anthropic")
            gen.set_attribute(GEN_AI_OPERATION, "chat")
            gen.set_attribute(GEN_AI_REQUEST_MODEL, "claude-opus-4-8")
            gen.set_attribute(GEN_AI_INPUT_TOKENS, 1843)
            gen.set_attribute(GEN_AI_OUTPUT_TOKENS, 412)


# --- Assertions over the captured spans ---------------------------------------
def assert_tree(spans) -> None:
    """Walk the captured spans and assert the tree shape + token attributes.
    `spans` is the tuple of ReadableSpans from InMemorySpanExporter."""
    by_name = {s.name: s for s in spans}

    # (1) Exactly the four spans we built.
    expected_names = {"agent.invoke", "chat plan",
                      "invoke_agent retrieve", "chat generate"}
    assert set(by_name) == expected_names, (
        f"expected {expected_names}, got {set(by_name)}")

    root = by_name["agent.invoke"]
    children = [by_name[n] for n in expected_names if n != "agent.invoke"]

    # (2) The root has NO parent; every child's parent is the root's span id.
    assert root.parent is None, "root span must have no parent"
    root_span_id = root.context.span_id
    for child in children:
        assert child.parent is not None, f"{child.name} should have a parent"
        assert child.parent.span_id == root_span_id, (
            f"{child.name}'s parent is not the root — orphaned span! "
            "context propagation broke (Lecture 1 §2).")

    # (3) Every span sharing the trace id (one trace, not four).
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1, f"all spans must share one trace id, got {trace_ids}"

    # (4) Operation names are the REAL convention values.
    for child in children:
        op = child.attributes.get(GEN_AI_OPERATION)
        assert op in OPS, f"{child.name}: bad gen_ai.operation.name {op!r}"

    # (5) The two chat spans carry BOTH token attributes, with the REAL names.
    for name in ("chat plan", "chat generate"):
        a = by_name[name].attributes
        assert GEN_AI_INPUT_TOKENS in a, f"{name} missing {GEN_AI_INPUT_TOKENS}"
        assert GEN_AI_OUTPUT_TOKENS in a, f"{name} missing {GEN_AI_OUTPUT_TOKENS}"
        assert isinstance(a[GEN_AI_INPUT_TOKENS], int)
        assert isinstance(a[GEN_AI_OUTPUT_TOKENS], int)


def print_tree(spans) -> None:
    """Pretty-print the captured tree (root first, then its children)."""
    by_name = {s.name: s for s in spans}
    root = by_name["agent.invoke"]
    print("\n==================== CAPTURED TREE ====================")
    print(f"{root.name}  (trace_id={root.context.trace_id:032x}, root)")
    for name in ("chat plan", "invoke_agent retrieve", "chat generate"):
        s = by_name[name]
        op = s.attributes.get(GEN_AI_OPERATION, "-")
        toks = ""
        if GEN_AI_INPUT_TOKENS in s.attributes:
            toks = (f"  in={s.attributes[GEN_AI_INPUT_TOKENS]} "
                    f"out={s.attributes[GEN_AI_OUTPUT_TOKENS]}")
        print(f"  └─ {name:24s} gen_ai.operation.name={op}{toks}")


def main() -> int:
    tracer, memory = init_tracing()
    run_simulated_agent(tracer)

    spans = memory.get_finished_spans()
    print(f"\ncaptured {len(spans)} spans in memory")

    try:
        assert_tree(spans)
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return 1

    print_tree(spans)
    print("\n==================== VERDICT ====================")
    print("  PASS: 4 spans, one trace, 3 children parented to the root,")
    print("  real gen_ai.* operation names, and both token attributes on the")
    print("  chat spans. This is exactly what an auto-instrumentor emits — and")
    print("  what every cost/latency dashboard reads. You built it by hand once")
    print("  so the dashboard is no longer magic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; the Console export of each span is verbose JSON-ish and
# is elided here — the parts that matter are the captured tree + the VERDICT)
# -----------------------------------------------------------------------------
#
# {
#   "name": "chat plan",
#   "context": { "trace_id": "0x...", "span_id": "0x..." },
#   "parent_id": "0x...",                      <-- the root's span id (nested!)
#   "attributes": {
#     "gen_ai.system": "anthropic",
#     "gen_ai.operation.name": "chat",
#     "gen_ai.request.model": "claude-haiku-4-5",
#     "gen_ai.usage.input_tokens": 612,
#     "gen_ai.usage.output_tokens": 28
#   }, ...
# }
# ... (chat plan, invoke_agent retrieve, chat generate, then agent.invoke last —
#      children end BEFORE the parent, so the root prints last) ...
#
# captured 4 spans in memory
#
# ==================== CAPTURED TREE ====================
# agent.invoke  (trace_id=..., root)
#   └─ chat plan                gen_ai.operation.name=chat  in=612 out=28
#   └─ invoke_agent retrieve    gen_ai.operation.name=invoke_agent
#   └─ chat generate            gen_ai.operation.name=chat  in=1843 out=412
#
# ==================== VERDICT ====================
#   PASS: 4 spans, one trace, 3 children parented to the root, real gen_ai.*
#   operation names, and both token attributes on the chat spans.
#
# WHY THE TREE IS FREE: start_as_current_span makes its span the ACTIVE span for
# the duration of the `with` block, so the next start_as_current_span inside it
# automatically links to it as parent. Nest the blocks and you get the parent/
# child tree with no manual parent-id bookkeeping. Break the nesting (start a
# child OUTSIDE the parent's block) and the child orphans into its own trace —
# the single most common instrumentation bug (Lecture 1 §2).
# -----------------------------------------------------------------------------
