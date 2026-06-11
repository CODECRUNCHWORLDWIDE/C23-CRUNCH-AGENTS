# Exercise 1 — Route the Supervisor

**Estimated time:** ~50 minutes. Guided.

## Goal

Build the supervisor router node for the capstone's multi-agent graph — the node whose *only* job is to decide which subordinate agent runs next — and prove, by reading a trace, that it routes by a structured decision rather than parsed prose. By the end you will have a supervisor that drives a four-agent graph (retrieval, code, writing, critique) and a trace that shows *why* it routed each way.

This is the brain of the capstone. The lesson is the lecture's first rule made concrete: **the supervisor decides; the subordinates do.** If your supervisor's prompt ever starts retrieving or writing, you've collapsed the router into a doer and lost the debuggability the pattern exists to give you.

## Setup

Work in your Sprint A capstone repo. Install the agent deps:

```bash
pip install langgraph langchain-core anthropic opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-http
export ANTHROPIC_API_KEY=sk-ant-...
```

For the trace backend, run Langfuse locally (the week-18 self-host) or Phoenix (`pip install arize-phoenix && phoenix serve`). Either is fine for this exercise — you just need somewhere the spans land that you can click through.

## Part A — the router decision (structured, not parsed)

Write `supervise(state) -> Route` exactly as the lecture shows: a `claude-opus-4-8` call with `thinking={"type":"adaptive"}`, `output_config={"effort":"medium", "format": {...json_schema...}}`, returning a single `next` enum field and a `reason`. The schema constrains `next` to `["retrieval", "code", "writing", "critique", "finish"]`.

**Acceptance criteria for Part A:**

- The router returns a value from the enum — never free text you have to parse.
- The `reason` field is captured into `state["route_reason"]` so it lands in the trace.
- You do **not** pass `budget_tokens` or `temperature` (both 400 on `claude-opus-4-8`).
- The system prompt forbids the supervisor from doing any retrieval/computation/writing itself.

**Hint.** The routing policy in the lecture's `ROUTER_SYSTEM` is the spec: no context yet → retrieve; needs computation and none done → code; context but no draft → write; draft not critiqued → critique; critique passed → finish; critique failed → write again. Encode that policy in the prompt and let the structured output enforce the enum.

## Part B — wire the conditional edge

Build the LangGraph state graph: a `supervisor` node, four subordinate nodes (stub them — each just appends a marker to state and returns to the supervisor), `START → supervisor`, every subordinate `→ supervisor`, and a conditional edge from `supervisor` keyed on `state["next"]` mapping each route to its node and `finish → END`.

**Acceptance criteria for Part B:**

- The graph compiles and runs a query end-to-end (with stub subordinates).
- The supervisor is the hub: every subordinate returns to it, and it alone decides the next hop.
- A query with no context first routes to `retrieval`; after a draft exists, it routes to `critique`; after critique passes, it routes to `finish`.

**Hint.** Use a `TypedDict` for the state with at least `messages`, `next`, `route_reason`, `contexts`, `draft`, and `critique_passed`. The stub subordinates set the field that proves they ran (the retrieval stub sets `contexts`, the writing stub sets `draft`, the critique stub sets `critique_passed=True`).

## Part C — read the routing decisions in a trace

Instrument the supervisor node with an OTel span (lecture 2, §3). Set `gen_ai.route` to the chosen route and add the `route_reason` as a span attribute. Run a query, open the trace, and read the routing chain.

**Acceptance criteria for Part C:**

- Each supervisor decision is a span with `gen_ai.route` and the `reason` attribute.
- You can open the trace and narrate the routing chain: which agent ran, in what order, and *why* the supervisor chose each — from the trace alone, no print statements.
- Write a one-paragraph note (`notes/week-23/supervisor-routing.md`) describing one query's full routing chain as read from the trace, including the supervisor's `reason` at each hop.

**Hint.** The point of Part C is the corollary: *an agentic system you cannot trace is an agentic system you cannot finish.* Deliberately make the critique stub fail once (set `critique_passed=False`) and confirm the trace shows `write → critique → write → critique → ...` — then add the step budget from the lecture and confirm a budget abort shows up as `gen_ai.abort_reason` in the trace.

## Part D — add the budget

Add the `Budget` class from the lecture and the budget check at the top of `supervisor_node`. Force a runaway (a critique that never passes) and confirm the step budget aborts the loop cleanly with `state["aborted"]` set and an answer that says the budget was exhausted.

**Acceptance criteria for Part D:**

- A runaway loop is aborted by the step budget (not by Ctrl-C).
- The abort reason lands in the trace as `gen_ai.abort_reason="step budget"`.
- The final answer is the last draft (or a clear "aborted" message), not a crash.

**Hint.** This is the production-vs-demo line from the lecture. The budget is the thing that stops a misrouting bug from looping forever at 3 AM. Build it now; you'll alert on `gen_ai.abort_reason` in week 24.

## What you've built

A supervisor that routes a four-agent graph by a structured decision, a conditional edge that dispatches by that decision, a trace that shows *why* each route was chosen, and a budget that aborts runaways. This is the control structure the mini-project assembles into the shipped capstone. Commit it; the next exercise builds the tool the retrieval-agent will call.
