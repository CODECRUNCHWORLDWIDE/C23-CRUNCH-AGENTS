# Lecture 1 — The Supervisor Graph and the MCP Tool Surface

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can build a LangGraph supervisor that *routes* turns to subordinate agents (rather than doing the work itself), put step/token/time/cost budgets on every route, make the graph resume after a crash with a checkpointer, and stand up four MCP servers (filesystem, calculator, web-fetch, custom corpus-search) over both stdio and streamable HTTP with argument validation, path-traversal defense, and rate limiting on every tool.

If you remember one sentence from this entire week, remember the lecture title:

> **The last 10% of an agent is 90% of the engineering. Pick what to drop early.**

There's a corollary you should tape next to it:

> **An agentic system you cannot trace is an agentic system you cannot finish.** When the supervisor routes to the wrong agent, the trace tells you in thirty seconds. Without it, you bisect by print statement for an hour.

In Sprint A you built the body — retrieval, memory, the corpus. This lecture wires the brain (the supervisor) and the hands (the MCP tool surface). The next lecture wires the metabolism (serving) and the senses (eval + tracing). Together they are the Production Agentic Research Assistant the syllabus specifies.

---

## 1. Why a supervisor, and why it routes rather than does

You have four jobs to do for any research query: *retrieve* the relevant context, *compute* anything the answer needs (a sum, a date difference, a unit conversion), *write* a grounded answer, and *critique* the draft before it ships. You could cram all four into one giant ReAct loop with one giant prompt and twelve tools. People do. It works until it doesn't, and when it doesn't, you cannot tell *which* of the four jobs went wrong, because they're all tangled in one context window.

The supervisor pattern unties them. Each job becomes a **subordinate agent** — a small, single-purpose graph node with its own prompt and its own narrow tool set. A **supervisor** node sits above them, and its only job is to look at the current state and decide *which subordinate runs next*. It is a router, not a doer.

> **The rule that defines the whole architecture:** the supervisor decides; the subordinates do. If you ever find the supervisor's prompt asking it to "retrieve the relevant clauses and then write a grounded answer," stop — you've collapsed the router into a doer, and you've lost the thing that made the pattern worth using: the ability to read a trace and see *retrieval-agent failed*, not *the agent failed*.

Why does routing-not-doing matter so much in a capstone specifically?

**Reason 1 — debuggability.** When the answer is wrong, the trace shows a chain: `route=retrieval → retrieval_agent → corpus.search → 4 chunks`, then `route=write → writing_agent → draft`, then `route=critique → critique_agent → not grounded ✗`. You see immediately that retrieval got chunks but the draft wasn't grounded in them — a writing failure, not a retrieval failure. One glance, one culprit.

**Reason 2 — independent prompts.** The retrieval-agent's prompt is about *finding* and *citing*. The writing-agent's prompt is about *synthesizing only from provided context*. The critique-agent's prompt is about *catching ungrounded claims*. Cram them together and each instruction dilutes the others. Separate them and each is sharp.

**Reason 3 — independent budgets.** Retrieval should be cheap and fast (a few tool calls). Writing can spend more tokens. Critique should be one shot. Per-agent budgets (next section) only make sense when the agents are separate.

**Reason 4 — independent models.** The cheap local 7B can handle routing and retrieval; the hard writing route can go to `claude-opus-4-8`. You can only route by model when the work is split by agent.

Here is the graph shape. The supervisor is the hub; every subordinate returns to it; the supervisor decides whether to route again or finish.

```
                 ┌──────────────┐
        ┌───────▶│  Supervisor  │◀────────┐
        │        │   (router)   │         │
        │        └──────┬───────┘         │
        │         decides next            │
   ┌────┴────┐  ┌───────┴──────┐  ┌────────┴───────┐  ┌──────────┐
   │Retrieval│  │  Code/Calc   │  │    Writing     │  │ Critique │
   │  agent  │  │    agent     │  │     agent      │  │  agent   │
   └────┬────┘  └──────┬───────┘  └────────┬───────┘  └────┬─────┘
        │              │                   │               │
        └──────────────┴───────────────────┴───────────────┘
                      (all return to supervisor)
```

---

## 2. The supervisor router node

The supervisor reads the conversation state and emits a single decision: which agent runs next, or `finish`. In LangGraph that decision drives a **conditional edge**. The cleanest way to make the decision reliable is *structured output* — you ask the model for one field, `next`, constrained to an enum, so it can't hand you prose you have to parse.

```python
from typing import Literal, TypedDict
from anthropic import Anthropic

client = Anthropic()  # ANTHROPIC_API_KEY from the environment

Route = Literal["retrieval", "code", "writing", "critique", "finish"]

ROUTER_SYSTEM = """You are the supervisor of a research-assistant agent system.
You do NOT answer questions, retrieve documents, compute, or write. Your ONLY job
is to choose which subordinate agent runs next, given the conversation so far.

Agents:
- retrieval: finds and returns relevant corpus chunks for the user's question.
- code: runs a calculator/Python tool for any arithmetic, date math, or unit work.
- writing: synthesizes a grounded answer ONLY from retrieved context.
- critique: checks the latest draft is grounded in the retrieved context.

Routing policy:
- If there is no retrieved context yet, route to retrieval.
- If the question needs computation and it has not been done, route to code.
- If there is context and no draft, route to writing.
- If there is a draft that has not been critiqued, route to critique.
- If critique passed, finish. If critique failed, route back to writing.
Choose exactly one next step."""


def supervise(state: "AgentState") -> Route:
    """Return the next route. The model returns ONE enum field, not prose."""
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "medium",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "next": {
                            "type": "string",
                            "enum": ["retrieval", "code", "writing",
                                     "critique", "finish"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["next", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        system=ROUTER_SYSTEM,
        messages=[{"role": "user", "content": render_state(state)}],
    )
    import json
    decision = json.loads(next(b.text for b in resp.content if b.type == "text"))
    state["route_reason"] = decision["reason"]   # logged into the trace
    return decision["next"]
```

Two senior notes on that code, both load-bearing for the capstone:

- **The model is `claude-opus-4-8` and thinking is adaptive.** This is the 2026-current Anthropic surface: you do **not** pass `budget_tokens` (it 400s on this model) and you do **not** pass `temperature` (also 400). You set `thinking={"type":"adaptive"}` and control depth with `output_config={"effort":"medium"}`. Routing is a judgment task that benefits from a little thinking; `medium` is the cost/quality sweet spot. If you route to the local 7B instead, drop the `thinking`/`effort` fields — those are Anthropic-model parameters.
- **The decision is structured, not parsed.** `output_config.format` with a `json_schema` constrains the output to your enum. You never regex "I think we should retrieve next" out of prose. The `reason` field is for the trace — it's why you can read *why* the supervisor routed where it did when you debug.

Wiring it into LangGraph is a conditional edge from the supervisor node:

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("retrieval", retrieval_agent)
graph.add_node("code", code_agent)
graph.add_node("writing", writing_agent)
graph.add_node("critique", critique_agent)

graph.add_edge(START, "supervisor")
# Subordinates always return to the supervisor:
for agent in ("retrieval", "code", "writing", "critique"):
    graph.add_edge(agent, "supervisor")

# The supervisor's conditional edge dispatches by the router decision:
graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],            # set by supervise()
    {"retrieval": "retrieval", "code": "code", "writing": "writing",
     "critique": "critique", "finish": END},
)
```

That's the whole control structure. The supervisor runs, sets `state["next"]`, and the conditional edge sends control to the matching subordinate — which does its narrow job, writes its result into state, and returns to the supervisor for the next decision. The loop ends when the supervisor routes to `finish`.

---

## 3. Budgets — the thing that makes it production, not a demo

The single most common agentic-system failure in production is not a wrong answer. It's a **runaway loop**: the supervisor routes `writing → critique → writing → critique → writing...` forever because the critique never passes and nobody told the loop to stop. In a demo you Ctrl-C it. In production it burns your token budget at 3 AM and pages you.

> **Most agent failures are not model failures — they are loop failures, budget failures, or tool failures.** (You learned this in week 5; the capstone is where ignoring it costs real money.)

Every route gets four budgets, checked in the supervisor before it dispatches:

- **Step budget** — max total node executions (e.g. 12). Hit it → force `finish` with a "budget exhausted" answer.
- **Token budget** — max cumulative tokens across the run (e.g. 50k). Tracked from each model response's `usage`.
- **Time budget** — wall-clock deadline (e.g. 60s). A long retrieval or a slow vendor call can blow this.
- **Cost budget** — max dollars (e.g. $0.10). Computed from token usage × per-model rate; the vendor tier is the expensive one.

```python
import time

class Budget:
    def __init__(self, max_steps=12, max_tokens=50_000,
                 max_seconds=60.0, max_dollars=0.10):
        self.max_steps, self.max_tokens = max_steps, max_tokens
        self.max_seconds, self.max_dollars = max_seconds, max_dollars
        self.steps = 0
        self.tokens = 0
        self.dollars = 0.0
        self.t0 = time.monotonic()

    def charge(self, *, steps=0, tokens=0, dollars=0.0):
        self.steps += steps
        self.tokens += tokens
        self.dollars += dollars

    def exhausted(self) -> str | None:
        if self.steps >= self.max_steps:
            return "step budget"
        if self.tokens >= self.max_tokens:
            return "token budget"
        if time.monotonic() - self.t0 >= self.max_seconds:
            return "time budget"
        if self.dollars >= self.max_dollars:
            return "cost budget"
        return None


def supervisor_node(state):
    state["budget"].charge(steps=1)
    reason = state["budget"].exhausted()
    if reason is not None:
        state["next"] = "finish"
        state["answer"] = (state.get("draft") or
                           f"[aborted: {reason} exhausted before a grounded answer]")
        state["aborted"] = reason          # shows up in the trace
        return state
    state["next"] = supervise(state)
    return state
```

The budget is not a nice-to-have. It is the difference between a capstone that "works on my machine for the demo query" and one that survives the chaos drill next week. The cost budget in particular is what stops a misrouting bug from quietly running up a vendor bill — and when you instrument this with OTel (next lecture), a budget abort is a span attribute (`gen_ai.abort_reason="cost budget"`) you can alert on.

---

## 4. The checkpointer — resume after a crash

LangGraph's checkpointer persists graph state after every node. With a SQLite checkpointer, if the process dies mid-run (a vLLM replica falls over, the box reboots), you resume from the last completed node instead of starting the whole research run over.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("capstone_checkpoints.sqlite") as saver:
    app = graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "research-run-0007"}}
    # First run dies after the retrieval node...
    # ...re-invoking with the SAME thread_id resumes from there:
    result = app.invoke({"messages": [user_msg], "budget": Budget()}, config)
```

The `thread_id` is the resume key. In the capstone, every user research run gets a thread id, and the checkpoint database is the thing that makes "kill a vLLM replica mid-run" (week 24's chaos drill) survivable rather than catastrophic. Build it in now; you'll be glad next week.

> **Why this is in Sprint B and not Sprint A:** in Sprint A the work was a pipeline (retrieve → memory) — a pipeline either completes or you re-run it. In Sprint B the work is a *graph with a loop*, and a loop that dies on step 9 of 12 is expensive to restart from step 1. Persistence earns its keep exactly when the work is long and looping.

---

## 5. The MCP tool surface — four servers, two transports

The subordinate agents don't reach into your codebase directly; they call **tools** exposed over the **Model Context Protocol**. This is the week-15 lesson made load-bearing: MCP is the cross-vendor tool protocol, and exposing your tools over it means the same surface works from a LangGraph agent, from Claude Desktop, from Cursor, or from any MCP client.

The capstone needs four servers:

1. **Filesystem** — read files from a sandboxed root (the retrieval-agent reads source documents; the writing-agent reads templates).
2. **Calculator / Python** — the code-agent's tool for arithmetic, date math, unit conversion.
3. **Web-fetch** — fetch a URL the corpus references (rate-limited, allow-listed).
4. **Custom corpus-search** — the load-bearing one: it wraps your Sprint A hybrid-retrieval pipeline so the retrieval-agent calls `corpus.search(query, k)` and gets ranked chunks.

Here is the corpus-search server, the heart of the surface. Note the structure: a server, a `list_tools` advertising the tool's schema, a `call_tool` dispatching and *validating arguments before doing anything*.

```python
# corpus_mcp_server.py — the custom private-corpus search MCP server
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types
from capstone.retrieval import hybrid_search   # your Sprint A pipeline, UNCHANGED

server = Server("capstone-corpus")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="corpus_search",
            description="Search the private research corpus. Call this when you "
                        "need source material to ground an answer. Returns the "
                        "top-k ranked chunks with their source doc ids.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "k": {"type": "integer", "minimum": 1, "maximum": 20,
                          "default": 5, "description": "Number of chunks to return."},
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "corpus_search":
        raise ValueError(f"unknown tool: {name}")

    # --- validate BEFORE doing anything (a tool is an RCE primitive) ---
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    k = arguments.get("k", 5)
    if not isinstance(k, int) or not (1 <= k <= 20):
        raise ValueError("k must be an integer in [1, 20]")

    hits = hybrid_search(query, k=k)     # BM25 + dense + reranker from Sprint A
    lines = [f"[{h.doc_id}] {h.text}" for h in hits]
    return [types.TextContent(type="text", text="\n\n".join(lines))]


if __name__ == "__main__":
    import asyncio

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(main())
```

The description matters more than beginners expect. **Be prescriptive about *when* to call the tool**, not just what it does — "Call this when you need source material to ground an answer." On recent models, which reach for tools more conservatively, a trigger condition in the description measurably raises the should-call rate. That's the difference between the retrieval-agent searching the corpus and the retrieval-agent confabulating from prior knowledge.

### 5.1 The two transports

The same server runs over two transports:

- **stdio** — the server is a subprocess; the agent talks to it over stdin/stdout. This is the default for *local* tools and the lowest-friction path. The `if __name__ == "__main__"` block above is the stdio entrypoint.
- **streamable HTTP** — the server is an HTTP endpoint; the agent (or Claude Desktop, or Cursor) talks to it over HTTP. This is the transport for a *deployed, cross-client* tool surface — the one your `docker compose up` exposes.

You write the tool logic once and choose the transport at startup. stdio for the LangGraph agents running in-process with the supervisor; streamable HTTP for the corpus server exposed behind your deploy so an external client can use it too.

### 5.2 The security review — a tool is RCE

> **A tool call is a request to take action in the world. Treat it like a remote API call from an untrusted client — because that is what it is.** (Week 4 and week 17; the capstone is where a missed check becomes the chaos-drill prompt-injection scenario.)

Every tool on the surface gets three defenses, no exceptions:

- **Argument validation.** Validate types, ranges, and shapes *before* the tool does anything. The corpus server above rejects a non-string query and an out-of-range `k`. The filesystem server is the dangerous one: validate the path.
- **Path-traversal defense.** The filesystem server must resolve every requested path against a sandbox root and reject anything that escapes it. The classic attack is `../../etc/passwd`; the classic bug is string-prefix-matching the root instead of resolving and comparing real paths.

  ```python
  from pathlib import Path

  SANDBOX = Path("/srv/corpus").resolve()

  def safe_path(requested: str) -> Path:
      p = (SANDBOX / requested).resolve()
      if not p.is_relative_to(SANDBOX):     # Python 3.9+: the correct check
          raise ValueError("path escapes the sandbox")
      return p
  ```

- **Rate limiting.** The web-fetch server in particular must not let an injected instruction in a retrieved document turn your agent into a request amplifier. A token-bucket per tool, per run, caps the blast radius.

Why so much paranoia in a capstone? Because next week you inject a malicious instruction through a retrieved document and find out whether these defenses hold. The week-17 threat model isn't a checkbox — it's the thing the chaos drill tests. Build the defenses in now; document them in the tool surface doc; you'll re-measure them under attack in seven days.

---

## 6. The thin slice — scope before you deepen

Now the lecture's title becomes a method. You have a supervisor, four subordinates, four MCP servers, two model tiers, an eval suite, and two tracing backends still to wire. You will not finish all of them excellently by Friday. The senior move is to get the **thin slice** working end-to-end first:

> **One query in → supervisor routes → retrieval-agent calls corpus.search → writing-agent drafts → critique-agent checks → answer out, with a trace in Langfuse and a Ragas score on it.**

That slice touches every layer once. Get *it* green, and you've de-risked the integration — the wiring is proven. *Then* you deepen by what the eval flags: if context precision is low, the retrieval-agent's `k` or the reranker needs work; if faithfulness is low, the writing-agent's prompt needs tightening; if the judge disagrees with humans, the calibration needs more examples. You triage by measurement.

The anti-pattern is the opposite: perfecting the reranker for two days before the supervisor has ever routed a single query, then discovering on Thursday that the writing-agent doesn't compile. Integration risk is front-loaded; resolve it first.

This is also why the **cut list** is a deliverable (homework Problem 4). Writing down "I dropped the web-fetch server because no gold question needed it; I dropped HITL approval on the code-agent because the eval showed the generated expressions were 98% correct" is not an admission of failure — it's the evidence that you scoped like an engineer rather than a completionist. A capstone with a clear cut list and a green eval beats a capstone with every feature half-wired and a red eval, every time.

---

## 7. What you can do now

You can:

- Build a supervisor that routes turns to subordinate agents via a structured-output decision, and explain why routing-not-doing makes the system debuggable.
- Put step/token/time/cost budgets on every route and force a clean `finish` when one is exhausted.
- Make the graph resume after a crash with a SQLite checkpointer keyed by `thread_id`.
- Stand up four MCP servers with the `mcp` Python SDK, exposed over stdio and streamable HTTP.
- Harden every tool with argument validation, path-traversal defense, and rate limiting — and explain why the chaos drill next week tests exactly those defenses.
- Scope the integration with the thin-slice-first method and justify a cut list by measurement.

The next lecture wires the metabolism and the senses: the vLLM/LiteLLM two-tier serving stack that powers the local-vs-vendor routing, the Ragas + calibrated-judge eval suite that gates the build, and the OpenTelemetry Gen-AI tracing that turns every route, tool call, and model request into a span you can read. That's the half of Sprint B that turns "it ran once" into "it's measured, served, and observable" — which is to say, into a system you can hand to next week's chaos drill.

---

*If you find errors in this material, please open an issue or send a PR.*
