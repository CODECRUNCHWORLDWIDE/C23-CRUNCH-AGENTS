# Week 23 — Capstone Sprint B — Agents, MCP, Eval, Serving

This is the week the capstone becomes a *system*. In week 22 (Sprint A) you built the foundation — hybrid retrieval over a 10 GB corpus, three memory tiers, a Mermaid architecture diagram, a 6-page architecture document. The retrieval works. The memory survives a multi-turn conversation. But nothing *drives* it yet: there's no supervisor deciding which agent runs, no tool surface the agents call, no cluster serving the local tier, no eval suite telling you whether any of it is good. Sprint B is where you wire the brain to the body.

By Friday you will have shipped the **Production Agentic Research Assistant**: a LangGraph supervisor graph routing to a retrieval-agent, a code-agent, a writing-agent, and a critique-agent; an MCP tool surface (filesystem, web, calculator, and a custom private-corpus search) live over stdio and streamable HTTP; a vLLM cluster serving your local 7B/13B tier with LiteLLM in front; a vendor frontier model (`claude-opus-4-8`) on the hard routes; a full eval suite — Ragas plus a calibrated LLM-as-judge — green on a 100-question gold set; and OpenTelemetry Gen-AI traces flowing to Langfuse and Arize Phoenix. The deliverable is a live URL or a `docker compose up`-runnable image.

The one sentence to internalize before you read another line — it's the lecture title, and it's the whole week:

> **The last 10% of an agent is 90% of the engineering. Pick what to drop early.**

Here's why that's not defeatist. By now you have eighteen weeks of components. The temptation in a capstone is to make every component *excellent* — the perfect reranker, the perfect memory eviction policy, the perfect prompt. That's how capstones miss the ship date with a half-integrated pile of brilliant parts. The senior move is the opposite: get the *thin slice* working end-to-end first — one query in, supervisor routes, one agent retrieves, one tool fires, eval scores it, trace lands in Langfuse — and *then* deepen the parts that the eval says are weak. You triage by measurement, not by ambition. Sprint B is a scoping exercise as much as an engineering one.

There's a corollary worth taping next to it:

> **An agentic system you cannot trace is an agentic system you cannot finish.** When the supervisor routes to the wrong agent, the trace tells you in thirty seconds. Without it, you bisect by print statement for an hour.

## Learning objectives

By the end of this week, you will be able to:

- **Build** a LangGraph supervisor graph with explicit nodes (plan, route, retrieve, execute-code, write, critique), conditional edges, a SQLite checkpointer for resumability, and per-route budgets (step, token, time, cost).
- **Wire** a multi-agent system where the supervisor is a router, not a doer — it decides *which* subordinate agent handles a turn and hands off, rather than trying to do retrieval and writing itself.
- **Author** an MCP tool surface — a filesystem server, a calculator, a web-fetch server, and a custom private-corpus search server — using the official `mcp` Python SDK, exposed over both stdio and streamable HTTP, consumed from the LangGraph agents.
- **Stand up** a vLLM cluster serving a local 7B/13B model with continuous batching, put LiteLLM in front as an OpenAI-compatible router, and configure cost-tracked routing that sends easy queries local and hard queries to `claude-opus-4-8`.
- **Assemble** an eval suite that runs Ragas (faithfulness, context recall, context precision, answer relevancy) plus a calibrated LLM-as-judge over a 100-question gold set, and gates the build on a pass threshold.
- **Instrument** the whole system with OpenTelemetry Gen-AI semantic conventions, exporting spans to Langfuse (self-hosted) and Arize Phoenix, so every agent step, tool call, and model request is a span you can read.
- **Triage** integration work under a deadline: get the thin end-to-end slice green first, then deepen the components the eval flags as weak — and write down what you dropped and why.
- **Ship** a runnable artifact: a live deploy URL or a `docker compose up` image that brings up the whole stack — supervisor, MCP servers, vLLM, LiteLLM, the eval harness, and the tracing backends.

## Prerequisites

This week assumes you have completed **C23 weeks 1–22**, or have equivalent fluency. Specifically:

- You finished **Sprint A (week 22)** and have the capstone's retrieval and memory layers working: hybrid retrieval (BM25 + dense + reranker) over the 10 GB corpus, the three memory tiers wired, and the architecture document committed. **This week builds directly on that repo** — if Sprint A is broken, fix it before you wire the supervisor.
- You can write a **LangGraph state graph** (week 13): nodes, conditional edges, a checkpointer, reading a trace. We extend that into a multi-agent supervisor here.
- You can write an **MCP server** in Python (week 15): the `mcp` SDK, the three transports, tool/resource/prompt primitives, and the security review (a tool is RCE). We stand up four servers this week.
- You have stood up **vLLM** (weeks 6 and 19) and put **LiteLLM** in front of it (week 19). We serve the capstone's local tier on it.
- You can run a **Ragas** suite and a **calibrated LLM-as-judge** (week 12), and instrument a system with **OpenTelemetry Gen-AI conventions** exporting to **Langfuse** and **Phoenix** (week 18). We make both load-bearing here.

You need access to a GPU for the vLLM tier. A rented L4 or A10 (~$0.50/h) carries the 7B local tier; the lab budget assumes a few hours of rented compute. An Anthropic API key is required for the vendor tier (`claude-opus-4-8`) and the LLM-as-judge — the eval and the hard-route serving both call it. If you have no GPU at all, the README documents an Ollama fallback for the local tier so the integration still runs end-to-end (slower, lower throughput, but the wiring is identical).

## Topics covered

- **The supervisor graph:** LangGraph state machine where the supervisor *routes* rather than *does*; explicit plan/route/retrieve/code/write/critique nodes; conditional edges driven by the router's decision; the SQLite checkpointer for resume-after-crash; per-route step/token/time/cost budgets that abort runaway loops.
- **Multi-agent handoff:** the retrieval-agent (calls hybrid retrieval + the corpus MCP server), the code-agent (runs the calculator/Python MCP tool), the writing-agent (synthesizes a grounded answer), the critique-agent (checks the draft against the retrieved context before it ships).
- **The MCP tool surface:** four servers with the official `mcp` Python SDK — filesystem, calculator, web-fetch, custom corpus-search — over stdio (for local agents) and streamable HTTP (for the deployed surface); argument validation, path-traversal defense, and rate limiting on every tool (a tool is an RCE primitive).
- **Serving the two tiers:** vLLM continuous-batching cluster for the local 7B/13B tier; LiteLLM as the OpenAI-compatible router with a vendor fallback to `claude-opus-4-8`; a small classifier that decides easy-vs-hard and routes accordingly; per-request cost accounting.
- **The eval suite:** Ragas (faithfulness, context recall, context precision, answer relevancy) over a 100-question gold set; a calibrated LLM-as-judge (`claude-opus-4-8` as judge, 10 human-labeled calibration examples) on a 50-question subset; a CI gate that fails the build if the suite regresses below threshold.
- **Tracing the system:** OpenTelemetry Gen-AI semantic conventions on every agent step, tool call, and model request; export to Langfuse (self-hosted) and Arize Phoenix; reading a trace to find a misroute, a budget abort, or a failed tool call.
- **Scoping under a deadline:** the thin-slice-first methodology; triaging deepening work by what the eval flags; writing down the cut list (what you dropped and why) as a first-class deliverable.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                              | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-------------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | The supervisor graph; routing vs doing; budgets; the thin slice   |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | MCP tool surface (4 servers, 2 transports); wiring agents to tools |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Serving — vLLM cluster + LiteLLM router + cost-tracked routing     |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | The eval suite (Ragas + calibrated judge) + OTel tracing wiring    |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | End-to-end integration; the eval-green gate; cut-list memo         |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Capstone deep work — close the loop, ship the artifact            |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, cut-list polish                                     |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                                   | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | LangGraph, MCP SDK, vLLM/LiteLLM, Ragas, OTel Gen-AI, and the Anthropic SDK references |
| [lecture-notes/01-supervisor-graph-and-mcp-surface.md](./lecture-notes/01-supervisor-graph-and-mcp-surface.md) | The supervisor-routes-not-does pattern, budgets, the four MCP servers, and the two transports |
| [lecture-notes/02-serving-eval-and-tracing.md](./lecture-notes/02-serving-eval-and-tracing.md) | The vLLM/LiteLLM two-tier serving stack, the Ragas + calibrated-judge eval suite, and OTel Gen-AI tracing |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-route-the-supervisor.md](./exercises/exercise-01-route-the-supervisor.md) | Build the supervisor router node and read its routing decisions in a trace |
| [exercises/exercise-02-mcp-corpus-server.py](./exercises/exercise-02-mcp-corpus-server.py) | Write the custom corpus-search MCP server and harden one tool against path traversal |
| [exercises/exercise-03-cost-tracked-router.py](./exercises/exercise-03-cost-tracked-router.py) | Build the easy-vs-hard classifier and route local-vs-vendor with per-request cost accounting |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-end-to-end-green.md](./challenges/challenge-01-end-to-end-green.md) | Wire the thin slice end-to-end and get the eval suite green on the 100-question gold set |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the Sprint B cut-list memo |
| [mini-project/README.md](./mini-project/README.md) | The capstone culmination — ship the runnable Production Agentic Research Assistant |

## The "the slice is green" promise

C23 uses a recurring marker for every step that ends in the integration actually working end-to-end, scored:

```
$ python -m capstone.eval run --gold gold/eval_100.jsonl --gate
[supervisor] q07 "What indemnity cap applies to data-breach claims?"
  route=retrieval -> retrieval_agent -> corpus.search (mcp/stdio) -> 4 chunks
  route=write     -> writing_agent   -> draft (claude-opus-4-8, hard route)
  route=critique  -> critique_agent  -> grounded ✓
  trace: langfuse/trace/3f9c...  phoenix/span/aa12...
ragas: faithfulness=0.91 context_recall=0.88 context_precision=0.84 answer_relevancy=0.90
judge (calibrated): 0.87 on 50-q subset
GATE: PASS (faithfulness 0.91 >= 0.85, judge 0.87 >= 0.80)
```

If that gate prints `FAIL`, the build does not ship — and the trace link in the line above is how you find *which* route or tool dragged the score down. The point of Sprint B is to make that gate go green on the full 100-question set, with the trace proving each route did what the line says.

## Stretch goals

If you finish the regular work early and want to push further:

- **Add a `pause_turn` / human-in-the-loop gate** on the code-agent: before the calculator/Python MCP tool runs a generated expression, surface it for approval. Measure how often the agent's generated expression was wrong (you just built a cheap eval).
- **Run the supervisor with `claude-opus-4-8` adaptive thinking on the routing decision** (`thinking={"type":"adaptive"}`, `output_config={"effort":"medium"}`) and compare route accuracy against a no-thinking baseline on a 25-query routing gold set. Does the thinking pay for itself?
- **Wire prompt caching on the retrieval-agent's system prompt** (the big grounded-answer instruction block is stable; the query is volatile). Verify `cache_read_input_tokens > 0` on the second query, and chart the cost delta across the 100-question run.
- **Stand up the MCP corpus server over streamable HTTP behind LiteLLM's auth** and consume it from Claude Desktop or Cursor as a sanity check that your tool surface is genuinely cross-client, not just LangGraph-internal.

## Up next

Week 24 is the final week: **Chaos Drill, Eval-in-Prod, On-Call.** You'll take the system you ship this week and *break it on purpose* — kill a vLLM replica and verify LiteLLM fails over, inject a prompt-injection attack through a retrieved document and verify your defenses hold, corrupt 5% of the vector store and measure the Ragas-faithfulness regression before restoring from backup. Then you'll write the postmortem. Everything you build this week is the thing you'll attack next week, so build it to be traced, budgeted, and recoverable — because next week you find out, in a controlled window, whether it actually is.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
