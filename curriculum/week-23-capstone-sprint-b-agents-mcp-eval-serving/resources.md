# Week 23 — Resources

Every resource here is **free** or has a free tier. LangGraph, the MCP Python SDK, vLLM, LiteLLM, Ragas, Langfuse, Arize Phoenix, and the OpenTelemetry Gen-AI conventions are all open source. The Anthropic SDK is free to install; the vendor tier (`claude-opus-4-8`) and the LLM-as-judge consume paid API tokens, but the lab budget is small and an open-only judge fallback is documented for every step that uses one.

Library names and APIs move every cohort — the *concepts* (supervisor routing, MCP transports, two-tier serving, Ragas metrics, OTel spans) are stable. When a specific page 404s, search the project's docs for the function name.

This week sits on top of Sprint A (week 22) and pulls forward weeks 13 (LangGraph), 15 (MCP), 18 (observability), 19 (vLLM/LiteLLM), and 12 (Ragas). The retrieval, memory, MCP, serving, eval, and tracing packages you built in those weeks are imported here — the resources below assume you have them.

## Required reading (work it into your week)

- **LangGraph — multi-agent / supervisor** — the canonical reference for the supervisor pattern, conditional edges, and handoff between agents. Read the supervisor and the persistence (checkpointer) sections twice:
  <https://langchain-ai.github.io/langgraph/concepts/multi_agent/>
- **Model Context Protocol — Python SDK** — the official `mcp` SDK: writing a server, the `@server.list_tools()` / `@server.call_tool()` primitives, and the three transports (stdio, SSE, streamable HTTP):
  <https://github.com/modelcontextprotocol/python-sdk>
- **vLLM — OpenAI-compatible server** — `vllm serve`, continuous batching, the `/v1/chat/completions` surface that LiteLLM routes to:
  <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
- **Ragas — metrics** — faithfulness, context recall, context precision, answer relevancy; the four metrics the eval gate checks:
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>

## The agent + MCP references

- **LangGraph — persistence & checkpointers** — the SQLite checkpointer that makes the supervisor resume after a process kill; threads and state:
  <https://langchain-ai.github.io/langgraph/concepts/persistence/>
- **MCP specification** — the protocol spec: tool / resource / prompt primitives, the transport definitions, and the lifecycle. Read the transports section:
  <https://modelcontextprotocol.io/specification>
- **MCP — TypeScript SDK** — the TS counterpart, useful if you expose any tool surface in TypeScript or test against Cursor:
  <https://github.com/modelcontextprotocol/typescript-sdk>
- **Anthropic — agent design patterns** — bash-vs-dedicated-tool, when to promote a tool, context management for long agent runs; the senior heuristics behind your tool surface:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>

## The serving stack (have these open on Wednesday)

- **vLLM (`vllm`)** — `pip install vllm`. `vllm serve <model>` brings up the OpenAI-compatible server with continuous batching and paged attention:
  <https://docs.vllm.ai/en/latest/>
- **LiteLLM proxy** — `pip install "litellm[proxy]"`. The OpenAI-compatible router that fronts vLLM *and* the vendor model, with fallbacks, retries, and per-key cost tracking:
  <https://docs.litellm.ai/docs/proxy/quick_start>
- **LiteLLM — routing & fallbacks** — the `fallbacks` and `model_list` config that fails a hard route over to `claude-opus-4-8` (and fails a dead vLLM replica over to a live one — you'll lean on this in week 24's chaos drill):
  <https://docs.litellm.ai/docs/routing>
- **Ollama** — the no-GPU fallback for the local tier; the wiring to LiteLLM is identical to vLLM's OpenAI surface, only slower:
  <https://github.com/ollama/ollama>

## Eval & observability (the gate and the trace)

- **Ragas — faithfulness** — the "is the answer grounded in the retrieved context?" metric; the spine of the eval gate:
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/>
- **Ragas — context precision / recall** — the retrieval-quality metrics that tell you whether the failure is the retriever or the generator:
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/>
- **OpenTelemetry — Gen-AI semantic conventions** — the cross-vendor span/attribute standard for LLM and agent traces; the schema your spans follow so Langfuse and Phoenix both understand them:
  <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
- **Langfuse (self-hosted)** — `docker compose up` the Langfuse stack; the OTel-compatible tracing + prompt-management backend:
  <https://langfuse.com/docs/deployment/self-host>
- **Arize Phoenix** — `pip install arize-phoenix`. The open tracing + eval-in-prod backend you'll lean on in week 24:
  <https://docs.arize.com/phoenix>

## The vendor tier & the judge (the Anthropic SDK)

- **Anthropic Python SDK** — `pip install anthropic`. The vendor frontier model for the hard routes and the LLM-as-judge. Default model `claude-opus-4-8`; adaptive thinking via `thinking={"type":"adaptive"}` and depth via `output_config={"effort":"..."}`:
  <https://github.com/anthropics/anthropic-sdk-python>
- **Anthropic — tool use** — the `tool_use` surface, if you call the vendor model with tools directly rather than through MCP:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>
- **Anthropic — prompt caching** — cache the stable grounded-answer system prompt so the volatile query is the only uncached part; verify with `usage.cache_read_input_tokens`:
  <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>

## Models you'll use this week

- **`claude-opus-4-8`** — the vendor frontier model for the hard routes *and* the calibrated LLM-as-judge. 1M context, adaptive thinking only (`thinking={"type":"adaptive"}`); control depth with `output_config={"effort":"low"|"medium"|"high"|"xhigh"|"max"}`. Do **not** pass `budget_tokens` or `temperature` — both 400 on this model.
- **`claude-haiku-4-5`** — the cheap option for the easy-vs-hard classifier if you route classification to a vendor model instead of a local one; 200K context, fast.
- **A local 7B/13B (`Qwen2.5-7B-Instruct` or `Qwen2.5-14B-Instruct`)** — served on vLLM as the local tier; the easy routes land here.
- **`BAAI/bge-reranker-v2-m3` + `BAAI/bge-large-en-v1.5`** — the reranker and embedding from Phase II, unchanged; the retrieval-agent calls the hybrid pipeline you built in Sprint A.

## Tools you'll use this week

- **`langgraph` / `langchain-core`** — the supervisor state graph and the SQLite checkpointer.
- **`mcp`** — `pip install mcp`. The official MCP Python SDK for the four tool servers.
- **`vllm`** — `pip install vllm`. The local-tier serving (GPU). Ollama is the no-GPU fallback.
- **`litellm[proxy]`** — `pip install "litellm[proxy]"`. The OpenAI-compatible router with fallbacks and cost tracking.
- **`ragas`** — `pip install ragas`. The eval-suite metrics.
- **`anthropic`** — `pip install anthropic`. The vendor tier and the judge.
- **`opentelemetry-sdk` + `openinference-instrumentation`** — OTel spans following the Gen-AI conventions, exported to Langfuse and Phoenix.
- **The Sprint A capstone package** — your week-22 repo. This week imports the retrieval pipeline, the memory tiers, and the gold set **unchanged**.

## A note on the corpus and the gold set

The capstone runs against the **10 GB private corpus** from Sprint A and the **100-question gold set** the syllabus specifies for the final eval. Sprint A landed retrieval and memory over that corpus; this week's job is to make the *agents* use them and to *score* the result. The gold set is in `{"query", "answer", "relevant_doc_ids"}` form — Ragas needs the reference answer for answer-relevancy, and the retrieval metrics need the relevant doc ids. If your Sprint A gold set is only 40 questions (the Phase II milestone size), extend it to 100 this week — the final-capstone rubric grades on the 100-question set, and a thin gold set is the most common reason a "green" eval doesn't generalize.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Supervisor** | The LangGraph node that *routes* a turn to a subordinate agent; it decides, it doesn't do. |
| **Subordinate agent** | A specialist node — retrieval, code, writing, critique — that the supervisor hands a turn to. |
| **Handoff** | The supervisor passing control (and state) to a subordinate agent and back. |
| **Conditional edge** | A LangGraph edge whose target depends on state (the router's decision). |
| **Checkpointer** | LangGraph's persistence layer (SQLite here) that lets the graph resume after a crash. |
| **Budget** | A per-route limit (steps / tokens / time / cost) that aborts a runaway loop. |
| **MCP server** | A process exposing tools/resources/prompts over a transport (stdio / SSE / streamable HTTP). |
| **stdio transport** | MCP over a subprocess's stdin/stdout; the default for local tools. |
| **streamable HTTP** | MCP over HTTP; the transport for a deployed, cross-client tool surface. |
| **Continuous batching** | vLLM's scheduler interleaving requests at the token level for high throughput. |
| **LiteLLM router** | The OpenAI-compatible proxy fronting vLLM + vendor, with fallbacks and cost tracking. |
| **Easy-vs-hard route** | The classifier decision sending a query to the local tier (easy) or the vendor (hard). |
| **Faithfulness** | Ragas metric: is the answer grounded in the retrieved context? |
| **Context precision/recall** | Ragas retrieval metrics: did you retrieve the right chunks, and only them? |
| **LLM-as-judge (calibrated)** | A model scoring answers, calibrated against human-labeled examples. |
| **OTel Gen-AI span** | A trace span following the Gen-AI semantic conventions (model, tokens, tool name). |
| **The gate** | The CI check that fails the build if the eval suite regresses below threshold. |
| **Cut list** | The written record of what you dropped from the capstone, and why. |

---

*If a link 404s, please open an issue so we can replace it.*
