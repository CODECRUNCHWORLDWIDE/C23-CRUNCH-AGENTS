# Week 15 — MCP, the Cross-Vendor Tool Protocol

Welcome to the week your tools stop being glued to one vendor's SDK. Back in week 4 you wrote tool schemas as JSON blobs that lived inside a single `messages.create` call, and in week 13 you wired those tools into a LangGraph node by hand. Every one of those tools was married to the framework that called it. This week you divorce them. You will learn the **Model Context Protocol (MCP)** — the open, JSON-RPC-based standard that lets a tool surface live in its own process, behind a stable interface, and be consumed by *any* MCP-aware client: Claude Desktop, Cursor, a LangGraph agent, a programmatic Python client, or a self-hosted runtime. Write the tool once; plug it in everywhere.

This is week 3 of **Phase III — Agents & Orchestration**, and it sits on top of week 13 (LangGraph) and week 14 (Mastra/Inngest). Everything here assumes you can stand up an agent loop, define a tool, and read a trace. What changes this week is *where the tool lives* and *how the agent reaches it*. The headline lab is two MCP servers — one filesystem surface, one private-corpus search — exposed over both stdio and streamable HTTP, consumed from a LangGraph agent, then put through a real security review.

The one sentence to internalize before you read another line:

> **MCP is the USB-C of agent tooling. It is not the future — it is the present, and it is open.**

Here's why that's not marketing. Before MCP, every integration was an N×M problem: N agent frameworks times M tools, each pair hand-wired. A filesystem tool written for the OpenAI Agents SDK didn't work in LangGraph without a rewrite; a database tool written for Claude Desktop didn't work in Cursor. MCP collapses that to N+M: a tool speaks MCP once (the server), a client speaks MCP once (the host), and any client talks to any server. The protocol is the contract. Your job this week is to write both sides of that contract correctly — and to remember that a tool is a remote-code-execution primitive, so the contract had better be airtight.

There's a corollary worth taping to your monitor:

> **A tool exposed over MCP is an API exposed to an untrusted client.** The model is the client. Treat every tool argument like it came from the public internet — because, by way of a prompt injection, it might have.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** the MCP architecture — host, client, server — and the three primitives a server exposes: **tools** (callable functions), **resources** (readable context), and **prompts** (reusable templates).
- **Enumerate** the three MCP transports — **stdio**, **SSE** (legacy), and **streamable HTTP** (the current remote standard) — and choose the right one for local-process versus networked deployment.
- **Write** an MCP server in Python with the official `mcp` SDK (`FastMCP`), exposing typed tools, resources, and prompts, and run it over stdio and streamable HTTP.
- **Consume** an MCP server three ways: from Claude Desktop / Cursor via config, from a programmatic Python client (`ClientSession`), and from a **LangGraph** agent via `langchain-mcp-adapters`.
- **Reason** about the JSON-RPC message flow — `initialize`, `tools/list`, `tools/call`, `resources/read` — and read an MCP session at the wire level.
- **Secure** an MCP server: validate every argument, defend against path traversal, rate-limit expensive tools, scope capabilities, and treat the tool surface as an attack surface (a dress rehearsal for week 17's red-team).
- **Situate** MCP in the 2026 ecosystem — the **OpenClaw family** of open MCP gateways, self-hosted servers, and community-maintained Claude-compatible agent runtimes that speak MCP natively.
- **Build** the `crunchmcp` toolkit: two production-shaped MCP servers with a documented tool surface, multi-transport support, a consuming LangGraph agent, and a security-review memo.

## Prerequisites

This week assumes you have completed **C23 weeks 1–14**, or have equivalent fluency. Specifically:

- You finished **week 13** and can build a **LangGraph** state graph with tool-calling nodes. This week plugs MCP tools into that graph via `langchain-mcp-adapters` — if your week-13 agent is broken, fix it first.
- You remember **week 4**'s tool-calling material: a tool is a JSON schema (name, description, `input_schema`) plus a function. MCP is that idea, moved into its own process and standardized.
- Python 3.12 on Linux, macOS, or WSL2; a virtualenv you can `pip install` into; comfort with `async`/`await` (MCP is async to the core — week 17 of C17, or the async chapters of weeks 4–6, cover what you need).
- You can read JSON-RPC conceptually — a request has a `method`, `params`, and an `id`; a response carries a `result` or an `error`. We start from the wire and build up.

You do **not** need a GPU (MCP is I/O-bound protocol plumbing). You do **not** need an Anthropic API key for the server work — servers are vendor-agnostic — though one of the consumption paths uses Claude (`claude-opus-4-8`) and Claude Desktop, and an open-only fallback (a local model from week 6 via a programmatic client) is documented for every lab.

## Topics covered

- **Why a protocol at all:** the N×M integration explosion, the N+M collapse MCP buys, and why "write the tool once" is the whole value proposition.
- **The architecture:** host (the LLM application), client (one per server connection, lives in the host), server (the tool/resource/prompt provider) — and the strict 1:1 client-server pairing.
- **The three primitives:** **tools** (model-controlled, side-effecting actions), **resources** (application-controlled, readable context like files or query results), **prompts** (user-controlled, reusable templates surfaced as slash-commands).
- **The three transports:** **stdio** (local subprocess, the default for desktop clients), **SSE** (the deprecated remote transport — know it exists, don't build new on it), **streamable HTTP** (the current remote standard — one endpoint, optional SSE upgrade for streaming).
- **Writing a server:** `FastMCP`, the `@mcp.tool()` / `@mcp.resource()` / `@mcp.prompt()` decorators, typed signatures that become JSON schemas, and the `mcp dev` inspector.
- **Consuming a server:** Claude Desktop / Cursor config files, the programmatic `ClientSession` lifecycle (`initialize` → `list_tools` → `call_tool`), and the LangGraph adapter path.
- **The JSON-RPC layer:** capability negotiation in `initialize`, the `tools/list` and `tools/call` round trips, and reading a session at the wire level.
- **Security review:** a tool is RCE — argument validation, path-traversal defense, rate limiting, capability scoping, and the threat model of an MCP tool surface.
- **The OpenClaw ecosystem:** open MCP gateways (route/aggregate many servers behind one endpoint), self-hosted MCP servers, and community Claude-compatible runtimes that consume MCP natively.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                          | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|---------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Why a protocol; architecture; the three primitives            |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Writing a `FastMCP` server; stdio transport; the exercises    |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Transports (stdio/SSE/streamable HTTP); consuming from clients|    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Security review; the OpenClaw ecosystem; building the toolkit |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | LangGraph consumption + the security memo; transport clinic   |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                         |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                      |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                               | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The MCP spec, the Python/TS SDK docs, transport references, client configs, and the OpenClaw-ecosystem reading |
| [lecture-notes/01-the-protocol-and-the-primitives.md](./lecture-notes/01-the-protocol-and-the-primitives.md) | Why a protocol, the host/client/server architecture, the three primitives, and the JSON-RPC message flow |
| [lecture-notes/02-transports-clients-and-security.md](./lecture-notes/02-transports-clients-and-security.md) | The three transports, consuming from Claude Desktop / programmatic / LangGraph, MCP security review, and the OpenClaw ecosystem |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-trace-an-mcp-session.md](./exercises/exercise-01-trace-an-mcp-session.md) | Run the MCP inspector against a server and read the JSON-RPC handshake at the wire level |
| [exercises/exercise-02-build-a-filesystem-server.py](./exercises/exercise-02-build-a-filesystem-server.py) | Implement a sandboxed filesystem MCP server with `FastMCP` and defend it against path traversal |
| [exercises/exercise-03-consume-from-a-client.py](./exercises/exercise-03-consume-from-a-client.py) | Drive an MCP server from a programmatic `ClientSession`: initialize, list, call, read a resource |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-two-servers-two-transports.md](./challenges/challenge-01-two-servers-two-transports.md) | Two MCP servers over stdio + streamable HTTP, consumed from a LangGraph agent, with a security review |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the one-page MCP tool-surface security memo |
| [mini-project/README.md](./mini-project/README.md) | The `crunchmcp` toolkit — two servers, multi-transport, a LangGraph consumer, and a documented surface |

## The "any client, any server" promise

C23 uses a recurring marker for every exercise that ends in a tool surface that *works across clients without modification*:

```
$ python -m crunchmcp.inspect --server corpus_search --transport stdio
initialize         -> protocolVersion=2025-06-18  capabilities={tools, resources}
tools/list         -> [search_corpus, get_document]
tools/call search_corpus {"query": "five-year confidentiality"}
  -> [{"clause_id": "clause_09", "score": 0.88,
       "text": "...protected for five years after termination."}]  ✓
```

If that same `corpus_search` server, unchanged, also answers a `tools/call` from Claude Desktop, from Cursor, and from your LangGraph agent, the promise is kept: **one server, every client.** The point of week 15 is to make the tool surface *portable* — and to prove it by pointing three different clients at the identical server process.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **MCP specification** (`2025-06-18` revision) end to end until you can explain the `initialize` capability-negotiation handshake from memory: <https://modelcontextprotocol.io/specification/2025-06-18>. Then implement a `prompts/list` + `prompts/get` flow and surface it in Claude Desktop as a slash-command.
- Build a **third transport path**: take your filesystem server and run it over streamable HTTP behind a reverse proxy with bearer-token auth. Confirm an unauthenticated `tools/call` is rejected at the proxy, before it reaches your server.
- Stand up an **OpenClaw-family MCP gateway** (an open aggregator) in front of both your servers, so a single client connection sees the union of both tool surfaces. Measure the added latency per `tools/call` hop.
- Add **resource subscriptions**: expose a resource that changes (a log file), subscribe to it from a client, and verify the client receives `notifications/resources/updated` when it changes.

## Up next

Week 16 steps off the orchestration track and asks a different question: when prompt-and-retrieve isn't enough, do you fine-tune? You'll build an SFT dataset, train a LoRA adapter on a 7B model, and decide — with a measured eval — whether the fine-tune was worth it. Then week 17 comes back to this week's tool surface and *attacks* it: the red-team lab takes your week-15 MCP server and tries to break it. Push your `crunchmcp` toolkit before you start week 17; the safety lab inherits it directly.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
