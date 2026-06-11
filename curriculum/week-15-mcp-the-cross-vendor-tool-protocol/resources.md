# Week 15 — Resources

Every resource here is **free** or has a free tier. The Model Context Protocol is an open standard; the official SDKs (Python, TypeScript) are open source. Claude Desktop and Cursor are free clients you can test against. The OpenClaw-family tooling is open source. Model cards live on Hugging Face.

The protocol moves on a dated-revision cadence (`2025-06-18` is the current spec revision as of this cohort) — the *concepts* (host/client/server, tools/resources/prompts, the JSON-RPC handshake, the transport split) are stable. When a specific spec page 404s, search the spec site for the method name (`tools/call`, `initialize`).

This week sits on top of weeks 13–14. The LangGraph agent and the tool-calling intuition come from there; the resources below assume you can stand up a state graph and read a trace.

## Required reading (work it into your week)

- **The MCP specification (`2025-06-18`)** — the canonical reference for the architecture, the lifecycle, and the JSON-RPC messages. Read the *Architecture*, *Lifecycle*, and *Tools* sections twice:
  <https://modelcontextprotocol.io/specification/2025-06-18>
- **MCP introduction & core concepts** — the gentle on-ramp: what a host/client/server is, and what the three primitives are for:
  <https://modelcontextprotocol.io/docs/concepts/architecture>
- **The official Python SDK (`mcp`)** — `FastMCP`, the `@mcp.tool()`/`@mcp.resource()`/`@mcp.prompt()` decorators, and the transport runners. Read the server quickstart and the `FastMCP` reference:
  <https://github.com/modelcontextprotocol/python-sdk>
- **Anthropic's MCP connector docs** — how Claude (the API and Claude Desktop) consumes MCP servers, including the `mcp_servers` request parameter and the `anthropic[mcp]` conversion helpers:
  <https://platform.claude.com/docs/en/agents-and-tools/mcp>

## The protocol references

- **MCP — Transports** — the stdio / SSE / streamable-HTTP split, the message framing, and when to use each. The streamable-HTTP section is the one that matters for remote servers:
  <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>
- **MCP — Tools** — the `tools/list` and `tools/call` methods, the tool schema shape, and the result/`isError` contract:
  <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>
- **MCP — Resources** — `resources/list`, `resources/read`, resource templates (URI templates), and subscriptions:
  <https://modelcontextprotocol.io/specification/2025-06-18/server/resources>
- **MCP — Prompts** — `prompts/list` and `prompts/get`; how a prompt becomes a user-invoked slash-command in a client:
  <https://modelcontextprotocol.io/specification/2025-06-18/server/prompts>
- **JSON-RPC 2.0 spec** — the wire format MCP rides on; you only need the request/response/notification shapes:
  <https://www.jsonrpc.org/specification>

## The SDKs (have these open on Tuesday)

- **`mcp` (Python SDK)** — `pip install "mcp[cli]"`. `FastMCP` for servers, `ClientSession` + `stdio_client` / `streamablehttp_client` for clients, and the `mcp dev` inspector:
  <https://github.com/modelcontextprotocol/python-sdk>
- **`@modelcontextprotocol/sdk` (TypeScript SDK)** — the canonical TS server/client; the same primitives, the same transports, for when your tool surface lives in a Node service:
  <https://github.com/modelcontextprotocol/typescript-sdk>
- **`langchain-mcp-adapters`** — converts MCP tools into LangChain/LangGraph tools so your week-13 agent can call them. `pip install langchain-mcp-adapters`:
  <https://github.com/langchain-ai/langchain-mcp-adapters>
- **`anthropic[mcp]`** — `pip install "anthropic[mcp]"`. The `anthropic.lib.tools.mcp` helpers (`mcp_tool`, `async_mcp_tool`) convert MCP tools for Claude's tool runner; or use the API's `mcp_servers` parameter to let Claude connect directly:
  <https://github.com/anthropics/anthropic-sdk-python>

## Clients to test against

- **The MCP Inspector** — `npx @modelcontextprotocol/inspector` (or `mcp dev server.py`) — a browser UI that connects to your server, lists its tools/resources/prompts, and lets you call them and watch the JSON-RPC. Exercise 1 lives here:
  <https://github.com/modelcontextprotocol/inspector>
- **Claude Desktop** — the reference MCP client. Add a server to `claude_desktop_config.json` and it appears as a tool surface in the chat. Free download:
  <https://modelcontextprotocol.io/quickstart/user>
- **Cursor** — the AI code editor; supports MCP servers via `.cursor/mcp.json`. A second client to prove the "any client" promise:
  <https://docs.cursor.com/context/model-context-protocol>

## Security (the tool-is-RCE spine — rehearsed again in week 17)

- **MCP security best practices** — the spec's own security guidance: input validation, the confused-deputy problem, token passthrough, and consent:
  <https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices>
- **OWASP LLM Top 10** — the broader threat catalog; `LLM01` (prompt injection) and the tool/agent entries frame *why* an MCP tool is an attack surface. You'll meet this in depth in week 17:
  <https://genai.owasp.org/llm-top-10/>
- **CWE-22: Path Traversal** — the canonical reference for the filesystem-server defense you build in Exercise 2 and the challenge:
  <https://cwe.mitre.org/data/definitions/22.html>

## The OpenClaw family (open MCP ecosystem)

The **OpenClaw family** is this cohort's shorthand for the open-source, MCP-native tooling ecosystem: gateways that aggregate many servers behind one endpoint, self-hosted MCP servers you can run on your own hardware, and community-maintained Claude-compatible agent runtimes that speak MCP as their native tool protocol. Specific projects rotate every cohort; the *pattern* — open runtimes that consume and route MCP — is stable.

- **Awesome MCP Servers** — the community catalog of open MCP servers (filesystem, git, database, search, browser, …). Browse it to see what a mature tool surface looks like before you write your own:
  <https://github.com/modelcontextprotocol/servers>
- **Open MCP gateways / aggregators (OpenClaw family)** — the "one endpoint, many servers" pattern: a gateway speaks MCP to the client and fans out to N upstream servers. Search the awesome-MCP catalog for "gateway", "proxy", and "router" entries; the stretch goal stands one up.
- **Open Claude-compatible runtimes (OpenClaw family)** — community agent loops that consume MCP natively, so the same server you wrote works in a self-hosted runtime exactly as it does in Claude Desktop. The point of the family is that MCP is *open*: no single vendor owns the client side.

## Model used in the consumption path

- **`claude-opus-4-8`** — the model behind the Claude Desktop / Claude-API consumption path. MCP servers are vendor-agnostic; this is just one client. An open-only fallback (a local model from week 6 driven by a programmatic `ClientSession`) is documented for every lab, so no Anthropic key is required to complete the week.

## Tools you'll use this week

- **`mcp` / `mcp[cli]`** — `pip install "mcp[cli]"`. The Python server/client SDK + the `mcp dev` inspector.
- **`langchain-mcp-adapters`** — convert MCP tools into LangGraph tools for the consuming agent.
- **`httpx`** — async HTTP client; the streamable-HTTP transport uses it under the hood, and you'll use it directly in the transport clinic.
- **`uv` (optional)** — `uvx` / `uv run` is the lowest-friction way to launch MCP servers, and it's what most client configs reference. `pip` + a venv works identically.
- **The MCP Inspector** — `npx @modelcontextprotocol/inspector` — the wire-level debugger for Exercise 1.
- **Your week-13 LangGraph agent** — this week plugs MCP tools into it via `langchain-mcp-adapters`.

## A note on the corpus

The custom-domain MCP server (the second server in the challenge and mini-project) wraps the same small **legal corpus** you've used since week 7 — the synthetic services agreement of ~50 clauses plus the gold set. That keeps the tool *testable*: a `search_corpus` call for "five-year confidentiality" should return `clause_09`, and you can assert it. The filesystem server operates on a throwaway sandbox directory. No tool in this week should touch anything outside its declared sandbox — that's the whole security point.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **MCP** | Model Context Protocol — the open JSON-RPC standard for connecting LLM apps to tools/context. |
| **Host** | The LLM application (Claude Desktop, Cursor, your agent) that wants to use tools. |
| **Client** | The connector inside the host; one client per server, 1:1. |
| **Server** | The process that exposes tools/resources/prompts. What you write this week. |
| **Tool** | A model-controlled, callable function with a JSON-schema input (`tools/call`). |
| **Resource** | App-controlled readable context (a file, a query result) addressed by URI (`resources/read`). |
| **Prompt** | A user-controlled reusable template, often surfaced as a slash-command (`prompts/get`). |
| **stdio transport** | Server runs as a subprocess; messages over stdin/stdout. The local default. |
| **streamable HTTP** | The current remote transport: one HTTP endpoint, optional SSE upgrade for streaming. |
| **SSE transport** | The deprecated remote transport (two endpoints). Know it; don't build new on it. |
| **`initialize`** | The handshake that negotiates protocol version + capabilities before any tool call. |
| **JSON-RPC** | The wire format: `{method, params, id}` request → `{result \| error, id}` response. |
| **`FastMCP`** | The high-level Python server class; decorators turn typed functions into MCP primitives. |
| **OpenClaw family** | This cohort's name for the open MCP ecosystem: gateways, self-hosted servers, Claude-compatible runtimes. |
| **Tool-is-RCE** | A tool call is a request to run code; validate every argument like it's hostile. |

---

*If a link 404s, please open an issue so we can replace it.*
