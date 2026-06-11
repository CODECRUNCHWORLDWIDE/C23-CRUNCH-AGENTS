# Lecture 2 — Transports, Clients, and Security: How the Messages Travel, Who Consumes Them, and Why a Tool Is RCE

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can choose the right transport (stdio vs SSE vs streamable HTTP) for a deployment, consume an MCP server three ways (Claude Desktop config, a programmatic `ClientSession`, a LangGraph agent via `langchain-mcp-adapters`), run a real security review of a tool surface (argument validation, path-traversal defense, rate limiting, capability scoping), and situate MCP in the open OpenClaw ecosystem.

Lecture 1 was *what the protocol is and what a server exposes*. This lecture is *how the JSON-RPC messages physically travel, who consumes them, and how an adversary attacks the tool surface* — because the cleanest server design in the world is a liability if it's reachable over the wrong transport or accepts an unvalidated path argument.

> **The transport is a deployment decision, not a protocol decision.** The same server logic runs over stdio (local subprocess) or streamable HTTP (networked) — you choose by *where the server lives relative to the host*, and the security posture changes completely between the two.

---

## Part 1 — The three transports

A transport is the pipe that carries JSON-RPC messages between client and server. MCP defines three, and the choice is almost entirely about *locality*: is the server a local subprocess, or a networked service?

### 1.1 stdio — the local subprocess transport

**stdio** is the default and the simplest. The host *spawns the server as a child process* and talks to it over the child's **stdin/stdout**: the client writes JSON-RPC requests to the server's stdin, the server writes responses to its stdout. (stderr is free for logging — never write protocol messages to stdout's neighbor by accident, or you'll corrupt the stream.)

stdio is the right transport when the server runs on the *same machine* as the host: Claude Desktop launching a local filesystem server, Cursor launching a local git server, your dev box running a corpus-search server. Its properties:

- **No network, no ports, no auth handshake** — the security boundary is the OS process boundary. If you can spawn the process, you can talk to it; there's no remote attacker because there's no remote surface.
- **One client, one server, one process** — stdio is inherently 1:1 and local. You cannot have two hosts share one stdio server; each host spawns its own.
- **Launched by the client** — the host config says "run `python server.py`" (or `uvx some-server`), and the host manages the subprocess lifecycle.

```python
# A FastMCP server defaults to stdio. This is the whole runner:
if __name__ == "__main__":
    mcp.run()                 # transport="stdio" is the default
    # equivalently: mcp.run(transport="stdio")
```

The client side (programmatic) spawns the subprocess and connects:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command="python", args=["server.py"])
async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()                 # the handshake (Lecture 1 §5.1)
        tools = await session.list_tools()
        result = await session.call_tool("add", {"a": 2, "b": 3})
```

That `read, write` pair *is* the stdio pipe — the client owns the subprocess's stdout (read) and stdin (write). When the `async with` blocks exit, the subprocess is terminated. This is Exercise 3.

### 1.2 SSE — the deprecated remote transport (know it, don't build on it)

Before streamable HTTP, MCP's remote transport was **HTTP+SSE**, which used *two* endpoints: a POST endpoint for the client to send messages, and a separate **Server-Sent Events** stream for the server to push messages back. It worked, but the two-endpoint design was awkward (session correlation across two connections, reconnection complexity, statefulness headaches).

You need to know SSE exists for two reasons: you'll encounter older servers and configs that use it, and the transport name still shows up in the SDKs (`mcp.client.sse`). But **do not build new servers on SSE.** It is superseded by streamable HTTP. The one-line takeaway: *SSE is the legacy remote transport; recognize it, migrate off it, don't start on it.*

### 1.3 Streamable HTTP — the current remote standard

**Streamable HTTP** is the modern remote transport and the one to use for any networked MCP server. It collapses SSE's two endpoints into **one**: the client POSTs JSON-RPC to a single MCP endpoint (e.g. `https://tools.example.com/mcp`), and the server responds with either a plain JSON response *or*, when it wants to stream (progress, partial results), upgrades that same response to an SSE stream. One endpoint, optional streaming, far simpler session handling.

Streamable HTTP is the right transport when the server is a *networked service* — a corpus-search server running in your cluster that multiple hosts hit, a tool surface you expose to Claude via the API's `mcp_servers` parameter, a server behind a load balancer. Its properties:

- **One endpoint, real HTTP** — it lives at a URL, behind your normal web infrastructure (reverse proxy, TLS, auth middleware, rate limiter).
- **Stateless-friendly** — designed so the server can be horizontally scaled; session state is carried in headers, not pinned to one long-lived socket.
- **Auth is your job** — and this is the crux: a networked server is *reachable by remote attackers*, so authentication, authorization, and rate limiting are no longer optional. With stdio the OS process boundary protected you; with streamable HTTP, *you* are the boundary.

```python
# The SAME server, run over streamable HTTP instead of stdio. The tool/resource
# code is byte-for-byte identical — only the runner changes.
if __name__ == "__main__":
    mcp.run(transport="streamable-http")   # serves on /mcp by default
```

That one-word change — `transport="streamable-http"` — is the whole point of the transport abstraction: **the server logic doesn't know or care how its messages travel.** You write the tools once; you deploy them locally (stdio) or remotely (streamable HTTP) by flipping the runner. The challenge has you run *both* transports against the *same* server to prove it.

### 1.4 Choosing — the decision in one table

| Transport | Use when | Auth | Security boundary |
|---|---|---|---|
| **stdio** | Server is a local subprocess (desktop client, dev box, local tool) | None (OS process) | The OS process boundary |
| **SSE** | *Legacy only* — you're maintaining an old remote server | Bolt-on | HTTP (two endpoints — awkward) |
| **streamable HTTP** | Server is a networked service (cluster, shared, behind a proxy) | **Required** | *You* — TLS, authn/z, rate limiting |

> **The rule:** local server → stdio; networked server → streamable HTTP; SSE only if you inherited it. And the moment you go networked, the security work in Part 3 stops being optional.

---

## Part 2 — Consuming a server three ways

A server is only useful if a client calls it. The "any client" half of the week's promise means the *same server* should answer all three of these consumers without modification. Prove it by pointing all three at one server.

### 2.1 Claude Desktop / Cursor — config-file consumption

Desktop clients consume MCP servers via a JSON config that tells the host *how to launch the server* (for stdio) or *where to reach it* (for HTTP). For Claude Desktop, it's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "corpus-search": {
      "command": "python",
      "args": ["/abs/path/to/corpus_server.py"]
    },
    "filesystem": {
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/abs/path/to/sandbox"]
    }
  }
}
```

Restart the client and the servers' tools appear in the chat — the model can now call `search_corpus`, and a paperclip/tool UI shows what's available. Cursor uses the same shape in `.cursor/mcp.json`. The key insight: **this config is the entire integration.** You wrote no client code, no glue, no adapter — the host already speaks MCP, so pointing it at your server is a few lines of JSON. *That* is N+M in practice: the client integration was written once, by the client author, and your server drops in.

A note on consent: a well-built desktop client *asks before running a side-effecting tool* — a dialog like "Allow `write_file`?" This is the host's consent gate from Lecture 1 §7. When you're testing, you'll click "allow" a lot; in production, that gate is a security control, and a host that auto-approves everything is a host the week-17 red team will love.

### 2.2 A programmatic client — `ClientSession`

When *your code* is the host — a script, a service, an agent — you drive the server with a `ClientSession`. The lifecycle mirrors the protocol exactly (Lecture 1 §7): open a transport, `initialize`, discover, call. Over stdio:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(command="python", args=["corpus_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                       # handshake first

            tools = await session.list_tools()               # tools/list
            print("tools:", [t.name for t in tools.tools])

            result = await session.call_tool(                # tools/call
                "search_corpus", {"query": "five-year confidentiality"}
            )
            print("result:", result.content)

            doc = await session.read_resource(               # resources/read
                "corpus://clause_09"
            )
            print("resource:", doc.contents)


asyncio.run(main())
```

Over streamable HTTP, only the transport opener changes — `from mcp.client.streamable_http import streamablehttp_client` and `async with streamablehttp_client(url) as (read, write, _):` — everything after `ClientSession(...)` is identical. That symmetry is the transport abstraction paying off on the *client* side too. Exercise 3 is this loop end to end.

### 2.3 A LangGraph agent — `langchain-mcp-adapters`

The week's headline consumption path: plug MCP tools into the week-13 LangGraph agent. The `langchain-mcp-adapters` package connects to one or more MCP servers, runs `tools/list`, and converts each MCP tool into a LangChain/LangGraph `BaseTool` your graph can bind to a model node:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Connect to BOTH servers at once; the agent sees the union of their tools.
client = MultiServerMCPClient(
    {
        "filesystem": {
            "command": "python",
            "args": ["fs_server.py"],
            "transport": "stdio",
        },
        "corpus": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
    }
)

tools = await client.get_tools()                 # MCP tools -> LangGraph tools
agent = create_react_agent("anthropic:claude-opus-4-8", tools)
result = await agent.ainvoke(
    {"messages": "Find the confidentiality clause and write it to notes.txt"}
)
```

Notice the agent connects to a **stdio** server and a **streamable-HTTP** server *simultaneously* and sees one merged tool surface — the filesystem tools and the corpus tools, side by side, transport-agnostic. The model calls `search_corpus` (from the HTTP server) and `write_file` (from the stdio server) in the same loop without knowing or caring that they live in different processes over different transports. **That is the whole protocol working:** the agent is portable across servers, the servers are portable across transports, and nothing had to be rewritten to combine them.

For the open-only / vendor-free path, swap the model: point `create_react_agent` at a local model (week 6, via an OpenAI-compatible local endpoint) instead of `claude-opus-4-8`. The MCP machinery is identical — only the model behind the agent changes. The week is completable with zero vendor credentials.

---

## Part 3 — Security review: a tool is RCE

Here is where the corollary from Lecture 1 becomes engineering work:

> **A tool exposed over MCP is an API exposed to an untrusted client.** The model is the client, the model can be steered by a prompt injection, so every tool argument is potentially hostile. Validate it like it came from the public internet.

This section is the dress rehearsal for week 17, where you'll *attack* exactly this surface. Build the defenses now so the red team has something real to test.

### 3.1 Argument validation — the first and most-skipped defense

The model supplies tool arguments, and the model can be manipulated. **Never trust a tool argument.** Validate type, range, and shape *inside the tool*, before you act on it:

```python
@mcp.tool()
def get_clause(clause_id: str) -> str:
    """Fetch one clause by id (e.g. 'clause_09')."""
    # Validate SHAPE before doing anything. clause_id must match a strict pattern,
    # not be an arbitrary string the model (or an injection) chose.
    if not re.fullmatch(r"clause_\d{2}", clause_id):
        raise ValueError(f"invalid clause_id {clause_id!r}; expected clause_NN")
    return CORPUS[clause_id]
```

Type hints get you *part* of the way — `FastMCP` rejects a non-string `clause_id` at the schema layer. But schema validation can't express "must be `clause_` followed by two digits." That semantic constraint is yours to enforce, and skipping it is the single most common MCP vulnerability: a tool that accepts a string and feeds it straight into a path, a query, or a shell.

### 3.2 Path traversal — the filesystem server's defining vulnerability

A filesystem tool that takes a path and reads it is a path-traversal hole waiting to happen. The attack: the model (steered by an injection) calls `read_file("../../../../etc/passwd")` and your "sandboxed" server happily reads outside the sandbox. **CWE-22.** The defense is to resolve the requested path and confirm it stays inside the sandbox root:

```python
from pathlib import Path

SANDBOX = Path("/abs/path/to/sandbox").resolve()


@mcp.tool()
def read_file(relative_path: str) -> str:
    """Read a file from inside the sandbox. Paths are relative to the sandbox root."""
    # Resolve the FULL path, then verify it's still under SANDBOX. resolve()
    # collapses '..' so '../../etc/passwd' resolves to /etc/passwd, which fails
    # the is_relative_to check.
    target = (SANDBOX / relative_path).resolve()
    if not target.is_relative_to(SANDBOX):
        raise ValueError(f"path escapes sandbox: {relative_path!r}")
    if not target.is_file():
        raise FileNotFoundError(relative_path)
    return target.read_text()
```

The load-bearing line is `target.is_relative_to(SANDBOX)` *after* `resolve()`. Resolving first is essential: `resolve()` turns `sandbox/../../../etc/passwd` into the real absolute path, and *then* the containment check catches the escape. Checking before resolving (e.g. naively rejecting strings that contain `..`) is brittle — symlinks, encoded separators, and absolute paths all bypass a string check but not a resolved-containment check. Exercise 2 builds this server and tries to break it; the challenge re-audits it.

### 3.3 Rate limiting — bound the expensive and the abusable

A tool that's cheap for the model to call but expensive for you to serve (an LLM-backed summarizer, a paid API wrapper, a heavy search) needs a rate limit, or a runaway agent loop (or an attacker) can run up cost or DoS the backend. A simple token-bucket or per-window counter inside the tool suffices for a single server:

```python
import time
from collections import deque

_calls: deque[float] = deque()
_MAX_PER_MIN = 30


@mcp.tool()
def expensive_search(query: str) -> list[dict]:
    """Search the corpus (rate-limited to 30 calls/min)."""
    now = time.monotonic()
    while _calls and _calls[0] < now - 60:
        _calls.popleft()
    if len(_calls) >= _MAX_PER_MIN:
        raise RuntimeError("rate limit exceeded; try again shortly")
    _calls.append(now)
    return _do_search(query)
```

For a networked (streamable-HTTP) server, you'd typically push rate limiting *up* into the reverse proxy or gateway as well, so abuse is rejected before it reaches your process. Defense in depth: limit at the edge *and* in the tool.

### 3.4 Capability scoping — least privilege for tool surfaces

Don't expose a `run_shell(command)` tool when the agent only needs to read three specific files. Every tool you expose is attack surface; the smallest surface that does the job is the safest. Scope ruthlessly:

- **Prefer narrow tools over general ones.** `read_clause(clause_id)` (validated, bounded) over `read_file(path)` (path-traversal risk). `query_orders(customer_id)` over `run_sql(query)` (injection risk). The narrower the tool, the smaller its abuse surface.
- **Don't expose write/delete unless required.** A read-only corpus server has a dramatically smaller threat model than one that can also write. If the task is retrieval, ship retrieval only.
- **Separate servers by trust level.** A low-trust public-corpus search and a high-trust internal-database tool shouldn't live in the same server with the same blast radius. Split them; let the host decide which to enable.

### 3.5 The MCP threat model, in one paragraph

The attack chain you'll execute in week 17: a malicious instruction reaches the model (directly in a user message, or *indirectly* via a poisoned document the model retrieved), the model is steered into calling a tool with hostile arguments, and the tool — if unvalidated — executes the attacker's intent (reads `/etc/passwd`, runs arbitrary SQL, exfiltrates data through an outbound call). MCP doesn't introduce this risk; it *concentrates* it, because the protocol makes it trivially easy to bolt powerful tools onto an agent. The defenses above — validate arguments, sandbox paths, rate-limit, scope capabilities, gate consent in the host — are what stand between "convenient tool surface" and "remote code execution for whoever can talk to your model." Write them now; you'll attack them in week 17.

---

## Part 4 — The OpenClaw ecosystem: MCP is open

The last piece is context: MCP is not Anthropic's private protocol with an Anthropic-only client. It's an **open standard**, and a whole ecosystem of open tooling has grown around it. This cohort's shorthand for that ecosystem is the **OpenClaw family** — open MCP gateways, self-hosted MCP servers, and community-maintained Claude-compatible runtimes that speak MCP natively. The specific projects rotate every cohort; the *pattern* is durable, and the pattern is the lesson: **no single vendor owns the client side of MCP.**

Three pieces of the family matter for your literacy:

- **Open MCP gateways / aggregators.** A gateway speaks MCP to the client and fans out to N upstream MCP servers, presenting their *combined* tool surface behind one connection. Instead of your host configuring ten servers, it configures one gateway that routes to ten. This is the "one endpoint, many servers" pattern — useful for centralizing auth, rate limiting, and discovery. The stretch goal stands one up in front of your two servers and measures the per-hop latency.
- **Self-hosted MCP servers.** Because a server is just a JSON-RPC process, you can run it on your own hardware, behind your own network controls, with your own data never leaving your perimeter. The open catalog (Awesome MCP Servers) is full of these — filesystem, git, database, search, browser — and they all work in any MCP client precisely because they speak the open protocol. Your `crunchmcp` servers are self-hosted MCP servers in exactly this sense.
- **Open Claude-compatible runtimes.** Community agent loops that consume MCP as their native tool protocol. The significance: the *same server you wrote this week* runs unchanged in these open runtimes, just as it runs in Claude Desktop. The client side is not vendor-locked. If a runtime speaks MCP, your server works in it — that's the open-standard guarantee made concrete.

The reason this matters beyond trivia: when you build a tool surface on MCP, you are not betting on one vendor. You're building on a protocol that an open ecosystem already implements on both sides. If your favorite client is deprecated, the server you wrote still works in every other MCP client — which is exactly the "the course is the engineering, not the import" contract from the C23 charter, applied to tooling.

---

## Part 4.5 — A worked transport decision, and the auth that comes with it

Let's make the transport choice concrete with a scenario you'll actually face, because the abstract table in §1.4 hides the *consequences* of the choice.

**Scenario:** you've written a corpus-search server. Two teams want it. Team A is a single developer running an agent on their laptop. Team B is a production service in the cluster where ten agent replicas need the same tool surface.

For **Team A**, the answer is stdio. The developer's host (Claude Desktop, or their own agent) spawns the server as a subprocess; the corpus data lives on their machine; there's no network exposure and no auth to configure. The config is three lines of JSON. Done. The security boundary is the developer's own OS — if someone can run processes on that laptop, they have bigger problems than your MCP server.

For **Team B**, the answer is streamable HTTP, and the choice *cascades into real work*:

- The server now lives at a URL (`https://tools.cluster.internal/mcp`) reachable by ten replicas — and, unless you stop it, by anything else that can route to that URL.
- **Authentication becomes mandatory.** A networked endpoint with no auth is an open tool surface; any process that can reach the URL can call `search_corpus` (and any *other* tool you exposed). You put a bearer token (or mTLS, or your cluster's service-mesh identity) in front, so only authorized callers reach the server.
- **Rate limiting moves to the edge.** With ten replicas hammering the server, a per-process rate limit isn't enough — you rate-limit at the reverse proxy or gateway so one misbehaving replica (or an attacker who got a token) can't exhaust the backend.
- **TLS is non-negotiable.** Tool arguments and results travel over the network now; they're plaintext JSON unless you terminate TLS at the proxy.

Here's the payoff that makes the abstraction worth it: **the server's Python is identical for both teams.** The same `search_corpus` function, the same `FastMCP` object, the same `safe_path` validation. Team A runs it with `mcp.run()`; Team B runs it with `mcp.run(transport="streamable-http")` behind their proxy. You wrote the tool once; the *deployment* differs, and the *deployment* is where the auth and rate-limiting work lives. This is precisely why the transport is a deployment decision, not a protocol one — and why the security memo for a networked server is longer than for a local one, even though the tool code is the same.

The general principle for the security memo: **a server's threat model is a function of its transport.** A stdio server's threats are bounded by the OS process model; a streamable-HTTP server inherits the entire threat model of a public (or semi-public) HTTP service — authentication bypass, authorization confusion, the confused-deputy problem (a server that holds credentials and can be tricked into using them on an attacker's behalf), token leakage. The MCP spec's security-best-practices page enumerates these; the moment you flip to streamable HTTP, that page applies to you and you cite it in the memo.

---

## Part 5 — Recap

You should now be able to:

- **Choose a transport** by locality: stdio for a local subprocess (no network, OS-process boundary), streamable HTTP for a networked service (auth required, *you* are the boundary), SSE only if you inherited it (deprecated).
- **Run one server over both transports** by flipping the runner (`mcp.run()` vs `mcp.run(transport="streamable-http")`) — the tool code is identical, proving the transport is a deployment decision, not a protocol one.
- **Consume a server three ways**: a Claude Desktop / Cursor JSON config (zero client code), a programmatic `ClientSession` (initialize → list → call → read), and a LangGraph agent via `langchain-mcp-adapters` (MCP tools become LangGraph tools, multiple servers merged into one surface).
- **Run a security review** of a tool surface: validate every argument (shape, not just type), defend filesystem tools against path traversal (`resolve()` then `is_relative_to`), rate-limit expensive tools, scope capabilities to least privilege, and gate consent in the host — the exact surface week 17 attacks.
- **Situate MCP in the OpenClaw open ecosystem**: gateways that aggregate servers, self-hosted servers that keep data in your perimeter, and open Claude-compatible runtimes — all proof that MCP is an open standard no single vendor owns.

Next: the exercises put this on real servers — trace a live MCP session at the wire level, build a sandboxed filesystem server and try to break it, and drive a server from a programmatic client. Continue to [the exercises](../exercises/README.md).

---

## References

- *MCP — Transports (stdio / SSE / streamable HTTP)*: <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>
- *MCP — Security best practices*: <https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices>
- *MCP Python SDK (server + client + transports)*: <https://github.com/modelcontextprotocol/python-sdk>
- *`langchain-mcp-adapters` (MCP tools → LangGraph)*: <https://github.com/langchain-ai/langchain-mcp-adapters>
- *Claude Desktop MCP quickstart (config-file consumption)*: <https://modelcontextprotocol.io/quickstart/user>
- *Anthropic MCP connector + `anthropic[mcp]`*: <https://platform.claude.com/docs/en/agents-and-tools/mcp>
- *CWE-22: Path Traversal*: <https://cwe.mitre.org/data/definitions/22.html>
- *OWASP LLM Top 10 (the threat catalog week 17 expands on)*: <https://genai.owasp.org/llm-top-10/>
- *Awesome MCP Servers (the open self-hosted catalog)*: <https://github.com/modelcontextprotocol/servers>
