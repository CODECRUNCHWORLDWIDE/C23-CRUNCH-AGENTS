# Lecture 1 — The Protocol and the Primitives: Host, Client, Server, and the Three Things a Server Exposes

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can explain the MCP architecture (host / client / server) and the strict 1:1 client-server pairing, name the three server primitives (tools, resources, prompts) and who controls each, write a minimal `FastMCP` server, and read the JSON-RPC handshake — `initialize`, `tools/list`, `tools/call` — at the wire level.

If you remember one sentence from this entire week, remember this one:

> **MCP is the USB-C of agent tooling. It is not the future — it is the present, and it is open.** Write a tool once, behind the protocol, and any MCP-aware client can use it: Claude Desktop, Cursor, a LangGraph agent, a self-hosted runtime.

There's a corollary you should tape next to it:

> **A tool exposed over MCP is an API exposed to an untrusted client.** The model is the client, and the model can be steered by a prompt injection. Validate every argument like it came from the public internet.

Back in week 4, a tool was a JSON blob inside a `messages.create` call and a Python function in the same file. In week 13 you wired tools into a LangGraph node by hand. Both approaches glue the tool to the caller. This week we cut that glue: the tool moves into its own process, behind a stable JSON-RPC interface, and the caller reaches it over a transport. The whole lecture is in service of one idea: **the tool surface becomes portable.**

---

## 1. Why a protocol at all — the N×M problem

Imagine you have built five tools — a filesystem reader, a database query tool, a web-fetcher, a calculator, and a private-corpus search. And imagine your team uses four different agent stacks: a LangGraph service, a Mastra TypeScript service, Claude Desktop for ad-hoc work, and Cursor for coding. Without a protocol, integrating every tool into every stack is **N×M**: 5 tools × 4 stacks = 20 bespoke integrations. The filesystem tool you wrote for LangGraph doesn't work in Mastra without a TypeScript rewrite; the corpus search you exposed to Claude Desktop doesn't work in Cursor without re-implementing its config glue. Every new tool multiplies the work by the number of stacks; every new stack multiplies it by the number of tools.

This is exactly the integration explosion that USB solved for peripherals and that LSP (the Language Server Protocol) solved for editors. Before LSP, every editor needed a bespoke plugin for every language — N editors × M languages. LSP collapsed that to N+M: a language ships *one* language server, an editor ships *one* LSP client, and any editor talks to any language. MCP is LSP for agent tools.

**MCP collapses N×M to N+M.** A tool speaks MCP once (you write a *server*). A stack speaks MCP once (it ships a *client*). Now any client talks to any server. Your five tools become five servers; your four stacks each ship one MCP client; the integrations drop from 20 to "write five servers and let the clients find them." Add a sixth tool and it's *one* new server that every stack can use immediately. Add a fifth stack and it's *one* new client that reaches every existing server. The whole point — and the sentence to internalize — is **write the tool once, plug it in everywhere.**

That is the entire value proposition. Everything else in this lecture — the architecture, the primitives, the transports — is the machinery that makes "write once, plug in everywhere" actually work in practice.

---

## 2. The architecture — host, client, server

MCP has exactly three roles. Get these straight and the rest of the protocol falls into place.

**The host** is the LLM application — the thing that *wants* to use tools. Claude Desktop is a host. Cursor is a host. Your LangGraph agent is a host. The host owns the conversation with the model, decides when a tool should be called, and is responsible for getting the user's consent before a tool runs.

**The client** is the connector that lives *inside* the host. Critically, **there is one client per server connection, and the pairing is 1:1.** If your host connects to three MCP servers, it instantiates three clients. Each client manages one stateful session with one server: it performs the handshake, lists what the server offers, and relays tool calls. The client is a thin protocol object, not application logic — its job is to speak MCP correctly to exactly one server.

**The server** is the process that *exposes* capabilities — tools, resources, prompts. This is what you write this week. A server knows nothing about the model, the conversation, or the host's UI. It receives a `tools/call` request, runs a function, and returns a result. It is, in the most literal sense, a small JSON-RPC service whose API happens to be designed for an LLM to consume.

```
┌──────────────────────────────── HOST (the LLM app) ───────────────────────────┐
│                                                                                │
│   conversation + model + consent UI                                            │
│                                                                                │
│   ┌──────────┐         ┌──────────┐         ┌──────────┐                       │
│   │ Client A │         │ Client B │         │ Client C │   (1 client per server)│
│   └────┬─────┘         └────┬─────┘         └────┬─────┘                       │
└────────┼────────────────────┼────────────────────┼───────────────────────────┘
         │ MCP                 │ MCP                 │ MCP
         v                     v                     v
   ┌───────────┐         ┌───────────┐         ┌───────────┐
   │ Server A  │         │ Server B  │         │ Server C  │
   │ filesystem│         │  corpus   │         │   git     │
   └───────────┘         └───────────┘         └───────────┘
```

The separation is the source of the portability. Because the server knows nothing about the host, the *same server* can be paired with a Claude Desktop client, a Cursor client, and a LangGraph client — simultaneously or in turn — with zero changes. That is the "any client, any server" promise made structural.

One more distinction that trips people up: the **client** and the **host** are not the same thing, even though the client lives inside the host. The host is the whole application; the client is the per-connection protocol object. When the spec says "the client sends `tools/call`," it means the per-server connector, not the chat app as a whole.

---

## 3. The three primitives — and who controls each

A server can expose three kinds of capability. The single most important thing to understand is **who controls each one**, because that control model is the security boundary.

### 3.1 Tools — model-controlled

A **tool** is a callable function with a JSON-schema input. It is **model-controlled**: the model decides, during the agent loop, to call `search_corpus` with `{"query": "..."}`. Tools are the primitive that *does things* — reads a file, runs a query, sends a request. Because the model controls invocation and the model can be steered by a prompt injection, **tools are the primary attack surface.** Every tool is, functionally, a remote-code-execution endpoint that an adversary may be able to trigger by poisoning the model's context. We come back to this hard in Lecture 2's security section and again, with red teaming, in week 17.

A tool has a `name`, a `description` (the model reads this to decide *when* to call it — write it prescriptively), and an `inputSchema` (JSON Schema the arguments must satisfy). A `tools/call` returns content (text, or structured data) and an `isError` flag.

### 3.2 Resources — application-controlled

A **resource** is readable context addressed by a URI — a file (`file:///sandbox/contract.txt`), a database row, a query result. It is **application-controlled**: the *host application*, not the model, decides which resources to pull into context. The model doesn't call a resource the way it calls a tool; instead the host reads resources and may inject them into the prompt. Resources are the "give the model *context*" primitive, as opposed to tools' "let the model *act*" primitive.

This control distinction matters for safety: because resources are application-controlled, a malicious model can't *invoke* a resource read on its own the way it can invoke a tool. The host mediates. (It can still be *tricked* into reading a poisoned resource — that's the indirect-injection vector you'll attack in week 17 — but the invocation pathway is the host's, not the model's.)

### 3.3 Prompts — user-controlled

A **prompt** is a reusable, parameterized message template. It is **user-controlled**: a prompt typically surfaces in the client as a slash-command the *user* picks (`/summarize-contract`). The server defines the template; the user invokes it; the host fills in arguments and seeds the conversation. Prompts are how a server ships "blessed" interaction patterns — a well-crafted summarization prompt, a structured-extraction prompt — without the user having to remember them.

The control triad — **tools = model, resources = app, prompts = user** — is the cleanest way to remember the primitives, and it's a quiz favorite. It's also a design discipline: when you add a capability to a server, ask "who should be allowed to trigger this?" An action with side effects is a tool (and gets validated like RCE); a piece of readable context is a resource; a blessed interaction is a prompt.

---

## 4. Your first server — `FastMCP`

The official Python SDK ships `FastMCP`, a high-level server where Python type hints become JSON schemas and decorators register primitives. Here is a complete, runnable server exposing one tool, one resource, and one prompt:

```python
# server.py — a minimal but complete MCP server.
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("crunch-demo")          # the server's name, seen by clients


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum.

    The docstring becomes the tool DESCRIPTION the model reads to decide
    when to call this. The type hints (a: int, b: int) become the JSON
    Schema for the arguments. Return type documents the result.
    """
    return a + b


@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """A resource template: a URI like greeting://ada resolves to this text.

    The host reads this; the model does not invoke it directly.
    """
    return f"Hello, {name}, from an MCP resource."


@mcp.prompt()
def review_contract(clause: str) -> str:
    """A user-invoked prompt template (a slash-command in the client)."""
    return (
        "You are a contracts analyst. Read the clause below and state its "
        f"obligation in one sentence.\n\nClause:\n{clause}"
    )


if __name__ == "__main__":
    mcp.run()           # defaults to the stdio transport (Lecture 2)
```

Three things to notice, because they're the whole ergonomics of `FastMCP`:

1. **Type hints are the schema.** `add(a: int, b: int)` generates an `inputSchema` requiring two integers. There is no separate schema to hand-maintain — the signature *is* the contract. (This is why typed signatures matter: a sloppy `def add(a, b)` with no hints gives the model an untyped schema and worse tool-calling accuracy.)
2. **The docstring is the description.** The model reads it to decide when to call the tool, so write it the way you'd write an API doc: say *what it does* and *when to use it*, not just *what it is*. "Add two integers" is fine; "Use this when the user asks for an arithmetic sum" is better for triggering.
3. **The decorator picks the primitive.** `@mcp.tool()` → model-controlled action. `@mcp.resource(uri)` → app-controlled context. `@mcp.prompt()` → user-controlled template. Same function-writing ergonomics, three different control models.

Run it under the inspector to see what a client sees:

```bash
pip install "mcp[cli]"
mcp dev server.py        # opens the MCP Inspector against your server
```

The inspector connects, runs `initialize`, lists your one tool / one resource / one prompt, and lets you call `add` and watch the JSON-RPC. Exercise 1 lives in exactly this loop — and reading that wire traffic is how you build a real mental model of the protocol, not a hand-wavy one.

---

## 5. The JSON-RPC layer — what's actually on the wire

MCP rides on **JSON-RPC 2.0**. You do not have to love JSON-RPC, but you do have to be able to read it, because debugging an MCP integration means reading the messages. The format is small:

- A **request** is `{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {...}}`. The `id` correlates the response.
- A **response** is `{"jsonrpc": "2.0", "id": 1, "result": {...}}` — or, on failure, `{"jsonrpc": "2.0", "id": 1, "error": {"code": ..., "message": ...}}`.
- A **notification** is a request with *no* `id` — fire-and-forget, no response expected (`notifications/initialized`, `notifications/resources/updated`).

A full session, in order, looks like this:

```
--> initialize          {protocolVersion, capabilities, clientInfo}
<-- initialize result    {protocolVersion, capabilities, serverInfo}
--> notifications/initialized        (no id — the handshake is done)
--> tools/list           {}
<-- tools/list result    {tools: [{name, description, inputSchema}, ...]}
--> tools/call           {name: "add", arguments: {a: 2, b: 3}}
<-- tools/call result    {content: [{type: "text", text: "5"}], isError: false}
```

### 5.1 `initialize` — the handshake that negotiates capabilities

Nothing happens before `initialize`. The client sends its desired `protocolVersion` and the `capabilities` it supports; the server replies with the version it agrees to and the `capabilities` it offers (does it have tools? resources? prompts? resource subscriptions?). This is **capability negotiation** — neither side assumes a feature exists until the handshake confirms it. A client that wants to subscribe to resource changes checks the server's advertised `resources.subscribe` capability first; a server that doesn't offer prompts simply doesn't list the `prompts` capability, and a well-behaved client won't call `prompts/list`.

The handshake completes with the client sending the `notifications/initialized` notification — *now* the session is live and `tools/list` / `tools/call` are legal. Sending a `tools/call` before the handshake finishes is a protocol error. (This ordering is a quiz favorite: **`initialize` first, always.**)

### 5.2 `tools/list` and `tools/call` — discovery, then invocation

`tools/list` returns the catalog: each tool's `name`, `description`, and `inputSchema`. The host hands this catalog to the model so the model knows what it can call. `tools/call` invokes one: `{"name": "search_corpus", "arguments": {"query": "..."}}`. The result carries `content` (a list of typed blocks — usually text) and an `isError` boolean.

The `isError` flag is worth dwelling on because it's a correctness trap. There are **two** kinds of failure: a *protocol* error (the method doesn't exist, the params are malformed) comes back as a JSON-RPC `error` object; a *tool* error (the file wasn't found, the query failed) comes back as a normal `result` with `isError: true` and an explanatory message in `content`. The distinction matters: a tool error is *information for the model* — "that file doesn't exist, try another path" — so it must reach the model as content, not blow up the JSON-RPC layer. Return tool failures as `isError: true` results; reserve JSON-RPC errors for genuine protocol violations.

```python
# A tool that returns a tool-error result (NOT a protocol error) on bad input.
@mcp.tool()
def read_note(note_id: str) -> str:
    """Read a stored note by id."""
    path = NOTES_DIR / f"{note_id}.txt"
    if not path.exists():
        # This reaches the model as `isError: true` content, so the model can
        # recover ("try a different id"). Do NOT raise a bare exception here —
        # that becomes an opaque protocol error the model can't reason about.
        raise FileNotFoundError(f"No note with id {note_id!r}")
    return path.read_text()
```

`FastMCP` converts the raised `FileNotFoundError` into an `isError: true` result automatically — that's the idiomatic way to signal a recoverable tool failure. The model sees the message and can try again; the protocol layer stays clean.

---

## 6. What the model actually sees — tools become a tool catalog

Step back to the agent loop you built in weeks 5 and 13. When the host connects to an MCP server, it runs `tools/list`, gets the catalog, and translates each MCP tool into whatever tool-format its model expects — Anthropic `tool_use` blocks, OpenAI function specs, a LangGraph `BaseTool`. From the model's point of view, **an MCP tool is just a tool.** The model doesn't know or care that `search_corpus` lives in a separate process behind JSON-RPC; it sees a name, a description, and a schema, exactly like the inline tools from week 4.

This is the second half of the portability story. Not only is the *server* portable across clients (§2) — the *model* is portable across servers. The same Claude model (`claude-opus-4-8`) that calls your filesystem server's tools will, with no code change, call your corpus server's tools, because both arrive at the model as the same tool-catalog shape. The protocol is the universal joint between "tools I wrote" and "models that call them."

Concretely, with Anthropic's SDK there are two consumption paths (you'll use both this week):

- **Direct connection:** pass `mcp_servers=[{...}]` on the API request and Claude connects to the (remote) server itself. The model calls the tools; you never touch the JSON-RPC.
- **SDK conversion helpers:** `from anthropic.lib.tools.mcp import mcp_tool`, list the server's tools via a `ClientSession`, convert each with `mcp_tool(t, session)`, and hand them to `client.beta.messages.tool_runner(...)`. This path keeps the MCP connection on *your* side — the right choice for local stdio servers and for fine-grained control. (Requires `pip install "anthropic[mcp]"`.)

Both paths end in the same place: the model sees a tool catalog and calls tools. The difference is *who holds the MCP connection* — Anthropic's infrastructure (direct) or your process (conversion helpers). For local servers you wrote, the conversion-helper path is the one to reach for.

---

## 7. The lifecycle, end to end

Putting the pieces together, the life of an MCP session is:

1. **Launch.** The host starts (or connects to) the server — spawns a subprocess for stdio, or opens an HTTP connection for streamable HTTP (Lecture 2).
2. **Initialize.** Client sends `initialize`; server replies with agreed version + capabilities; client sends `notifications/initialized`. The session is now live.
3. **Discover.** Client runs `tools/list` (and `resources/list`, `prompts/list` if it cares). The host hands the tool catalog to the model.
4. **Operate.** The model, mid-conversation, decides to call a tool. The host requests consent (if configured), then the client sends `tools/call`. The server runs the function and returns a result. This repeats for the life of the conversation.
5. **Shutdown.** The client closes the session; for stdio, the subprocess is terminated.

Two properties of this lifecycle are load-bearing. First, **the session is stateful** — `initialize` establishes a connection that persists across many tool calls, which is why the client is a long-lived object, not a per-call function. Second, **consent lives in the host** — the protocol assumes a human (or a policy) may need to approve a side-effecting tool call before it runs. A well-built host *gates* `tools/call`; a careless one auto-approves everything, which is exactly the foothold week 17's red team will look for.

---

## 8. Resources and prompts in depth — the other two primitives

§3 named the three primitives; tools got the lion's share of attention because they're where the action (and the risk) is. But resources and prompts are not afterthoughts — they're how a server provides *context* and *blessed interactions*, and a server that exposes all three is a richer, safer surface than a tools-only one. Let's make them concrete.

### 8.1 Resources — addressable context, read by the host

A resource is identified by a **URI**, and the server can expose either *fixed* resources (`note://welcome`) or *templated* ones (`corpus://{clause_id}`, where the `{clause_id}` is a parameter the client fills in). The client discovers resources with `resources/list` and reads one with `resources/read`:

```python
@mcp.resource("corpus://{clause_id}")
def clause_resource(clause_id: str) -> str:
    """A templated resource: corpus://clause_09 resolves to clause 9's text.

    The HOST decides to read this and inject it into the model's context. The
    model does not 'call' it the way it calls a tool — which is exactly the
    control distinction: resources are application-controlled context, not
    model-controlled actions.
    """
    return CORPUS.get(clause_id, f"<no such clause: {clause_id}>")
```

Why a resource and not a tool? The same data — clause 9's text — *could* be served by a `get_clause` tool. The difference is the control model and the intent. A tool is for the model to *decide* to invoke when it needs to act; a resource is for the host to *pull in* as context, often deterministically (e.g. "always load the user's profile resource at the start of the session"). When you're designing a server, the question "should the *model* choose to fetch this, or should the *host* always provide it?" decides between a tool and a resource. Context the host should mediate → resource. An action the model should choose → tool. (Many servers expose *both* a `get_X` tool and an `X://` resource for the same data, so the host can pull it as context *and* the model can fetch it on demand. That's fine — they serve different control flows.)

Resources also support **subscriptions**: a client can subscribe to a resource (if the server advertises the `subscribe` capability in `initialize`), and the server pushes a `notifications/resources/updated` notification when it changes. This is how a client stays in sync with a log file, a live document, or a changing dataset without polling. The stretch goal in the README builds this.

### 8.2 Prompts — blessed templates the user invokes

A prompt is a reusable, parameterized message template that the *user* invokes — typically surfaced in the client as a slash-command. The server defines it; the client lists it with `prompts/list` and fetches a filled-in version with `prompts/get`:

```python
@mcp.prompt()
def summarize_clause(clause_id: str) -> str:
    """A user-invoked template. In Claude Desktop this becomes a slash-command
    like /summarize-clause; the user picks it and supplies clause_id, and the
    host seeds the conversation with the returned text."""
    return (
        "You are a contracts analyst. Read the clause below and state its single "
        f"obligation in one sentence, plainly.\n\nClause {clause_id}:\n{CORPUS[clause_id]}"
    )
```

Why prompts exist: they let a server ship *good interaction patterns* so users don't have to remember (or re-type) them. A well-crafted summarization prompt, a structured-extraction prompt, a "explain this like I'm five" prompt — the server author wrote them well once, and every user of the server gets them as a one-click slash-command. Prompts are the "here's the right way to use this server" primitive. They're user-controlled (the user picks them), which is why they can't be triggered by a malicious model or an injection — a useful safety property that follows directly from the control model.

The triad, one more time, now with the *design question* each answers:
- **Tool** — "should the model be able to *do* this?" (model-controlled action; the attack surface).
- **Resource** — "should the host be able to *provide* this as context?" (app-controlled context).
- **Prompt** — "should the user have a *blessed way to invoke* this?" (user-controlled template).

---

## 9. MCP versus week 4's inline tools — what actually changed

It's worth being precise about what MCP buys you over the inline tool-calling you did in week 4, because the *capability* is nearly identical — the model calls a named function with JSON arguments — and yet the *engineering* is transformed.

In week 4, a tool was: a JSON schema in the `tools` array of a `messages.create` call, plus a Python function in the same file, plus the glue that dispatched the model's tool-call to that function. Everything lived in one process, married to one SDK. To use that tool from a *different* agent, you copied the schema and the function and rewrote the glue. To use it from Claude Desktop, you couldn't — Desktop doesn't run your Python.

MCP changes three things, and only three:

1. **The tool moves into its own process.** It's no longer a function in your agent file; it's a server you launch. This is what makes it shareable — a separate process can be reached by *any* client, not just the one that imported the function.
2. **The interface is standardized.** `tools/list` and `tools/call` are the same for every server and every client. The schema shape, the result shape, the error contract — all fixed by the protocol. This is what makes a server *portable*: a client that speaks MCP can talk to a server it's never seen.
3. **The transport is pluggable.** The same server runs locally (stdio) or networked (streamable HTTP), so "where does the tool run" becomes a deployment decision, not a code rewrite (Lecture 2).

What *didn't* change: the model's experience. The model still sees a name, a description, and a schema, and still emits a tool-call. From the model's seat, an MCP tool is indistinguishable from a week-4 inline tool. That's deliberate — MCP standardizes the *plumbing between the tool and the caller*, not the *interface between the caller and the model*. You already knew how to define a good tool (clear name, prescriptive description, tight schema — week 4); MCP just lets that tool live somewhere portable. The lesson learned in week 4 about *writing* tools carries over unchanged; what's new this week is *where they live and how they're reached.*

One subtlety worth flagging now, because it bites people who treat MCP as "free portability": **a tool description that was tuned for one model isn't automatically optimal for every model.** The description is what the model reads to decide *when* to call the tool, and different models weight descriptions differently (Lecture from the model-migration notes: recent models reach for tools more conservatively and reward prescriptive "call this when..." descriptions). MCP makes the *mechanism* portable, but the *triggering quality* still depends on the description meeting the consuming model's expectations. So when you wire your server into a new client with a new model and tool-calling accuracy dips, suspect the description before the protocol — the plumbing is portable, the prompt-engineering of the description is still yours to tune per the model behind the client.

---

## 9.5 Common questions, answered

A few questions that come up every cohort, because the answers sharpen the mental model:

**"Why JSON-RPC and not REST?"** MCP is *stateful and bidirectional* — the session persists across many calls, and the server can send notifications *to* the client (resource updates, progress). REST is stateless request/response; it doesn't naturally model a long-lived session with server-initiated messages. JSON-RPC, riding over a persistent transport (stdio pipe, or streamable HTTP's optional SSE upgrade), fits the bidirectional-session shape MCP needs. You don't have to love JSON-RPC; you have to be able to read it (§5).

**"Is the client the same as the agent?"** No — and this trips people up. The *host* is the agent (the whole application). The *client* is a thin per-server connector *inside* the host. Your LangGraph agent is a host; when it connects to two MCP servers, it has two clients. When the spec says "the client sends `tools/call`," it means that per-connection object, not your agent as a whole (§2).

**"Can two agents share one MCP server?"** Over **stdio**, no — stdio is inherently 1:1 and local; each host spawns its own subprocess. Over **streamable HTTP**, yes — a networked server can serve many clients, which is exactly why you'd choose HTTP for a shared corpus server (Lecture 2). The transport decides the sharing model.

**"If MCP standardizes tools, why does my tool-calling accuracy differ across clients?"** Because the protocol standardizes the *mechanism*, not the *description quality* relative to a given model (§9). The description is prompt-engineering, and different models weight it differently. Portable plumbing, model-specific tuning of the description.

**"Do I need an Anthropic API key to write a server?"** No. A server is vendor-agnostic — it knows nothing about any model. You need a key only for the *consumption path* that happens to use Claude, and even that has an open-only fallback. The whole week is completable with zero vendor credentials, which is the point of an *open* protocol.

---

## 10. Recap

You should now be able to:

- Explain **why a protocol** — the N×M integration explosion that MCP collapses to N+M, and the "write the tool once, plug it in everywhere" payoff.
- Name the **three roles** — host (the LLM app), client (the per-server connector, 1:1 with servers), server (the tool/resource/prompt provider you write) — and explain why the host↔server separation makes servers portable across clients.
- State the **three primitives and their control models** — tools (model-controlled, the action + attack surface), resources (app-controlled, readable context), prompts (user-controlled, slash-command templates).
- Write a minimal **`FastMCP` server** where type hints are the schema, docstrings are the description, and decorators pick the primitive.
- Read the **JSON-RPC handshake** — `initialize` (capability negotiation) → `notifications/initialized` → `tools/list` → `tools/call` — and distinguish a protocol error (JSON-RPC `error`) from a tool error (`result` with `isError: true`).
- Explain how an **MCP tool reaches the model** as an ordinary tool catalog, and the two Anthropic consumption paths (direct `mcp_servers` vs SDK conversion helpers).

Next: the transports that carry these messages (stdio vs SSE vs streamable HTTP), how to consume a server from real clients (Claude Desktop, a programmatic `ClientSession`, a LangGraph agent), the security review that treats every tool as RCE, and the OpenClaw open-ecosystem context. Continue to [Lecture 2 — Transports, Clients, and Security](./02-transports-clients-and-security.md).

---

## References

- *Model Context Protocol specification (`2025-06-18`)* — architecture, lifecycle, primitives: <https://modelcontextprotocol.io/specification/2025-06-18>
- *MCP core concepts (architecture)*: <https://modelcontextprotocol.io/docs/concepts/architecture>
- *MCP — Tools (the `tools/list` / `tools/call` contract)*: <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>
- *MCP — Resources*: <https://modelcontextprotocol.io/specification/2025-06-18/server/resources>
- *MCP — Prompts*: <https://modelcontextprotocol.io/specification/2025-06-18/server/prompts>
- *JSON-RPC 2.0 specification*: <https://www.jsonrpc.org/specification>
- *MCP Python SDK (`FastMCP`)*: <https://github.com/modelcontextprotocol/python-sdk>
- *Anthropic MCP connector + `anthropic[mcp]` helpers*: <https://platform.claude.com/docs/en/agents-and-tools/mcp>
