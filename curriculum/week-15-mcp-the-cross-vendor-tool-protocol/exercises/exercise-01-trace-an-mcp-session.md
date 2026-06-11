# Exercise 1 — Trace an MCP Session

**Goal:** Run a real MCP server under the **MCP Inspector**, drive it by hand, and read the JSON-RPC handshake — `initialize` → `notifications/initialized` → `tools/list` → `tools/call` — at the wire level. You will train the single most important MCP debugging habit: **the SDK hides the protocol, but you debug at the protocol.** When an integration breaks, you read the messages.

**Estimated time:** 45 minutes. Guided.

---

## Setup

Install the SDK with the CLI extras (the inspector ships with it):

```bash
pip install "mcp[cli]"
# The inspector UI is launched by `mcp dev`; it needs Node available
# (npx is invoked under the hood). Install Node 18+ if you don't have it.
```

Write a tiny server to inspect. Save this as `demo_server.py`:

```python
# demo_server.py — a 3-primitive server to trace.
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("crunch-trace-demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers. Use when the user wants an arithmetic sum."""
    return a + b


@mcp.tool()
def echo(text: str) -> str:
    """Return the input text unchanged (a trivial tool for tracing)."""
    return text


@mcp.resource("note://welcome")
def welcome() -> str:
    """A static resource the host can read into context."""
    return "Welcome to MCP. This text came from a resource read."


if __name__ == "__main__":
    mcp.run()
```

---

## Step 1 — Launch the inspector

```bash
mcp dev demo_server.py
```

This spawns your server over **stdio** and opens the MCP Inspector in your browser. The inspector *is* an MCP client — it performs the same handshake any client does, and (this is the point) it shows you the raw JSON-RPC in a history/log pane.

Find the message log. Depending on the inspector version it's a "History", "Notifications", or "Messages" pane. You're looking for the sequence of JSON-RPC requests and responses.

---

## Step 2 — Read the handshake

In the log, locate the **`initialize`** request the inspector sent on connect. Confirm you can see:

- The client's requested `protocolVersion` (a date string like `2025-06-18`).
- The client's advertised `capabilities`.
- The server's `initialize` **response**, carrying the agreed `protocolVersion`, the server's `capabilities` (it should advertise `tools` and `resources`), and `serverInfo` with your server name `crunch-trace-demo`.

Then find the **`notifications/initialized`** message the client sends right after. Notice it has **no `id`** — it's a notification, not a request, so there's no response. This is the moment the session goes live.

> **Checkpoint:** Write down, in your own words, what the `initialize` handshake *negotiated*. The answer is "protocol version + which capabilities each side supports" — that's capability negotiation, and it's why a client checks the server's advertised capabilities before assuming a feature exists.

---

## Step 3 — Discover the tools

In the inspector's **Tools** tab, click "List Tools" (or it may auto-list). In the message log, find the **`tools/list`** request and its response. The response `result` should contain `add` and `echo`, each with:

- a `name`
- a `description` (your docstring — confirm `add`'s description is the docstring text)
- an `inputSchema` (a JSON Schema; confirm `add`'s schema requires two integers `a` and `b`)

> **Checkpoint:** The `inputSchema` for `add` was generated from your *type hints* (`a: int, b: int`). You never wrote a schema by hand. Confirm this by reading the schema — it should say `"type": "integer"` for both. This is the `FastMCP` ergonomic: hints → schema.

---

## Step 4 — Call a tool and read the result

In the **Tools** tab, select `add`, enter `a=2`, `b=3`, and run it. In the log, find the **`tools/call`** request:

```json
{"method": "tools/call", "params": {"name": "add", "arguments": {"a": 2, "b": 3}}}
```

and its response — a `result` with `content: [{"type": "text", "text": "5"}]` and `isError: false`.

Now call `echo` with `text="hello"` and confirm the round trip. You've now seen the full operational loop: discover (`tools/list`), then invoke (`tools/call`).

---

## Step 5 — Read a resource

In the **Resources** tab, find `note://welcome` and read it. In the log, find the **`resources/read`** request with `params: {"uri": "note://welcome"}` and its response carrying the welcome text. Notice the *difference* from a tool call: a resource is read by URI, and it returns *context* (text to put in the prompt), not the result of an *action*. This is the model-controlled-tool vs app-controlled-resource distinction from Lecture 1, visible on the wire.

---

## Step 6 — Provoke an error and read it

Call `add` with a deliberately wrong argument — put a string where an integer goes (`a="two"`). Watch what comes back. The schema layer rejects it; find the error in the log. Note that this is a *validation* failure at the protocol/schema layer, distinct from a *tool* error (`isError: true` in a result). Being able to tell these apart in the log is the debugging skill.

---

## Step 7 — Write down what you found

Create `notes/week-15/trace.md` with a short table mapping each protocol message you observed to what it did:

| Message | id? | What it did |
|---|---|---|
| `initialize` (request) | yes | Client proposed protocol version + capabilities |
| `initialize` (response) | yes | Server agreed version + advertised tools/resources |
| `notifications/initialized` | **no** | Client signalled handshake complete; session live |
| `tools/list` | yes | Returned the catalog (add, echo) with schemas |
| `tools/call` | yes | Invoked `add(2,3)` → result `5`, `isError: false` |
| `resources/read` | yes | Read `note://welcome` → welcome text |

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] The inspector connected and you located the `initialize` request **and** response in the log.
- [ ] You identified `notifications/initialized` and confirmed it has **no `id`** (it's a notification).
- [ ] You found the `tools/list` response and confirmed `add`'s `inputSchema` was generated from your type hints (two integers).
- [ ] You called `add(2,3)`, found the `tools/call` request and its `result` (`5`, `isError: false`).
- [ ] You read `note://welcome` via `resources/read` and noted how a resource read differs from a tool call.
- [ ] `notes/week-15/trace.md` has the message-to-meaning table filled from your own observation.
- [ ] You can state, in one sentence, *why* `initialize` must come before `tools/call` (capability negotiation establishes the session before any operation is legal).

---

## Stretch

- Add a `@mcp.prompt()` to `demo_server.py`, restart the inspector, and find `prompts/list` + `prompts/get` in the log. Confirm the prompt arrives as a *template* the user would invoke, not a tool the model calls.
- Run the same server over streamable HTTP (`mcp.run(transport="streamable-http")`) and point the inspector at the URL instead of spawning a subprocess. Confirm the *protocol messages are identical* — only the transport changed. That's the transport abstraction, observed.
- Find where the SDK logs to stderr. Confirm that protocol traffic is on stdout and your `print()` debugging would corrupt the stream — which is exactly why stdio servers must log to stderr, never stdout.

---

When this feels comfortable, move to [Exercise 2 — Build a filesystem server](exercise-02-build-a-filesystem-server.py).
