# Week 15 — Exercises

Three focused drills that take you from "what's on the wire" to "I wrote a server and drove it from a client." Each takes 30–60 minutes. Do them in order — exercise 3 consumes the kind of server you build in exercise 2, which assumes the wire-level intuition from exercise 1.

## Index

1. **[Exercise 1 — Trace an MCP session](exercise-01-trace-an-mcp-session.md)** — run the MCP Inspector against a server, read the `initialize` → `tools/list` → `tools/call` handshake at the JSON-RPC level, and confirm you can identify each message. (~45 min, guided)
2. **[Exercise 2 — Build a filesystem server](exercise-02-build-a-filesystem-server.py)** — implement a sandboxed filesystem MCP server with `FastMCP`, then defend it against path traversal and prove the defense holds. (~50 min, runnable)
3. **[Exercise 3 — Consume from a client](exercise-03-consume-from-a-client.py)** — drive an MCP server from a programmatic `ClientSession`: initialize, list tools, call a tool, read a resource, and assert the results. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install the deps: `pip install "mcp[cli]" httpx`. The inspector for Exercise 1 also needs Node: `npx @modelcontextprotocol/inspector` (or just `mcp dev server.py`, which bundles it).
- **Read the wire before you trust the abstraction.** Exercise 1's whole point is that the SDK hides JSON-RPC, but you debug at the JSON-RPC layer. Watch the messages.
- **Initialize first, always.** Every client lifecycle is `initialize` → discover → call. A `tools/call` before `initialize` is a protocol error — Exercise 3's header reminds you.
- **Treat every tool argument as hostile.** Exercise 2 is about a path-traversal defense; the lesson is that the model (or an injection) supplies arguments, so you validate them. This is the week-17 rehearsal.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone. Exercise 2 is a server you run *and* inspect; Exercise 3 is a client that spawns a tiny server subprocess (defined inline) so it has no external dependencies beyond `mcp`.

```bash
# Exercise 1 — inspect a server in a browser UI:
mcp dev exercise-02-build-a-filesystem-server.py

# Exercise 2 — run the server's own self-test (path-traversal probes):
python3 exercise-02-build-a-filesystem-server.py --self-test

# Exercise 3 — run the client against an inline server subprocess:
python3 exercise-03-consume-from-a-client.py
```

## A note on async

MCP is async to the core — `ClientSession`, `call_tool`, `stdio_client` are all coroutines. Exercise 3 runs inside `asyncio.run(main())`. If you've never written async Python, the pattern is mechanical: `async def`, `await` the coroutine, wrap the top-level call in `asyncio.run`. The exercises stay inside that pattern; you don't need deep asyncio to complete them.

## A note on determinism

The protocol is deterministic: the same `tools/call` with the same arguments against the same server gives the same result. The path-traversal probes in Exercise 2 either escape the sandbox or they don't — there's no flakiness to chase. If a probe that *should* be blocked gets through, your containment check is wrong, and that's worth finding before week 17's red team finds it for you.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-15` to compare.
