# Week 15 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 16. Answer key is at the bottom — don't peek.

---

**Q1.** What problem does MCP fundamentally solve? Pick the *most complete* answer.

- A) It makes LLMs faster by caching tool results.
- B) It collapses the N×M tool-integration explosion (N frameworks × M tools, each hand-wired) to N+M: a tool speaks MCP once (a server), a client speaks MCP once, and any client talks to any server.
- C) It replaces JSON-RPC with a faster binary protocol.
- D) It forces every tool to run on Anthropic's servers.

---

**Q2.** In MCP's architecture, what is the relationship between clients and servers?

- A) One client can connect to many servers over a shared connection.
- B) There is exactly one client per server connection (1:1); a host that connects to three servers instantiates three clients, each managing one stateful session.
- C) Clients and servers are the same object.
- D) One server serves many clients over one connection.

---

**Q3.** The three MCP server primitives are tools, resources, and prompts. Who controls each?

- A) The model controls all three.
- B) Tools are **model**-controlled (the model invokes them), resources are **application**-controlled (the host reads them into context), prompts are **user**-controlled (the user picks them, often as a slash-command).
- C) The server controls all three; the client just displays them.
- D) Resources are model-controlled and tools are user-controlled.

---

**Q4.** Why is a tool the primary attack surface of an MCP server?

- A) Tools are slower than resources.
- B) The model decides when to call a tool, the model can be steered by a prompt injection, and a tool runs code with arguments the model supplied — so a tool is effectively a remote-code-execution endpoint an adversary may trigger by poisoning the model's context.
- C) Tools are the only primitive that uses JSON-RPC.
- D) Tools require an API key and resources don't.

---

**Q5.** In a `FastMCP` server, where does a tool's JSON input schema come from?

- A) You hand-write it as a separate JSON file.
- B) From the function's **type hints** — `def add(a: int, b: int)` generates a schema requiring two integers; you never write the schema by hand.
- C) The model infers it at call time.
- D) From the docstring only.

---

**Q6.** What must happen before any `tools/call` is legal in an MCP session?

- A) Nothing — you can call a tool immediately.
- B) The `initialize` handshake must complete (client and server negotiate protocol version and capabilities), followed by the client's `notifications/initialized`. Only then is the session live.
- C) The server must call the client first.
- D) You must read a resource first.

---

**Q7.** A tool fails because the requested file doesn't exist. How should that reach the model?

- A) As a JSON-RPC `error` object (a protocol error).
- B) As a normal `result` with `isError: true` and an explanatory message in `content`, so the model can *reason about it* and recover ("try a different path") — a JSON-RPC error is reserved for genuine protocol violations.
- C) As a crash that terminates the session.
- D) Silently — return an empty result.

---

**Q8.** When should you use the **stdio** transport versus **streamable HTTP**?

- A) Always use stdio; HTTP is deprecated.
- B) stdio when the server is a local subprocess (the OS process boundary is the security boundary, no auth needed); streamable HTTP when the server is a networked service (it's reachable by remote attackers, so auth/authz/rate-limiting become required and *you* are the boundary).
- C) HTTP for local servers, stdio for remote.
- D) They're interchangeable in every situation.

---

**Q9.** What is the status of the **SSE** transport in 2026?

- A) It's the current recommended remote transport.
- B) It's the **deprecated** remote transport (two-endpoint design), superseded by streamable HTTP. Recognize it in old servers/configs, but don't build new servers on it.
- C) It's the only transport that supports streaming.
- D) It's faster than streamable HTTP and preferred.

---

**Q10.** A filesystem tool takes a `relative_path`. What is the correct path-traversal defense?

- A) Reject any path string that contains `..`.
- B) `resolve()` the joined path FIRST (collapsing `..`, following symlinks, normalizing separators), THEN check `is_relative_to(SANDBOX)` — because a string check on `..` misses absolute paths (`/etc/passwd` has no `..`), symlinks, and encoded separators.
- C) Only allow paths shorter than 50 characters.
- D) Trust the model not to send a malicious path.

---

**Q11.** You change a server's runner from `mcp.run()` to `mcp.run(transport="streamable-http")`. What else must change in the tool code?

- A) Every tool must be rewritten for HTTP.
- B) **Nothing** — the tool/resource code is byte-for-byte identical; the transport is a deployment decision the server logic doesn't know about. (Security posture changes, but the code doesn't.)
- C) You must add type hints.
- D) You must remove all resources.

---

**Q12.** A LangGraph agent connects (via `MultiServerMCPClient`) to a stdio filesystem server and a streamable-HTTP corpus server at once. What does the model see?

- A) Two separate tool lists it must switch between.
- B) One merged, flat tool surface — `search_corpus`, `get_clause`, `read_file`, `write_file` — and it calls tools across both servers without knowing or caring that some came over HTTP and some over stdio.
- C) Only the stdio server's tools.
- D) An error, because you can't mix transports.

---

**Q13.** Your corpus server runs over streamable HTTP on `localhost` with no auth, and it works fine in the lab. What must the security memo say?

- A) Nothing — it works, so it's secure.
- B) That a *real* (non-localhost) deployment requires authentication, authorization, and edge rate-limiting, because over HTTP the server is reachable by remote attackers and the OS-process boundary that protected the stdio server no longer applies — the server's threat model changes the instant it goes networked.
- C) That HTTP is always insecure and the server should use stdio.
- D) That auth is the client's job, not the server's.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — N×M → N+M is the whole value proposition: write the tool once (a server), and any MCP client can use it. (Lecture 1 §1.)
2. **B** — Strict 1:1 client-server pairing; one client per server connection, each a stateful session. (Lecture 1 §2.)
3. **B** — The control triad: tools = model, resources = app, prompts = user. This is the security boundary *and* a design discipline. (Lecture 1 §3.)
4. **B** — A tool is RCE: model-invoked, injectable, runs code with model-supplied args. Validate every argument like it's hostile. (Lecture 1 §3.1, Lecture 2 §3.)
5. **B** — `FastMCP` generates the schema from type hints; the signature *is* the contract. Docstrings become the description. (Lecture 1 §4.)
6. **B** — `initialize` (capability negotiation) → `notifications/initialized`, then the session is live. A `tools/call` before the handshake is a protocol error. (Lecture 1 §5.1, §7.)
7. **B** — Tool errors are *information for the model*: return `isError: true` results so the model can recover; reserve JSON-RPC errors for protocol violations. (Lecture 1 §5.2.)
8. **B** — Locality decides: stdio = local subprocess (OS-process boundary), streamable HTTP = networked service (you are the boundary, auth required). (Lecture 2 §1.1, §1.3, §1.4.)
9. **B** — SSE is the deprecated two-endpoint remote transport; know it, don't build new on it; streamable HTTP superseded it. (Lecture 2 §1.2.)
10. **B** — Resolve first, then check containment with `is_relative_to`. A string `..` check misses absolute paths, symlinks, and encoded separators. (Lecture 2 §3.2; Exercise 2; the challenge trap.)
11. **B** — Nothing in the tool code; the transport is a deployment decision. Flip the runner, ship locally or remotely. (Lecture 2 §1.3.)
12. **B** — One merged transport-agnostic surface; the agent (client) is portable across servers, the servers are portable across transports. That's the protocol working. (Lecture 2 §2.3.)
13. **B** — Going networked changes the threat model: HTTP exposes the server to remote attackers, so authn/authz/rate-limiting become required. The OS-process boundary is gone. (Lecture 2 §1.3; the challenge's second trap.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
