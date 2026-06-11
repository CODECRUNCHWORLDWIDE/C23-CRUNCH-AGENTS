# Week 15 Homework

Six problems that revisit the week's topics and put MCP into your fingers. The full set should take about **5 hours**. Work in your Week 15 Git repository (the same workspace as the exercises and the `crunchmcp` mini-project) so every problem produces at least one commit you can point to when week 17 attacks this surface.

The headline deliverable is **Problem 4 — the one-page MCP tool-surface security memo**, the artifact that proves your tool surface is safe before the red team tries to break it.

Have your **week-13 LangGraph agent** importable (Problem 5 plugs MCP tools into it) and `mcp[cli]` installed. If week 13 is broken, fix it first — Problem 5 depends on it.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Trace a session and map the messages

**Problem statement.** Run the MCP Inspector against a 2-tool, 1-resource `FastMCP` server (Exercise 1's `demo_server.py`, or your own). Produce `notes/week-15/trace.md` containing the JSON-RPC message-to-meaning table: every protocol message you observed (`initialize` request + response, `notifications/initialized`, `tools/list`, `tools/call`, `resources/read`) and what each did. The required check: confirm a tool's `inputSchema` was generated from your type hints, not hand-written.

**Acceptance criteria.**

- `notes/week-15/trace.md` exists with a row per observed protocol message.
- The `notifications/initialized` row is correctly marked as having **no `id`** (a notification).
- You show a tool's `inputSchema` and confirm it matches the function's type hints.
- One sentence stating *why* `initialize` must precede `tools/call`.
- Committed.

**Hint.** `mcp dev demo_server.py` opens the inspector with a message log. If you can't find the log pane, it's usually labelled "History" or "Notifications". The handshake messages appear right at connect time, before you click anything.

**Estimated time.** 40 minutes.

---

## Problem 2 — Implement and unit-test the path-traversal defense

**Problem statement.** In your `crunchmcp` package, implement `safe_path(sandbox, relative_path)` in `security.py` (resolve-then-contain) and the `read_file` / `write_file` / `list_files` tools that use it. Write `tests/test_security.py` proving that (a) every traversal probe — `../SECRET.txt`, `/etc/passwd`, an absolute path, `sub/../../SECRET.txt` — is blocked, and (b) every legitimate in-sandbox read succeeds.

**Acceptance criteria.**

- `safe_path` resolves the joined path **before** checking `is_relative_to(SANDBOX)`.
- `pytest tests/test_security.py` passes with at least: all four traversal probes blocked, two legit reads succeed.
- A planted out-of-sandbox secret is provably never returned.
- Committed.

**Hint.** Port your Exercise 2 `_safe_path`. For the test, plant a `SECRET.txt` as a *sibling* of the sandbox (not inside it), then assert `read_file("../SECRET.txt")` raises and never returns the secret text. Include an **absolute-path** probe (`/etc/passwd`) — that's the one a naive `..`-string check misses.

**Estimated time.** 50 minutes.

---

## Problem 3 — Run one server over both transports

**Problem statement.** Take your corpus server and run it over **stdio** and over **streamable HTTP** from the *same* server object, via your `transport.serve(server, transport)`. Produce `notes/week-15/transports.md` showing: the two commands, a successful `tools/call` over each, and a one-paragraph statement of what changed between them (the runner) and what didn't (the tool code).

**Acceptance criteria.**

- The corpus server serves over both `stdio` and `streamable-http` with no change to the server's tool code.
- A successful `search_corpus` call is shown over each transport (returning clause_09 for the confidentiality query).
- A paragraph naming exactly what differed (transport runner) and what stayed identical (tools/resources).
- Committed.

**Hint.** For the HTTP path, run the server (`serve corpus --transport streamable-http`) in one terminal and hit it with a `streamablehttp_client` in another. For stdio, the client spawns the server. The `tools/call` result should be byte-identical across transports — that's the proof.

**Estimated time.** 50 minutes.

---

## Problem 4 — The one-page MCP tool-surface security memo (headline deliverable)

**Problem statement.** This is the artifact week 17 will test. For your two-server surface (filesystem + corpus), write a **one-page** memo at `notes/week-15/security-memo.md` against this template:

1. **The surface** — one sentence per server naming the tools and their control model.
2. **The threats** — per server: path traversal (fs), argument injection (corpus `get_clause`), resource exhaustion (corpus `search_corpus`), over-broad capability (any).
3. **The defenses** — per threat: the specific mechanism (resolve-then-contain, `clause_\d{2}` validation, rate limiter, least-privilege scoping).
4. **The evidence** — for each defense, the probe and its result (e.g. "`read_file('/etc/passwd')` → blocked; secret never returned").
5. **The networked caveat** — a statement that the HTTP server, once deployed beyond localhost, requires authn/authz/edge-rate-limiting because the OS-process boundary no longer applies.
6. **One scoping decision** — a tool you deliberately *did not* expose (`run_shell`, `read_file(absolute_path)`, `run_sql`) and why.

**Acceptance criteria.**

- `notes/week-15/security-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- Each defense is tied to a **specific mechanism**, not "I validate inputs."
- Each defense has **evidence** (a probe and its observed result).
- The networked caveat is stated explicitly.
- Committed.

**Hint.** The memo is read by a reviewer (and, in two weeks, by you, deciding whether your defenses will survive 25 adversarial prompts). Make every claim falsifiable: don't write "the server is safe from traversal," write "`read_file('../SECRET.txt')` and `read_file('/etc/passwd')` both raise `ValueError` and the planted secret is never returned — see `audit` output." A defense without evidence is a hope.

**Estimated time.** 1 hour.

---

## Problem 5 — Consume MCP tools from your week-13 LangGraph agent

**Problem statement.** Take your week-13 LangGraph agent and wire your corpus server into it via `langchain-mcp-adapters`. Run a task that requires a corpus tool (e.g. "what's the confidentiality duration?") and confirm the agent calls `search_corpus`, retrieves `clause_09`, and answers. Record the trace in `notes/week-15/langgraph-mcp.md`.

**Acceptance criteria.**

- The LangGraph agent connects to your corpus MCP server and the MCP tools appear in its tool list.
- A task requiring `search_corpus` completes, with `clause_09` retrieved (the answer survived the protocol).
- The trace shows the tool call and the final answer.
- The embedding/model can be `claude-opus-4-8` *or* a local model from week 6 (vendor-free path) — your choice, stated in the notes.
- Committed.

**Hint.** `MultiServerMCPClient({"corpus": {...}}).get_tools()` returns LangGraph-compatible tools; bind them with `create_react_agent`. For the vendor-free path, point the agent at a local model via an OpenAI-compatible endpoint instead of `claude-opus-4-8` — the MCP machinery is identical.

**Estimated time.** 50 minutes.

---

## Problem 6 — Expose all three primitives

**Problem statement.** Extend your corpus server so it exposes a **tool** (`get_clause`), a **resource** (`corpus://{clause_id}`), and a **prompt** (`summarize_clause`). Inspect it and confirm all three appear with the correct lifecycle methods (`tools/list`, `resources/list`/`resources/read`, `prompts/list`/`prompts/get`). Record which control model each primitive has in `notes/week-15/primitives.md`.

**Acceptance criteria.**

- All three primitives are implemented and discoverable (visible in the inspector or via a programmatic client).
- `notes/week-15/primitives.md` states, for each, its control model: tool = model-controlled, resource = app-controlled, prompt = user-controlled.
- A one-sentence justification for *why* each capability is the primitive it is (an action is a tool, readable context is a resource, a blessed template is a prompt).
- Committed.

**Hint.** `@mcp.tool()` / `@mcp.resource(uri)` / `@mcp.prompt()` are the three decorators. The prompt should return a *template string* (a contracts-analyst summarization prompt), not perform an action — that's what makes it a prompt and not a tool. In the inspector, the prompt shows up under a separate "Prompts" tab and would appear as a slash-command in Claude Desktop.

**Estimated time.** 40 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Trace a session, map the messages | 40 min |
| 2 — Path-traversal defense + tests | 50 min |
| 3 — One server, both transports | 50 min |
| 4 — Tool-surface security memo (headline) | 1 h 0 min |
| 5 — Consume MCP tools from LangGraph | 50 min |
| 6 — Expose all three primitives | 40 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchmcp` [mini-project](./mini-project/README.md) is in the same workspace — week 17 attacks it and week 23 may ship it. Then take the [quiz](./quiz.md) with your notes closed.
