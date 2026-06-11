# Mini-Project — `crunchmcp`: A Multi-Transport MCP Toolkit

> Build a reusable MCP toolkit — two production-shaped servers (filesystem + private-corpus search), each runnable over stdio *and* streamable HTTP, a documented tool surface, a programmatic consumer, and a security review — so "expose this capability as a tool any agent can use" becomes `python -m crunchmcp serve corpus --transport streamable-http` and a config line, not a bespoke per-framework integration.

This is the artifact that turns a pile of tool functions into a portable tool surface. After this week, adding a capability to your agents is "write an MCP server, point a client at it" — not "re-implement this tool for LangGraph, then again for Mastra, then again for Claude Desktop." The toolkit is transport-agnostic, client-agnostic, and security-reviewed, and it's the surface week 17 attacks and week 23's capstone reuses.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This toolkit is consumed directly by your **week-17 safety lab**, where you red-team the corpus + filesystem agent with 25 adversarial prompts and harden it. It's also a candidate for the **week-23 capstone MCP tool surface** (filesystem, web, calculator, custom). The syllabus says the safety lab attacks "your week-15 MCP-tool agent"; *this toolkit is that agent's tool surface*. Build it well now; you'll defend it under fire in two weeks and ship it in the capstone.

---

## What you will build

A small Python package `crunchmcp` with five deliverables:

1. **`crunchmcp/servers/filesystem.py`** — a sandboxed filesystem MCP server: `list_files`, `read_file`, `write_file`, all confined to a sandbox root by a resolve-then-contain path check. The defining security surface of the toolkit.
2. **`crunchmcp/servers/corpus.py`** — a private-corpus-search MCP server over the week-7 legal corpus: `search_corpus(query)` (rate-limited), `get_clause(clause_id)` (argument-validated), plus a `corpus://{clause_id}` resource and a `summarize_clause` prompt. The "custom domain tool" the syllabus calls for.
3. **`crunchmcp/transport.py`** — a thin runner that serves *any* server over a chosen transport (`stdio` or `streamable-http`), so the same server logic deploys both ways with one flag. The single source of truth for "how a server is launched."
4. **`crunchmcp/inspect.py`** — a programmatic `ClientSession`-based inspector: connect to a named server over a chosen transport, run the full lifecycle (initialize → list → call → read), and print the JSON-RPC-level summary. Your own version of the MCP Inspector, scriptable and CI-able.
5. **`crunchmcp/cli.py`** — `serve`, `inspect`, and `audit` commands that tie it together.

By the end you have a public repo of ~450–550 lines of Python that any future agent project can point a client at and stop re-implementing tools per framework.

---

## Why a package and not a pile of scripts

You could leave each server as a loose `.py` file. Don't — not as the artifact. A package gives you:

- **Reuse.** Week 17 imports your servers to attack them; week 23 imports them for the capstone surface. A loose script gets copy-pasted, drifts, and rots.
- **One transport story.** "Serve this over HTTP" is `serve corpus --transport streamable-http`, not "go edit the runner at the bottom of the file." The transport choice lives in one place, version-controlled.
- **A real consumer.** `inspect` is a programmatic client you can run in CI to assert "the corpus server still returns clause_09 for the confidentiality query" on every push. A manual inspector session is none of those.
- **An audit command.** `audit` runs the security probes (traversal, malformed args, rate limit) and prints PASS/FAIL — so "is this surface still safe?" is a command, not a memory.

Loose scripts are great for *exploring* a single server by hand (Exercise 1/2 territory). The thing you ship and depend on — and red-team in two weeks — is a package. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchmcp/
├── pyproject.toml
├── README.md                     # the tool-surface docs + the security memo
├── corpus/
│   └── legal_corpus.json         # the week-7 clauses: {"clause_09": "...", ...}
├── crunchmcp/
│   ├── __init__.py
│   ├── servers/
│   │   ├── __init__.py
│   │   ├── filesystem.py         # sandboxed fs server (list/read/write)
│   │   └── corpus.py             # corpus search server (search/get + resource + prompt)
│   ├── transport.py              # serve(server, transport) — stdio | streamable-http
│   ├── inspect.py                # programmatic ClientSession lifecycle runner
│   ├── security.py               # _safe_path, validate_clause_id, RateLimiter
│   └── cli.py                    # serve / inspect / audit
└── tests/
    ├── test_security.py          # path traversal + arg validation + rate limit
    └── test_roundtrip.py         # client drives server: initialize -> call -> assert
```

---

## Deliverable 1 — `servers/filesystem.py` (the sandboxed surface)

This is the security heart of the toolkit. Every tool routes its path through `_safe_path` from `security.py`; nothing touches the disk without a resolved-containment check.

```python
"""crunchmcp.servers.filesystem — a sandboxed filesystem MCP server.

Every path argument is hostile until proven otherwise (the model, possibly
steered by an injection, supplies it). _safe_path resolves THEN checks
containment; no tool bypasses it.
"""
from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from crunchmcp.security import safe_path   # resolve-then-contain

SANDBOX = Path("./sandbox").resolve()
SANDBOX.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("crunch-filesystem")


@mcp.tool()
def read_file(relative_path: str) -> str:
    """Read a text file from inside the sandbox. Path is relative to the root."""
    target = safe_path(SANDBOX, relative_path)        # raises if it escapes
    if not target.is_file():
        raise FileNotFoundError(relative_path)
    return target.read_text()


# TODO 1: list_files(subdir="") — list relative paths under a sandbox subdir,
#   routing `subdir` through safe_path(). Reject a subdir that escapes.

# TODO 2: write_file(relative_path, content) — write inside the sandbox, again
#   via safe_path(). Create parent dirs. Return a "wrote N chars" confirmation.


def get_server() -> FastMCP:
    """Return the configured server so transport.py can run it any transport."""
    return mcp
```

> **The rule the project enforces:** no tool touches the filesystem except through `safe_path()`. If `grep -rn "open(\|read_text\|write_text\|Path(" crunchmcp/servers` shows a disk access that didn't route through `safe_path`, you've opened a hole. The audit command's traversal probe exists to catch exactly that.

---

## Deliverable 2 — `servers/corpus.py` (the custom-domain surface)

The corpus server wraps the week-7 legal corpus and exposes all three primitives — a tool, a resource, and a prompt — so the toolkit demonstrates the full surface, not just tools.

```python
from mcp.server.fastmcp import FastMCP

from crunchmcp.security import RateLimiter, validate_clause_id

mcp = FastMCP("crunch-corpus")
_limiter = RateLimiter(max_per_minute=30)
CORPUS = _load_corpus()   # {"clause_09": "...protected for five years...", ...}


@mcp.tool()
def search_corpus(query: str) -> list[dict]:
    """Search the legal corpus for clauses matching a query. Use when the user
    asks about contract terms (confidentiality, termination, fees, ...)."""
    _limiter.check()                                  # rate limit (raises if over)
    return _rank(query, CORPUS)[:3]


@mcp.tool()
def get_clause(clause_id: str) -> str:
    """Fetch one clause by id (e.g. 'clause_09')."""
    validate_clause_id(clause_id)                     # argument validation (clause_NN)
    return CORPUS[clause_id]


@mcp.resource("corpus://{clause_id}")
def clause_resource(clause_id: str) -> str:
    """App-controlled context: read a clause by URI."""
    return CORPUS.get(clause_id, f"<no such clause: {clause_id}>")


@mcp.prompt()
def summarize_clause(clause_id: str) -> str:
    """User-invoked prompt: a blessed template for one-line clause summaries."""
    return (
        "You are a contracts analyst. Summarize this clause's obligation in one "
        f"sentence.\n\nClause {clause_id}:\n{CORPUS.get(clause_id, '')}"
    )
```

> The three primitives here aren't decoration — they're the point. `search_corpus`/`get_clause` are **model-controlled** tools (the agent calls them); `corpus://{clause_id}` is an **app-controlled** resource (the host reads it into context); `summarize_clause` is a **user-controlled** prompt (a slash-command). One server, all three control models — exactly the triad from Lecture 1.

---

## Deliverable 3 — `transport.py` (one server, any transport)

The runner that proves the transport is a deployment decision, not a code decision.

```python
from mcp.server.fastmcp import FastMCP


def serve(server: FastMCP, transport: str = "stdio") -> None:
    """Run a FastMCP server over the chosen transport. The server's tool code is
    byte-for-byte identical across transports — only this runner changes."""
    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "streamable-http":
        server.run(transport="streamable-http")       # serves on /mcp
    else:
        raise ValueError(f"unknown transport: {transport!r}")
```

The non-negotiable this enforces: **the same server object serves both transports.** You never fork the server for HTTP vs stdio; you flip the runner. The CLI's `serve` command is a one-liner over this.

---

## Deliverable 4 — `inspect.py` (your own programmatic client)

A scriptable client that runs the full lifecycle against a named server and prints a wire-level summary — the thing you'd run in CI to assert the surface still works.

```python
async def inspect_server(name: str, transport: str) -> dict:
    """Connect to a server, run initialize -> list -> call -> read, return a summary."""
    async with _open(name, transport) as session:    # stdio or streamablehttp client
        await session.initialize()                    # handshake FIRST
        tools = await session.list_tools()
        # TODO 3: call a representative tool, read a representative resource,
        #   and return {"tools": [...], "sample_call": ..., "sample_resource": ...}
        ...
```

This is the file behind the week's "any client, any server" promise marker:

```
$ python -m crunchmcp inspect --server corpus --transport stdio
initialize    -> protocolVersion=2025-06-18  capabilities={tools, resources, prompts}
tools/list    -> [get_clause, search_corpus]
tools/call search_corpus {"query": "five-year confidentiality"}
  -> clause_09 (score 3): "...protected for five years after termination."  ✓
```

---

## Deliverable 5 — `cli.py` (serve / inspect / audit)

```bash
# Serve a server over a chosen transport:
python -m crunchmcp serve corpus --transport streamable-http
python -m crunchmcp serve filesystem --transport stdio

# Inspect a running/spawnable server (the full lifecycle, printed):
python -m crunchmcp inspect --server corpus --transport stdio

# Run the security audit (traversal + arg-validation + rate-limit probes):
python -m crunchmcp audit
```

`audit` should print, per server, the probe results:

```
SERVER       PROBE                       RESULT
filesystem   path-traversal (8 probes)   PASS (all escapes blocked)
corpus       arg-validation (5 probes)   PASS (all malformed ids rejected)
corpus       rate-limit                  PASS (rejects past 30/min)
--------------------------------------------------------------------
audit: PASS — surface is safe to consume. (week 17 will try harder.)
```

The point is that "is this tool surface safe?" is a command with a printed answer and a green/red verdict — not a vibe and not a memory.

---

## Rules

- **You may** read the MCP spec, the SDK docs, the lecture notes, and your exercise code.
- **You must not** let any filesystem tool touch the disk except through `safe_path()` — the resolve-then-contain check is the toolkit's load-bearing defense. Bypassing it is how the surface gets owned in week 17.
- **You must not** skip argument validation on `get_clause` — the `clause_\d{2}` check is what turns a malformed id into a clean tool error instead of a crash or an injection.
- **You must** serve each server over *both* transports from the *same* server object — no forking the server for HTTP. The transport abstraction is the week's structural lesson.
- Python 3.12, `mcp[cli]`, `httpx`, plus `pytest`. The LangGraph consumption path (`langchain-mcp-adapters`) is the challenge's job; the mini-project's consumer is your own `inspect.py`, so the toolkit has no agent-framework dependency.
- Keep tools **narrow and scoped** — no `run_shell`, no `read_file(absolute_path)`, no `run_sql`. Least privilege is graded.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-15-crunchmcp-<yourhandle>`.
- [ ] `crunchmcp/servers/filesystem.py` and `corpus.py` each run over **both** stdio and streamable HTTP via `transport.serve(...)` — proven by `serve` working with either `--transport` value.
- [ ] The corpus server exposes a **tool**, a **resource**, and a **prompt** (all three primitives), with the correct control models.
- [ ] Every filesystem tool routes through `safe_path()`; `audit`'s traversal probe shows all escapes blocked.
- [ ] `get_clause` validates `clause_id`; `audit`'s arg-validation probe shows malformed ids rejected as tool errors.
- [ ] `search_corpus` is rate-limited; `audit`'s rate-limit probe shows rejection past the limit.
- [ ] `python -m crunchmcp inspect --server corpus --transport stdio` prints the full lifecycle and retrieves `clause_09` for the confidentiality query (the promise marker).
- [ ] `pytest` passes, with at least:
  - `test_security.py`: traversal blocked, malformed `clause_id` rejected, rate limit enforced.
  - `test_roundtrip.py`: a programmatic client drives a server through initialize → call → assert.
- [ ] A `README.md` with the tool-surface docs (every tool: name, args, what it does, why it's scoped that way) and the **one-page security memo** (per-server threats, defenses, evidence).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Server correctness** | 25 | Both servers implement their tools correctly; the corpus server exposes all three primitives with the right control models; tool descriptions are prescriptive (the model would call them at the right time). |
| **Transport portability** | 20 | The same server object serves both stdio and streamable HTTP via one runner; no server is forked per transport; `serve` works with either flag. |
| **Security** | 25 | `safe_path` resolves-then-contains (not a string check); all fs tools route through it; `get_clause` validates; `search_corpus` rate-limits; `audit` proves all three; capabilities are scoped to least privilege. |
| **Consumer & lifecycle** | 15 | `inspect.py` runs the full `ClientSession` lifecycle (initialize first) and retrieves clause_09; it's scriptable, not a manual session. |
| **Tests** | 10 | `test_security` covers traversal/validation/rate-limit; `test_roundtrip` drives a real client/server round trip; `pytest` green. |
| **Docs & hygiene** | 5 | Clear tool-surface README + security memo, no secrets committed, sensible commits, no `__pycache__`/`.venv`/`sandbox/` checked in. |

**90+** is portfolio-grade and ready to be attacked in week 17 / shipped in week 23. **70–89** works but has a soft defense (a string-based path check, a missing rate limit) or a forked-per-transport server. **Below 70** means the surface isn't safe or isn't portable — fix that first, because week 17's red team starts from this exact toolkit.

---

## Stretch goals

- **The third client.** Wire your corpus server into Claude Desktop via `claude_desktop_config.json` and confirm the *same* server answers from Desktop. Add a `clients.md` documenting the config for Desktop, Cursor, and a programmatic client — the "any client" promise, documented.
- **Resource subscriptions.** Make a resource that changes (a log the server appends to), advertise the `subscribe` capability, and have `inspect.py` subscribe and observe `notifications/resources/updated`. The full resource lifecycle, not just reads.
- **An aggregating gateway.** Stand up an OpenClaw-family MCP gateway in front of both servers and add a `serve gateway` command so one client connection sees the union of both surfaces. Measure the per-hop latency in the README.
- **CI.** A GitHub Actions workflow that runs `pytest` *and* `python -m crunchmcp audit` headless on every push, failing the build if any security probe regresses. Green check = the surface is still safe.

---

## How this connects to the rest of C23

- **Week 4 (tool calling)** gave you tools as JSON schemas inside one API call; this toolkit moves those tools into their own processes behind the open protocol, so any client can use them.
- **Week 13 (LangGraph)** gave you the agent that *consumes* these tools — the challenge wires this toolkit into that agent via `langchain-mcp-adapters`.
- **Week 17 (safety)** *attacks* this exact toolkit: 25 adversarial prompts, direct and indirect injection, tool-argument abuse, measured attack-success-rate before and after hardening. Your `safe_path`, your validation, your rate limit are what the red team tries to break — build them to survive.
- **Week 23 (capstone)** reuses this as a candidate for the capstone's MCP tool surface (filesystem, web, calculator, custom). A scoped, transport-portable, security-reviewed toolkit is exactly what the capstone wants.

When you've finished, push the repo and take the [quiz](../quiz.md).
