# Lecture 2 — Structured Output, MCP, and the Security Surface

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can pick between JSON-mode and grammar-constrained decoding for an extraction task and justify it, explain MCP and its three transports well enough to consume one, and defend a file/web/code tool against the arguments a hostile party would choose.

Lecture 1 was the *request* — how the model asks for an action. This lecture is the three things that turn a toy tool call into a production one: making the model emit data you can deserialize (§1–2), the open protocol that lets a tool surface outlive any one vendor (§3), and the discipline that keeps a tool from becoming a remote-code-execution hole (§4–6).

---

## Part 1 — Structured output: two ways to get JSON you can trust

There is a difference between "the model returned JSON" and "the model returned JSON that conforms to my schema, every time, so I can `Model.model_validate(...)` without a `try/except`." Getting from the first to the second is *structured output*, and you have two families of technique.

### 1.1 JSON-mode / structured outputs (the vendor path)

On the Anthropic API, you constrain the *response format* with `output_config.format` and a JSON Schema, or — cleaner — hand `messages.parse()` a Pydantic model and get a validated instance back:

```python
from pydantic import BaseModel
import anthropic

class Contact(BaseModel):
    name: str
    email: str
    plan: str
    demo_requested: bool

client = anthropic.Anthropic()

response = client.messages.parse(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "Jane Doe (jane@co.com) wants the Enterprise plan and asked for a demo.",
    }],
    output_format=Contact,
)

contact = response.parsed_output       # a validated Contact instance
print(contact.name, contact.demo_requested)   # "Jane Doe"  True
```

Under the hood this sends `output_config={"format": {"type": "json_schema", "schema": Contact.model_json_schema()}}` and validates the result for you. Same JSON Schema limitations as strict tool use (Lecture 1 §7): supported types + `enum`/`const`/`anyOf`/`$ref`, the string *formats* (`email`, `date-time`, `uri`, `uuid`, …), and `additionalProperties: false`; **not** supported are numeric bounds (`minimum`/`maximum`), string-length bounds, or recursive schemas. The SDK strips unsupported constraints from what it sends and validates them client-side.

The raw form, when you don't want Pydantic:

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[{"role": "user", "content": "..."}],
    output_config={
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "score": {"type": "integer"}},
                "required": ["name", "score"],
                "additionalProperties": False,
            },
        }
    },
)
# The first text block is guaranteed to be valid JSON for that schema.
```

### 1.2 Grammar-constrained decoding (the local path)

The vendor path needs vendor support. For a **local** model — Qwen on Ollama, anything on vLLM/SGLang — you constrain generation yourself with a **grammar**: at each decoding step, the sampler is masked so only tokens that keep the output schema-valid can be chosen. The model is *structurally incapable* of producing invalid JSON. The two libraries you'll meet:

- **`outlines`** — JSON-Schema- and regex-constrained generation, runs over a local model:

```python
import outlines
from pydantic import BaseModel

class Contact(BaseModel):
    name: str
    email: str
    plan: str
    demo_requested: bool

model = outlines.models.transformers("Qwen/Qwen2.5-7B-Instruct")
generator = outlines.generate.json(model, Contact)
contact = generator("Jane Doe (jane@co.com) wants Enterprise and a demo.")
# contact is a validated Contact — the decoder could not emit anything else.
```

- **`xgrammar`** — a fast grammar engine used as the constraint backend inside serving stacks (SGLang, vLLM). You rarely call it directly; you enable it at the server and pass a `json_schema` (or an EBNF grammar) per request. SGLang's structured-output backend is `xgrammar` by default.

### 1.3 Choosing between them

| Dimension | JSON-mode / structured outputs (vendor) | Grammar-constrained decoding (local) |
|---|---|---|
| Where it runs | Vendor API (Anthropic, OpenAI, …) | Your hardware (`outlines`, SGLang/`xgrammar`) |
| Guarantee | Schema-valid output (vendor-enforced) | Schema-valid output (decoder-enforced — *cannot* deviate) |
| Schema power | Types + formats + `enum`; no numeric/length bounds | Full JSON Schema or arbitrary EBNF grammars |
| Latency cost | One-time schema compile (~cached 24h), then cheap | Per-token mask compute; small tax on tokens/sec |
| Portability | Tied to the vendor's API | Tied to your serving stack, but runs offline |
| When to reach for it | You're on a frontier API and want a typed record | You're self-hosting, or you need a grammar JSON Schema can't express (a custom DSL) |

The honest 2026 guidance a senior engineer gives: **on the frontier API, use `messages.parse()` with a Pydantic model — it's the least code and the most robust.** When you're driving a *local* model and "mostly valid JSON" isn't good enough, reach for grammar-constrained decoding, because the decoder *cannot* produce an invalid token — no retry loop, no repair step. The cost is a small tokens/sec tax, which you'll measure in the exercises.

> **Structured output vs tool calling — they're cousins, not the same thing.** A forced tool call (`tool_choice: {type: "tool"}`) *also* gives you a schema-conforming object — the tool's `input`. The difference is intent: tool calling is "take an action," structured output is "this is the answer's shape." For pure extraction (no side effect), `messages.parse()` is cleaner than a fake tool you never actually run.

---

## Part 2 — MCP: the cross-vendor tool protocol

You can hand-write tool definitions for every vendor forever. Or you can speak one protocol that any MCP-aware client understands. That protocol is **MCP — the Model Context Protocol** — an open standard (originated by Anthropic, now broadly adopted) for exposing tools, resources, and prompts to an LLM client over a defined wire format. The pitch a senior engineer gives a new hire: *"MCP is the USB-C of agent tooling — one plug, any client, and it's open."*

### 2.1 The architecture

```
┌────────────┐     MCP (JSON-RPC)     ┌────────────────┐
│ MCP Client │◄──────────────────────►│   MCP Server   │
│ (the host: │   initialize           │ (your tools,   │
│  Claude    │   tools/list           │  resources,    │
│  Desktop,  │   tools/call           │  prompts)      │
│  Cursor,   │   resources/read       │                │
│  your app) │   prompts/get          │                │
└────────────┘                        └────────────────┘
```

An **MCP server** exposes three kinds of capability:

- **Tools** — functions the model can call (the focus of this week). Same idea as Lecture 1's tools, but discovered at runtime via `tools/list` and invoked via `tools/call`.
- **Resources** — read-only data the model can pull in (a file, a record, a query result), discovered via `resources/list` and read via `resources/read`.
- **Prompts** — reusable, parameterized prompt templates the server offers, fetched via `prompts/get`.

An **MCP client** (Claude Desktop, Cursor, or a programmatic client you write) connects to the server, calls `initialize`, lists what's available, and routes the model's tool calls through it. The handshake is JSON-RPC: `initialize` → `initialized` → `tools/list` → `tools/call`.

### 2.2 The three transports

MCP defines *how* the client and server talk over the wire. There are three:

| Transport | What it is | When to use it |
|---|---|---|
| **stdio** | The server runs as a local subprocess; client and server talk over stdin/stdout. | Local tools — a filesystem server, a git server, anything on the same machine. The default for local. |
| **SSE** (Server-Sent Events) | A remote server pushes events over an HTTP SSE stream; the client posts requests separately. | The *legacy* remote transport. You'll still see it; new servers prefer the next one. |
| **streamable HTTP** | A single HTTP endpoint handles request/response and streaming over one connection. | The **current** remote default — simpler, firewall-friendly, the one to choose for new remote servers. |

You will write an MCP server in **Week 15** and consume one this week. The thing to internalize now: MCP decouples the *tool surface* from the *model vendor*. A filesystem MCP server works with Claude Desktop, Cursor, and your own programmatic client equally — you write the tool once, every MCP-speaking client gets it. That's the payoff, and it's why the syllabus treats MCP as the spine and raw per-vendor tool definitions as the thing MCP eventually replaces.

### 2.3 Consuming an MCP server (the shape)

The `anthropic` SDK ships helpers to convert MCP tools into Anthropic tool definitions so the tool runner can call them. The shape, with the official `mcp` Python SDK driving a stdio server:

```python
from anthropic import AsyncAnthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

client = AsyncAnthropic()

async with stdio_client(StdioServerParameters(command="my-fs-server")) as (read, write):
    async with ClientSession(read, write) as mcp:
        await mcp.initialize()
        tools = (await mcp.list_tools()).tools          # tools/list
        runner = client.beta.messages.tool_runner(
            model="claude-opus-4-8", max_tokens=1024,
            messages=[{"role": "user", "content": "List the files in /tmp."}],
            tools=[async_mcp_tool(t, mcp) for t in tools],   # adapt MCP → Anthropic
        )
        async for message in runner:
            print(message)
```

The point of the exercise isn't to master MCP this week — it's to *see* that the model's tool calls route through a protocol that doesn't care which vendor's model is on the other end. Security review of an MCP server (a tool is RCE) is Week 15 and Week 17; here we just consume one.

### 2.4 Raw tool calls vs MCP — when to reach for which

You will leave this week able to do tool calling *two* ways: raw per-vendor definitions (Lecture 1) and MCP (this section). They are not competitors so much as different altitudes:

- **Raw per-vendor tool definitions** are right when the tools are *yours*, live in *your* process, and you're targeting one vendor — the situation for this week's mini-project and the Week 5 agent loop. You hand the model a `tools` list and run a loop. Minimum ceremony, full control.
- **MCP** earns its keep when a tool surface must be *reused across clients* (your app, Claude Desktop, Cursor, a teammate's agent), *discovered at runtime* (the client calls `tools/list` instead of you hard-coding definitions), or *owned by a different team or process* (a database server, an internal-search server). The protocol is the contract; any MCP-speaking client gets the tools for free.

The honest 2026 framing: start with raw tool calls because they're simpler, and graduate a tool to an MCP server the moment it needs to be shared, discovered, or run out-of-process. The syllabus reflects this — you hand-roll raw tools now (Weeks 4–5), and you *author* MCP servers in Week 15 once you have a tool surface worth sharing. The skill this week is recognizing which altitude a given tool belongs at, not memorizing the MCP wire format.

---

## Part 3 — The security surface: a tool is a remote-code-execution primitive

Now the part that separates a demo from a product. Recall the frame: **a tool call is a request to take an action in the world, and the model that emits it is an untrusted client.** Worse — by Week 17 you'll see that a *retrieved document* can inject instructions that steer the model's tool arguments (indirect prompt injection). So you must treat every tool argument as if a hostile party chose it, because one eventually will.

The threat model has a name in the OWASP LLM Top 10: **LLM06, Excessive Agency** — giving a tool more power than the task needs — sitting on top of **LLM01, Prompt Injection** as the delivery mechanism. The defense is the same instinct you saw on the calculator in Lecture 1: **parse and whitelist; never trust, never `eval`, always bound.** Let's apply it to the three dangerous tools.

### 3.1 File-read — path traversal (CWE-22)

The naive file tool:

```python
def read_file(path: str) -> str:
    return open(path).read()      # CATASTROPHIC
```

The model passes `path="../../../../etc/passwd"`, or `path="/home/user/.ssh/id_rsa"`, and your tool happily exfiltrates secrets. The fix is to **confine every access to a sandbox root** and reject anything that escapes it:

```python
import os

SANDBOX = os.path.realpath("./agent_sandbox")

def read_file(path: str) -> str:
    # Resolve against the sandbox, then verify the result is still inside it.
    candidate = os.path.realpath(os.path.join(SANDBOX, path))
    if not (candidate == SANDBOX or candidate.startswith(SANDBOX + os.sep)):
        return "ERROR: path escapes the sandbox"
    if not os.path.isfile(candidate):
        return "ERROR: not a file"
    with open(candidate, "r", encoding="utf-8", errors="replace") as f:
        return f.read(64_000)      # bound the size, too
```

`os.path.realpath` resolves `..` *and* symlinks before the check, which closes the symlink-escape variant (`path="link_to_etc"` where `link_to_etc → /etc`). The `startswith(SANDBOX + os.sep)` guard avoids the classic prefix bug where `./agent_sandbox_evil` would pass a naive `startswith(SANDBOX)`. And you bound the read size, because a tool that returns a 4 GB file is a denial-of-service on your own context window.

### 3.2 Web-fetch — SSRF and private-IP egress

The naive web tool:

```python
import httpx
def fetch_url(url: str) -> str:
    return httpx.get(url).text     # SSRF
```

The model passes `url="http://169.254.169.254/latest/meta-data/iam/security-credentials/"` (the cloud metadata endpoint) and your tool hands cloud credentials to the model. Or `url="http://localhost:6379/"` to poke your Redis, or `url="file:///etc/passwd"`. This is **Server-Side Request Forgery (SSRF)**. The defense: allow only `http`/`https`, resolve the hostname, and **reject private, loopback, and link-local IPs**:

```python
import ipaddress
import socket
from urllib.parse import urlparse
import httpx

def fetch_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "ERROR: only http/https allowed"
    host = parsed.hostname
    if host is None:
        return "ERROR: no host"
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except (socket.gaierror, ValueError):
        return "ERROR: cannot resolve host"
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return "ERROR: refusing to fetch a private/loopback/link-local address"
    resp = httpx.get(url, timeout=5.0, follow_redirects=False)   # don't follow redirects blindly
    return resp.text[:64_000]
```

Two subtleties. First, `follow_redirects=False` — otherwise an allowed public URL can `302` you straight to `169.254.169.254`, defeating the check (this is the classic SSRF-via-redirect bypass; the rigorous fix re-validates after each hop). Second, there's a theoretical DNS-rebinding race (resolve passes, then the name re-resolves to a private IP before the connect); the production-grade fix pins the connection to the validated IP. For the course, the check above is the required bar; the rebinding fix is a stretch goal.

### 3.3 Python sandbox — arbitrary code by definition

A "run this Python" tool is arbitrary code execution *as a feature*. You cannot whitelist your way out of `exec`. The only real answer is **isolation**: run the code in a container or microVM with no network, a read-only filesystem (or a throwaway tmpfs), a CPU/memory/time budget, and dropped capabilities. The Anthropic **code execution tool** is exactly this as a managed server-side sandbox (1 CPU, 5 GiB RAM, no internet, ephemeral) — when you can use a hosted sandbox, do, because rolling your own securely is genuinely hard. If you must self-host, the floor is: a container per call, `--network=none`, `--read-only`, a `mem_limit`, a wall-clock timeout that kills the container, and never the host Python interpreter. **Never `exec()` model code in your own process.** The calculator's AST-whitelist (Lecture 1 §8) is the right pattern when the "code" is just arithmetic; the moment it's general Python, you need a sandbox, not a parser.

---

## Part 4 — Defense in depth, the checklist

No single control is enough; you layer them. For every tool you ship:

1. **Schema validation** (Lecture 1 §6) — reject structurally invalid arguments before anything runs. Cheap, catches garbage.
2. **Semantic validation** — the sandbox check, the IP check, the size bound. JSON Schema can't express "inside this directory"; your code must.
3. **Least privilege** — the tool gets the *minimum* power for the job. A read-only file tool has no write path. A web tool has an allowlist if the task allows one.
4. **Bound everything** — read size, request timeout, output length, recursion. An unbounded tool is a DoS on your own context and wallet.
5. **Confirmation gates for irreversible actions** — `send_email`, `delete`, `git push` should pause for a human (or a policy check) before executing. Reversibility is the criterion: hard-to-undo → gate it.
6. **Return errors *to the model*, not exceptions to yourself** — `is_error: true` results let a capable model recover; an unhandled exception just crashes the loop.

> **The one-line summary of the whole security half:** validate the argument as if a hostile party chose it (because one will), give the tool the least power that does the job, and bound everything it can consume.

---

## Part 5 — Putting it together: the failure-mode decision tree

When a tool-using turn goes wrong, walk this tree — it covers the calling mechanics *and* the security path:

```
The tool-using turn misbehaved.
│
├─ Did the model call the tool at all (stop_reason == "tool_use")?
│   ├─ No  → description too vague, or task didn't need it.
│   │        Fix the description before forcing tool_choice.
│   └─ Yes ↓
│
├─ Did the next request 400?
│   ├─ Yes → a tool_use has no matching tool_result, or vice versa.
│   │        One result per tool_use_id, all in one user turn.
│   └─ No  ↓
│
├─ Did the arguments validate against the schema?
│   ├─ No  → return is_error=True; a capable model self-corrects.
│   │        (A local model fails this more — that's your benchmark.)
│   └─ Yes ↓
│
├─ Are the arguments SEMANTICALLY safe (in sandbox, not a private IP, bounded)?
│   ├─ No  → you have a security bug. Add the semantic check. Do NOT run it.
│   └─ Yes ↓
│
└─ Tool ran but the answer is wrong → check the tool's own logic / the result format you returned.
```

Tape this next to the JSON Schema reference. Between the two, you can diagnose almost any tool-calling problem — and, more importantly, you won't ship the one that matters: running an argument you never validated.

---

## Part 6 — Recap

You should now be able to:

- Get schema-conforming output two ways: `messages.parse()` with a Pydantic model on the frontier path, and `outlines`/`xgrammar` grammar-constrained decoding on the local path — and justify which for a given task.
- Explain MCP — server/client, tools/resources/prompts, and the three transports (stdio, SSE, streamable HTTP) — and consume one MCP server's tools through the SDK.
- State why a tool is a remote-code-execution primitive and apply the parse-and-whitelist instinct to a file tool (path traversal), a web tool (SSRF), and a code tool (isolation).
- Layer defenses: schema validation, semantic validation, least privilege, bounding, confirmation gates, and error-returns-to-the-model.
- Walk the calling-and-security decision tree to find any tool-use failure fast — and to refuse to run an unvalidated argument.

Next: the exercises put all of this on a real tool surface, and the mini-project benchmarks it frontier-vs-local. Continue to [the exercises](../exercises/README.md).

---

## References

- *Structured outputs* — Anthropic docs: <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>
- *`outlines`*: <https://github.com/dottxt-ai/outlines>
- *`xgrammar`*: <https://github.com/mlc-ai/xgrammar>
- *MCP — specification*: <https://modelcontextprotocol.io/specification/>
- *MCP — introduction*: <https://modelcontextprotocol.io/introduction>
- *MCP Python SDK*: <https://github.com/modelcontextprotocol/python-sdk>
- *OWASP Top 10 for LLM Applications*: <https://genai.owasp.org/llm-top-10/>
- *SSRF prevention cheat sheet (OWASP)*: <https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html>
- *CWE-22 (path traversal)*: <https://cwe.mitre.org/data/definitions/22.html>
