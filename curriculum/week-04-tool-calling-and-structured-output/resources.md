# Week 4 — Resources

Every resource here is **free**. The Anthropic docs are open. The MCP spec and SDKs are open-source. The structured-output libraries (`outlines`, `xgrammar`) are open-source. Ollama is open-source. No paywalled books are linked.

Model IDs move; the concepts don't. Where a doc pins a model, the IDs current in 2026 are `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5` for Anthropic, and `qwen2.5:7b-instruct` for the local path. If you read this in a later cohort, swap the SKUs — the tool-call contract is stable.

## Required reading (work it into your week)

- **Tool use with Claude — overview** — the canonical reference for `tools`, `tool_use`, `tool_result`, and the loop. Read it Monday and again Thursday:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>
- **Implement tool use** — the request/response shapes, `tool_choice`, parallel tool use, error handling:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use>
- **Structured outputs** — `output_config.format`, `strict: true`, JSON Schema limitations, and `messages.parse`:
  <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>
- **Model Context Protocol — specification** — server/client, the three transports, tools/resources/prompts:
  <https://modelcontextprotocol.io/specification/>
- **MCP — introduction** — the "why" and the architecture diagram, in the maintainers' words:
  <https://modelcontextprotocol.io/introduction>

## Cross-vendor function calling (read the shapes side by side)

You will not memorize every vendor's JSON. But you should be able to look at two and say what's portable.

- **OpenAI — function calling** — the `tools` / `tool_calls` shape and the loop:
  <https://platform.openai.com/docs/guides/function-calling>
- **Google Gemini — function calling** — `function_declarations` and the call/response cycle:
  <https://ai.google.dev/gemini-api/docs/function-calling>
- **Ollama — tool support** — the `tools` field (OpenAI-shaped) and which local models support it:
  <https://ollama.com/blog/tool-support>
- **Qwen 2.5 — function calling / Hermes tool template** — what the open-weights model was trained to emit:
  <https://qwen.readthedocs.io/en/latest/framework/function_call.html>

## Structured output & grammar-constrained decoding

- **`outlines`** — regex- and JSON-Schema-constrained generation for local models:
  <https://github.com/dottxt-ai/outlines>
- **`xgrammar`** — fast grammar-constrained decoding, the backend behind several serving stacks:
  <https://github.com/mlc-ai/xgrammar>
- **SGLang — structured outputs** — grammar/JSON-Schema constraints at serving time:
  <https://docs.sglang.ai/backend/structured_outputs.html>
- **`pydantic` v2 — JSON Schema generation** — `model_json_schema()` is how you turn a model into a tool's `input_schema`:
  <https://docs.pydantic.dev/latest/concepts/json_schema/>
- **JSON Schema — understanding** — the reference you keep open while hand-writing schemas:
  <https://json-schema.org/understanding-json-schema/>

## The security surface (read before you write a single tool)

- **OWASP Top 10 for LLM Applications** — LLM01 prompt injection, LLM06 excessive agency, the tool-use threats:
  <https://genai.owasp.org/llm-top-10/>
- **Anthropic — tool-use best practices & safety** — argument validation, least privilege, confirmation gates:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>
- **CWE-22 — path traversal** — the file-tool attack you will defend against in Exercise 3:
  <https://cwe.mitre.org/data/definitions/22.html>
- **SSRF prevention cheat sheet (OWASP)** — the web-fetch attack you will defend against:
  <https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html>

## API references (the ones you'll have open all week)

- **Anthropic Python SDK** — `messages.create`, `messages.parse`, the `@beta_tool` tool runner:
  <https://github.com/anthropics/anthropic-sdk-python>
- **`anthropic` tool-use examples** — runnable manual-loop and tool-runner samples:
  <https://github.com/anthropics/anthropic-sdk-python/tree/main/examples>
- **MCP Python SDK (`mcp`)** — what you'll use in Week 15; skim the `Server`/`Client` and transport classes now:
  <https://github.com/modelcontextprotocol/python-sdk>

## Talks & posts worth your time (free, no signup)

- **Anthropic — "Building effective agents"** — the foundational post; the tool-use and workflow patterns the course leans on:
  <https://www.anthropic.com/research/building-effective-agents>
- **Anthropic engineering — writing tools for agents** — how to design a tool surface a model will actually use well:
  <https://www.anthropic.com/engineering>
- **MCP — example servers** — read a real server's tool definitions before you write your own:
  <https://github.com/modelcontextprotocol/servers>

## Tools you'll use this week

- **`anthropic`** — `pip install anthropic`. The frontier path for every lab.
- **`ollama`** — the local path; `ollama pull qwen2.5:7b-instruct`. Serves an OpenAI-compatible `/v1` endpoint and a native `/api/chat`.
- **`pydantic`** — `pip install pydantic`. Schemas, validation, and `model_json_schema()`.
- **`outlines`** — `pip install outlines`. Grammar/JSON-Schema-constrained generation for the local model.
- **`httpx`** — `pip install httpx`. The web-fetch tool's HTTP client (with the SSRF guard you'll write).
- **`jsonschema`** — `pip install jsonschema`. Validate a model's tool `input` against your schema before you run anything.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Tool** | A function the model can request; you define its name, description, and `input_schema`. |
| **`input_schema`** | A JSON Schema describing the tool's arguments. The contract the model fills in. |
| **`tool_use` block** | The assistant's request to call a tool — carries `id`, `name`, and `input`. |
| **`tool_result` block** | Your reply — carries `tool_use_id` (must match), `content`, and optional `is_error`. |
| **`tool_use_id`** | The string tying a `tool_result` back to its `tool_use`. Mismatch → 400. |
| **`tool_choice`** | `auto` / `any` / `tool` / `none`; whether and which tool the model must call. |
| **Parallel tool use** | One assistant turn emitting several `tool_use` blocks at once. |
| **`stop_reason=tool_use`** | The model wants a tool run; the loop continues. |
| **JSON-mode / structured output** | Constraining the *response* to a JSON Schema (`output_config.format`). |
| **`strict: true`** | Constraining a *tool's* arguments to exactly its schema; no extra/missing fields. |
| **Grammar-constrained decoding** | Masking the sampler so only schema-valid tokens can be emitted (`outlines`, `xgrammar`). |
| **MCP** | Model Context Protocol — the open, cross-vendor tool/resource/prompt protocol. |
| **stdio / SSE / streamable HTTP** | MCP's three transports: local subprocess / legacy remote / current remote. |
| **Tool-call accuracy** | Fraction of a fixed task set where the model called the right tool with valid args. |
| **Excessive agency** | OWASP LLM06 — giving a tool more power than the task needs. |

---

*If a link 404s, please open an issue so we can replace it.*
