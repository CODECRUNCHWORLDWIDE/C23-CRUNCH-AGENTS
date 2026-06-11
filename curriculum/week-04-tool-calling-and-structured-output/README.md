# Week 4 — Tool Calling and Structured Output

Welcome to the week where the model stops being a text generator and starts being a thing that *acts in the world*. By Friday you will be able to define a tool schema that any 2026 frontier model — or a local Qwen — will call correctly, read a `tool_use` block the way a backend engineer reads a request body, validate the arguments like they came from an untrusted client (because they did), and force a model to emit JSON that conforms to a schema you can deserialize without a `try/except` around `json.loads`.

We assume you finished Week 3 and can version a prompt, diff it, and run it through a `promptfoo` regression harness. We also assume you have an Anthropic API key in your environment **and** a local `qwen2.5:7b-instruct` pulled in Ollama, because every lab this week runs against *both* — a frontier vendor model and an open-weights local model — using the *same tool schema*. If your Ollama isn't serving, fix that first; `ollama run qwen2.5:7b-instruct "hi"` should print a reply.

The one thing to internalize before you read another line: **a tool call is a request to take an action in the world, and the model that emits it is an untrusted client.** The model does not run your tool. *You* run your tool, with arguments the model chose, and the model will sometimes choose arguments that are wrong, malformed, or actively hostile (especially once a retrieved document can influence them — that's Week 17, but the discipline starts now). The schema is the contract. JSON-mode and grammar-constrained decoding are how you make the model *hold up its end*. Argument validation is how you hold up yours.

This week is where you stop trusting model output and start contracting it.

## Learning objectives

By the end of this week, you will be able to:

- **Define** a tool with a correct JSON Schema `input_schema` — typed properties, `required`, `enum`, `additionalProperties: false` — that the Anthropic API, OpenAI, Gemini, and a local Qwen all accept and call.
- **Trace** a full tool-use turn: the `tool_use` block in the assistant message, the `tool_result` you send back, the `tool_use_id` that ties them, and the second model turn that consumes the result.
- **Compare** function-calling across vendors — Anthropic `tool_use`, OpenAI `tools`, Gemini function calling, and the open-source equivalent on Llama/Qwen through Ollama — and state what is portable and what is not.
- **Explain** MCP (the Model Context Protocol) as the cross-vendor tool layer: its server/client model and its three transports (stdio, SSE, streamable HTTP), and where it fits relative to raw tool-calling.
- **Choose** between JSON-mode (`output_config.format` / structured outputs) and grammar-constrained decoding (`outlines`, `xgrammar`) for a given extraction task, and justify the choice.
- **Defend** every tool against malicious or malformed arguments: path-traversal on a file tool, arbitrary-code on a Python tool, SSRF on a web-fetch tool, injection on a calculator.
- **Measure** tool-call accuracy on a fixed benchmark and report a number, not a vibe.
- **Build** a reusable four-tool agent surface (calculator, file-read, web-fetch, Python sandbox) that runs identically against a frontier model and a local model.

## Prerequisites

This week assumes you have completed **C23 weeks 1–3**, or have equivalent fluency. Specifically:

- Python 3.12, comfortable with type hints, `dataclasses`, and `pydantic` v2 (`BaseModel`, `model_validate`, `model_json_schema`).
- The `anthropic` SDK installed (`pip install anthropic`) and `ANTHROPIC_API_KEY` exported. `python -c "import anthropic; print(anthropic.__version__)"` works.
- **Ollama** installed with `qwen2.5:7b-instruct` pulled. `ollama list` shows it. You can `curl http://localhost:11434/api/chat`.
- You can read and write JSON Schema by hand — `type`, `properties`, `required`, `enum`, `additionalProperties`. If you can't, the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) is your Monday-morning reading.
- You finished Week 2 and know what JSON-mode and grammar-constrained decoding *are* at a concept level. This week we make them load-bearing.

You do **not** need prior MCP experience. We introduce it at the protocol level and you will write a tiny MCP server in Week 15; this week it's awareness plus one hands-on consume.

## Topics covered

- **The anatomy of a tool call**: `input_schema` as a JSON Schema contract; the `tool_use` content block (`id`, `name`, `input`); the `tool_result` content block (`tool_use_id`, `content`, `is_error`); the two-turn round trip; `stop_reason == "tool_use"`.
- **Cross-vendor function calling**: Anthropic `tools` + `tool_use`, OpenAI `tools` + `tool_calls`, Gemini `function_declarations`, and the Ollama/Qwen `tools` field that mirrors the OpenAI shape. What's portable (the JSON Schema), what's not (block names, the loop mechanics, streaming of partial args).
- **`tool_choice`**: `auto`, `any`, `tool` (forced), `none`, and `disable_parallel_tool_use`. When to force a tool and when forcing one is a smell.
- **Parallel tool use**: a single assistant turn can emit multiple `tool_use` blocks; you must return one `tool_result` per `tool_use_id` or the next turn 400s.
- **MCP as the cross-vendor protocol**: server, client, and the three transports — **stdio** (local subprocess), **SSE** (legacy remote), **streamable HTTP** (the current remote default); tools vs resources vs prompts; why MCP is "the USB-C of agent tooling."
- **Structured output two ways**: JSON-mode / structured outputs (`output_config.format` with a `json_schema`, `client.messages.parse()` with a Pydantic model, `strict: true` tool use) versus grammar-constrained decoding (`outlines`, `xgrammar`, SGLang's grammar backend) for local models. The accuracy/latency/portability trade-off.
- **The tool-use security surface**: a tool is a remote-code-execution primitive. Path traversal on file-read, SSRF and private-IP egress on web-fetch, arbitrary-code and resource exhaustion on a Python sandbox, and why you validate every argument as if a hostile party chose it.
- **Measuring tool-call accuracy**: a fixed task set, a deterministic grader, and a reported pass rate — frontier vs local, with the cost and latency next to the accuracy.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                              | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|----------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Anatomy of a tool call; the two-turn round trip    |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Cross-vendor calling; Ollama parity; `tool_choice` |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Structured output; JSON-mode vs grammar decoding   |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Tool security; sanitizing untrusted arguments      |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | MCP overview; the four-tool surface; benchmark      |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                             |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, benchmark write-up polish            |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                    | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The Anthropic tool-use docs, MCP spec, structured-output guides, and the talks worth your time |
| [lecture-notes/01-the-tool-call-contract.md](./lecture-notes/01-the-tool-call-contract.md) | The tool-use round trip, cross-vendor calling, `tool_choice`, parallel tools, and Ollama parity |
| [lecture-notes/02-structured-output-mcp-and-the-security-surface.md](./lecture-notes/02-structured-output-mcp-and-the-security-surface.md) | JSON-mode vs grammar decoding, MCP and its transports, and defending each tool against hostile arguments |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-design-a-tool-schema.md](./exercises/exercise-01-design-a-tool-schema.md) | Hand-write three tool schemas, validate them, and call them against Claude and Qwen |
| [exercises/exercise-02-structured-extraction.py](./exercises/exercise-02-structured-extraction.py) | Extract a typed `Pydantic` record three ways: `messages.parse`, raw `output_config.format`, and `outlines` on a local model |
| [exercises/exercise-03-sanitize-the-tools.py](./exercises/exercise-03-sanitize-the-tools.py) | Harden a file-read and a web-fetch tool against path traversal and SSRF; prove the attack fails |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-cross-vendor-tool-bridge.md](./challenges/challenge-01-cross-vendor-tool-bridge.md) | Make one tool registry drive Claude *and* Qwen with no per-vendor tool code, then benchmark both |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the tool-call accuracy benchmark write-up |
| [mini-project/README.md](./mini-project/README.md) | The reusable four-tool agent surface with a 50-task accuracy benchmark, frontier vs local |

## The "the model called the tool correctly" promise

C23 uses a recurring marker for every exercise that ends in a model actually calling a tool with valid arguments and consuming the result:

```text
>>> turn 1: stop_reason=tool_use
    tool_use  calculator(expression="(1234 * 7) + 19")
>>> ran calculator -> "8657"
>>> turn 2: stop_reason=end_turn
    "The answer is 8657."
```

If `stop_reason` is `end_turn` when you expected `tool_use`, the model declined the tool — usually because the description was vague or the question didn't actually need it. If you sent a `tool_result` without a matching `tool_use_id`, the API 400s. If the `input` doesn't validate against your schema and you ran it anyway, you have a security bug, not a feature. The point of Week 4 is to make that clean two-turn trace ordinary — and to make every deviation *loud*.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **MCP specification** end to end and draw the stdio handshake (`initialize` → `initialized` → `tools/list` → `tools/call`) from memory: <https://modelcontextprotocol.io/specification/>.
- Wire the **`anthropic` tool runner** (`client.beta.messages.tool_runner` with the `@beta_tool` decorator) and compare its loop to your hand-rolled one. Note exactly what it does for you and what it hides.
- Implement **strict tool use** (`"strict": True` on a tool definition) and confirm the model can no longer emit an out-of-schema argument. Compare against the non-strict failure rate on a deliberately ambiguous task.
- Run the **`xgrammar`** backend behind SGLang for grammar-constrained decoding of a local model and measure the tokens/sec tax versus unconstrained generation.

## Up next

Week 5 takes the tool surface you built here and wraps it in **the agent loop** — reason, act, observe, repeat — with step, token, time, and cost budgets so the thing actually terminates. Push your mini-project before you start it; the agent loop imports this week's tool registry directly.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
