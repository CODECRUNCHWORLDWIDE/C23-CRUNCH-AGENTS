# Lecture 1 — The Tool-Call Contract: What the Model Actually Asks For

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can define a tool with a correct JSON Schema, trace a full two-turn tool-use round trip, run the same tool schema against a frontier model and a local Qwen, and use `tool_choice` deliberately instead of by accident.

If you remember one sentence from this entire week, remember this one:

> **A tool call is a request to take an action in the world. The model that emits it is an untrusted client. The schema is the contract; you are the runtime; validation is your half of the deal.**

A lot of engineers meet tool calling and think "the model can run functions now." It cannot. The model emits a structured *request* — "I would like to call `get_weather` with `{"city": "Paris"}`" — and then *stops*. Your code reads that request, decides whether to honor it, runs the function with arguments **the model chose**, and hands the result back. The model never touches your filesystem, your network, or your database. You do, on its behalf, with inputs from a source you do not fully control. Hold that frame and the rest of the week is mechanics.

---

## 1. Where tool calling lives in the request

A tool-enabled request to the Anthropic Messages API has three load-bearing pieces:

1. **`tools`** — a list of tool definitions. Each has a `name`, a `description`, and an `input_schema` (a JSON Schema object).
2. **`messages`** — the conversation so far.
3. **`tool_choice`** (optional) — whether the model *may*, *must*, or *must not* call a tool, and if it must, which one.

Here is the minimal shape in the `anthropic` Python SDK:

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns temperature in Celsius and a short condition string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris' or 'San Francisco, CA'.",
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    }
]

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris right now?"}],
)
print(response.stop_reason)   # 'tool_use'
```

The `input_schema` is the part you must get right. It is a JSON Schema `object`, and the model reads it to decide both *whether* the tool applies and *how* to fill in the arguments. Three rules carry most of the weight:

- **Every property gets a `description`.** The model reads property descriptions the same way it reads the tool description. A `city` field with no description is a field the model will fill in inconsistently.
- **`required` is honest.** List only the fields the tool genuinely cannot run without. Everything else is optional with a sane default in *your* code.
- **`additionalProperties: false`.** This tells the model — and, with strict mode, the decoder — that no extra fields are allowed. It also makes structured-output strict mode legal (Anthropic requires it).

> **The description is the prompt.** A vague tool description is the single most common reason a model "won't call the tool." Before you debug the loop, read your description out loud and ask: would a competent intern know exactly when to use this and what each argument means?

---

## 2. The two-turn round trip

Tool use is **stateless and two-turn**. The API does not run your tool and continue; it returns the `tool_use` request and stops. You run the tool, append the result, and call the API *again*. Walk the trace:

### Turn 1 — the model requests a tool

The response from §1 contains a `tool_use` content block:

```python
for block in response.content:
    if block.type == "tool_use":
        print(block.id)     # 'toolu_01A...'  — the tool_use_id
        print(block.name)   # 'get_weather'
        print(block.input)  # {'city': 'Paris'}  — already a parsed dict
```

`block.input` is a **parsed Python dict**, not a JSON string. (Do not raw-string-match the serialized form — current models may escape Unicode or slashes differently. Trust the parsed object.) The `block.id` — the `tool_use_id` — is the string that ties your result back to this request. Lose it and you can't reply.

### You run the tool

This is your code, your runtime, your security boundary:

```python
def get_weather(city: str) -> str:
    # ... call a weather API, or stub it ...
    return "18°C, partly cloudy"

tool_use = next(b for b in response.content if b.type == "tool_use")
result_text = get_weather(**tool_use.input)
```

### Turn 2 — you send the result back

You append **two** messages: the assistant's turn (echoing its `tool_use` so the model sees its own request) and a user turn carrying the `tool_result`:

```python
messages = [
    {"role": "user", "content": "What's the weather in Paris right now?"},
    {"role": "assistant", "content": response.content},   # echo the tool_use block
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use.id,    # MUST match the tool_use block's id
                "content": result_text,
            }
        ],
    },
]

final = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    tools=tools,
    messages=messages,
)
print(final.stop_reason)   # 'end_turn'
print(next(b.text for b in final.content if b.type == "text"))
# "It's currently 18°C and partly cloudy in Paris."
```

That's the whole dance. The `tool_use_id` is the load-bearing link: every `tool_use` in the assistant turn must have exactly one matching `tool_result` in the next user turn, or the API returns a 400. This is the tool-calling equivalent of Week 3's "you must close every bracket" — except the API enforces it loudly, which is a mercy.

> **The manual loop, in full.** Most real agents wrap this in a `while` loop: call the model, if `stop_reason == "tool_use"` run every requested tool and feed the results back, repeat until `stop_reason == "end_turn"`. That loop *is* the agent — which is exactly what Week 5 builds. This week we keep it to one or two turns so the contract stays visible.

---

## 3. `tool_choice` — may, must, which, or never

By default (`tool_choice` unset, equivalent to `{"type": "auto"}`) the model decides whether to call a tool. Four settings:

| `tool_choice` | Behavior | Use when |
|---|---|---|
| `{"type": "auto"}` (default) | Model decides whether to use a tool. | Almost always. Let the model judge. |
| `{"type": "any"}` | Model *must* call **some** tool (its pick). | You know a tool is needed but not which. |
| `{"type": "tool", "name": "extract"}` | Model *must* call **this** tool. | Forced structured extraction — the tool *is* the output shape. |
| `{"type": "none"}` | Model *cannot* call any tool. | You want a plain text turn even though tools are defined. |

Any of these can carry `"disable_parallel_tool_use": true` to force at most one tool per turn.

Forcing a tool (`type: "tool"`) is a legitimate pattern for structured extraction — "always call `record_contact` with the fields you found" — and we'll use it. But reaching for a forced tool *to make the model behave* is usually a smell: if the model won't call your tool on `auto`, the description or the task framing is wrong, and forcing it just papers over that. Fix the description first.

```python
# Forced extraction: the tool IS the output format.
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    tools=[contact_tool],
    tool_choice={"type": "tool", "name": "record_contact"},
    messages=[{"role": "user", "content": "Jane Doe, jane@co.com, Enterprise plan."}],
)
```

---

## 4. Parallel tool use

A single assistant turn can request **multiple** tools at once. Ask "what's the weather in Paris and Tokyo?" and a capable model emits two `tool_use` blocks in one turn. The rule that bites people: **you must return one `tool_result` for every `tool_use`, all in the next single user turn.** Drop one and the next request 400s.

```python
tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

results = []
for tu in tool_use_blocks:
    out = run_tool(tu.name, tu.input)   # your dispatcher
    results.append({
        "type": "tool_result",
        "tool_use_id": tu.id,
        "content": out,
    })

messages.append({"role": "assistant", "content": response.content})
messages.append({"role": "user", "content": results})   # ALL results, one turn
```

Two design notes. First, parallel-safe tools (read-only lookups) can run concurrently — a `ThreadPoolExecutor` over `run_tool` is fine and faster. Parallel-*unsafe* tools (anything with side effects — `send_email`, `git_push`) should not, which is one reason you sometimes set `disable_parallel_tool_use: true`. Second, the *order* of results in the list does not matter; the `tool_use_id` does the matching, not the position.

---

## 5. Cross-vendor calling — what's portable and what isn't

Every major vendor in 2026 supports tool/function calling, and the **JSON Schema is portable across all of them**. What differs is the envelope. Here is the same `get_weather` tool, three ways.

**Anthropic** (`tools` + `tool_use` blocks):

```python
tools = [{
    "name": "get_weather",
    "description": "...",
    "input_schema": {"type": "object", "properties": {...}, "required": ["city"]},
}]
# Response: content blocks, one of type "tool_use" with .input
```

**OpenAI** (`tools` with a `function` wrapper + `tool_calls`):

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "...",
        "parameters": {"type": "object", "properties": {...}, "required": ["city"]},
    },
}]
# Response: message.tool_calls[i].function.arguments  (a JSON *string* — you parse it)
```

**Ollama / Qwen** (OpenAI-shaped `tools`, native `/api/chat`):

```python
import ollama

response = ollama.chat(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "...",
            "parameters": {"type": "object", "properties": {...}, "required": ["city"]},
        },
    }],
)
# Response: response["message"]["tool_calls"][i]["function"]["arguments"]  (a dict in Ollama)
```

What's portable:

- **The `input_schema` / `parameters` JSON Schema.** Write it once; it works everywhere. This is the whole reason a single tool *registry* can drive multiple vendors (the mini-project leans on this).

What's **not** portable:

- **The envelope.** Anthropic nests `input_schema` directly; OpenAI and Ollama wrap the tool in `{"type": "function", "function": {...}}` and call the schema `parameters`.
- **Where the call lands and what type it is.** Anthropic: a `tool_use` content block with a parsed `.input` dict. OpenAI: `message.tool_calls`, with `arguments` as a JSON *string* you must `json.loads`. Ollama: `tool_calls` with `arguments` already a dict.
- **Result format.** Anthropic uses a `tool_result` content block keyed by `tool_use_id`. OpenAI uses a `role: "tool"` message keyed by `tool_call_id`. Same idea, different keys.
- **Reliability on small models.** A frontier model nails tool calls. A 7B local model is *good* but will occasionally emit malformed arguments, hallucinate a tool name, or wrap the JSON in prose. Your dispatcher must validate (next section) and your benchmark must measure the gap — that's the whole point of running both.

> **The portability lesson, stated once:** keep your tools as plain Python functions plus JSON Schemas in a vendor-neutral registry, and write a thin per-vendor *adapter* that translates the registry into that vendor's envelope and translates the vendor's call back into `(name, args)`. Two thin adapters, one tool surface. We build exactly this in the challenge.

---

## 6. Validate before you run — the half of the contract that's yours

The model fills in the schema, but it is *not guaranteed* to fill it in correctly — especially a local model, and especially once a retrieved document can influence the arguments (Week 17). So you validate the `input` against your schema **before** you dispatch:

```python
import jsonschema

SCHEMAS = {"get_weather": tools[0]["input_schema"]}

def run_tool(name: str, args: dict) -> str:
    schema = SCHEMAS.get(name)
    if schema is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as e:
        # Return the error to the model as a tool_result with is_error=True.
        return f"ERROR: invalid arguments for {name}: {e.message}"
    return DISPATCH[name](**args)
```

Two things to notice. First, an unknown tool name is a *normal* failure mode (a small model hallucinates `get_wether`), and you handle it gracefully, not with an unhandled `KeyError`. Second, when validation fails you send the error *back to the model* as a `tool_result` with `is_error: true`:

```python
{
    "type": "tool_result",
    "tool_use_id": tu.id,
    "content": "ERROR: 'city' is a required property",
    "is_error": True,
}
```

The model reads that, and a capable model corrects itself on the next turn — it re-emits the call with the missing field. This is the difference between a brittle agent that dies on the first malformed call and a robust one that self-heals. You will see this self-correction live in Exercise 1.

This validation step is *also* your first security boundary. JSON Schema can enforce `type`, `enum`, `required`, and (with `additionalProperties: false`) reject unexpected fields. It cannot enforce "this path is inside the sandbox" or "this URL isn't a private IP" — that's Lecture 2's semantic validation. But schema validation catches the structural garbage cheaply, before any semantic check runs.

---

## 7. `strict` tool use — making the schema un-violatable

By default, the model is *steered* toward your schema but can still, occasionally, produce an out-of-schema argument (an extra field, a wrong type, a missing required field). **Strict tool use** (`"strict": True` on the tool definition, on supporting models) constrains the model's decoding so it *cannot* emit anything that violates the schema:

```python
strict_tool = {
    "name": "book_flight",
    "description": "Book a flight.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "destination": {"type": "string"},
            "passengers": {"type": "integer", "enum": [1, 2, 3, 4]},
        },
        "required": ["destination", "passengers"],
        "additionalProperties": False,
    },
}
```

Strict mode requires `additionalProperties: false` and every property in `required` (the schema must be fully closed). The trade-off: the first request with a new strict schema pays a one-time compilation cost (then it's cached ~24h), and strict mode does not support every JSON Schema feature — no `minimum`/`maximum`, no `minLength`, no recursive schemas. For those constraints you validate in code (§6). Reach for strict when a malformed argument would be *expensive* — a financial transaction, a destructive action — and you want the model to be structurally incapable of producing one.

---

## 8. A worked example: the calculator tool, end to end

Let's put §§1–7 together on the simplest possible real tool — a calculator — because it surfaces every concept including the security one.

```python
import ast
import operator

import anthropic
import jsonschema

client = anthropic.Anthropic()

CALC_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "An arithmetic expression using + - * / ** () and numbers, e.g. '(1234 * 7) + 19'.",
        }
    },
    "required": ["expression"],
    "additionalProperties": False,
}

calculator_tool = {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression. Use this for any exact numeric computation.",
    "input_schema": CALC_SCHEMA,
}

# A SAFE evaluator. NEVER eval() model-chosen strings — that's arbitrary code execution.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")

def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")   # parse, do NOT eval()
    return str(_safe_eval(tree.body))

def run_calc(args: dict) -> str:
    jsonschema.validate(args, CALC_SCHEMA)
    try:
        return calculator(**args)
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"ERROR: {e}"

# --- the loop ---
messages = [{"role": "user", "content": "What is (1234 * 7) + 19?"}]
while True:
    resp = client.messages.create(
        model="claude-opus-4-8", max_tokens=1024,
        tools=[calculator_tool], messages=messages,
    )
    print(f">>> stop_reason={resp.stop_reason}")
    if resp.stop_reason != "tool_use":
        print(next(b.text for b in resp.content if b.type == "text"))
        break
    messages.append({"role": "assistant", "content": resp.content})
    results = []
    for tu in (b for b in resp.content if b.type == "tool_use"):
        print(f"    tool_use  {tu.name}({tu.input})")
        out = run_calc(tu.input)
        print(f">>> ran calculator -> {out!r}")
        results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
    messages.append({"role": "user", "content": results})
```

Run it and you get the connection-formed promise from the README:

```text
>>> stop_reason=tool_use
    tool_use  calculator({'expression': '(1234 * 7) + 19'})
>>> ran calculator -> '8657'
>>> stop_reason=end_turn
The answer is 8657.
```

The single most important line in that whole block is the comment on `_safe_eval`: **never `eval()` a model-chosen string.** A naive `def calculator(expression): return eval(expression)` is arbitrary remote-code execution — the model (or a prompt-injected document) can pass `__import__('os').system('rm -rf ~')`. The AST-walking evaluator only permits the four operators and numbers; anything else raises. That instinct — "the argument came from an untrusted client, so I parse and whitelist, I never `eval`" — is the entire security half of this week, previewed on the simplest tool. Lecture 2 generalizes it to files, the network, and a Python sandbox.

---

## 9. Recap

You should now be able to:

- Write a tool with a correct `input_schema` — typed properties with descriptions, honest `required`, `additionalProperties: false`.
- Trace the two-turn round trip: `tool_use` (with `id`, `name`, `input`) → run the tool → `tool_result` (with matching `tool_use_id`) → final turn.
- Handle parallel tool use: one `tool_result` per `tool_use`, all in one user turn.
- Use `tool_choice` deliberately — `auto` by default, `tool` for forced extraction, and know that forcing-to-fix is a smell.
- Run one JSON Schema against Claude, OpenAI, and Ollama/Qwen, and say what's portable (the schema) and what isn't (the envelope, the result type, the reliability).
- Validate the model's `input` against the schema before dispatch, return `is_error` results so the model self-corrects, and reach for `strict` when a bad argument would be expensive.
- Never `eval()` a model-chosen string.

Next up: how to make the model emit *structured data* (not just call tools) two different ways, what MCP is and why it's the cross-vendor tool layer, and how to defend every tool against arguments chosen by a hostile party. Continue to [Lecture 2 — Structured Output, MCP, and the Security Surface](./02-structured-output-mcp-and-the-security-surface.md).

---

## References

- *Tool use with Claude — overview* — Anthropic docs: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>
- *Implement tool use* — Anthropic docs: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use>
- *OpenAI function calling*: <https://platform.openai.com/docs/guides/function-calling>
- *Gemini function calling*: <https://ai.google.dev/gemini-api/docs/function-calling>
- *Ollama tool support*: <https://ollama.com/blog/tool-support>
- *Building effective agents* — Anthropic: <https://www.anthropic.com/research/building-effective-agents>
- *JSON Schema — understanding*: <https://json-schema.org/understanding-json-schema/>
