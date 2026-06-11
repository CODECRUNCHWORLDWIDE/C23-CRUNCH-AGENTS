# Exercise 1 — Design a Tool Schema

**Goal:** Hand-write three tool schemas, wire a validate-then-dispatch loop, watch a frontier model self-correct after you reject a malformed call, and run the exact same schemas against a local Qwen. You will train the single most important habit of the week: reading the `tool_use` request and your `tool_result` reply as a matched pair.

**Estimated time:** 50 minutes. Guided.

---

## Setup

```bash
pip install anthropic ollama jsonschema
export ANTHROPIC_API_KEY=sk-ant-...
ollama pull qwen2.5:7b-instruct      # if you haven't
ollama serve &                       # if it isn't already running
```

Verify both paths answer:

```bash
python3 -c "import anthropic; anthropic.Anthropic().messages.create(model='claude-haiku-4-5', max_tokens=16, messages=[{'role':'user','content':'hi'}]); print('claude ok')"
python3 -c "import ollama; print(ollama.chat(model='qwen2.5:7b-instruct', messages=[{'role':'user','content':'hi'}])['message']['content'][:20], '<- qwen ok')"
```

---

## Step 1 — Write three schemas by hand

Create `tools.py`. Define three tools as plain dicts. **Do not use a schema-generation library here** — the point is to feel the JSON Schema by hand. Get the descriptions right; they are the prompt.

```python
# tools.py
CALCULATOR = {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression. Use for any exact numeric computation the user asks for.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "An arithmetic expression with + - * / ** () and numbers, e.g. '(1234 * 7) + 19'.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
}

UNIT_CONVERT = {
    "name": "convert_units",
    "description": "Convert a value between two units of the same dimension (length, mass, or temperature).",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The numeric quantity to convert."},
            "from_unit": {"type": "string", "enum": ["m", "ft", "kg", "lb", "C", "F"],
                          "description": "The source unit."},
            "to_unit": {"type": "string", "enum": ["m", "ft", "kg", "lb", "C", "F"],
                        "description": "The target unit. Must share a dimension with from_unit."},
        },
        "required": ["value", "from_unit", "to_unit"],
        "additionalProperties": False,
    },
}

LOOKUP_CAPITAL = {
    "name": "lookup_capital",
    "description": "Return the capital city of a country. Use this instead of answering from memory when asked for a capital.",
    "input_schema": {
        "type": "object",
        "properties": {
            "country": {"type": "string", "description": "The country name in English, e.g. 'France'."}
        },
        "required": ["country"],
        "additionalProperties": False,
    },
}

ALL_TOOLS = [CALCULATOR, UNIT_CONVERT, LOOKUP_CAPITAL]
```

Now implement the three functions and a **validate-then-dispatch** wrapper:

```python
# tools.py (continued)
import ast, operator, jsonschema

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")

def calculator(expression: str) -> str:
    return str(_safe_eval(ast.parse(expression, mode="eval").body))

_FACTORS = {("m", "ft"): 3.28084, ("ft", "m"): 0.3048, ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592}
def convert_units(value, from_unit, to_unit) -> str:
    if from_unit == to_unit:
        return str(value)
    if (from_unit, to_unit) == ("C", "F"): return str(value * 9 / 5 + 32)
    if (from_unit, to_unit) == ("F", "C"): return str((value - 32) * 5 / 9)
    f = _FACTORS.get((from_unit, to_unit))
    if f is None:
        raise ValueError(f"no conversion {from_unit}->{to_unit}")
    return str(value * f)

_CAPITALS = {"france": "Paris", "japan": "Tokyo", "brazil": "Brasília", "kenya": "Nairobi"}
def lookup_capital(country: str) -> str:
    c = _CAPITALS.get(country.strip().lower())
    return c if c else f"ERROR: unknown country '{country}'"

DISPATCH = {"calculator": calculator, "convert_units": convert_units, "lookup_capital": lookup_capital}
SCHEMAS = {t["name"]: t["input_schema"] for t in ALL_TOOLS}

def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Returns (result_text, is_error)."""
    if name not in DISPATCH:
        return f"ERROR: unknown tool '{name}'", True
    try:
        jsonschema.validate(args, SCHEMAS[name])
    except jsonschema.ValidationError as e:
        return f"ERROR: invalid arguments: {e.message}", True
    try:
        return DISPATCH[name](**args), False
    except Exception as e:
        return f"ERROR: {e}", True
```

---

## Step 2 — The Claude loop

Create `run_claude.py`:

```python
import anthropic
from tools import ALL_TOOLS, run_tool

client = anthropic.Anthropic()

def ask(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    while True:
        resp = client.messages.create(
            model="claude-haiku-4-5", max_tokens=1024, tools=ALL_TOOLS, messages=messages,
        )
        print(f">>> stop_reason={resp.stop_reason}")
        if resp.stop_reason != "tool_use":
            return next(b.text for b in resp.content if b.type == "text")
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in (b for b in resp.content if b.type == "tool_use"):
            print(f"    tool_use  {tu.name}({tu.input})")
            out, is_err = run_tool(tu.name, tu.input)
            print(f">>> ran {tu.name} -> {out!r} (is_error={is_err})")
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": out, "is_error": is_err})
        messages.append({"role": "user", "content": results})

if __name__ == "__main__":
    for q in ["What is (1234 * 7) + 19?",
              "How many feet is 100 meters?",
              "What's the capital of Kenya?"]:
        print(f"\n=== {q} ===")
        print(ask(q))
```

Run it. You should see clean two-turn traces, each ending in `end_turn`.

---

## Step 3 — Force a self-correction

Add a question the model is likely to fumble on a small schema, and **temporarily break your validation to be stricter** so you can watch the recovery. Add this question:

```python
"Convert 50 from Celsius to Fahrenheit and also tell me the capital of France."
```

This needs *parallel* tool use (a convert and a lookup in one turn). Confirm you see **two** `tool_use` blocks in one turn and you return **two** `tool_result`s. Then, to see self-correction, edit `convert_units`'s schema to *also* require a non-existent `"precision"` field, re-run, and watch: the model's first call fails validation, you return `is_error=True` with the message, and on the next turn the model... still can't satisfy an impossible schema (that's the lesson — an impossible schema is *your* bug). Now make `precision` optional instead and confirm the call succeeds. Document what you saw.

---

## Step 4 — The same schemas against Qwen

Create `run_qwen.py`. The schema is portable; the envelope is not — Ollama wants the OpenAI shape:

```python
import json, ollama
from tools import ALL_TOOLS, run_tool

def to_ollama(t):
    return {"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}

OLLAMA_TOOLS = [to_ollama(t) for t in ALL_TOOLS]

def ask(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(6):   # step budget — a small model can loop
        resp = ollama.chat(model="qwen2.5:7b-instruct", messages=messages, tools=OLLAMA_TOOLS)
        msg = resp["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg["content"]
        messages.append(msg)
        for c in calls:
            args = c["function"]["arguments"]
            if isinstance(args, str):   # some builds return a string
                args = json.loads(args)
            out, is_err = run_tool(c["function"]["name"], args)
            print(f"    qwen tool_use {c['function']['name']}({args}) -> {out!r}")
            messages.append({"role": "tool", "content": out})
    return "ERROR: step budget exceeded"

if __name__ == "__main__":
    for q in ["What is (1234 * 7) + 19?", "How many feet is 100 meters?",
              "What's the capital of Kenya?"]:
        print(f"\n=== {q} ===\n{ask(q)}")
```

Run it. Note where Qwen differs: it may wrap JSON in prose, miss a parallel call, or get a unit conversion arithmetically wrong. **Write down one concrete difference** between Claude's and Qwen's behavior on the same task.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] Your three schemas validate (run `jsonschema.Draft202012Validator.check_schema(s)` on each).
- [ ] `run_claude.py` produces clean two-turn traces ending in `end_turn` for all three base questions.
- [ ] The parallel question produces **two** `tool_use` blocks and **two** `tool_result`s in matched turns.
- [ ] You triggered an `is_error=True` result and can explain, in one sentence, why an *impossible* schema is your bug, not the model's.
- [ ] `run_qwen.py` runs against the same schemas with only an envelope adapter, and you've documented one concrete Claude-vs-Qwen behavioral difference.

---

## Stretch

- Add `"strict": True` to the `calculator` tool (drop it into the Anthropic call) and try to make Claude emit an out-of-schema argument. Can you? Document the result.
- Add a deliberately *vague* fourth tool (`"name": "helper", "description": "does helpful things"`) and confirm the model under-calls or mis-calls it. Fix the description and re-measure. This is the §5 decision tree's top branch, demonstrated.
- Run the Qwen path with `qwen2.5:3b-instruct` instead of 7B and note how tool-call reliability degrades with size — preview of the Week 6 quantization/size trade-offs.

---

When this feels comfortable, move to [Exercise 2 — Structured extraction three ways](exercise-02-structured-extraction.py).
