# Challenge 1 — The Cross-Vendor Tool Bridge

**Time estimate:** ~90 minutes.

## Problem statement

You are the platform engineer for a team that wants to run "easy" agent tasks on a cheap local model and "hard" ones on a frontier model — over the *same* tool surface (this is literally the Week 21 routing lab and the capstone's serving story). The non-negotiable: a tool is defined **once**. Its name, its JSON Schema, its Python implementation, and its validation live in one place. No tool logic is duplicated per vendor. The only per-vendor code is a thin *adapter* that translates the registry into that vendor's envelope and translates the vendor's tool call back into a plain `(name, args)`.

You will build that registry and two adapters, then run a 20-task benchmark against both `claude-opus-4-8` (or `claude-haiku-4-5` to save cost) and a local `qwen2.5:7b-instruct`, and report the tool-call accuracy gap with a number.

This mirrors the real skill: the model vendor is the *most volatile* part of the stack (the README says so explicitly). Keep your tools vendor-neutral and you can swap the model in a weekend. Couple them and you can't.

## The shape you must build

```
toolkit/
├── registry.py        # ONE definition per tool: name, schema, impl, validation
├── adapters/
│   ├── anthropic.py    # registry -> Anthropic `tools`; tool_use -> (name, args)
│   └── ollama.py       # registry -> Ollama `tools`;   tool_calls -> (name, args)
├── agent.py            # a thin run loop that takes an adapter + a question
└── benchmark.py        # runs 20 tasks through each adapter, prints accuracy
```

### `registry.py` — the single source of truth

A `Tool` carries everything: name, description, JSON Schema, the callable, and uses your validate-then-dispatch wrapper from Exercise 1. Sketch:

```python
from dataclasses import dataclass
from typing import Callable
import jsonschema

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    impl: Callable[..., str]

    def run(self, args: dict) -> tuple[str, bool]:
        try:
            jsonschema.validate(args, self.input_schema)
        except jsonschema.ValidationError as e:
            return f"ERROR: invalid arguments: {e.message}", True
        try:
            return self.impl(**args), False
        except Exception as e:
            return f"ERROR: {e}", True

# REGISTRY = {tool.name: tool for tool in [calculator, convert_units, lookup_capital, read_file]}
```

Reuse the four tools you've already written (calculator, convert_units, lookup_capital from Exercise 1, and the **hardened** read_file from Exercise 3 — sandboxed). No naked `eval`, no naked `open`.

### The adapters — the only per-vendor code

`adapters/anthropic.py` exposes two functions:

```python
def to_anthropic_tools(registry) -> list[dict]:
    """Each Tool -> {'name', 'description', 'input_schema'}."""

def parse_tool_calls(response) -> list[tuple[str, str, dict]]:
    """Anthropic response -> list of (tool_use_id, name, args)."""
```

`adapters/ollama.py` exposes the same two functions with the OpenAI envelope (`{'type':'function','function':{...,'parameters': schema}}`) and parsing from `message['tool_calls']` (remember: `arguments` may be a dict or a JSON string).

### `agent.py` — vendor-agnostic loop

The loop takes an adapter and a question, calls the model, runs every requested tool through `Tool.run`, feeds results back, and stops on `end_turn` (Anthropic) / no-tool-calls (Ollama). **Enforce a step budget** (max 6 turns) — a small model can loop forever, and the agent loop in Week 5 makes budgets a first-class concern. The loop body must contain **no tool names** — it dispatches purely through the registry.

### `benchmark.py` — the number

Write 20 tasks, each with a deterministic expected answer. Mix: arithmetic (calculator), unit conversion, capital lookup, and file-read. A task passes if the **final answer contains the expected value**. Run all 20 through the Anthropic adapter and all 20 through the Ollama adapter. Print a table.

## Acceptance criteria

- [ ] `grep -rn "calculator\|convert_units\|lookup_capital\|read_file" toolkit/agent.py adapters/` finds tool names **only** in `registry.py` and `benchmark.py` (the tasks), **never** in `agent.py` or inside the adapter loop bodies. The loop is vendor- and tool-agnostic.
- [ ] Both adapters expose the identical two-function interface; `agent.py` imports an adapter by name and is otherwise unchanged between vendors.
- [ ] `benchmark.py` runs all 20 tasks against both models and prints a table:
  ```
  TOOL-CALL ACCURACY (20 tasks)
  model                  passed  accuracy  avg_turns  avg_latency
  claude-opus-4-8        20/20   100.0%    2.0        0.9s
  qwen2.5:7b-instruct    16/20    80.0%    2.4        0.3s
  ```
- [ ] The hardened `read_file` is the one from Exercise 3 (sandboxed); a benchmark task that asks to read outside the sandbox must come back refused, and you count that refusal as a *pass* (correct behavior).
- [ ] A `README.md` with the table, the gap analysis (which task classes did the local model miss, and why), and one sentence on what you'd change to close the gap (better descriptions? strict mode? a bigger model?).
- [ ] Committed to your Week 4 repo under `challenges/challenge-01/`.

## The trap (read after a first attempt)

The subtle bug everyone hits: the Ollama `arguments` field is sometimes a **dict** and sometimes a **JSON string**, depending on the build and the model. If you assume one, half your local tool calls silently fail validation (a string isn't an object) and your local accuracy looks artificially terrible. The fix is in the *adapter*, not the registry — `json.loads(args) if isinstance(args, str) else args` — which is exactly why the per-vendor quirk belongs in the per-vendor adapter. If you find yourself patching the registry to handle a vendor's wire quirk, you've put the seam in the wrong place.

## Stretch

- Add a **third** adapter for the OpenAI API (`tools` / `tool_calls` with `arguments` as a JSON string, results as `role: "tool"` messages). Confirm `agent.py` and `registry.py` don't change at all. Three vendors, one tool surface — that's the whole thesis.
- Turn on `"strict": True` in the Anthropic adapter and re-run the benchmark. Does frontier accuracy change? Does latency? (First strict request pays a compile cost.)
- Add a **cost** column: tokens-in × input-price + tokens-out × output-price for the vendor, $0 for local. Now your table is the Week 21 routing decision in miniature — when is the frontier model worth it?

## Why this matters

In Week 5 you hand-roll an agent loop that imports a tool registry. In Week 13 you re-implement that loop as a LangGraph graph. In Week 21 you route between a local 7B and a frontier model by difficulty. In the capstone you ship all of it. **Every one of those weeks assumes your tools are vendor-neutral.** This challenge builds that assumption once, in 90 minutes, so the rest of the course is swapping models, not rewriting tools. The engineer who can say "the tool surface doesn't care which model is behind it" is the one who survives the next framework churn — which, per the README, is the entire point of the course.
