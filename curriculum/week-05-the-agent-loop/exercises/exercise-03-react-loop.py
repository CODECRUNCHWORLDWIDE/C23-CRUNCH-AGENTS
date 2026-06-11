#!/usr/bin/env python3
# Exercise 3 — A complete ReAct loop (Claude and Qwen, same tool schema)
#
# Goal: A full, ~150-line ReAct agent you can read top to bottom. It runs against
#       claude-opus-4-8 (Anthropic SDK) OR qwen2.5:7b-instruct (Ollama), using the
#       SAME tool schema, with the four budgets from Lecture 2 and a legible trace.
#       This is the spine of the week's mini-project.
#
# Estimated time: 50 minutes. Runnable.
#
# WHAT IT DEMONSTRATES
#
#   * The loop is the same underneath for both providers (Lecture 1 §5).
#   * What is PORTABLE: the JSON-Schema tool definition.
#   * What is NOT: tool_use blocks vs tool_calls, tool_result vs role:"tool".
#   * Budgets make it terminate; the trace makes it legible.
#
# HOW TO USE THIS FILE
#
#   pip install anthropic openai
#   export ANTHROPIC_API_KEY=sk-ant-...
#
#   # Claude path (default):
#   python3 exercise-03-react-loop.py
#
#   # Local path (needs Ollama serving qwen2.5:7b-instruct):
#   python3 exercise-03-react-loop.py --provider qwen
#
#   This file ships a tiny self-contained tool registry (calculator + word_count)
#   so it runs WITHOUT your Week 4 package. In the mini-project you swap in your
#   real crunch_tools registry; the loop does not change.
#
# ACCEPTANCE CRITERIA
#
#   [ ] Both providers solve the calculator task and terminate on end_turn.
#   [ ] The trace shows reason/act/observe/final lines and a termination summary.
#   [ ] Switching --provider changes ONLY which adapter runs, not the loop logic.
#   [ ] You can point at the two places the providers differ (block shape, result
#       envelope) and the one place they agree (the JSON Schema).
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

# --- A tiny self-contained tool registry ------------------------------------
# In the mini-project: from crunch_tools.registry import REGISTRY


def _calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Week 4 hardens the eval; short here."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return f"Error: expression has characters outside {sorted(allowed)}."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating {expression!r}: {exc}"


def _word_count(text: str) -> str:
    """Count words in a piece of text."""
    return str(len(text.split()))


REGISTRY = {
    "calculator": {
        "fn": _calculator,
        "schema": {
            "name": "calculator",
            "description": (
                "Evaluate a basic arithmetic expression. Call this whenever the "
                "answer depends on a precise calculation you should not do in your head."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. (1234 * 7) + 19"}
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    "word_count": {
        "fn": _word_count,
        "schema": {
            "name": "word_count",
            "description": "Count the number of words in a piece of text.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
}

SYSTEM = (
    "You are a problem-solving agent with tools. Work step by step. Call a tool when "
    "the answer depends on a precise computation or a fact you cannot produce reliably "
    "yourself. After each tool result, decide whether you can answer or need another "
    "tool. Answer concisely. Do not call a tool you do not need."
)

PRICE_IN = 5.00 / 1_000_000
PRICE_OUT = 25.00 / 1_000_000


@dataclass
class Budgets:
    max_steps: int = 8
    max_tokens: int = 20_000
    max_seconds: float = 60.0
    max_dollars: float = 0.20
    steps: int = 0
    tokens: int = 0
    dollars: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def record(self, in_tok: int, out_tok: int) -> None:
        self.tokens += in_tok + out_tok
        self.dollars += in_tok * PRICE_IN + out_tok * PRICE_OUT

    def exceeded(self) -> str | None:
        if self.steps >= self.max_steps:
            return "step budget"
        if self.tokens >= self.max_tokens:
            return "token budget"
        if time.monotonic() - self.started >= self.max_seconds:
            return "time budget"
        if self.dollars >= self.max_dollars:
            return "cost budget"
        return None

    def summary(self, reason: str) -> str:
        el = time.monotonic() - self.started
        return (
            f"--- terminated: {reason} | steps={self.steps}/{self.max_steps} "
            f"tokens={self.tokens}/{self.max_tokens} time={el:.1f}s/{self.max_seconds:.0f}s "
            f"cost=${self.dollars:.4f}/${self.max_dollars:.2f}"
        )


def log(step: int, kind: str, content: str) -> None:
    print(f"step {step:<2} {kind:<7} {content[:90]}")


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    tool = REGISTRY.get(name)
    if tool is None:
        return (f"Error: no tool named {name!r}. Available: {list(REGISTRY)}", True)
    try:
        return (tool["fn"](**args), False)
    except Exception as exc:  # noqa: BLE001
        return (f"Error running {name}: {exc}", True)


# ============================================================================
# CLAUDE PATH — Anthropic SDK. tool_use blocks, tool_result blocks.
# ============================================================================
def agent_claude(task: str, b: Budgets) -> str:
    import anthropic

    client = anthropic.Anthropic()
    tools = [t["schema"] for t in REGISTRY.values()]
    messages = [{"role": "user", "content": task}]

    while True:
        breached = b.exceeded()
        if breached:
            print(b.summary(breached + " exceeded"))
            return "Stopped: budget exceeded."
        resp = client.messages.create(
            model="claude-opus-4-8", max_tokens=2048, system=SYSTEM,
            tools=tools, messages=messages,
        )
        b.steps += 1
        b.record(resp.usage.input_tokens, resp.usage.output_tokens)
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            final = next((blk.text for blk in resp.content if blk.type == "text"), "")
            log(b.steps, "final", final)
            print(b.summary("end_turn"))
            return final

        results = []
        for blk in resp.content:
            if blk.type != "tool_use":
                continue
            log(b.steps, "act", f"{blk.name}({dict(blk.input)})")
            out, is_err = run_tool(blk.name, dict(blk.input))
            log(b.steps, "observe", out)
            results.append({
                "type": "tool_result", "tool_use_id": blk.id,
                "content": out, "is_error": is_err,
            })
        messages.append({"role": "user", "content": results})


# ============================================================================
# QWEN PATH — Ollama OpenAI-compatible API. tool_calls, role:"tool" messages.
# ============================================================================
def agent_qwen(task: str, b: Budgets) -> str:
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    tools = [
        {"type": "function", "function": {
            "name": t["schema"]["name"],
            "description": t["schema"]["description"],
            "parameters": t["schema"]["input_schema"],  # SAME JSON Schema
        }}
        for t in REGISTRY.values()
    ]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]

    while True:
        breached = b.exceeded()
        if breached:
            print(b.summary(breached + " exceeded"))
            return "Stopped: budget exceeded."
        resp = client.chat.completions.create(
            model="qwen2.5:7b-instruct", messages=messages, tools=tools,
        )
        b.steps += 1
        usage = resp.usage
        b.record(usage.prompt_tokens, usage.completion_tokens)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            log(b.steps, "final", msg.content or "")
            print(b.summary("end_turn"))
            return msg.content or ""

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            log(b.steps, "act", f"{call.function.name}({args})")
            out, _ = run_tool(call.function.name, args)
            log(b.steps, "observe", out)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})


def main() -> None:
    parser = argparse.ArgumentParser(description="A ReAct loop, two providers.")
    parser.add_argument("--provider", choices=["claude", "qwen"], default="claude")
    args = parser.parse_args()

    task = (
        "What is (1234 * 7) + 19? Then count the words in the sentence "
        "'the agent loop is just a while loop with budgets' and add it to that number."
    )
    print(f"provider={args.provider}\ntask: {task}\n")
    b = Budgets()
    answer = (agent_claude if args.provider == "claude" else agent_qwen)(task, b)
    print(f"\nANSWER: {answer}")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Expected output (--provider claude; numbers vary)
# -----------------------------------------------------------------------------
#
# provider=claude
# task: What is (1234 * 7) + 19? Then count the words ...
#
# step 1  act     calculator({'expression': '(1234 * 7) + 19'})
# step 1  observe 8657
# step 2  act     word_count({'text': 'the agent loop is just a while loop with budgets'})
# step 2  observe 10
# step 3  act     calculator({'expression': '8657 + 10'})
# step 3  observe 8667
# step 4  final   The result is 8667.
# --- terminated: end_turn | steps=4/8 tokens=2841/20000 time=3.2s/60s cost=$0.012/$0.20
#
# ANSWER: The result is 8667.
#
# The Qwen path produces the SAME shape of trace and the same answer when the 7B
# behaves; it fails more often (hallucinated names, answering without acting). The
# loop is identical — only the adapter (tool_use vs tool_calls) differs. Exact
# token counts/timings vary; the SHAPE is invariant.
# -----------------------------------------------------------------------------
