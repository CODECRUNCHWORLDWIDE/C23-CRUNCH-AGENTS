# Lecture 2 — Budgets, Failure Modes, and the SDKs

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can enforce four budgets (step, token, time, cost) so an agent always terminates, diagnose every common agent failure mode from its trace, and compare your hand-rolled loop against the `claude-agent-sdk` honestly — pass rate, cost, code surface, and what the SDK hides.

Lecture 1 built the loop. This lecture makes it safe to run unsupervised, teaches you to recognize the ways it fails, and then hands you the SDK that wraps it — so you can decide, with a number in hand, when to keep your loop and when to delete it.

The thesis of the whole week, restated for this lecture:

> **Most agent failures are not model failures. They are loop failures, budget failures, or tool failures. A framework hides all three. You built the loop by hand so that when the box misbehaves, you can open it and read the trace instead of filing an issue.**

---

## 1. Why one budget is not enough

The §2 agent in Lecture 1 had a step budget. That stops one runaway — the agent that takes too many turns. It does not stop the others. Each budget catches a *different* class of runaway, and a production agent needs all four:

| Budget | Catches | The runaway it stops |
|---|---|---|
| **Step** | turns | The agent that ping-pongs forever, calling tools without converging. |
| **Token** | cumulative tokens (in + out) | The agent whose context balloons — long tool outputs, growing history — until each call is enormous and expensive even at low step counts. |
| **Time** | wall-clock | The agent stuck on a slow tool (a hung web fetch, a sandbox that won't return) where steps and tokens look fine but the run never ends. |
| **Cost** | dollars (tokens × price) | The agent that is *technically* within step and token limits per call but, across a long run on an expensive model, quietly burns $40 overnight. |

Note they are not redundant. A 3-step run can blow the token budget if one tool returns a 200 KB document. A run that respects steps and tokens can blow the time budget if a tool hangs. A run that respects steps, tokens, and time on a cheap model can still blow the cost budget on an expensive one. **You need all four because each stops a runaway the others miss.**

---

## 2. Budgets as first-class loop state

Here is the Lecture 1 loop with all four budgets wired in. The structure is the same; the new code is a small `Budgets` object and four checks at the top of the loop.

```python
"""ReAct loop with four budgets. Each budget stops a different runaway.

Budgets are loop state, checked at the top of every iteration. The agent
ALWAYS terminates: either the model says end_turn, or a budget fires.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic

MODEL = "claude-opus-4-8"

# claude-opus-4-8 pricing, USD per token (see resources.md / Anthropic pricing).
PRICE_IN = 5.00 / 1_000_000   # $5.00 / 1M input tokens
PRICE_OUT = 25.00 / 1_000_000  # $25.00 / 1M output tokens


@dataclass
class Budgets:
    max_steps: int = 8
    max_tokens: int = 20_000      # cumulative input + output across the run
    max_seconds: float = 30.0
    max_dollars: float = 0.10

    # live counters
    steps: int = 0
    tokens: int = 0
    dollars: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def record_usage(self, usage) -> None:
        """Accumulate tokens and dollars from one response's usage block."""
        self.tokens += usage.input_tokens + usage.output_tokens
        self.dollars += usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT

    def exceeded(self) -> str | None:
        """Return the name of the first exceeded budget, or None."""
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
        elapsed = time.monotonic() - self.started
        return (
            f"--- terminated: {reason} | steps={self.steps}/{self.max_steps} "
            f"tokens={self.tokens}/{self.max_tokens} "
            f"time={elapsed:.1f}s/{self.max_seconds:.0f}s "
            f"cost=${self.dollars:.4f}/${self.max_dollars:.2f}"
        )


def agent(task: str, system: str, tool_schemas: list[dict], run_tool, b: Budgets) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": task}]

    while True:
        # Check budgets BEFORE the (costly) model call.
        breached = b.exceeded()
        if breached:
            print(b.summary(breached + " exceeded"))
            return "Stopped: budget exceeded before reaching an answer."

        response = client.messages.create(
            model=MODEL, max_tokens=4096, system=system,
            tools=tool_schemas, messages=messages,
        )
        b.steps += 1
        b.record_usage(response.usage)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final = next((blk.text for blk in response.content if blk.type == "text"), "")
            print(b.summary("end_turn"))
            return final

        results = []
        for blk in response.content:
            if blk.type != "tool_use":
                continue
            out, is_error = run_tool(blk.name, dict(blk.input))
            results.append({
                "type": "tool_result", "tool_use_id": blk.id,
                "content": out, "is_error": is_error,
            })
        messages.append({"role": "user", "content": results})
```

The discipline:

- **Check budgets at the top of the loop, before the model call.** The model call is the expensive part; you want to refuse *before* spending, not after.
- **Record usage from `response.usage` every turn.** The Anthropic SDK hands you `input_tokens` and `output_tokens` on every response — that is your ground truth for the token and cost budgets. Do not estimate with a third-party tokenizer; the real count is right there. (For *pre-flight* estimates use `client.messages.count_tokens`, never `tiktoken` — that is OpenAI's tokenizer and it miscounts Claude.)
- **The termination summary always prints.** Whether the agent finished cleanly or hit a budget, you get the one-line summary. That line is the "the agent terminated cleanly" promise from the week README. A run with *no* summary line is a run whose loop has no budget and will eventually hang.

> **Where each check goes is a design choice.** We check all four at the top. You could also check the time budget *inside* a slow tool call (so a single hung tool cannot blow past the wall-clock limit) — that is a refinement Exercise 2 explores. The principle is invariant: every path out of the loop is either `end_turn` or a named budget. There is no third exit.

---

## 3. The failure-mode catalog

You will see these over and over. Learn to name each from the trace, because naming it tells you the fix.

### 3.1 The infinite tool-call loop

The canonical failure. The agent calls tools forever without converging on an answer. In the trace, the step count climbs and climbs and the `final` line never comes. **Cause:** usually a task the model cannot actually solve with the tools it has, or a tool whose output never gives it what it needs. **What saves you:** the step (or token, or time, or cost) budget — the agent hits the ceiling and stops. The budget firing is not a failure of your system; it is your system *working*. The fix for the underlying problem is upstream: a better tool, a clearer task, or accepting that this task is out of scope.

### 3.2 Re-calling a failing tool forever

A sharper version: the same `act` line repeats with the *same arguments* and the *same failing `observe`*. The model gets an error, does not learn from it, and tries the identical thing again. **Cause:** the error message in your `tool_result` was unhelpful ("Error") so the model has nothing to adjust on. **Fix:** make tool errors *actionable* — say *what* was wrong and *what valid input looks like*. "Error: 'date' must be ISO-8601 (YYYY-MM-DD); got '3/15'." gives the model something to fix. Compare to Lecture 1's hallucinated-name handling, which names the valid tools — same principle.

### 3.3 The hallucinated tool name

The model emits a `tool_use` for a tool that does not exist (`web_search` when you only registered `web_fetch`). **Cause:** vague or overlapping tool descriptions, or a model (often a smaller local one) that pattern-matched to a tool it has seen elsewhere. **Fix:** the error-result recovery from Lecture 1 (return an error naming the valid tools), plus tighter, more distinct tool descriptions. Smaller models do this more — it is a prime place the 7B/frontier gap shows up in the mini-project benchmark.

### 3.4 Answering without acting

The model produces a final answer without calling the tool it should have — guessing the arithmetic instead of using the calculator, asserting a fact instead of fetching it. **Cause:** under-eager tool use, often from a too-soft system prompt or (on `claude-opus-4-8`) the model's tendency to reach for tools more conservatively than older models. **Fix:** make the tool-use trigger explicit *in the tool description and the system prompt* — "Call this when the answer depends on a current fact or a precise computation." This is the inverse of over-triggering; both are prompt-tuning problems, calibrated by reading the trace.

### 3.5 Stuck on the same sub-goal

The agent makes progress, then circles one sub-goal — re-deriving a fact it already established, re-reading a file it already read, re-litigating a decision. **Cause:** the agent is not carrying forward what it learned, often because a long context pushed the relevant observation out of effective attention, or because nothing in the loop summarizes progress. **Fix:** at minimum, the budgets bound it. Better: a progress note or a summary step (a Week 11 memory topic). For this week, recognizing it in the trace is the deliverable.

### 3.6 The "looks done but isn't" finish

The agent returns `end_turn` with a confident answer that is wrong — it stopped too early, satisfied with an incomplete result. **Cause:** the model judged itself done when it was not. **Fix:** this is exactly where a *reflection pass* (Lecture 1 §4.2) can earn its tokens — a critique step that checks the draft against the task before accepting it. Measure whether it helps on your benchmark.

> **The decision tree for "my agent misbehaved":** Did it terminate? If no → your loop has no budget; add one. If it terminated on a budget → which budget, and is the underlying task solvable with these tools (§3.1)? If it terminated on `end_turn` but the answer is wrong → read the trace for which failure mode (§3.2–3.6) and apply that fix. Always: read the trace first. The trace tells you which of these you have.

---

## 4. The SDK survey

You built the loop. Now meet the libraries that build it for you, and learn to compare them honestly.

### 4.1 The Anthropic `claude-agent-sdk` and the in-SDK tool runner

There are two Claude-native ways to not-write-the-loop:

**The in-SDK tool runner** (beta) lives in the regular `anthropic` package. You decorate tools with `@beta_tool` and the runner drives the loop:

```python
import anthropic
from anthropic import beta_tool

client = anthropic.Anthropic()


@beta_tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: A safe arithmetic expression like "(1234 * 7) + 19".
    """
    import ast, operator

    # A real implementation sanitizes; see Week 4. Shown short here.
    return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 (Week 4 hardens this)


runner = client.beta.messages.tool_runner(
    model="claude-opus-4-8",
    max_tokens=4096,
    tools=[calculator],
    messages=[{"role": "user", "content": "What is (1234 * 7) + 19?"}],
)
for message in runner:        # each iteration is one assistant turn
    print(message.stop_reason)
```

The runner does everything your loop did: calls the API, sees `tool_use`, runs the decorated function, feeds the `tool_result` back, repeats until `end_turn`. The schema is generated from your function signature and docstring — no hand-written JSON Schema. That is genuinely less code.

**The `claude-agent-sdk`** is the higher-level, Claude-native agent runtime — sessions, a managed loop, context management, built-in tool surfaces. It is the "I want Anthropic to run a real agent for me" path. At survey level for this week: you should know it exists, know it wraps the same loop you built, and know that the further up the abstraction you go, the more it does for you and the more it hides.

### 4.2 The other vendors' SDKs (place them, don't master them)

- **OpenAI Agents SDK** — OpenAI-native agent loop with handoffs between agents; the OpenAI ecosystem's answer to the same problem.
- **AWS Strands Agents** — AWS-native agent surface, leans on the AWS ecosystem (Bedrock, etc.).
- **Google ADK (Agent Development Kit)** — Google-native agent surface, leans on the Google/Gemini ecosystem.

The week's thesis about these: **the loop is the same underneath; they differ in ergonomics, observability, and how much they hide.** You read these to *place* them — to be the engineer who, asked "should we use Strands or roll our own?", can answer with the trade-offs rather than a guess. You do not need to master four SDKs. You need to have built the loop once by hand and to understand that each SDK is a wrapper around it with a different ecosystem attached.

---

## 5. Hand-rolled vs SDK: the honest comparison

The mini-project and the week's challenge both ask you to implement your agent *both* ways — hand-rolled and via the SDK runner — and report four numbers. Here is how to think about each:

| Dimension | What to measure | What you usually find |
|---|---|---|
| **Pass rate** | Fraction of the 25-task benchmark solved correctly, same tasks both ways. | Roughly equal — same model, same tools, same loop underneath. If they differ a lot, suspect a bug in one harness, not a model difference. |
| **Cost** | Total dollars across the benchmark (sum `usage` × price). | Close, with the SDK sometimes slightly higher if it adds scaffolding tokens (extra system text, retries). Measure, don't assume. |
| **Code surface** | Lines of code you wrote and maintain. | The SDK wins clearly — no loop, no schema plumbing, no result-envelope handling. This is the SDK's real selling point. |
| **What it hides** | What you can no longer see or control. | The interleaving, the exact retry behavior, where budgets are enforced, the precise trace format. You traded control for brevity. |

The senior judgment: **the SDK is the right default for production once you understand the loop, because the code surface win is real and durable.** You hand-roll when you need fine-grained control the SDK does not expose — a custom budget check inside a tool call, a human-in-the-loop approval gate, a bespoke trace format your observability stack demands, or a research experiment on the loop itself. The reason this course makes you hand-roll first is *not* that hand-rolling is better; it is that you cannot make that judgment, or debug the SDK when it misbehaves, until you have felt the thing it wraps. By Week 13 you graduate the loop to a LangGraph *graph*; this week you earn the right to use either by building neither-from-a-library first.

---

## 6. Token and cost accounting, precisely

Because the cost budget is only as good as your accounting, two rules from Week 4 carry forward and matter here:

- **Use `response.usage` for actuals.** Every response carries `input_tokens` and `output_tokens`. Sum them across the run; multiply by the per-token price for the model. That is your real spend, not an estimate.
- **Use `client.messages.count_tokens` for pre-flight estimates.** When you want to predict a call's input cost *before* sending it (to refuse a call that would blow the budget), count the prompt with the model's own tokenizer via the API. Never reach for `tiktoken` — it is OpenAI's tokenizer and undercounts Claude by 15–20% on typical text, more on code.

For `claude-opus-4-8` the rates are $5.00 per million input tokens and $25.00 per million output tokens (the numbers in the `Budgets` class above). For the local Qwen path through Ollama the dollar cost is effectively zero (you pay in your own electricity and GPU time, not per token) — which is half the point of the local path and a recurring theme of the course.

---

## 7. Recap

You should now be able to:

- Explain why an agent needs four budgets, not one, and what runaway each of step / token / time / cost catches that the others miss.
- Wire all four budgets into the loop as first-class state, checked before each model call, with a termination summary that always prints.
- Name every common failure mode — infinite tool-call loop, re-calling a failing tool, hallucinated tool name, answering-without-acting, stuck-on-a-sub-goal, looks-done-but-isn't — from the trace, and state the fix for each.
- Use the `claude-agent-sdk` / in-SDK tool runner to run the loop without writing it, and place the OpenAI, AWS Strands, and Google ADK SDKs as ecosystem variants of the same loop.
- Compare hand-rolled vs SDK on pass rate, cost, code surface, and what the SDK hides — and state when to keep your own loop.
- Account for tokens and cost precisely with `response.usage` and `count_tokens`, and never with `tiktoken`.

Next: the exercises put this on a real agent. You will annotate a trace and find the failure, force each budget to fire and prove termination, and run a complete ~150-line ReAct agent against both Claude and Qwen. Continue to [the exercises](../exercises/README.md).

---

## References

- *Building effective agents* — Anthropic: <https://www.anthropic.com/research/building-effective-agents>
- *Anthropic Agent SDK / `claude-agent-sdk`*: <https://github.com/anthropics/claude-agent-sdk-python>
- *Anthropic Python SDK tool runner* (`tool_runner`, `@beta_tool`): <https://github.com/anthropics/anthropic-sdk-python>
- *Token counting* — Anthropic docs (`messages.count_tokens`): <https://platform.claude.com/docs/en/build-with-claude/token-counting>
- *Pricing* — Anthropic docs: <https://platform.claude.com/docs/en/about-claude/pricing>
- *OpenAI Agents SDK*: <https://openai.github.io/openai-agents-python/>
- *AWS Strands Agents*: <https://github.com/strands-agents/sdk-python>
- *Google ADK*: <https://google.github.io/adk-docs/>
