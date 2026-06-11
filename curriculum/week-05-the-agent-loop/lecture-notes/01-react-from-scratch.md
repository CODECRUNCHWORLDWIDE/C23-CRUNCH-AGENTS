# Lecture 1 — The Agent Loop, ReAct, and a Hand-Rolled Agent

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can explain the agent loop as the Week 4 tool-use round trip wrapped in a `while`, write a ReAct agent in ~150 lines against both Claude and a local Qwen, name the patterns (ReAct, plan-and-execute, reflection) and say when each earns its tokens, and read the trace your agent emits.

If you remember one sentence from this entire week, remember this one:

> **An agent is not magic. It is a `while` loop that calls a model, runs whatever tools the model asked for, feeds the results back, and repeats until the model says it is done — and almost everything that goes wrong is the loop, the budget, or a tool, not the model.**

Week 4 gave you the tool-call *round trip*: you send a request with `tools`, the model replies with `stop_reason == "tool_use"` and one or more `tool_use` blocks, you run the tools, you send `tool_result` blocks back, and the model produces a final answer. That was *one* round trip — the model asked for tools once, you answered once, done. This week, the model asks again. And again. The thing that lets it keep going until the task is actually solved is the loop. That is the entire idea. Everything else is budgets (so it terminates), patterns (so it reasons well), and trace-reading (so you can tell what happened).

This lecture builds the loop by hand. We use the **Anthropic Python SDK** (`anthropic`, model `claude-opus-4-8`) for the frontier path and **Ollama** serving `qwen2.5:7b-instruct` for the local path. No agent framework — that is deliberate. LangGraph is Week 13; you will appreciate it far more once you have felt the thing it wraps.

---

## 1. The loop, formally

Here is the agent loop with nothing removed and nothing added:

```
1. messages = [ system, user_task ]
2. loop:
3.     response = model(messages, tools)
4.     append response to messages          # the assistant turn, verbatim
5.     if response.stop_reason != "tool_use":
6.         return final answer               # the model is done
7.     for each tool_use block in response:
8.         result = run_tool(block.name, block.input)
9.         collect a tool_result(block.id, result)
10.    append all tool_results as one user turn
11.    # back to step 3
```

That is it. Lines 3–10 are exactly the Week 4 round trip. The only new thing is the word **loop** on line 2 and the **back to step 3** on line 11. Read it twice. When a framework's agent does something baffling in Week 13, this is the box you open.

Three details matter enormously and are where beginners lose hours:

- **Append the assistant turn verbatim (line 4).** You append `response.content` — the *whole* list of content blocks, including the `tool_use` blocks — not a string you extracted. If you summarize it or drop the `tool_use` blocks, the next request 400s because a `tool_result` has no matching `tool_use_id`.
- **One `tool_result` per `tool_use_id`, all in a single user turn (lines 7–10).** A single assistant turn can contain multiple `tool_use` blocks (parallel tool use, Week 4). You must return exactly one `tool_result` for each, batched into one user message. Miss one and the API rejects the next turn.
- **The stop condition is the model's `stop_reason` (line 5).** `tool_use` means "keep going." `end_turn` means "I'm done." You do not decide when the agent is finished — the model does, *within the budget you enforce* (Lecture 2).

### Why the system prompt matters here

In a one-shot call the system prompt is flavor. In an agent loop it is load-bearing, because it is the only place you tell the model *how to behave across many turns*: when to use tools, when to stop, how to handle a tool error, how terse to be. A good agent system prompt for this week says roughly:

> You are a problem-solving agent with access to tools. Work step by step. When you need a fact you do not have, or a computation you cannot do reliably in your head, call the appropriate tool rather than guessing. After each tool result, decide whether you have enough to answer or need another tool call. When you can answer, answer directly and concisely. Do not call a tool you do not need.

That last sentence matters. On `claude-opus-4-8`, over-aggressive instructions like "ALWAYS use a tool" cause *over-triggering* — the model calls tools it does not need. State the trigger condition plainly ("call the tool when the answer depends on a current fact or a precise computation") and let the model judge. This is the 2026 prompting reality from Week 3, applied to agents.

---

## 2. A hand-rolled ReAct agent against Claude

Let us build the loop for real. We will reuse the Week 4 tool registry — assume you have `REGISTRY`, a dict from tool name to an object with `.schema` (the JSON-Schema tool definition the API wants) and `.run(args) -> str` (validates and executes). If your Week 4 mini-project produced something shaped differently, adapt the two adapter functions and the rest is unchanged.

```python
"""A hand-rolled ReAct agent loop against Claude (claude-opus-4-8).

Reuses the Week 4 tool registry. The loop is ~40 lines; everything else is
plumbing for a legible trace. No framework.
"""
from __future__ import annotations

import anthropic

# from crunch_tools.registry import REGISTRY   # your Week 4 mini-project
# For this lecture we inline a tiny registry so the file runs standalone.

MODEL = "claude-opus-4-8"

SYSTEM = (
    "You are a problem-solving agent with access to tools. Work step by step. "
    "When you need a fact you do not have or a computation you cannot do reliably "
    "in your head, call the appropriate tool rather than guessing. After each tool "
    "result, decide whether you can answer or need another tool. Answer concisely "
    "when you can. Do not call a tool you do not need."
)


def tool_schemas() -> list[dict]:
    """The `tools` argument the Messages API wants: name, description, input_schema."""
    return [t.schema for t in REGISTRY.values()]


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Dispatch one tool call. Returns (result_text, is_error)."""
    tool = REGISTRY.get(name)
    if tool is None:
        # The model hallucinated a tool name. Tell it so — loudly, as an error
        # result — instead of crashing the loop. (Failure mode, Lecture 2.)
        return (f"Error: no tool named {name!r}. Available: {list(REGISTRY)}", True)
    try:
        return (tool.run(args), False)
    except Exception as exc:  # noqa: BLE001 — surface any tool failure to the model
        return (f"Error running {name}: {exc}", True)


def agent(task: str, max_steps: int = 8) -> str:
    """Run the ReAct loop until end_turn or the step budget (Lecture 2 expands this)."""
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=tool_schemas(),
            messages=messages,
        )
        # Append the assistant turn VERBATIM — including tool_use blocks.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # The model is done. Pull the text out and return it.
            final = next((b.text for b in response.content if b.type == "text"), "")
            print(f"--- terminated: end_turn at step {step} ---")
            return final

        # Run every tool the model asked for; collect one result per tool_use_id.
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"step {step}  act     {block.name}({block.input})")
            out, is_error = run_tool(block.name, dict(block.input))
            print(f"step {step}  observe {out[:80]}")
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": out,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})

    print(f"--- terminated: step budget ({max_steps}) exhausted ---")
    return "I ran out of steps before reaching an answer."
```

Read the loop body. Lines 3–11 of §1 are right there: call the model, append the assistant turn, check `stop_reason`, run tools, append results, repeat. The `print` statements are your trace — we make them rigorous in §6 and in the exercises.

Note three things this code does that beginners forget:

1. **It never raises on a tool failure.** A tool that throws becomes a `tool_result` with `is_error: True`. The model reads that error and can recover — retry with different arguments, or apologize and answer without the tool. If you let the exception propagate, you kill the agent over a recoverable hiccup.
2. **It never raises on a hallucinated tool name.** Same treatment: the unknown name becomes an error result naming the valid tools, and the model usually corrects itself on the next turn.
3. **It has a step budget already** (`max_steps`). Even this first cut cannot loop forever. Lecture 2 turns that single guard into four (step, token, time, cost).

### Running it

```python
print(agent("What is (1234 * 7) + 19, and what is the capital of the country whose "
            "name in English starts with the 12th letter of the alphabet?"))
```

A healthy trace looks like:

```
step 1  act     calculator({'expression': '(1234 * 7) + 19'})
step 1  observe 8657
step 2  act     ...
--- terminated: end_turn at step 3 ---
```

The model decomposed the task, called the calculator for the arithmetic, reasoned about the lettering itself (no tool needed), and answered. That decomposition — *reason, then act, then observe, then reason again* — is ReAct, and it is the next section.

---

## 3. ReAct: reason + act + observe

**ReAct** (Yao et al., 2022 — on the reading list) named the pattern your loop already implements: the model **reasons** about what to do, **acts** by calling a tool, **observes** the result, and then reasons again with the observation in hand. The power is in the *interleaving*. A model that must produce a whole plan before acting commits to steps before it knows what the early steps return. A ReAct model adapts: it sees that the web search returned nothing useful and tries a different query, instead of barreling down a dead plan.

There are two ways to realize ReAct, and the distinction matters in 2026:

### 3.1 Native tool-use ReAct (the default)

This is what the §2 code does. "Reasoning" is the model's own thinking and any text it emits; "acting" is a real `tool_use` block; "observing" is the `tool_result` you feed back. The structure is enforced by the API's tool protocol. You do not parse anything — the SDK hands you typed `tool_use` blocks. This is the correct default on every 2026 frontier model and on tool-capable open models like Qwen 2.5.

With `claude-opus-4-8`, you can also enable **adaptive thinking** so the model reasons explicitly between tool calls:

```python
response = client.messages.create(
    model=MODEL,
    max_tokens=4096,
    thinking={"type": "adaptive"},        # the model decides how much to think
    output_config={"effort": "high"},     # depth/spend knob: low|medium|high|max
    system=SYSTEM,
    tools=tool_schemas(),
    messages=messages,
)
```

Adaptive thinking interleaves reasoning between tool calls automatically — the model thinks, decides to act, observes, thinks again. The `thinking` blocks come back in `response.content` alongside the `tool_use` blocks; append them verbatim like everything else (the API rejects modified thinking blocks). This is ReAct with the "reason" step made first-class by the model rather than scaffolded by you.

### 3.2 Text-scaffolded ReAct (the classic, mostly historical)

The original ReAct paper predates native tool-calling APIs. It scaffolded the loop *in text*: the prompt instructed the model to emit `Thought:`, then `Action:`, then `Action Input:`, and the harness parsed those strings, ran the action, and appended an `Observation:` line. You will see this in older codebases and in tutorials.

```
Thought: I need to compute 1234 * 7 + 19.
Action: calculator
Action Input: (1234 * 7) + 19
Observation: 8657
Thought: Now I can answer.
Final Answer: 8657.
```

It works, but it is brittle: you are regex-parsing free text, and the model can emit a malformed `Action Input` that your parser chokes on. In 2026 you reach for native tool-use unless you are on a model with no tool API at all. We mention the scaffold so you recognize it and so you understand that ReAct is a *pattern*, not an API feature — the API just makes it clean.

> **The honest 2026 take:** ReAct is the right default pattern for tool-using agents. Native tool-use is the right way to implement it. You will spend the rest of this course on native tool-use; you should be able to *recognize* the text scaffold and explain why it is brittle, and that is enough.

---

## 4. The other patterns: plan-and-execute and reflection

ReAct is not the only pattern, and part of being a senior engineer is knowing when a different one earns its tokens.

### 4.1 Plan-and-execute

**Plan-and-execute** front-loads the reasoning: a first model call produces a numbered plan, then subsequent calls execute the steps. The pitch is that an explicit upfront plan keeps a long, multi-step task coherent — the agent does not wander.

```python
PLAN_SYSTEM = (
    "Produce a numbered plan to accomplish the task. Each step should be a single "
    "concrete action. Do not execute anything yet — only plan."
)
```

When it helps: genuinely long tasks with many interdependent steps, where ReAct's myopia ("just do the next thing") causes it to lose the thread. When it hurts: most tasks. The upfront plan is *premature commitment* — the agent commits to step 4 before it has seen what step 2 returns, and if step 2 surprises it, the plan is wrong and the agent either follows it off a cliff or pays to replan. For the 25-task benchmark in this week's mini-project, plain ReAct usually beats plan-and-execute on pass rate *and* cost. Reach for plan-and-execute when you have measured that ReAct wanders, not as a default.

### 4.2 Reflection / self-critique

**Reflection** adds a critique pass: after the agent drafts an answer, a second model call critiques that draft against the task, and a third revises it. (Reflexion, Shinn et al., 2023 — on the reading list — is the research root.)

```python
CRITIQUE_SYSTEM = (
    "You are a critic. Given a task and a candidate answer, list concrete, specific "
    "problems with the answer: factual errors, missed requirements, unjustified "
    "claims. If the answer is correct and complete, say 'No issues.'"
)
```

When it helps: tasks where the failure mode is a subtle, *checkable* error — a math result that can be verified, a piece of code that can be run, a claim that can be cross-checked against a retrieved document. A critique pass that *grounds its critique in a tool* (re-run the code, re-read the source) is worth real money. When it hurts: open-ended tasks where the critique is just the same model second-guessing itself with no new information — you pay double the tokens for a wash, or worse, the model "fixes" a correct answer into a wrong one. **Measure the lift.** The stretch goal this week is exactly this: add a reflection pass, measure pass-rate change *and* token cost on the benchmark, and decide whether it earned its keep. Usually it does not. Sometimes it does. The point is that you measured.

> **The pattern-selection rule of thumb:** Start with ReAct. Add plan-and-execute only if you have measured that ReAct wanders on your task. Add reflection only if the failure mode is a checkable error and the critique can ground itself in a tool. Every added pattern is added tokens and added latency — make it justify itself with a number, not a vibe.

---

## 5. The same loop against a local model

The thesis of this whole course is that the loop is the same underneath. Let us prove it: the identical ReAct loop, pointed at `qwen2.5:7b-instruct` through Ollama. Qwen 2.5 7B supports tool-calling in the OpenAI-compatible shape, so we use Ollama's `/v1` endpoint with the `openai` client.

```python
"""The same ReAct loop against a local Qwen 2.5 7B via Ollama's OpenAI-compatible API.

Ollama exposes an OpenAI-shaped /v1 endpoint. The tool schema is the SAME JSON
Schema you wrote in Week 4 — only the block names and the loop's plumbing differ
from the Anthropic path.
"""
from openai import OpenAI

local = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key is ignored
LOCAL_MODEL = "qwen2.5:7b-instruct"


def to_openai_tools() -> list[dict]:
    """Wrap each Week 4 tool schema in the OpenAI `function` envelope Qwen expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.schema["name"],
                "description": t.schema["description"],
                "parameters": t.schema["input_schema"],  # same JSON Schema
            },
        }
        for t in REGISTRY.values()
    ]


def local_agent(task: str, max_steps: int = 8) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]
    for step in range(1, max_steps + 1):
        resp = local.chat.completions.create(
            model=LOCAL_MODEL,
            messages=messages,
            tools=to_openai_tools(),
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))  # append assistant turn verbatim

        if not msg.tool_calls:
            print(f"--- terminated: stop at step {step} ---")
            return msg.content or ""

        for call in msg.tool_calls:
            import json

            args = json.loads(call.function.arguments)  # Qwen returns args as a JSON string
            print(f"step {step}  act     {call.function.name}({args})")
            out, _is_error = run_tool(call.function.name, args)
            print(f"step {step}  observe {out[:80]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": out,
                }
            )
    print(f"--- terminated: step budget ({max_steps}) exhausted ---")
    return "I ran out of steps before reaching an answer."
```

Compare this to the Anthropic version in §2. The *loop* is identical: call model, append assistant turn, check for tool calls, run them, append results, repeat. What differs is plumbing that the vendors did not standardize:

- **Block shape.** Anthropic returns typed `tool_use` content blocks; OpenAI/Qwen returns `tool_calls` on the message with `arguments` as a *JSON string* you must `json.loads`.
- **Result shape.** Anthropic wants a `tool_result` block with `tool_use_id`; OpenAI/Qwen wants a `role: "tool"` message with `tool_call_id`.
- **The tool schema is the same.** The JSON Schema you wrote in Week 4 drops straight into both — that is the portable part, and it is the part that matters.

The 7B will be less reliable than the frontier model — it hallucinates tool names more, sometimes answers without acting, sometimes loops. That is not a bug in your loop; it is the capability gap, and *reading the trace to see exactly where the 7B went wrong* is the skill this week builds. The mini-project benchmarks both so you can put a number on that gap.

---

## 6. Reading the trace

An agent you cannot trace is a closed box. The single most valuable habit this week is structuring the agent's output so every step is legible, then narrating a run out loud. A good trace line carries: the step number, the kind of event (reason / act / observe / final / budget), and the content.

```
step 1  reason  "I need 17 * 23 first; I'll use the calculator."
step 1  act     calculator(expression="17 * 23")
step 1  observe "391"
step 2  reason  "Now I can answer."
step 2  final   "17 * 23 = 391."
--- terminated: end_turn | steps=2/8 tokens=812/20000 time=1.4s/30s cost=$0.004/$0.10
```

That last line is the **termination summary** — the recurring "the agent terminated cleanly" marker from the week README. It tells you *how* the agent stopped (`end_turn`, not a budget) and how much of each budget it used. When you read a failed run, you are looking for the step where the trace stops making sense:

- A `reason` line that misunderstands the task → a prompt problem.
- An `act` line calling a tool that does not exist → a hallucinated tool name (the model needs better tool descriptions, or the error-result recovery in §2 to nudge it).
- The same `act` line repeating with the same arguments and the same failing `observe` → the **re-calling-a-failing-tool** loop (Lecture 2). The agent is stuck; only a budget saves it.
- An `act` that ignores the previous `observe` entirely → the model is not actually reading tool results; often a context or formatting problem.

Exercise 1 has you annotate a real trace and find the failure step. Do it slowly. This is the skill that separates the engineer who can debug an agent from the one who files an issue against the framework.

---

## 7. Recap

You should now be able to:

- State the agent loop as the Week 4 round trip wrapped in a `while`, and name the three plumbing details that bite beginners (append the assistant turn verbatim; one `tool_result` per `tool_use_id`; stop on the model's `stop_reason`).
- Write a hand-rolled ReAct agent against `claude-opus-4-8` in ~40 lines of loop, with tool-error and hallucinated-name recovery built in.
- Explain ReAct (reason + act + observe), why interleaving beats plan-then-act, and why native tool-use is the 2026 way to implement it.
- Say when plan-and-execute and reflection earn their tokens — and that the answer is "measure it," not "always add them."
- Point the *same loop* at a local Qwen 7B, and name exactly what is portable (the JSON-Schema tool definition) and what is not (block names, result envelopes).
- Read an agent trace and find the step where a run went wrong.

Next: how to make the loop *always terminate* with four budgets, the full catalog of agent failure modes, and an honest comparison of your hand-rolled loop against the `claude-agent-sdk`. Continue to [Lecture 2 — Budgets, Failure Modes, and the SDKs](./02-budgets-failure-modes-and-the-sdks.md).

---

## References

- *ReAct: Synergizing Reasoning and Acting in Language Models* (Yao et al., 2022): <https://arxiv.org/abs/2210.03629>
- *Building effective agents* — Anthropic: <https://www.anthropic.com/research/building-effective-agents>
- *Tool use — implement the agentic loop* — Anthropic docs: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use>
- *Reflexion: Language Agents with Verbal Reinforcement Learning* (Shinn et al., 2023): <https://arxiv.org/abs/2303.11366>
- *Plan-and-Solve Prompting* (Wang et al., 2023): <https://arxiv.org/abs/2305.04091>
- *Ollama OpenAI-compatibility* — `/v1` endpoint and tool calls: <https://github.com/ollama/ollama/blob/main/docs/openai.md>
