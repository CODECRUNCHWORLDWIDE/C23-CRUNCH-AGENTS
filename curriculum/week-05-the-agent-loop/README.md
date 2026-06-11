# Week 5 — The Agent Loop

Welcome to the week where the tool surface from Week 4 becomes a *thing that decides for itself what to do next*. By Friday you will be able to hand-roll a ReAct agent in ~150 lines of Python — no framework — point it at a local 7B and a frontier model, read its trace step by step, and tell a friend *exactly* why it failed on the third question. You will also be able to stop it from running forever, because you will have wired a step budget, a token budget, a time budget, and a cost budget into the loop from the start.

We assume you finished Week 4 and have a vendor-neutral tool registry (`crunch_tools` or equivalent) with at least a calculator, a sandboxed file-read, and an SSRF-guarded web-fetch. This week's agent imports that registry directly — the loop is new, the tools are not. If your `crunch_tools` isn't a clean `from crunch_tools.registry import REGISTRY`, fix that first; every exercise this week loops over it.

The one thing to internalize before you read another line: **most agent failures are not model failures — they are loop failures, budget failures, or tool failures.** The model is usually fine. What kills agents is the loop that never terminates (the "infinite tool-call loop"), the budget nobody set (so a stuck agent burns $40 overnight), and the tool that returned a confusing error the model couldn't recover from. A framework will hide all three from you. This week you build the loop by hand precisely so that when a framework's agent misbehaves in Week 13, you can open the box and read the trace instead of filing an issue.

This week is where you stop treating the agent as magic and start treating it as a `while` loop with budgets.

## Learning objectives

By the end of this week, you will be able to:

- **Implement** a ReAct (reason → act → observe) agent loop from scratch in ~150 lines, against both a frontier model and a local Qwen, over the Week 4 tool registry.
- **Read** an agent trace — every thought, tool call, observation, and the final answer — and narrate step by step *why* a run succeeded or failed.
- **Distinguish** the agent patterns — ReAct, plan-and-execute, reflection / self-critique — and state when each helps and when it's overhead.
- **Diagnose** the canonical failure modes: the infinite tool-call loop, the model that re-calls a failing tool forever, the model that hallucinates a tool name, the model that answers without acting.
- **Enforce** four budgets — step, token, time, and cost — so an agent always terminates, and explain why each is necessary (each catches a different runaway).
- **Compare** the hand-rolled loop against a real SDK — the **Anthropic `claude-agent-sdk`** / the `messages` tool runner — honestly: pass rate, cost, code surface area, and what the SDK does for you versus hides from you.
- **Survey** the 2026 agent SDK landscape — Anthropic `claude-agent-sdk`, OpenAI Agents SDK, AWS Strands, Google ADK — and place each.
- **Measure** an agent on a fixed task benchmark and report a pass rate with cost and latency, not a vibe.

## Prerequisites

This week assumes you have completed **C23 weeks 1–4**, or have equivalent fluency. Specifically:

- The Week 4 **tool registry** (`crunch_tools` or equivalent) importable, with at least calculator + file-read + web-fetch, each hardened and validated.
- Comfort with the **two-turn tool-use round trip** from Week 4: `tool_use` (with `id`/`name`/`input`) → run the tool → `tool_result` (matching `tool_use_id`) → next turn. The agent loop is this round trip in a `while`.
- Python 3.12, the `anthropic` SDK, `ANTHROPIC_API_KEY` exported, and **Ollama** serving `qwen2.5:7b-instruct`. `ollama list` shows it.
- You can read JSON Schema and validate a dict against it (Week 4). The agent dispatches through `REGISTRY[name].run(args)`, which validates for you.
- You know what `stop_reason == "tool_use"` means and why a mismatched `tool_use_id` 400s. If not, re-read Week 4 Lecture 1 before Monday.

You do **not** need prior framework experience (LangGraph is Week 13). This week is deliberately framework-free — you build the loop so you understand the thing the frameworks wrap.

## Topics covered

- **The agent loop, formally**: the cycle of *model call → if `tool_use`, run tools → feed results → repeat → until `end_turn`*. Why it's just the Week 4 round trip wrapped in `while`, and what the system prompt has to say to make it work.
- **ReAct** (reason + act + observe): the pattern where the model interleaves thinking, tool calls, and observations. Native tool-use vs the classic text-based `Thought:/Action:/Observation:` scaffold, and why native tool-use is the 2026 default.
- **Plan-and-execute**: make a plan first, then execute the steps. When the upfront plan helps (long multi-step tasks) and when it's premature commitment.
- **Reflection / self-critique**: the agent critiques its own draft and revises. When a critique pass earns its tokens and when it's just doubling cost.
- **The failure modes**: the infinite tool-call loop, re-calling a failing tool forever, hallucinated tool names, answering-without-acting, and the "stuck on the same sub-goal" loop. How to detect each from the trace.
- **Budgets as first-class loop state**: step budget (max turns), token budget (cumulative tokens), time budget (wall clock), cost budget (dollars). Why you need all four — each stops a different runaway — and where in the loop each check goes.
- **Reading a trace**: structuring the agent's output so every thought, action, observation, and budget tick is legible; narrating a run; finding the step where it went wrong.
- **The SDK survey**: the **Anthropic `claude-agent-sdk`** and the `messages` tool runner as the Claude-native loop; **OpenAI Agents SDK**, **AWS Strands Agents**, **Google ADK** at a survey level. What an SDK loop does for you and what it hides.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                              | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|----------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | The loop; ReAct; the system prompt; first run      |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Patterns: plan-and-execute, reflection; when each  |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Failure modes; the four budgets; trace reading     |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | SDK survey; hand-rolled vs `claude-agent-sdk`      |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The benchmark; pass rate vs cost vs code surface    |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                             |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, comparison write-up polish           |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                    | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The ReAct paper, the agent-SDK docs, the "building effective agents" canon, and the talks worth your time |
| [lecture-notes/01-react-from-scratch.md](./lecture-notes/01-react-from-scratch.md) | The loop, ReAct, the system prompt, the patterns, and a full hand-rolled agent |
| [lecture-notes/02-budgets-failure-modes-and-the-sdks.md](./lecture-notes/02-budgets-failure-modes-and-the-sdks.md) | The four budgets, the failure-mode catalog, trace reading, and the SDK survey |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-trace-a-run.md](./exercises/exercise-01-trace-a-run.md) | Run a hand-rolled agent on three tasks; annotate every step of one trace and find the failure |
| [exercises/exercise-02-budget-guards.py](./exercises/exercise-02-budget-guards.py) | Add step/token/time/cost budgets to a loop; force each one to fire and prove the agent terminates |
| [exercises/exercise-03-react-loop.py](./exercises/exercise-03-react-loop.py) | A complete ~150-line ReAct agent over the Week 4 registry, runnable against Claude and Qwen |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-hand-rolled-vs-sdk.md](./challenges/challenge-01-hand-rolled-vs-sdk.md) | Re-implement your agent with `claude-agent-sdk`; benchmark both; report pass rate, cost, and code surface honestly |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the hand-rolled-vs-SDK comparison write-up |
| [mini-project/README.md](./mini-project/README.md) | The 25-task agent benchmark harness, hand-rolled loop vs SDK, frontier vs local |

## The "the agent terminated cleanly" promise

C23 uses a recurring marker for every exercise that ends in an agent that *finished* — reached an answer or hit a budget, but did not hang:

```text
step 1  reason  "I need to compute 17 * 23 first."
step 1  act     calculator(expression="17 * 23")
step 1  observe "391"
step 2  reason  "Now I can answer."
step 2  final   "17 * 23 = 391."
--- terminated: end_turn | steps=2/8 tokens=812/20000 time=1.4s/30s cost=$0.003/$0.10
```

If the run ends with `terminated: step budget exceeded` instead of `end_turn`, the agent got stuck — and the budget *saved you*, which is the point. If it ends with no terminating line at all, your loop has no budget and *will* hang on the wrong input. The point of Week 5 is to make that clean terminating line ordinary — and to make a runaway *loud and bounded* instead of silent and unbounded.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **ReAct paper** (Yao et al., 2022) until you can draw the reason→act→observe cycle from memory and explain why interleaving beats reason-then-act: <https://arxiv.org/abs/2210.03629>.
- Add a **reflection pass** to your agent: after the draft answer, have the model critique it against the question and optionally revise. Measure the pass-rate lift *and* the token cost on your benchmark — is it worth it?
- Wire the **`claude-agent-sdk`** properly (sessions, the managed loop) and compare its trace format to your hand-rolled one. Note what it logs that you didn't think to.
- Implement a **plan-and-execute** variant: first call asks the model for a numbered plan, subsequent calls execute each step. Compare its pass rate and token cost to ReAct on a *long* multi-step task.

## Up next

Week 6 takes the agent you built here and stands it up on **local inference** — Ollama, llama.cpp, vLLM — so the loop you hand-rolled runs entirely on your own hardware with a measured tokens/sec and a chosen quantization. By the end of Week 6 you have the Phase I milestone: a working ReAct agent on a local 7B with a benchmark score. Push your mini-project before you start it; Week 6 points the same loop at a local server.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
