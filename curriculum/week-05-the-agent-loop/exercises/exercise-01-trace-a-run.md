# Exercise 1 — Trace a Run and Find the Failure

**Goal:** Run a hand-rolled agent on three tasks, capture a structured trace of each, then annotate one trace step by step and name the failure mode. You will train the single most important diagnostic habit of the week: reading an agent trace the way a backend engineer reads a request log, and pointing at the exact step where a run went wrong.

**Estimated time:** 45 minutes. Guided.

---

## Setup

You need your **Week 4 tool registry** and the hand-rolled agent from Lecture 1. If you have not pulled the lecture code into a file yet, copy the `agent()` function from [Lecture 1 §2](../lecture-notes/01-react-from-scratch.md) into `trace_agent.py`. Add a small structured-trace helper so every step prints one legible line:

```python
def log(step: int, kind: str, content: str) -> None:
    """One trace line per event. kind in {reason, act, observe, final, budget}."""
    print(f"step {step:<2} {kind:<7} {content[:100]}")
```

Wire it into the loop: log `act` before each tool call (`f"{name}({args})"`), log `observe` after (`out`), and log `final` when the model returns `end_turn`. If you enabled adaptive thinking, log the thinking text as `reason`.

Confirm it runs:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 trace_agent.py
```

---

## Step 1 — Run three tasks of increasing difficulty

Run your agent on these three tasks and capture each full trace to a file:

```python
TASKS = [
    # T1: single tool, should succeed cleanly in ~2 steps.
    "What is (1234 * 7) + 19?",
    # T2: two tools, should succeed in ~3-4 steps.
    "Fetch https://example.com and tell me how many characters the page body is, "
    "then multiply that by 3.",
    # T3: a task designed to be hard / ambiguous — watch what the agent does.
    "What is the population of the capital of the country that won the most recent "
    "FIFA World Cup, divided by 1000?",
]

for i, task in enumerate(TASKS, 1):
    print(f"\n==================== TASK {i} ====================")
    print(agent(task))
```

```bash
python3 trace_agent.py | tee notes/week-05/traces.txt
```

T1 should terminate on `end_turn` in a couple of steps. T2 should chain two tools. T3 is the interesting one — depending on the model, the tools you registered, and the day, it may answer cleanly, it may stall, or it may hit a budget. **That is the point.** You are collecting one clean run and at least one run with a wrinkle.

---

## Step 2 — Annotate the cleanest trace

Pick the T1 or T2 trace (a clean one). In `notes/week-05/trace-annotated.md`, paste the trace and annotate every line with what the agent was *doing* and *why*. Example:

```
step 1  reason  I need 17*23 first.        # decomposed the task: arithmetic first
step 1  act     calculator(expr=17*23)     # chose the right tool for the computation
step 1  observe 391                         # tool returned the value
step 2  final   17 * 23 = 391.              # had what it needed; answered. end_turn.
--- terminated: end_turn | steps=2/8 ... ---
```

The annotation forces you to read the trace as a *causal story*: each step's action follows from the previous step's observation. A clean run is a clean chain.

---

## Step 3 — Annotate a wrinkled trace and name the failure

Now take a trace with a wrinkle (most likely T3, or engineer one — see below). Annotate it the same way, and at the bottom, **name the failure mode** from the Lecture 2 §3 catalog:

- infinite tool-call loop
- re-calling a failing tool forever
- hallucinated tool name
- answering without acting
- stuck on the same sub-goal
- looks-done-but-isn't

If T3 ran cleanly for you (frontier models often handle it), *engineer* a failure to study. Three easy ways:

1. **Force a hallucinated name:** add a system-prompt line claiming a tool you did not register (`You also have a 'database' tool.`) and watch the model call it.
2. **Force re-calling a failing tool:** make one tool always raise with the *unhelpful* message `"Error"`, and watch the model retry it identically.
3. **Force answering-without-acting:** soften the system prompt to `Use tools only if absolutely necessary.` on T1 and watch the model guess the arithmetic.

Capture the trace, annotate it, and name what you induced.

---

## Step 4 — State the fix

Below your annotation, write one sentence: given this failure mode, what is the fix? Map it to Lecture 2 §3:

- hallucinated name → tighter tool descriptions + error-result naming the valid tools
- re-calling a failing tool → make the tool's error message *actionable*
- answering without acting → make the tool-use trigger explicit in the prompt
- infinite loop / stuck → the budget bounds it; the real fix is upstream (better tool or scope)

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] `notes/week-05/traces.txt` contains full structured traces for all three tasks, each ending in a termination summary line (`end_turn` or a named budget).
- [ ] `notes/week-05/trace-annotated.md` annotates one clean trace as a causal chain (each act follows from the prior observe).
- [ ] The same file annotates one *wrinkled* trace and names the failure mode from the Lecture 2 §3 catalog.
- [ ] You state the specific fix for that failure mode in one sentence, mapped to the lecture.
- [ ] You can say, out loud, *which step* the wrinkled run went wrong and why.

---

## Stretch

- Run the same three tasks through the **local Qwen 7B** path (Lecture 1 §5) and diff the traces. The 7B will fail more often and in more interesting ways — hallucinated names and answering-without-acting are common. Annotate one 7B failure the frontier model did not have.
- Enable **adaptive thinking** (`thinking={"type": "adaptive"}`, `output_config={"effort": "high"}`) on the Claude path and re-run T3. Does the explicit reasoning in the `reason` lines change where (or whether) it fails? Note the difference.
- Add a `reason`-line counter and a `act`-line counter to your summary. A run with many `reason` lines and few `act` lines is over-thinking; the reverse is acting-without-reasoning. What does each task's ratio tell you?

---

When this feels comfortable, move to [Exercise 2 — Budget guards](exercise-02-budget-guards.py).
