# Week 5 Homework

Six problems that drive the agent loop into your fingers. The full set should take about **5 hours**. Work in your Week 5 Git repository (the same workspace as the exercises and the `crunch_agent` mini-project) so every problem produces at least one commit you can point to at the Phase I milestone review in Week 6.

The headline deliverable is **Problem 4 — the hand-rolled-vs-SDK comparison write-up**, the syllabus's "compare SDKs honestly" skill. Treat it as the artifact a reviewer reads, not a journal entry.

Each problem includes a short **problem statement**, **acceptance criteria** so you know when you're done, a **hint** if you get stuck, and an **estimated time**.

Export `ANTHROPIC_API_KEY` in every terminal. Have Ollama serving `qwen2.5:7b-instruct` for the local problems. Have your **Week 4 tool registry** importable — Problems 1, 2, 4, and 6 loop over it.

---

## Problem 1 — The trace, made legible

**Problem statement.** Take the hand-rolled agent from Lecture 1 (or your mini-project loop) and add a structured trace: one line per event, with the step number, the kind (`reason` / `act` / `observe` / `final` / `budget`), and the content. Run it on a two-tool task (e.g. "fetch a page, count its words, multiply by 3") and capture the full trace plus the termination summary line to `notes/week-05/trace.txt`.

**Acceptance criteria.**

- `notes/week-05/trace.txt` shows a complete run with `act` / `observe` lines for each tool call and a `final` line.
- The run ends with a termination summary: `--- terminated: end_turn | steps=.../... tokens=.../... time=.../... cost=$.../$... ---`.
- The trace reads as a causal chain — each `act` follows from the previous `observe`.
- Committed.

**Hint.** The summary line is the "the agent terminated cleanly" promise from the week README. If your trace has no summary line, your loop has no budget — add the `Budgets` object from Lecture 2 §2 first.

**Estimated time.** 30 minutes.

---

## Problem 2 — Force every budget to fire

**Problem statement.** Using the `Budgets` dataclass (step/token/time/cost), construct four runs of the *same* task where a *different* budget fires each time — by setting one budget tight and the rest loose. Prove the agent terminates under each. You may use the `FakeModel` approach from Exercise 2 (a model that loops forever) so the runs are deterministic and free.

**Acceptance criteria.**

- `notes/week-05/budgets.md` shows four runs, each terminating on the intended budget (step, then token, then time, then cost), with the termination summary line for each.
- You state, in one sentence, why an infinitely-looping model still terminates (the loop, not the model, owns the exit).
- Committed.

**Hint.** To make the *cost* budget fire before the *token* budget, set `max_dollars` low and `max_tokens` high — recall cost = input×$5/M + output×$25/M, so output tokens are 5× as expensive. To make *time* fire, give the fake model a small `sleep`.

**Estimated time.** 40 minutes.

---

## Problem 3 — Induce and name three failure modes

**Problem statement.** Deliberately induce three of the Lecture 2 §3 failure modes on your agent and capture each trace. Suggested: (a) **hallucinated tool name** — tell the system prompt the agent has a tool you did not register; (b) **re-calling a failing tool** — make one tool always raise with an unhelpful `"Error"` message; (c) **answering without acting** — soften the prompt to discourage tool use on a task that needs it. For each, capture the trace, name the failure mode, and state the fix.

**Acceptance criteria.**

- `notes/week-05/failure-modes.md` documents three induced failures, each with a trace excerpt, the failure-mode name from the §3 catalog, and the specific fix.
- For the re-calling-a-failing-tool case, you show that an *actionable* error message (naming what was wrong) changes the model's behavior versus the bare `"Error"`.
- Committed.

**Hint.** The cleanest demonstration of fix (b): run the same task twice — once with the tool raising `"Error"`, once with it raising `"Error: 'date' must be ISO-8601 (YYYY-MM-DD); got '3/15'"` — and diff the traces. The actionable message usually breaks the loop.

**Estimated time.** 1 hour.

---

## Problem 4 — Hand-rolled vs SDK, the comparison (headline deliverable)

**Problem statement.** This is the syllabus deliverable. Run the *same* agent two ways — your hand-rolled loop and the in-SDK tool runner (`client.beta.messages.tool_runner` with `@beta_tool`) — on a shared 8-task set with a deterministic grader, all on `claude-opus-4-8`. Measure pass rate, total cost (from real `usage`), and lines of agent code you maintain. Then write a one-page comparison at `notes/week-05/handrolled-vs-sdk.md`.

**Acceptance criteria.**

- `notes/week-05/handrolled-vs-sdk.md` exists, fits on roughly one page, and contains:
  1. A table: pass rate, cost, code surface (LOC) for both implementations on the same 8 tasks.
  2. A "what the SDK hides" list — at least three concrete items (trace format, budget placement, retry behavior, interleaving).
  3. A recommendation (keep hand-rolled / move to SDK) tied to at least two of the measured numbers — not taste.
- Pass rate comes from a deterministic grader; cost comes from real `usage`.
- The recommendation names a trade-off, not just a winner.
- Committed.

**Hint.** Keep the tools, system prompt, and model identical across both — the only difference is who drives the loop. If your cost numbers say the SDK is cheaper *and* fewer lines *and* equal pass rate, re-check the cost accounting; the honest answer usually has a trade-off in it.

**Estimated time.** 1 hour.

---

## Problem 5 — The frontier/local gap, measured

**Problem statement.** Run your agent on the same 8-task set on *both* `claude-opus-4-8` (Anthropic) and `qwen2.5:7b-instruct` (Ollama). Report pass rate and median latency per provider, broken down by task category (math / lookup / text). Identify which category the local model loses the most ground on and explain why from a trace.

**Acceptance criteria.**

- `notes/week-05/frontier-vs-local.md` shows pass rate and p50 latency per provider, with a per-category breakdown.
- You name the category where the local model loses the most and quote a trace where the 7B failed a task the frontier model passed.
- You note that the local path's dollar cost is effectively zero (electricity, not API) — half the point of the local path.
- Committed.

**Hint.** The 7B most often loses on multi-step lookup (it gives up or hallucinates a tool name) and multi-step math (it answers without acting). The trace will show exactly which. The same loop runs both — only the adapter differs (Lecture 1 §5).

**Estimated time.** 50 minutes.

---

## Problem 6 — Does reflection earn its tokens?

**Problem statement.** Add a reflection pass to your agent: after the draft answer, a critique call lists concrete problems, and a revision call optionally fixes them (Lecture 1 §4.2). Run your 8-task set with and without reflection on `claude-opus-4-8`. Report the pass-rate change *and* the cost change. Decide, with the numbers, whether reflection earned its tokens on this task set.

**Acceptance criteria.**

- `notes/week-05/reflection.md` reports pass rate and total cost for {no reflection, with reflection} on the same 8 tasks.
- You state a verdict — "worth it" or "not worth it" — backed by the two numbers, and note what *kind* of task (if any) it helped on.
- If reflection ever turned a correct answer into a wrong one, you flag it (a real risk of un-grounded critique).
- Committed.

**Hint.** Reflection helps most on tasks with a *checkable* error and hurts on open-ended ones where it's just the model second-guessing itself. On an 8-task math/lookup/text set, expect a small or zero lift at a real cost increase — and that "small or zero" is a legitimate, defensible finding. The point is that you measured.

**Estimated time.** 50 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — The trace, made legible | 30 min |
| 2 — Force every budget to fire | 40 min |
| 3 — Induce and name three failure modes | 1 h 0 min |
| 4 — Hand-rolled vs SDK (headline) | 1 h 0 min |
| 5 — Frontier/local gap, measured | 50 min |
| 6 — Does reflection earn its tokens? | 50 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunch_agent` [mini-project](./mini-project/README.md) is in the same workspace — Week 6 imports it. Then take the [quiz](./quiz.md) with your notes closed.

---

## Grading rubric (per the canonical weekly mini-project rubric)

The homework is graded on the same four axes as every C23 deliverable. For the homework specifically:

| Axis | Weight | What "meets" looks like |
|---|---:|---|
| **Correctness** | 30% | The agent runs both providers, terminates on every task (end_turn or a named budget), and the induced failure modes reproduce. Budgets fire as intended in Problem 2. |
| **Engineering quality** | 25% | The loop body carries no provider branches; tool errors and hallucinated names don't crash the loop; the trace is legible. Readable code, sensible structure. |
| **Measurement** | 25% | Pass rate from a deterministic grader; cost from real `usage`; latency reported; the frontier/local and with/without-reflection deltas are real numbers with the method documented. Vibes do not count. |
| **Write-up** | 20% | The Problem 4 comparison reads as an artifact a reviewer would act on: a table, a "what it hides" list, and a recommendation that names a trade-off. |

Graders are instructed to **fail vibes-only submissions** — a working agent demo with no measured pass rate or cost is not a "meets."
