# Week 14 Homework

Six problems that revisit the week's topics and put durable, typed agent design into your fingers. The full set should take about **5 hours**. Work in your Week 14 Git repository (the same workspace as the exercises and the `crunchagent-durable` mini-project) so every problem produces at least one commit you can point to in the production weeks.

The headline deliverable is **Problem 4 — the Mastra-vs-LangGraph + Inngest resume-after-crash comparison memo**, the syllabus lab. Treat it as the artifact a reviewer reads, not a journal entry.

Have **Node 20+** installed (`node --version`), `tsx` available (`npx tsx ...`), and the Inngest dev server runnable (`npx inngest-cli@latest dev`). Have your **week-13 LangGraph supervisor** importable for the comparison. For the frontier path set `ANTHROPIC_API_KEY`; for the open path point at a local vLLM/Ollama endpoint; every problem has a deterministic offline fallback so a missing key never blocks you.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Scaffold a Mastra agent with a typed tool

**Problem statement.** Build a minimal Mastra agent: `instructions`, `model: anthropic("claude-sonnet-4-6")`, and one typed tool (zod `inputSchema`/`outputSchema`). Call it on a question that needs the tool, and confirm via `result.toolCalls` that the agent *called* the tool rather than guessing. Then deliberately introduce a type error (e.g. read `result.txt`) and capture the compiler rejecting it.

**Acceptance criteria.**

- `notes/week-14/agent.ts` (or a committed file) with a runnable Mastra agent + one typed tool.
- A captured run showing `result.text` and `result.toolCalls` (the tool was actually called).
- A note showing the deliberate type error and the compiler's rejection — and one sentence on why catching it at build time beats catching it at runtime.
- Committed.

**Hint.** Use `createTool({ id, description, inputSchema, outputSchema, execute })`. The agent calls the tool when the instructions tell it to and the question needs it. If you have no key, the agent's text will be a stub, but the tool-call wiring and the type check are what you're proving.

**Estimated time.** 40 minutes.

---

## Problem 2 — A two-step Mastra workflow with a checked seam

**Problem statement.** Build a Mastra workflow of two `createStep`s chained with `.then()` and `.commit()`. The first step outputs a shape; the second step's `inputSchema` must match it. Then deliberately mismatch the second step's input schema and capture the compiler rejecting the `.then()` seam — the workflow-level type-safety win.

**Acceptance criteria.**

- A runnable two-step workflow (`createWorkflow` + two `createStep` + `.then()` + `.commit()`) that produces a typed output.
- A captured compile error from a deliberate input/output schema mismatch at the seam.
- One sentence stating when you'd use a workflow (you decide the flow) versus an agent (the model decides).
- Committed.

**Hint.** Forgetting `.commit()` is the classic "it won't run" bug — call it. For the mismatch, change the second step's `inputSchema` to expect a field the first step doesn't produce and watch the `.then()` type-check fail.

**Estimated time.** 45 minutes.

---

## Problem 3 — Prove resume-from-step-N

**Problem statement.** Take Exercise 3's durable pipeline (or your mini-project's `workflow.ts`). Run it with a simulated crash after step 2, then re-run, and produce a captured trace showing steps 1–2 **replayed from cache** and the run **resuming at step 3**, ending in `PASS: it resumed from step 3`. Then re-run with the crash after step 1 and after step 3, and confirm the resume point moves accordingly.

**Acceptance criteria.**

- `notes/week-14/resume.md` with captured traces for crash-after-1, crash-after-2, and crash-after-3, each showing the right replayed-from-cache steps and resume point.
- A one-sentence explanation of *why* the completed steps don't re-run (the step record, not the function body, is the source of truth).
- A note on what happens if you delete the step store between attempts (the run starts over — proving the store *is* the durability).
- Committed.

**Hint.** Run `npx tsx exercise-03-durable-resume.ts --simulate-crash-after 2`, then `1`, then `3`. The replayed-from-cache count should equal the crash-after number. If steps re-run when they shouldn't, a step id collided or a side-effect leaked outside a step.

**Estimated time.** 40 minutes.

---

## Problem 4 — The Mastra-vs-LangGraph + Inngest resume memo (headline deliverable)

**Problem statement.** This is the syllabus lab. Build the same supervisor in **Mastra** (wired to **Inngest** for event-driven, durable invocation) and reference your **week-13 LangGraph** supervisor for the Python side. Crash both mid-run and resume each. Write a **one-page** memo at `notes/week-14/polyglot-memo.md` against this template:

1. **Recommendation** — one sentence: which stack you'd ship for a stated use case, and *which axis* drove it.
2. **The comparison table** — Mastra+Inngest vs LangGraph+checkpointer across: lines to a working supervisor, type-safety (compile-time vs runtime), where a shape mismatch is caught, durability mechanism, resume granularity, what the crash test took, deploy-target fit, ecosystem proximity.
3. **The resume story, both sides** — how each resumed after the crash (Inngest step memoization vs LangGraph checkpointer), and a resume trace for the Mastra side in the promise format (`it resumed from step N`).
4. **Where durability comes from** — one paragraph: durability is the *execution engine*, not the agent framework; Mastra is ergonomics, Inngest is durability.
5. **The honest trade** — what you gave up on each side (TS: ecosystem proximity; Python: compile-time type safety / edge fit).

**Acceptance criteria.**

- `notes/week-14/polyglot-memo.md` exists, fits roughly one page (350–550 words), and hits all five headings.
- The Mastra side is genuinely wired to Inngest (event-triggered) and **resumes from the failed step** on a crash (completed steps replayed from cache); the LangGraph side resumes from its checkpoint.
- Every Mastra+Inngest **side-effect is inside a `step.run`** and steps are **idempotent** (deterministic persist key) — stated and true.
- The recommendation names the *driving axis*, not "I like TypeScript."
- At least one resume trace in the promise format.
- Committed.

**Hint.** Don't reimplement the LangGraph supervisor — point at week 13's. The crash test is the spine of the memo: if your Mastra "crashed" run re-pays for completed steps on restart, you fell in the challenge's trap (a side-effect outside a step, or units that aren't `step.run`s). Run the Inngest dev server and watch the replay in the UI to capture the trace.

**Estimated time.** 1 hour.

---

## Problem 5 — Idempotency, broken then fixed

**Problem statement.** In a durable step, write a `persist` step two ways: first **non-idempotently** (key the output on `Date.now()`), then **idempotently** (key on `runId`). Force a replay of that step (crash after it, or call it twice) and show that the non-idempotent version produces a *duplicate* output while the idempotent version overwrites harmlessly. Capture both results.

**Acceptance criteria.**

- `notes/week-14/idempotency.md` showing the non-idempotent step producing two outputs on replay, and the idempotent step producing one.
- A one-sentence statement of the rule: a step may be attempted more than once, so its destination must be deterministic (or use an idempotency key).
- The same demonstration for a *read-only* gather step (safe to replay by nature) for contrast.
- Committed.

**Hint.** `reports/${Date.now()}.md` makes a new file every attempt; `reports/${runId}.md` overwrites the same one. The lesson is Lecture 2 §5 made visible: idempotency is the price of durability, and you pay it on every step that touches the outside world.

**Estimated time.** 45 minutes.

---

## Problem 6 — Survey Trigger.dev and Temporal honestly

**Problem statement.** Without building a full system, sketch how the research run would look in **Trigger.dev** (TS-native background task) *or* **Temporal** (workflow + activities), and write a short comparison to the Inngest version: the durability mechanism, the developer ergonomics, and the operational weight. State, in one sentence each, when you'd reach for Inngest, Trigger.dev, and Temporal.

**Acceptance criteria.**

- `notes/week-14/durability-engines.md` with a sketch (pseudocode is fine) of the research run in Trigger.dev *or* Temporal.
- A three-row comparison (Inngest / Trigger.dev / Temporal) on durability mechanism, language reach, and operational weight.
- A one-sentence "reach for it when…" for each of the three.
- One sentence connecting all three to the single durability spine (persist what finished, replay, skip the done, resume at the first incomplete unit).
- Committed.

**Hint.** Temporal's *activity* is the analogue of Inngest's `step.run`; the workflow body is deterministic orchestration. The honest contrast is operational weight: Inngest's local dev server vs Temporal's cluster + workers. You're not building it — you're showing you know *when* each is the right pick (Lecture 2 §7–8).

**Estimated time.** 30 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Scaffold a Mastra agent with a typed tool | 40 min |
| 2 — Two-step workflow with a checked seam | 45 min |
| 3 — Prove resume-from-step-N | 40 min |
| 4 — Mastra-vs-LangGraph + Inngest resume memo (headline) | 1 h 0 min |
| 5 — Idempotency, broken then fixed | 45 min |
| 6 — Survey Trigger.dev and Temporal | 30 min |
| **Total** | **~4 h 20 min** |

When you've finished all six, push your repo and make sure the `crunchagent-durable` [mini-project](./mini-project/README.md) is in the same workspace — week 15 plugs MCP tools into your durable supervisor. Then take the [quiz](./quiz.md) with your notes closed.
