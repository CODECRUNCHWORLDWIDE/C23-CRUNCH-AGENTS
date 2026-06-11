# Week 14 — Mastra, Inngest, and TypeScript Agent Stacks

Welcome to the week your agent learns to survive a crash. Last week you built a supervisor in **LangGraph** (Python) and gave it a SQLite checkpointer so it could resume after a process kill. This week you build *the same supervisor* in **Mastra** (TypeScript), wire it to **Inngest** for event-driven, durable execution, and crash it on purpose — then watch it pick up from the exact step it died on, without re-running the work it already finished. By Friday you will be able to look at any agent stack, in either language, and answer the only question that matters in production: **if the process dies at step 7, does it resume at step 7, or does it start over at step 1?**

This is week 2 of **Phase III — Agents & Orchestration**, and it is the deliberate TypeScript counterpart to week 13. Everything you learned about supervisors, sub-agents, routers, and checkpointed resume in Python carries over — the *concepts* are language-agnostic. What changes is the stack, the ergonomics, and the type system. The whole point of this week is **polyglot agent design**: the same architecture expressed in two ecosystems, so you can pick the right one for a given job instead of defaulting to whatever language you happen to know.

The one sentence to internalize before you read another line:

> **Your agent platform is your durability platform. If it cannot resume from step 7 after a crash, it is not production — it is a demo that hasn't crashed yet.**

Here is why that's not hyperbole. An agent is a sequence of expensive, side-effecting steps: call a model, hit a tool, write to a store, call the model again. Each step costs money, latency, and sometimes an irreversible external action (a payment, an email, an S3 write). When the process dies mid-run — and it *will*, because deploys, OOM kills, spot-instance preemptions, and rate-limit timeouts are facts of life — a non-durable agent restarts from zero: it re-pays for every model call, re-runs every tool, and re-sends every email. A **durable** agent remembers which steps completed, skips them on replay, and resumes from the first incomplete one. That memory — Inngest's step memoization, LangGraph's checkpointer, Temporal's activity history — is the difference between an agent you can run for real and a toy.

There's a corollary worth taping next to last week's mantra:

> **Durability does not come from your agent framework. It comes from your execution engine.** Mastra makes the agent *ergonomic*; Inngest makes it *durable*. Conflate the two and you will ship something that reads beautifully and loses all its work on the first crash.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** why a TypeScript-first agent stack exists at all — full-stack and edge deployment, one language front-to-back, and end-to-end type safety from the API route to the agent's tool schema — and when that beats Python's ML-ecosystem proximity.
- **Build** a Mastra agent: `instructions`, a `model` wired through the Vercel AI SDK's `@ai-sdk/anthropic` provider, and typed `tools`, then call it and narrow its typed output.
- **Compose** a Mastra **workflow** from typed steps (`createStep` with zod input/output schemas, `.then()` chaining, `.commit()`) and build a **supervisor agent** that routes a task to one of several sub-agents.
- **Explain** durable execution mechanically — how Inngest memoizes each completed `step.run` result and *skips it on replay* so a crashed function resumes from the first incomplete step rather than from the top.
- **Wire** an agent to **event-driven invocation**: a new file in S3 emits an event, the event triggers a durable research run, and `inngest.send` / `inngest.createFunction` carry the work.
- **Demonstrate** resume-after-crash: implement a multi-step pipeline, crash it after step 2, re-run it, and *prove* that steps 1–2 were replayed from cache and the run resumed at step 3.
- **Compare**, honestly, the Python-first (LangGraph) and TypeScript-first (Mastra) stacks — where each wins, where each is a stretch — and survey **Trigger.dev** and **Temporal** as the heavier-duty neighbors, knowing when each is the right pick.
- **Apply** the two durability disciplines that make replay safe: **steps must be idempotent**, and **side-effects only happen inside steps**.

## Prerequisites

This week assumes you have completed **C23 weeks 1–13**, or have equivalent fluency. Specifically:

- You finished **week 13** and have the **LangGraph supervisor** (Python): a `StateGraph` with plan/retrieve/execute/critique nodes, a checkpointer, and resume-after-kill. **This week rebuilds that supervisor in TypeScript** and references the Python version directly — have it open.
- **Node.js 20+** (Node 22 LTS is ideal) and `npm`. You will run a TypeScript file with `tsx` and run the **Inngest dev server** locally (`npx inngest-cli@latest dev`).
- **Enough TypeScript to read this week's lab.** The course does *not* assume you write production TypeScript. It assumes you can read a typed function signature, a zod schema, and an `async/await` call. Every code block is annotated; if you can follow week 13's Python, you can follow this week's TypeScript.
- An **Anthropic API key** (`ANTHROPIC_API_KEY`) for the frontier path, *or* a local OpenAI-compatible endpoint (vLLM/Ollama) for the open-weights path. Every lab has a **deterministic offline fallback** so it runs with neither.

You do **not** need a GPU. You do **not** need an AWS account — the S3 trigger has a local equivalent (a watched folder or a manually-sent event) documented in every lab.

## Topics covered

- **Why TypeScript-first:** full-stack/edge deployment, one language from the Next.js route to the agent's tool, and end-to-end type safety; the honest trade against Python's ML/RAG ecosystem proximity.
- **Mastra agents:** `new Agent({ name, instructions, model, tools })`, the `@ai-sdk/anthropic` model wiring (`anthropic("claude-sonnet-4-6")`), typed tools, and calling the agent.
- **Mastra workflows:** `createWorkflow` / `createStep` with zod-typed input/output, `.then()` chaining, `.commit()`, and the supervisor-as-workflow pattern that routes to sub-agents.
- **Mastra memory and evals:** `@mastra/memory` for agent memory, and scorers/evals for measuring agent quality (survey + one concrete example).
- **Durable execution with Inngest:** `inngest.createFunction`, `step.run` memoization, the crash/retry/replay model, and *why* completed steps are skipped on replay (resume from step 7).
- **Event-driven invocation:** `inngest.send({ name, data })`, an S3-new-file event → a research run, and the local equivalent for the lab.
- **The durability spine across stacks:** LangGraph's checkpointer ≈ Inngest's step memoization ≈ Temporal's activities — the same idea in three frameworks.
- **The neighbors, honestly:** **Trigger.dev** (TS-native background jobs) and **Temporal** (heavyweight, language-agnostic, battle-tested durable workflows) at survey level, and when each is the right pick.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                          | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|---------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Why TS-first; Mastra agents, tools, AI SDK model wiring       |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Mastra workflows, steps, the supervisor; exercises 1–2        |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Durable execution: Inngest steps, memoization, resume         |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Event-driven invocation; Trigger.dev/Temporal survey          |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The polyglot lab: Mastra-vs-LangGraph + Inngest resume        |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                        |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                     |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                               | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | Mastra/Inngest/Trigger.dev/Temporal docs, the AI SDK + `@ai-sdk/anthropic` references, the LangGraph-vs-Mastra framing, and a glossary cheat-sheet |
| [lecture-notes/01-mastra-and-the-typescript-agent-stack.md](./lecture-notes/01-mastra-and-the-typescript-agent-stack.md) | Why TS-first, Mastra agents/workflows/steps/memory/evals, the supervisor in Mastra, and the honest LangGraph-vs-Mastra comparison |
| [lecture-notes/02-durable-execution-and-event-driven-agents.md](./lecture-notes/02-durable-execution-and-event-driven-agents.md) | Inngest steps + memoization + resume-from-step-7, event-driven invocation (S3 → event → research run), Trigger.dev/Temporal survey, the shared durability spine |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises + how to run them |
| [exercises/exercise-01-scaffold-a-mastra-agent.md](./exercises/exercise-01-scaffold-a-mastra-agent.md) | Scaffold a minimal Mastra agent (instructions + `@ai-sdk/anthropic` + one tool), call it, observe the typed output, then add a second tool and a workflow step |
| [exercises/exercise-02-mastra-supervisor.ts](./exercises/exercise-02-mastra-supervisor.ts) | Build a Mastra-style supervisor that routes a task to one of two sub-agents and prints the routing trace (offline fallback included) |
| [exercises/exercise-03-durable-resume.ts](./exercises/exercise-03-durable-resume.ts) | A multi-step memoized pipeline: crash after step 2, re-run, and prove steps 1–2 replayed-from-cache and the run resumed at step 3 |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge (the syllabus polyglot lab) |
| [challenges/challenge-01-polyglot-supervisor.md](./challenges/challenge-01-polyglot-supervisor.md) | The same supervisor in Mastra (TS) and LangGraph (Python, week 13), wired to Inngest for event-driven invocation; compare ergonomics, type safety, and the resume-after-crash story |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the Mastra-vs-LangGraph + Inngest resume-after-crash memo |
| [mini-project/README.md](./mini-project/README.md) | `crunchagent-durable` — a reusable Mastra + Inngest durable-agent package: supervisor + event-triggered, resumable research workflow + a budget |

## The "it resumed from step 7" promise

C23 uses a recurring marker for every exercise that ends in durability actually *working* — the agent crashing and picking up exactly where it died, not re-running the work it already finished:

```
$ npx tsx exercise-03-durable-resume.ts --simulate-crash-after 2
run abc123 :: attempt 1
  [step 1] plan            RAN FRESH  -> "3 sub-questions"
  [step 2] gather-sources  RAN FRESH  -> "7 sources"
  [step 3] synthesize      💥 CRASH (simulated)
run abc123 :: attempt 2 (replay)
  [step 1] plan            REPLAYED FROM CACHE  (not re-run)
  [step 2] gather-sources  REPLAYED FROM CACHE  (not re-run)
  [step 3] synthesize      RAN FRESH  -> "report (1,240 words)"
  [step 4] persist         RAN FRESH  -> "s3://reports/abc123.md"
PASS: it resumed from step 3 — steps 1–2 were memoized and skipped on replay.
```

If steps 1 and 2 *re-run* on attempt 2 — re-calling the model, re-paying, re-hitting the tool — then your "durable" agent is not durable; it just hasn't lost enough work yet for you to notice. The point of week 14 is to make resume-from-step-N *measurable*: the steps that already finished print `REPLAYED FROM CACHE`, the run resumes at the first incomplete one, and the `PASS` line names the step it resumed from. That is durability you can prove, not durability you hope for.

## Stretch goals

If you finish the regular work early and want to push further:

- **Build the Temporal version.** Express the same supervisor as a Temporal workflow with the model/tool calls as **activities**. Notice that the durability story is *identical* (activity results are persisted and replayed) but the operational weight is much higher (a Temporal cluster, workers, the Temporal SDK). That contrast *is* the "when is Temporal the right pick" lesson, felt in your hands.
- **Wire a real S3 trigger.** Point Inngest at a real S3 bucket notification (via an SQS/EventBridge → Inngest bridge or an API route that receives the S3 event) so dropping a file into the bucket genuinely kicks off a research run. Then kill the worker mid-run and confirm the run resumes on restart.
- **Add a Trigger.dev background job.** Re-express the research run as a Trigger.dev task and compare its developer ergonomics and durability model to Inngest's. Where do they agree (durable steps, retries)? Where do they differ (event model, deployment)?
- **Type-safety stress test.** Add a sub-agent whose tool returns a shape the workflow doesn't expect, and watch the TypeScript compiler reject it *before* you run it. Then break the equivalent in the Python version and watch it fail at *runtime*. That delta is the type-safety argument, measured.

## Up next

Week 15 takes the agents you can now build *and resume* in two languages and gives them a **standard way to talk to tools**: **MCP — the cross-vendor tool protocol.** You'll stop hand-wiring each tool into each framework and instead expose tools over MCP so any agent — Mastra, LangGraph, or a Claude client — can call them through one protocol. The durability spine you built this week carries forward: an MCP tool call is still a *step*, and a step still has to be idempotent and replayable. Push your mini-project before you start week 15; the durable supervisor is the thing MCP plugs tools *into*.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
