# Challenge 1 — The Polyglot Supervisor

**Time estimate:** ~150 minutes.

## Problem statement

You have an agent architecture — a supervisor that routes a research task to one of two sub-agents — and a decision to make: which stack do you ship it on? Two reasonable engineers on your team disagree. One swears by Python and LangGraph (week 13); the other swears by TypeScript and Mastra. You are going to end the argument the only honest way it can end: **build the same supervisor in both stacks, wire one of them to a durable execution engine (Inngest) for event-driven invocation, crash it on purpose, and let the evidence pick the winner — on each axis separately.**

This is the syllabus deliverable in lab form. The output is not "TypeScript is better" or "Python is better." It's a **scored comparison** across three axes — developer ergonomics, type safety, and the resume-after-crash story — plus a recommendation that names *which axis* drove the decision for *this* use case.

## What is fixed (do not let these vary)

- **The architecture.** Both implementations are the *same* supervisor: a router that classifies a task into `research` or `math` and delegates to the matching sub-agent. Same sub-agents, same routing logic, same task set. Only the *stack* varies.
- **The model.** Both use `claude-sonnet-4-6` for the sub-agents (frontier path) — Mastra via `@ai-sdk/anthropic`, LangGraph via the Anthropic Python client. (Open-weights path: both point at the same local vLLM/Ollama endpoint. Hold the model constant so the comparison is *stack*, not *model*.)
- **The durability target.** "Resume from the failed step after a crash." Both stacks must be able to demonstrate it: LangGraph via its checkpointer (week 13), Mastra-wired-to-Inngest via step memoization (this week).
- **The crash test.** Same crash point (after step 2 of the research run) for both, so the resume story is compared on equal terms.

## What varies (this is what you're measuring)

- **The stack:** Mastra (TypeScript) vs LangGraph (Python).
- **The durability engine:** Inngest (for Mastra) vs the LangGraph checkpointer (for the Python side). Note honestly that these are *different engines*, not the same engine in two languages — that's part of the comparison.
- **Developer ergonomics, type safety, and the resume mechanics** — the three axes you score.

## The harness approach

### Side A — the Mastra supervisor, wired to Inngest

Build the supervisor as a Mastra workflow (or the hand-rolled equivalent from Exercise 2), then express the *research run* as an Inngest function so it's event-triggered and durable. The shape:

```typescript
import { Inngest } from "inngest";
import { Agent } from "@mastra/core/agent";
import { anthropic } from "@ai-sdk/anthropic";

export const inngest = new Inngest({ id: "polyglot-supervisor" });

const researchAgent = new Agent({
  name: "research-sub-agent",
  instructions: "Answer factual research questions, grounded and concise.",
  model: anthropic("claude-sonnet-4-6"),
});
const mathAgent = new Agent({
  name: "math-sub-agent",
  instructions: "Solve quantitative problems; show the calculation.",
  model: anthropic("claude-sonnet-4-6"),
});

// The durable, event-triggered research run. Each step.run is a durable checkpoint;
// every side-effect lives INSIDE a step (Lecture 2 §5). The supervisor's route
// decision is itself a step so it's memoized on replay.
export const supervisorRun = inngest.createFunction(
  { id: "supervisor-run", retries: 3 },
  { event: "research/requested" }, // an S3 new file becomes this event
  async ({ event, step }) => {
    const route = await step.run("route", async () => {
      const t = event.data.task as string;
      return /\b(calculate|sum|percent|how much|\d+%)\b/i.test(t) ? "math" : "research";
    });

    const plan = await step.run("plan", async () => {
      const agent = route === "math" ? mathAgent : researchAgent;
      const r = await agent.generate(`Plan how to answer: ${event.data.task}`);
      return r.text;
    });

    const sources = await step.run("gather-sources", async () => {
      // hits external APIs — idempotent because it only READS
      return gatherSources(plan);
    });

    const report = await step.run("synthesize", async () => {
      const agent = route === "math" ? mathAgent : researchAgent;
      const r = await agent.generate(`Using these sources, answer: ${event.data.task}\n${sources}`);
      return r.text;
    });

    const location = await step.run("persist", async () => {
      // DETERMINISTIC key -> idempotent: a replay overwrites harmlessly
      return writeToS3(`reports/${event.data.runId}.md`, report);
    });

    return { route, location };
  },
);
```

Trigger it from the **S3-new-file** path (or a local equivalent — a watched folder, or a manual `inngest.send`):

```typescript
// The bridge: a new file in S3 -> an Inngest event -> the durable run.
export async function onS3File(bucket: string, key: string) {
  await inngest.send({
    name: "research/requested",
    data: { runId: key.replace(/\W+/g, "-"), task: deriveTaskFromKey(key), bucket, key },
  });
}

// Local equivalent for the lab (no AWS needed): just send the event directly.
await inngest.send({
  name: "research/requested",
  data: { runId: "doc-42", task: "confidentiality clause durations" },
});
```

Run the Inngest dev server (`npx inngest-cli@latest dev`, UI at `localhost:8288`), fire the event, then **kill the worker process mid-run** (or throw inside `synthesize`). On restart, the run replays: `route`, `plan`, `gather-sources` come back from cache; `synthesize` runs fresh; the run resumes from step 4. Watch it in the UI.

### Side B — the LangGraph supervisor (week 13, Python)

You already built this. The same supervisor lives in your week-13 repo: a `StateGraph` with a router node, two sub-graphs, and a checkpointer. **Point at it** — don't rebuild it. Its durability story is the checkpointer: persisted graph state you resume from after a process kill (week 13's resume-after-crash demo). For the comparison, run *its* crash test (kill the process mid-run, resume from the checkpoint) and record what it took.

> See your week-13 `challenge` / `mini-project` for the LangGraph supervisor and its checkpointed resume. This challenge does **not** reimplement it; it *compares against* it.

### The comparison

Fill this table from your own hands-on experience, not from blog posts:

| Axis | Mastra + Inngest (TS) | LangGraph + checkpointer (Python) |
|---|---|---|
| **Lines to a working supervisor** | _measure_ | _measure_ |
| **Type safety at the step seams** | compile-time (zod + TS) | runtime (Pydantic/TypedDict) |
| **Catch a shape mismatch when?** | at the keystroke / build | at runtime, on the unlucky request |
| **Durability mechanism** | Inngest step memoization | LangGraph checkpointer |
| **Resume granularity** | per `step.run` | per node |
| **What it took to crash-test** | _record_ | _record_ |
| **Deploy target fit** | edge / full-stack TS | Python service near ML/RAG |
| **Ecosystem proximity (embeddings, rerankers, evals)** | HTTP hop to Python | native |

## Acceptance criteria

- [ ] A `challenge-01/` directory with a runnable Mastra-supervisor-wired-to-Inngest (`supervisor.ts` + an event-trigger script), and a pointer to the week-13 LangGraph supervisor.
- [ ] The Mastra side runs against the Inngest dev server; firing the `research/requested` event starts the durable run, and you can show the steps in the UI.
- [ ] A **crash test on both sides**: the Mastra+Inngest run is killed mid-run and **resumes from the failed step** (completed steps replay from cache); the LangGraph run is killed and resumes from its checkpoint. Both are demonstrated, not asserted.
- [ ] Every Mastra+Inngest **side-effect lives inside a `step.run`**, and the persist step uses a **deterministic key** (idempotent). No `fetch`/write in the bare function body.
- [ ] A one-page `polyglot-memo.md` that fills the comparison table from your own runs and names a **recommendation** — which stack you'd ship for a stated use case, and *which axis* drove it (e.g. "edge deploy + type safety → Mastra" or "lives next to a Python RAG pipeline → LangGraph").
- [ ] At least one **resume trace** in the promise format for the Mastra side: `it resumed from step N`, showing the completed steps replayed-from-cache.

## The trap (read after a first attempt)

The trap is **conflating Mastra's ergonomics with durability.** Mastra makes the *agent* beautiful — typed tools, clean workflows, nice `generate()` calls. None of that is durability. If you write the research run as a plain `async` function calling `researchAgent.generate()` four times and *don't* wrap each call in an Inngest `step.run`, then a crash mid-run re-runs every model call from the top — and the gorgeous Mastra code lost all its work. **Durability comes from Inngest (the execution engine), not from Mastra (the agent framework).** The proof: your crash test must show steps replaying *from cache*, which only happens if each unit is a `step.run`. If your "crashed" run re-pays for `plan` on restart, you fell in the trap.

A second trap: **doing side-effects outside a step.** A `fetch` or an S3 write sitting in the bare function body between two `step.run`s re-executes on *every* replay — because nothing memoizes it. So a crash makes it run twice (or N times). Every side-effect goes *inside* a step, and every step is idempotent (deterministic destination / idempotency key). Skip this and your "durable" agent re-hits external systems on every retry, which is *worse* than the naive version.

A third, subtler trap: **a non-idempotent step.** If `persist` writes to `reports/${Date.now()}.md` instead of `reports/${runId}.md`, then a replay of that step writes a *second* file — durable replay turned into duplicate output. Idempotency is the price of durability; pay it on every step that touches the outside world.

## Stretch goals

- **The Temporal version.** Express the same supervisor as a Temporal workflow with the model/tool calls as **activities**. The durability story is identical (activities are persisted and replayed); the operational weight is much higher (a cluster, workers, the Temporal SDK). Record what it cost you to stand up — that contrast *is* the "when is Temporal the right pick" lesson.
- **A Trigger.dev background job.** Re-express the research run as a Trigger.dev task and compare its durable-step model and developer ergonomics to Inngest's. Where do they agree (durable steps, retries)? Where do they differ (event model, deploy)?
- **Real S3.** Wire a real S3 bucket notification (via SQS/EventBridge → an API route that calls `inngest.send`) so dropping a file in the bucket genuinely starts a run. Kill the worker mid-run; confirm it resumes on restart.
- **Type-safety stress test.** Add a sub-agent tool that returns a shape the workflow doesn't expect. Watch TypeScript reject it at build time; then break the equivalent in the LangGraph version and watch it fail at runtime. That delta is the type-safety axis, measured.

## Why this matters

Polyglot durability is *the* production-agent skill. In the serving and capstone weeks (19–24) you will run agents for real, where crashes are routine and a non-durable run is a money leak and a duplicate-action bug waiting to happen. The engineer who can express an architecture in two stacks — and who scores *agent ergonomics* and *durability* as separate axes instead of buying a framework as a bundle — is the one who picks the right tool for the deploy and ships an agent that survives its first OOM kill. You built the same supervisor twice, crashed it, and watched it resume from the failed step. You can name the winner, on each axis, with a crash test behind the claim. That's the difference between a demo and production.
