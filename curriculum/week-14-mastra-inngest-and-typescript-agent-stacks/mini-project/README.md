# Mini-Project — `crunchagent-durable`: A Reusable Durable-Agent Package

> Build a reusable TypeScript package that wraps a **Mastra supervisor agent** in an **Inngest-driven, event-triggered, durably-resumable research workflow** with a **budget** — so any project can `import { runResearch } from "crunchagent-durable"`, drop a file in a bucket (or send an event), and get a research run that survives a crash, resumes from the failed step, and refuses to overspend.

This is the artifact that turns this week's lessons into something you depend on. After this week, kicking off a durable agent run is `inngest.send({ name: "research/requested", data })` and watching it resume after a crash — not a one-off script that loses all its work the first time the process dies. The package is supervisor-routed, event-triggered, budget-bounded, and — the point of the whole week — **durable**: every unit of work is an idempotent `step.run`, so a crash resumes from step N instead of restarting at step 1.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** the durable supervisor here is the thing **week 15 (MCP)** plugs tools into — an MCP tool call is just another `step.run`, idempotent and replayable. The same durability spine recurs in **week 18 (observability)**, where you trace these steps, and in **weeks 19–24 (production & capstone)**, where this supervisor scales up into the capstone's serving story. The week-13 LangGraph supervisor is its Python counterpart; this is the TypeScript half of your polyglot toolkit. Build it well now.

---

## What you will build

A small TypeScript package `crunchagent-durable` with four deliverables:

1. **`src/agents.ts`** — the Mastra supervisor: a router plus two sub-agents (research, math), each a `new Agent({ instructions, model, tools })` wired through `@ai-sdk/anthropic`. The agent layer — ergonomic, typed, and *not* where durability lives.
2. **`src/workflow.ts`** — the Inngest durable research function: `inngest.createFunction` whose body is a sequence of `step.run` units (route → plan → gather → synthesize → persist), each idempotent, each a durable checkpoint. The durability layer.
3. **`src/budget.ts`** — a budget guard: a per-run token/dollar ceiling that each step checks before doing expensive work, so a runaway agent stops instead of draining the account. The safety layer.
4. **`src/index.ts` + a CLI / event-sender** — the public API (`runResearch`) and a way to trigger a run: a CLI that sends the event, plus the S3-new-file bridge (and a local watched-folder / manual-send equivalent).

By the end you have a package of ~400–500 lines of TypeScript that any project can install, point at a model (Claude or a local open model), trigger with an event, and trust to resume after a crash without re-paying for completed steps.

---

## Why a package and not a script

You could do all of this in one `index.ts`. Don't — not as the artifact. A package gives you:

- **Reuse.** Week 15 imports your durable supervisor to plug MCP tools into; the capstone scales it up. A script gets copy-pasted, drifts, and rots.
- **A durability contract in code.** "Every side-effect is inside a `step.run`, every step is idempotent" is enforced by the structure of the package (steps are functions in `workflow.ts`), version-controlled, and testable — not a discipline you hope you remembered.
- **A CLI + an event surface.** `crunchagent send --task "..."` and the S3 bridge are scriptable and CI-able. They're how the durable run gets *triggered* in production.

The senior-shop convention in 2026: the thing you depend on is a typed package with a durability contract; the notebook is for exploring.

---

## Package layout

```
crunchagent-durable/
├── package.json                # deps: @mastra/core, @ai-sdk/anthropic, inngest, zod
├── tsconfig.json               # "module": "NodeNext", "strict": true, target ES2022
├── .env.example                # ANTHROPIC_API_KEY=... (or a local model endpoint)
├── README.md                   # run commands + the resume-after-crash demo
├── src/
│   ├── index.ts                # public API: runResearch(), the inngest client
│   ├── agents.ts               # the Mastra supervisor: router + 2 sub-agents
│   ├── workflow.ts             # the Inngest durable function (the step.run chain)
│   ├── budget.ts               # the per-run budget guard
│   ├── triggers.ts             # S3-new-file bridge + local watched-folder/manual send
│   └── cli.ts                  # `crunchagent send|serve` commands
└── test/
    ├── workflow.resume.test.ts # crash after step 2, assert resume from step 3
    └── budget.test.ts          # a run that exceeds the budget stops cleanly
```

The runtime is **Node 20+** (Node 22 LTS recommended). You run a `.ts` file with `tsx`, run the durable function under the **Inngest dev server**, and run tests with your test runner of choice (`vitest`/`node --test`).

---

## Run commands

```bash
# Install (frontier path):
npm install @mastra/core @ai-sdk/anthropic inngest zod
npm install -D tsx vitest

# Set the model key (or point at a local vLLM/Ollama endpoint for the open path):
cp .env.example .env   # then edit ANTHROPIC_API_KEY=...

# Terminal 1 — the Inngest dev server (event bus + runner + UI at :8288):
npx inngest-cli@latest dev

# Terminal 2 — serve the function so the dev server can run it, then trigger it:
npx tsx src/cli.ts serve          # registers the durable function with the dev server
npx tsx src/cli.ts send --task "confidentiality clause durations"

# Watch the run in the UI at http://localhost:8288. Then KILL terminal 2 mid-run
# and restart `serve` — the run RESUMES from the failed step (completed steps
# replay from cache). That is the deliverable, demonstrated.
```

---

## Deliverable 1 — `agents.ts` (the Mastra supervisor)

The agent layer. A router plus two sub-agents, each behind `@ai-sdk/anthropic`. This is ergonomic and typed — and it is *not* where durability comes from (that's Deliverable 2).

```typescript
// src/agents.ts — the Mastra supervisor (router + sub-agents). Agent layer only.
import { Agent } from "@mastra/core/agent";
import { anthropic } from "@ai-sdk/anthropic";

export const researchAgent = new Agent({
  name: "research-sub-agent",
  instructions: "Answer factual research questions, grounded and concise. Cite what you used.",
  model: anthropic("claude-sonnet-4-6"), // EXACT id; no temperature/top_p (they 400)
});

export const mathAgent = new Agent({
  name: "math-sub-agent",
  instructions: "Solve quantitative problems. Show the calculation, then the answer.",
  model: anthropic("claude-sonnet-4-6"),
});

export type Route = "research" | "math";

// The router: classify a task into a route. In production a Haiku call; a keyword
// classifier here keeps tests deterministic. TODO 1: optionally upgrade to a
// claude-haiku-4-5 classifier behind a flag, keeping the deterministic default.
export function route(task: string): Route {
  return /\b(calculate|sum|average|percent|how much|how many|\d+%)\b/i.test(task) ? "math" : "research";
}

export function agentFor(r: Route): Agent {
  return r === "math" ? mathAgent : researchAgent;
}
```

---

## Deliverable 2 — `workflow.ts` (the Inngest durable function)

The durability layer. The research run as an Inngest function whose body is a chain of `step.run` units. **Every side-effect is inside a step; every step is idempotent.** This is the heart of the project.

```typescript
// src/workflow.ts — the durable, event-triggered research run. Durability layer.
import { Inngest } from "inngest";
import { route, agentFor } from "./agents.ts";
import { Budget } from "./budget.ts";

export const inngest = new Inngest({ id: "crunchagent-durable" });

export const research = inngest.createFunction(
  { id: "research-run", retries: 3 }, // step failures retry without re-running prior steps
  { event: "research/requested" },
  async ({ event, step }) => {
    const { runId, task, maxUsd } = event.data as { runId: string; task: string; maxUsd: number };
    const budget = new Budget(maxUsd);

    // Every unit is a step.run -> a durable checkpoint, skipped on replay.
    const decided = await step.run("route", async () => route(task));

    const plan = await step.run("plan", async () => {
      budget.assertUnder("plan"); // TODO 2: check budget BEFORE the expensive call
      const r = await agentFor(decided).generate(`Plan how to answer: ${task}`);
      budget.add(r.usage); // TODO 3: record token usage toward the budget
      return r.text;
    });

    const sources = await step.run("gather-sources", async () => {
      // idempotent: only READS external sources (safe to replay)
      return gatherSources(plan); // TODO 4: implement gatherSources (HTTP read or stub)
    });

    const report = await step.run("synthesize", async () => {
      budget.assertUnder("synthesize");
      const r = await agentFor(decided).generate(`Sources:\n${sources}\n\nAnswer: ${task}`);
      budget.add(r.usage);
      return r.text;
    });

    const location = await step.run("persist", async () => {
      // DETERMINISTIC key -> idempotent: a replay overwrites harmlessly
      return persistReport(`reports/${runId}.md`, report); // TODO 5: write (S3 or local file)
    });

    return { route: decided, location, usd: budget.spent() };
  },
);
```

The non-negotiables `workflow.ts` enforces:

- **Side-effects only inside `step.run`.** No `fetch`/write/model-call in the bare body. The bare body is re-walked on every replay; anything outside a step re-runs every retry (Lecture 2 §5).
- **Idempotent steps.** `gather-sources` only reads; `persist` writes to a deterministic key (`reports/${runId}.md`). A replay of any step is safe.
- **Step ids are unique and stable.** Steps are skipped-on-replay *by id*; two steps sharing an id collide, and renaming an id between deploys breaks resume for in-flight runs.

---

## Deliverable 3 — `budget.ts` (the safety layer)

A per-run ceiling. Each expensive step checks the budget *before* the model call and records usage *after*, so a runaway agent stops cleanly instead of draining the account.

```typescript
// src/budget.ts — a per-run token/dollar budget guard.
export class BudgetExceeded extends Error {}

// Rough per-MTok prices for claude-sonnet-4-6 (illustrative; check current pricing).
const USD_PER_INPUT_MTOK = 3.0;
const USD_PER_OUTPUT_MTOK = 15.0;

export class Budget {
  private usd = 0;
  constructor(private readonly maxUsd: number) {}

  // TODO 6: accumulate cost from a usage object { inputTokens, outputTokens }.
  add(usage: { inputTokens?: number; outputTokens?: number } | undefined): void {
    if (!usage) return;
    this.usd +=
      ((usage.inputTokens ?? 0) / 1e6) * USD_PER_INPUT_MTOK +
      ((usage.outputTokens ?? 0) / 1e6) * USD_PER_OUTPUT_MTOK;
  }

  assertUnder(stepName: string): void {
    if (this.usd >= this.maxUsd) {
      throw new BudgetExceeded(`budget $${this.maxUsd} exceeded before ${stepName} ($${this.usd.toFixed(4)})`);
    }
  }

  spent(): number {
    return Number(this.usd.toFixed(4));
  }
}
```

> **Note on durability + budget:** the budget is reconstructed from memoized step results on replay (completed steps already recorded their usage), so a resumed run doesn't double-count or re-pay. Keep the budget derivable from step results, not from a mutable global that a crash would lose.

---

## Deliverable 4 — `index.ts` + `triggers.ts` + `cli.ts` (the public surface)

The public API and the ways to trigger a run.

```typescript
// src/index.ts — the public API.
export { inngest, research } from "./workflow.ts";

import { inngest } from "./workflow.ts";

// Kick off a durable run by sending the event. Returns immediately; the run is
// durable and proceeds in the background (and resumes if it crashes).
export async function runResearch(task: string, opts?: { runId?: string; maxUsd?: number }) {
  const runId = opts?.runId ?? `run-${Date.now()}`;
  await inngest.send({
    name: "research/requested",
    data: { runId, task, maxUsd: opts?.maxUsd ?? 1.0 },
  });
  return { runId };
}
```

```typescript
// src/triggers.ts — the S3-new-file bridge + local equivalent.
import { inngest } from "./workflow.ts";

// A new file in S3 -> an Inngest event -> a durable run. (Wire to S3 notifications
// via SQS/EventBridge -> this handler, or a Lambda that calls it.)
export async function onS3File(bucket: string, key: string) {
  await inngest.send({
    name: "research/requested",
    data: { runId: key.replace(/\W+/g, "-"), task: deriveTask(key), maxUsd: 1.0, bucket, key },
  });
}

// TODO 7: a local watched-folder equivalent (fs.watch a dir; on a new file, send
// the same event) so the lab needs no AWS account.
```

The CLI ties it together: `serve` registers the function with the dev server; `send --task "..."` fires the event.

---

## Rules

- **You may** read the Mastra/Inngest docs, the lecture notes, and your week-13 LangGraph supervisor for the architecture.
- **You must not** put any side-effect (model call, HTTP, file/S3 write) **outside** a `step.run`. The bare function body is glue only. (This is the week's central discipline; violating it is the challenge's trap.)
- **You must** make every step **idempotent** — deterministic destinations, read-only gathers, idempotency keys for any external mutation — so replay is safe.
- **You must not** rename or reuse a `step.run` id once a run can be in flight against it; ids are how resume skips completed steps.
- **You must** reuse the week-13 supervisor *concepts* (router + sub-agents, the research decomposition) — this is the TS counterpart, not a from-scratch redesign.
- Node 20+, `@mastra/core`, `@ai-sdk/anthropic`, `inngest`, `zod`, plus a test runner. The model is `claude-sonnet-4-6` (frontier) or a local OpenAI-compatible model (open path); no `temperature`/`top_p`/`top_k`.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-14-crunchagent-durable-<yourhandle>`.
- [ ] `npx inngest-cli@latest dev` + `npx tsx src/cli.ts serve` register the `research` function; `send --task "..."` triggers a durable run visible in the UI.
- [ ] The research run is a chain of `step.run` units (route → plan → gather → synthesize → persist), **every side-effect inside a step**, **every step idempotent** (deterministic persist key).
- [ ] **Resume-after-crash, demonstrated:** kill the worker mid-run (or throw in `synthesize`); on restart the run **resumes from the failed step**, with completed steps **replayed from cache** (no re-pay). `test/workflow.resume.test.ts` asserts it.
- [ ] The **budget guard** stops a run that exceeds `maxUsd` cleanly (`BudgetExceeded`), and the budget is correctly reconstructed on replay (no double-count). `test/budget.test.ts` asserts it.
- [ ] The supervisor **routes** to the right sub-agent, and the router decision is itself a memoized step.
- [ ] A `README.md` with the run commands and a **resume trace** in the promise format (`it resumed from step N`, completed steps replayed-from-cache).
- [ ] Tests pass; no secrets committed; no `node_modules`/`.env` checked in.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Durability correctness** | 30 | Every side-effect inside a `step.run`; steps idempotent (deterministic persist key, read-only gather); resume-from-failed-step demonstrated and tested; completed steps replay from cache, no re-pay. |
| **Supervisor & agents** | 20 | Mastra supervisor routes to the right sub-agent; sub-agents wired via `@ai-sdk/anthropic` with correct model ids and no rejected sampling params; the router decision is a memoized step. |
| **Event-driven invocation** | 15 | `inngest.send`/`createFunction` wired correctly; the S3-new-file bridge (or local watched-folder/manual-send equivalent) starts a run; the event surface is clean. |
| **Budget guard** | 15 | Per-run ceiling checked before expensive steps; usage recorded; clean stop on exceed; budget correct on replay (no double-count). |
| **Tests** | 10 | `workflow.resume.test.ts` proves resume-from-step-N; `budget.test.ts` proves the ceiling; green. |
| **Docs & hygiene** | 10 | Clear README + run commands + resume trace; type-safe (`strict: true`); no secrets; sensible commits. |

**90+** is portfolio-grade and ready for week 15 to plug MCP tools into. **70–89** works but has a soft idempotency story or an untested resume. **Below 70** means it isn't actually durable (a side-effect leaked outside a step, or resume re-pays) — fix that first, because the rest of Phase III assumes this resumes.

---

## Stretch goals

- **Temporal port.** Re-express the research run as a Temporal workflow with the steps as activities. The durability spine is identical; record the operational weight delta — that *is* the "when Temporal" lesson.
- **Trigger.dev port.** Re-express the run as a Trigger.dev task; compare its durable-step model and ergonomics to Inngest's.
- **Open-weights end-to-end.** Run the whole stack (Mastra + Inngest + a local vLLM/Ollama model) with no cloud dependency, changing only the `model:` line. Confirm the durability behaves identically — durability is model-agnostic.
- **Suspend/resume for human-in-the-loop.** Add a `step.waitForEvent` so the run pauses for a human approval event, then resumes — durably — when the approval arrives. (Inngest persists across the wait, so an approval an hour later still resumes the exact run.)
- **CI.** A GitHub Actions workflow that runs the resume and budget tests headless on every push.

---

## How this connects to the rest of C23

- **Week 13 (LangGraph, Python)** gave you the supervisor architecture and the checkpointed resume; this is the **TypeScript counterpart**, with Inngest's step memoization as the durability spine instead of LangGraph's checkpointer — the two halves of your polyglot toolkit.
- **Week 15 (MCP)** plugs tools into *this* durable supervisor over a standard protocol; an MCP tool call is just another idempotent `step.run`.
- **Week 18 (observability)** traces these steps — each `step.run` is a span — so the durability you built here is also the thing you watch in production.
- **Weeks 19–24 (production & capstone)** scale this supervisor into the serving story; the "does it resume from step N?" discipline you proved here is exactly what a production agent has to guarantee.

When you've finished, push the repo and take the [quiz](../quiz.md).
