# Lecture 2 — Durable Execution and Event-Driven Agents

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can explain mechanically how Inngest memoizes each completed `step.run` and skips it on replay (resume from step 7), wire an agent to event-driven invocation (an S3 new file → an event → a research run), state the crash/retry/replay model, survey Trigger.dev and Temporal and say when each is the right pick, and explain why LangGraph's checkpointer, Inngest's step memoization, and Temporal's activities are the *same durability spine*.

Lecture 1 built an agent that's ergonomic and typed and — if you kill the process mid-run — completely amnesiac. It re-pays for every model call, re-runs every tool, re-sends every email, from the top. This lecture fixes that, and it's the heart of the week.

> **Your agent platform is your durability platform. If it cannot resume from step 7 after a crash, it is not production.**

That sentence is the whole point of Phase III's production thread. An agent is a sequence of expensive, side-effecting steps. In the real world the process *will* die mid-run — deploys, OOM kills, spot-instance preemption, a 30-second rate-limit timeout, a `kill -9` from an autoscaler. The only question that matters is what happens next: does the run resume from where it died, or start over? **Durable execution** is the machinery that makes the first answer true, and — the lesson you must not miss — **it does not come from Mastra.** It comes from the execution engine. This week the engine is **Inngest**.

---

## 1. The problem, stated precisely

Take the research run from Lecture 1: four steps — `plan`, `gather-sources`, `synthesize`, `persist`. Each is expensive: `plan` and `synthesize` are Claude calls (money + latency), `gather-sources` hits external APIs (rate limits), `persist` writes to S3 (an irreversible side-effect). Suppose the process dies during `synthesize`.

A **naive** agent — a plain `async` function with four `await`s — has no memory of what finished. On restart it runs `plan` again (pay again), `gather-sources` again (pay again, re-hit the rate-limited API), then finally reaches `synthesize`. You paid for `plan` and `gather` *twice*, and if `persist` had already run before a *later* crash, you'd write the report to S3 *twice*. Multiply by every crash in production and you have an agent that's expensive, slow, and occasionally double-acting.

A **durable** agent remembers. It records, after each step, "this step finished, here is its result." On restart it reads that record, *skips* `plan` and `gather` (returning their recorded results instantly, with no re-execution), and resumes at `synthesize` — the first step that didn't finish. That's resume-from-step-7. The machinery that does the recording-and-skipping is the entire subject of this lecture.

---

## 2. Inngest functions and steps

Inngest is an open-source, event-driven durable-execution engine with a local dev server (`npx inngest-cli@latest dev`) so you can run the whole thing — event bus, runner, UI — with no cloud account.

You create a client, then a function:

```typescript
import { Inngest } from "inngest";

export const inngest = new Inngest({ id: "agents" });

export const research = inngest.createFunction(
  { id: "research-run" },
  { event: "research/requested" }, // this function runs when this event fires
  async ({ event, step }) => {
    // Each step.run is a DURABLE checkpoint. The string id names the step;
    // the async fn is its body. The RESULT is what gets memoized.
    const plan = await step.run("plan", async () => {
      return await makePlan(event.data.topic); // a Claude call, say
    });

    const sources = await step.run("gather-sources", async () => {
      return await gatherSources(plan); // hits external APIs
    });

    const report = await step.run("synthesize", async () => {
      return await synthesize(plan, sources); // another Claude call
    });

    const location = await step.run("persist", async () => {
      return await writeToS3(report); // the irreversible side-effect
    });

    return { location };
  },
);
```

The anatomy:

- **`createFunction({ id }, trigger, handler)`** — `id` names the function; the trigger (`{ event: "research/requested" }`) says *when* it runs; the handler is the body.
- **The handler receives `{ event, step }`.** `event` is the triggering event (with its `data`). `step` is the durability primitive.
- **`step.run("name", async () => {...})`** — wraps a unit of work as a *durable step*. The string name identifies it; the async body does the work; the **return value is what Inngest memoizes**. This is the load-bearing line of the whole week.

---

## 3. How memoization makes "resume from step 7" work — mechanically

Here is the mechanism, precisely, because hand-waving it is how people misuse it.

When an Inngest function runs, the engine executes it as a series of **replays**. On the *first* attempt:

1. The function body starts. It reaches `step.run("plan", ...)`. Inngest has no record of `plan`, so it **executes** the body, gets the result, and **persists** `{ step: "plan", result: <plan> }` to its durable store. Then it returns the result to your code, and execution continues.
2. It reaches `step.run("gather-sources", ...)`. No record → execute → persist → continue.
3. It reaches `step.run("synthesize", ...)`. No record → it starts executing… and the **process dies** mid-`synthesize`. Nothing was persisted for `synthesize` (it didn't finish).

Now Inngest **retries** the function — a *replay*. The function body starts again **from the top**:

4. It reaches `step.run("plan", ...)`. Inngest **has a record** for `plan`. So it does **not** execute the body — it returns the *memoized* result instantly. `plan` is **replayed from cache**. No Claude call. No cost.
5. It reaches `step.run("gather-sources", ...)`. Has a record → return memoized result. **Replayed from cache.** No API hit.
6. It reaches `step.run("synthesize", ...)`. *No* record (it never finished). So Inngest **executes it fresh** — and the run has **resumed from step 3**, exactly where it died.
7. `persist` runs fresh. The function completes.

That's it. The whole "resume from step 7" magic is: **a completed step's result is persisted; on replay, a step that already has a persisted result is skipped and its cached result returned; the first step *without* a result is where execution actually resumes.** Steps are skipped by **id**, so the names matter — two steps with the same id collide.

```
attempt 1:  plan(run) → gather(run) → synthesize(run, 💥 die)
attempt 2:  plan(cache) → gather(cache) → synthesize(run) → persist(run) ✓
                  ↑ skipped        ↑ skipped       ↑ resumed here
```

> **The mental model that makes it click:** the function body is *not* the source of truth — the **step record is**. The body is a recipe that gets re-walked on every replay; each `step.run` either does the work (no record) or hands back a cached result (record exists). The body is deterministic glue; the steps are the durable facts. Write the body so that re-walking it is always safe (§5), and durability is automatic.

This is the "it resumed from step N" promise from the README, and it's what Exercise 3 makes *measurable*: crash after step 2, re-run, and print which steps ran fresh versus replayed-from-cache, ending in `PASS: it resumed from step 3`.

### 3.1 Why the function body must be deterministic

There's a corollary to "the body is re-walked on every replay" that trips people up: **the body must be deterministic outside of steps.** If your bare body branches on `Math.random()`, on `Date.now()`, or on a value you fetched *outside* a step, then attempt 2 might walk a *different path* than attempt 1 — reach a different `step.run`, in a different order — and the memoized results no longer line up with the steps the body is asking for. The replay desyncs.

The rule that prevents this: **any non-determinism must come *out of a step*, so it's memoized.** Need a random id? `const id = await step.run("make-id", () => crypto.randomUUID())` — now the id is recorded on attempt 1 and *replayed identically* on attempt 2, so the body walks the same path both times. Need the current time? `await step.run("now", () => Date.now())`. The body's control flow can then branch on `id`/`now` safely, because those values are stable across replays. Determinism in the body + memoized non-determinism in the steps = a replay that always re-walks the same path and lines up with the recorded results.

```typescript
// ❌ WRONG: the body branches on fresh non-determinism. Replay may desync.
async ({ step }) => {
  if (Math.random() > 0.5) {            // different on attempt 2!
    await step.run("a", () => doA());
  } else {
    await step.run("b", () => doB());
  }
};

// ✅ RIGHT: the non-determinism is memoized; the branch is stable across replays.
async ({ step }) => {
  const coin = await step.run("coin", () => Math.random()); // recorded once
  if (coin > 0.5) {
    await step.run("a", () => doA());   // same path every replay
  } else {
    await step.run("b", () => doB());
  }
};
```

This is the same determinism requirement Temporal places on its workflow code, for the same reason: replay only works if re-walking the orchestration is deterministic. The steps are where the messy, non-deterministic, side-effecting real world is allowed to live — and each one is recorded so the next replay sees the same answer.

---

## 4. Event-driven invocation — S3 new file → event → research run

The second half of "event-driven durable execution" is the **event**. Inngest functions are triggered by events, not called directly. You emit an event with `inngest.send`:

```typescript
await inngest.send({
  name: "research/requested",
  data: { topic: "confidentiality clause durations", source: "s3://uploads/doc-42.pdf" },
});
```

That single `send` causes Inngest to find every function listening for `research/requested` and run it — durably, with all the step memoization above. The decoupling is the point: the thing that *emits* the event knows nothing about the function that *handles* it. That's what makes "a new file in S3 triggers a research run" clean.

The lab's trigger, concretely: **a new file lands in an S3 bucket → S3 emits a notification → that becomes an Inngest event → the durable research function runs.**

```typescript
// An API route (or a small Lambda) that receives the S3 event notification and
// turns it into an Inngest event. The bridge from "AWS noticed a file" to "a
// durable agent run starts."
export async function handleS3Notification(s3Event: S3Notification) {
  for (const record of s3Event.Records) {
    await inngest.send({
      name: "research/requested",
      data: {
        bucket: record.s3.bucket.name,
        key: record.s3.object.key, // the new file
        topic: deriveTopic(record.s3.object.key),
      },
    });
  }
}
```

For the lab you do **not** need a real AWS account. The S3 trigger has two local equivalents, both of which produce the same `inngest.send`:

- **A watched folder.** Watch a local directory; on a new file, call `inngest.send({ name: "research/requested", data: { key: filename } })`. Dropping a file into the folder kicks off a run, exactly like S3 would.
- **A manual send.** Just call `inngest.send(...)` from a script (or the dev-server UI's "send event" button). This is the simplest way to *prove* the event → function → durable-steps path before you wire any file source.

The event is the *trigger*; the steps are the *durability*. Keep them distinct: event-driven is about *what starts the run*; durable execution is about *what happens when the run crashes*. The lab uses both, but they're separable concerns.

---

## 5. The two disciplines that make replay safe

Memoization only works if re-walking the function body is safe. Two rules make it safe, and breaking either is the classic durable-execution bug.

**Discipline 1 — steps must be idempotent.** A step may be *attempted* more than once (a crash between "side-effect happened" and "result persisted" forces a retry of that step). So a step's side-effect must be safe to repeat: writing to S3 at a **deterministic key** (`reports/${runId}.md`, not `reports/${Date.now()}.md`) overwrites harmlessly; charging a credit card must use an **idempotency key** so the second attempt is a no-op. If a step's side-effect is *not* idempotent, a retry double-acts. Idempotency is the price of durability, and it's non-negotiable.

**Discipline 2 — side-effects only happen *inside* steps.** Any side-effect (model call, API hit, DB write, S3 put) must live **inside a `step.run`**, never in the bare function body between steps. Why: the bare body is re-walked on *every* replay. A `fetch` sitting between two steps, outside any `step.run`, runs **again on every retry** — because it has no memoized record to skip it. Put it in a step and it runs once, then replays from cache. This is the rule people break most:

```typescript
// ❌ WRONG: the fetch is in the bare body. It re-runs on EVERY replay.
async ({ event, step }) => {
  const data = await fetch(url).then((r) => r.json()); // runs every retry!
  const plan = await step.run("plan", () => makePlan(data));
  // ...
};

// ✅ RIGHT: the side-effect is INSIDE a step. It runs once, then replays from cache.
async ({ event, step }) => {
  const data = await step.run("fetch-input", () => fetch(url).then((r) => r.json()));
  const plan = await step.run("plan", () => makePlan(data));
  // ...
};
```

The bare function body should contain only **deterministic glue** — control flow, variable wiring, decisions based on step results. Anything that touches the outside world goes in a step. Get this wrong and your "durable" agent re-hits external systems on every crash, which is worse than the naive version (the naive one at least only ran once per run).

A worked idempotency-key example, because the irreversible-action case is the one that bites hardest. Say a step charges a customer. A retry of that step (the process died after the charge succeeded but before the result was persisted) must *not* charge twice:

```typescript
const charge = await step.run("charge-customer", async () => {
  // The idempotency key is DETERMINISTIC per run+step, so a retry sends the SAME
  // key. The payment provider sees the duplicate key and returns the ORIGINAL
  // charge instead of creating a second one. The retry is a safe no-op.
  return await payments.charge({
    amount: 5000,
    idempotencyKey: `${runId}:charge-customer`, // NOT Date.now(), NOT random
  });
});
```

The key insight: the *step* may run more than once, but the *external effect* happens at most once, because the downstream system de-duplicates on the key. For your own writes (S3, a database), the analogue is a deterministic destination (`reports/${runId}.md`) or an upsert keyed on `runId` — a second write overwrites the first instead of appending a duplicate. The discipline scales: every step that mutates the outside world needs *either* a deterministic destination *or* an idempotency key the downstream honors.

> **The one-line test for a step:** "If this ran twice, would anything be wrong?" If yes, you need an idempotency key or a deterministic destination. If a side-effect lives outside a step, move it in. These two rules are 90% of getting durable execution right.

---

## 6. The crash/retry/resume model in practice

Inngest doesn't just resume after a *process* crash — it handles step-level *failures* the same way. If a step **throws** (a flaky API, a transient 500), Inngest **retries that step** (with backoff) up to a configured limit, without re-running the steps before it. A step failure and a process crash are the same shape: completed steps are memoized, the failed/incomplete step is the resume point.

```typescript
const sources = await step.run("gather-sources", async () => {
  const res = await fetch(api); // may throw / 500 transiently
  if (!res.ok) throw new Error(`gather failed: ${res.status}`); // Inngest retries THIS step
  return res.json();
});
// `plan` (before this) is never re-run while `gather-sources` retries.
```

This unifies two things people usually treat separately: **retries** (handling a flaky step) and **resume-after-crash** (handling a dead process) are *the same mechanism* — memoize what finished, re-attempt what didn't. You configure retry counts and backoff per function; the durability is free either way.

The dev server makes this *visible*. Run `npx inngest-cli@latest dev`, trigger the function, and the UI at `localhost:8288` shows each step, its memoized result, and — when a step throws or you kill the process — the replay, with the completed steps marked done and execution picking up at the failed one. Watching the replay frame-by-frame is the fastest way to *believe* the resume story rather than take it on faith.

---

## 7. The neighbors — Trigger.dev and Temporal (and when each wins)

Inngest is not the only durable-execution engine. Two neighbors matter, at survey level.

**Trigger.dev** — a **TypeScript-native** background-jobs / long-running-tasks platform. Like Inngest, it gives you durable steps and retries; unlike Inngest's event-first model, it's framed around **tasks** you trigger and long jobs you want to survive deploys. It's open-source with a hosted tier. **When it wins:** you're a TypeScript shop that thinks in *background jobs* ("run this long task, survive a deploy, retry the flaky bits") more than in *events*, and you want a TS-native developer experience for that. The durability story rhymes with Inngest's — durable steps, memoized results — with a different surface and event model.

**Temporal** — the **heavyweight, language-agnostic** durable-workflow engine, battle-tested at scale (Uber, Stripe, Netflix-class workloads). You write **workflows** (deterministic orchestration code) and **activities** (the side-effecting units); Temporal persists the workflow's event history and **replays** it to resume — *exactly* the same durability idea as Inngest's step memoization, but with a much heavier operational model: you run a **Temporal cluster**, deploy **workers**, and use the Temporal SDK (available in Go, Java, TypeScript, Python, and more). **When it wins:** you need durability at serious scale, across **multiple languages**, with strong guarantees and a mature operational story — and you're willing to run the cluster to get it. Temporal is the "we operate this ourselves, at scale, in any language" choice; you do **not** reach for it for a weekend agent.

The honest ranking by weight, for *this* kind of work:

| Engine | Model | Language | Op weight | Reach for it when |
|---|---|---|---|---|
| **Inngest** | event-driven functions + durable steps | TS (+others) | light (local dev server) | event-driven agents, full-stack TS, you want resume-from-step fast |
| **Trigger.dev** | TS-native background tasks + durable steps | TS | light–medium (hosted/self-host) | TS background jobs, long tasks that survive deploys |
| **Temporal** | workflows + activities, replay-based | language-agnostic | heavy (cluster + workers) | durability at scale, polyglot, strong guarantees, you'll run infra |

You will **not** teach Temporal or Trigger.dev deeply this week — name them, contrast them, and know when each is the right pick. The lab is Inngest, because its local dev server makes the resume story tangible in minutes.

---

## 8. The unifying idea — one durability spine, three frameworks

Step back and look at all three durability mechanisms you've now seen:

- **LangGraph (week 13, Python):** a **checkpointer** persists the graph's typed state after each node; on resume you load the checkpoint and continue from the next node.
- **Inngest (this week, TypeScript):** **step memoization** persists each `step.run` result; on replay, completed steps are skipped and the run resumes at the first incomplete one.
- **Temporal (survey):** **activity history** persists each activity's result; on replay, the workflow is re-run and completed activities return their recorded results, resuming at the first incomplete one.

These are **the same idea**. Persist what finished; on restart, re-walk the orchestration and skip what's already done; resume at the first thing that isn't. The vocabulary differs (checkpoint / memoized step / activity), the language differs (Python / TypeScript / any), the operational weight differs (in-process SQLite / a dev server / a cluster) — but the **durability spine is identical**. Once you see that, switching stacks stops being scary: you're not learning a new concept, you're learning a new spelling of one you already know.

That's the deep polyglot-design payoff of the week. The *agent* differs across stacks (Mastra's agents vs LangGraph's graphs). The *durability* is the same everywhere. So when you evaluate a stack, you ask two separate questions: "is the agent ergonomics good for my team?" (Mastra vs LangGraph) and "is the durability spine solid and operable for my scale?" (Inngest vs Temporal vs LangGraph's checkpointer). They are *different axes*, and a senior engineer scores them separately instead of buying "Mastra" or "LangGraph" as a bundle.

---

## 9. Recap

You should now be able to:

- **State the problem durability solves:** a naive agent re-pays and re-acts on every crash; a durable agent skips finished steps and resumes at the first incomplete one.
- **Explain Inngest mechanically:** `step.run("name", fn)` memoizes its return value; on replay, a step with a persisted result is skipped (returned from cache) and the run resumes at the first step *without* one — that's resume-from-step-N, and steps are skipped by **id**.
- **Wire event-driven invocation:** `inngest.send({ name, data })` triggers `createFunction({ id }, { event }, handler)`; an S3 new file (or a local watched folder, or a manual send) becomes an event that starts a durable run.
- **Apply the two disciplines:** steps must be **idempotent** (deterministic destinations / idempotency keys), and **side-effects only happen inside steps** (the bare body is re-walked every replay, so anything outside a `step.run` re-runs on every retry).
- **Survey Trigger.dev and Temporal** and say when each is the right pick — Trigger.dev for TS background jobs, Temporal for heavyweight, language-agnostic durability at scale — versus Inngest for fast, event-driven, full-stack-TS durable agents.
- **See the one durability spine** behind LangGraph's checkpointer, Inngest's step memoization, and Temporal's activities, and score *agent ergonomics* and *durability* as separate axes when picking a stack.

Next: the exercises put this in your hands — scaffold a Mastra agent, build a Mastra supervisor that routes to sub-agents, and implement a memoized pipeline that crashes after step 2 and *proves* it resumed from step 3. Continue to [the exercises](../exercises/README.md).

---

## References

- *Inngest — steps & durable execution (the memoization model)*: <https://www.inngest.com/docs/learn/inngest-steps>
- *Inngest — functions (`createFunction`, `{ event, step }`)*: <https://www.inngest.com/docs/functions>
- *Inngest — events & `inngest.send`*: <https://www.inngest.com/docs/events>
- *Inngest — dev server (`npx inngest-cli@latest dev`)*: <https://www.inngest.com/docs/dev-server>
- *Trigger.dev — TS-native background jobs*: <https://trigger.dev/docs>
- *Temporal — workflows & activities (the replay-based durability spine)*: <https://docs.temporal.io/activities>
- *LangGraph — persistence / checkpointers (the Python durability spine)*: <https://langchain-ai.github.io/langgraph/concepts/persistence/>
- *Mastra docs (the agent layer the steps wrap)*: <https://mastra.ai/docs>
