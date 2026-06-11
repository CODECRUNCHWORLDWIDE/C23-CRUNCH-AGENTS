#!/usr/bin/env -S npx tsx
// Exercise 3 — Durable resume (crash after step 2, resume from step 3)
//
// Goal: Make "your agent platform is your durability platform" MEASURABLE.
//       Implement a multi-step research pipeline where each step is MEMOIZED.
//       Run it once, simulate a crash AFTER step 2 (throw), then re-run and PROVE
//       that steps 1-2 are NOT re-executed — they are replayed from cache — and
//       the run RESUMES from step 3. The script ends in a PASS line naming the
//       step it resumed from. This is the week's "it resumed from step N" promise.
//
// Estimated time: 50 minutes. Runnable.
//
// HOW TO USE THIS FILE
//
//   Standalone. Run it directly with tsx:
//
//       npx tsx exercise-03-durable-resume.ts --simulate-crash-after 2
//
//   Attempt 1 runs steps 1-2 fresh, then "crashes" at step 3 (the body throws
//   to mimic an OOM kill / deploy / preemption). The script then RE-RUNS the same
//   run id (a replay) and prints which steps RAN FRESH vs were REPLAYED FROM CACHE,
//   resuming at the first incomplete step, and finishes the pipeline.
//
//   The memoization is backed by a tiny JSON-file step-store (`.durable-run.json`)
//   that records each completed step's result by id — exactly the SHAPE of what
//   Inngest's `step.run` does internally (persist the result; skip on replay).
//
//   USING REAL INNGEST: the bottom of the file shows the identical pipeline as an
//   inngest.createFunction with step.run — same step ids, same resume semantics.
//   Run `npx inngest-cli@latest dev` to SEE the replay in the UI at :8288.
//
// ACCEPTANCE CRITERIA
//
//   [ ] Attempt 1 runs steps 1-2 fresh and crashes at the configured step.
//   [ ] Attempt 2 (replay) prints steps 1-2 as REPLAYED FROM CACHE (NOT re-run)
//       and runs step 3+ fresh — the run resumed from the crash point.
//   [ ] The script ends with: "PASS: it resumed from step N".
//   [ ] Re-running with --simulate-crash-after 1 vs 3 moves the resume point
//       accordingly (durability tracks the actual crash point).
//   [ ] Deleting `.durable-run.json` between runs makes attempt 2 start over —
//       which PROVES the JSON file IS the durability (the step record is the
//       source of truth, not the function body). Lecture 2 §3.
//
// Expected output is at the bottom of the file.

import { readFileSync, writeFileSync, existsSync, rmSync } from "node:fs";

// --- The durable step store: persist a completed step's result by id ----------
// This is a hand-rolled stand-in for Inngest's internal memoization. The CONTRACT
// is the whole lesson: a step with a recorded result is SKIPPED on replay; the
// first step WITHOUT one is where execution resumes.
const STORE_PATH = ".durable-run.json";

type StepRecord = Record<string, unknown>;

function loadStore(runId: string): StepRecord {
  if (!existsSync(STORE_PATH)) return {};
  const all = JSON.parse(readFileSync(STORE_PATH, "utf8")) as Record<string, StepRecord>;
  return all[runId] ?? {};
}

function saveStep(runId: string, stepId: string, result: unknown): void {
  const all: Record<string, StepRecord> = existsSync(STORE_PATH)
    ? JSON.parse(readFileSync(STORE_PATH, "utf8"))
    : {};
  all[runId] = { ...(all[runId] ?? {}), [stepId]: result };
  writeFileSync(STORE_PATH, JSON.stringify(all, null, 2));
}

// A `step.run` look-alike. If the step already has a recorded result, return it
// from cache (REPLAYED). Otherwise execute the body, persist, return (RAN FRESH).
function makeStep(runId: string, cache: StepRecord, log: string[]) {
  return async function step<T>(id: string, body: () => Promise<T>): Promise<T> {
    if (id in cache) {
      log.push(`  [${id}] REPLAYED FROM CACHE  (not re-run)`);
      return cache[id] as T; // the memoized result — no re-execution
    }
    const result = await body(); // RUN FRESH (the side-effect happens here, inside the step)
    saveStep(runId, id, result); // persist BEFORE returning -> durable checkpoint
    cache[id] = result;
    log.push(`  [${id}] RAN FRESH  -> ${JSON.stringify(result)}`);
    return result;
  };
}

// --- The pipeline: four idempotent, step-wrapped units ------------------------
// Every side-effect lives INSIDE a step (Lecture 2 §5). The bare body is only glue.
async function researchPipeline(
  runId: string,
  topic: string,
  crashAfter: number,
  attempt: number,
): Promise<{ log: string[]; location?: string; crashed: boolean }> {
  const cache = loadStore(runId);
  const log: string[] = [];
  const step = makeStep(runId, cache, log);

  // step.run bodies must be idempotent: re-running them yields the same result
  // and (for persist) writes to a DETERMINISTIC destination keyed by runId.
  const plan = await step("1-plan", async () => {
    return { subQuestions: [`What is ${topic}?`, `Why does ${topic} matter?`, `Tradeoffs of ${topic}?`] };
  });
  maybeCrash(1, crashAfter, attempt, log);

  const sources = await step("2-gather-sources", async () => {
    return { sources: plan.subQuestions.map((q, i) => `source-${i}: ${q}`) };
  });
  maybeCrash(2, crashAfter, attempt, log);

  const report = await step("3-synthesize", async () => {
    const words = 200 + sources.sources.length * 30;
    return { report: `Report on ${topic}`, words };
  });
  maybeCrash(3, crashAfter, attempt, log);

  const persisted = await step("4-persist", async () => {
    // DETERMINISTIC key -> idempotent: a retry overwrites harmlessly (Lecture 2 §5).
    return { location: `s3://reports/${runId}.md`, bytes: report.words * 6 };
  });

  return { log, location: persisted.location, crashed: false };
}

// Simulate a process death AFTER a given step, but only on the FIRST attempt.
class SimulatedCrash extends Error {}
function maybeCrash(afterStep: number, crashAfter: number, attempt: number, log: string[]): void {
  if (attempt === 1 && afterStep === crashAfter) {
    log.push(`  💥 CRASH (simulated) — process died after step ${afterStep}`);
    throw new SimulatedCrash(`died after step ${afterStep}`);
  }
}

// --- Driver: run, crash, replay, prove resume ----------------------------------
async function main(): Promise<number> {
  const arg = process.argv.indexOf("--simulate-crash-after");
  const crashAfter = arg !== -1 ? Number(process.argv[arg + 1]) : 2;
  const runId = `abc123-crash${crashAfter}`;
  const topic = "durable execution";

  // Start clean so the demo is reproducible (real Inngest keeps the store; here we
  // reset so each invocation shows the full crash-then-resume arc end to end).
  if (existsSync(STORE_PATH)) {
    const all = JSON.parse(readFileSync(STORE_PATH, "utf8"));
    delete all[runId];
    writeFileSync(STORE_PATH, JSON.stringify(all, null, 2));
  }

  // ----- Attempt 1: runs fresh until the simulated crash -----
  console.log(`run ${runId} :: attempt 1`);
  let firstCrashed = false;
  try {
    const r1 = await researchPipeline(runId, topic, crashAfter, 1);
    r1.log.forEach((l) => console.log(l));
  } catch (e) {
    if (e instanceof SimulatedCrash) {
      firstCrashed = true;
      // Print the log accumulated before the crash by re-loading the store state.
      const partial = loadStore(runId);
      Object.keys(partial)
        .sort()
        .forEach((id) => console.log(`  [${id}] RAN FRESH  -> ${JSON.stringify(partial[id])}`));
      console.log(`  💥 CRASH (simulated) at step ${crashAfter + 1} — process died`);
    } else {
      throw e;
    }
  }

  if (!firstCrashed) {
    console.log("NOTE: no crash occurred (crashAfter beyond the pipeline). Try --simulate-crash-after 2.");
    return 0;
  }

  // ----- Attempt 2: REPLAY. Completed steps come from cache; resume at the crash point -----
  console.log(`run ${runId} :: attempt 2 (replay)`);
  const r2 = await researchPipeline(runId, topic, crashAfter, 2);
  r2.log.forEach((l) => console.log(l));

  // ----- Prove it: which step did we actually resume from? -----
  const resumedFrom = crashAfter + 1;
  const replayedCount = r2.log.filter((l) => l.includes("REPLAYED FROM CACHE")).length;
  console.log("");
  if (replayedCount === crashAfter && r2.location) {
    console.log(
      `PASS: it resumed from step ${resumedFrom} — steps 1-${crashAfter} were ` +
        `memoized and skipped on replay (replayed-from-cache=${replayedCount}). ` +
        `Final output: ${r2.location}`,
    );
    return 0;
  }
  console.log(
    `FAIL: expected ${crashAfter} replayed-from-cache steps, saw ${replayedCount}. ` +
      `Did a side-effect leak outside a step, or two steps share an id? (Lecture 2 §5)`,
  );
  return 1;
}

main().then((code) => process.exit(code));

// -----------------------------------------------------------------------------
// Expected output (shape) — with --simulate-crash-after 2
// -----------------------------------------------------------------------------
//
// run abc123-crash2 :: attempt 1
//   [1-plan] RAN FRESH  -> {"subQuestions":["What is durable execution?", ...]}
//   [2-gather-sources] RAN FRESH  -> {"sources":["source-0: ...", ...]}
//   💥 CRASH (simulated) at step 3 — process died
// run abc123-crash2 :: attempt 2 (replay)
//   [1-plan] REPLAYED FROM CACHE  (not re-run)
//   [2-gather-sources] REPLAYED FROM CACHE  (not re-run)
//   [3-synthesize] RAN FRESH  -> {"report":"Report on durable execution","words":290}
//   [4-persist] RAN FRESH  -> {"location":"s3://reports/abc123-crash2.md","bytes":1740}
//
// PASS: it resumed from step 3 — steps 1-2 were memoized and skipped on replay
// (replayed-from-cache=2). Final output: s3://reports/abc123-crash2.md
//
// READ IT: on attempt 2, steps 1-2 did NOT re-run (no model call, no API hit, no
// re-pay) — their results came from the JSON step-store. The run resumed at step 3,
// the first step without a recorded result. That is resume-from-step-N, the whole
// durability promise, made measurable. Delete .durable-run.json between attempts
// and attempt 2 starts over — proving the STEP RECORD, not the function body, is
// the source of truth (Lecture 2 §3).
//
// -----------------------------------------------------------------------------
// The SAME pipeline as a real Inngest function (identical resume semantics):
//
//   import { Inngest } from "inngest";
//   const inngest = new Inngest({ id: "agents" });
//
//   export const research = inngest.createFunction(
//     { id: "research-run" },
//     { event: "research/requested" },
//     async ({ event, step }) => {
//       const plan    = await step.run("1-plan",            () => makePlan(event.data.topic));
//       const sources = await step.run("2-gather-sources",  () => gatherSources(plan));
//       const report  = await step.run("3-synthesize",      () => synthesize(plan, sources));
//       const loc     = await step.run("4-persist",         () => writeToS3(event.data.runId, report));
//       return { location: loc };
//     },
//   );
//
//   // Trigger it (the S3-new-file -> event path):
//   await inngest.send({ name: "research/requested",
//                        data: { runId: "abc123", topic: "durable execution" } });
//
// Each step.run is memoized; on a crash Inngest replays, skips completed steps,
// and resumes at the first incomplete one — the exact behavior the hand-rolled
// store above demonstrates. Run `npx inngest-cli@latest dev` to watch it at :8288.
// -----------------------------------------------------------------------------
