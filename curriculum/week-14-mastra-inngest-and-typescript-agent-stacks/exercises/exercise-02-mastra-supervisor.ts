#!/usr/bin/env -S npx tsx
// Exercise 2 — The Mastra supervisor (route a task to one of two sub-agents)
//
// Goal: Build a supervisor that ROUTES a task to one of two specialized
//       sub-agents (a research agent and a math agent) and PRINTS THE ROUTING
//       TRACE — the TypeScript counterpart of week 13's LangGraph supervisor.
//       The lesson is the pattern: the supervisor does NO domain work; its only
//       job is to classify the task and delegate. You will SEE each task land at
//       the right sub-agent in the printed trace.
//
// Estimated time: 50 minutes. Runnable.
//
// HOW TO USE THIS FILE
//
//   Standalone. Run it directly with tsx (no build step):
//
//       npx tsx exercise-02-mastra-supervisor.ts
//
//   It feeds a handful of tasks through the supervisor and prints, per task,
//   the route the supervisor chose and which sub-agent handled it.
//
//   FRONTIER vs OFFLINE: if `@mastra/core` AND `@ai-sdk/anthropic` are installed
//   AND ANTHROPIC_API_KEY is set, it wires real Mastra Agents behind
//   anthropic("claude-sonnet-4-6"). Otherwise it falls back to a tiny hand-rolled
//   "agent" interface (a deterministic stub model) that demonstrates the IDENTICAL
//   supervisor+routing concept with zero dependencies. The header prints the mode.
//
// ACCEPTANCE CRITERIA
//
//   [ ] The supervisor routes each task to exactly one sub-agent and prints the
//       routing trace (task -> route -> handling sub-agent).
//   [ ] Math-flavored tasks ("how much is 18% of 4200?") route to the math agent;
//       factual tasks route to the research agent.
//   [ ] The supervisor itself does NO domain work — it only classifies and
//       delegates (read run_supervisor: it never answers a question itself).
//   [ ] You can map each piece onto week 13's LangGraph supervisor (router node
//       -> sub-graph) and name the one difference you feel: types at the seams.
//
// Expected output is at the bottom of the file.

// --- A uniform sub-agent interface so the supervisor treats both identically ---
// (This is the shape Mastra's Agent.generate() exposes, narrowed to what we use.)
interface SubAgent {
  name: string;
  generate(task: string): Promise<{ text: string }>;
}

type Route = "research" | "math";

interface RouteDecision {
  route: Route;
  reason: string;
}

// --- The router: the supervisor's ONLY job. Classify the task into a route. ----
// In production this is a cheap Claude (Haiku) call; a deterministic keyword
// classifier keeps the exercise reproducible and offline-safe.
function route(task: string): RouteDecision {
  const mathish = /\b(calculate|sum|average|percent|how much|how many|\d+\s*[-+*/%]|\d+%)\b/i;
  if (mathish.test(task)) {
    return { route: "math", reason: "matched a quantitative pattern" };
  }
  return { route: "research", reason: "no quantitative pattern; treat as factual research" };
}

// --- The supervisor: route, then delegate. It does NO domain work itself. ------
async function runSupervisor(
  task: string,
  subAgents: Record<Route, SubAgent>,
): Promise<{ route: Route; handledBy: string; answer: string; reason: string }> {
  const decision = route(task); // the supervisor decides
  const chosen = subAgents[decision.route]; // ...then delegates
  const result = await chosen.generate(task); // the sub-agent does the work
  return {
    route: decision.route,
    handledBy: chosen.name,
    answer: result.text,
    reason: decision.reason,
  };
}

// --- Build the two sub-agents: real Mastra if available, else deterministic ----
async function buildSubAgents(): Promise<{ mode: string; agents: Record<Route, SubAgent> }> {
  const haveKey = !!process.env.ANTHROPIC_API_KEY;
  if (haveKey) {
    try {
      // Dynamic import so the file still runs (and type-checks) without the deps.
      const { Agent } = await import("@mastra/core/agent");
      const { anthropic } = await import("@ai-sdk/anthropic");

      const researchAgent = new Agent({
        name: "research-sub-agent",
        instructions: "You answer factual research questions concisely, grounded in what you know.",
        model: anthropic("claude-sonnet-4-6"),
      });
      const mathAgent = new Agent({
        name: "math-sub-agent",
        instructions: "You solve quantitative problems. Show the calculation, then the answer.",
        model: anthropic("claude-sonnet-4-6"),
      });

      // Adapt Mastra's Agent to our SubAgent interface (it already matches).
      const adapt = (a: { name: string; generate: (t: string) => Promise<{ text: string }> }): SubAgent => ({
        name: a.name,
        generate: (t) => a.generate(t),
      });

      return {
        mode: "FRONTIER (Mastra agents + anthropic('claude-sonnet-4-6'))",
        agents: { research: adapt(researchAgent), math: adapt(mathAgent) },
      };
    } catch {
      // Deps not installed — fall through to the deterministic stub.
    }
  }

  // Deterministic offline sub-agents: no model, no network, identical interface.
  const research: SubAgent = {
    name: "research-sub-agent",
    generate: async (t) => ({ text: `(stub research) Here is what I know about: "${t}".` }),
  };
  const math: SubAgent = {
    name: "math-sub-agent",
    generate: async (t) => {
      // A toy calculator so the math route does something real & checkable.
      const pct = t.match(/(\d+(?:\.\d+)?)%\s*of\s*(\d+(?:\.\d+)?)/i);
      if (pct) {
        const ans = (Number(pct[1]) / 100) * Number(pct[2]);
        return { text: `(stub math) ${pct[1]}% of ${pct[2]} = ${ans}` };
      }
      return { text: `(stub math) computed a result for: "${t}".` };
    },
  };
  return {
    mode: "OFFLINE FALLBACK (deterministic stub agents — install deps + set ANTHROPIC_API_KEY for the real thing)",
    agents: { research, math },
  };
}

// --- Driver ---------------------------------------------------------------------
async function main(): Promise<number> {
  const { mode, agents } = await buildSubAgents();
  console.log(`mode: ${mode}\n`);

  const tasks = [
    "What is the typical confidentiality duration in a services agreement?",
    "How much is 18% of 4200?",
    "Summarize why durable execution matters for agents.",
    "Calculate the average of 10, 20, and 30.",
    "Which state law commonly governs US tech contracts?",
  ];

  console.log("ROUTING TRACE");
  console.log("-".repeat(72));
  let researchCount = 0;
  let mathCount = 0;
  for (const task of tasks) {
    const out = await runSupervisor(task, agents);
    if (out.route === "research") researchCount++;
    else mathCount++;
    console.log(`task   : ${task}`);
    console.log(`  route -> ${out.route.padEnd(8)} (${out.reason})`);
    console.log(`  by    -> ${out.handledBy}`);
    console.log(`  ans   -> ${out.answer.slice(0, 64)}`);
    console.log("-".repeat(72));
  }

  console.log(`\nrouted: research=${researchCount}, math=${mathCount}`);
  if (researchCount > 0 && mathCount > 0) {
    console.log(
      "PASS: the supervisor routed tasks to BOTH sub-agents and did no domain\n" +
        "work itself — it only classified and delegated. That is the supervisor\n" +
        "pattern, identical to week 13's LangGraph router; the difference you feel\n" +
        "in TypeScript is the types at every seam (Route is a union, not a string).",
    );
    return 0;
  }
  console.log("NOTE: tweak the task list so at least one task hits each route.");
  return 0;
}

main().then((code) => process.exit(code));

// -----------------------------------------------------------------------------
// Expected output (shape; the FRONTIER mode's `ans` text varies, OFFLINE is fixed)
// -----------------------------------------------------------------------------
//
// mode: OFFLINE FALLBACK (deterministic stub agents — ...)
//
// ROUTING TRACE
// ------------------------------------------------------------------------
// task   : What is the typical confidentiality duration in a services agreement?
//   route -> research (no quantitative pattern; treat as factual research)
//   by    -> research-sub-agent
//   ans   -> (stub research) Here is what I know about: "What is the typi...
// ------------------------------------------------------------------------
// task   : How much is 18% of 4200?
//   route -> math     (matched a quantitative pattern)
//   by    -> math-sub-agent
//   ans   -> (stub math) 18% of 4200 = 756
// ------------------------------------------------------------------------
// ... (three more tasks) ...
//
// routed: research=3, math=2
// PASS: the supervisor routed tasks to BOTH sub-agents and did no domain
// work itself ... the types at every seam (Route is a union, not a string).
//
// NOTE: in FRONTIER mode the `ans` lines are real Claude answers (non-deterministic),
// but the ROUTES are identical — routing keys on the task text, not the model output.
// That is the supervisor's contract: deterministic delegation, regardless of the
// sub-agent's (non-deterministic) answer.
// -----------------------------------------------------------------------------
