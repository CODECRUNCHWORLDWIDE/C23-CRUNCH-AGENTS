# Week 14 — Exercises

Three focused drills that take you from "a Mastra agent that calls one tool" to "a pipeline that crashes and proves it resumed from the step it died on." Each takes 40–60 minutes. Do them in order — exercise 3's durability lesson lands harder once you've felt the supervisor in exercise 2, which assumes the agent scaffold from exercise 1.

## Index

1. **[Exercise 1 — Scaffold a Mastra agent](exercise-01-scaffold-a-mastra-agent.md)** — build a minimal Mastra agent (instructions + a `@ai-sdk/anthropic` model + one tool), call it, observe the typed output, then add a second tool and a tiny workflow step. (~45 min, guided)
2. **[Exercise 2 — The Mastra supervisor](exercise-02-mastra-supervisor.ts)** — build a Mastra-style supervisor that routes a task to one of two sub-agents and prints the routing trace. Mirrors week 13's supervisor, in TypeScript. (~50 min, runnable)
3. **[Exercise 3 — Durable resume](exercise-03-durable-resume.ts)** — implement a multi-step memoized pipeline, simulate a crash after step 2, re-run, and prove steps 1–2 were replayed-from-cache and the run resumed at step 3, ending in `PASS: it resumed from step N`. (~50 min, runnable)

## How to work the exercises

- Use **Node 20+** (Node 22 LTS recommended). Check with `node --version`. Make a folder, `npm init -y`, and install deps as each exercise needs them.
- **Run a `.ts` file directly** with `tsx` — no build step: `npx tsx exercise-02-mastra-supervisor.ts`. (`npm install -D tsx` once, or let `npx` fetch it.)
- **Every runnable exercise has an offline fallback.** If `@mastra/core` / `inngest` aren't installed, or there's no `ANTHROPIC_API_KEY`, the file falls back to a small hand-rolled equivalent (a stub "agent" / a JSON-file step-memoizer) that demonstrates the *identical* concept deterministically. So the file **always runs** — the header prints which mode is active.
- **Type safety is a feature, not a hurdle.** When the compiler flags a step's input not matching the previous step's output, that's the lesson from Lecture 1 §5 doing its job — fix the shape, don't `any`-cast around it.
- When durability "doesn't resume," walk the §5 disciplines from Lecture 2 *before* you blame the engine: is the side-effect *inside* a step? Is the step idempotent? Are two steps sharing an id?
- Each runnable exercise (`.ts`) ends with an **`// Expected output (shape)`** block. If your output doesn't match the *shape*, you're not done.

## Running the exercises (and the Inngest dev server)

The two `.ts` files are standalone and run with `tsx`. Exercise 3 demonstrates Inngest's step model; it runs fully offline with a hand-rolled memoizer, but to see the *real* thing — including the replay UI — run the Inngest dev server alongside it:

```bash
# Node 20+ required; check it:
node --version

# Install deps for the runnable exercises (frontier path):
npm install @mastra/core @ai-sdk/anthropic inngest zod
npm install -D tsx

# Run an exercise directly (no build step):
npx tsx exercise-02-mastra-supervisor.ts
npx tsx exercise-03-durable-resume.ts --simulate-crash-after 2

# Optional: the Inngest dev server + UI (to SEE the resume frame-by-frame):
npx inngest-cli@latest dev
#   -> open http://localhost:8288, trigger the function, watch each step's
#      memoized result and the replay after a crash.
```

For the frontier (Claude) path, put your key in the environment: `export ANTHROPIC_API_KEY=sk-...`. For the open-weights path, point the AI-SDK model at a local vLLM/Ollama endpoint via an OpenAI-compatible provider (the agent code doesn't change — only the `model:` line does). With neither, the deterministic fallback runs and the lessons are identical.

## A note on determinism

The durability lesson is **deterministic**: given the same crash point, the same steps replay-from-cache and the run resumes at the same step, every run. That reproducibility is the whole point — if you can't reproduce "it resumed from step 3," something changed (a step id collision, a side-effect leaked outside a step, a non-idempotent destination), and that's worth finding. The model's *text* output is non-deterministic (and the frontier models reject `temperature`, so you can't pin it), which is exactly why the durability proof keys on *which steps ran*, not on *what they returned*.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-14` to compare.
