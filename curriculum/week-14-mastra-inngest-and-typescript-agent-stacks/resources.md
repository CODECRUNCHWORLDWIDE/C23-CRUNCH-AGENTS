# Week 14 — Resources

Every resource here is **free** or has a free tier. **Mastra** and **Inngest** are open source and self-hostable; the Inngest **dev server** runs locally with zero cloud account. **Trigger.dev** is open-source with a hosted free tier; **Temporal** is open-source (and Temporal Cloud has a trial). The **Vercel AI SDK** and the Anthropic SDKs are open. The model behind every agent can be a **local open-weights model** (vLLM/Ollama via an OpenAI-compatible provider) *or* **Anthropic Claude** (the frontier path) — both wirings are shown.

Framework APIs move every cohort — the *concepts* (agent, workflow, step, durable memoization, event-driven invocation, idempotency) are stable. When a specific page 404s, search the project's docs for the primitive name (`createWorkflow`, `step.run`, `inngest.send`).

This week sits beside week 13. The supervisor architecture, the plan/retrieve/execute/critique decomposition, and the resume-after-crash idea come from there; the resources below assume you have that LangGraph supervisor to compare against.

## Required reading (work it into your week)

- **Mastra — overview & agents** — the canonical reference for `new Agent({ name, instructions, model, tools })`, the AI-SDK model interface, and calling an agent. Read the agents and tools pages first:
  <https://mastra.ai/docs>
- **Mastra — workflows** — `createWorkflow` / `createStep`, zod-typed input/output, `.then()` chaining, `.commit()`, and branching. This is the supervisor's backbone:
  <https://mastra.ai/docs/workflows/overview>
- **Inngest — durable execution & steps** — `createFunction`, `step.run`, and *why* a completed step is memoized and skipped on replay. Read the "steps" and "durable execution" pages until "resume from step 7" is mechanical to you:
  <https://www.inngest.com/docs/learn/inngest-steps>
- **Vercel AI SDK — providers** — the model interface Mastra uses; how a provider like `@ai-sdk/anthropic` plugs a model into `generateText`/`streamText` and into a Mastra agent:
  <https://ai-sdk.dev/docs/foundations/providers-and-models>

## Mastra (the TypeScript-first agent framework)

- **Mastra docs home** — agents, workflows, memory, evals/scorers, tools, RAG, deployment:
  <https://mastra.ai/docs>
- **Mastra agents** — `instructions`, `model`, `tools`, structured output, calling `.generate()` / `.stream()`:
  <https://mastra.ai/docs/agents/overview>
- **Mastra workflows** — typed steps, chaining, branching, suspend/resume, `.commit()`:
  <https://mastra.ai/docs/workflows/overview>
- **Mastra memory (`@mastra/memory`)** — conversation history, working memory, and semantic recall for agents:
  <https://mastra.ai/docs/memory/overview>
- **Mastra evals / scorers** — measuring agent output quality (relevancy, faithfulness, tool-use correctness):
  <https://mastra.ai/docs/evals/overview>
- **Mastra GitHub** — open source; read the examples directory for runnable agent + workflow code:
  <https://github.com/mastra-ai/mastra>

## Inngest (event-driven durable execution)

- **Inngest docs home** — functions, events, steps, durability, retries, flow control:
  <https://www.inngest.com/docs>
- **Inngest steps & durable execution** — the memoization model: each `step.run` result is persisted and replayed; on a crash/retry, completed steps are skipped and the run resumes from the first incomplete one:
  <https://www.inngest.com/docs/learn/inngest-steps>
- **Inngest functions** — `inngest.createFunction({ id }, { event }, handler)`, the `{ event, step }` handler args, concurrency and retries:
  <https://www.inngest.com/docs/functions>
- **Inngest events & `inngest.send`** — emitting events (`{ name, data }`) that trigger functions; the S3-new-file → event pattern:
  <https://www.inngest.com/docs/events>
- **Inngest dev server** — `npx inngest-cli@latest dev` runs the whole thing locally with a UI at `http://localhost:8288`; no cloud account needed:
  <https://www.inngest.com/docs/dev-server>
- **Inngest GitHub** — open source; the SDK and the dev server:
  <https://github.com/inngest/inngest>

## The heavier-duty neighbors (survey level)

- **Trigger.dev** — TS-native background jobs / long-running tasks with durable steps and retries; open-source with a hosted free tier. The "background agent jobs" option:
  <https://trigger.dev/docs>
- **Temporal** — the heavyweight, language-agnostic durable-workflow engine: workflows + activities, replay-based durability, battle-tested at scale. The "durability at scale, any language" option:
  <https://docs.temporal.io/>
- **Temporal — workflows & activities** — the activity is Temporal's unit of durable, retryable side-effect (the analogue of Inngest's `step.run`):
  <https://docs.temporal.io/activities>

## The Python-vs-TypeScript framing (the honest comparison)

- **LangGraph (week 13, Python)** — the StateGraph, checkpointers, and the supervisor you built last week. This week's Mastra supervisor is the direct counterpart:
  <https://langchain-ai.github.io/langgraph/>
- **LangGraph persistence / checkpointers** — the Python durability spine; compare it to Inngest's step memoization and Temporal's activity history:
  <https://langchain-ai.github.io/langgraph/concepts/persistence/>
- The framing to hold in your head: **LangGraph (Python)** wins on ML/RAG ecosystem proximity (the embeddings, the rerankers, the eval libraries all live in Python); **Mastra (TypeScript)** wins on full-stack/edge deployment, one language front-to-back, and end-to-end type safety. **Durability is the shared spine** — LangGraph's checkpointer ≈ Inngest's step memoization ≈ Temporal's activities — and it does *not* come from the agent framework; it comes from the execution engine.

## The Anthropic SDKs (TypeScript)

- **`@anthropic-ai/sdk`** — `npm install @anthropic-ai/sdk`. The official TypeScript SDK. `const client = new Anthropic()`; `client.messages.create({ model, max_tokens, messages })`. The `content` field is a discriminated union — narrow `block.type === "text"` before reading `block.text`:
  <https://github.com/anthropics/anthropic-sdk-typescript>
- **Anthropic API — models & messages** — the model IDs (`claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5`), adaptive thinking (`thinking: { type: "adaptive" }`, `output_config: { effort: "high" }`), and the message format:
  <https://docs.anthropic.com/en/api/messages>
- **`@ai-sdk/anthropic`** — `npm install @ai-sdk/anthropic`. The Vercel AI SDK provider for Claude; `import { anthropic } from "@ai-sdk/anthropic"; model: anthropic("claude-sonnet-4-6")`. This is the clean, idiomatic way to put Claude behind a Mastra agent:
  <https://ai-sdk.dev/providers/ai-sdk-providers/anthropic>

## Models you'll use this week

- **`claude-sonnet-4-6`** — the good agent default: capable, fast enough, cheaper than Opus. Most sub-agents use this. Adaptive thinking, no `temperature`/`top_p`/`top_k` (they 400).
- **`claude-opus-4-8`** — the most capable model; reach for it on the supervisor's hardest routing/synthesis steps. Same adaptive-thinking interface.
- **`claude-haiku-4-5`** — the fast/cheap tier for trivial routing or classification sub-steps.
- **A local open-weights model** (e.g. a Llama/Qwen-class model served by **vLLM** or **Ollama**) behind an **OpenAI-compatible AI-SDK provider** — the self-hostable path, so the whole stack (Mastra + Inngest + a local model) runs with no cloud dependency at all.

## Tools you'll use this week

- **`@mastra/core`** — `npm install @mastra/core`. `Agent`, `Mastra`, `createWorkflow`, `createStep`.
- **`@mastra/memory`** — `npm install @mastra/memory`. Agent memory (`new Memory(...)`).
- **`@ai-sdk/anthropic`** — `npm install @ai-sdk/anthropic`. The Claude provider for the AI-SDK model interface.
- **`inngest`** — `npm install inngest`. `new Inngest({ id })`, `createFunction`, `step.run`, `inngest.send`.
- **`@anthropic-ai/sdk`** — `npm install @anthropic-ai/sdk`. The direct Anthropic client (used where you want raw Messages API access).
- **`zod`** — `npm install zod`. The schema library Mastra uses for typed workflow I/O and tool inputs.
- **`tsx`** — `npm install -D tsx`. Run a `.ts` file directly: `npx tsx file.ts`. No build step.
- **`inngest-cli`** — `npx inngest-cli@latest dev`. The local dev server + UI.

## A note on Node and the Inngest dev server

- **Node 20+ required; Node 22 LTS recommended.** Check with `node --version`. The Mastra and Inngest SDKs target modern Node; older versions will throw on ESM/`fetch` features.
- **Run the Inngest dev server locally** with `npx inngest-cli@latest dev`. It starts an event bus, a runner, and a UI at `http://localhost:8288` where you can *see* each step's memoized result and watch a function resume after a crash — no cloud account, no signup. The dev server is the single best tool for *believing* the resume story, because it shows you the replay frame by frame.
- **Run a `.ts` file** with `npx tsx file.ts` — no compile step, no `tsconfig` gymnastics for a single file. For the mini-project (a real package) you'll have a `tsconfig.json`; for the exercises, `tsx` is enough.
- **Environment:** put `ANTHROPIC_API_KEY=...` in a `.env` (and load it with `--env-file=.env` on Node 22, or `dotenv`). Every lab has a deterministic offline fallback, so a missing key never blocks you.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Agent** | An LLM with `instructions`, a `model`, and `tools` it can call — the unit that *decides and acts*. In Mastra: `new Agent({...})`. |
| **Workflow** | A typed, ordered composition of steps with explicit input/output schemas. In Mastra: `createWorkflow` + `createStep` + `.then()` + `.commit()`. |
| **Step** | One unit of work inside a workflow/function. In Inngest: `step.run("name", fn)` — and a step is the unit of *durability*. |
| **Durable execution** | Execution that survives a crash: completed work is persisted, and on restart the run resumes from where it died instead of from the top. |
| **Memoization** | Inngest's durability mechanism: a completed `step.run`'s result is recorded; on replay the step is *not* re-run — its cached result is returned. |
| **Event** | A named message with data (`{ name, data }`) that can trigger a function. An S3-new-file notification becomes an event; the event starts a research run. |
| **Supervisor** | An agent (or workflow) that *routes* a task to one of several sub-agents instead of doing the work itself — the multi-agent router pattern. |
| **Sub-agent** | A specialized agent the supervisor delegates to (e.g. a research agent, a math agent). |
| **AI SDK provider** | The Vercel AI SDK's pluggable model interface. `@ai-sdk/anthropic` is the provider that puts Claude behind `anthropic("claude-sonnet-4-6")`. |
| **Idempotent** | A step that produces the same result and the same side-effects no matter how many times it runs — the property that makes replay safe. |
| **Checkpointer** | LangGraph's durability mechanism (Python): persisted graph state you resume from. The Python analogue of Inngest's step memoization. |
| **Activity** | Temporal's unit of durable, retryable side-effect — the Temporal analogue of an Inngest `step.run`. |

---

*If a link 404s, please open an issue so we can replace it.*
