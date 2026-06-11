# Lecture 1 — Mastra and the TypeScript Agent Stack

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can explain why a TypeScript-first agent framework exists, build a Mastra agent (`instructions` + a `@ai-sdk/anthropic` model + typed tools), compose a workflow from zod-typed steps, build a supervisor that routes to sub-agents, attach memory and an eval, and state — honestly — when the Python (LangGraph) stack wins and when the TypeScript (Mastra) stack wins.

Last week you built a supervisor in **LangGraph** — Python, a `StateGraph`, explicit nodes and edges, a checkpointer. It worked, and it taught you the graph pattern. This week you build *the same supervisor* in **Mastra** — TypeScript, agents and workflows, a different but equally legitimate way to express the same architecture. The point is not "TypeScript is better." The point is **polyglot agent design**: the architecture is language-agnostic, and a senior engineer picks the stack that fits the *deployment*, not the stack they happen to know.

If you remember one sentence from this lecture, remember this one:

> **Mastra makes your agent ergonomic. It does not make your agent durable.** Durability is Lecture 2's job (Inngest), and it comes from the execution engine, not the agent framework. Keep those two concerns separate in your head from the first line of code, because conflating them is how people ship beautiful agents that lose all their work on the first crash.

---

## 1. Why a TypeScript-first agent stack exists

For two years the agent world was Python-by-default. LangChain, LangGraph, LlamaIndex, the eval libraries, the embedding models, the rerankers — all Python. So why would anyone build agents in TypeScript?

Three real reasons, and they're not "because we like JavaScript."

**Reason 1 — one language, front to back.** A huge fraction of real agent products are *web products*: a Next.js app with an API route that calls an agent that streams tokens back to a React component. If your front end, your API route, your agent, and your tool definitions are all TypeScript, there is **one language, one type system, one build, one deploy**. No Python service to stand up beside the Node app, no serialization boundary between the route and the agent, no second runtime to operate. For a full-stack team, that's not a small win — it's the difference between one deployable and two.

**Reason 2 — the edge.** TypeScript agents run where Python can't easily go: Vercel Edge Functions, Cloudflare Workers, Deno Deploy. An agent that runs at the edge, milliseconds from the user, with no cold-start Python container, is a real architectural option that the Python stack makes awkward. If your latency budget is tight and your deploy target is edge, TypeScript isn't a preference — it's a requirement.

**Reason 3 — end-to-end type safety.** This is the one that compounds. In a typed agent stack, the tool's input schema, the workflow step's output, the agent's structured response, and the API route's response body are all **statically checked**. A tool that returns `{ sources: string[] }` and a step that expects `{ documents: string[] }` is a **compile error** — caught before you run, before you deploy, before a user hits it. In the Python version the same mismatch is a `KeyError` at runtime, in production, on the unlucky request. Type safety moves a whole class of agent bugs from "3am page" to "red squiggle in the editor."

Now the honest other side, because a senior engineer states the trade both ways:

> **Python wins on ecosystem proximity.** If your agent lives next to a RAG pipeline — embeddings, rerankers, a vector store client, Ragas evals, a fine-tuned model — that ecosystem is *overwhelmingly* Python. Reaching it from TypeScript means HTTP hops or reimplementation. When the agent's center of gravity is ML/RAG, Python (LangGraph) is the right default, and the type-safety win doesn't pay for the ecosystem tax.

So the rule is not "use TypeScript." The rule is: **full-stack/edge/type-safety-critical → Mastra; ML/RAG-ecosystem-critical → LangGraph.** And — the through-line of the whole week — *neither choice gives you durability for free*. That's a separate axis (Lecture 2), and it's the same on both sides.

---

## 2. The Mastra agent — `instructions`, a `model`, and `tools`

A Mastra agent is the TypeScript analogue of the ReAct agent you wrote from scratch in week 5 and graphed in week 13: an LLM with a system prompt, a model, and tools it can call. In Mastra it's a single constructor.

```typescript
import { Agent } from "@mastra/core/agent";
import { anthropic } from "@ai-sdk/anthropic";
import { createTool } from "@mastra/core/tools";
import { z } from "zod";

// A tool: a typed function the agent can call. Inputs and outputs are zod schemas,
// so the agent (and the compiler) know the exact shape on both sides.
const searchTool = createTool({
  id: "search_docs",
  description: "Search the internal document corpus for a query and return snippets.",
  inputSchema: z.object({ query: z.string() }),
  outputSchema: z.object({ snippets: z.array(z.string()) }),
  execute: async ({ context }) => {
    const { query } = context;
    // Real implementation would hit your retriever; here, a stub.
    return { snippets: [`(stub) top snippet for "${query}"`] };
  },
});

const researchAgent = new Agent({
  name: "research-agent",
  instructions:
    "You are a research assistant. Use the search_docs tool to ground every claim. " +
    "Cite the snippet you used. If you cannot ground a claim, say so.",
  model: anthropic("claude-sonnet-4-6"),
  tools: { searchTool },
});
```

Read the four things that matter:

- **`instructions`** — the system prompt. Same role as week 13's node prompts, just attached to the agent object.
- **`model`** — wired through the Vercel **AI SDK** provider. `anthropic("claude-sonnet-4-6")` returns the AI-SDK model interface Mastra speaks. This is the clean, idiomatic Mastra path for Claude (more on the wiring in §3).
- **`tools`** — a map of `createTool` objects, each with a **zod `inputSchema` and `outputSchema`**. The schemas are the type-safety win: the agent is told the exact input shape, the result is validated against the output shape, and *your code* gets the narrowed type for free.
- **`name`** — identifies the agent in logs, evals, and the supervisor's routing table.

Calling the agent is one `await`:

```typescript
const result = await researchAgent.generate(
  "How long is the confidentiality obligation after termination?",
);
console.log(result.text); // the agent's grounded answer (a string)
```

`result.text` is typed `string`. `result.toolCalls` is a typed array of the tool calls the agent made. There is no `dict` to fish a key out of and hope — the result is a typed object, and the editor knows its shape. That's the day-to-day texture of the TypeScript stack: you stop guessing what came back.

---

## 3. The model wiring — `@ai-sdk/anthropic` and the authoritative Claude facts

Mastra does not embed a model client. It speaks the **Vercel AI SDK** model interface, and you bring a *provider*. For Claude, that provider is **`@ai-sdk/anthropic`**:

```typescript
import { anthropic } from "@ai-sdk/anthropic";

// The model id is an EXACT string — no date suffix.
const model = anthropic("claude-sonnet-4-6");
```

The 2026 Claude model facts you must get right (these are authoritative — do not invent around them):

- **Model IDs are exact strings, no date suffixes:** `claude-opus-4-8` (most capable), `claude-sonnet-4-6` (the good agent default), `claude-haiku-4-5` (fast/cheap). `claude-sonnet-4-6` is the right default for most sub-agents; reach for Opus on the supervisor's hardest synthesis.
- **Opus 4.8 and Sonnet 4.6 use *adaptive* thinking.** Through the raw Messages API that's `thinking: { type: "adaptive" }` and `output_config: { effort: "high" }` (`low | medium | high | max`). There is **no `budget_tokens`**.
- **No `temperature`, `top_p`, or `top_k` on Opus 4.8 / Sonnet 4.6** — they return a 400. If you're used to setting `temperature: 0` for determinism, stop: these models reject it. (The AI-SDK provider handles the supported params; just don't pass sampling knobs.)

When you need the **raw Messages API** (e.g. inside an Inngest step that wants full control, or to read the content-block union directly), use the official TypeScript SDK:

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic(); // reads ANTHROPIC_API_KEY from env

const response = await client.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 16000,
  messages: [{ role: "user", content: "Summarize the termination clause." }],
});

// response.content is a DISCRIMINATED UNION of blocks. You must NARROW.
for (const block of response.content) {
  if (block.type === "text") {
    console.log(block.text); // only here is `.text` available
  }
}
```

That content-block narrowing is the single most common beginner mistake in the TS SDK: `response.content` is not a string, it's an array of typed blocks (`text`, `tool_use`, `thinking`, …), and you reach `.text` only after `block.type === "text"`. The compiler enforces it — which is, again, the type-safety win doing its job.

**The open-weights path.** Mastra speaks the AI-SDK interface, so you can point it at a *local* OpenAI-compatible endpoint (vLLM or Ollama) via an OpenAI-compatible provider instead of `@ai-sdk/anthropic`. The agent code doesn't change — only the `model:` line does. That's how the whole stack (Mastra + Inngest + a local model) runs with no cloud dependency. The frontier path (Claude) and the open path (local) differ in one line.

---

## 4. Mastra workflows — typed steps you can chain

An agent decides and acts in a loop. A **workflow** is the other Mastra primitive: an *explicit, typed, ordered* composition of steps. This is the analogue of week 13's `StateGraph` — where LangGraph gives you nodes and edges, Mastra gives you steps and `.then()`. You use a workflow when the control flow is known and you want it explicit and type-checked, rather than left to the model.

A step is `createStep` with a zod input schema, a zod output schema, and an `execute`:

```typescript
import { createWorkflow, createStep } from "@mastra/core/workflows";
import { z } from "zod";

const planStep = createStep({
  id: "plan",
  inputSchema: z.object({ task: z.string() }),
  outputSchema: z.object({ subQuestions: z.array(z.string()) }),
  execute: async ({ inputData }) => {
    const { task } = inputData; // typed { task: string }
    // (real impl would call an agent; stub here)
    return { subQuestions: [`What is ${task}?`, `Why does ${task} matter?`] };
  },
});

const gatherStep = createStep({
  id: "gather",
  // The INPUT of this step must match the OUTPUT of the previous one — the
  // compiler checks the seam. This is the type-safety win at the workflow level.
  inputSchema: z.object({ subQuestions: z.array(z.string()) }),
  outputSchema: z.object({ sources: z.array(z.string()) }),
  execute: async ({ inputData }) => {
    const { subQuestions } = inputData;
    return { sources: subQuestions.map((q) => `source for: ${q}`) };
  },
});

const researchWorkflow = createWorkflow({
  id: "research",
  inputSchema: z.object({ task: z.string() }),
  outputSchema: z.object({ sources: z.array(z.string()) }),
})
  .then(planStep)
  .then(gatherStep)
  .commit(); // .commit() finalizes the workflow — you MUST call it.
```

The four things that matter:

- **`createStep`** — each step has a zod **`inputSchema`** and **`outputSchema`**. The schemas are not documentation; they're enforced. The output of `planStep` (`{ subQuestions }`) must satisfy the input of `gatherStep` (`{ subQuestions }`), and the compiler checks that seam. Mismatch the shapes and the build fails — the workflow-level analogue of the agent-tool type check.
- **`.then(step)`** — chains steps in order, passing the previous step's output as the next step's input. Mastra also has `.branch()` for conditional routing (the supervisor uses it, §5), `.parallel()` for fan-out, and `.dountil()`/`.foreach()` for loops.
- **`.commit()`** — finalizes the workflow definition. Forget it and the workflow isn't registered; this is the most common "why isn't my workflow running?" bug.
- **`inputSchema`/`outputSchema` on the workflow itself** — the whole workflow is one typed unit, so the *caller* gets the same type guarantees the steps do.

You register agents and workflows on the central `Mastra` object:

```typescript
import { Mastra } from "@mastra/core";

export const mastra = new Mastra({
  agents: { researchAgent },
  workflows: { researchWorkflow },
});
```

`mastra` is the registry the rest of your app (and the dev playground, and the deployment) talks to. It's the one object that knows about every agent and workflow you defined.

> **Agent vs workflow, the senior heuristic:** use an **agent** when the model should *decide* the next step (open-ended reasoning, tool choice). Use a **workflow** when *you* decide the next step and want it explicit, typed, and inspectable. Real systems mix them: a workflow whose steps *call* agents. The supervisor in §5 is exactly that — a workflow that routes to agents.

---

## 5. The supervisor in Mastra

The supervisor is the week-13 architecture, rebuilt: an agent (or workflow) that **routes** a task to one of several specialized sub-agents instead of doing the work itself. Last week it was a router node over a research sub-graph and a math sub-graph. Here it's a workflow with a routing step and a branch.

```typescript
import { Agent } from "@mastra/core/agent";
import { createWorkflow, createStep } from "@mastra/core/workflows";
import { anthropic } from "@ai-sdk/anthropic";
import { z } from "zod";

// Two specialized sub-agents.
const researchSubAgent = new Agent({
  name: "research-sub-agent",
  instructions: "You answer factual research questions, grounded in sources.",
  model: anthropic("claude-sonnet-4-6"),
});

const mathSubAgent = new Agent({
  name: "math-sub-agent",
  instructions: "You solve quantitative/calculation problems. Show the steps.",
  model: anthropic("claude-sonnet-4-6"),
});

// The router step: classify the task into a route. The supervisor's one job.
const routeStep = createStep({
  id: "route",
  inputSchema: z.object({ task: z.string() }),
  outputSchema: z.object({ route: z.enum(["research", "math"]), task: z.string() }),
  execute: async ({ inputData }) => {
    const { task } = inputData;
    // A cheap, fast classifier. In production this is a small Claude call (Haiku);
    // a keyword heuristic keeps the lecture deterministic.
    const isMath = /\b(calculate|sum|how much|average|percent|\d+\s*[-+*/])\b/i.test(task);
    return { route: isMath ? "math" : "research", task };
  },
});

const researchStep = createStep({
  id: "run-research",
  inputSchema: z.object({ route: z.enum(["research", "math"]), task: z.string() }),
  outputSchema: z.object({ answer: z.string(), handledBy: z.string() }),
  execute: async ({ inputData }) => {
    const r = await researchSubAgent.generate(inputData.task);
    return { answer: r.text, handledBy: "research-sub-agent" };
  },
});

const mathStep = createStep({
  id: "run-math",
  inputSchema: z.object({ route: z.enum(["research", "math"]), task: z.string() }),
  outputSchema: z.object({ answer: z.string(), handledBy: z.string() }),
  execute: async ({ inputData }) => {
    const r = await mathSubAgent.generate(inputData.task);
    return { answer: r.text, handledBy: "math-sub-agent" };
  },
});

const supervisor = createWorkflow({
  id: "supervisor",
  inputSchema: z.object({ task: z.string() }),
  outputSchema: z.object({ answer: z.string(), handledBy: z.string() }),
})
  .then(routeStep)
  .branch([
    [async ({ inputData }) => inputData.route === "research", researchStep],
    [async ({ inputData }) => inputData.route === "math", mathStep],
  ])
  .commit();
```

Read the shape: `routeStep` decides, `.branch()` picks the matching sub-agent step, the sub-agent does the work, and the workflow returns a typed `{ answer, handledBy }`. The supervisor itself does *no* domain work — it only routes. That separation (router vs worker) is the whole supervisor pattern, and it's identical to week 13's; only the syntax changed.

Compare the two, side by side, because the comparison *is* the polyglot-design lesson:

| Concern | LangGraph (week 13, Python) | Mastra (this week, TypeScript) |
|---|---|---|
| Decompose into units | nodes on a `StateGraph` | steps via `createStep` |
| Wire the units | edges (`add_edge`, conditional edges) | `.then()`, `.branch()`, `.parallel()` |
| Typed state at the seam | a `TypedDict`/Pydantic state, checked loosely | zod schemas, checked by the **compiler** |
| Route to sub-agents | a router node + conditional edge | a route step + `.branch()` |
| Where types are enforced | runtime (Pydantic) | compile time (TypeScript) + runtime (zod) |

The architecture is the same. The difference you can *feel*: in Mastra, mismatch a step's input to the previous step's output and the editor flags it before you run. In LangGraph, the same mismatch surfaces as a runtime `KeyError` on the unlucky request. That's the type-safety axis, concrete.

---

## 6. Agent memory

A bare agent is stateless: each `generate` call is independent. Real agents need **memory** — conversation history, working memory across a session, and (optionally) semantic recall of past interactions. Mastra provides it via `@mastra/memory`:

```typescript
import { Memory } from "@mastra/memory";
import { Agent } from "@mastra/core/agent";
import { anthropic } from "@ai-sdk/anthropic";

const memory = new Memory(); // backed by a store; defaults work for dev

const chatAgent = new Agent({
  name: "chat-agent",
  instructions: "You are a helpful assistant. Remember what the user told you earlier.",
  model: anthropic("claude-sonnet-4-6"),
  memory,
});

// Calls that share a thread/resource share memory.
await chatAgent.generate("My favorite store is pgvector.", {
  memory: { thread: "t1", resource: "user-42" },
});
const r = await chatAgent.generate("What did I say my favorite store was?", {
  memory: { thread: "t1", resource: "user-42" },
});
// r.text recalls "pgvector" because the thread carried the earlier turn.
```

The three memory kinds Mastra supports: **conversation history** (the recent turns, automatically included), **working memory** (a persistent scratchpad of facts about the user/session), and **semantic recall** (embedding-based retrieval of older relevant turns — the same retrieval machinery from Phase II, pointed at the conversation). For this week you'll use conversation history; the deeper memory patterns are a stretch goal. The key mental model: **memory is per-thread/per-resource**, so two users (two resources) don't bleed into each other's context.

> **Memory is not durability.** A subtle but important distinction: memory is *what the agent knows across turns*; durability (Lecture 2) is *whether a single run survives a crash*. An agent can have rich memory and zero durability (crash mid-run, lose the whole run) — they're orthogonal. Don't let "it remembers things" fool you into thinking "it's production." Memory is conversational state; durability is execution state.

---

## 7. Evals and scorers — measuring agent quality

You don't ship an agent because it "seems good." You measure it. Mastra provides **scorers** (evals) that judge agent output along axes like answer relevancy, faithfulness to provided context, and tool-use correctness — the same eval discipline you applied to retrieval in Phase II, now pointed at agents.

```typescript
import { Agent } from "@mastra/core/agent";
import { anthropic } from "@ai-sdk/anthropic";
// Mastra ships built-in scorers; you can also write custom ones.
import { createAnswerRelevancyScorer } from "@mastra/evals/scorers";

const relevancy = createAnswerRelevancyScorer({ model: anthropic("claude-sonnet-4-6") });

const agent = new Agent({
  name: "qa-agent",
  instructions: "Answer the user's question concisely.",
  model: anthropic("claude-sonnet-4-6"),
  scorers: { relevancy }, // run automatically on the agent's outputs
});
```

A scorer is an LLM-as-judge (or a heuristic) that returns a score and a reason for each output, so you can track whether a prompt change *helped* or *hurt* across a fixed eval set — the same "change one thing, measure, decide" discipline from Phase II. The senior point: **an agent without an eval set is an agent you can't improve safely**, because every prompt edit is a vibe. The eval set turns "I think this is better" into "relevancy went 0.71 → 0.79 on the 40-case set." Evals are a survey topic this week (you'll lean on them harder in the week-18 observability material), but build the habit now: define the metric before you tune the prompt.

---

## 8. Recap

You should now be able to:

- **Explain why TypeScript-first agent stacks exist** — one language front-to-back, edge deployment, and end-to-end type safety — and state the honest counter: Python wins when the agent's center of gravity is the ML/RAG ecosystem.
- **Build a Mastra agent** — `instructions`, a `model` wired through `@ai-sdk/anthropic` (`anthropic("claude-sonnet-4-6")`), and typed `tools` via `createTool` with zod schemas — and call it, reading the typed result.
- **Get the Claude facts right** — exact model IDs (no date suffix), adaptive thinking (`thinking: { type: "adaptive" }`, `output_config.effort`), no `temperature`/`top_p`/`top_k`, and content-block narrowing (`block.type === "text"`) on the raw SDK.
- **Compose a Mastra workflow** — `createStep` with zod input/output, `.then()`/`.branch()` chaining, `.commit()` — and know when to use a workflow (you decide the flow) versus an agent (the model decides).
- **Build a supervisor** as a workflow that routes to sub-agents, and map it one-to-one onto week 13's LangGraph supervisor.
- **Attach memory** (`@mastra/memory`, per-thread) and an **eval/scorer**, and keep clear that memory ≠ durability.

The one thing you have **not** done yet: make any of this survive a crash. Everything above is ergonomic and typed and lovely — and if the process dies mid-run, it starts over from zero. That gap is the entire subject of Lecture 2. Continue to [Lecture 2 — Durable Execution and Event-Driven Agents](./02-durable-execution-and-event-driven-agents.md).

---

## References

- *Mastra docs (agents, workflows, memory, evals)*: <https://mastra.ai/docs>
- *Mastra workflows (createWorkflow / createStep / branch / commit)*: <https://mastra.ai/docs/workflows/overview>
- *Mastra memory (`@mastra/memory`)*: <https://mastra.ai/docs/memory/overview>
- *Vercel AI SDK — providers and models*: <https://ai-sdk.dev/docs/foundations/providers-and-models>
- *`@ai-sdk/anthropic` provider*: <https://ai-sdk.dev/providers/ai-sdk-providers/anthropic>
- *`@anthropic-ai/sdk` (TypeScript SDK, content-block union)*: <https://github.com/anthropics/anthropic-sdk-typescript>
- *Anthropic API — messages & models*: <https://docs.anthropic.com/en/api/messages>
- *LangGraph (the Python supervisor this week mirrors)*: <https://langchain-ai.github.io/langgraph/>
