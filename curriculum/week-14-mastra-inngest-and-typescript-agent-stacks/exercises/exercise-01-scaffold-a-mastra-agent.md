# Exercise 1 — Scaffold a Mastra Agent

**Goal:** Build a minimal Mastra agent from nothing — `instructions`, a `model` wired through `@ai-sdk/anthropic`, and one typed tool — call it, observe the **typed** output, then grow it: add a second tool and a tiny workflow step. You will train the core habit of the TypeScript stack: **read the types**. The shape of every result is known to the compiler, so you stop guessing what came back.

**Estimated time:** 45 minutes. Guided.

---

## Setup

Make a folder, init, and install the frontier-path deps:

```bash
mkdir mastra-agent-scaffold && cd mastra-agent-scaffold
npm init -y
npm pkg set type=module          # so import/export work without a build step
npm install @mastra/core @ai-sdk/anthropic zod
npm install -D tsx
```

Set your key for the frontier path (or skip it and use the open-weights path below):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

> **Node 20+ required.** Check `node --version`. If it's older, upgrade — the SDKs target modern Node.

---

## Step 1 — The smallest agent that works

Create `agent.ts`:

```typescript
import { Agent } from "@mastra/core/agent";
import { anthropic } from "@ai-sdk/anthropic";

const agent = new Agent({
  name: "scaffold-agent",
  instructions: "You are a concise assistant. Answer in one or two sentences.",
  model: anthropic("claude-sonnet-4-6"), // EXACT id, no date suffix
});

const result = await agent.generate("What is durable execution, in one sentence?");
console.log(result.text); // result.text is typed `string`
```

Run it:

```bash
npx tsx agent.ts
```

You should see a one-sentence answer. Notice what you did *not* do: no `temperature`, no `top_p` — Sonnet 4.6 rejects those (a 400). The AI-SDK provider handles the supported params; you just pick the model.

> **Open-weights path:** to run this with a local model instead of Claude, install an OpenAI-compatible provider and point it at your vLLM/Ollama endpoint, then change *only* the `model:` line. The rest of the agent is identical. That one-line swap is the whole "frontier vs open" story.

---

## Step 2 — Add a typed tool

A bare agent can't *act*. Give it a tool — a typed function it can call. Create `tools.ts`:

```typescript
import { createTool } from "@mastra/core/tools";
import { z } from "zod";

export const wordCountTool = createTool({
  id: "word_count",
  description: "Count the words in a piece of text.",
  inputSchema: z.object({ text: z.string() }),
  outputSchema: z.object({ count: z.number() }),
  execute: async ({ context }) => {
    const { text } = context; // typed { text: string }
    return { count: text.trim().split(/\s+/).filter(Boolean).length };
  },
});
```

Wire it into the agent and ask a question that needs it:

```typescript
import { Agent } from "@mastra/core/agent";
import { anthropic } from "@ai-sdk/anthropic";
import { wordCountTool } from "./tools.ts";

const agent = new Agent({
  name: "scaffold-agent",
  instructions:
    "You are a concise assistant. When asked to count words, USE the word_count tool — " +
    "do not count by hand.",
  model: anthropic("claude-sonnet-4-6"),
  tools: { wordCountTool },
});

const result = await agent.generate(
  'How many words are in: "durable execution resumes from the failed step"?',
);
console.log("answer:", result.text);
console.log("tool calls:", result.toolCalls); // typed array of the calls it made
```

Run it and confirm `result.toolCalls` shows the agent *called* `word_count` rather than guessing. The `inputSchema`/`outputSchema` are the type-safety win: the agent is told the exact input shape, and the result is validated against the output shape before your code touches it.

---

## Step 3 — Observe the typed output

The texture of the TypeScript stack is that **the result is a typed object, not a `dict` to fish in**. In your editor, hover `result` and see:

- `result.text: string` — the agent's text answer.
- `result.toolCalls` — a typed array of tool calls (id, args, result).
- `result.usage` — token usage.

There is no `result["text"]` that might `KeyError`. If you typo `result.txt`, the compiler stops you *before* you run. Try it: change `result.text` to `result.txt`, run `npx tsx`, and watch it refuse to run. That refusal is the type system catching a bug at the keystroke instead of in production.

---

## Step 4 — Add a second tool and a tiny workflow step

Now grow toward the supervisor (Exercise 2). Add a second tool, then wrap a unit of work as a **workflow step** with typed I/O.

Add a second tool to `tools.ts`:

```typescript
export const upperTool = createTool({
  id: "shout",
  description: "Return the text in upper case.",
  inputSchema: z.object({ text: z.string() }),
  outputSchema: z.object({ shouted: z.string() }),
  execute: async ({ context }) => ({ shouted: context.text.toUpperCase() }),
});
```

Now a one-step workflow that uses the typed-step machinery from Lecture 1 §4:

```typescript
import { createWorkflow, createStep } from "@mastra/core/workflows";
import { z } from "zod";

const countStep = createStep({
  id: "count",
  inputSchema: z.object({ text: z.string() }),
  outputSchema: z.object({ text: z.string(), count: z.number() }),
  execute: async ({ inputData }) => {
    const count = inputData.text.trim().split(/\s+/).filter(Boolean).length;
    return { text: inputData.text, count }; // shape must match outputSchema
  },
});

const flow = createWorkflow({
  id: "count-flow",
  inputSchema: z.object({ text: z.string() }),
  outputSchema: z.object({ text: z.string(), count: z.number() }),
})
  .then(countStep)
  .commit(); // forget .commit() and the workflow won't run — the classic bug
```

The lesson: a **step** has a zod `inputSchema` and `outputSchema`, and when you chain steps (Exercise 2), the *output* of one must satisfy the *input* of the next — checked by the compiler. You just built the unit the supervisor is made of.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] A Mastra agent with `instructions` + `anthropic("claude-sonnet-4-6")` runs and prints a typed `result.text`.
- [ ] You added a **typed tool** (zod `inputSchema`/`outputSchema`) and confirmed via `result.toolCalls` that the agent *called* it rather than guessing.
- [ ] You triggered a **compile error** on purpose (e.g. `result.txt`) and saw the type system catch it before running — and can state why that's the type-safety win.
- [ ] You added a **second tool** and a **one-step workflow** (`createStep` + `.then()` + `.commit()`) with matching input/output schemas.
- [ ] You can state, in one sentence, when you'd use a **workflow** (you decide the flow) versus let the **agent** decide.

---

## Stretch

- **Open-weights swap.** Re-run Step 1 against a local vLLM/Ollama model via an OpenAI-compatible AI-SDK provider, changing *only* the `model:` line. Confirm the agent code is otherwise identical — that's the portability of the AI-SDK model interface.
- **Two-step workflow.** Chain `countStep` into a second step that takes `{ text, count }` and returns `{ summary: string }`. Deliberately mismatch the second step's `inputSchema` and watch the compiler reject the `.then()` seam — the Lecture 1 §5 type check, felt.
- **Memory.** Add `@mastra/memory` (`new Memory()`) to the agent, run two `generate` calls on the same `thread`, and confirm the second call remembers the first. Then confirm memory is per-thread by switching threads and watching the recall disappear.

---

When this feels comfortable, move to [Exercise 2 — The Mastra supervisor](exercise-02-mastra-supervisor.ts).
