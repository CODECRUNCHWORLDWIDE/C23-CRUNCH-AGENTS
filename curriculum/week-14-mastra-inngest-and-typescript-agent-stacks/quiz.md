# Week 14 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 15. Answer key is at the bottom — don't peek.

---

**Q1.** In Mastra, how do you put Claude behind an agent's `model`?

- A) `model: "claude-sonnet-4-6"` as a bare string.
- B) Through the Vercel AI SDK provider: `import { anthropic } from "@ai-sdk/anthropic"; model: anthropic("claude-sonnet-4-6")`.
- C) `model: new Anthropic()` from `@anthropic-ai/sdk`.
- D) Mastra has no model field; the model is global.

---

**Q2.** You set `temperature: 0` on a Mastra agent using `claude-sonnet-4-6` to make it deterministic. What happens?

- A) The agent becomes fully deterministic.
- B) Opus 4.8 and Sonnet 4.6 reject `temperature`/`top_p`/`top_k` (a 400) — those sampling params aren't supported; you don't pass them.
- C) Temperature is ignored silently.
- D) It switches the model to greedy decoding.

---

**Q3.** In a Mastra workflow, what does `.commit()` do, and what breaks if you forget it?

- A) It saves the workflow to disk; forgetting it loses your code.
- B) It finalizes/registers the workflow definition; forget it and the workflow isn't registered and won't run.
- C) It commits to git; forgetting it is harmless.
- D) It runs the workflow; forgetting it means it runs twice.

---

**Q4.** What is the relationship between the *output* schema of one Mastra step and the *input* schema of the next when chained with `.then()`?

- A) They're unrelated; Mastra coerces anything.
- B) The previous step's output must satisfy the next step's input schema — and the TypeScript compiler checks that seam, so a mismatch is a build error.
- C) Both must be `any`.
- D) Only the workflow's top-level schema matters; steps are untyped.

---

**Q5.** What is the core durability mechanism of Inngest?

- A) It writes the whole program state to disk every second.
- B) Each `step.run` result is **memoized** (persisted by step id); on replay a step that already has a result is skipped and its cached result returned, so the run resumes at the first incomplete step.
- C) It runs every function twice and compares.
- D) It uses database transactions around the entire function.

---

**Q6.** A function crashes during step 3 of 5. Inngest retries it. What happens to steps 1 and 2 on the retry?

- A) They re-run from scratch (re-calling the model, re-paying).
- B) They are replayed from cache — their memoized results are returned without re-execution — and execution resumes at step 3.
- C) The whole function is abandoned.
- D) Steps 1 and 2 run, but step 3 is skipped.

---

**Q7.** Why must a `step.run` body be **idempotent**?

- A) To make the code shorter.
- B) Because a step may be attempted more than once (a crash between the side-effect and the result being persisted forces a retry), so its side-effect must be safe to repeat — e.g. write to a deterministic key, or use an idempotency key for a charge.
- C) Idempotency isn't required in Inngest.
- D) Only the first step needs to be idempotent.

---

**Q8.** You put a `fetch(url)` in the bare function body, *between* two `step.run` calls (not inside a step). What's the bug?

- A) Nothing — it runs once like any step.
- B) The bare body is re-walked on every replay, so the un-stepped `fetch` re-runs on *every* retry/crash; side-effects must live *inside* a `step.run` to be memoized.
- C) `fetch` isn't allowed in Inngest functions at all.
- D) It runs only on the first attempt and never again.

---

**Q9.** How is the durable research run *triggered* in the event-driven model?

- A) By calling the function directly like a normal async function.
- B) By an event: `inngest.send({ name: "research/requested", data })` fires the event, and `createFunction({ id }, { event: "research/requested" }, handler)` runs in response — so a new file in S3 can become an event that starts the run.
- C) By a cron schedule only.
- D) By polling a database every second.

---

**Q10.** Where does an agent's durability actually come from in the Mastra + Inngest stack?

- A) From Mastra — its typed workflows are inherently durable.
- B) From Inngest (the execution engine) — Mastra makes the agent *ergonomic*, but durability comes from `step.run` memoization, not from the agent framework.
- C) From the Anthropic API.
- D) From TypeScript's type system.

---

**Q11.** When is **Temporal** the right pick over Inngest for durable agent execution?

- A) Always — Temporal is strictly better.
- B) When you need durability at serious scale, across **multiple languages**, with strong guarantees and a mature operational story — and you're willing to run a Temporal cluster and workers to get it.
- C) Only for TypeScript projects.
- D) When you have no database.

---

**Q12.** Honestly comparing LangGraph (Python) and Mastra (TypeScript), when does the Python stack win?

- A) Never; TypeScript is always better.
- B) When the agent's center of gravity is the **ML/RAG ecosystem** (embeddings, rerankers, eval libraries, fine-tuned models) — that ecosystem is overwhelmingly Python, and reaching it from TypeScript means HTTP hops or reimplementation.
- C) When you deploy to the edge.
- D) When you need end-to-end compile-time type safety.

---

**Q13.** What do LangGraph's checkpointer, Inngest's step memoization, and Temporal's activities have in common?

- A) Nothing; they're unrelated mechanisms.
- B) They are the **same durability spine**: persist what finished, re-walk the orchestration on restart, skip what's already done, and resume at the first incomplete unit — differing only in vocabulary, language, and operational weight.
- C) They all require Postgres.
- D) They all run the function twice and diff the results.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Mastra speaks the Vercel AI SDK model interface; `@ai-sdk/anthropic`'s `anthropic("claude-sonnet-4-6")` is the idiomatic Claude wiring. (Lecture 1 §2–3.)
2. **B** — Opus 4.8 / Sonnet 4.6 reject `temperature`/`top_p`/`top_k` with a 400; you don't pass sampling knobs to these models. (Lecture 1 §3.)
3. **B** — `.commit()` finalizes/registers the workflow; forgetting it is the classic "why isn't my workflow running?" bug. (Lecture 1 §4.)
4. **B** — The previous step's output must satisfy the next step's input schema, and the compiler checks that seam — the workflow-level type-safety win. (Lecture 1 §4–5.)
5. **B** — Step memoization: each `step.run` result is persisted by id; on replay a step with a record is skipped and its result returned; the run resumes at the first incomplete step. (Lecture 2 §2–3.)
6. **B** — Completed steps replay from cache (no re-execution); the run resumes at the crashed step. That's resume-from-step-N. (Lecture 2 §3.)
7. **B** — A step may be attempted more than once, so its side-effect must be repeat-safe (deterministic destination / idempotency key). Idempotency is the price of durability. (Lecture 2 §5.)
8. **B** — The bare body is re-walked every replay, so an un-stepped `fetch` re-runs on every retry; side-effects must live inside a `step.run` to be memoized. (Lecture 2 §5.)
9. **B** — Events: `inngest.send` fires `research/requested`; `createFunction` listening for it runs durably. An S3 new file becomes that event. (Lecture 2 §4.)
10. **B** — Durability comes from the execution engine (Inngest's `step.run` memoization), not the agent framework. Mastra is ergonomics; Inngest is durability. (Lecture 2 intro, §8; Lecture 1 §1.)
11. **B** — Temporal: heavyweight, language-agnostic, battle-tested durability at scale — when you need polyglot + strong guarantees and will run the cluster. (Lecture 2 §7.)
12. **B** — Python (LangGraph) wins on ML/RAG ecosystem proximity; that's the honest counter to the TypeScript type-safety/edge/full-stack win. (Lecture 1 §1.)
13. **B** — One durability spine: persist what finished, re-walk, skip the done, resume at the first incomplete unit; only the vocabulary/language/op-weight differ. (Lecture 2 §8.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
