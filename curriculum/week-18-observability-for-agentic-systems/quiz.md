# Week 18 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 19. Answer key is at the bottom — don't peek.

---

**Q1.** Why is an uninstrumented multi-agent system described as a "closed-box"? Pick the *most complete* reason.

- A) Because the model weights are not open-source.
- B) Because when it fails — a wrong answer, a latency spike, a cost blowout — you have no per-step record to attribute the failure to a specific span (which agent, which tool, which model call), so you reconstruct the run by hand the slow way.
- C) Because the agent runs inside a Docker container you cannot inspect.
- D) Because LLM outputs are non-deterministic and therefore cannot be logged.

---

**Q2.** In the OpenTelemetry Gen-AI semantic conventions, which attribute names are correct for a chat-completion span?

- A) `llm.model`, `llm.prompt_tokens`, `llm.completion_tokens`.
- B) `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, with `gen_ai.system` naming the provider and `gen_ai.operation.name` the operation (e.g. `chat`).
- C) `openai.model`, `openai.tokens_in`, `openai.tokens_out`.
- D) `model_name`, `tokens`, `cost` — the convention leaves naming to the vendor.

---

**Q3.** What is the relationship between a *trace* and a *span* for one agent run?

- A) A trace is a single span; the terms are interchangeable.
- B) A trace is the whole run as one tree; each span is one unit of work (an LLM call, a tool call, a retrieval, the supervisor step), nested by parent–child so the tree mirrors the agent's call structure.
- C) A span contains many traces, one per user.
- D) A trace is a metric and a span is a log line; they live in different systems.

---

**Q4.** You instrument only the top-level `invoke()` call of your supervisor agent. A retrieval sub-step is silently returning empty results. What goes wrong in your dashboard?

- A) Nothing — the top-level span captures everything below it.
- B) The failing retrieval has no span of its own, so the dashboard shows only that the whole run was slow/wrong; you can't see *which* sub-step failed, which is the exact debugging signal you instrumented for. Instrument each sub-step.
- C) The OTLP exporter drops the trace because it is incomplete.
- D) Phoenix refuses to render a single-span trace.

---

**Q5.** Why do you report **p95** (and p99) latency per agent step rather than the mean?

- A) The mean is harder to compute than a percentile.
- B) The mean is dominated by the bulk of fast requests and hides the tail; the slow 5% — the ones that breach the budget and anger users — show up in p95/p99, not in the average.
- C) p95 and the mean are always equal for latency.
- D) Percentiles cost less to store than a mean.

---

**Q6.** An SLO says "95% of supervisor runs complete in under 8 s." Over a window you served 10,000 runs and 700 breached 8 s. What does the error budget tell you?

- A) You are within budget — 700 is fewer than 1,000.
- B) Your budget was 5% (500 runs); you spent 700, so you are *over* budget (7% breached). The error budget is exhausted and the SLO is violated — freeze risky changes and fix the tail.
- C) Error budgets do not apply to latency, only to availability.
- D) The SLO passes because most runs were fast.

---

**Q7.** "Eval-on-traces" (replaying a production trace through a new prompt version) is valuable because:

- A) It deletes the old trace to save storage.
- B) It lets you test a prompt change against *real* production inputs that already failed or succeeded — you re-run the recorded inputs through the new prompt and diff the outputs/metrics, instead of guessing on synthetic examples.
- C) It re-trains the model on the trace.
- D) It is the only way to compute p95 latency.

---

**Q8.** Per-route / per-user / per-model token accounting is built from:

- A) Counting the lines in your application log file.
- B) Reading `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` off each LLM span and grouping by the route/user/model attributes you also recorded on the span, then multiplying by the model's price.
- C) The vendor's monthly invoice, after the fact.
- D) Estimating with a character count, since spans don't carry token counts.

---

**Q9.** What does an OTLP exporter do, and why does it let you send to Langfuse *and* Phoenix at once?

- A) It compiles your prompt into bytecode.
- B) It serializes spans in the OpenTelemetry wire format and ships them to any OTLP-compatible collector/backend; because both Langfuse and Phoenix accept OTLP, you attach two span processors (one per endpoint) and the same spans fan out to both with no app changes.
- C) It is a Langfuse-only feature that cannot target Phoenix.
- D) It converts spans into a relational database schema.

---

**Q10.** You auto-instrument LangChain/LangGraph with OpenInference and also add a few *manual* spans. Why both?

- A) Manual spans are required because auto-instrumentation is broken.
- B) Auto-instrumentation gives you the LLM/tool/chain spans for free with the right gen_ai attributes; manual spans add the *domain* steps the framework doesn't know about (e.g. "retrieval-precision check," a business-logic gate), so the trace tells the whole story.
- C) You should never mix them — pick one.
- D) Manual spans are only for non-Python code.

---

**Q11.** A dashboard shows p95 latency per step is fine, but a user complains a specific run was slow. Where do you look?

- A) The aggregate p95 chart — it must be wrong.
- B) The individual *trace* for that run: open the span tree, sort by span duration, and the longest span is the culprit step — trace-driven debugging starts at the trace, not the aggregate.
- C) The model's release notes.
- D) Nowhere — a single slow run inside budget is unobservable.

---

**Q12.** When would you reach for self-hosted **Langfuse** over hosted **LangSmith**?

- A) Never — LangSmith is strictly better.
- B) When you need the traces (and the PII inside prompts/completions) to stay inside your own infrastructure, want an open-source/self-hostable bill-of-materials, and are willing to run the Postgres/ClickHouse stack yourself; LangSmith is the hosted-convenience option, Phoenix the open eval-leaning one.
- C) Only when you have no GPU.
- D) When your agent is written in TypeScript.

---

**Q13.** Your replay shows the new prompt version scores 0.01 higher on one metric over the old version, on a 12-trace replay set. What's the honest conclusion?

- A) The new prompt is definitively better — ship it.
- B) A 0.01 delta on 12 traces is inside the noise; you've shown "not obviously worse," not "better." Replay on a larger, representative trace set (and check it didn't regress other metrics or latency/cost) before shipping.
- C) Replay is broken and should be discarded.
- D) The old prompt must be deleted immediately.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Without per-step spans you cannot attribute a failure to a specific call; you re-open the box the hard way. (Lecture 1 §1.)
2. **B** — The OTel Gen-AI conventions standardize `gen_ai.request.model`, `gen_ai.usage.input_tokens`/`output_tokens`, `gen_ai.system`, `gen_ai.operation.name`. (Lecture 1 §3.)
3. **B** — A trace is the whole run as a tree; spans are the nested units of work mirroring the call structure. (Lecture 1 §2.)
4. **B** — Only the top span exists, so the failing sub-step is invisible; instrument each sub-step. (Lecture 1 §5; Lecture 2 §3.)
5. **B** — The mean hides the tail; p95/p99 surface the slow minority that breaches the budget. (Lecture 2 §1.)
6. **B** — Budget was 500 (5%); 700 breached (7%), so the budget is over-spent and the SLO is violated. (Lecture 2 §1, §4.5.)
7. **B** — Replay re-runs *real* recorded inputs through the new prompt and diffs outputs/metrics — no guessing on synthetic data. (Lecture 2 §4.)
8. **B** — Read the token-usage attributes off each LLM span, group by route/user/model, multiply by price. (Lecture 1 §6.)
9. **B** — OTLP is the wire format every compatible backend accepts; two span processors fan the same spans to Langfuse and Phoenix. (Lecture 1 §4.)
10. **B** — Auto-instrumentation gives the framework spans; manual spans add the domain steps the framework can't know about. (Lecture 1 §5.)
11. **B** — Trace-driven debugging starts at the individual trace; the longest span is the culprit. (Lecture 2 §3.)
12. **B** — Self-hosted Langfuse keeps traces/PII in your infra and is open-source; LangSmith is hosted convenience, Phoenix the open eval-leaning option. (Lecture 2 §5.)
13. **B** — A 0.01 delta on 12 traces is noise; replay larger and check other metrics before shipping. (Lecture 2 §4, §4.5.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
