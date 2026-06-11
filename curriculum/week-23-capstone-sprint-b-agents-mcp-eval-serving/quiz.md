# Week 23 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 24. Answer key is at the bottom — don't peek.

---

**Q1.** In the supervisor pattern, what is the supervisor's job?

- A) To retrieve the relevant context and write the grounded answer.
- B) To *route* — decide which subordinate agent runs next given the state — and nothing else; the subordinates do the retrieval, computation, writing, and critique.
- C) To call every tool in parallel and merge the results.
- D) To replace the subordinate agents once it has learned their behavior.

---

**Q2.** Why is "the supervisor routes, it doesn't do" worth enforcing in a capstone?

- A) It makes the system faster.
- B) It makes the system *debuggable*: when the answer is wrong, the trace shows which subordinate failed (retrieval vs writing vs critique), instead of one tangled "the agent failed."
- C) It is required by LangGraph.
- D) It reduces the number of model calls to one.

---

**Q3.** The supervisor's routing decision is returned as a structured enum field, not free text. Why?

- A) Free text is cheaper.
- B) So the decision drives a conditional edge reliably without parsing prose; an enum can't hand you "I think we should retrieve" that you'd have to regex.
- C) Because LangGraph cannot read strings.
- D) Structured output disables the model's reasoning.

---

**Q4.** What is the purpose of per-route budgets (step / token / time / cost)?

- A) To make the agent answer faster.
- B) To abort runaway loops (e.g. write → critique → write forever) cleanly before they burn the token/cost budget; most agent failures are loop/budget/tool failures, not model failures.
- C) To force the supervisor to use every agent.
- D) Budgets are only for the vendor tier.

---

**Q5.** Why does the capstone use a SQLite checkpointer on the graph?

- A) To store the gold set.
- B) So the graph resumes from the last completed node after a crash (a killed vLLM replica, a reboot) instead of restarting a long looping run from scratch — keyed by `thread_id`.
- C) To cache model outputs.
- D) Checkpointers are required for conditional edges.

---

**Q6.** You expose the corpus-search tool over MCP. What must you do *before* the tool does anything with its arguments?

- A) Nothing; the model only sends valid arguments.
- B) Validate the arguments (types, ranges, shapes) and, for a filesystem tool, resolve the path against a sandbox root and reject traversal — a tool is an RCE primitive from an untrusted caller.
- C) Log the arguments to the trace and proceed.
- D) Re-embed the arguments.

---

**Q7.** Which is the correct path-traversal defense for a filesystem MCP tool?

- A) Check that the requested path string does not contain "..".
- B) Resolve `(SANDBOX / requested).resolve()` and reject it if it is not `is_relative_to(SANDBOX)` — compare *resolved real paths*, don't prefix-match strings.
- C) Allow any path under `/tmp`.
- D) Trust the path if the file exists.

---

**Q8.** Why front both the local vLLM tier and the vendor model with LiteLLM?

- A) LiteLLM trains the models.
- B) It presents one OpenAI-compatible surface so agents route by model name, and it provides fallback (a dead local tier degrades to the vendor) and per-request cost tracking — the basis of the cost report and the week-24 failover.
- C) vLLM cannot serve without LiteLLM.
- D) To disable continuous batching.

---

**Q9.** What does vLLM's continuous batching buy you, and what makes self-hosting economical?

- A) It encrypts requests.
- B) It interleaves requests at the token level so a new request joins the in-flight batch immediately and a finished one frees its slot at once — the throughput multiplier that makes self-hosting cheaper than per-call vendor pricing at steady traffic.
- C) It increases the context window.
- D) It removes the need for a GPU.

---

**Q10.** Ragas reports low context *recall* but high context *precision*. What does that tell you, and where do you fix it?

- A) The writing-agent is confabulating; fix the writing prompt.
- B) You're *missing* relevant material — the answer can't be grounded in what was never retrieved; fix the retriever / chunking / `k`, not the generator.
- C) The judge is uncalibrated; add more labels.
- D) Nothing; recall and precision are interchangeable.

---

**Q11.** Why must the LLM-as-judge be *calibrated*?

- A) Calibration makes it run faster.
- B) Without anchoring its 1–5 scale to human-labeled examples, the judge's "4" is a vibe wearing a number — uninterpretable; calibration (10 human labels in the prompt, spot-checked) makes its score mean what a human's score means.
- C) Calibration is only needed for Ragas.
- D) An uncalibrated judge always scores everything 1.

---

**Q12.** Why export OpenTelemetry Gen-AI spans to *both* Langfuse and Phoenix?

- A) Redundancy in case one crashes.
- B) The Gen-AI semantic conventions are a cross-vendor standard both backends understand, and they have different strengths — Langfuse for prompt management + cost, Phoenix for the eval-in-prod / shadow-traffic work you lean on in week 24.
- C) Phoenix can't read traces alone.
- D) To double the storage cost.

---

**Q13.** "The last 10% of an agent is 90% of the engineering. Pick what to drop early." In practice, this means:

- A) Drop the eval; it's the least important.
- B) Get the thin end-to-end slice green first (one query through every layer, scored and traced), *then* deepen the components the eval flags as weak — and write down what you cut and why. Scope by measurement, not ambition.
- C) Perfect each component in isolation before integrating.
- D) Ship whatever runs by Friday with no measurement.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — The supervisor routes; the subordinates do. Collapsing it into a doer loses the pattern's whole value. (Lecture 1 §1.)
2. **B** — Debuggability: the trace localizes the failure to one subordinate. (Lecture 1 §1, reason 1.)
3. **B** — Structured enum drives the conditional edge reliably; no prose parsing. (Lecture 1 §2.)
4. **B** — Budgets abort runaway loops; most agent failures are loop/budget/tool failures. (Lecture 1 §3.)
5. **B** — The checkpointer resumes from the last node after a crash, keyed by `thread_id`. (Lecture 1 §4.)
6. **B** — Validate arguments and defend paths *before* acting; a tool is an RCE primitive. (Lecture 1 §5.2.)
7. **B** — Resolve real paths and use `is_relative_to`; never prefix-match strings. (Lecture 1 §5.2.)
8. **B** — One OpenAI-compatible surface, fallback, and cost tracking; the basis of failover and the cost report. (Lecture 2 §1.2.)
9. **B** — Token-level interleaving = throughput multiplier = economical self-hosting. (Lecture 2 §1.1.)
10. **B** — Low recall = missing material; fix the retriever, not the generator. The precision/recall split is the diagnostic. (Lecture 2 §2.1.)
11. **B** — An uncalibrated judge is a vibe with a number; anchor it to human labels. (Lecture 2 §2.2.)
12. **B** — Cross-vendor conventions both understand; different strengths; Phoenix is your week-24 eval-in-prod home. (Lecture 2 §3.)
13. **B** — Thin slice first, deepen by measurement, write the cut list. (Lecture 1 §6.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
