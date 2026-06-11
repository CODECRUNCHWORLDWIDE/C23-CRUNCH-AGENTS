# Week 5 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 6. Answer key is at the bottom — don't peek.

---

**Q1.** The agent loop is, at its core:

- A) A neural network that decides actions end-to-end.
- B) The Week 4 tool-use round trip wrapped in a `while`: call the model, run the tools it asks for, feed results back, repeat until the model stops asking.
- C) A fixed pipeline of stages with no branching.
- D) A framework feature that cannot be implemented by hand.

---

**Q2.** In a hand-rolled loop, you must append the assistant turn to `messages` as `response.content` (the full list of content blocks), not as an extracted string. Why?

- A) It saves tokens.
- B) The string form is unreadable.
- C) If you drop the `tool_use` blocks, the next turn 400s — a `tool_result` you send back has no matching `tool_use_id`.
- D) The API requires Markdown.

---

**Q3.** What is the stop condition for the loop?

- A) A fixed number of iterations, always.
- B) The model's `stop_reason`: `tool_use` means keep going, `end_turn` means done — within the budgets you enforce.
- C) When the user presses Ctrl-C.
- D) When the token count reaches exactly zero.

---

**Q4.** In ReAct, why does interleaving reason → act → observe beat producing a full plan up front and then executing it?

- A) It uses fewer tokens always.
- B) The agent adapts: it sees what an early action returns and adjusts, instead of committing to later steps before knowing the earlier ones' results.
- C) Plans are not allowed by the API.
- D) Observation is optional in plan-and-execute.

---

**Q5.** You have an agent with only a *step* budget. Which runaway does it fail to stop?

- A) The agent that takes too many turns.
- B) The agent stuck on a hung tool that never returns — steps stay low, but wall-clock runs forever (needs a time budget).
- C) An agent that finishes in two steps.
- D) None — a step budget stops every runaway.

---

**Q6.** Why do you need all four budgets (step, token, time, cost) rather than just one?

- A) Redundancy for safety.
- B) Each catches a *different* runaway: a 3-step run can blow tokens on a huge tool output; a within-token run can hang on a slow tool; a within-time run on an expensive model can still blow dollars.
- C) The API requires four.
- D) Three of them are deprecated.

---

**Q7.** Your agent's trace shows the same `act calculator(expr="3/15")` and the same `observe Error` repeating every step. Which failure mode is this, and what's the fix?

- A) Infinite tool-call loop; add more tools.
- B) Re-calling a failing tool; make the tool's error message *actionable* (say what was wrong and what valid input looks like) so the model can correct.
- C) Hallucinated tool name; rename the tool.
- D) Answering without acting; soften the prompt.

---

**Q8.** The model emits a `tool_use` for `web_search`, but you only registered `web_fetch`. The best loop behavior is:

- A) Crash the loop with an exception.
- B) Silently ignore the call and continue.
- C) Return a `tool_result` with `is_error: true` that names the valid tools, so the model can correct on the next turn.
- D) Retry the same call automatically forever.

---

**Q9.** For the *cost* budget, where do you get the true token counts to multiply by price?

- A) `tiktoken`.
- B) `response.usage.input_tokens` and `response.usage.output_tokens` from each Anthropic response.
- C) A fixed estimate of 4 characters per token.
- D) The `max_tokens` you requested.

---

**Q10.** Why must you NOT use `tiktoken` to count tokens for a Claude cost budget?

- A) It is too slow.
- B) It is OpenAI's tokenizer and miscounts Claude (under by ~15–20% on typical text, more on code); use `messages.count_tokens` for pre-flight estimates and `response.usage` for actuals.
- C) It requires a GPU.
- D) It only works on images.

---

**Q11.** You point the *same* ReAct loop at `claude-opus-4-8` and at `qwen2.5:7b-instruct` via Ollama. What is portable across both, and what is not?

- A) Nothing is portable; you rewrite everything.
- B) Portable: the JSON-Schema tool definition. Not portable: block shape (`tool_use` blocks vs `tool_calls` with stringified args) and result envelope (`tool_result` vs `role:"tool"`).
- C) Everything is portable, including block names.
- D) Only the system prompt is portable.

---

**Q12.** When does a *reflection* (self-critique) pass reliably earn its extra tokens?

- A) Always — more critique is always better.
- B) Never — it only doubles cost.
- C) When the failure mode is a *checkable* error and the critique can ground itself in a tool (re-run the code, re-read the source); measure the lift on your benchmark.
- D) Only on the local model.

---

**Q13.** What is the honest senior take on hand-rolled loop vs the `claude-agent-sdk` runner?

- A) Always hand-roll; SDKs are for beginners.
- B) Always use the SDK; never write a loop.
- C) The SDK is the right default once you understand the loop — the code-surface win is real; hand-roll when you need control the SDK doesn't expose (custom budget placement, approval gates, bespoke traces). You build it by hand first so you can debug the SDK and make that call.
- D) They produce different answers, so you must use both.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — The loop is the Week 4 round trip wrapped in `while`; everything else is budgets, patterns, and trace-reading. (Lecture 1 §1.)
2. **C** — Append `response.content` verbatim; dropping the `tool_use` blocks orphans the `tool_result` and the next turn 400s. (Lecture 1 §1.)
3. **B** — The model's `stop_reason` is the stop condition: `tool_use` continues, `end_turn` finishes — bounded by your budgets. (Lecture 1 §1, Lecture 2 §2.)
4. **B** — Interleaving lets the agent adapt to what each action returns; plan-then-execute is premature commitment. (Lecture 1 §3, §4.1.)
5. **B** — A step budget misses the hung-tool runaway; that needs a time budget. (Lecture 2 §1.)
6. **B** — Each budget catches a different runaway the others miss. (Lecture 2 §1.)
7. **B** — Repeating identical act/error is re-calling a failing tool; the fix is an actionable error message. (Lecture 2 §3.2.)
8. **C** — Return an `is_error` result naming the valid tools so the model can recover; never crash. (Lecture 1 §2, Lecture 2 §3.3.)
9. **B** — Real token counts come from `response.usage`; multiply by the model's price. (Lecture 2 §2, §6.)
10. **B** — `tiktoken` is OpenAI's tokenizer and miscounts Claude; use `count_tokens` / `response.usage`. (Lecture 2 §6.)
11. **B** — The JSON Schema is portable; block shapes and result envelopes are not. (Lecture 1 §5.)
12. **C** — Reflection earns its tokens on checkable errors where the critique can ground in a tool; measure it. (Lecture 1 §4.2, Lecture 2 §3.6.)
13. **C** — The SDK is the right default once you understand the loop; hand-roll for control. You build by hand first to earn that judgment. (Lecture 2 §5.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
