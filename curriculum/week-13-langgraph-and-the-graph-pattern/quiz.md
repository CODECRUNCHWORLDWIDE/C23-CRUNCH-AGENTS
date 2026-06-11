# Week 13 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 14. Answer key is at the bottom — don't peek.

---

**Q1.** The week's heuristic is "reach for LangGraph before the loop gets a fourth `if`." What is the *real* reason a branching agent loop should become a state graph?

- A) Graphs run faster than loops on the same hardware.
- B) A `while` loop hides its state in the program counter and local variables — un-inspectable, un-serializable, un-resumable — while a `StateGraph` makes the state an explicit typed object you can print, log, checkpoint, and resume.
- C) LangGraph is required to call the Anthropic SDK.
- D) Loops cannot call more than three tools.

---

**Q2.** In a LangGraph `StateGraph`, what does a node function return?

- A) The entire new state, fully reconstructed.
- B) A *partial* dict of only the keys it changed; LangGraph merges that delta into the running state according to each key's reducer.
- C) Nothing — nodes mutate the state in place.
- D) A boolean indicating success or failure.

---

**Q3.** A state key is declared `docs: Annotated[list[str], operator.add]`. A node returns `{"docs": ["x"]}` when `docs` already holds `["a", "b"]`. What is `docs` after the merge?

- A) `["x"]` — the return value overwrites the old value.
- B) `["a", "b", "x"]` — the `operator.add` reducer *appends* the returned list to the existing one.
- C) `["x", "a", "b"]` — reducers prepend.
- D) An error — you can't return a list for a reducer key.

---

**Q4.** What does `add_conditional_edges("critique", route_fn, {"replan": "plan", "end": END})` do?

- A) Always routes `critique` to both `plan` and `END`.
- B) After `critique`, calls `route_fn(state)`; its returned key is looked up in the path map to pick the next node — so `"replan"` goes to `plan` (the loop-back) and `"end"` goes to `END`.
- C) Runs `route_fn` before `critique`.
- D) Defines two unconditional edges out of `critique`.

---

**Q5.** Where does the step budget belong in the four-node ReAct graph, and why?

- A) In the `execute` node, because that's where tokens are spent.
- B) In the conditional-edge routing function after `critique`, because that's the only place the re-plan loop can close — `if state["steps"] >= MAX: return "end"` is what guarantees termination.
- C) Nowhere — LangGraph caps loops automatically.
- D) In the initial state, as a constant the nodes read but never act on.

---

**Q6.** What is the difference between `MemorySaver` and `SqliteSaver`?

- A) `MemorySaver` is faster but less accurate.
- B) `MemorySaver` stores state in a Python dict in the current process (gone when the process dies); `SqliteSaver` writes state to a file on disk, so a fresh process can resume the same `thread_id` after a kill.
- C) They are aliases for the same class.
- D) `SqliteSaver` only works with PostgreSQL.

---

**Q7.** You ran a checkpointed graph in one process; it completed `plan` and `retrieve`, then the process was killed. In a fresh process with the same `SqliteSaver` file and the same `thread_id`, how do you *resume* (not restart)?

- A) Call `invoke(initial_state, config)` again with the full initial state.
- B) Call `invoke(None, config={"configurable": {"thread_id": "..."}})` — passing `None` as the input tells LangGraph to continue from the last checkpoint; `plan` and `retrieve` do not re-run.
- C) Delete the SQLite file and start over.
- D) There is no way to resume; you must re-run from the start.

---

**Q8.** What does the `thread_id` do?

- A) It sets the number of OS threads the graph uses.
- B) It is the key under which the checkpointer stores a run's state; two invokes with the same `thread_id` share/resume state, two with different `thread_id`s are independent runs.
- C) It selects which model the nodes call.
- D) It is a random UUID LangGraph generates and you never set.

---

**Q9.** Which multi-agent pattern has a central router node that delegates to specialized sub-agents and collects their results?

- A) Swarm.
- B) Supervisor — the default pattern, and the skeleton of the weeks 22–23 capstone.
- C) Hierarchical.
- D) Pipeline.

---

**Q10.** In a *swarm*, who decides what runs next?

- A) A single central supervisor holds the plan.
- B) Each agent decides locally when to *hand off* control to a peer — there is no central router; control is a baton the peers pass to each other.
- C) The checkpointer routes based on state size.
- D) The user picks each step manually.

---

**Q11.** Why does the lecture say CrewAI's role-play (`Agent`/`Task`/`Crew`) abstraction *leaks*?

- A) Because CrewAI is closed source.
- B) Because branching and loops are second-class in a role/task model, state becomes implicit again (hard to checkpoint and resume), and the role labels ("researcher", "writer") are not the control flow — so the moment you need a real conditional re-plan-with-budget edge, you're fighting the abstraction and encoding control flow in prompts.
- C) Because it can't call LLMs.
- D) Because it only supports one agent at a time.

---

**Q12.** When does a deterministic *state machine* beat an agent?

- A) Never — an LLM should always decide the routing.
- B) When the control flow is fixed and known in advance (you can draw the diagram before running it, and it doesn't change run to run): the state machine is cheaper, faster, can't loop forever, and is fully reproducible, because no LLM call is wasted deciding routing that was never in doubt.
- C) Only when there is no model available.
- D) Only for tasks under 100 tokens.

---

**Q13.** In the loop-vs-graph comparison, the graph's pass rate on the 25-task benchmark comes out roughly equal to the week-5 loop's. What is the honest conclusion?

- A) The graph made the agent smarter, and the equal pass rate is a measurement error.
- B) Equal pass rate is the *expected* result — the graph is a *refactor*, not a smarter agent. The wins are not accuracy; they are lines-of-code clarity, observability (you can stream every node), and resumability (it survives a kill), which the loop lacked.
- C) The graph is worse because it didn't improve the pass rate.
- D) The benchmark is broken because both implementations scored the same.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — The real win is *explicit state* (printable, loggable, checkpointable, resumable) replacing the loop's implicit state in the call stack. (Lecture 1 §1.)
2. **B** — A node returns a partial dict; LangGraph merges it per the reducer. You return your delta, not the whole state. (Lecture 1 §3–4.)
3. **B** — `operator.add` on a reducer key appends: `["a","b"] + ["x"] = ["a","b","x"]`. Choosing overwrite where you meant append is the classic state bug. (Lecture 1 §3.)
4. **B** — `add_conditional_edges(source, router_fn, path_map)`: the router's returned key indexes the path map to pick the next node; `"replan"→plan` is the loop-back, `"end"→END` terminates. (Lecture 1 §5.)
5. **B** — The budget lives in the conditional-edge router, the only place the re-plan loop can close. (Lecture 1 §5; Lecture 2 Part 2.)
6. **B** — `MemorySaver` = RAM, dies with the process; `SqliteSaver` = disk, survives a kill and lets a fresh process resume the same `thread_id`. (Lecture 2 §1.2.)
7. **B** — `invoke(None, config=THREAD)` resumes from the last checkpoint; completed nodes don't re-run, so you don't re-pay their tokens. (Lecture 2 §1.3.)
8. **B** — The `thread_id` is the checkpointer's key; same `thread_id` resumes, different `thread_id` is an independent run. (Lecture 2 §1.1.)
9. **B** — Supervisor: a central router delegates to sub-agents and collects results. It's the capstone shape. (Lecture 2 §3.1.)
10. **B** — Swarm: peers hand off to each other locally; no central router. (Lecture 2 §3.2.)
11. **B** — Branching/loops second-class, state implicit (hard to checkpoint/resume), role labels ≠ control flow; you end up encoding control flow in prompts. (Lecture 2 §4.2.)
12. **B** — If you can draw the control flow in advance and it doesn't change run to run, it's a state machine: cheaper, faster, can't loop forever, reproducible. (Lecture 2 §5.)
13. **B** — Equal pass rate is expected; the graph is a refactor. The wins are LoC clarity, observability, and resumability, not accuracy. (Challenge 1; README.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
