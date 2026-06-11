# Week 22 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 23. Answer key is at the bottom — don't peek.

---

**Q1.** Why does the architecture document come *before* the code in a capstone?

- A) It's a formality the rubric requires; the code is what matters.
- B) The parts have hard dependencies (the supervisor needs the retrieval interface first), the decisions need to be made once with reasons, and the document is itself a graded deliverable — so writing it first sequences the build correctly and records the decisions.
- C) Writing documentation is faster than writing code.
- D) The document replaces the need for tests.

---

**Q2.** Which section of the 6-page architecture document is the most important, and why?

- A) The context section, because it states the goal.
- B) The interfaces-and-data-flow section, because that's where the system is actually *defined* — the contracts between components — and vague interfaces here become re-architecture in Sprint B.
- C) The risks section, because risks are scary.
- D) They're all equally important; there's no priority.

---

**Q3.** Why is Mermaid (diagram-as-code) the right tool for the capstone's architecture diagram?

- A) It produces prettier diagrams than other tools.
- B) It's text that lives in the repo, diffs, and can be changed in the same commit as a code change — which is what lets it *stay in sync* with the running system. A drawn-image diagram drifts; a diagram-as-code one is version-controlled with the code.
- C) It's the only diagram tool that renders on GitHub.
- D) It requires no learning.

---

**Q4.** What makes the `retrieve(query, k, filters)` interface the most important design move in Sprint A?

- A) It's the fastest retrieval implementation.
- B) It hides the hybrid internals (BM25/dense/RRF/reranker) from the agents, so the agents call `retrieve(query)` and don't know how it works — which lets you evolve the internals (add GraphRAG, swap the reranker, restore after corruption) without touching a single agent.
- C) It's required by pgvector.
- D) It makes retrieval slower but more accurate.

---

**Q5.** At 10 GB, why must the corpus ingest be checkpointed and resumable?

- A) Postgres requires checkpoints.
- B) The ingest is hours of work; a crash at document 9,000 of 10,000 should cost minutes, not the whole run — and the ability to re-run the ingest reliably is exactly what week 24's chaos drill (restore from backup) requires under time pressure.
- C) Checkpointing makes the ingest faster.
- D) It's optional; you can always restart from zero.

---

**Q6.** Why is the chunk metadata schema "a commitment" that must be complete on the first ingest?

- A) Metadata is read-only after ingest by law.
- B) Adding a metadata field later means re-extracting and re-attaching it to every chunk — a full 2+ hour re-ingest — so you must enumerate every field the *whole capstone* needs (citation, filtering, memory) before the big ingest, not after.
- C) Metadata fields can't be changed in pgvector.
- D) The schema is fixed by the embedding model.

---

**Q7.** Reciprocal Rank Fusion (RRF) combines the dense and sparse rankings. How, and why is it attractive?

- A) It averages the two similarity scores; it needs careful tuning.
- B) It sums `1/(k + rank)` across both rankings, rewarding chunks that rank high in *either* signal — no training, no score-normalization, just rank-based fusion. It's attractive because it's a few lines with no tuning and it captures the "dense catches meaning, sparse catches exact terms" complementarity.
- C) It trains a model to merge the rankings; it's expensive.
- D) It picks whichever ranking is longer.

---

**Q8.** The three memory tiers answer three different questions. Match them.

- A) They all answer "what was said?"; they're redundant copies.
- B) Episodic = "what have we said?" (rolling summary); semantic = "what do we know?" (facts as vectors); procedural = "what have we done?" (action log). They're complementary — drop any one and the agent is missing a piece of its state.
- C) Episodic = facts, semantic = actions, procedural = summary.
- D) All three are the same vector store with different names.

---

**Q9.** In the turn-38 regression test, a fact planted at turn 1 is recalled at turn 38. *Why* does it survive, and what does the test prove?

- A) It survives because the model has a long context window; the test proves the model.
- B) It survives because the *semantic* tier stored it as a durable fact at turn 1 (the raw transcript of turns 1–37 is long gone from active context), and `read_semantic` surfaced it for the recall query. The test proves the memory *architecture*, not the model's context length.
- C) It survives in the episodic summary; the test proves summarization.
- D) It survives by luck; the test proves nothing.

---

**Q10.** When the turn-38 test *fails*, the tiers make it diagnosable. What's the diagnosis path?

- A) There's no way to diagnose it; you start over.
- B) Walk the fact's path: was it EXTRACTED to a semantic fact when said? STORED with a usable embedding? RETRIEVED by the recall query? INCLUDED in the turn-38 context? Each checkpoint is a different tier/stage and a different fix — so the failure points at the broken stage.
- C) Increase the context window until it passes.
- D) Re-run the test until it passes by chance.

---

**Q11.** Why draft the supervisor and sub-agent *contracts* in Sprint A when the agents are built in Sprint B?

- A) To have something to grade.
- B) Because drafting the consumer (the agent contracts) *validates* the producer (the foundation interfaces) — if the agents you'll build can't cleanly call `retrieve()` and the memory reads, the interfaces are wrong, and finding that in Sprint A is cheap while finding it in Sprint B is a re-architecture.
- C) The agents must be built this week.
- D) Contracts are easier to write than diagrams.

---

**Q12.** How do you distinguish *retrieval* from *semantic memory*, given both use a vector store?

- A) They're the same thing.
- B) Retrieval pulls from the *corpus* (fixed, external, same for every user — "what's in the documents?"); semantic memory pulls from the *conversation* (per-session/per-user facts — "what do I know about this user and task?"). Both use vectors, but they store different things and answer different questions.
- C) Retrieval uses BM25; memory never does.
- D) Memory is faster than retrieval.

---

**Q13.** What does it mean to say "Sprint A is done"?

- A) The whole capstone works end to end.
- B) The foundation is landed: the architecture doc + diagram match the code; the corpus is ingested and hybrid-indexed with a measured Recall@5; the three memory tiers pass the regression test; and the agent contracts are *drafted* (not implemented). The agents, serving, and eval are Sprint B — you land the foundation cleanly and stop.
- C) All four agents are built and tested.
- D) The chaos drill has been run.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Hard dependencies, decisions-once-with-reasons, and a graded deliverable; writing first sequences the build. (Lecture 1 §1.)
2. **B** — Interfaces-and-data-flow defines the system; vague there = re-architecture later. (Lecture 1 §2.)
3. **B** — Diagram-as-code lives in the repo, diffs, and stays in sync; a drawn image drifts. (Lecture 1 §3.)
4. **B** — The interface hides the internals so they evolve without touching the agents — the key decoupling. (Lecture 1 §5, §5c.)
5. **B** — A crash mustn't cost the whole multi-hour run; resumable re-ingest is the chaos-drill recovery requirement. (Lecture 1 §4, §4b.)
6. **B** — Adding a field later means a full re-ingest; enumerate the whole capstone's needs first. (Lecture 1 §4c.)
7. **B** — RRF sums `1/(k+rank)`, rewarding either-signal high ranks; no tuning, captures dense+sparse complementarity. (Lecture 1 §5.)
8. **B** — Episodic (said) / semantic (know) / procedural (done); complementary, not redundant. (Lecture 2 §1.)
9. **B** — The semantic tier stored the fact; `read_semantic` recalled it; the test proves the architecture, not context length. (Lecture 2 §3.)
10. **B** — Walk extract → store → retrieve → include; each checkpoint is a different tier/stage and fix. (Lecture 2 §3b.)
11. **B** — Drafting the consumers validates the producer interfaces; cheap to fix a mismatch now, a re-architecture later. (Lecture 2 §5.)
12. **B** — Retrieval = corpus (external, fixed); semantic memory = conversation (per-session facts); both use vectors, different content. (Lecture 2 §1.)
13. **B** — The foundation is landed and measured; the agents/serving/eval are Sprint B. Land cleanly and stop. (Lecture 2 §6.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
