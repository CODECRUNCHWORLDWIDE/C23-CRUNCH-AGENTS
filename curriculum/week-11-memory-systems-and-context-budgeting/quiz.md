# Week 11 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 12. Answer key is at the bottom — don't peek.

---

**Q1.** What's the difference between *memory* and *retrieval*?

- A) They're the same thing.
- B) Retrieval reads a fixed corpus that exists before the conversation; memory accumulates from the interaction itself (what the user said, what tools ran) — different questions, shared machinery.
- C) Memory is faster than retrieval.
- D) Retrieval only works with vector stores; memory only works with knowledge graphs.

---

**Q2.** Name the three memory tiers and what each holds.

- A) Short, medium, and long — all holding the same transcript.
- B) Episodic (what was said — turn history, rolling summary + recent), semantic (what's true — facts about user/world, vector + KG), procedural (what I did — tool histories and outcomes).
- C) RAM, disk, and cache.
- D) System, user, and assistant messages.

---

**Q3.** A user tells the agent their project name in turn 3. By what mechanism does the agent still know it in turn 38?

- A) The model is stateful and remembers automatically.
- B) The fact was extracted into *semantic memory* (a vector store) at turn 3, and retrieved by similarity to the turn-38 question — surviving even if the rolling summary dropped it.
- C) The full transcript is always in the context window.
- D) The agent re-asks the user.

---

**Q4.** The week's mantra is "context is the most expensive cache on the planet." What does treating it as a cache imply?

- A) Make the context window as large as possible.
- B) The window is small, finite, and expensive, with hits (fact in context) and misses (fact evicted); you manage it with a *budget* and an *eviction policy* instead of appending forever.
- C) Cache the model's weights.
- D) Never use memory.

---

**Q5.** Why is "more context is not more memory" true?

- A) Larger windows cost the same as smaller ones.
- B) A huge window stuffed with raw history is worse than a small one with the right facts — slower, costlier, and (because of lost-in-the-middle) less accurate on the buried facts.
- C) Models can't read more than 4k tokens.
- D) Memory is unrelated to context.

---

**Q6.** What is the rolling-summary strategy, and what is its main weakness?

- A) It stores every turn forever; its weakness is cost.
- B) It keeps a running summary updated each turn (bounding the token footprint); its weakness is that it's lossy — each pass can drop a fact, so durable facts must also go to semantic memory.
- C) It only works on documents; its weakness is speed.
- D) It summarizes images; its weakness is accuracy.

---

**Q7.** When would you choose hierarchical summarization over a flat rolling summary?

- A) Never — flat is always better.
- B) For long conversations where one flat summary becomes too coarse (the specific thing is averaged away); hierarchical keeps summaries at multiple levels so you retrieve at the right resolution.
- C) Only for code.
- D) When you have no vector store.

---

**Q8.** Why must you count the context budget in *tokens* using the model's tokenizer, not characters?

- A) Characters are always larger than tokens.
- B) A token is the model's unit and the budget the model actually sees; counting characters (or `tiktoken` for a Claude model — a different tokenizer that undercounts) gives a budget that lies.
- C) The API rejects character counts.
- D) Tokens are free; characters cost money.

---

**Q9.** What is the "lost in the middle" effect, and why does it make budgeting a *quality* lever?

- A) Models lose tokens in transmission; budgeting prevents data loss.
- B) Long-context models recall content at the start and end far better than the middle (a U-shaped curve), so a tight, edge-placed context where the answer isn't buried produces a *better answer* than a full one — budgeting keeps the answer findable, not just cheaper.
- C) The middle of a conversation is always irrelevant.
- D) It only affects images.

---

**Q10.** Pure LRU eviction drops the oldest content. Why can this fail on a memory benchmark, and what saves it?

- A) LRU never fails.
- B) The *important* fact (the turn-3 project name) is the oldest by turn 38, so LRU evicts it — but it's saved because durable facts were promoted to *semantic memory*, which LRU eviction of the transcript doesn't touch.
- C) LRU only works on caches, not memory.
- D) Nothing saves it; LRU always loses old facts permanently.

---

**Q11.** How does salience-weighted eviction differ from LRU, and when does it win?

- A) It's identical to LRU.
- B) It scores each memory by importance (salience), optionally combined with recency, and evicts the *lowest-scoring* even if recent — so it keeps the important old fact and drops a recent "thanks!"; it wins on memory benchmarks where old facts matter.
- C) It evicts the newest content.
- D) It only works with TTLs.

---

**Q12.** What does the memory regression test measure, and what is its essential control?

- A) Latency; the control is a faster model.
- B) Whether a fact planted early survives to be recalled late (the turn-38 test); the essential control is a *no-memory baseline* (recent window only), so the recall delta measures what memory actually bought.
- C) Token count; the control is a smaller prompt.
- D) Cost; the control is a cheaper API.

---

**Q13.** In the regression test, you ask about the turn-3 fact at turn 6 and the no-memory baseline "passes." What's wrong with the test?

- A) Nothing; turn 6 is fine.
- B) Turn 6 is *within* the recent window, so the baseline still has the fact — the test must ask *late* (after the fact scrolls out of the recent window) to measure survival; a gap shorter than the recent window measures nothing.
- C) The baseline should never pass.
- D) The fact was planted at the wrong turn.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Retrieval reads a pre-existing fixed corpus; memory accumulates from the interaction. Different questions, shared machinery. (Lecture 1 §1.)
2. **B** — Episodic (what was said), semantic (what's true), procedural (what I did). (Lecture 1 §2.)
3. **B** — The fact was promoted to the semantic vector store at turn 3 and retrieved by similarity at turn 38 — surviving summary loss. (Lecture 1 §2.2; Lecture 2 §5.)
4. **B** — Small, finite, expensive cache with hits/misses; manage with a budget + eviction, not endless append. (Lecture 2 §1.)
5. **B** — A full window of raw history is worse than a tight one with the right facts — cost *and* lost-in-the-middle accuracy. (Lecture 1 corollary; Lecture 2 §3.)
6. **B** — Rolling summary bounds tokens but is lossy; durable facts must also go to semantic memory. (Lecture 1 §3.1.)
7. **B** — Hierarchical keeps multi-level resolution for long conversations a flat summary blurs. (Lecture 1 §3.2.)
8. **B** — Count in the model's tokens (the real budget); characters or `tiktoken`-on-Claude lie. (Lecture 2 §2.)
9. **B** — U-shaped recall (edges beat middle) makes a tight, edge-placed context more accurate, not just cheaper. (Lecture 2 §3.)
10. **B** — LRU evicts the old-but-important fact; semantic memory saves it because transcript eviction doesn't touch it. (Lecture 2 §4.1.)
11. **B** — Salience scores by importance (± recency), evicts the lowest even if recent; wins where old facts matter. (Lecture 2 §4.2.)
12. **B** — Does an early fact survive to be recalled late (turn-38); the no-memory baseline is the control for the delta. (Lecture 2 §5.)
13. **B** — Turn 6 is inside the recent window, so the baseline trivially "passes"; ask late to measure survival. (Lecture 2 §5; the challenge's second trap.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
