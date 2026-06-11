# Week 22 Homework

Six problems that build the Sprint A foundation into your fingers. The full set should take about **5 hours**. Work in your **capstone repository** (the `crunchcap` mini-project workspace) — every problem produces a commit that's part of your actual capstone, not throwaway practice.

The headline deliverable is **Problem 6 — the 6-page architecture document**, called out explicitly in the syllabus (the Phase II milestone). Treat it as the artifact a sealed-review panel reads to decide whether your foundation is sound, not a journal entry.

Have your **week-8 chunker, week-9 hybrid retriever, and week-11 memory tiers** importable — Sprint A *integrates* them, and the homework reuses them. Have pgvector running (`docker run -d -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17`). If those prior packages are broken, fix them first — the whole sprint depends on them.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Draw the architecture diagram and component table

**Problem statement.** Write `capstone/ARCHITECTURE.md` (start it) with a Mermaid architecture diagram and a component/interface table (Exercise 1's deliverables). The diagram is at the right level (components + edges, passes the new-engineer-bug test); the table gives each component's responsibility and *interface*.

**Acceptance criteria.**

- A Mermaid diagram in `ARCHITECTURE.md`, committed and rendering.
- A component/interface table with the interface (contract) for each component.
- The diagram passes the level-of-detail test (a new engineer could find the right box for a bug).
- Committed.

**Hint.** If you can't draw an edge between two components, you've found a missing interface — define it, then draw the edge. The diagram exposing a missing connection is the diagram doing its job. Keep it to components and edges; not a call graph.

**Estimated time.** 40 minutes.

---

## Problem 2 — Ingest the corpus (checkpointed and measured)

**Problem statement.** Build the ingest (Lecture 1 §4, your `crunchcap/ingest.py`) over your corpus (≥10 docs, ≥100 pages; ideally ~10 GB, a 1–2 GB subset is acceptable with reported scaled numbers). It must be *checkpointed and resumable*. Produce `notes/week-22/ingest.md` with the operational numbers: ingest time, chunk count, index build time, index size, and the rebuild-after-schema-change estimate.

**Acceptance criteria.**

- The corpus is ingested into pgvector (+ a BM25 index) using your week-8 winning chunker.
- The ingest is checkpointed (a crash loses at most one document, not the run) — demonstrated by interrupting and resuming.
- The operational numbers (ingest time, chunk count, index build time, index size, rebuild cost) are reported.
- Committed.

**Hint.** Save the checkpoint *after* each document commits, and check it at the top of the loop to skip done docs. Checkpoint the embedding stage too — re-embedding 150K chunks because of a late crash is the painful, avoidable cost. These operational numbers are the chaos-drill recovery metrics (week 24); record them now.

**Estimated time.** 50 minutes (plus background ingest time).

---

## Problem 3 — Land the retrieval interface and measure it

**Problem statement.** Build `retrieve(query, k, filters)` (Lecture 1 §5, your `crunchcap/retrieval.py`) over the hybrid internals (BM25 + dense + RRF + reranker), hiding them behind the one interface. Run `evaluate()` on the 100-question gold set and produce `notes/week-22/retrieval.md` with Recall@5 and MRR measured *layer by layer* (dense → +BM25 → +reranker).

**Acceptance criteria.**

- A single `retrieve(query, k, filters)` interface; the hybrid internals are hidden behind it.
- Recall@5 and MRR on the 100-question gold set, reported for each layer (dense, +BM25, +reranker).
- The layer-by-layer progression shows each layer's contribution (and any layer that didn't help is noted, not blindly kept).
- Committed.

**Hint.** Reuse your week-9 `evaluate()` unchanged. Map chunk hits back to source doc ids before scoring (the week-8/9 trap). The layer-by-layer numbers are what answer "why hybrid?" to a reviewer — a single 0.91 hides the story; the 0.78 → 0.85 → 0.91 progression tells it.

**Estimated time.** 50 minutes.

---

## Problem 4 — Wire the three memory tiers and pass the regression test

**Problem statement.** Build the three memory tiers (Lecture 2 §2–3, your `crunchcap/memory.py`) behind read/write interfaces, and pass the turn-38 regression test (a load-bearing fact survives to a late turn). Produce `notes/week-22/memory.md` showing the test passing and which tier preserved the fact.

**Acceptance criteria.**

- Three tiers (episodic, semantic, procedural) behind read/write interfaces.
- The turn-38 regression test passes (`pytest` or a script): a fact planted at turn 1 is recalled at turn 38.
- The notes show the fact is *gone from the episodic summary* but recalled via the semantic tier — proving the architecture, not the context length.
- Committed.

**Hint.** Build and test incrementally (Lecture 2 §3d): get `write_semantic`/`read_semantic` passing a 3-turn version first, then add filler turns, then the full 38. The quiet failure point is *extraction* — if "Halibut" was never extracted to a fact, no read can recall it. Run the test as you build, not at the end.

**Estimated time.** 50 minutes.

---

## Problem 5 — Draft the agent contracts and validate the interfaces

**Problem statement.** Draft the supervisor and sub-agent contracts (Lecture 2 §5, your `crunchcap/contracts.py`) as interface stubs (the bodies `raise NotImplementedError("Sprint B")`). Then write `test_contracts.py` showing each drafted contract can *cleanly call* the foundation interfaces (`retrieve()`, the memory reads). Produce `notes/week-22/contracts.md` noting any interface gap drafting the contracts revealed (and how you fixed it).

**Acceptance criteria.**

- Drafted contracts for the supervisor and the four sub-agents (retrieval, code, writing, critique).
- `test_contracts.py` shows the contracts cleanly call `retrieve()` and the memory interfaces (the call type-checks / runs up to the `NotImplementedError`).
- The notes name any interface gap the drafting revealed and how you resolved it — or confirm the interfaces fit cleanly.
- Committed.

**Hint.** Drafting the writing-agent contract is the best test: it needs *both* retrieved chunks *and* memory (episodic + semantic). If your interfaces don't cleanly give it both, you found a foundation gap — fix it now (Sprint A) instead of mid-Sprint-B. The point of the drafts is to *validate the foundation*, so an interface gap found here is a success, not a failure.

**Estimated time.** 40 minutes.

---

## Problem 6 — The 6-page architecture document (headline deliverable)

**Problem statement.** This is the syllabus deliverable (the Phase II milestone's architecture memo). Complete `capstone/ARCHITECTURE.md` as a **6-page** document against the six-section template (Lecture 1 §2):

1. **Context and goal** — the capstone in a paragraph, the corpus, the success criteria (the eight deliverables).
2. **Component overview** — the boxes (supervisor, agents, retrieval, MCP, memory, serving, eval/tracing), one paragraph each; the annotated Mermaid diagram.
3. **Interfaces and data flow** — the contracts (`retrieve()`, memory reads/writes, the agent signatures) and a one-query data-flow trace.
4. **Key decisions and rejected alternatives** — 4–6 ADR-style entries (store, chunker, serving, memory schema, fusion) with what you chose, rejected, and why.
5. **The build sequence** — what's done (Sprint A), next (Sprint B), deferred (chaos drill), with dependencies named.
6. **Risks and open questions** — 2–3 honest risks (ingest time, index recovery, a memory edge case), and (stretch) the "if I had two more weeks" section.

**Acceptance criteria.**

- `capstone/ARCHITECTURE.md` exists, is roughly 6 pages, and hits all six sections.
- The interfaces-and-data-flow section names the real contracts and includes a one-query trace.
- The decisions section has ≥4 ADR-style entries (chosen / rejected / why).
- The Mermaid diagram matches the code you built in Problems 2–5.
- The build sequence names Sprint A as done and the dependencies for B and the chaos drill.
- Committed.

**Hint.** Six pages is the discipline — if you can't explain the architecture in six pages, you don't yet understand which parts matter, and compressing to six pages is how you figure that out. The decisions and risks sections are what make it an *architecture* document rather than a description; don't skip them. This is a portfolio artifact (the career pack wants it) — write it for a smart reader with fifteen minutes.

**Estimated time.** 1 hour 10 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Architecture diagram + component table | 40 min |
| 2 — Ingest the corpus (checkpointed, measured) | 50 min |
| 3 — Retrieval interface + layer-by-layer measure | 50 min |
| 4 — Three memory tiers + regression test | 50 min |
| 5 — Draft agent contracts + validate interfaces | 40 min |
| 6 — 6-page architecture document (headline) | 1 h 10 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your capstone repo and make sure the `crunchcap` [mini-project](./mini-project/README.md) foundation is in it — Sprint B (week 23) builds the agents *on top of* these interfaces, and the chaos drill (week 24) corrupts this ingest's index. Then take the [quiz](./quiz.md) with your notes closed.

Up next: **Week 23 — Capstone Sprint B**, where you build the multi-agent supervisor and the retrieval/code/writing/critique agents against the contracts you drafted this week, wire the MCP tool surface, deploy the vLLM cluster, and stand up the eval suite with OTel tracing.
