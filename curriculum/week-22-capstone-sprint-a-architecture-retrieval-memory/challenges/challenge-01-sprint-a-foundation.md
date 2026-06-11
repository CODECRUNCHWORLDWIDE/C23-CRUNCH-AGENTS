# Challenge 1 — The Sprint A Foundation

**Time estimate:** ~150+ minutes (this is the sprint; the corpus ingest runs in the background while you work on the rest).

## Problem statement

This is Capstone Sprint A. You are building the foundation of the **Production Agentic Research Assistant** — the retrieval and memory substrate, the architecture document, and the agent contracts — that weeks 23–24 stand on. By the end you have a corpus ingested and hybrid-indexed (with a measured Recall@5), three memory tiers wired (passing the turn-38 regression test), the supervisor and sub-agent contracts drafted, a Mermaid diagram that matches the code, and a 6-page architecture document. This is the Phase II milestone, made real.

This is the syllabus capstone-sprint deliverable. The output is a **foundation** — interfaced, measured, documented — that makes Sprint B assembly rather than improvisation, not a half-built system you'll re-architect under deadline.

## What you build

Five things, in dependency order:

1. **The architecture document + Mermaid diagram** (write this *first*, Lecture 1 §1) — components, interfaces, data flow, decisions-and-alternatives, build sequence, risks. The diagram matches the code you write.
2. **The corpus ingest** — extraction → clean → chunk (your week-8 winner) → metadata → embed → store (pgvector + Tantivy), over the 10 GB corpus, as a *checkpointed, resumable, measured* operation. Report ingest time, index build time, index size, rebuild cost.
3. **Hybrid retrieval behind `retrieve()`** — BM25 + dense + RRF + reranker (your week-9 pipeline), hidden behind the single `retrieve(query, k, filters)` interface. Measured: Recall@5 and MRR on the 100-question gold set, layer by layer.
4. **The three memory tiers behind their interfaces** — episodic, semantic, procedural (your week-11 tiers), with read/write per tier. Passing: the turn-38 regression test.
5. **The supervisor + sub-agent contracts** — drafted (not implemented): the supervisor's `run()` and the retrieval/code/writing/critique signatures, validated against the foundation interfaces.

## What's fixed (the milestone bar)

- **The corpus:** ≥10 documents, ≥100 pages (the Phase II bar), ideally ~10 GB. A 1–2 GB representative subset is acceptable *if* you report the scaled numbers and the doc reasons about full scale.
- **The gold set:** 100 questions over the corpus (the capstone eval bar) for the Recall@5/MRR measurement.
- **The interfaces:** `retrieve(query, k, filters) -> [Chunk]` and `memory.read_*/write_*` per tier — the contracts week 23 depends on. Design them to be stable.
- **The measurements:** Recall@5 + MRR on the gold set (retrieval); the turn-38 regression test (memory); the ingest operational numbers.

## Acceptance criteria

- [ ] A `capstone/` repo (the start of your capstone) with a runnable `sprint_a.py` (or equivalent) that ingests the corpus, builds the hybrid index, and exposes `retrieve()`.
- [ ] The corpus is ingested and hybrid-indexed; the ingest is **checkpointed and resumable**, and the operational numbers (ingest time, index build time, index size, rebuild cost) are reported.
- [ ] `retrieve(query, k, filters)` is the single interface; the BM25/dense/RRF/reranker internals are hidden behind it. **Recall@5 and MRR on the 100-question gold set are reported, measured layer by layer** (dense → +BM25 → +reranker).
- [ ] The three memory tiers are wired behind read/write interfaces and the **turn-38 regression test passes** (a load-bearing fact survives to a late turn).
- [ ] The supervisor + sub-agent contracts are drafted and validated against the foundation interfaces (the agents-to-be can cleanly call `retrieve()` and the memory reads).
- [ ] A **Mermaid architecture diagram** committed to the repo, matching the code.
- [ ] A **6-page architecture document** (`ARCHITECTURE.md`) with the six sections (context, components, interfaces/data-flow, decisions-and-alternatives, build sequence, risks).
- [ ] One **per-query trace** in the promise format showing the foundation works: `retrieve("...") -> [ranked chunks]; memory recalls "Project Halibut" at turn 38 ✓`.

## The trap (read after a first attempt)

The trap is **building components instead of a foundation.** It's tempting to get the chunker working, then the retriever, then the memory tiers, as three separate scripts — and end the week with three things that work in isolation and don't connect. That's not a foundation; it's a pile. **The foundation is the *interfaces* — `retrieve()` and the memory reads — that the agents plug into**, and a pile of working components with no clean interfaces between them is exactly what makes Sprint B a re-architecture. The test: can you draft the week-23 retrieval agent's contract and have it *cleanly call* your `retrieve()`? If yes, you have a foundation. If the agent contract needs a retrieval signature your `retrieve()` doesn't offer, you have a pile — fix the interface now.

A second, subtler trap: **skipping the architecture document because "the code is the design."** The code is *a* design — an implicit, undocumented one that lives in your head and breaks when you (or a teammate, or future-you) need to reason about the system. The document is where the *decisions* (why pgvector, why this chunker, why three memory tiers) are recorded with their alternatives, where the *risks* are named, and where the *build sequence* is planned. A capstone with no architecture document fails the Phase II milestone and reads, to a sealed-review panel, as "I built something and I'm not sure why it's shaped this way." Write the document; it's graded.

## Stretch goals

- **GraphRAG leg.** Extract an entity graph over a corpus slice and let `retrieve()` optionally consult it for multi-hop questions; measure the Recall@5 lift on the multi-hop gold questions (week 10).
- **Salience-weighted episodic eviction.** Replace plain rolling-summary truncation with salience-weighted eviction, and show it keeps a load-bearing fact alive longer under context pressure (week 11).
- **Semantic-cache the retrieval.** Wire the week-21 semantic cache in front of `retrieve()` so repeated queries skip retrieval — a down-payment on the capstone's cost report.
- **The "if I had two more weeks" section.** Write the portfolio artifact (career pack): an honest assessment of what you'd improve in the foundation with more time. Naming what's imperfect is a senior skill.

## Why this matters

This *is* the capstone foundation — weeks 23 and 24 build the agents, MCP, serving, eval, and chaos drill *on top of* the retrieval and memory interfaces you land here. The Phase II milestone is graded on exactly this: hybrid retrieval over a real corpus, chunking A/B'd, three memory tiers, a 6-page architecture memo. A reviewer will read your `ARCHITECTURE.md`, run your `retrieve()`, and check your regression test — and decide whether you have a foundation a multi-agent system can stand on. The interfaces held, the retrieval measured, the memory survived to turn 38, and you can defend every decision in the document. That's a Sprint A that makes the rest of the capstone fast.
