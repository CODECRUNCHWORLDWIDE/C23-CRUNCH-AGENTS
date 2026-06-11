# Week 22 — Capstone Sprint A: Architecture, Retrieval, Memory

Welcome to the week the capstone stops being a syllabus paragraph and starts being a repository. For twenty-one weeks you built the pieces — a ReAct loop, a chunking A/B harness, a hybrid retriever, three memory tiers, a vLLM cluster, a cost-routing layer. This week you draw the architecture that connects them, write the document that defends it, and *land the foundation* — the retrieval and memory layers of the **Production Agentic Research Assistant** — over a real 10 GB private corpus. By Friday you have a Mermaid diagram that matches running code, a 6-page architecture document, and a retrieval-plus-memory substrate that next week's agents will stand on.

This is week 4 of **Phase IV — Production AI & Capstone**, and it's the first of the three capstone sprints (A here, B in week 23, the chaos drill in week 24). The mantra that sets the week's posture:

> **A capstone is an architecture document with code attached. Write the document first.**

Here is why that's not a writing exercise. The capstone is a *multi-agent system* — a supervisor delegating to retrieval, code, writing, and critique agents, over hybrid retrieval and three memory tiers, served on a vLLM cluster with cost-tracked routing, fully traced and evaluated. That is too much to build by improvisation; the parts have hard dependencies (the supervisor can't be built before the retrieval interface exists; memory can't be wired before the agent loop is defined). The architecture document is how you *sequence* the build — decide the interfaces, the data flow, and the integration points *before* you write the code that has to honor them. A team that codes first and documents later builds four agents that can't talk to each other; a team that documents first builds four agents against an agreed interface. Sprint A is the document plus the foundation it specifies.

There's a corollary worth taping next to it:

> **Sprint A is the foundation, not the demo. Get the retrieval and memory *interfaces* right and the rest of the capstone clicks onto them; get them wrong and you re-architect under deadline in week 23.** The retrieval layer's API (what does `retrieve(query)` return?) and the memory tiers' API (how does an agent read and write episodic / semantic / procedural memory?) are the contracts every later component depends on. This week is where you make those contracts good.

## Learning objectives

By the end of this week, you will be able to:

- **Write** a 6-page architecture document for a production agentic system — components, interfaces, data flow, the build sequence, and the explicit decisions (and rejected alternatives) — and keep a **Mermaid diagram** in sync with the running code.
- **Integrate** Phase I–III work into one coherent foundation: the chunking strategy (week 8), the hybrid retriever — BM25 + dense + reranker (week 9) — over a real corpus, behind a single clean retrieval interface.
- **Ingest and index** a 10 GB private corpus: the extraction → clean → chunk → embed → store pipeline (week 8) at a scale where the operational story (ingest throughput, index build time, rebuild cost) matters.
- **Land** hybrid retrieval — BM25 lexical + dense vector + a reranker — fused (RRF) and measured (MRR / Recall@5) on a gold set, as the capstone's retrieval substrate.
- **Wire** the three memory tiers — **episodic** (rolling turn summary), **semantic** (vector + KG facts about the user/task), **procedural** (tool/action history) — behind interfaces the supervisor and sub-agents will use, with a memory regression test.
- **Define** the supervisor-agent draft and the sub-agent interfaces (retrieval, code, writing, critique) as contracts, even though the agents themselves are built in week 23 — so the foundation is shaped for what plugs into it.
- **Sequence** the remaining capstone work: what's done (Sprint A), what's next (Sprint B agents/MCP/eval/serving), and what's deferred to the chaos drill — a realistic plan, not a wish.

## Prerequisites

This week assumes you have completed **C23 weeks 1–21**, or have equivalent fluency, and — critically — that you have the *working artifacts* from the key weeks, because Sprint A *integrates* them rather than rebuilding them:

- **Week 8** — the `crunchrag_chunk` chunking A/B harness and a chosen chunking strategy. The capstone's ingest uses your winning chunker.
- **Week 9** — the hybrid-retrieval pipeline (BM25 + dense + reranker, RRF fusion) and its `evaluate()`. The capstone's retrieval *is* this, scaled to the real corpus.
- **Week 10** — vector-store operational literacy (pgvector / Qdrant); you pick the store and own its 2-AM story.
- **Week 11** — the three memory tiers and the memory regression test. The capstone wires exactly these.
- **Week 19 / 21** — the vLLM serving tier and the cost-routing layer (you don't deploy them this week, but the architecture document accounts for them).

You'll want a machine that can hold and index a ~10 GB corpus (or a representative subset if disk/compute is tight — the README's notes give a scaled-down path). The retrieval and memory work is mostly CPU/DB-bound; you don't need a GPU this week unless you're running embeddings locally at scale.

## Topics covered

- **The architecture document:** components, interfaces, data flow, build sequence, decisions-and-alternatives; the Mermaid diagram as a living artifact kept in sync with code.
- **Corpus ingestion at scale:** extraction → clean → chunk → metadata → embed → store over 10 GB; ingest throughput, index build time, the rebuild-after-schema-change cost (week-10 operational reasoning, now real).
- **Hybrid retrieval as the substrate:** BM25 (Tantivy/ES) + dense (pgvector/Qdrant) + RRF fusion + a bge-reranker, behind one `retrieve(query) -> ranked_chunks` interface; measured MRR / Recall@5 on a gold set.
- **The three memory tiers:** episodic (rolling/hierarchical summary of turns), semantic (vector + KG facts), procedural (tool/action log); their read/write interfaces; eviction and context budgeting (week 11), now as capstone components.
- **The memory regression test:** does the agent still remember the user's project name in turn 38? — the test that proves memory works across a long session.
- **The supervisor + sub-agent contracts:** the supervisor's delegation interface and the retrieval/code/writing/critique sub-agent signatures, drafted as contracts for week 23 to implement.
- **Sequencing the capstone:** the realistic plan — Sprint A done, Sprint B next, chaos drill last — with dependencies named.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                            | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | The architecture document; the Mermaid diagram; component design |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Corpus ingestion at scale; the retrieval interface              |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | The three memory tiers; the memory regression test              |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Wiring retrieval + memory; the supervisor draft; the build       |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The architecture doc + diagram; the Sprint-A review              |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project (capstone foundation) deep work                     |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, document polish                                    |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                                  | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The capstone spec + rubric, the Mermaid/architecture-doc references, the integration-points docs from weeks 8–11 |
| [lecture-notes/01-architecture-document-and-corpus-ingestion.md](./lecture-notes/01-architecture-document-and-corpus-ingestion.md) | Writing the architecture doc, the Mermaid diagram, and ingesting + hybrid-indexing the 10 GB corpus |
| [lecture-notes/02-memory-tiers-and-the-supervisor-draft.md](./lecture-notes/02-memory-tiers-and-the-supervisor-draft.md) | The three memory tiers, the memory regression test, and the supervisor + sub-agent contracts |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-architecture-diagram.md](./exercises/exercise-01-architecture-diagram.md) | Draw the capstone's Mermaid architecture diagram and write the component/interface table |
| [exercises/exercise-02-retrieval-interface.py](./exercises/exercise-02-retrieval-interface.py) | Define and test the single `retrieve()` interface that fuses BM25 + dense + reranker |
| [exercises/exercise-03-memory-tiers.py](./exercises/exercise-03-memory-tiers.py) | Implement the three memory tiers behind their interfaces and pass the turn-38 regression test |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-sprint-a-foundation.md](./challenges/challenge-01-sprint-a-foundation.md) | The full Sprint A: ingest the corpus, land hybrid retrieval + memory, ship the diagram + 6-page doc |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the 6-page architecture document |
| [mini-project/README.md](./mini-project/README.md) | The capstone foundation — `crunchcap` retrieval + memory substrate the agents plug into |

## The "the foundation holds the agents" promise

C23 uses a recurring marker for every exercise that ends in the capstone foundation working *because* the interfaces were designed right:

```
$ python sprint_a.py --corpus corpus/ --gold gold.json
ingested: 10.2 GB -> 184,302 chunks (recursive-512) in pgvector + tantivy
hybrid retrieval (BM25 + dense + bge-reranker, RRF):
  Recall@5: 0.91   MRR: 0.78   (gold set: 100 questions)
memory tiers wired: episodic + semantic + procedural
  regression test: agent recalls "Project Halibut" at turn 38  ✓
retrieve(query) -> [ranked_chunks]   memory.read(tier) -> facts   [interfaces ready]
```

If that interface line reads "TODO: agents will call this somehow," your foundation isn't a foundation — it's a pile of components the agents can't plug into. The point of Sprint A is to make the retrieval and memory *interfaces* real and tested, so week 23's supervisor and sub-agents click onto them — and to prove the foundation works with a measured Recall@5 and a passing memory regression test, not a vibe about how "the retrieval is basically done."

## Stretch goals

If you finish the regular work early and want to push further:

- Add **GraphRAG** (week 10) over a slice of the corpus — extract an entity graph and let the retrieval interface optionally consult it — and measure whether it lifts Recall@5 on multi-hop questions.
- Implement **salience-weighted eviction** (week 11) for the episodic memory instead of plain rolling summary, and show it keeps a load-bearing fact (the user's project name) alive longer under context pressure.
- Wire the retrieval layer to the **week-21 semantic cache** so repeated queries skip retrieval — a down-payment on the capstone's cost report.
- Write the **"if I had two more weeks" section** of the architecture doc now (the portfolio artifact the career pack wants), naming what you'd improve in the foundation if the sprint were longer — an honest assessment is itself a senior skill.

## Up next

Week 23 is **Capstone Sprint B** — you build the multi-agent supervisor and the retrieval/code/writing agents *on top of* the foundation you land this week, wire the MCP tool surface, deploy the vLLM cluster, and stand up the Ragas + LLM-judge eval suite with OTel tracing. Everything in Sprint B plugs into the retrieval and memory interfaces you design now. Push your `crunchcap` foundation and your architecture document before you start — Sprint B builds directly on both, and a foundation with vague interfaces makes Sprint B a re-architecture instead of an assembly.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
