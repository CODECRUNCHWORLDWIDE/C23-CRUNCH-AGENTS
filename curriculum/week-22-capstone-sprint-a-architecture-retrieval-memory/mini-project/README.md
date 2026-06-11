# Mini-Project — `crunchcap`: The Capstone Foundation (Retrieval + Memory Substrate)

> Build the foundation of the Production Agentic Research Assistant — the retrieval and memory substrate the agents plug into — as a real package: a corpus ingest, a single `retrieve()` interface over hybrid retrieval, the three memory tiers behind clean read/write interfaces, and the drafted agent contracts. This is not a practice harness; it's the actual first layer of your capstone, the one weeks 23–24 build on.

This is the artifact that turns the capstone from a syllabus paragraph into a repository. After this week, the capstone has a *foundation*: a measured retrieval substrate (`Recall@5` on the gold set), memory that survives a real session (the turn-38 test), and interfaces the agents will click onto. The substrate is corpus-agnostic, interface-stable, and measured — and it produces the Phase II milestone deliverables.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This *is* the capstone foundation. **Week 23 (Sprint B)** builds the supervisor and the four sub-agents directly against the `retrieve()` and memory interfaces you define here. **Week 24 (chaos drill)** corrupts the index this foundation builds and grades your recovery time against the ingest numbers you measure here. Build it well now; everything else in the capstone stands on it, and a foundation with vague interfaces makes the rest of the capstone a re-architecture.

---

## What you will build

A Python package `crunchcap` with four deliverables:

1. **`crunchcap/ingest.py`** — the extraction → clean → chunk → metadata → embed → store pipeline over the corpus, as a *checkpointed, resumable, measured* operation. The single source of truth for "how the corpus becomes a searchable index."
2. **`crunchcap/retrieval.py`** — the single `retrieve(query, k, filters) -> [Chunk]` interface over the hybrid internals (BM25 + dense + RRF + reranker), with `evaluate()` for the gold-set measurement. The contract the agents call.
3. **`crunchcap/memory.py`** — the three memory tiers (episodic, semantic, procedural) behind read/write interfaces, with the regression test. The conversation-state substrate.
4. **`crunchcap/contracts.py`** — the drafted supervisor and sub-agent interfaces (type stubs / protocols), validated against the retrieval and memory interfaces. The shape week 23 implements.

By the end you have the first ~500–700 lines of your capstone repo, plus the architecture document and Mermaid diagram, that weeks 23–24 build directly on.

---

## Why a package and not a notebook

The capstone is a *system*, and a system is a package, not a notebook. A package gives you:

- **The interfaces are real.** `retrieve()` and `memory.read_semantic()` are importable functions with stable signatures — exactly what week 23's agents call. A notebook's cells aren't importable contracts.
- **The measurement is fixed.** The gold-set `evaluate()` and the regression test live in code, version-controlled, re-runnable. "Did the retrieval regress?" is answered by re-running the test, not by eyeballing a cell.
- **It's the capstone repo.** This package *is* the start of the thing you ship. There's no "translate the notebook to production later" — it's production from the first commit.

Notebooks are fine for *exploring* the corpus by eye. The thing you ship and build the agents on is a package. That's not a convention here — it's a requirement, because the agents import it.

---

## Package layout

```
crunchcap/
├── pyproject.toml
├── docker-compose.yml          # pgvector/pgvector:pg17 + (optional) tantivy
├── ARCHITECTURE.md             # the 6-page architecture document
├── architecture.mmd            # the Mermaid diagram (rendered in ARCHITECTURE.md)
├── corpus/                     # the 10 GB private corpus (gitignored; documented)
├── gold/
│   └── questions.jsonl         # 100 gold questions: {"query": "...", "relevant": ["doc_id"]}
├── crunchcap/
│   ├── __init__.py
│   ├── ingest.py               # extract -> clean -> chunk -> embed -> store (checkpointed)
│   ├── retrieval.py            # retrieve() interface + evaluate()
│   ├── memory.py               # episodic + semantic + procedural tiers + interfaces
│   └── contracts.py            # supervisor + sub-agent interface stubs (for week 23)
└── tests/
    ├── test_retrieval.py       # retrieve() returns ranked chunks; gold-set Recall@5 above bar
    ├── test_memory.py          # the turn-38 regression test
    └── test_contracts.py       # the drafted agent contracts cleanly call the interfaces
```

This is your capstone repo's first commit. Week 23 adds `agents/`, `serving/`, `eval/`, `tracing/`; week 24 adds the chaos-drill scripts and the postmortem. The foundation stays.

---

## Deliverable 1 — `ingest.py` (the checkpointed pipeline)

The ingest is an operation, not a script. It must be resumable and measured.

```python
"""crunchcap.ingest — corpus -> hybrid index, checkpointed and measured."""
from __future__ import annotations

import json
import time
from pathlib import Path

CHECKPOINT = Path("ingest_checkpoint.json")


def load_done() -> set[str]:
    if CHECKPOINT.exists():
        return set(json.loads(CHECKPOINT.read_text())["done"])
    return set()


def ingest(corpus_dir: Path, store, chunker, embedder) -> dict:
    """Extract -> clean -> chunk -> metadata -> embed -> store, resumable.
    Returns the operational numbers (the chaos-drill recovery metrics)."""
    done = load_done()
    t0 = time.perf_counter()
    n_chunks = 0
    for doc in sorted(corpus_dir.rglob("*")):
        if doc.name in done:
            continue
        # TODO 1: extract(doc) -> clean() -> chunker.chunk() -> attach metadata
        #   (source_doc_id, section, page, ...) -> embedder.embed_documents() ->
        #   store.insert into BOTH pgvector and tantivy. Then add doc.name to
        #   `done` and SAVE THE CHECKPOINT (so a crash loses at most one doc).
        ...
    ingest_s = time.perf_counter() - t0
    # TODO 2: build the HNSW index; time it separately (rebuild-without-reextract).
    return {"ingest_s": ingest_s, "n_chunks": n_chunks, "index_build_s": ...,
            "index_size_bytes": ...}
```

The non-negotiable: **the metadata schema is complete on the first ingest** (Lecture 1 §4c) — `source_doc_id` and `page` for citation, plus any filter fields the agents will need — because adding a field later is a full re-ingest.

---

## Deliverable 2 — `retrieval.py` (the single interface + evaluate)

The one contract the agents call, hiding the hybrid internals.

```python
def retrieve(query: str, k: int = 5, filters: dict | None = None) -> list[Chunk]:
    """THE interface. Agents call this; they know nothing of the internals."""
    dense = pgvector_search(embed(query), k=20, filters=filters)
    sparse = bm25_search(query, k=20, filters=filters)
    fused = rrf_fuse(dense, sparse)
    # TODO 3: rerank the fused candidates with bge-reranker-v2, return top-k.
    ...


def evaluate(gold: list[dict], retrieve_fn, k: int = 5) -> dict:
    """Recall@k and MRR on the gold set. Reuse your week-9 evaluate() unchanged."""
    # TODO 4: for each gold question, retrieve_fn(query), map chunk hits back to
    #   source_doc_ids, compute Recall@k and MRR. (This is week-9's function.)
    ...
```

Measure layer by layer (dense → +BM25 → +reranker) and report the progression — a reviewer asking "why hybrid?" is answered by the numbers.

---

## Deliverable 3 — `memory.py` (the three tiers + regression test)

The conversation-state substrate, behind read/write interfaces.

```python
class Memory:
    def write_episodic(self, turn: Turn) -> None: ...   # fold into rolling summary
    def read_episodic(self) -> str: ...                  # the bounded summary
    def write_semantic(self, fact: Fact) -> None: ...    # store a durable fact + vec
    def read_semantic(self, query: str, k: int = 5) -> list[Fact]: ...  # by relevance
    def write_procedural(self, action: Action) -> None: ...  # append to the log
    def read_procedural(self, k: int = 10) -> list[Action]: ...  # recent window

    # TODO 5: the EXTRACTION pass -- pull durable facts from a user turn and
    #   write_semantic() them. This is the quiet failure point: a fact never
    #   extracted can never be recalled (the turn-38 test catches it).
```

The non-negotiable: **the turn-38 regression test passes** — a load-bearing fact survives to a late turn, *because* the semantic tier stored it, not because it's still in context.

---

## Deliverable 4 — `contracts.py` (the drafted agent interfaces)

The shape week 23 implements, validated against the foundation now.

```python
from typing import Protocol


class SubAgent(Protocol):
    def __call__(self, task: str, context: "Context") -> "AgentResult": ...


# Drafted contracts (stubs for week 23). Each must cleanly CALL the foundation
# interfaces -- which is how drafting them VALIDATES the interfaces (Lecture 2 §5).
def retrieval_agent(task: str, context) -> "RetrievalResult":
    """Calls retrieve(task) -- so retrieve()'s signature must fit. (It does.)"""
    raise NotImplementedError("Sprint B")

def writing_agent(task: str, context) -> "WritingResult":
    """Reads memory.read_episodic() + retrieved chunks -- so the memory interface
    must give it both. (It does.)"""
    raise NotImplementedError("Sprint B")
# ... code_agent, critique_agent, Supervisor
```

`test_contracts.py` checks that the drafted contracts can *call* the real interfaces (even if they `raise NotImplementedError` for the body) — proving the foundation's shape fits the agents-to-be.

---

## Rules

- **You may** (and should) import your week-8 chunker, week-9 hybrid retriever, and week-11 memory tiers — Sprint A *integrates*, it doesn't rebuild.
- **You must** measure retrieval on the gold set (Recall@5 + MRR) and pass the turn-38 memory regression test — a foundation without these numbers is unproven.
- **You must** make the ingest checkpointed and resumable, and report the operational numbers (the chaos-drill recovery metrics).
- **You must** keep the `retrieve()` and memory interfaces stable — they're the contracts week 23 depends on.
- **You must** write the 6-page `ARCHITECTURE.md` and the Mermaid diagram, and keep the diagram matching the code.
- Any Claude call (the episodic summarizer, the fact extractor, a future supervisor) uses `client.messages.create(...)` with `thinking={"type": "adaptive"}` and `output_config={"effort": ...}` — never `budget_tokens` or `temperature`.
- Python 3.12, your week-8/9/11 packages, `sentence-transformers`, `psycopg[binary]`, `rank-bm25`/`tantivy`, `numpy`, `pytest`.

---

## Acceptance criteria

- [ ] A `capstone/` (or `crunchcap/`) repo — the start of your capstone, not a throwaway.
- [ ] `docker compose up -d` brings up pgvector; the ingest builds the hybrid index.
- [ ] `ingest.py` is checkpointed/resumable and reports the operational numbers (ingest time, index build time, index size, rebuild cost).
- [ ] `retrieve(query, k, filters)` is the single interface; Recall@5 + MRR on the 100-question gold set are reported, measured layer by layer.
- [ ] The three memory tiers are wired behind interfaces; `test_memory.py`'s turn-38 regression test passes.
- [ ] `contracts.py` drafts the supervisor + sub-agent interfaces; `test_contracts.py` shows they cleanly call the foundation interfaces.
- [ ] `ARCHITECTURE.md` (6 pages, six sections) + the Mermaid diagram, matching the code.
- [ ] `pytest` passes; committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Architecture document + diagram** | 20 | 6 pages, six sections (context/components/interfaces/decisions/sequence/risks); Mermaid diagram matches the code; decisions name alternatives. |
| **Corpus ingest** | 15 | Checkpointed, resumable; operational numbers reported (ingest/build time, size, rebuild cost); metadata schema complete. |
| **Hybrid retrieval interface** | 25 | Single `retrieve()` hides the internals; Recall@5 + MRR measured on the gold set, layer by layer; the interface is clean and stable. |
| **Memory tiers** | 25 | Three tiers behind read/write interfaces; the turn-38 regression test passes; extraction works; the tiers are genuinely distinct. |
| **Agent contracts** | 10 | Supervisor + sub-agent contracts drafted; they cleanly call the foundation interfaces (validation). |
| **Tests & hygiene** | 5 | `pytest` green; clean repo; no secrets; sensible commits. |

**90+** is a foundation Sprint B is assembly on. **70–89** works but has a soft interface or a missing measurement. **Below 70** means the foundation is a pile of components, not interfaced layers — fix that first, because Sprint B *imports* these interfaces.

---

## Stretch goals

- **GraphRAG leg** behind `retrieve()` for multi-hop questions; measure the lift (week 10).
- **Salience-weighted eviction** for episodic memory; show a fact survives longer under pressure (week 11).
- **Semantic-cache** the retrieval (week 21) — repeated queries skip retrieval; a cost-report down-payment.
- **The "if I had two more weeks" section** of `ARCHITECTURE.md` — the honest portfolio artifact.

---

## How this connects to the rest of C23

- **Weeks 8/9/11** built the chunker, the hybrid retriever, and the memory tiers; this is where they become one interfaced foundation.
- **Week 19/21** built the serving and cost-routing layers; the architecture doc accounts for them, and Sprint B deploys them.
- **Week 23 (Sprint B)** builds the supervisor and sub-agents *against the contracts you draft here*, on the retrieval and memory interfaces.
- **Week 24 (chaos drill)** corrupts the index *this* ingest builds and grades recovery against *these* operational numbers.

When you've finished, push the repo and take the [quiz](../quiz.md). Then prepare for Sprint B — the agents you contracted this week.
