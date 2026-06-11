# Mini-Project — `crunchstore`: The Vector-Store Adapter + Operational Bakeoff

> Build a reusable vector-store abstraction that any RAG pipeline can import to run the *same* retrieval against pgvector, Qdrant, or Weaviate behind one interface — and an operational bakeoff that scores each store on ingest, latency, filtered-search recall, config complexity, and **time-to-recover** — so "which vector store, and how do you know?" becomes a command, not an argument.

This is the artifact that turns store selection from folklore into an operational measurement. After this week, picking a store is `python -m crunchstore bakeoff --stores pgvector,qdrant,weaviate` and reading a scorecard — not copying a choice from a blog post that benchmarked QPS on a good day. The abstraction is store-agnostic (one `VectorStore` interface), the bakeoff is operations-honest (recovery time and filtered-recall, not just latency), and it reuses week 7's `evaluate()` **unchanged**.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This adapter is imported directly by your **week-11 memory-systems lab**, where the *semantic memory tier* is a vector store — your `crunchstore` interface *is* that tier's backing store. The capstone's hybrid retrieval runs on whichever store this bakeoff picks, and it has to survive the week-24 chaos drill's index-corruption scenario — your recovery drill *here* is the rehearsal. Build it well now; you'll lean on it for the rest of Phase II and the capstone.

---

## What you will build

A small Python package `crunchstore` with four deliverables:

1. **`crunchstore/base.py`** — the uniform `VectorStore` interface (`create`, `upsert`, `search`, `search_filtered`, `snapshot`, `restore`, `drop`), so the rest of the code never has to remember that Qdrant uses payload indexes and pgvector uses GIN indexes. One interface; the per-store quirks live *inside* the adapters.
2. **`crunchstore/adapters/`** — one adapter per store (`pgvector.py`, `qdrant.py`, `weaviate.py`), each implementing the interface, each indexing the filterable metadata fields, each with a real `snapshot`/`restore`.
3. **`crunchstore/bakeoff.py`** — the operational scorecard: for each store, ingest the corpus (timed), run week-7's `evaluate()` (Recall@5/MRR), measure query p50/p95, measure filtered-recall at a selective filter, and run the recovery drill (time-to-recover). Collect a row per store.
4. **`crunchstore/cli.py`** — a `bakeoff` command that ties it together and prints the scorecard with a recommendation line.

By the end you have a public repo of ~450–550 lines of Python that any future RAG project can `from crunchstore.adapters import load` and stop guessing about the store.

---

## Why a module and not a notebook

You could do all of this in a Jupyter notebook. Don't — not as the artifact. A module gives you:

- **Reuse.** Week 11's semantic memory tier imports this adapter; the capstone's retrieval runs on it. A notebook gets copy-pasted, drifts, and rots.
- **A fixed measurement.** The corpus, the gold set, the operational metrics, and the "everything but the store is frozen" discipline live in code, version-controlled. "Did switching to Qdrant help?" is answered by re-running the *same* `bakeoff.py`, not by eyeballing a new cell.
- **A CLI.** `bakeoff --stores qdrant,pgvector` is greppable, scriptable, and CI-able. A notebook cell is none of those.

Notebooks are great for *exploring* one store's behavior by eye. The thing you ship and depend on is a module. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchstore/
├── pyproject.toml
├── docker-compose.yml          # pgvector + qdrant + weaviate, one `up`
├── README.md                   # the bakeoff scorecard + the store memo
├── corpus/
│   ├── legal_docs.jsonl        # {"doc_id", "text", "tenant", "clause_type", "version"}
│   └── gold.json               # 40 queries: [{"query", "relevant": ["clause_14"]}]
├── crunchstore/
│   ├── __init__.py
│   ├── base.py                 # the VectorStore interface
│   ├── adapters/
│   │   ├── pgvector.py
│   │   ├── qdrant.py
│   │   └── weaviate.py
│   ├── bakeoff.py              # the operational scorecard (reuses week-7 evaluate)
│   └── cli.py                  # the `bakeoff` command
└── tests/
    ├── test_adapters.py        # each adapter round-trips upsert -> search
    └── test_recovery.py        # snapshot -> drop -> restore returns Recall@5 to baseline
```

Your week-7 `crunchrag_embed` package is a dependency; `bakeoff.py` imports `evaluate` from it **unchanged**.

---

## Deliverable 1 — `base.py` (the uniform interface)

This is the heart of the project. Every store has a different shape — Qdrant has payload indexes and snapshots, pgvector has GIN indexes and `pg_dump`, Weaviate has a schema and a backup module. The interface hides all of that so `bakeoff.py` treats them identically.

```python
"""crunchstore.base — one interface over production vector stores.

The per-store quirks (payload vs GIN index, snapshot vs pg_dump) live in the
adapters and nowhere else. Callers just create, upsert, search, and recover.
"""
from __future__ import annotations

from typing import Protocol


class VectorStore(Protocol):
    name: str

    def create(self, collection: str, dim: int) -> None:
        """Create the collection AND index the filterable metadata fields."""

    def upsert(self, collection: str,
               rows: list[tuple[str, list[float], dict]]) -> None:
        """rows = (chunk_id, vector, metadata). metadata carries tenant/clause_type."""

    def search(self, collection: str, qv: list[float], k: int) -> list[str]:
        """Unfiltered ANN. Returns chunk_ids in rank order."""

    def search_filtered(self, collection: str, qv: list[float], k: int,
                        where: dict) -> list[str]:
        """Filtered ANN — must NOT post-filter naively (the recall-collapse trap)."""

    def snapshot(self, collection: str) -> str:
        """Back up -> a restore handle. The recovery story lives here."""

    def restore(self, collection: str, handle: str) -> None:
        """Restore from a snapshot handle. Timed in the recovery drill."""

    def drop(self, collection: str) -> None: ...


# TODO 1: a `load(store_name: str) -> VectorStore` factory that returns the right
#   adapter ("pgvector" -> PgvectorStore(), etc.) so callers never import adapters
#   directly — they ask for a store by name.
```

> **The rule the project enforces:** every adapter **indexes its filterable metadata fields** in `create`/`upsert`, and `search_filtered` must use the store's *native* filtering (Qdrant `query_filter`, pgvector `WHERE` on an indexed column), never a naive "ANN then drop in Python." If `search_filtered` fetches a big `k` and filters in Python, you've reintroduced the post-filter recall collapse (Lecture 1 §3) — and `test_recovery`/a filtered-recall test should catch it.

---

## Deliverable 2 — `adapters/` (one per store)

Each adapter implements the interface. The non-obvious parts:

- **pgvector** — HNSW index for vectors, a GIN index on the `jsonb` metadata for filters; `snapshot`/`restore` via `pg_dump`/`pg_restore` (or a table copy). Filtered search is `WHERE meta->>'tenant' = %s ORDER BY embedding <=> %s`.
- **Qdrant** — `create_payload_index` for each filterable field; `search_filtered` via `query_filter`; `snapshot`/`restore` via the real snapshot API (the fast recovery story).
- **Weaviate** — a schema with the filterable properties; `search_filtered` via a `where` filter; `snapshot`/`restore` via the backup module.

```python
# TODO 2: PgvectorStore — port your Exercise 1 adapter; add snapshot/restore via
#   pg_dump/pg_restore (or COPY to/from a file) so the recovery drill is real.

# TODO 3: QdrantStore — port your Exercise 1 adapter; implement snapshot/restore
#   with client.create_snapshot / client.recover_snapshot. This is the fast path.

# TODO 4: WeaviateStore — schema-first; vectorizer=none (you bring vectors);
#   filtered search via Filter; snapshot/restore via the backup module.
```

---

## Deliverable 3 — `bakeoff.py` (the operational scorecard)

```python
from crunchrag_embed.eval import evaluate     # week 7, UNCHANGED


def score_store(store, collection, rows, gold, build_retrieve_fn,
                selective_filter):
    """Ingest (timed) -> Recall@5/MRR -> p50/p95 -> filtered-recall -> time-to-recover."""
    import time
    t0 = time.perf_counter()
    store.create(collection, dim=1024)
    store.upsert(collection, rows)
    ingest_s = time.perf_counter() - t0

    rfn = build_retrieve_fn(store, collection)
    m = evaluate(gold, rfn, k=5)                            # the spine, unchanged
    # TODO 5: measure query p50/p95 over the gold queries.
    # TODO 6: measure filtered-recall at `selective_filter` (the §3 number) — and
    #   assert it does NOT collapse (that's the whole point of native filtering).
    # TODO 7: run the recovery drill: snapshot -> drop -> restore (timed) ->
    #   confirm Recall@5 back to baseline. Return time_to_recover_s.
    ...
```

The non-negotiables `bakeoff.py` enforces:

- **One pipeline, one gold set, one metric.** `evaluate()` is imported unchanged; you do not re-implement Recall@5 or MRR.
- **The operational axes are measured, not just latency** — ingest, filtered-recall *at a selective filter*, and time-to-recover are first-class outputs (the trap is over-weighting latency).
- **Unfiltered recall must agree across stores** (same ANN quality); if it doesn't, an adapter has a bug (wrong distance metric, missing normalization).

---

## Deliverable 4 — `cli.py` (the `bakeoff` command)

```bash
python -m crunchstore bakeoff \
    --corpus corpus/legal_docs.jsonl \
    --gold corpus/gold.json \
    --stores pgvector,qdrant,weaviate \
    --selective-filter tenant=rare \
    --k 5
```

It should run the scorecard against each store and print:

```
STORE      RECALL@5  MRR    INGEST_v/s  Q_p95  FILT_RECALL  TTR     CONFIG_LINES
pgvector     0.88    0.74     8,200     16ms      0.84      31s        14
qdrant       0.88    0.75    12,400     14ms      0.86      47s*       11
weaviate     0.87    0.73     6,900     21ms      0.82      88s        23
--------------------------------------------------------------------------------
recommendation: pgvector for a team already on Postgres (familiar recovery, joins);
   qdrant if filtered/multi-tenant search is the bottleneck (filt_recall 0.86, fast
   snapshots). See memo. (*qdrant TTR is snapshot-restore; pgvector is pg_restore.)
```

The recommendation line makes the *judgment call* the scorecard sets up: the stores agree on unfiltered recall, so the decision is operational — familiarity, filtered-recall, recovery time. The point is a *decision for a stated workload*, printed and defended.

---

## Rules

- **You may** read the store docs, the GraphRAG paper, the lecture notes, and your week-7/8/9 code.
- **You must not** re-implement `evaluate()`, `recall_at_k`, or `reciprocal_rank` — import them from `crunchrag_embed.eval` unchanged.
- **You must not** vary the pipeline, corpus, embedding, or gold set across stores. The whole validity is one-variable-at-a-time.
- **You must not** post-filter naively in `search_filtered` — use each store's native filtering (the recall-collapse trap).
- **You must** run a real recovery drill (snapshot/restore, not just "re-ingest") for at least one store and report time-to-recover.
- Python 3.12, `qdrant-client`, `weaviate-client`, `psycopg[binary]`, `sentence-transformers`, `numpy`, plus `pytest`. Docker for the three store servers.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-10-crunchstore-<yourhandle>`.
- [ ] `docker compose up -d` brings up pgvector, Qdrant, and Weaviate; the bakeoff runs against all three.
- [ ] All three adapters implement the same `VectorStore` interface, each indexing its filterable metadata fields, each with a working `snapshot`/`restore`.
- [ ] `bakeoff.py` imports `evaluate` from `crunchrag_embed` **unchanged** and reports Recall@5, MRR, ingest, p50/p95, filtered-recall, and time-to-recover per store.
- [ ] Unfiltered **Recall@5 agrees** across stores (same ANN quality); filtered-recall is measured **at a selective filter** and does not collapse for the native-filtering stores.
- [ ] `pytest` passes, with at least:
  - `test_adapters.py`: each adapter round-trips `upsert` → `search` and returns the inserted ids.
  - `test_recovery.py`: snapshot → drop → restore returns Recall@5 to baseline for at least one store.
- [ ] `python -m crunchstore bakeoff --stores pgvector,qdrant,weaviate` prints a scorecard with a recommendation line.
- [ ] A `README.md` with the scorecard, the run commands, and the **one-page store memo** (the store you'd ship for a stated workload, its numbers, why it won *operationally*, the trade-off accepted, and the recovery story).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Store abstraction** | 20 | All three stores behind one `VectorStore` interface; adding a store needs no core edit; each indexes its filterable fields. |
| **One-variable validity** | 20 | Pipeline/corpus/embedding/gold fixed; only the store varies; `evaluate()` imported unchanged; unfiltered recall agrees across stores. |
| **Filtered ANN done right** | 20 | `search_filtered` uses native filtering (not Python post-filter); filtered-recall is measured at a *selective* filter and does not collapse. |
| **Recovery drill** | 20 | A real snapshot/restore (not re-ingest) for ≥1 store; time-to-recover measured; Recall@5 confirmed back to baseline; the number is in the memo. |
| **Bakeoff & decision** | 15 | The CLI runs all stores and prints the operational scorecard; the README names a store with numbers and a defensible operational reason. |
| **Docs & hygiene** | 5 | Clear README + memo, no secrets, sensible commits, no data/volumes checked in. |

**90+** is portfolio-grade and ready to back week 11's semantic memory tier. **70–89** works but over-weights latency or has a soft recovery story. **Below 70** means the bakeoff isn't operational or the filtered path collapses — fix that first, because week 11 and the capstone run on this adapter.

---

## Stretch goals

- **GraphRAG leg.** Build a tiny GraphRAG over the corpus (Microsoft's library or a hand-rolled entity graph) and add one global/multi-hop question to the gold set that flat hybrid retrieval misses — the justification for the pattern (Lecture 2 §3).
- **Agentic router.** Add a `--router` mode where a small classifier picks vector / filter / skip per query; measure the lift over the fixed pipeline on a heterogeneous query set (Lecture 2 §4).
- **Three-way recovery.** Run the drill on all three stores (pgvector `pg_restore`, Qdrant snapshot, Weaviate backup) and compare recovery times head-to-head — the heart of a great memo.
- **CI.** A GitHub Actions workflow that spins up the three stores in service containers, runs `pytest`, and runs a two-store headless bakeoff. Green check on every push.

---

## How this connects to the rest of C23

- **Weeks 7–9 (embeddings, chunking, reranking)** gave you the pipeline and `evaluate()`; this bakeoff points them at three stores and picks one operationally.
- **Week 11 (memory systems)** uses a vector store as the *semantic memory tier*; this adapter is that tier's backing store.
- **Week 12 (architecture review)** grades whether you chose your store by measurement (this bakeoff) and can defend its recovery story.
- **Week 24 (chaos drill)** corrupts the vector index on purpose; the recovery drill *here* is the rehearsal, and your time-to-recover number is what you'll be held to.

When you've finished, push the repo and take the [quiz](../quiz.md).
