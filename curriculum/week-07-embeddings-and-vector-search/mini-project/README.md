# Mini-Project — `crunchrag_embed`: The Embedding Bakeoff Harness

> Build a reusable embedding-and-evaluation module that any RAG pipeline can import to embed a corpus, index it in pgvector, run a gold-set evaluation, and produce top-1 / Recall@k / MRR — so "which embedding, and how do you know?" becomes a command, not an argument.

This is the artifact that turns retrieval from a vibe into a measurement. After this week, picking an embedding is `python -m crunchrag_embed bakeoff --corpus legal --models bge,gte,nomic` and reading a table — not scrolling MTEB and guessing. The harness is corpus-agnostic, model-agnostic, and metric-honest, and it becomes the spine of the next two weeks.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This module is imported directly by your **week-8 chunking A/B harness** and your **week-9 reranking-lift charts**. The syllabus chunking lab says: "Use the same embedding (BGE-large) and the same vector store (pgvector). Report retrieval MRR, Recall@5..." — that "same embedding, same store, report MRR" *is this module*. Build it well now; you'll lean on it for the rest of Phase II, and you'll defend it at the week-12 architecture review.

---

## What you will build

A small Python package `crunchrag_embed` with four deliverables:

1. **`crunchrag_embed/embedders.py`** — a thin, uniform wrapper over the embedding models, so the rest of the code never has to remember that BGE needs a prefix and Cohere needs an `input_type`. One `Embedder` interface; the per-model conventions live *inside* the wrapper, in one place.
2. **`crunchrag_embed/store.py`** — the pgvector layer: create a table sized to a model's dimension, batch-insert vectors, build an HNSW index, and run a k-NN query. The single source of truth for "how we talk to the vector store."
3. **`crunchrag_embed/eval.py`** — the metrics: `recall_at_k`, `reciprocal_rank`, and `evaluate(gold, retrieve_fn, k)` returning top-1 / Recall@k / MRR. This is the function weeks 8 and 9 call.
4. **`crunchrag_embed/cli.py`** — a `bakeoff` command that ties it together: embed the corpus with N models, run the gold set against each, and print the comparison table.

By the end you have a public repo of ~350–450 lines of Python (excluding the corpus) that any future RAG project can `from crunchrag_embed.eval import evaluate` and stop guessing.

---

## Why a module and not a notebook

You could do all of this in a Jupyter notebook. Don't — not as the artifact. A module gives you:

- **Reuse.** Week 8 imports `evaluate()` unchanged. A notebook gets copy-pasted, drifts, and rots.
- **A fixed measurement.** The gold set and the metric live in code, version-controlled. "Did chunking help?" is answered by re-running the *same* `evaluate()`, not by eyeballing a new notebook.
- **A CLI.** `bakeoff --models bge,gte` is greppable, scriptable, and CI-able. A notebook cell is none of those.

Notebooks are great for *exploring*. The thing you ship and depend on is a module. That's the senior-shop convention.

---

## Package layout

```
crunchrag_embed/
├── pyproject.toml
├── docker-compose.yml          # pgvector/pgvector:pg17 on localhost:5432
├── README.md                   # the bakeoff results table + run commands
├── corpus/
│   ├── legal_clauses.jsonl     # 50 clauses: {"doc_id": "...", "text": "..."}
│   └── gold.json               # 40 queries: [{"query": "...", "relevant": ["clause_14"]}]
├── crunchrag_embed/
│   ├── __init__.py
│   ├── embedders.py            # the uniform Embedder interface (prefix logic lives here)
│   ├── store.py                # pgvector: create / insert / index / knn
│   ├── eval.py                 # recall_at_k, reciprocal_rank, evaluate  (weeks 8-9 import this)
│   └── cli.py                  # the `bakeoff` command
└── tests/
    ├── test_eval.py            # metric correctness on hand-checked cases
    └── test_embedders.py       # prefix/convention applied per model
```

---

## Deliverable 1 — `embedders.py` (the uniform interface)

This is the heart of the project's *correctness*. Every model has a different query/document convention (Lecture 1 §2, §3). The whole point of the wrapper is that the convention is applied **once, in one place**, so no caller can forget it.

```python
"""crunchrag_embed.embedders — one interface over many embedding models.

The per-model query/document conventions (BGE's prefix, nomic's search_query:,
Cohere's input_type) live HERE and nowhere else. Callers just embed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Embedder:
    name: str
    dim: int
    _model: SentenceTransformer

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class BGEEmbedder(Embedder):
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # BGE: NO prefix on documents.
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32)

    def embed_query(self, text: str) -> np.ndarray:
        # BGE: prefix ON queries.
        return self._model.encode(self.QUERY_PREFIX + text, normalize_embeddings=True)


# TODO: GTEEmbedder (no prefix on either side),
#       NomicEmbedder ("search_document: " / "search_query: ", trust_remote_code=True),
#       and OPTIONALLY a vendor embedder (OpenAI/Cohere) behind the same interface.

def load(model_key: str) -> Embedder:
    """Factory: 'bge' -> BGEEmbedder, 'gte' -> GTEEmbedder, 'nomic' -> NomicEmbedder."""
    if model_key == "bge":
        m = SentenceTransformer("BAAI/bge-large-en-v1.5")
        return BGEEmbedder(name="bge-large-en-v1.5", dim=1024, _model=m)
    # TODO: the other models.
    raise ValueError(f"unknown model key: {model_key}")
```

> **The rule the project enforces:** no caller ever prepends a prefix or sets an `input_type`. If `grep -rn "Represent this sentence" --include=*.py | grep -v embedders.py` returns anything, you've leaked the convention out of its one home. The whole reason the module exists is to make the prefix bug impossible.

---

## Deliverable 2 — `store.py` (the pgvector layer)

A small class that owns the database. It must:

- `create_table(name, dim)` — drop-and-create a table sized to the model's dimension (different models need different tables — dimension lock-in, Lecture 1 §7).
- `insert(name, rows)` — batch-insert `(doc_id, content, vector)` using `COPY` for speed.
- `build_hnsw(name)` — `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`.
- `knn(name, query_vec, k)` — `ORDER BY embedding <=> q LIMIT k`, returning `[doc_id]` (and optionally scores). Set `hnsw.ef_search` first.

The `vector_cosine_ops` + `<=>` pairing must match (Lecture 2 §2.2). Encode that pairing in *one* place so it can't drift.

---

## Deliverable 3 — `eval.py` (the metrics — weeks 8–9 import this)

This is the function you'll reuse for the rest of the phase. It must:

- `recall_at_k(retrieved_ids, relevant_ids, k) -> float`
- `reciprocal_rank(retrieved_ids, relevant_ids) -> float`
- `evaluate(gold, retrieve_fn, k=5) -> dict` returning `{"queries", "top1", "Recall@k", "MRR"}`.

Use the implementations from Lecture 2 §3.3 as your starting point. The contract: `retrieve_fn(query)` returns a ranked list of `doc_id`s; `gold` is `[(query, {relevant_ids})]`. Keep `evaluate` **pure** — it takes a retrieve function and a gold set, knows nothing about pgvector or embeddings. That purity is what lets week 8 pass it a *chunking-aware* retriever and week 9 pass it a *reranking* retriever, unchanged.

---

## Deliverable 4 — `cli.py` (the bakeoff command)

```bash
python -m crunchrag_embed bakeoff \
    --corpus corpus/legal_clauses.jsonl \
    --gold corpus/gold.json \
    --models bge,gte,nomic \
    --k 5
```

It should: for each model, load the embedder, create a model-specific table, embed+insert the corpus, build the index, define a `retrieve_fn` that embeds the query (via the embedder's `embed_query`) and calls `store.knn`, run `evaluate`, and collect the row. Then print:

```
MODEL                    DIM   TOP-1   RECALL@5    MRR   EMBED_S
bge-large-en-v1.5       1024    0.62       0.85   0.71      4.1
gte-large-en-v1.5       1024    0.60       0.87   0.70      4.4
nomic-embed-text-v1.5    768    0.58       0.82   0.67      2.9
------------------------------------------------------------------
winner by Recall@5: gte-large-en-v1.5 (0.87)
```

The winner line picks by Recall@5 by default, with a `--metric` flag to switch. The point is a *decision*, printed.

---

## Rules

- **You may** read the model cards, the pgvector docs, `sentence-transformers` source, and the lecture notes.
- **You must not** prepend a query prefix or set an `input_type` anywhere except inside `embedders.py`. The `grep` check above must be clean.
- **You must not** hand-tune the gold set per model — one fixed gold set, all models.
- Python 3.12, `sentence-transformers`, `psycopg[binary]`, `numpy`, plus `pytest`. The vendor leg (OpenAI/Cohere) is optional; the module must work with open models only.
- `evaluate()` must be pure (no DB or model dependency in its signature) so weeks 8–9 can reuse it.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-07-crunchrag-embed-<yourhandle>`.
- [ ] `docker compose up -d` brings up pgvector; the bakeoff runs against it.
- [ ] `embedders.py` implements at least three open models behind one `Embedder` interface, each with its **correct** query/document convention.
- [ ] `grep -rn "Represent this sentence" --include=*.py | grep -v embedders.py` finds **nothing** (the prefix lives in one place).
- [ ] `eval.py` exposes a **pure** `evaluate(gold, retrieve_fn, k)` returning top-1 / Recall@k / MRR.
- [ ] `pytest` passes, with at least:
  - `test_eval.py`: `recall_at_k` and `reciprocal_rank` on hand-checked cases (relevant at rank 1 → RR 1.0; rank 3 → RR 0.333; not found → 0.0).
  - `test_embedders.py`: confirms the query encoding differs from the document encoding for a prefix model (i.e. the convention is actually applied).
- [ ] `python -m crunchrag_embed bakeoff --models bge,gte,nomic` prints a three-row table and a winner line.
- [ ] A `README.md` with the results table, the run commands, and one paragraph on which model you'd ship and why.
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Convention correctness** | 25 | Every model uses its right prefix/input_type; the `grep` check is clean; no model is unfairly starved. |
| **Metric correctness** | 25 | `evaluate` returns correct top-1 / Recall@k / MRR; `eval.py` is pure and reusable; matches the hand-checked test cases. |
| **Store discipline** | 15 | One pgvector layer; `vector_cosine_ops` + `<=>` matched; per-model tables sized to the dimension; batched inserts. |
| **Tests** | 15 | `test_eval` covers the metric edge cases; `test_embedders` proves the convention is applied; `pytest` green. |
| **Bakeoff & decision** | 15 | The CLI runs three models and prints a comparison table with a winner; the README names a pick with a reason. |
| **Docs & hygiene** | 5 | Clear README, no secrets committed, sensible commits, no `__pycache__`/`.venv` checked in. |

**90+** is portfolio-grade and ready to drop into week 8's chunking A/B. **70–89** works but has a convention leak or a soft metric. **Below 70** means the harness isn't actually a fair, reusable measurement — fix that first, because weeks 8 and 9 depend on it.

---

## Stretch goals

- **Vendor leg.** Add an `OpenAIEmbedder` (or `CohereEmbedder`) behind the same interface, with the right `dimensions`/`input_type`, and a `--models bge,gte,openai` run. Put the per-query cost in the table.
- **`ef_search` sweep.** Add `bakeoff --ef-sweep 40,100,200` that runs the winning model at multiple `ef_search` values and shows the embedding choice and index tuning are separate levers.
- **Matryoshka.** Add `--truncate-dims 256` for nomic/jina and report the storage-vs-recall trade-off.
- **CI.** A GitHub Actions workflow that spins up pgvector in a service container, runs `pytest`, and runs a one-model bakeoff headless. Green check on every push.

---

## How this connects to the rest of C23

- **Week 8 (chunking)** imports `evaluate()` and `store.py` to A/B chunking strategies with the *same* embedding and store — the syllabus's "same embedding, same store, report MRR/Recall@5" lab is this module plus a chunker.
- **Week 9 (reranking)** wraps your `retrieve_fn` with a reranker and a BM25 leg and calls the *same* `evaluate()` to chart the lift at each layer.
- **Week 12 (evaluation)** layers Ragas on top; your top-1/Recall@5/MRR are the *retrieval* metrics that sit under Ragas's *answer* metrics.

When you've finished, push the repo and take the [quiz](../quiz.md).
