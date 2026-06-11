# Mini-Project — `crunchrag_hybrid`: The Layered Retriever and Lift Chart

> Build a reusable layered retriever — dense + BM25 + RRF + reranker, with optional HyDE — that wraps your week-7/8 `retrieve_fn`, reuses the week-7 `evaluate()` unchanged, and prints a lift table showing what every layer is worth on the same 40-query gold set. So "is the reranker helping?" becomes a command, not an argument.

This is the artifact that turns "we added the standard RAG layers" into "we measured each layer and here's the number." After this week, justifying a retrieval pipeline is `python -m crunchrag_hybrid lift --corpus legal` and reading a table — not hand-waving about best practices. The module is the spine of your Phase II retrieval stack, and it's the thing you'll deploy and defend.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This module is the retriever you carry into the rest of Phase II. **Week 10 (vector stores in production)** takes this exact layered retriever and swaps the in-memory BM25 leg for Postgres full-text or a real search engine, and pgvector for a production store — without touching the fusion or reranking code, because they sit behind an interface. **Week 12 (evaluation)** layers Ragas on top; your per-layer Recall@5/MRR are the *retrieval* metrics that sit under Ragas's *answer* metrics, and your lift table is what you present at the architecture review. Build it well now; the next three weeks lean on it.

---

## What you will build

A small Python package `crunchrag_hybrid` with four deliverables:

1. **`crunchrag_hybrid/legs.py`** — the individual retrieval legs behind one uniform interface: a `dense` leg (wraps your week-7 `store.knn` + embedder), a `bm25` leg (`rank-bm25`), and an optional `hyde` transform that produces a hypothetical-answer vector for the dense leg. Each leg returns a ranked list of `doc_id`s.
2. **`crunchrag_hybrid/fuse.py`** — Reciprocal Rank Fusion: `rrf_fuse(ranked_lists, k=60)`. Pure, rank-based, no score calibration. The single source of truth for "how we combine retrievers."
3. **`crunchrag_hybrid/rerank.py`** — the cross-encoder reranker: load `BAAI/bge-reranker-v2-m3`, score the first-stage top-k `(query, passage)` pairs, return the reranked top-n. Applied **only** to the first-stage candidates.
4. **`crunchrag_hybrid/cli.py`** — a `lift` command that ties it together: build each layer (bm25 → dense → hybrid → +reranker → +HyDE), run the **same** week-7 `evaluate()` on each over the **same** gold set, and print the cumulative-lift table.

By the end you have a public repo of ~350–450 lines of Python (excluding the corpus) that any future RAG project can `from crunchrag_hybrid import LayeredRetriever` and stop guessing which layers earn their keep.

---

## Why a layered module and not a script

You could wire all this into one script. Don't — not as the artifact. A layered module gives you:

- **Reuse with substitution.** Week 10 swaps the BM25 leg and the store *behind the interface* without touching fusion or reranking. A script gets rewritten; a module gets reconfigured.
- **A fixed measurement.** Each layer is a `retrieve_fn` measured by the *same* `evaluate()`. "Did the reranker help?" is answered by re-running the same function with one more layer, not by eyeballing a new script.
- **A CLI.** `lift --corpus legal` is greppable, scriptable, and CI-able. The lift table regenerates on every corpus or chunking change.

The thing you ship and depend on is a module with clean seams between legs, fusion, and reranking. That's the senior-shop convention, and it's what makes week 10's swap a one-line config change instead of a rewrite.

---

## Package layout

```
crunchrag_hybrid/
├── pyproject.toml
├── docker-compose.yml          # pgvector/pgvector:pg17 on localhost:5432 (reused from week 7)
├── README.md                   # the lift table + run commands
├── corpus/
│   ├── legal_clauses.jsonl     # 50 clauses: {"doc_id": "...", "text": "..."}
│   └── gold.json               # 40 queries: [{"query": "...", "relevant": ["clause_14"]}]
├── crunchrag_hybrid/
│   ├── __init__.py             # exports LayeredRetriever
│   ├── legs.py                 # dense leg, bm25 leg, hyde transform (uniform interface)
│   ├── fuse.py                 # rrf_fuse(ranked_lists, k=60)
│   ├── rerank.py               # CrossEncoder reranker over first-stage top-k
│   ├── retriever.py            # LayeredRetriever: composes legs -> fuse -> rerank
│   └── cli.py                  # the `lift` command
└── tests/
    ├── test_fuse.py            # RRF correctness on hand-checked cases
    └── test_retriever.py       # reranker only sees the first-stage top-k; layers compose
```

Note what's **not** here: you do **not** re-implement `evaluate()`, `store.py`, or the embedder. You **import** them from your week-7 `crunchrag_embed`. This module is strictly the *new* layers (bm25, fusion, reranking, HyDE) plus the glue that measures them with the old function.

---

## Deliverable 1 — `legs.py` (the retrieval legs)

Each leg is a function from a query to a ranked list of `doc_id`s, with a uniform signature so the retriever can compose them. The dense leg wraps your week-7 pipeline; the BM25 leg is new this week.

```python
"""crunchrag_hybrid.legs — the individual retrieval legs, one uniform shape.

Each leg: (query: str, top_k: int) -> list[str]   # ranked doc_ids, best first.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

# Reuse week 7 — do NOT re-implement these.
from crunchrag_embed import embedders
from crunchrag_embed.store import Store


def tokenize(text: str) -> list[str]:
    """Keep $, commas, digits together so '$1,000,000' and ids stay matchable."""
    return re.findall(r"[\w$,]+", text.lower())


class BM25Leg:
    """Lexical retrieval over the corpus. Pure Python, no infra."""

    def __init__(self, corpus: list[tuple[str, str]]) -> None:
        self.doc_ids = [doc_id for doc_id, _ in corpus]
        # id trick: prepend the doc_id so 'clause 14' can match lexically.
        tokenized = [tokenize(f"{doc_id} {text}") for doc_id, text in corpus]
        self.bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)

    def search(self, query: str, top_k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self.doc_ids[i] for i in order]


class DenseLeg:
    """Semantic retrieval: wraps the week-7 embedder + pgvector store."""

    def __init__(self, store: Store, embedder, table: str = "clauses") -> None:
        self.store = store
        self.embedder = embedder       # BGE query prefix lives INSIDE the embedder
        self.table = table

    def search(self, query: str, top_k: int) -> list[str]:
        qvec = self.embedder.embed_query(query)        # prefix applied in embedder
        return self.store.knn(self.table, qvec, k=top_k)

    # TODO 1: add `search_with_vector(self, vec, top_k)` so the HyDE transform can
    #         retrieve with a hypothetical-answer vector instead of the query's.


def hyde_vector(query: str, embedder, llm_client) -> "np.ndarray":
    """HyDE: generate a hypothetical answer, embed THAT (not the query).

    Cite: Gao et al. 2022 (arXiv:2212.10496). The hypothetical may be wrong —
    we only use its EMBEDDING to find the real document.
    """
    # TODO 2: call the LLM (claude-opus-4-8, thinking={"type":"adaptive"}) with a
    #         system prompt that asks for a single plausible clause answering the
    #         query, then embed that text like a DOCUMENT (no BGE query prefix) and
    #         return the vector. See Lecture 2 §5 for the exact call.
    raise NotImplementedError("TODO 2: implement HyDE per Lecture 2 §5")
```

> **The rule the project enforces:** the BGE query prefix lives in *one* place — the week-7 embedder — and nowhere else. The HyDE transform embeds its hypothetical as a *document* (no query prefix), because the hypothetical is document-shaped. If `grep -rn "Represent this sentence"` finds anything outside `crunchrag_embed`, the convention has leaked.

---

## Deliverable 2 — `fuse.py` (Reciprocal Rank Fusion)

The smallest, most important file. Rank-based, no calibration, k=60.

```python
"""crunchrag_hybrid.fuse — Reciprocal Rank Fusion."""
from __future__ import annotations


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """score(d) = sum over rankers of 1 / (k + rank_r(d)), rank 1-based.

    A doc missing from a list contributes 0 from that list (no penalty).
    Returns [(doc_id, score)] best-first.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):   # 1-based — load-bearing
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
```

Three invariants the tests must pin: 1-based rank, missing-doc-contributes-0, and `k` is the fusion constant (not the result count). This file should be < 20 lines and have zero dependencies — that simplicity is the point of RRF.

---

## Deliverable 3 — `rerank.py` (the cross-encoder)

Load `BAAI/bge-reranker-v2-m3` once, score the first-stage top-k pairs, return the reranked top-n. **It must only ever see the first-stage candidates** — never the whole corpus.

```python
"""crunchrag_hybrid.rerank — cross-encoder reranking of first-stage candidates."""
from __future__ import annotations

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],   # (doc_id, passage) — the first-stage top-k ONLY
        top_n: int = 5,
    ) -> list[str]:
        # TODO 3: build (query, passage) pairs, call self.model.predict(pairs),
        #         sort candidates by score descending, return the top_n doc_ids.
        #         The cost is one forward pass per candidate — that's why we only
        #         ever pass the first-stage top-k here, never the whole corpus.
        raise NotImplementedError("TODO 3: implement rerank per Lecture 2 §2")
```

---

## Deliverable 4 — `cli.py` (the lift command)

```bash
python -m crunchrag_hybrid lift \
    --corpus corpus/legal_clauses.jsonl \
    --gold corpus/gold.json \
    --first-stage-k 50 \
    --final-k 5
```

It should build the five layers as `retrieve_fn`s, run the **same** week-7 `evaluate()` on each over the **same** gold set, and print:

```
LAYER                       TOP-1   RECALL@5    MRR     Δ Recall@5
bm25 only                    0.50       0.72   0.60          —
dense only                   0.62       0.85   0.71      +0.13
hybrid (dense + bm25, RRF)   0.68       0.90   0.76      +0.05
  + reranker (bge-v2-m3)     0.78       0.93   0.83      +0.03
  + HyDE                     0.78       0.93   0.82      +0.00
-----------------------------------------------------------------
biggest single lift: dense over bm25 (+0.13 Recall@5)
cheapest meaningful lift: reranker (+0.03 Recall@5, +0.07 MRR)
```

The exact numbers are yours to measure; the *structure* — five rows, a `Δ` column, two summary lines — is fixed. The point is a measured pipeline, printed.

```python
# cli.py sketch — the five layers, each a retrieve_fn handed to the SAME evaluate().
from crunchrag_embed.eval import evaluate   # week-7 function, imported UNCHANGED

def build_layers(retriever, first_stage_k, final_k):
    R = retriever
    return {
        "bm25 only":   lambda q: R.bm25.search(q, final_k),
        "dense only":  lambda q: R.dense.search(q, final_k),
        "hybrid (RRF)": lambda q: [d for d, _ in R.hybrid(q, first_stage_k)][:final_k],
        "+ reranker":  lambda q: R.hybrid_reranked(q, first_stage_k, final_k),
        # TODO 4: add the "+ HyDE" layer: HyDE-vector dense leg -> fuse -> rerank.
    }

def run_lift(retriever, gold, first_stage_k, final_k):
    rows, prev = [], None
    for name, fn in build_layers(retriever, first_stage_k, final_k).items():
        m = evaluate(gold, fn, k=final_k)         # SAME function, SAME gold, every layer
        delta = "—" if prev is None else f"{m['Recall@k'] - prev:+.2f}"
        rows.append((name, m["top1"], m["Recall@k"], m["MRR"], delta))
        prev = m["Recall@k"]
    return rows
```

---

## Rules

- **You may** read the model cards, the `rank-bm25` and `sentence-transformers` docs, and the lecture notes.
- **You must reuse** the week-7 `evaluate()` **unchanged** — import it, don't re-implement it. The same is true of `store.py` and the embedder. This module adds layers; it does not fork the measurement.
- **The reranker must only see the first-stage top-k.** No reranking the whole corpus. The test enforces this.
- **The gold set is fixed** across all five layers — one `gold.json`, every layer.
- **RRF must be correct:** `1/(k + rank)`, k=60, 1-based rank, missing-doc-contributes-0.
- **HyDE embeds the hypothetical as a document** (no query prefix); the prefix stays in the week-7 embedder.
- Python 3.12, `rank-bm25`, `sentence-transformers`, `psycopg[binary]`, `numpy`, plus `pytest`. The HyDE leg needs an LLM client (`anthropic`); it's optional — the module must run the first four layers without it.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-09-crunchrag-hybrid-<yourhandle>`.
- [ ] `docker compose up -d` brings up pgvector; the dense leg runs against it (reusing the week-7 store).
- [ ] `legs.py` implements the BM25 and dense legs behind a uniform `(query, top_k) -> [doc_id]` shape.
- [ ] `fuse.py` exposes a correct `rrf_fuse` (1-based rank, k=60, missing-doc-contributes-0).
- [ ] `rerank.py` reranks **only** the first-stage top-k with `BAAI/bge-reranker-v2-m3`.
- [ ] `cli.py`'s `lift` command imports the week-7 `evaluate()` **unchanged** and prints the five-row lift table with a `Δ Recall@5` column.
- [ ] `grep -rn "Represent this sentence" --include=*.py | grep -v crunchrag_embed` finds **nothing** (the prefix stayed in week 7's embedder).
- [ ] `pytest` passes, with at least:
  - `test_fuse.py`: `rrf_fuse` on hand-checked cases (the Lecture 1 §3.3 worked example; a doc in both lists beats a doc #1 in one and absent from the other).
  - `test_retriever.py`: the reranker receives exactly the first-stage top-k (not the corpus), and the layers compose.
- [ ] A `README.md` with the lift table, the run commands, and one paragraph on which layer you'd ship and which you'd cut (and why).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Reuse discipline** | 20 | `evaluate()`, `store.py`, the embedder imported from week 7 **unchanged**; no forked measurement; the prefix-leak `grep` is clean. |
| **Fusion correctness** | 20 | `rrf_fuse` is correct (1-based, k=60, missing→0); matches the hand-checked tests; rank-based, not score-normalized. |
| **Reranker discipline** | 20 | Cross-encoder applied to first-stage top-k only (test-enforced); `bge-reranker-v2-m3` loaded once; reranked top-n returned. |
| **The lift table** | 20 | Five layers, same gold set, same `evaluate()`; the `Δ` column is computed; the table is honest (a +0.00 row is reported, not hidden). |
| **Tests** | 15 | `test_fuse` covers RRF edge cases; `test_retriever` proves the top-k discipline; `pytest` green. |
| **Docs & hygiene** | 5 | Clear README with the table and a ship/cut decision; no secrets; sensible commits; no `__pycache__`/`.venv` checked in. |

**90+** is portfolio-grade and ready to drop into week 10's production-store swap. **70–89** works but has a reuse leak (re-implemented `evaluate()`) or reranks too wide. **Below 70** means the lift table isn't a fair, reproducible measurement — fix that first, because week 10 and the week-12 review depend on it.

---

## Stretch goals

- **ColBERT leg.** Add a late-interaction leg (RAGatouille, `answerai-colbert-small-v1`) as its own row in the lift table. Where does token-level MaxSim land between your bi-encoder and your cross-encoder, on quality and latency?
- **Latency column.** Add a median `ms/query` column per layer so the table answers "was it worth it?" not just "did it help?" The reranker's lift-per-ms is the senior-review number.
- **The `text2sql` leg.** Add a structured-retrieval path: a 3-table SQLite schema of the same contracts, a read-only role, a schema allowlist, and a `LayeredRetriever.sql(question)` that answers "which agreements expire before 2027?" — a question the vector store can't answer. Validate the generated SQL is a single read-only SELECT (Lecture 2 §6).
- **CI.** A GitHub Actions workflow that spins up pgvector in a service container, runs `pytest`, and runs a headless `lift` over a few queries. Green check on every push.

---

## How this connects to the rest of C23

- **Week 10 (vector stores in production)** takes this `LayeredRetriever` and swaps the in-memory BM25 leg for Postgres full-text or Tantivy, and pgvector for a production store — *behind the interface*, so fusion and reranking don't change. The clean seams you build now are what make that swap a config change.
- **Week 12 (evaluation)** layers Ragas on top; your per-layer Recall@5/MRR are the *retrieval* metrics under Ragas's *answer* metrics, and your lift table is the artifact you present at the architecture review.
- The whole of Phase II's retrieval story is this module: week 7 gave you the dense leg and the metric, week 8 gave you the chunking, and week 9 wraps it all in the layered retriever you'll deploy.

When you've finished, push the repo and take the [quiz](../quiz.md).
