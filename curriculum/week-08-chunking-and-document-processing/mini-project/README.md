# Mini-Project — `crunchrag_chunk`: The Chunking A/B Harness

> Build a reusable chunking-and-evaluation module that any RAG pipeline can import to chunk a corpus five different ways, embed with a fixed model, retrieve from a fixed store, and report Recall@5 / MRR / faithfulness — so "which chunking strategy, and how do you know?" becomes a command, not an argument.

This is the artifact that turns chunking from folklore into a measurement. After this week, picking a chunker is `python -m crunchrag_chunk ab --strategies fixed512,recursive,semantic,late` and reading a table — not copying a `chunk_size` from a blog post. The harness is corpus-agnostic, strategy-pluggable, and metric-honest, and it reuses week 7's `evaluate()` and `store.py` **unchanged**.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This module is imported directly by your **week-9 reranking-and-hybrid-search lab**. Week 9 takes your *winning* chunking strategy from this harness, adds a BM25 lexical leg and a reranker on top, and charts the lift at each layer — using the same `evaluate()` you call here. The syllabus says the reranking lab starts from "your best chunking strategy from week 8"; *this harness is how you found it*. Build it well now; you'll lean on it for the rest of Phase II and defend it at the week-12 architecture review.

---

## What you will build

A small Python package `crunchrag_chunk` with four deliverables:

1. **`crunchrag_chunk/chunkers.py`** — a uniform `Chunker` interface over the five strategies (fixed token-window, sliding-window, recursive, semantic-paragraph, late chunking), so the rest of the code never has to remember that late chunking embeds differently or that semantic chunking needs sentence vectors. One interface; the per-strategy quirks live *inside* the wrappers.
2. **`crunchrag_chunk/pipeline.py`** — the extraction→clean→chunk→metadata glue: take a corpus, run it through a chosen chunker, and emit `(chunk_id, source_doc_id, text, metadata)` rows ready to embed. The single source of truth for "how a document becomes chunks."
3. **`crunchrag_chunk/ab.py`** — the A/B loop: for each strategy, build a `retrieve_fn` (chunk → embed with the fixed model → index in the fixed store → retrieve → map chunks back to source ids), call week-7's pure `evaluate()`, collect the row, and pick a winner. Faithfulness is a pluggable secondary metric.
4. **`crunchrag_chunk/cli.py`** — an `ab` command that ties it together and prints the comparison table with a winner line.

By the end you have a public repo of ~400–500 lines of Python (excluding the corpus) that any future RAG project can `from crunchrag_chunk.chunkers import load` and stop guessing about chunk size.

---

## Why a module and not a notebook

You could do all of this in a Jupyter notebook. Don't — not as the artifact. A module gives you:

- **Reuse.** Week 9 imports your A/B loop and your winning chunker. A notebook gets copy-pasted, drifts, and rots.
- **A fixed measurement.** The gold set, the metric, and the "everything but the chunker is frozen" discipline live in code, version-controlled. "Did this chunker help?" is answered by re-running the *same* `ab.py`, not by eyeballing a new notebook cell.
- **A CLI.** `ab --strategies fixed512,recursive,late` is greppable, scriptable, and CI-able. A notebook cell is none of those.

Notebooks are great for *exploring* a single document's boundaries by eye (Exercise 2 territory). The thing you ship and depend on is a module. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchrag_chunk/
├── pyproject.toml
├── docker-compose.yml          # pgvector/pgvector:pg17 on localhost:5432 (from week 7)
├── README.md                   # the A/B results table + the winner memo
├── corpus/
│   ├── legal_docs.jsonl        # multi-clause documents: {"doc_id": "...", "text": "..."}
│   └── gold.json               # 40 queries: [{"query": "...", "relevant": ["clause_14"]}]
├── crunchrag_chunk/
│   ├── __init__.py
│   ├── chunkers.py             # the uniform Chunker interface (5 strategies)
│   ├── pipeline.py             # extract -> clean -> chunk -> metadata
│   ├── ab.py                   # the A/B loop (reuses week-7 evaluate + store)
│   ├── faithfulness.py         # the lightweight LLM-as-judge tie-breaker
│   └── cli.py                  # the `ab` command
└── tests/
    ├── test_chunkers.py        # boundary correctness per strategy
    └── test_ab_mapping.py      # chunk-id -> source-id mapping is correct
```

Your week-7 `crunchrag_embed` package is a dependency (installed editable or vendored); `ab.py` imports `evaluate` and `store` from it **unchanged**.

---

## Deliverable 1 — `chunkers.py` (the uniform interface)

This is the heart of the project. Every strategy has a different shape — fixed/sliding/recursive return text chunks and embed normally; semantic needs sentence vectors to find boundaries; late chunking returns *spans* and embeds the whole document first. The wrapper hides all of that behind one interface so `ab.py` treats them identically.

```python
"""crunchrag_chunk.chunkers — one interface over five chunking strategies.

The per-strategy quirks (late chunking's whole-doc forward pass, semantic
chunking's sentence vectors) live HERE and nowhere else. Callers just chunk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer

_TOK = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")


def n_tokens(text: str) -> int:
    return len(_TOK.encode(text, add_special_tokens=False))


@dataclass
class Chunk:
    text: str
    token_span: tuple[int, int] | None = None   # set only for late chunking


class Chunker:
    name: str

    def chunk(self, text: str) -> list[Chunk]:
        raise NotImplementedError


class FixedTokenChunker(Chunker):
    name = "fixed512"

    def __init__(self, size: int = 512, overlap: int = 0) -> None:
        self.size, self.overlap = size, overlap

    def chunk(self, text: str) -> list[Chunk]:
        ids = _TOK.encode(text, add_special_tokens=False)
        step = self.size - self.overlap
        out = []
        for start in range(0, len(ids), step):
            window = ids[start:start + self.size]
            if not window:
                break
            out.append(Chunk(text=_TOK.decode(window),
                             token_span=(start, start + len(window))))
            if start + self.size >= len(ids):
                break
        return out


# TODO 1: SlidingWindowChunker — same as FixedTokenChunker but overlap > 0 by
#   default (e.g. size=512, overlap=64). You may subclass FixedTokenChunker.

# TODO 2: RecursiveChunker — split on ("\n\n", "\n", ". ", " ", "") with
#   length measured by n_tokens(); recurse into pieces still over `size`.
#   (Port your Exercise 2 recursive_chunks; wrap each piece in a Chunk.)

# TODO 3: SemanticChunker — embed each sentence with BGE, compute adjacent-
#   sentence cosine distance, split at the `percentile`-th distance spike
#   (Lecture 1 §5). Takes a SentenceTransformer in __init__.

# TODO 4: LateChunker — this one is special. It does NOT embed here; it returns
#   Chunks with token_span set (boundaries only, via any sub-strategy). The
#   whole-document forward pass + per-span mean-pool happens in the embedder
#   path in ab.py (Lecture 1 §6), because late chunking changes HOW chunks are
#   embedded, not just where they're cut.


def load(strategy: str, **kwargs) -> Chunker:
    """Factory: 'fixed512' -> FixedTokenChunker, 'recursive' -> RecursiveChunker, ..."""
    if strategy == "fixed512":
        return FixedTokenChunker(size=512, **kwargs)
    if strategy == "fixed1024":
        return FixedTokenChunker(size=1024, **kwargs)
    # TODO 5: wire up 'sliding', 'recursive', 'semantic', 'late'.
    raise ValueError(f"unknown strategy: {strategy}")
```

> **The rule the project enforces:** no caller measures chunk size in characters. Every size is in `n_tokens()` — the BGE tokenizer's units, the budget the encoder actually sees. If `grep -rn "len(" --include=*.py crunchrag_chunk | grep -i "chunk_size\|size ="` shows a character-length size, you've reintroduced the classic bug.

---

## Deliverable 2 — `pipeline.py` (extract → clean → chunk → metadata)

The glue that turns a corpus into embeddable rows. It must:

- Load the corpus (`legal_docs.jsonl`: `{"doc_id", "text"}` per line).
- **Clean** each document (rejoin hyphenation, collapse whitespace, drop obvious boilerplate) — Lecture 2 §4. Even on a clean synthetic corpus, the cleaning step must exist and be a no-op-safe function, because real corpora need it and week 9 reuses this.
- **Chunk** each document with a given `Chunker`.
- **Attach metadata** to each chunk: its `source_doc_id` (the clause id, for gold scoring), a stable `chunk_id`, the position, and (stretch) the section heading.
- Emit `[(chunk_id, source_doc_id, text, metadata)]`.

```python
def chunk_corpus(docs, chunker):
    """docs: [(doc_id, text)]. Returns [(chunk_id, source_doc_id, text, meta)]."""
    rows = []
    for doc_id, raw in docs:
        text = clean(raw)                       # TODO 6: implement clean()
        for i, ch in enumerate(chunker.chunk(text)):
            rows.append((
                f"{doc_id}::{i}",               # chunk_id
                doc_id,                          # source_doc_id (gold unit!)
                ch.text,
                {"position": i, "token_span": ch.token_span},
            ))
    return rows
```

The `source_doc_id` is load-bearing: gold is in clause ids, you retrieve chunks, and this is the thread that maps a chunk hit back to the clause it came from. Get this mapping wrong and the whole A/B is invalid (the challenge's trap).

---

## Deliverable 3 — `ab.py` (the A/B loop — reuses week-7 `evaluate()` + `store.py`)

The function that runs the bakeoff. For each strategy it builds a `retrieve_fn` and calls the *same* pure `evaluate()`:

```python
from crunchrag_embed.eval import evaluate     # week 7, UNCHANGED
from crunchrag_embed import store             # week 7, UNCHANGED


def build_retrieve_fn(rows, embed_documents, embed_query, dim, table):
    """rows: [(chunk_id, source_doc_id, text, meta)]. Fixed store, one strategy."""
    store.create_table(table, dim=dim)
    vecs = embed_documents([r[2] for r in rows])
    store.insert(table, [(r[0], r[2], v) for r, v in zip(rows, vecs)])
    store.build_hnsw(table)
    chunk_to_doc = {r[0]: r[1] for r in rows}

    def retrieve_fn(query):
        # TODO 7: embed the query, knn the store (k=20), map chunk ids back to
        #   source_doc_ids, de-duplicate preserving rank, return the clause ranking.
        ...
    return retrieve_fn


def run_ab(strategies, docs, gold, embedders, k=5):
    table_rows = []
    for name in strategies:
        chunker = chunkers.load(name)
        rows = pipeline.chunk_corpus(docs, chunker)
        emb = embedders[name]               # BGE for most; jina-v3 for 'late'
        fn = build_retrieve_fn(rows, emb.embed_documents, emb.embed_query,
                               emb.dim, f"chunks_{name}")
        m = evaluate(gold, fn, k=k)         # SAME evaluate() every strategy
        table_rows.append((name, len(rows), m["Recall@k"], m["MRR"]))
    return table_rows
```

The non-negotiables `ab.py` enforces:

- **One store, one gold set, one metric.** `evaluate()` is imported unchanged; you do not re-implement Recall@5 or MRR (you proved them correct in week 7).
- **The embedder is fixed per the A/B rules** — BGE-large for fixed/sliding/recursive/semantic. **Late chunking is the documented exception**: it uses jina-v3 and the whole-doc-forward-pass-then-pool embed path. `ab.py` makes that asymmetry explicit in the table (a `model` column), so nobody mistakes a model swap for a chunking win.
- **The chunk→source mapping** is the only correct way to score; it lives here once.

---

## Deliverable 4 — `cli.py` (the `ab` command)

```bash
python -m crunchrag_chunk ab \
    --corpus corpus/legal_docs.jsonl \
    --gold corpus/gold.json \
    --strategies fixed512,fixed1024,recursive,semantic,late \
    --k 5
```

It should run the A/B and print:

```
STRATEGY     MODEL     CHUNKS   RECALL@5    MRR   FAITHFUL   EMBED_S
fixed512     bge        62       0.78      0.61     0.74       3.9
fixed1024    bge        34       0.72      0.55     0.79       3.1
recursive    bge        88       0.88      0.74     0.81       4.6
semantic     bge        71       0.86      0.72     0.80       9.2
late         jina-v3    88       0.89      0.75     0.83      11.7
--------------------------------------------------------------------
winner by Recall@5: late (0.89)  — but recursive (0.88) at 1/3 the cost;
   see memo for the recommendation.
```

The winner line picks by Recall@5 by default (a `--metric` flag switches it), and the memo (the README) makes the *judgment call* the table sets up: late edged recursive by 0.01 but cost a long-context model and 2.5× the embed time — on this corpus, is that worth it? The point is a *decision*, printed and defended.

---

## Rules

- **You may** read the strategy docs, the late-chunking paper, the lecture notes, and your week-7 code.
- **You must not** re-implement `evaluate()`, `recall_at_k`, or `reciprocal_rank` — import them from `crunchrag_embed.eval` unchanged. Re-implementing them is how the measurement drifts.
- **You must not** vary the store, the gold set, or (for non-late strategies) the embedding across runs. The whole validity is one-variable-at-a-time.
- **You must** map every chunk hit back to its `source_doc_id` before scoring. Scoring raw chunk ids against a clause-id gold set is the trap.
- Python 3.12, `transformers`, `sentence-transformers`, `psycopg[binary]`, `numpy`, plus `pytest`. The faithfulness leg needs a generator + judge LLM; an open-only path (a local model from week 6) is acceptable and the metric is optional-but-recommended.
- `evaluate()` must stay pure; if you found yourself editing week-7 code, stop — the design is that you don't have to.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-08-crunchrag-chunk-<yourhandle>`.
- [ ] `docker compose up -d` brings up pgvector; the A/B runs against it.
- [ ] `chunkers.py` implements all five strategies behind one `Chunker` interface, each with correct boundaries (proven by `test_chunkers.py`).
- [ ] `ab.py` imports `evaluate` and `store` from `crunchrag_embed` **unchanged** (no copy-pasted metric code).
- [ ] The chunk→source mapping is correct and tested (`test_ab_mapping.py`): a chunk of clause_09 maps to `clause_09`, and a strategy "finds" clause_09 iff one of its chunks ranks in top-k.
- [ ] `pytest` passes, with at least:
  - `test_chunkers.py`: fixed-window respects the size; recursive keeps a paragraph-sized clause whole; semantic splits at a planted topic boundary; late returns spans.
  - `test_ab_mapping.py`: the de-duplicating chunk→clause mapping preserves rank order and collapses multiple chunks of one clause to a single clause hit.
- [ ] `python -m crunchrag_chunk ab --strategies fixed512,fixed1024,recursive,semantic,late` prints a five-row table with a winner line.
- [ ] A `README.md` with the results table, the run commands, and the **one-page winner memo** (winner, its Recall@5/MRR/faithfulness, why it won *on this corpus*, the trade-off accepted, and the late-chunking model-swap caveat).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Chunker correctness** | 25 | All five strategies implemented behind one interface; boundaries correct; size measured in BGE tokens, never characters; late chunking is genuinely whole-doc-forward-pass-then-pool, not independent embedding mislabeled. |
| **A/B validity** | 25 | Store/gold/metric fixed; only the chunker varies; `evaluate()` imported unchanged; the late-chunking model swap is isolated and labeled, not hidden. |
| **Chunk→source mapping** | 20 | Every chunk maps to its source clause id; scoring is on clauses, not raw chunks; de-dup preserves rank. The trap is avoided and tested. |
| **Tests** | 15 | `test_chunkers` covers all five; `test_ab_mapping` covers the de-dup and rank-preservation; `pytest` green. |
| **A/B & decision** | 10 | The CLI runs all strategies and prints a comparison table; the README names a winner with a number and a defensible reason. |
| **Docs & hygiene** | 5 | Clear README + memo, no secrets committed, sensible commits, no `__pycache__`/`.venv`/`models/` checked in. |

**90+** is portfolio-grade and ready to drop into week 9's reranking lab. **70–89** works but has a soft mapping or an unlabeled model swap. **Below 70** means the A/B isn't a fair, reusable measurement — fix that first, because week 9 starts from this harness's winner.

---

## Stretch goals

- **Early-vs-late control.** Add a `late-early` strategy: jina-v3 with *independent* chunk embedding on the same spans. The `late` vs `late-early` gap (model held constant) is the *pure* late-chunking lift — the honest measurement the paper reports.
- **Metadata injection.** Add a `--inject-heading` flag that prepends each chunk's section heading to its text before embedding. Measure the Recall@5 delta — the README stretch goal, with a number.
- **Overlap sweep.** Add `ab --overlap-sweep 0,32,64,128` on the winning strategy. On self-contained clauses, does overlap move recall at all? Predict, then measure.
- **CI.** A GitHub Actions workflow that spins up pgvector in a service container, runs `pytest`, and runs a two-strategy A/B headless. Green check on every push.

---

## How this connects to the rest of C23

- **Week 7 (embeddings)** gave you `evaluate()` and `store.py`; this harness imports both unchanged and points them at the chunker instead of the model — the "same embedding, same store, report MRR/Recall@5" lab the syllabus describes.
- **Week 9 (reranking & hybrid search)** starts from *your winning chunker* here, adds a BM25 lexical leg and a cross-encoder reranker, and charts the lift at each layer — calling the *same* `evaluate()`. Your chunk→source mapping carries straight over.
- **Week 12 (evaluation & architecture review)** layers Ragas on top; your Recall@5/MRR are the *retrieval* metrics under Ragas's *answer* metrics, and the review grades whether you chose your chunking strategy by measurement (this A/B) rather than by default.

When you've finished, push the repo and take the [quiz](../quiz.md).
