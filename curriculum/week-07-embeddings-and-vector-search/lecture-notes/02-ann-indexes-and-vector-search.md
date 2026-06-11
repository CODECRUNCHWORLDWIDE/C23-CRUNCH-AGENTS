# Lecture 2 — ANN Indexes, pgvector, and Measuring Retrieval

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can explain how HNSW, IVF, and ScaNN trade recall for latency and memory, build and tune an HNSW index in pgvector, and measure retrieval quality with Recall@k and MRR so your embedding choice is a number, not a vibe.

Lecture 1 turned text into vectors. This lecture searches them — fast, at scale, and with a measurement at the end. Three parts: (1) how approximate nearest-neighbour search works, (2) pgvector as the production default, (3) evaluating retrieval like an engineer.

---

## Part 1 — Why approximate, and how

### 1.1 The exact baseline and why it doesn't scale

The honest way to find the nearest vectors to a query is **brute force**: compute the similarity to *every* stored vector, sort, take the top k. This is **exact** — it returns the true nearest neighbours, always. For a few thousand vectors it's instant and you should just do it.

The problem is scale. With N vectors of dimension d, each query is O(N·d) work. At N = 10 million and d = 1024, that's ten billion multiply-adds *per query*. A single query takes hundreds of milliseconds to seconds on a CPU. Run a thousand queries a second and you need a server farm. Brute force is correct and unscalable.

So we trade a little correctness for a lot of speed. An **Approximate Nearest Neighbour (ANN)** index returns *probably* the nearest neighbours — say, 95% of the true top-10 — in a tiny fraction of the time. The 5% you miss is the price. For RAG that price is almost always worth paying: a reranker (week 9) cleans up the top results anyway, and a 95%-recall index at 2 ms beats a 100%-recall scan at 800 ms every time.

> **The iron triangle:** you cannot maximize recall, minimize latency, and minimize memory all at once. Every ANN index is a chosen point on that triangle, and every runtime knob moves you along an edge. Internalize this and the index docs stop being mysterious.

### 1.2 HNSW — the graph

**Hierarchical Navigable Small World** is the index pgvector, Qdrant, Weaviate, and most modern stores default to. The idea is a *navigable graph*: every vector is a node, connected to its nearby neighbours by edges. To search, you start at an entry point and greedily walk toward the query — at each step, move to the neighbour closest to the query — until you can't get closer.

The "hierarchical" part adds layers. The top layer has few nodes with long-range links (think highways); each lower layer adds more nodes with shorter links (local roads); the bottom layer has everyone. A search starts at the top, descends through the layers, and zooms in. This is what makes it logarithmic-ish instead of linear.

The knobs:

- **`m`** — how many edges each node gets (typical 16). Higher `m` = better recall, more memory, slower build. Set at index-build time; can't change without rebuilding.
- **`ef_construction`** — how hard the index works while *building* (typical 64–200). Higher = better graph, slower build. Build-time only.
- **`ef_search`** — how many candidates the search explores at query time (typical 40–400). **This is the one runtime knob that matters.** Higher `ef_search` = higher recall, higher latency. You can change it per query. It is the dial you turn to move along the recall/latency edge of the triangle.

HNSW's strength: excellent recall/latency, no training step, handles incremental inserts. Its weakness: memory. The graph lives in RAM and the edges cost real bytes — an HNSW index can be larger than the vectors themselves. At a billion vectors, that hurts.

### 1.3 IVF — the inverted file

**IVF (Inverted File)** takes a different tack: *cluster first, search less*. At build time it runs k-means to partition the vector space into (say) 1000 clusters, each with a centroid. To search, you find the few centroids nearest the query and scan *only* the vectors in those clusters — ignoring the rest entirely.

The knobs:

- **`lists`** (a.k.a. `nlist`) — how many clusters. More clusters = finer partition, faster search, but you need enough vectors per cluster. Build-time.
- **`nprobe`** — how many clusters to actually scan at query time. **The runtime knob.** `nprobe=1` is fast and low-recall (you might miss neighbours that fell in an adjacent cluster); higher `nprobe` scans more clusters for better recall at higher latency.

IVF's strength: low memory (no graph, just centroids + cluster assignments) and it's great for billion-scale when combined with **product quantization** (PQ), which compresses each vector to a few bytes. Its weakness: it needs a *training* step (the k-means) on a representative sample before you can insert, and recall is more sensitive to how the data clusters. The classic boundary failure: a true neighbour sits just across a cluster border and gets missed unless `nprobe` is high enough to scan that neighbour's cluster too.

### 1.4 ScaNN — Google's anisotropic quantization

**ScaNN** (Scalable Nearest Neighbors) is Google's library, and the headline idea is **anisotropic quantization**. Ordinary quantization compresses every vector to minimize reconstruction error uniformly. ScaNN observes that for *maximum inner product search*, errors in the direction that matters for ranking hurt more than errors in directions that don't — so it quantizes *anisotropically*, spending precision where it changes the score. The result is excellent recall-per-byte at very large scale. You'll rarely run ScaNN yourself (it's most relevant inside Google-scale systems and some managed stores), but you should know the concept: **not all quantization error is equal, and a good index spends its bits where ranking depends on them.**

### 1.5 The decision, in one table

| Index | Memory | Build | Best for | Runtime knob |
|---|---|---|---|---|
| **Brute force** | low | none | < ~10k vectors, or ground truth | none (always exact) |
| **HNSW** | high | medium | most RAG (10k–100M), incremental inserts | `ef_search` |
| **IVF (+PQ)** | low | needs k-means | 100M–10B, memory-constrained | `nprobe` |
| **ScaNN** | low | needs training | Google-scale MIPS | (library-specific) |

For nearly every system you build in this course, the answer is **HNSW**, because your corpus is in the 10k–10M range where HNSW's recall/latency is unbeatable and its memory cost is affordable. Reach for IVF+PQ only when you blow past memory at hundreds of millions of vectors. Use brute force for your *ground truth* when measuring recall (Exercise 3) — you need the exact answer to know how approximate your ANN really is.

---

## Part 2 — pgvector in practice

You could run a dedicated vector database (week 10 surveys them). But in 2026 the sane default for most teams is **pgvector**: a Postgres extension that adds a `vector` type and ANN indexes. Why pgvector first? Because your data, your metadata, your access control, your backups, and your operational know-how already live in Postgres. Adding vectors to the database you already run beats standing up a second system you have to learn, monitor, and back up separately. "Pick the store with the operational story you can live with at 2 AM" (week 10's lecture) — and for most teams that's the Postgres they already operate.

### 2.1 The vector type and the operators

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    content   TEXT NOT NULL,
    embedding vector(1024)   -- dimension is fixed here, forever, per Lecture 1 §7
);
```

The three distance operators — and you must match them to your metric:

| Operator | Distance | Use when |
|---|---|---|
| `<=>` | cosine distance (1 − cosine similarity) | normalized text embeddings (the default) |
| `<#>` | negative inner product | normalized vectors, slightly faster than `<=>` |
| `<->` | Euclidean (L2) distance | un-normalized vectors where magnitude matters |

A k-NN query for the 5 nearest chunks to a query vector:

```sql
SELECT id, doc_id, content, 1 - (embedding <=> :query_vec) AS cosine_sim
FROM chunks
ORDER BY embedding <=> :query_vec   -- smaller cosine DISTANCE = more similar
LIMIT 5;
```

Note the subtlety: `<=>` is a *distance* (smaller is better), so `ORDER BY embedding <=> q` puts the most similar first. To report a *similarity* score you do `1 - distance`. Mixing these up — sorting by similarity ascending, say — silently inverts your results. It's the pgvector version of week 5's "I sorted the wrong way" bug.

### 2.2 Building the HNSW index

Without an index, that query is a brute-force scan — fine for thousands of rows, slow for millions. Add an HNSW index:

```sql
CREATE INDEX ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

`vector_cosine_ops` tells the index to use cosine distance — **it must match the operator your queries use (`<=>`)**. If you build a `vector_l2_ops` index and query with `<=>`, Postgres can't use the index and falls back to a full scan; you get correct results, mysteriously slowly. This pairing is the pgvector gotcha that sends people to Stack Overflow.

At query time, set `ef_search`:

```sql
SET hnsw.ef_search = 100;   -- the runtime recall/latency dial
SELECT id, content FROM chunks ORDER BY embedding <=> :q LIMIT 5;
```

Higher `ef_search` = better recall, slower query. Exercise 3 has you sweep it against ground truth and chart the curve — that chart is the single most useful artifact for understanding your index.

### 2.3 The full Python round-trip

```python
import psycopg
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def to_pgvector(v: np.ndarray) -> str:
    # pgvector accepts the literal '[0.1,0.2,...]' string form.
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

conn = psycopg.connect("postgresql://postgres:crunch@localhost:5432/postgres")
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

# Insert a document chunk.
doc = "Either party may terminate this agreement upon thirty days written notice."
vec = model.encode(doc, normalize_embeddings=True)
conn.execute(
    "INSERT INTO chunks (doc_id, content, embedding) VALUES (%s, %s, %s)",
    ("clause_14", doc, to_pgvector(vec)),
)
conn.commit()

# Query.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
qvec = model.encode(QUERY_PREFIX + "notice to end the contract",
                    normalize_embeddings=True)
conn.execute("SET hnsw.ef_search = 100")
rows = conn.execute(
    "SELECT doc_id, content, 1 - (embedding <=> %s) AS sim "
    "FROM chunks ORDER BY embedding <=> %s LIMIT 5",
    (to_pgvector(qvec), to_pgvector(qvec)),
).fetchall()
for doc_id, content, sim in rows:
    print(f"{doc_id}  sim={sim:.3f}  {content[:60]}")
```

The exercises build this out into a real corpus with a real index. The pattern — embed query (with the prefix!), `ORDER BY <=> q LIMIT k`, read the rows — is the spine of every RAG retrieval call you'll write this phase.

---

## Part 3 — Measuring retrieval like an engineer

Here is the part that separates a senior RAG engineer from someone who shipped a demo: **you measure retrieval with numbers, on a gold set, and you let the numbers pick the model.** "It seems to find good stuff" is not a result. The course's grading rubric *fails vibes-only submissions* for exactly this reason.

### 3.1 The gold set

A **gold set** is a list of (query, relevant-doc-ids) pairs that you (a human, or a careful LLM you spot-checked) labeled. For the legal corpus this week: 40 questions, each tagged with which clause(s) actually answer it. The gold set is the ground truth your metrics are computed against. It's small (40 is fine to start), it's hand-curated, and it's the most valuable artifact in your whole pipeline — more valuable than the code, because it's what tells you whether a change helped.

Building one is tedious and worth every minute. A rough but real shortcut: write the questions, run a strong baseline retriever, and have a human confirm or correct the top result for each. You're not labeling from scratch; you're auditing a draft.

### 3.2 The metrics

For each query, you retrieve a ranked list and compare it to the gold relevant docs.

**Recall@k** — did the relevant doc make it into the top k?

```
Recall@k for one query = (relevant docs in top-k) / (total relevant docs)
Recall@k overall       = mean over all queries
```

If each query has exactly one relevant doc, Recall@k is just "fraction of queries where the right doc was in the top k." Recall@5 of 0.85 means: for 85% of queries, the right answer was somewhere in the top 5. This is the headline RAG metric, because if the right chunk isn't in what you retrieve, the LLM can't use it — retrieval recall is a *ceiling* on answer quality.

**MRR (Mean Reciprocal Rank)** — how high up was the first relevant doc?

```
RR for one query = 1 / (rank of the first relevant doc)   (0 if none found)
MRR              = mean of RR over all queries
```

If the relevant doc is rank 1, RR = 1.0. Rank 2 → 0.5. Rank 5 → 0.2. Not found → 0. MRR rewards putting the answer *first*, which matters because the LLM weights the top of the context more (the "lost in the middle" effect you'll study in week 11). MRR of 0.71 means the first relevant doc is, on average, around rank 1.4.

**top-1 / top-5** — the simplest cut: fraction of queries whose #1 result (or top-5) is relevant. top-1 is Recall@1 when there's one relevant doc; people report it because it's intuitive.

### 3.3 Computing them in Python

```python
def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for d in retrieved_ids[:k] if d in relevant_ids)
    return hits / len(relevant_ids)

def evaluate(gold: list[tuple[str, set[str]]], retrieve, k: int = 5) -> dict:
    """gold is [(query, {relevant_doc_ids})]; retrieve(query) -> ranked [doc_id]."""
    rrs, recalls, top1s = [], [], []
    for query, relevant in gold:
        ranked = retrieve(query)
        rrs.append(reciprocal_rank(ranked, relevant))
        recalls.append(recall_at_k(ranked, relevant, k))
        top1s.append(1.0 if ranked and ranked[0] in relevant else 0.0)
    n = len(gold)
    return {
        "queries": n,
        "MRR": sum(rrs) / n,
        f"Recall@{k}": sum(recalls) / n,
        "top1": sum(top1s) / n,
    }
```

Run that with two different embeddings and you have a *comparison*, not an opinion. That function is the heart of the mini-project, and you'll carry it into weeks 8 and 9 — the chunking A/B and the reranking-lift charts both call the exact same `evaluate()`.

### 3.4 The trap: measuring the wrong thing

Two failure modes to avoid:

- **Recall at retrieval ≠ answer quality.** A high Recall@5 means the right chunk *was available*. Whether the LLM *used* it correctly is a separate question (Ragas faithfulness, week 12). Don't claim your *answers* are good because your *retrieval* recall is high; they're different metrics on different things. But do remember: bad retrieval recall *caps* answer quality — you can't answer from a chunk you never retrieved.
- **Overfitting to the gold set.** If you tune `ef_search`, the chunk size, and the embedding all against the same 40 questions, you may be fitting noise. Keep the gold set fixed, change one thing at a time, and be suspicious of improvements smaller than the noise. With 40 questions, a one-point Recall@5 move is roughly one query flipping — that's noise, not signal.

---

## 4. Putting it together: the retrieval debugging tree

When retrieval is bad, walk this tree:

```
Recall@5 is low.
│
├─ Did you embed QUERIES with the right prefix/input_type?
│   ├─ No  → fix the query/document asymmetry (Lecture 1 §7). Most common cause.
│   └─ Yes ↓
│
├─ Are query and document vectors BOTH normalized the same way?
│   ├─ No  → normalize everything in one place; re-embed if needed.
│   └─ Yes ↓
│
├─ Does the pgvector index op match the query operator? (vector_cosine_ops + <=>)
│   ├─ No  → rebuild the index with the matching op, or you're doing a full scan.
│   └─ Yes ↓
│
├─ Is ef_search high enough? (compare ANN recall vs brute-force ground truth)
│   ├─ No  → raise ef_search; re-measure on the curve (Exercise 3).
│   └─ Yes ↓
│
└─ Index is fine, embedding asymmetry is fine → it's the CHUNKING (week 8)
    or you need a RERANKER / hybrid search (week 9). The embedding isn't the problem.
```

That last branch is the week's humility lesson, and it's why the next two weeks exist. Once your embedding is normalized, prefixed correctly, and indexed with a tuned `ef_search`, you've extracted most of what the embedding has to give. The remaining gains are in *what you embed* (chunking) and *how you re-rank* (reranking + hybrid). Tape this tree next to the model table from Lecture 1.

---

## 5. Recap

You should now be able to:

- Explain why exact search doesn't scale and what "approximate" buys you, in terms of the recall/latency/memory triangle.
- Describe HNSW (the layered navigable graph, `m`/`ef_construction`/`ef_search`), IVF (clusters + `nprobe`), and ScaNN (anisotropic quantization), and pick HNSW for almost everything you'll build.
- Build and tune an HNSW index in pgvector: the `vector` type, the `<=>`/`<#>`/`<->` operators, `vector_cosine_ops`, and `SET hnsw.ef_search`.
- Build a gold set and compute Recall@k, MRR, and top-1/top-5 in plain Python — and use those numbers to pick an embedding instead of guessing.
- Walk the retrieval debugging tree to find why recall is low, and recognize when the answer is "it's not the embedding."

Next: the exercises put all of this on a real corpus, and the mini-project turns the `evaluate()` function into a reusable bakeoff harness you'll carry through the rest of Phase II. Continue to [the exercises](../exercises/README.md).

---

## References

- *HNSW paper* (Malkov & Yashunin, 2018): <https://arxiv.org/abs/1603.09320>
- *pgvector README and indexing guide*: <https://github.com/pgvector/pgvector>
- *FAISS index-selection guidelines*: <https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index>
- *ScaNN*: <https://github.com/google-research/google-research/tree/master/scann>
- *MTEB paper* (retrieval-metric definitions): <https://arxiv.org/abs/2210.07316>
- *Billion-scale similarity search with GPUs* (Johnson et al., 2017): <https://arxiv.org/abs/1702.08734>
