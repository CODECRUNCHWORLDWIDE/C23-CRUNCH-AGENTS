# Week 9 — Reranking, Hybrid Search, and Structured Retrieval

Welcome to the week where retrieval stops being one model and becomes a *pipeline*. By Friday you will be able to look at any RAG system and state, with measured numbers, exactly where each layer earns its place: what dense retrieval gets, what BM25 adds, what fusing them buys, what a reranker recovers on top, and when the answer isn't in the vector store at all and you should be querying a database instead. You will read a cumulative-lift table the way a backend engineer reads a flame graph — finding the layer that's actually paying for itself.

This is week 3 of **Phase II — RAG & Memory Systems**, and it sits directly on weeks 7 and 8. Everything here assumes you can embed a corpus, index it in pgvector, chunk it well, and *measure* retrieval with the `crunchrag_embed` `evaluate()` you built in week 7 and reused in week 8. This week you wrap your best `retrieve_fn` with three new layers and call the **same** `evaluate()` to chart what each one is worth.

The one sentence to internalize before you read another line:

> **Dense retrieval matches meaning and misses words. The whole job this week is to put the words back — with BM25 — and then let a reranker read the query and the passage *together* before you trust the order.**

Here's why that's not hyperbole. A `bge-large` embedding is a blurry average of a passage's meaning. Ask it for "clause 14" or "$1,000,000" or "Delaware" and it shrugs — those are exact tokens, rare strings, identifiers, and the embedding smeared them into the topical soup. BM25 nails them, because BM25 is *about* the words. Neither alone is enough; fused with Reciprocal Rank Fusion they cover each other's blind spots; and a cross-encoder reranker, applied only to the fused top-k, reads each candidate against the query and fixes the order that first-stage retrieval got *almost* right. That reranker is the headline of the week, and the syllabus mantra says it plainly:

> **A reranker is the cheapest meaningful win in RAG. Use one.**

## Learning objectives

By the end of this week, you will be able to:

- **Explain** why dense retrieval alone is insufficient — the exact-match, rare-term, identifier, and acronym failures that a single embedding vector cannot represent — and predict which queries it will miss.
- **Describe** BM25 precisely: term frequency, inverse document frequency, the `k1` term-frequency-saturation knob, and the `b` length-normalization knob, and what each does to a score.
- **Build** a hybrid retriever that fuses a dense ranked list and a BM25 ranked list with **Reciprocal Rank Fusion** — `score(d) = Σ_r 1/(k + rank_r(d))`, k≈60 — and explain why rank-based fusion beats score normalization in practice.
- **Apply** a **cross-encoder reranker** (`BAAI/bge-reranker-v2-m3`, Cohere `rerank-3.5`) to the first-stage top-k, and explain the bi-encoder/cross-encoder distinction and the latency reason you rerank only the top-k.
- **Reason** about **ColBERT** late-interaction (token-level MaxSim) and where it sits between a bi-encoder and a cross-encoder on the quality/cost curve.
- **Implement** query rewriting and **HyDE** (Hypothetical Document Embeddings) — generate a fake answer, embed *that*, retrieve with it — and measure where it helps recall and where it hurts precision.
- **Generate** SQL from natural language for **structured retrieval**, and lock down the safety surface: a read-only role, parameterized/validated execution, and a schema allowlist.
- **Measure** the cumulative lift at each layer (BM25 → dense → hybrid+RRF → +reranker → +HyDE) on the **same 40-query gold set**, and defend the pipeline in one table.

## Prerequisites

This week assumes you have completed **C23 weeks 1–8**, or have equivalent fluency. Specifically:

- You finished **week 7** and have the `crunchrag_embed` mini-project: an `Embedder` interface, the pgvector `store.py`, and a **pure** `evaluate(gold, retrieve_fn, k)` returning top-1 / Recall@k / MRR. **This week imports `evaluate()` unchanged** — if it's broken, fix it first.
- You finished **week 8** and have the `crunchrag_chunk` harness, so you have a *best* chunking strategy and a `retrieve_fn` that already beats the naive baseline. Week 9 wraps that `retrieve_fn`; it does not replace it.
- Python 3.12 on Linux, macOS (Apple Silicon), or WSL2; a virtualenv you can `pip install` into; Docker for pgvector (`pgvector/pgvector:pg17`).
- You can call a hosted LLM (week 1) — HyDE and text-to-SQL both need one — and you're comfortable with the week-7 retrieval metrics. We reuse Recall@5 and MRR all week.

You do **not** need a GPU. `BAAI/bge-reranker-v2-m3` runs on CPU or Apple Silicon (slower, but unblocked) for the small candidate sets we rerank; the labs document the CPU path. You do **not** need prior search-engine experience — we start at "what is BM25" and build up to a five-layer pipeline.

## Topics covered

- **Why dense is not enough:** the exact-match miss (IDs like `clause_14`, dollar amounts, statute numbers), the rare-term and acronym miss, the negation blind spot from week 7, and the failure mode where the topically-similar wrong clause outranks the lexically-exact right one.
- **BM25 and sparse/lexical search:** term frequency, the IDF weighting that rewards rare query terms, the `k1` saturation parameter (why the 50th occurrence of a word adds almost nothing), the `b` length-normalization parameter, and the Okapi BM25 scoring function in full. `rank-bm25` for the labs; Postgres full-text and **Tantivy/Elasticsearch** as the production path.
- **Hybrid search and Reciprocal Rank Fusion:** combining dense and sparse rankings; the RRF formula `Σ_r 1/(k + rank_r(d))` with k=60; why RRF is robust (rank-based, needs no score calibration) and weighted/normalized score fusion is fragile (the two scorers live on incomparable scales).
- **Rerankers (cross-encoders):** the bi-encoder vs cross-encoder distinction; `BAAI/bge-reranker-v2-m3` via `sentence-transformers` `CrossEncoder` (and `FlagEmbedding`'s `FlagReranker` as the alternative loader); Cohere `rerank-3.5` as the vendor path; why you score `(query, passage)` jointly and only on the first-stage top-k (top 50 → rerank → top 5).
- **ColBERT and late interaction:** token-level MaxSim, the middle ground between bi- and cross-encoders, via **RAGatouille** / the `colbert-ir` models (`answerai-colbert-small-v1`, ColBERTv2).
- **Query rewriting and HyDE:** rewriting a vague query into a retrievable one; **HyDE** — generate a hypothetical answer with an LLM, embed the hypothetical, retrieve with that vector (Gao et al., 2022).
- **Structured retrieval / text-to-SQL:** when the answer lives in a database, not a vector store; generating SQL from natural language; the safety surface — read-only role, parameterized/validated execution, schema allowlist, never executing raw model SQL on a writable connection.
- **The discipline of measuring lift:** the same 40-query gold set, the same `evaluate()`, one row per layer, so "did the reranker help?" is a number, not a vibe.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                       | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Why dense isn't enough; BM25 (TF/IDF/k1/b); the baseline    |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Hybrid search; RRF; the fusion exercise                     |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Rerankers (cross-encoder/ColBERT); the reranker-lift drill  |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Query rewriting + HyDE; text-to-SQL; the lift harness       |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The cumulative-lift run + memo; the safety clinic           |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                       |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                   |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                             | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | BM25 references, the RRF and HyDE papers, the reranker model cards, ColBERT/RAGatouille, and the text-to-SQL safety docs |
| [lecture-notes/01-why-dense-is-not-enough-bm25-and-hybrid.md](./lecture-notes/01-why-dense-is-not-enough-bm25-and-hybrid.md) | Why dense misses words, BM25 in full (TF/IDF/k1/b), hybrid search, and RRF with a worked fusion |
| [lecture-notes/02-rerankers-and-structured-retrieval.md](./lecture-notes/02-rerankers-and-structured-retrieval.md) | Cross-encoders, ColBERT, query rewriting, HyDE, and text-to-SQL with the safety surface |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-bm25-baseline.md](./exercises/exercise-01-bm25-baseline.md) | Build a `rank-bm25` index, run the gold set, and prove BM25 wins where dense missed (IDs, "$1,000,000", "Delaware") |
| [exercises/exercise-02-rrf-fusion.py](./exercises/exercise-02-rrf-fusion.py) | Implement RRF over a dense and a BM25 ranked list and prove the fused order beats either alone |
| [exercises/exercise-03-reranker-lift.py](./exercises/exercise-03-reranker-lift.py) | Score a candidate set with `bge-reranker-v2-m3` and watch it pull the right doc from rank 4 to rank 1 |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-cumulative-lift.md](./challenges/challenge-01-cumulative-lift.md) | Build the full stack and chart cumulative lift (BM25 → dense → hybrid → +reranker → +HyDE) on the 40-query gold set |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the headline cumulative-lift memo |
| [mini-project/README.md](./mini-project/README.md) | The `crunchrag_hybrid` layered retriever — dense + BM25 + RRF + reranker (+ optional HyDE), printing a lift table per layer |

## The "the lift was real, and measured" promise

C23 uses a recurring marker for every exercise that ends in a retrieval system measurably better than the layer below it. Week 9's marker is the cumulative-lift table — one row per layer, the **same** 40-query gold set under each:

```
$ python -m crunchrag_hybrid lift --corpus legal --k 5
LAYER                       TOP-1   RECALL@5    MRR     Δ Recall@5
bm25 only                    0.50       0.72   0.60          —
dense only                   0.62       0.85   0.71      +0.13
hybrid (dense + bm25, RRF)   0.68       0.90   0.76      +0.05
  + reranker (bge-v2-m3)     0.78       0.93   0.83      +0.03
  + HyDE                     0.78       0.93   0.82      +0.00
-----------------------------------------------------------------
biggest single lift: dense over bm25 (+0.13 Recall@5)
cheapest meaningful lift: reranker (+0.03 Recall@5, +0.07 MRR, ~30 ms/query)
```

If your reranker row doesn't move MRR even when Recall@5 barely budges, look again — the reranker's job is to *reorder*, so it shows up in MRR (rank of the right doc) more than in Recall@5 (whether it's in the top 5 at all). And notice the honest row at the bottom: HyDE added nothing here, and a measured "+0.00" is a *result*, not a failure — it's the difference between a pipeline you can defend and a pile of techniques you cargo-culted. The point of week 9 is to make every layer's contribution a number you measured on one fixed gold set, so the architecture review in week 12 is a conversation about evidence.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **RRF paper** (Cormack, Clarke & Büttcher, 2009) until you can explain why a parameter-light rank-based fusion beats the elaborate score-combination methods it was compared against: <https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf>. Then sweep `k` ∈ {10, 30, 60, 100} on your gold set and chart how little it moves.
- Swap `bge-reranker-v2-m3` for a **ColBERT** late-interaction retriever via **RAGatouille** (`answerai-colbert-small-v1`) and put it in the lift table as its own layer. Where does token-level MaxSim land between your bi-encoder and your cross-encoder, on both quality and latency?
- Implement **HyDE** with a real LLM call (generate a hypothetical clause, embed it, retrieve) and measure the recall lift *and* the precision cost on your gold set. Find a query where HyDE helps and one where its hallucinated answer drags retrieval off-topic.
- Build a tiny **text-to-SQL** leg over a 3-table SQLite schema of the same contracts (parties, clauses, payments) with a read-only role and a schema allowlist, and answer "which agreements expire before 2027?" — a question the vector store *cannot* answer because it's structured.

## Up next

Week 10 takes the layered retriever you build here and asks what it takes to run it for real: **vector stores in production** — Qdrant, Weaviate, Milvus, and managed pgvector at scale; sharding, replication, filtering, metadata, and the operational facts that bite when your 50-clause corpus becomes 50 million. Your `crunchrag_hybrid` retriever is the thing you'll deploy. Push your mini-project before you start it — week 10 builds on it.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
