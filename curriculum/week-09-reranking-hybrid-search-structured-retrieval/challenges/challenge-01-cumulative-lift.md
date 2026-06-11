# Challenge 1 — The Cumulative-Lift Chart

**Time estimate:** ~90 minutes.

## Problem statement

Your team's RAG pipeline has grown a BM25 leg, a fusion step, a reranker, and someone just merged a HyDE branch. In the architecture review, the lead asks the only question that matters: **"Which of these layers is actually helping, and by how much?"** The wrong answer is "all of them, probably." The right answer is a table — one row per layer, the same gold set under each, measured by the same function — that shows exactly where the recall came from and which layer earned its latency.

This challenge is that table. You will take your best chunking strategy from week 8 and build the full retrieval stack, measuring the cumulative lift at each layer on the **same 40-query gold set** with the **same** `evaluate()` you wrote in week 7. This mirrors the syllabus lab exactly ("Chart the cumulative lift on the same 40-query set: BM25 alone → dense alone → hybrid → +reranker → +HyDE").

This is the real skill: not stacking techniques because a blog post said to, but measuring each one on *your* data and keeping only the layers that pay for themselves.

## The layers (build them in this order)

Each layer wraps the one below it. The corpus, the chunking, and the gold set are held fixed across all five — only the retrieval strategy changes.

1. **`bm25` only** — `rank-bm25` over the corpus (Exercise 1). Your lexical baseline.
2. **`dense` only** — your week-7 pgvector + `bge-large-en-v1.5` retriever, with your best week-8 chunking. Your semantic baseline.
3. **`hybrid` (dense + bm25, RRF)** — fuse the two ranked lists with RRF, k=60 (Exercise 2).
4. **`+ reranker`** — take the hybrid top-k (e.g. 50), rerank with `BAAI/bge-reranker-v2-m3`, keep the top-5 (Exercise 3).
5. **`+ HyDE`** — generate a hypothetical answer per query, embed *that* for the dense leg, then run the full hybrid+reranker stack on top (Lecture 2 §5).

## The harness approach

The whole point is that **every layer is measured by the same `evaluate()`** — so each is just a different `retrieve_fn` you hand to the function you already have:

```python
from crunchrag_embed.eval import evaluate   # the week-7 function, UNCHANGED

# Each layer is a retrieve_fn: query -> ranked list of doc_ids.
layers = {
    "bm25 only":            lambda q: bm25_ids(q, top_k=5),
    "dense only":           lambda q: dense_ids(q, top_k=5),
    "hybrid (RRF)":         lambda q: rrf_fuse([dense_ids(q, 50), bm25_ids(q, 50)])[:5],
    "+ reranker":           lambda q: rerank(q, rrf_fuse([dense_ids(q,50), bm25_ids(q,50)])[:50])[:5],
    "+ HyDE":               lambda q: rerank(q, hyde_hybrid(q, top_k=50))[:5],
}

rows = []
prev_recall = None
for name, fn in layers.items():
    m = evaluate(GOLD, fn, k=5)        # same gold set, same function, every layer
    delta = "" if prev_recall is None else f"{m['Recall@k'] - prev_recall:+.2f}"
    rows.append((name, m["top1"], m["Recall@k"], m["MRR"], delta))
    prev_recall = m["Recall@k"]
# ...print rows as a table.
```

The discipline is non-negotiable: **the gold set is identical, the metric is identical, and `evaluate()` is the week-7 function imported unchanged.** If you tweak the gold set between layers, or re-implement the metric, the comparison is meaningless and the table is a lie.

## The deliverable

### The cumulative-lift table

A markdown table, one row per layer, with the `Δ Recall@5` column showing what *that layer* added over the one below it:

| Layer | top-1 | Recall@5 | MRR | Δ Recall@5 |
|---|---:|---:|---:|---:|
| bm25 only | ... | ... | ... | — |
| dense only | ... | ... | ... | ... |
| hybrid (dense + bm25, RRF) | ... | ... | ... | ... |
| + reranker (bge-v2-m3) | ... | ... | ... | ... |
| + HyDE | ... | ... | ... | ... |

Plus two sentences naming **the biggest single lift** and **the cheapest meaningful lift** (lift per unit of added latency/cost — almost always the reranker).

### The one-paragraph read

Below the table, write the honest read: which layer surprised you, which one added nothing (a `+0.00` row is a *finding*, not a failure), and which layer you'd ship to production and which you'd cut. This is the paragraph the architecture reviewer actually reads.

## Acceptance criteria

- [ ] A file `cumulative-lift.md` with the five-row table, every number computed from a real `evaluate()` run (not estimated).
- [ ] All five layers measured on the **identical** 40-query gold set, with `evaluate()` imported from `crunchrag_embed` **unchanged**.
- [ ] The reranker is applied **only** to the first-stage top-k (e.g. top 50 → rerank → top 5), never to the whole corpus.
- [ ] The `Δ Recall@5` column is filled in for every layer (the first is "—").
- [ ] You note which lift is biggest (usually dense over bm25, or the reranker) and which is cheapest-per-cost (usually the reranker).
- [ ] You report at least one layer honestly — including if HyDE came back `+0.00` or negative on this corpus.
- [ ] Committed to your week-9 repo under `challenges/challenge-01/`.

## The trap (read after a first attempt)

There are three classic ways to get a *confident wrong* table here. Watch for all three:

1. **Reranking the wrong top-k.** If you rerank only the hybrid *top-5* and call it the "reranker layer," you've capped the lift: the reranker can only reorder docs that made the top-5, so it can't rescue a right answer that landed at first-stage rank 12. **Rerank a generous candidate set (top 50), then keep the top 5.** The reranker's whole value is reordering a *wide* candidate set down to a *precise* short one. Rerank too few and you'll conclude "the reranker barely helped" when really you starved it.

2. **RRF `k` sensitivity (or the lack of it).** It's tempting to spend an hour tuning RRF's `k`. Don't — sweep `k` ∈ {10, 30, 60, 100} and you'll find Recall@5 barely moves (that's *why* RRF is robust). If your hybrid layer is wildly sensitive to `k`, you have a bug (a 0-based rank, or you fused score-normalized lists instead of rank lists), not a tuning opportunity. The non-sensitivity is the feature.

3. **HyDE helping recall but hurting precision.** HyDE can lift Recall@5 (the hypothetical answer embeds closer to the real clause) while *lowering* MRR or top-1 (the hallucinated answer drags a near-miss above the true answer). If your HyDE row shows Recall@5 up but MRR down, that's not noise — it's the precision cost of the hypothetical, and it's exactly the trade-off you must report. A layer that helps one metric and hurts another is a *decision*, not a free win.

## Stretch

- **Add a ColBERT layer.** Insert a late-interaction leg (RAGatouille, `answerai-colbert-small-v1`) as its own row between dense and the reranker. Where does token-level MaxSim land on quality, and what does it cost in latency versus the cross-encoder?
- **Per-query breakdown.** Beyond the aggregate table, dump the per-query Recall@5 for `dense only` vs `+ reranker` and find the 3–4 queries the reranker *flipped* from miss to hit. Those queries are your evidence for the architecture review — show the lead the exact clauses the reranker rescued.
- **Latency column.** Add a `ms/query` column (median) per layer. Now the table answers not just "did it help?" but "was it worth it?" — the reranker's `+0.03 Recall@5` at `+30 ms` is a different decision than HyDE's `+0.01` at `+400 ms` (an extra LLM call).

## Why this matters

In week 12 you defend your Phase II pipeline at the architecture review. The reviewer will not ask you to recite the RRF formula — they'll point at your retrieval stack and ask "why these layers, and how do you know each one is helping?" This challenge *is* that conversation, rehearsed. Every applied-AI interview that touches RAG eventually asks you to justify a pipeline; "we measured the cumulative lift on a fixed gold set and here's the table — the reranker bought us the most MRR per millisecond, and we cut HyDE because it was +0.00 on our corpus" is the answer that gets you hired over "we added all the standard RAG techniques." The mantra is the thesis, and now you can prove it: *a reranker is the cheapest meaningful win in RAG — and here's the number.*
