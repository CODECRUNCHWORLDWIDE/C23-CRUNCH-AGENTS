# Challenge 1 — The Chunking A/B

**Time estimate:** ~150 minutes.

## Problem statement

You have a corpus and a retrieval target. Four reasonable people on your team each swear by a different chunking strategy. You are going to end the argument the only way it can honestly end: run all four against the *same* embedding and the *same* store, on the *same* gold set, and let the metric pick the winner. Then you write down *why* it won — because a winner you can't explain is a winner you got lucky with.

This is the syllabus deliverable in lab form. The output is a decision: one chunking strategy, with Recall@5, MRR, and faithfulness numbers, and a paragraph of reasons grounded in *this* corpus.

## The four strategies

Run exactly these four, and nothing else varies:

1. **Token-window 512** — fixed window, 512 BGE tokens, no overlap (or a small fixed overlap you hold constant). The baseline.
2. **Token-window 1024** — same, at 1024 tokens. Tests the "bigger is more self-contained but more diluted" axis directly.
3. **Recursive** — LangChain-style separator hierarchy (`["\n\n", "\n", ". ", " ", ""]`), `chunk_size` measured in BGE tokens, modest overlap. The structure-respecting workhorse.
4. **Late chunking** — embed the *whole document* through a long-context model (jina-embeddings-v3) in one forward pass, then mean-pool token embeddings per chunk span (arXiv 2409.04701).

> **The one honest asymmetry:** strategies 1–3 use BGE-large; late chunking *requires* a long-context model (jina-v3) because its whole mechanism is one forward pass over the entire document. You cannot late-chunk through BGE's 512-token window. Call this out in the memo — it means late chunking's number includes a model change, so its win (or loss) is "late chunking *with jina-v3*," not late chunking in isolation. That's a real caveat, not a flaw; just state it.

## What is fixed (do not let these vary)

- **Store:** pgvector, `vector_cosine_ops`, the `<=>` operator, the same `ef_search` for every run. (Your week-7 `store.py`, unchanged.)
- **Gold set:** the 40-query legal gold set from week 7, unchanged. Gold is in *clause ids*; you retrieve *chunks*; you map chunk hits back to clause ids before scoring.
- **Metric suite:** Recall@5 and MRR (the spine) via week-7's `evaluate()` unchanged, plus faithfulness (the tie-breaker).
- **Embedding:** BGE-large for 1–3 (the jina-v3 swap for late chunking is the noted exception).

## The harness approach

The whole A/B reduces to: build a *different* `retrieve_fn` per strategy, pass each to the *same* `evaluate()`.

```python
from crunchrag_embed.eval import evaluate     # week 7, UNCHANGED
from crunchrag_embed import store             # week 7, UNCHANGED

def retriever_for(strategy_name, chunks_with_source, embed_fn):
    """chunks_with_source: [(chunk_id, clause_id, text)]. embed_fn embeds them."""
    table = f"chunks_{strategy_name}"
    store.create_table(table, dim=embed_fn.dim)
    store.insert(table, [(cid, txt, v)
                         for (cid, _, txt), v in zip(chunks_with_source,
                                                     embed_fn.embed_documents(
                                                         [c[2] for c in chunks_with_source]))])
    store.build_hnsw(table)
    chunk_to_clause = {c[0]: c[1] for c in chunks_with_source}

    def retrieve_fn(query):
        hits = store.knn(table, embed_fn.embed_query(query), k=20)
        seen, ranked = set(), []
        for h in hits:
            clause = chunk_to_clause[h]
            if clause not in seen:
                seen.add(clause); ranked.append(clause)
        return ranked
    return retrieve_fn

for name, (chunks, embed_fn) in strategies.items():
    fn = retriever_for(name, chunks, embed_fn)
    print(name, evaluate(gold, fn, k=5))
```

For **late chunking**, `chunks` are token *spans*, `embed_fn.embed_documents` does the one-forward-pass-then-pool from Lecture 1 §6, and `embed_fn.dim` is jina-v3's dimension — but the `retrieve_fn` shape and the `evaluate()` call are *identical*. That identical eval call is the whole point: you changed only the chunker.

For **faithfulness**, take the winning-by-recall two or three strategies, generate an answer per gold query from each strategy's top-k chunks, and score grounding with an LLM-as-judge (Ragas-style, Lecture 2 §5.3). Keep it lightweight; it's the tie-breaker, not the spine.

## Acceptance criteria

- [ ] A `challenge-01/` directory with a runnable `chunk_ab.py` that runs all four strategies against the fixed embedding + store + gold set and prints a comparison table.
- [ ] The table reports **Recall@5, MRR, and faithfulness** for all four strategies, plus chunk count and embed time.
- [ ] The store, gold set, and (for 1–3) embedding are demonstrably **identical** across runs — only the chunker varies. The late-chunking model swap is explicitly noted.
- [ ] Gold ids are clause ids; chunk hits are **mapped back to clause ids** before scoring (a strategy "finds clause 9" if any of its chunks ranks in top-k).
- [ ] A one-page `chunking-ab-memo.md` that names the **winner**, gives its three numbers, and explains in a paragraph **why it won on this corpus** (not in general).
- [ ] At least one **per-query trace** in the promise format, e.g. `q12 ("five-year confidentiality") -> chunk_09 (rank 1)`, showing the answer survived the chunking for the winner — and a counter-example query where a losing strategy split or buried an answer.

## The trap (read after a first attempt)

The trap is **scoring chunks instead of answers.** Your gold set is in clause ids, but the four strategies produce *different chunks* — token-window 512 makes one chunk per ~512 tokens, recursive makes one per clause, late chunking pools spans. If you score "did the right *chunk* come back?" you can't compare strategies, because they don't share a chunk vocabulary. You **must** map every chunk back to its source clause id and score "did the right *clause's* chunk come back in top-k?" Two strategies that both find clause 9 — one in a tight 512-token chunk, one in a 1024-token chunk — both score a hit, and now the comparison is fair. Skip the mapping and your A/B is comparing apples to differently-cut apples, and the numbers mean nothing. (This is exactly the `chunk_to_clause` map in the harness above; if you delete it, you've fallen in the trap.)

A second, subtler trap: **letting late chunking's model change masquerade as a chunking win.** If late-chunking-with-jina beats recursive-with-BGE, you have *not* shown late chunking beats recursive — you've shown jina-v3 + late chunking beats BGE + recursive, two changes at once. To isolate the chunking effect honestly, also run jina-v3 with *early* (independent) chunking as a control; the late-vs-early gap *at the same model* is the real late-chunking lift. Note this even if you don't run the full control.

## Stretch goals

- **Add the early-vs-late control.** Run jina-v3 with independent chunking and with late chunking on the same spans. The delta is the pure late-chunking effect, model held constant — the honest measurement the paper reports.
- **Overlap sweep on the winner.** Take the winning strategy and sweep overlap (0, 32, 64, 128). Does overlap move Recall@5 on a corpus of self-contained clauses? Predict first, then measure.
- **Metadata injection.** For the winning strategy, prepend each chunk's clause heading ("Termination", "Confidentiality") to the chunk text before embedding. Measure whether the heading injection moves Recall@5 — the README stretch goal, now with a number.
- **Faithfulness on the close calls.** If two strategies tie on Recall@5, generate answers and judge faithfulness on the 5 queries where they disagree most. The strategy whose chunks produce *complete* answers wins the tie.

## Why this matters

In Week 12 you defend your whole retrieval pipeline at the architecture review. The reviewer will not ask you to recite the five strategies — they'll point at your index and ask "why *that* chunking, and how do you know it's better than the obvious alternative?" This challenge *is* that conversation, rehearsed: you ran the alternatives, you have the table, you can name the winner and the number that justifies it. Every RAG system you ship after this gets chunked by some strategy whether you chose it or not — the engineer who *chose* it, with a measured A/B behind the choice, is the one whose retrieval doesn't quietly fail in production. The clause survived the cut, and you can prove it.
