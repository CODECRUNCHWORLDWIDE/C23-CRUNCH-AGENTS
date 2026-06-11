# Challenge 1 — The Embedding Bakeoff

**Time estimate:** ~90 minutes.

## Problem statement

Your team is starting a RAG product over a corpus of services agreements. Someone asks, in the architecture review, "which embedding model are we using and why?" The wrong answer is "the one at the top of MTEB." The right answer is "we ran a bakeoff on our own corpus with our own gold set; here are the numbers; here's the pick and the trade-off." This challenge is that bakeoff.

You will embed **one corpus** with **four embedding models**, index each in pgvector, run a **40-query gold set**, report **top-1 / top-5 / MRR** for each, and write a **one-page memo** picking a winner with reasons. This mirrors the syllabus lab exactly ("embed a legal corpus with three open embeddings and one vendor; report top-1/top-5/MRR; defend a choice in a 1-page memo").

This is the real skill: not knowing which model is "best" in the abstract, but running a *fair* comparison on *your* data and reading the result without flinching when the leaderboard's favorite loses.

## The corpus and gold set

Use the legal corpus and gold set from the mini-project skeleton (`mini-project/corpus/`). If you haven't pulled it yet, the minimal version is:

- **Corpus:** 50 clauses from a synthetic services agreement, one clause per document, each with a `doc_id`. (The exercise-02 corpus is a starter; the full 50-clause set is in the mini-project.)
- **Gold set:** 40 questions, each labeled with the `doc_id` of the clause that answers it. Format: a JSON list of `{"query": "...", "relevant": ["clause_14"]}`.

If you want to build your own, that's allowed and instructive — but **the gold set must be fixed across all four models**, or the comparison is meaningless.

## The four models

Three open, one vendor:

1. `BAAI/bge-large-en-v1.5` (1024-dim, query prefix required)
2. `Alibaba-NLP/gte-large-en-v1.5` (1024-dim, no prefix)
3. `nomic-ai/nomic-embed-text-v1.5` (768-dim, `search_query:`/`search_document:` prefixes, `trust_remote_code=True`)
4. **One vendor:** OpenAI `text-embedding-3-large` (set `dimensions=1024`) **or** Cohere `embed-english-v3.0` (set `input_type` correctly). **If you have no API key**, substitute a fourth open model — `BAAI/bge-m3` (1024-dim) — and note in your memo that you ran four open models instead of three-plus-one.

> **The fairness rule.** Each model gets its *own* correct prefix/input_type convention. Running BGE without its prefix or Cohere with the wrong `input_type` isn't a fair loss — it's a bug you introduced. A fair bakeoff gives every model its best shot. Getting this right *is* the challenge.

## Your task

For **each of the four models**, produce:

1. **A separate pgvector table** (or a `model` column to filter on) — vectors of different dimensions cannot share a column.
2. **The three metrics** on the 40-query gold set: top-1, Recall@5, MRR. (Reuse the `evaluate()` function from Lecture 2 §3.3.)
3. **The per-model row** in a results table.

Then write the memo.

## The deliverables

### 1. The results table

A markdown table, one row per model:

| Model | Dim | top-1 | Recall@5 | MRR | Embed time (s) | Notes |
|---|---:|---:|---:|---:|---:|---|
| bge-large-en-v1.5 | 1024 | ... | ... | ... | ... | query prefix applied |
| gte-large-en-v1.5 | 1024 | ... | ... | ... | ... | no prefix |
| nomic-embed-text-v1.5 | 768 | ... | ... | ... | ... | search_query/document |
| (vendor or bge-m3) | ... | ... | ... | ... | ... | input_type / api cost |

### 2. The one-page memo (`bakeoff-memo.md`, 350–550 words)

Hit these sections:

1. **Recommendation** — one sentence: which model, for this corpus.
2. **The numbers** — the table above, plus one line on what they show (is the spread big or small?).
3. **The trade-off** — why this model over the runner-up. Recall? Dimension (storage/latency)? License? Per-query cost? Operational simplicity? Name the axis that decided it.
4. **The MTEB reality** — one sentence comparing your pick's *measured* result to its MTEB leaderboard position. Did the leaderboard's favorite win on your data? If not, say so — that's the most valuable sentence in the memo.
5. **What I'd do with two more weeks** — the honest caveat (bigger gold set, a reranker leg, a second corpus).

## Acceptance criteria

- [ ] A file `bakeoff-results.md` with the four-model results table, every number computed from a real `evaluate()` run (not estimated).
- [ ] Each model used its **correct** prefix/input_type convention — state which convention per model in the table's Notes column.
- [ ] The gold set was **identical** across all four models (40 queries, same labels).
- [ ] A `bakeoff-memo.md` of 350–550 words hitting all five sections, with a *specific* reason for the pick (not "it had the best numbers" — *which* number, and the trade-off you accepted).
- [ ] The memo's MTEB sentence is honest: it either confirms the leaderboard or names the discrepancy.
- [ ] Committed to your week-7 repo under `challenges/challenge-01/`.

## The trap (read after a first attempt)

The subtle, realistic trap is the **prefix/input_type fairness bug**. The most common way people botch a bakeoff is to write one clean encode loop and reuse it for every model — which means three of the four models silently run with the *wrong* convention. BGE without its query prefix loses ~5 points of recall for no reason; Cohere with `input_type="search_document"` on the *queries* loses more. If your bakeoff shows one model crushing the others by 15+ points, **suspect a fairness bug before you believe the result** — you probably gave one model its prefix and starved the rest. Re-check that every model's query encoding uses *that model's* convention. A bakeoff that isn't fair is worse than no bakeoff, because it produces a confident wrong decision.

## Stretch

- **Add a Matryoshka leg.** Run nomic at full 768-dim and truncated to 256-dim (re-normalized). Report both rows. How much Recall@5 do you trade for the 3x storage saving? This is a real production lever and a great memo paragraph.
- **Add `ef_search` to the table.** Run the winning model at `ef_search` ∈ {40, 100, 200} and show that the embedding choice and the index tuning are *separate* levers — the bakeoff picks the model, `ef_search` tunes the index.
- **Cost column.** For the vendor model, compute the actual dollar cost to embed the corpus once and to run the 40 queries. Put it in the table. The cost-vs-recall trade-off is often what actually decides the pick in production.

## Why this matters

In week 12 you defend your Phase II pipeline at the architecture review. The reviewer will not ask you to recite embedding dimensions — they'll point at your retrieval layer and ask "why this embedding, and how would you know if a different one was better?" This challenge *is* that conversation, rehearsed. Every applied-AI interview that touches RAG eventually asks you to justify a model choice; "we benchmarked it on our data with a gold set and here's the memo" is the answer that gets you hired over "it's #1 on MTEB."
