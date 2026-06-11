# Lecture 1 — Why Dense Is Not Enough: BM25 and Hybrid Search

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can predict which queries dense retrieval will miss, score documents with BM25 by hand and explain `k1` and `b`, fuse two ranked lists with Reciprocal Rank Fusion, and say precisely why rank-based fusion beats score normalization.

If you remember one sentence from this lecture, remember this one:

> **A dense embedding represents what a passage *means*; it does not represent the *words* the passage is made of — so the moment a query hinges on an exact token (an id, a number, an acronym, a rare proper noun), you need a lexical signal back in the loop, and that signal is BM25.**

Week 7 taught you that a `bge-large` embedding is a lossy compression of meaning into 1024 floats. Week 8 taught you that the chunk you embed had better contain the answer. This week is about the failure that remains *even when the embedding is good and the chunk is right*: the failure to match a word. We fix it by adding lexical search, fusing the two retrievers, and — next lecture — reranking the result. By Friday you will have measured exactly what each of those layers is worth on the same 40-query gold set.

---

## 1. Where dense retrieval fails

Dense retrieval is the bi-encoder pipeline you built in week 7: embed the query, embed every document once, and rank by cosine similarity. It is *astonishingly* good at one thing — matching meaning across different words. Ask "how do I end the contract early" and it returns the termination clause even though the clause never says "end" or "early." That is the magic, and it is real.

It is also blind in specific, predictable ways. Here are the four that bite hardest.

### 1.1 The exact-match miss

Ask a dense retriever for **"clause 14"** and watch it flounder. The string `clause_14` is an *identifier*. The embedding of the query "clause 14" lands somewhere in the topical neighborhood of "legal clauses," not on the specific document whose id happens to be `clause_14`. The embedding has no notion that `clause_14` is a *literal token to be matched*; it only knows it's vaguely contract-shaped. The same happens with:

- **Dollar amounts.** "$1,000,000" in `clause_12` is, to the embedding, just "a large money amount, insurance-ish." A query for "the one-million-dollar insurance requirement" might land near it, but "1000000" or "$1M" can drift to a different clause entirely.
- **Statute and section numbers.** "Section 12(b)(3)" is a string. Dense retrieval smears it into "legal cross-reference."
- **Proper nouns and jurisdictions.** "Delaware" in `clause_18` is a rare token with a sharp meaning. A dense model trained on the open web has seen "Delaware" in a thousand non-legal contexts; the embedding doesn't privilege the exact match the way you want.

The general rule: **the more a query depends on a specific surface string, the worse dense retrieval does.** Embeddings are built to *generalize away* from surface form. That is exactly the wrong instinct when the surface form *is* the query.

### 1.2 The rare-term and acronym miss

Embeddings are trained on what's common. A rare term — a niche legal term of art, a product SKU, an acronym that means one thing in your corpus and another on the web — gets a weak, averaged representation. "SOW" might mean "statement of work" in your contracts and "start of week" everywhere else; the embedding splits the difference and matches neither well. BM25, by contrast, doesn't care what "SOW" *means* — it matches the three characters and weights them *up* precisely because they're rare (that's IDF, §3.2).

### 1.3 The negation blind spot (recap from week 7)

You proved this in week 7's homework: "the fee is refundable" and "the fee is **not** refundable" embed with cosine similarity often above 0.9, despite being opposites. The negation token is a tiny fraction of the shared meaning, so the embedding barely moves. Lexical search at least *sees* the word "not" as a distinct token — it doesn't fully solve negation (the real fix is downstream, in the reranker or the LLM reading the actual text), but it stops the two clauses from being indistinguishable.

### 1.4 The topical-distractor miss

This is the subtle one, and it's the one you saw in week 7's exercise-02 note. Query: "how do I end the contract early." The right answer is `clause_14` (termination, "thirty days written notice"). But `clause_09` ("...protected for five years **after termination**") shares the word "termination" and is topically adjacent, so it often ranks #2 — and on a harder query, it can rank #1, pushing the truly-relevant clause down. Dense retrieval ranks by *topical* similarity, and topical similarity is not relevance. This is the gap a reranker closes next lecture; lexical fusion narrows it first.

> **The honest summary:** dense retrieval is the right *first stage* and the wrong *only stage*. It has excellent recall on paraphrase and terrible precision on exact strings. The fix is not a better embedding (week 7 showed that moves recall a point or two); the fix is a *different kind* of signal layered on top.

---

## 2. BM25: search by the words, done right

BM25 ("Best Match 25," the 25th iteration of a family of probabilistic-relevance functions) is the lexical retriever that has been the backbone of search engines for thirty years. It is *sparse*: a document is represented not by a dense vector but by which terms it contains and how often. It scores a document for a query by summing, over every query term, a weight that rewards (a) the term appearing often in the document, (b) the term being rare across the corpus, and (c) the document not being padded with filler.

Let's build the formula one piece at a time, because the pieces are the whole lesson.

### 2.1 Term frequency (TF) — but saturating

The naive idea: a document that contains "termination" five times is more about termination than one that contains it once. True — but *not five times as much*. The jump from 0 to 1 occurrence is enormous (the term is now present at all); the jump from 20 to 21 is meaningless. So BM25 **saturates** term frequency: more occurrences always help, but with sharply diminishing returns.

The saturating TF term is:

```
tf_component(t, d) = f(t, d) * (k1 + 1) / ( f(t, d) + k1 * (1 - b + b * |d| / avgdl) )
```

where `f(t, d)` is the raw count of term `t` in document `d`. Ignore the `b` and `|d|` parts for a moment (set `b = 0`) and it collapses to `f * (k1 + 1) / (f + k1)`, a curve that rises fast then flattens toward `k1 + 1`. That flattening is the saturation, and **`k1` controls how fast it flattens.**

### 2.2 Inverse document frequency (IDF) — rare terms win

A query for "the termination notice" shouldn't be dominated by "the." "the" appears in every document; it discriminates nothing. "termination" appears in a handful; matching it is *informative*. IDF encodes exactly this: a term's weight is inversely related to how many documents contain it.

The BM25 IDF (the Robertson-Sparck-Jones form `rank-bm25` uses) is:

```
idf(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
```

where `N` is the number of documents and `n(t)` is the number containing term `t`. A term in 1 of 50 documents gets a large IDF; a term in 49 of 50 gets a near-zero one. This is *why* BM25 nails the rare-term and identifier queries that dense retrieval fumbles: the rarer the exact string you're matching, the *more* BM25 weights it. "Delaware," appearing in one clause, gets a big IDF and lands `clause_18` at the top.

### 2.3 The two knobs: `k1` and `b`

Put it together. The full Okapi BM25 score of document `d` for query `q`:

```
BM25(q, d) = Σ_{t in q}  idf(t) * [ f(t,d) * (k1 + 1) ]
                                  / [ f(t,d) + k1 * (1 - b + b * |d| / avgdl) ]
```

Two parameters, and you must understand both:

- **`k1` — term-frequency saturation.** Typical range **1.2–2.0** (`rank-bm25`'s `BM25Okapi` defaults to `k1=1.5`). Higher `k1` means term frequency keeps mattering as it grows (the curve flattens later); lower `k1` means TF saturates almost immediately (one occurrence is nearly as good as ten). At `k1 = 0`, term frequency is ignored entirely — a document either has the term or doesn't, binary. You raise `k1` when repetition genuinely signals relevance (long documents where a term recurring matters); you lower it when one mention is enough (short, dense clauses like ours).

- **`b` — length normalization.** Range **0.0–1.0** (`BM25Okapi` defaults to `b=0.75`). It controls how hard you penalize long documents. The `|d| / avgdl` ratio is "how long is this document relative to the average." At `b = 1.0`, you fully normalize: a long document needs proportionally more occurrences to score as high as a short one (this stops a 50-page document from winning every query just by being big enough to contain everything). At `b = 0.0`, length is ignored entirely. The default `0.75` says "mostly penalize length, but not completely." You lower `b` when your documents are uniform in length (our clauses are all short — length normalization barely matters); you raise it when document lengths vary wildly and the long ones are gaming you.

> **The intuition to keep:** `k1` is about *one document's* term counts (does repetition help?), `b` is about *comparing documents of different lengths* (is this document long because it's thorough or because it's padded?). For the legal-clause corpus — short, uniform documents — the defaults are fine and you'll see why in the exercise.

### 2.4 BM25 in code (the real thing)

We use `rank-bm25` for the labs — pure Python, `pip install rank-bm25`, no infrastructure. Here is the whole pattern over the legal corpus:

```python
from rank_bm25 import BM25Okapi

# Documents as (doc_id, text). The same 8-clause starter from week 7's exercise-02.
CORPUS = [
    ("clause_01", "This Agreement is entered into between the Company and the Contractor."),
    ("clause_07", "The annual fee shall be paid in twelve equal monthly installments."),
    ("clause_09", "All confidential information must be protected for five years after termination."),
    ("clause_12", "The Contractor shall maintain professional liability insurance of $1,000,000."),
    ("clause_14", "Either party may terminate this Agreement upon thirty days written notice."),
    ("clause_18", "This Agreement is governed by the laws of the State of Delaware."),
    ("clause_22", "Neither party shall be liable for delays caused by events beyond its control."),
    ("clause_27", "Any dispute shall be resolved by binding arbitration in San Francisco."),
]

doc_ids = [doc_id for doc_id, _ in CORPUS]


def tokenize(text: str) -> list[str]:
    """A real tokenizer would lowercase, strip punctuation, maybe stem.
    For a lexical baseline, lowercasing + splitting on non-alphanumerics is
    enough — but keep '$1,000,000' and 'clause_14' as matchable tokens."""
    import re
    # Keep digits, letters, $ and , together so money survives; split the rest.
    return re.findall(r"[\w$,.]+", text.lower())


tokenized_corpus = [tokenize(text) for _, text in CORPUS]
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)  # the defaults, made explicit


def bm25_search(query: str, k: int = 3) -> list[tuple[str, float]]:
    scores = bm25.get_scores(tokenize(query))           # one score per document
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [(doc_ids[i], float(scores[i])) for i in order[:k]]


for q in ["Delaware", "one million dollar insurance", "clause 14",
          "how do I end the contract early"]:
    print(f"\nquery: {q!r}")
    for rank, (doc_id, score) in enumerate(bm25_search(q), start=1):
        print(f"  #{rank}  {doc_id}  bm25={score:.3f}")
```

Run it and read the results against §1:

- **`"Delaware"`** → `clause_18` at #1, with a high score, because "delaware" is rare (IDF large) and present. Dense retrieval was fuzzy here; BM25 is certain.
- **`"clause 14"`** → BM25 matches the literal token if it's in the text or metadata; even partial overlap ("clause") plus the rarity of the id ranks it. (In the exercise you'll add the `doc_id` to the searchable text so the id is matchable — a real-world trick.)
- **`"how do I end the contract early"`** → BM25 *fails* this one. It shares no words with "Either party may terminate... upon thirty days written notice." No "end," no "early," no "contract" overlap strong enough. BM25 might return `clause_14` weakly or miss it entirely. **This is BM25's blind spot — the paraphrase query — and it's exactly dense retrieval's strength.**

That last bullet is the entire argument for hybrid search. BM25 and dense fail on *opposite* queries. So you run both.

### 2.5 A worked BM25 score, by hand

To make `k1`, `b`, and IDF concrete, score one document for one query the long way. Take the corpus above (8 clauses, `N = 8`) and the query `"liability insurance"` against `clause_12` ("The Contractor shall maintain professional liability insurance of $1,000,000."). The query terms are `liability` and `insurance`.

First, IDF. Count how many of the 8 clauses contain each term:

- `liability` appears in `clause_12` and `clause_22` ("Neither party shall be **liable**..." — *not* a match unless you stem; assume the exact token `liability` appears only in `clause_12`), so `n(liability) = 1`.
- `insurance` appears only in `clause_12`, so `n(insurance) = 1`.

Both are rare (1 of 8), so both get a large IDF:

```
idf(liability) = ln((8 - 1 + 0.5) / (1 + 0.5) + 1) = ln(7.5/1.5 + 1) = ln(6) ≈ 1.79
idf(insurance) = ln((8 - 1 + 0.5) / (1 + 0.5) + 1) = ln(6) ≈ 1.79
```

Now the saturating TF term. Each query term appears **once** in `clause_12` (`f = 1`). The clauses are short and roughly average length, so `|d| / avgdl ≈ 1`, which makes the length factor `(1 - b + b·1) = 1` regardless of `b` (this is *why* `b` barely matters on a uniform corpus). With `k1 = 1.5`:

```
tf_component = f·(k1 + 1) / (f + k1·1) = 1·2.5 / (1 + 1.5) = 2.5 / 2.5 = 1.0
```

So each term contributes `idf × tf_component = 1.79 × 1.0 ≈ 1.79`, and the document's BM25 score for the query is the sum over query terms:

```
BM25("liability insurance", clause_12) ≈ 1.79 + 1.79 = 3.58
```

Two things to take from this. First, the score is **large precisely because the terms are rare** — IDF did the work, which is exactly why BM25 nails the niche-term queries dense retrieval fumbles. Second, notice that `k1` and `b` made almost no difference here: TF was 1 (no repetition to saturate) and the document was average-length (nothing to normalize). On *this* corpus the BM25 parameters are nearly inert, and the defaults are fine — a real result you'll confirm empirically in Exercise 1, and a reminder that the parameters earn their keep only on long, length-varying documents where terms repeat.

---

## 3. Hybrid search: run both, then combine

Hybrid search is the deliberately unglamorous idea of running a dense retriever and a sparse (BM25) retriever over the same corpus, getting two ranked lists, and combining them into one. The dense list is strong on paraphrase; the sparse list is strong on exact terms; the combined list is strong on both. The only question is *how* to combine two ranked lists, and there are two answers — one fragile, one robust.

### 3.1 The fragile way: score normalization / weighted fusion

The obvious idea: each retriever produces a score per document, so combine the scores:

```
combined(d) = α * dense_score(d) + (1 - α) * bm25_score(d)
```

This *can* work, but it's a minefield, because **the two scores live on incomparable scales.** A dense cosine similarity is in `[-1, 1]` (or `[0, 1]` for normalized embeddings), tightly clustered around 0.3–0.8 in practice. A BM25 score is *unbounded* — it can be 2 for one query and 18 for another, depending on the IDF of the query terms and how many matched. You cannot add 0.7 (cosine) to 14.2 (BM25) and get anything meaningful.

So you must **normalize** first — min-max scale each list to `[0, 1]`, or z-score it — and now you've introduced a new problem: the normalization depends on the *set* of scores you happened to retrieve, which depends on `k`, which means the same document gets a different normalized score depending on how deep you looked. You're also now tuning `α` (the weight) *and* the normalization scheme, per corpus, per query distribution. It's calibration all the way down, and it breaks quietly when the score distribution shifts.

### 3.2 The robust way: Reciprocal Rank Fusion (RRF)

RRF (Cormack, Clarke & Büttcher, SIGIR 2009) throws the scores away and fuses on **rank** instead. The insight: you don't trust either retriever's *scores*, but you trust its *ordering*. A document that both retrievers rank near the top is probably relevant, regardless of what numbers they assigned. The formula:

```
RRF_score(d) = Σ_{r in rankers}  1 / (k + rank_r(d))
```

where `rank_r(d)` is the **1-based** position of document `d` in ranker `r`'s list (rank 1 is the top), the sum runs over every ranker, and `k` is a constant (the paper's value, and everyone's default, is **`k = 60`**). A document missing from a ranker's list simply contributes nothing from that ranker (equivalently, its rank is treated as infinite, so `1/(k+∞) ≈ 0`).

Why `1/(k + rank)`? Two properties:

1. **It's steeply top-weighted but bounded.** Rank 1 contributes `1/61 ≈ 0.0164`; rank 2 contributes `1/62 ≈ 0.0161`; rank 10 contributes `1/70 ≈ 0.0143`. The gaps between top ranks are small but real, and they shrink as you go down — so being in the top of *both* lists beats being #1 in one and absent from the other, while a single retriever's #1 still counts for a lot. The `k = 60` offset is what flattens the curve so a lucky #1 doesn't dominate; a smaller `k` makes the top rank far more decisive, a larger `k` flattens it toward "did you appear at all."
2. **It needs no calibration.** There are no scores to normalize, no `α` to tune, no per-query scale to estimate. Rank is rank. This is why RRF is the default in production hybrid search and why we use it for the rest of the course.

### 3.3 RRF worked by hand

Two retrievers, query "how do I end the contract early." Suppose:

```
dense ranks:  [clause_14, clause_09, clause_18, clause_07]   # paraphrase-strong
bm25 ranks:   [clause_09, clause_27, clause_14, clause_18]   # matched "termination"/"dispute" words
```

Dense nailed `clause_14` at #1 (it understood "end the contract" = terminate). BM25 put `clause_09` first (it shares the literal word "termination") and pushed the right answer, `clause_14`, down to #3. Neither list alone is ideal — dense is right but we want corroboration; BM25 is led astray by a topical word. Fuse them with `k = 60`:

```
clause_14:  1/(60+1)  [dense #1]  + 1/(60+3)  [bm25 #3]  = 0.01639 + 0.01587 = 0.03226
clause_09:  1/(60+2)  [dense #2]  + 1/(60+1)  [bm25 #1]  = 0.01613 + 0.01639 = 0.03252
clause_18:  1/(60+3)  [dense #3]  + 1/(60+4)  [bm25 #4]  = 0.01587 + 0.01563 = 0.03150
clause_07:  1/(60+4)  [dense #4]  + 0          [bm25 ∅]   = 0.01563
clause_27:  0          [dense ∅]   + 1/(60+2)  [bm25 #2]  = 0.01613
```

Fused order: `clause_09` (0.03252), `clause_14` (0.03226), `clause_18`, `clause_27`, `clause_07`. Notice what happened: `clause_14` and `clause_09` are now neck-and-neck — fusion *corroborated* both, and the right answer `clause_14` is at rank 2, up from where BM25 had it (rank 3) and pulled toward the top by the dense retriever's confidence. This is fusion working: it didn't magically put `clause_14` first (the reranker next lecture does that), but it produced a list where the right answer is reliably near the top *across query types*, which is the recall property hybrid search is for. Change the lists so dense and BM25 *agree* on `clause_14` and it shoots to #1 — agreement between rankers is what RRF rewards.

### 3.4 RRF in code

```python
def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked lists of doc_ids into one by Reciprocal Rank Fusion.

    ranked_lists: e.g. [dense_ids, bm25_ids], each best-first (rank 1 = index 0).
    Returns [(doc_id, rrf_score)] sorted best-first.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):   # 1-based rank — this matters
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


dense_ids = ["clause_14", "clause_09", "clause_18", "clause_07"]
bm25_ids = ["clause_09", "clause_27", "clause_14", "clause_18"]
for doc_id, score in rrf_fuse([dense_ids, bm25_ids], k=60):
    print(f"{doc_id}  rrf={score:.5f}")
```

Three things to never get wrong, because they're the bugs we see every cohort:

1. **Rank is 1-based.** `enumerate(..., start=1)`. If you use 0-based ranks, your top document gets `1/(k+0) = 1/60` and the formula is subtly off — and at `k=0` it would divide by zero. Start at 1.
2. **A missing document contributes 0, not a penalty.** You don't punish a document for being absent from one list; you just don't reward it from that list. The `dict.get(doc_id, 0.0)` accumulation handles this automatically.
3. **`k` is a fusion constant, not a retrieval `k`.** Don't confuse RRF's `k=60` with "top-k results." They're different `k`s that unfortunately share a letter. We write the fusion constant as `k=60` and the result count as `top_k` to keep them apart in the mini-project.

> **The mantra for §3:** *Fuse on rank, not on score. RRF with k=60 is the robust default; reach for weighted score fusion only when you have a calibrated reason and a held-out set to tune it on — which, at the legal-corpus scale, you don't.*

### 3.5 How deep to retrieve before fusing

One parameter you *do* control: how many candidates each leg returns before you fuse. Call it the *first-stage depth*. If the dense leg returns only its top-5 and BM25 only its top-5, then a document that dense ranked #8 — but which BM25 ranked #1 — can never be fused, because dense never surfaced it. You've thrown away exactly the cross-retriever rescue RRF exists to perform.

So retrieve **wide** in the first stage, fuse, then narrow. A common shape: each leg returns its top **50–100**, you fuse those into one list, and you keep the fused top-k for whatever comes next (the reranker, in Lecture 2). The cost is cheap — a bigger `LIMIT` on a vector query and a few more BM25 scores — and the benefit is real: the more candidates each leg contributes, the more chances RRF has to find a document both legs liked.

There's a tension with the reranker, though, and it's worth flagging now. The reranker (next lecture) is *expensive per candidate*, so you can't rerank 1,000 fused documents. The standard resolution: **fuse wide (top 50–100 per leg), then rerank the fused top-50, then keep the top-5.** The first stage is generous because retrieval is cheap; the second stage is selective because reranking is not. Get this wrong — fuse only the top-5 and call the reranker on those — and you cap the whole pipeline's recall at whatever the first stage's top-5 caught. We return to this in the Challenge's "trap" section, because it's the single most common way to build a pipeline that *looks* complete but quietly underperforms.

> **Rule of thumb:** first-stage depth ≥ reranker input ≥ final result count. Retrieve 50, rerank 50, return 5. Never let the final count constrain the first-stage depth.

---

## 4. Where the lift actually comes from

You now have two retrievers and a way to fuse them. Before we add the reranker next lecture, internalize the *shape* of the lift you should expect, because measuring it is the whole point of the week.

On a typical RAG corpus:

- **BM25 alone** has decent precision on keyword queries and poor recall on paraphrase. Call it a baseline.
- **Dense alone** has the opposite profile: strong recall on paraphrase, weak on exact terms. On most gold sets it *beats* BM25 alone on aggregate Recall@5, often by a clear margin — which is why dense became the default. But it's not uniformly better: on the keyword/identifier queries, BM25 wins outright.
- **Hybrid (dense + BM25, RRF)** beats *either* alone, because it recovers the queries each one missed. The lift is real but usually modest in aggregate (a handful of points of Recall@5) — its value is in the *tail*, the specific queries dense missed entirely now get rescued by BM25 and vice versa.
- **+ reranker** (next lecture) is typically the single best lift-per-effort, especially on MRR — it reorders the fused candidates so the right one lands at rank 1.
- **+ HyDE** (next lecture) sometimes helps recall on genuinely hard paraphrase queries and sometimes hurts precision when the hypothetical answer hallucinates off-topic. It's the layer you must *measure* rather than assume.

The discipline this week is to put each of those on **one row of one table**, measured by the **same** `evaluate()` from week 7, on the **same** 40-query gold set. "Did hybrid help?" is a `Δ Recall@5`. "Did the reranker earn its 30 ms?" is a `Δ MRR`. Numbers, not vibes — same as weeks 7 and 8.

---

## 5. A note on the production path

`rank-bm25` is perfect for learning and for a 50-document corpus. It is *not* what you ship at scale, because it tokenizes and scores in pure Python and holds the whole index in memory. The production lexical-search options, in rough order of how far they scale:

- **Postgres full-text search** (`tsvector` / `tsquery`, `ts_rank_cd`) — already in your database, fuses naturally with pgvector in one query (the pgvector hybrid-search docs show exactly this), and good to surprisingly large corpora. The sane next step.
- **`bm25s`** — a fast pure-Python BM25 on sparse `scipy` matrices, orders of magnitude quicker than `rank-bm25`, drop-in for the same API shape.
- **Tantivy** (Rust, with `tantivy-py` bindings) — a Lucene-in-Rust; this is the "real search engine" tier without standing up a cluster.
- **Elasticsearch / OpenSearch** — the heavyweight, when you need distributed lexical search and you're already running the cluster for other reasons.

The *concepts* — TF, IDF, `k1`, `b`, and RRF — are identical across all of them. Learn them on `rank-bm25`; the syllabus lab's "Tantivy or Elasticsearch" is the same BM25 with a faster engine underneath. The mini-project keeps the BM25 leg behind an interface precisely so you can swap `rank-bm25` for Postgres full-text in week 10 without touching the fusion code.

One concrete preview of that production path, because you already have pgvector running: Postgres can do *both* legs in one place. You add a `tsvector` column (`ALTER TABLE chunks ADD COLUMN ts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;`), build a GIN index on it, and now a single database can rank by `<=>` (dense cosine) *and* by `ts_rank_cd(ts, plainto_tsquery('english', $1))` (lexical). You can even fuse them inside SQL with a CTE per leg and an RRF expression over the row numbers — which is exactly what the pgvector hybrid-search docs demonstrate. For a 50-clause corpus that's overkill; for the production system you'll design in week 10 it's the sane default, because it keeps both retrievers, the fusion, and the documents in one transactional store instead of three services you have to keep in sync. Either way the fusion math is the RRF you just learned — the engine underneath changes, the `Σ 1/(k + rank)` does not.

---

## 6. Recap

You should now be able to:

- Predict the four ways dense retrieval fails — exact-match, rare-term/acronym, negation, and topical-distractor — and name a query that triggers each.
- Write the Okapi BM25 score from its parts: IDF × saturating-TF with length normalization.
- Explain `k1` (term-frequency saturation; ~1.5) and `b` (length normalization; ~0.75) correctly, and say when you'd move each.
- Build a `rank-bm25` index over a corpus and show it winning exactly the queries dense missed.
- Fuse two ranked lists with RRF — `Σ 1/(k + rank)`, k=60, 1-based rank — by hand and in code.
- Argue why RRF (rank-based, calibration-free) beats weighted score fusion (scale-mismatched, tuning-heavy) at this scale.
- Set the first-stage depth correctly (retrieve wide, fuse, narrow) so the fusion has candidates to rescue and the reranker has a real candidate set to work on.

Next up: the layer that turns a good candidate set into a *correctly ordered* one. A cross-encoder reads the query and each passage together and re-scores them — the cheapest meaningful win in RAG. Then HyDE and text-to-SQL, for the queries that retrieval alone can't reach. Continue to [Lecture 2 — Rerankers and Structured Retrieval](./02-rerankers-and-structured-retrieval.md).

---

## References

- *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* — Cormack, Clarke & Büttcher, SIGIR 2009 (the RRF paper, source of k=60): <https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf>
- *The Probabilistic Relevance Framework: BM25 and Beyond* — Robertson & Zaragoza, 2009 (the definitive BM25 treatment): <https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf>
- *`rank-bm25`* — the library used in the labs (`BM25Okapi`, defaults `k1=1.5`, `b=0.75`): <https://github.com/dorianbrown/rank_bm25>
- *`bm25s`* — the fast sparse-matrix BM25 for when the corpus outgrows the toy size: <https://github.com/xhluca/bm25s>
- *Tantivy* — the Rust full-text engine (the production lexical path): <https://github.com/quickwit-oss/tantivy>
- *pgvector hybrid search* — combining `<=>` with Postgres full-text in one query: <https://github.com/pgvector/pgvector#hybrid-search>
- *SPLADE* — Formal et al., 2021 (learned sparse retrieval, where bge-m3's sparse output comes from): <https://arxiv.org/abs/2107.05720>
