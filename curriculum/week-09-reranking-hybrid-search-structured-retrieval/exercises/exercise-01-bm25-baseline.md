# Exercise 1 — BM25 Baseline

**Goal:** Build a real BM25 index over the legal corpus with `rank-bm25`, run the 40-query gold set through it, and *prove* — with numbers — that BM25 wins exactly the queries dense retrieval missed: an identifier, a jurisdiction ("Delaware"), and a dollar amount ("$1,000,000"). By the end you'll have a measured BM25 baseline that becomes the first row of your cumulative-lift table.

**Estimated time:** 45 minutes. Guided.

---

## Setup

```bash
# Reuse your week-7 venv, then add rank-bm25 (pure Python, no infra):
pip install rank-bm25 numpy
```

No model download, no Postgres needed for the BM25 part. That's the point — lexical search is cheap.

---

## Step 1 — Build a BM25 index over the corpus

Save as `bm25_baseline.py`. We use the same legal corpus you've carried since week 7, plus a few more clauses so the gold set has range. **Note the trick on line marked `# id trick`:** we prepend the `doc_id` to the searchable text so an identifier query like "clause 14" can match lexically — a real-world move, because ids are exactly what dense retrieval can't represent.

```python
from rank_bm25 import BM25Okapi
import re

# The legal corpus: (doc_id, clause text). Same style as weeks 7-8.
CORPUS = [
    ("clause_01", "This Agreement is entered into between the Company and the Contractor."),
    ("clause_03", "The initial term of this Agreement is two years from the effective date."),
    ("clause_07", "The annual fee shall be paid in twelve equal monthly installments."),
    ("clause_09", "All confidential information must be protected for five years after termination."),
    ("clause_12", "The Contractor shall maintain professional liability insurance of $1,000,000."),
    ("clause_14", "Either party may terminate this Agreement upon thirty days written notice."),
    ("clause_18", "This Agreement is governed by the laws of the State of Delaware."),
    ("clause_22", "Neither party shall be liable for delays caused by events beyond its control."),
    ("clause_27", "Any dispute shall be resolved by binding arbitration in San Francisco."),
    ("clause_31", "Intellectual property created under this Agreement is owned by the Company."),
]

doc_ids = [doc_id for doc_id, _ in CORPUS]


def tokenize(text: str) -> list[str]:
    """Lowercase and split, keeping $, commas, and digits together so that
    '$1,000,000' and 'clause_14' survive as single matchable tokens."""
    return re.findall(r"[\w$,]+", text.lower())


# id trick: make the doc_id itself searchable by prepending it to the text.
# 'clause 14' in a query can now match the 'clause_14' token.
searchable = [f"{doc_id} {text}" for doc_id, text in CORPUS]
tokenized = [tokenize(s) for s in searchable]

bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)  # the Okapi defaults, made explicit


def bm25_search(query: str, k: int = 3) -> list[tuple[str, float]]:
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [(doc_ids[i], float(scores[i])) for i in order[:k]]


if __name__ == "__main__":
    for q in ["clause 14", "Delaware", "$1,000,000 insurance", "how do I end the contract early"]:
        print(f"\nquery: {q!r}")
        for rank, (doc_id, score) in enumerate(bm25_search(q), start=1):
            print(f"  #{rank}  {doc_id}  bm25={score:.3f}")
```

Run it. **What to notice:**

- `"clause 14"` → `clause_14` at #1, because the id trick made the literal token matchable. Dense retrieval cannot do this — an id is not a meaning.
- `"Delaware"` → `clause_18` at #1, with a high score, because "delaware" is rare (large IDF) and present. This is BM25's home turf.
- `"$1,000,000 insurance"` → `clause_12` at #1, because the exact money token and "insurance" both match.
- `"how do I end the contract early"` → BM25 **fails or scores low**, because the query shares no words with "Either party may terminate... upon thirty days written notice." *This* is the query dense retrieval nails and BM25 fumbles — the exact mirror image of the three above.

---

## Step 2 — Run the gold set through BM25

Now measure BM25 on the real metric. Here is a 12-query slice of the 40-query gold set (the full set ships in the mini-project skeleton). Each query is labeled with the `doc_id` that answers it.

```python
GOLD = [
    {"query": "what notice is required to terminate", "relevant": ["clause_14"]},
    {"query": "how do I end the contract early", "relevant": ["clause_14"]},
    {"query": "how long is confidential information protected", "relevant": ["clause_09"]},
    {"query": "confidentiality duration after termination", "relevant": ["clause_09"]},
    {"query": "how is the annual fee paid", "relevant": ["clause_07"]},
    {"query": "twelve monthly installments fee", "relevant": ["clause_07"]},
    {"query": "what insurance must the contractor carry", "relevant": ["clause_12"]},
    {"query": "$1,000,000 liability insurance", "relevant": ["clause_12"]},
    {"query": "which state law governs", "relevant": ["clause_18"]},
    {"query": "Delaware", "relevant": ["clause_18"]},
    {"query": "where are disputes resolved", "relevant": ["clause_27"]},
    {"query": "clause 14", "relevant": ["clause_14"]},
]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate_bm25(gold, k=5):
    recalls, rrs = [], []
    for row in gold:
        relevant = set(row["relevant"])
        retrieved = [doc_id for doc_id, _ in bm25_search(row["query"], k=len(CORPUS))]
        recalls.append(recall_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))
    return {
        "queries": len(gold),
        f"Recall@{k}": sum(recalls) / len(recalls),
        "MRR": sum(rrs) / len(rrs),
    }


print(evaluate_bm25(GOLD, k=5))
```

> **The metric functions here are the SAME shape as week-7's `evaluate()`.** In the mini-project you'll import the real `evaluate()` from `crunchrag_embed` and pass it a BM25 `retrieve_fn`, so BM25 becomes one row of the lift table measured by the exact same function as every other layer. Here we inline the metrics so the exercise is self-contained.

You should see a respectable Recall@5 driven up by the keyword/identifier queries ("Delaware", "$1,000,000", "clause 14") and dragged down by the paraphrase queries ("how do I end the contract early") — the queries that need *meaning*, where BM25 is blind.

---

## Step 3 — Compare against dense, query by query

If you have your week-7 pgvector pipeline working, run the *same* 12 queries through dense retrieval and tabulate side by side. (If you don't, you can still reason about it from week-7's exercise-02 output, which showed dense nailing the paraphrase queries.)

Build a table like this — fill in dense from your own run:

| query | gold | BM25 top-1 | dense top-1 | who wins |
|---|---|---|---|---|
| how do I end the contract early | clause_14 | (often misses) | clause_14 ✓ | **dense** |
| Delaware | clause_18 | clause_18 ✓ | (fuzzy) | **BM25** |
| $1,000,000 liability insurance | clause_12 | clause_12 ✓ | (fuzzy) | **BM25** |
| clause 14 | clause_14 | clause_14 ✓ | (misses) | **BM25** |
| confidentiality duration after termination | clause_09 | clause_09 ✓ | clause_09 ✓ | both |
| what notice is required to terminate | clause_14 | clause_14 ✓ | clause_14 ✓ | both |

**The lesson, made concrete:** BM25 and dense fail on *opposite* queries. The keyword/identifier/jurisdiction/money queries go to BM25; the paraphrase queries go to dense. Neither dominates. This table *is* the argument for hybrid search — and exercise 2 fuses the two so you don't have to choose.

---

## Step 4 — Tune `k1` and `b` (see them move)

Rebuild the index at `k1=0.0` (term frequency ignored) and at `b=0.0` (length normalization off), and re-run a couple of queries:

```python
bm25_no_tf = BM25Okapi(tokenized, k1=0.0, b=0.75)   # TF ignored: present-or-not, binary
bm25_no_len = BM25Okapi(tokenized, k1=1.5, b=0.0)   # length ignored entirely
```

On this corpus the clauses are short and uniform, so `b` barely matters (length normalization has little to normalize) and `k1` barely matters (no clause repeats a term enough to saturate). **That's a real result:** the defaults are right *because* of the corpus shape. Note in one line *why* — short, uniform documents — so you understand when you *would* need to tune them (long documents of wildly varying length, where `b` stops a 50-page doc from winning everything and `k1` controls whether repetition signals relevance).

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] `bm25_baseline.py` builds a `rank-bm25` index over the corpus and runs without error.
- [ ] You show BM25 ranking `clause_18` #1 for "Delaware", `clause_12` #1 for "$1,000,000 insurance", and `clause_14` #1 for "clause 14" — the three queries dense retrieval fumbles.
- [ ] You show BM25 *missing or under-ranking* `clause_14` for "how do I end the contract early" — the paraphrase query dense nails.
- [ ] You compute a BM25 Recall@5 and MRR on the gold-set slice with metric functions matching week-7's `evaluate()` shape.
- [ ] You state, in one sentence, why `k1` and `b` barely move on this corpus (short, uniform clauses) and when they *would* matter.

---

## Stretch

- **Swap the tokenizer.** Add a stemmer (e.g. `PorterStemmer` from `nltk`) so "terminate"/"termination"/"terminating" collapse to one token, and re-run. Does Recall@5 move? (On this corpus, a little — stemming helps the morphological-variant queries.)
- **Try `bm25s`.** Reinstall the index with the faster `bm25s` library (sparse-matrix BM25) and confirm you get the *same* ranking far faster. This is the drop-in upgrade path when the corpus outgrows `rank-bm25`.
- **Postgres full-text.** If you have pgvector running, build a `tsvector` column with `to_tsvector('english', content)` and query it with `ts_rank_cd` and a `plainto_tsquery`. Confirm it ranks "Delaware" and "$1,000,000" the same way — this is the production lexical path that fuses with pgvector in one query.

---

When this feels comfortable, move to [Exercise 2 — RRF fusion](exercise-02-rrf-fusion.py).
