#!/usr/bin/env python3
# Exercise 3 — Reranker lift (a cross-encoder fixes a wrong first-stage order)
#
# Goal: Take a candidate set where the RIGHT document is at rank 4 (a realistic
#       first-stage result — it's in the top-k, but not on top), score it with the
#       BAAI/bge-reranker-v2-m3 cross-encoder, and watch the reranked order pull
#       the right document to rank 1. This is "the cheapest meaningful win in RAG."
#
# Estimated time: 45 minutes. Runnable.
#
# MODEL DOWNLOAD
#
#   The first run downloads BAAI/bge-reranker-v2-m3 (~600 MB) from Hugging Face and
#   caches it under ~/.cache/huggingface. Do this on good wifi; subsequent runs are
#   fast. It runs on CPU (slower than GPU, but we rerank only 5 candidates, so it's
#   well under a second).
#
# HOW TO USE THIS FILE
#
#   pip install sentence-transformers     # you have this from week 7
#   python3 exercise-03-reranker-lift.py
#
# WHAT THIS PROVES
#
#   * A cross-encoder scores (query, passage) PAIRS jointly — applied only to the
#     small first-stage candidate set (here, 5), never the whole corpus.
#   * The reranker reads the query and each passage together, so it can tell the
#     truly-relevant clause from a topically-similar distractor.
#   * The lift shows up in MRR (rank of the right doc), not just Recall@5.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The first-stage list has the right doc (clause_09) at rank 4.
#   [ ] After reranking with bge-reranker-v2-m3, clause_09 is at rank 1.
#   [ ] The reciprocal rank improves from 1/4 = 0.25 to 1.0 (a big MRR jump).
#   [ ] The program prints PASS and exits 0.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import sys


# The query and a first-stage candidate set. clause_09 (the confidentiality clause
# that actually answers the query) is at RANK 4: it's in the top-k, but dense/hybrid
# put three topically-adjacent distractors above it (they all share "termination" /
# "Agreement" / contract vocabulary). The reranker's job is to reorder this.
QUERY = "how many years must confidential information stay protected after the contract ends?"

# (doc_id, passage) — order is the FIRST-STAGE ranking (e.g. hybrid+RRF top-5).
FIRST_STAGE = [
    ("clause_14", "Either party may terminate this Agreement upon thirty days written notice."),
    ("clause_18", "This Agreement is governed by the laws of the State of Delaware."),
    ("clause_03", "The initial term of this Agreement is two years from the effective date."),
    ("clause_09", "All confidential information must be protected for five years after termination."),
    ("clause_07", "The annual fee shall be paid in twelve equal monthly installments."),
]
GOLD = "clause_09"


def reciprocal_rank(ranked_ids: list[str], relevant: str) -> float:
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == relevant:
            return 1.0 / i
    return 0.0


def main() -> int:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("FAIL: sentence-transformers not installed. "
              "Run: pip install sentence-transformers")
        return 1

    print("loading BAAI/bge-reranker-v2-m3 (first run downloads ~600 MB)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    first_stage_ids = [doc_id for doc_id, _ in FIRST_STAGE]
    rr_before = reciprocal_rank(first_stage_ids, GOLD)
    rank_before = first_stage_ids.index(GOLD) + 1

    print(f"\nquery: {QUERY!r}\n")
    print("FIRST-STAGE order (e.g. hybrid + RRF):")
    for rank, (doc_id, text) in enumerate(FIRST_STAGE, start=1):
        marker = "  <-- the right answer, buried at rank 4" if doc_id == GOLD else ""
        print(f"  #{rank}  {doc_id}  {text[:55]}{marker}")

    # Score (query, passage) PAIRS jointly. This is the cross-encoder forward pass —
    # one per candidate, which is why we only ever run it on the first-stage top-k.
    pairs = [(QUERY, text) for _, text in FIRST_STAGE]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(FIRST_STAGE, scores), key=lambda x: -x[1])
    reranked_ids = [doc_id for (doc_id, _), _ in reranked]
    rr_after = reciprocal_rank(reranked_ids, GOLD)
    rank_after = reranked_ids.index(GOLD) + 1

    print("\nRERANKED order (BAAI/bge-reranker-v2-m3):")
    for rank, ((doc_id, text), score) in enumerate(reranked, start=1):
        marker = "  <-- the reranker lifted it to #1" if (doc_id == GOLD and rank == 1) else ""
        print(f"  #{rank}  {doc_id}  rerank={score:+.3f}  {text[:45]}{marker}")

    print(f"\nrank of {GOLD}:  before={rank_before}  after={rank_after}")
    print(f"reciprocal rank:  before={rr_before:.3f}  after={rr_after:.3f}  "
          f"(MRR contribution {rr_after - rr_before:+.3f})")

    if rank_after == 1:
        print("\nPASS: the cross-encoder read the query and each passage together, "
              "recognized that clause_09 literally states the five-year protection "
              "period, and pulled it from rank 4 to rank 1. That movement is the "
              "lift — and it shows up in MRR, the rank of the right answer.")
        return 0
    print(f"\nFAIL: expected {GOLD} at rank 1 after reranking, got rank {rank_after}. "
          "Check the model loaded correctly.")
    return 1


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# Wiring this into the pipeline (the mini-project does exactly this)
# -----------------------------------------------------------------------------
#
#   1. First stage retrieves a generous top-k (e.g. 50) with hybrid + RRF.
#   2. Rerank ONLY those 50 with the cross-encoder (never the whole corpus —
#      the cost is one forward pass per candidate).
#   3. Keep the reranked top-5 and hand them to the LLM / evaluate().
#
#   def rerank(query, candidates, reranker, top_k=5):
#       pairs = [(query, text) for _, text in candidates]
#       scores = reranker.predict(pairs)
#       ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
#       return [doc_id for (doc_id, _), _ in ranked[:top_k]]
#
# -----------------------------------------------------------------------------
# Expected output (SHAPE; exact reranker logits vary by model version / machine,
# but the ORDER — clause_09 at #1 after reranking — is what's invariant)
# -----------------------------------------------------------------------------
#
# loading BAAI/bge-reranker-v2-m3 (first run downloads ~600 MB)...
#
# query: 'how many years must confidential information stay protected after the contract ends?'
#
# FIRST-STAGE order (e.g. hybrid + RRF):
#   #1  clause_14  Either party may terminate this Agreement upon th...
#   #2  clause_18  This Agreement is governed by the laws of the Stat...
#   #3  clause_03  The initial term of this Agreement is two years fr...
#   #4  clause_09  All confidential information must be protected for...  <-- the right answer, buried at rank 4
#   #5  clause_07  The annual fee shall be paid in twelve equal month...
#
# RERANKED order (BAAI/bge-reranker-v2-m3):
#   #1  clause_09  rerank=+6.almost  All confidential information must be ...  <-- the reranker lifted it to #1
#   #2  clause_14  rerank=-3.xxx  Either party may terminate this Agreem...
#   #3  clause_03  rerank=-5.xxx  The initial term of this Agreement is ...
#   #4  clause_18  rerank=-7.xxx  This Agreement is governed by the laws...
#   #5  clause_07  rerank=-9.xxx  The annual fee shall be paid in twelve...
#
# rank of clause_09:  before=4  after=1
# reciprocal rank:  before=0.250  after=1.000  (MRR contribution +0.750)
#
# PASS: the cross-encoder read the query and each passage together ...
#
# THE LESSON: first-stage retrieval got clause_09 into the top-5 (good recall) but
# not to the top (mediocre precision). Recall@5 was already 1.0 — it can't improve.
# What MOVED is MRR: 0.25 -> 1.0, because the reranker fixed the ORDER. That is why
# "a reranker is the cheapest meaningful win in RAG" — one model, ~5 forward passes,
# and the right answer is now where the LLM reads most carefully: the top.
# -----------------------------------------------------------------------------
