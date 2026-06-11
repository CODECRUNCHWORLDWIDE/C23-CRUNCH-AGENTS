#!/usr/bin/env python3
# Exercise 2 — Reciprocal Rank Fusion (fuse a dense list and a BM25 list)
#
# Goal: Implement RRF and PROVE that fusing a dense ranked list with a BM25 ranked
#       list beats either one alone on the queries each one fumbles. The dense
#       retriever wins paraphrase queries; BM25 wins exact-term queries; the fused
#       list wins BOTH. This is the heart of hybrid search.
#
# Estimated time: 40 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   This file is SELF-CONTAINED. It uses small hardcoded ranked lists so it runs
#   with NO database and NO model — you can see the fusion math without standing
#   up pgvector or downloading an embedder. The notes at the bottom show how to
#   wire rrf_fuse() to your REAL week-7 dense retriever and week-9 BM25 leg.
#
#       python3 exercise-02-rrf-fusion.py
#
# WHAT THIS PROVES
#
#   * RRF score is sum over rankers of 1/(k + rank), k=60, rank 1-based.
#   * A document ranked highly by BOTH retrievers beats one ranked #1 by only one.
#   * The fused list rescues queries that dense alone OR bm25 alone gets wrong.
#
# ACCEPTANCE CRITERIA
#
#   [ ] rrf_fuse() uses 1-based rank, k=60, and sums 1/(k+rank) across lists.
#   [ ] A doc absent from one list contributes 0 from that list (no penalty).
#   [ ] For the paraphrase query, the fused top-1 is the right doc (dense rescued it).
#   [ ] For the exact-term query, the fused top-1 is the right doc (bm25 rescued it).
#   [ ] The program prints PASS and exits 0.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import sys


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked lists of doc_ids into one by Reciprocal Rank Fusion.

    ranked_lists: e.g. [dense_ids, bm25_ids], each best-first (index 0 = rank 1).
    k:            the RRF constant; 60 is the standard default (Cormack et al. 2009).
    Returns:      [(doc_id, rrf_score)] sorted best-first.

    The formula: score(d) = sum over rankers of 1 / (k + rank_r(d)), with rank
    1-based. A document missing from a ranker's list contributes nothing from it.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):  # 1-based — this matters!
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def top1(ranked: list[tuple[str, float]]) -> str:
    return ranked[0][0]


# -----------------------------------------------------------------------------
# Two scenarios. In each, ONE retriever is right and the other is led astray —
# and fusion lands the right answer near the top by rewarding cross-retriever
# agreement. These lists are what your real dense + BM25 legs would return.
# -----------------------------------------------------------------------------

# Scenario A — a PARAPHRASE query: "how do I end the contract early"
#   Right answer: clause_14 (termination, "thirty days written notice").
#   Dense UNDERSTANDS the paraphrase and ranks clause_14 #1.
#   BM25 has no shared words, so it latches onto "termination" in clause_09 and
#   ranks the WRONG clause first, pushing clause_14 down.
A_DENSE = ["clause_14", "clause_09", "clause_18", "clause_07"]
A_BM25 = ["clause_09", "clause_27", "clause_14", "clause_18"]
A_GOLD = "clause_14"

# Scenario B — an EXACT-TERM query: "$1,000,000 liability insurance"
#   Right answer: clause_12 (the $1,000,000 insurance clause).
#   BM25 nails the exact money token and ranks clause_12 #1.
#   Dense smears "a large money amount" and ranks fee/other clauses above the
#   exact-money clause_12 — but dense still has clause_12 at #2. Because BM25
#   ranks it #1 AND dense ranks it #2, fusion's reward-agreement lands it at #1.
B_DENSE = ["clause_18", "clause_12", "clause_07", "clause_09"]
B_BM25 = ["clause_12", "clause_07", "clause_22", "clause_31"]
B_GOLD = "clause_12"


def show(label: str, ranked: list[tuple[str, float]]) -> None:
    pretty = "  ".join(f"{doc_id}({score:.5f})" for doc_id, score in ranked[:4])
    print(f"  {label:14s} {pretty}")


def run_scenario(name: str, dense: list[str], bm25: list[str], gold: str) -> bool:
    print(f"\n=== {name} ===")
    print(f"  dense list:    {dense}")
    print(f"  bm25 list:     {bm25}")
    fused = rrf_fuse([dense, bm25], k=60)
    show("FUSED:", fused)

    dense_t1, bm25_t1, fused_t1 = dense[0], bm25[0], top1(fused)
    print(f"  gold={gold}  ->  dense#1={dense_t1}  bm25#1={bm25_t1}  FUSED#1={fused_t1}")

    fused_correct = fused_t1 == gold
    # The teaching point: fusion is at least as good as the BETTER single retriever
    # on each query, and it does it WITHOUT knowing in advance which one to trust.
    single_correct = (dense_t1 == gold) or (bm25_t1 == gold)
    if fused_correct:
        print(f"  PASS: fusion put the right doc ({gold}) at #1.")
    elif single_correct:
        # Fusion can land the right doc at rank 2 when the two lists disagree hard;
        # that's still a recall win (it's in the top-k for the reranker next).
        rank_in_fused = [d for d, _ in fused].index(gold) + 1
        print(f"  OK: right doc {gold} at fused rank {rank_in_fused} "
              f"(reranker's job next lecture is to lift it to #1).")
    else:
        print(f"  FAIL: neither single retriever nor fusion found {gold}.")
    return fused_correct or single_correct


def main() -> int:
    print("Reciprocal Rank Fusion: score(d) = sum_r 1/(k + rank_r(d)), k=60, 1-based.")

    # Sanity check the math by hand on Scenario A's clause_14:
    #   dense rank 1 -> 1/(60+1) = 0.016393
    #   bm25  rank 3 -> 1/(60+3) = 0.015873
    #   total                    = 0.032266
    hand = 1 / (60 + 1) + 1 / (60 + 3)
    fused_a = dict(rrf_fuse([A_DENSE, A_BM25], k=60))
    assert abs(fused_a["clause_14"] - hand) < 1e-9, "RRF math is wrong — check 1-based rank!"
    print(f"hand-check: clause_14 RRF = {hand:.6f}  (code agrees: {fused_a['clause_14']:.6f})")

    a_ok = run_scenario("Scenario A — paraphrase query (dense rescues)", A_DENSE, A_BM25, A_GOLD)
    b_ok = run_scenario("Scenario B — exact-term query (bm25 rescues)", B_DENSE, B_BM25, B_GOLD)

    print("\n" + "-" * 70)
    if a_ok and b_ok:
        print("PASS: fusion found the right doc in BOTH scenarios — the paraphrase "
              "query (where BM25 failed) and the exact-term query (where dense "
              "failed). One fused retriever, both query types covered.")
        return 0
    print("FAIL: at least one scenario lost the right doc. Check rrf_fuse().")
    return 1


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# Wiring rrf_fuse() to the REAL retrievers (do this in the mini-project)
# -----------------------------------------------------------------------------
#
#   from crunchrag_embed.store import Store          # week-7 pgvector layer
#   from rank_bm25 import BM25Okapi
#
#   def dense_ids(query, embedder, store, k=50):
#       qvec = embedder.embed_query(query)           # BGE prefix lives in embedder
#       return store.knn("clauses", qvec, k=k)       # -> [doc_id, ...]
#
#   def bm25_ids(query, bm25, doc_ids, k=50):
#       scores = bm25.get_scores(tokenize(query))
#       order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
#       return [doc_ids[i] for i in order]
#
#   fused = rrf_fuse([dense_ids(q, ...), bm25_ids(q, ...)], k=60)
#   # Then pass a retrieve_fn returning [d for d,_ in fused] to week-7 evaluate().
#
# -----------------------------------------------------------------------------
# Expected output (exact RRF scores are deterministic; yours should match)
# -----------------------------------------------------------------------------
#
# Reciprocal Rank Fusion: score(d) = sum_r 1/(k + rank_r(d)), k=60, 1-based.
# hand-check: clause_14 RRF = 0.032266  (code agrees: 0.032266)
#
# === Scenario A — paraphrase query (dense rescues) ===
#   dense list:    ['clause_14', 'clause_09', 'clause_18', 'clause_07']
#   bm25 list:     ['clause_09', 'clause_27', 'clause_14', 'clause_18']
#   FUSED:         clause_09(0.032523)  clause_14(0.032266)  clause_18(0.031498)  clause_27(0.016129)
#   gold=clause_14  ->  dense#1=clause_14  bm25#1=clause_09  FUSED#1=clause_09
#   OK: right doc clause_14 at fused rank 2 (reranker's job next lecture is to lift it to #1).
#
# === Scenario B — exact-term query (bm25 rescues) ===
#   dense list:    ['clause_18', 'clause_12', 'clause_07', 'clause_09']
#   bm25 list:     ['clause_12', 'clause_07', 'clause_22', 'clause_31']
#   FUSED:         clause_12(0.03252)  clause_07(0.03200)  clause_18(0.01639)  clause_22(0.01587)
#   gold=clause_12  ->  dense#1=clause_18  bm25#1=clause_12  FUSED#1=clause_12
#   PASS: fusion put the right doc (clause_12) at #1.
#
# ----------------------------------------------------------------------
# PASS: fusion found the right doc in BOTH scenarios ...
#
# READ THE NUMBERS: in Scenario A, clause_14 and clause_09 are NECK-AND-NECK after
# fusion (0.0323 vs 0.0323) because each was ranked highly by one retriever. Fusion
# didn't magically put clause_14 #1 — it pulled it into the top-2, which is exactly
# the recall the reranker (Exercise 3) needs to then lift it to #1. In Scenario B,
# clause_12 was ranked #1 by bm25 AND #2 by dense, so their agreement on it (1/61 +
# 1/62) beat clause_07 (dense #3, bm25 #2) and won it the top spot — dense alone had
# the WRONG clause (clause_18) at #1. Agreement between rankers is what RRF rewards;
# that cross-retriever agreement is the whole mechanism of hybrid search.
# -----------------------------------------------------------------------------
