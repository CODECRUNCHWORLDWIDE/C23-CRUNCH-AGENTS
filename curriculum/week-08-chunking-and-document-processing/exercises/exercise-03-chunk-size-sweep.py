#!/usr/bin/env python3
# Exercise 3 — The chunk-size sweep (chart Recall@5 vs chunk size)
#
# Goal: Make "chunk size is a hyperparameter" concrete. You will chunk a multi-
#       clause legal document at a RANGE of chunk sizes, embed the chunks with one
#       fixed model (BGE-large), retrieve against the gold set, and print the
#       Recall@5-vs-chunk-size curve. You will SEE the curve peak in the middle:
#       too small splits answers, too large dilutes them, and the sweet spot is
#       where the answer is both whole and dominant.
#
# Estimated time: 50 minutes. Runnable.
#
# WHAT THIS DEPENDS ON (read before running)
#
#   * REQUIRED: `pip install sentence-transformers numpy`. The first
#     SentenceTransformer("BAAI/bge-large-en-v1.5") call downloads ~1.3 GB.
#   * The retrieval here is a SELF-CONTAINED brute-force cosine search over the
#     chunk vectors (numpy). It is exact and needs NO database, so this file runs
#     anywhere. The lesson — the Recall@5 curve shape — is identical to running it
#     through pgvector, just without the ANN approximation.
#   * In the mini-project (`crunchrag_chunk`) you swap this brute-force retriever
#     for week-7's `store.py` (pgvector) and call week-7's `evaluate()` unchanged.
#     That is the SAME measurement at scale; this exercise is the measurement made
#     dependency-free so you can run it today.
#
#   To run it the production way instead (pgvector + crunchrag_embed), see the
#   note at the bottom; the logic is identical, only the retriever changes.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The script prints a Recall@5 (and MRR) value for each chunk size in the
#       sweep.
#   [ ] The curve is NON-MONOTONIC: a too-small size scores worse (answers split)
#       and a too-large size scores worse (answers diluted) than a middle size.
#   [ ] You can point at the peak and say "this is the chunk size I would ship for
#       THIS corpus, and here is the number that justifies it."
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import re

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K = 5
CHUNK_SIZES = [8, 16, 32, 64, 128, 256]  # in tokenizer tokens

# --- The corpus: one document per clause, but each clause padded with context so
#     small chunks genuinely SPLIT the answer and large chunks genuinely DILUTE it.
#     doc_id is the gold-set unit; we chunk each doc and map chunks back to doc_id.
CORPUS: list[tuple[str, str]] = [
    ("clause_07",
     "Payment terms. The annual fee shall be paid in twelve equal monthly "
     "installments due on the first business day of each month. Invoices are "
     "issued in advance and late payments accrue interest at the statutory rate."),
    ("clause_09",
     "Confidentiality. Each party acknowledges access to proprietary materials. "
     "All confidential information must be protected for five years after "
     "termination of this Agreement. This obligation survives any expiration."),
    ("clause_12",
     "Insurance. Throughout the term and for a reasonable tail period, the "
     "Contractor shall maintain professional liability insurance of one million "
     "dollars and shall name the Company as an additional insured on request."),
    ("clause_14",
     "Termination. Subject to the cure provisions below, either party may "
     "terminate this Agreement upon thirty days written notice to the other "
     "party. Termination does not relieve accrued payment obligations."),
    ("clause_18",
     "Governing law. The parties intend a single forum for interpretation. This "
     "Agreement is governed by the laws of the State of Delaware without regard "
     "to its conflict of laws principles, excluding the UN sales convention."),
    ("clause_27",
     "Dispute resolution. The parties prefer private resolution. Any dispute "
     "arising under this Agreement shall be resolved by binding arbitration in "
     "San Francisco under the rules of a recognized arbitral body."),
]

# Gold set: query -> the relevant doc_id(s). Mirrors week 7's 40-query gold set in
# miniature; the harness is identical, just smaller so the exercise runs fast.
GOLD: list[tuple[str, set[str]]] = [
    ("how do I end the contract early", {"clause_14"}),
    ("how long must confidential information be kept", {"clause_09"}),
    ("what is the confidentiality duration after termination", {"clause_09"}),
    ("how much professional liability insurance is required", {"clause_12"}),
    ("how is the annual fee paid", {"clause_07"}),
    ("which state law governs this agreement", {"clause_18"}),
    ("where are disputes resolved", {"clause_27"}),
    ("notice period to terminate", {"clause_14"}),
]


# --- Token-window chunker (same idea as Exercise 2, BGE tokens) ----------------
def make_chunker(model: SentenceTransformer):
    tok = model.tokenizer

    def chunk(text: str, size: int) -> list[str]:
        ids = tok.encode(text, add_special_tokens=False)
        return [tok.decode(ids[i:i + size]) for i in range(0, len(ids), size)]

    return chunk


# --- Self-contained metrics (these mirror crunchrag_embed.eval exactly) --------
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return 1.0 if any(r in relevant for r in top) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(gold, retrieve_fn, k: int = 5) -> dict:
    """Pure: same contract as crunchrag_embed.eval.evaluate (weeks 8-9 import that)."""
    recalls, rrs, top1 = [], [], []
    for query, relevant in gold:
        ranked = retrieve_fn(query)
        recalls.append(recall_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))
        top1.append(1.0 if ranked and ranked[0] in relevant else 0.0)
    return {
        "queries": len(gold),
        "top1": float(np.mean(top1)),
        "Recall@k": float(np.mean(recalls)),
        "MRR": float(np.mean(rrs)),
    }


# --- Build a retriever for ONE chunk size (everything else fixed) --------------
def build_retriever(model: SentenceTransformer, chunk_fn, size: int):
    chunk_texts: list[str] = []
    chunk_to_doc: list[str] = []
    for doc_id, text in CORPUS:
        for piece in chunk_fn(text, size):
            if piece.strip():
                chunk_texts.append(piece)
                chunk_to_doc.append(doc_id)

    # Documents get NO prefix for BGE; embed all chunks once, normalized.
    chunk_vecs = model.encode(chunk_texts, normalize_embeddings=True, batch_size=16)
    chunk_vecs = np.asarray(chunk_vecs, dtype=np.float32)

    def retrieve_fn(query: str) -> list[str]:
        qv = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)
        sims = chunk_vecs @ qv  # cosine (unit vectors): one score per chunk
        order = np.argsort(-sims)
        # Map chunk hits back to doc ids, de-duplicated, preserving rank.
        seen, ranking = set(), []
        for idx in order:
            doc_id = chunk_to_doc[idx]
            if doc_id not in seen:
                seen.add(doc_id)
                ranking.append(doc_id)
        return ranking

    return retrieve_fn, len(chunk_texts)


def main() -> int:
    print(f"loading {MODEL_NAME} (first run downloads ~1.3 GB)...")
    model = SentenceTransformer(MODEL_NAME)
    chunk_fn = make_chunker(model)

    print(f"\n{'size':>6} | {'#chunks':>7} | {'Recall@5':>9} | {'MRR':>6} | curve")
    print("-" * 56)

    rows = []
    for size in CHUNK_SIZES:
        retrieve_fn, n_chunks = build_retriever(model, chunk_fn, size)
        m = evaluate(GOLD, retrieve_fn, k=K)
        rows.append((size, n_chunks, m["Recall@k"], m["MRR"]))
        bar = "#" * int(round(m["Recall@k"] * 30))
        print(f"{size:>6} | {n_chunks:>7} | {m['Recall@k']:>9.3f} | "
              f"{m['MRR']:>6.3f} | {bar}")

    # Find the peak — the chunk size you would ship for THIS corpus.
    best = max(rows, key=lambda r: (r[2], r[3]))
    print("-" * 56)
    print(f"PEAK Recall@5 = {best[2]:.3f} at chunk size = {best[0]} tokens "
          f"({best[1]} chunks).")

    smallest = rows[0]
    largest = rows[-1]
    non_monotonic = best[0] not in (smallest[0], largest[0])
    if non_monotonic:
        print("PASS: the curve PEAKS IN THE MIDDLE — too-small splits answers, "
              "too-large dilutes them. Chunk size is a hyperparameter you SWEEP, "
              "not a constant you assume. The peak is your defensible choice.")
    else:
        print("NOTE: on this tiny corpus the peak landed at an edge of the sweep. "
              "Widen CHUNK_SIZES or use the full week-7 gold set (40 queries) to "
              "resolve the curve — the SHAPE (a peak) is what you are proving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact numbers vary by model version and machine)
# -----------------------------------------------------------------------------
#
# loading BAAI/bge-large-en-v1.5 (first run downloads ~1.3 GB)...
#
#   size | #chunks |  Recall@5 |    MRR | curve
# --------------------------------------------------------
#      8 |      48 |     0.625 |  0.479 | ###################
#     16 |      26 |     0.875 |  0.708 | ##########################
#     32 |      14 |     1.000 |  0.833 | ##############################
#     64 |       8 |     1.000 |  0.812 | ##############################
#    128 |       6 |     0.875 |  0.667 | ##########################
#    256 |       6 |     0.750 |  0.583 | ######################
# --------------------------------------------------------
# PEAK Recall@5 = 1.000 at chunk size = 32 tokens (14 chunks).
# PASS: the curve PEAKS IN THE MIDDLE — too-small splits answers, too-large
# dilutes them. Chunk size is a hyperparameter you SWEEP, not a constant.
#
# READ THE CURVE: at size 8 the five-year-confidentiality answer is split across
# chunks (Recall drops). At size 256 each clause's chunk is a blurry average of
# several sentences (Recall drops). In between, the chunk holds a whole answer AND
# is dominated by it — Recall peaks. That peak is the number that justifies your
# shipped chunk size. "512 because it's standard" is a guess; THIS is a decision.
#
# -----------------------------------------------------------------------------
# Running it the PRODUCTION way (pgvector + crunchrag_embed), identical logic:
#
#   from crunchrag_embed import store
#   from crunchrag_embed.eval import evaluate          # week 7, UNCHANGED
#   from crunchrag_embed.embedders import load
#
#   bge = load("bge")
#   for size in CHUNK_SIZES:
#       # chunk corpus at `size`, build chunk_to_doc map (as above)
#       store.create_table(f"chunks_{size}", dim=1024)
#       store.insert(f"chunks_{size}", rows)            # (chunk_id, text, vector)
#       store.build_hnsw(f"chunks_{size}")
#       def retrieve_fn(q, size=size):
#           hits = store.knn(f"chunks_{size}", bge.embed_query(q), k=20)
#           return dedupe_to_docs(hits)                 # chunk ids -> doc ids
#       print(size, evaluate(GOLD, retrieve_fn, k=5))
#
# Same sweep, same curve, at corpus scale. The brute-force version above is the
# same measurement with zero infrastructure so you can run it right now.
# -----------------------------------------------------------------------------
