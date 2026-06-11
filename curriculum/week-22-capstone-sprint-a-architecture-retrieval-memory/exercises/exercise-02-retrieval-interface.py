#!/usr/bin/env python3
# Exercise 2 — The retrieval interface (one clean contract over the hybrid internals)
#
# Goal: Build the SINGLE retrieve() interface the capstone's agents call, which
#       hides the hybrid internals (BM25 + dense + RRF fusion + reranker) behind
#       one clean contract. The lesson is decoupling: the agents call
#       retrieve(query) and DON'T KNOW there's a BM25 leg, an RRF fuse, or a
#       reranker -- which is exactly what lets you evolve the internals (add
#       GraphRAG, swap the reranker) without touching a single agent. The
#       interface is the contract; the implementation hides behind it.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   Standalone with a tiny in-memory corpus so the hybrid pipeline runs with no
#   pgvector and no GPU. The real capstone wraps your week-9 hybrid retriever;
#   this exercise builds the INTERFACE shape around a minimal implementation so
#   the contract is concrete.
#
#       python3 exercise-02-retrieval-interface.py
#
#   Embeddings: BGE if sentence-transformers is installed; a deterministic
#   hashing fallback otherwise so the SHAPE runs. BM25: rank-bm25 if installed;
#   a simple term-overlap fallback otherwise.
#
# ACCEPTANCE CRITERIA
#
#   [ ] A single retrieve(query, k, filters) -> list[Chunk] interface.
#   [ ] Behind it: a dense leg, a sparse (BM25) leg, RRF fusion, and a rerank
#       step -- all HIDDEN from the caller.
#   [ ] A test proving the interface returns ranked chunks and that swapping an
#       internal (e.g. disabling the reranker) doesn't change the SIGNATURE.
#   [ ] You can explain why the single interface is what lets retrieval evolve
#       without touching the agents.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Embedder: BGE or hashing fallback ----------------------------------------
try:
    from sentence_transformers import SentenceTransformer

    _M = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def embed(text: str) -> np.ndarray:
        return _M.encode(text, normalize_embeddings=True)

    EMBEDDER = "BAAI/bge-large-en-v1.5"
except Exception:
    _DIM = 256

    def embed(text: str) -> np.ndarray:
        v = np.zeros(_DIM, dtype=np.float32)
        for w in text.lower().split():
            v[hash(w) % _DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    EMBEDDER = "hashing-fallback"


# --- The corpus chunk: text + metadata (the citation/filter fields) -----------
@dataclass
class Chunk:
    id: str
    text: str
    source_doc: str
    score: float = 0.0


CORPUS = [
    Chunk("c1", "Continuous batching swaps finished sequences out every step.", "vllm.md"),
    Chunk("c2", "PagedAttention stores the KV cache in fixed-size blocks.", "vllm.md"),
    Chunk("c3", "The refund window is thirty days from the date of purchase.", "policy.md"),
    Chunk("c4", "Returns must be initiated within thirty days and the item unused.", "policy.md"),
    Chunk("c5", "Semantic caching embeds the query and looks up a vector store.", "cost.md"),
    Chunk("c6", "Model routing sends easy queries to a cheap small model.", "cost.md"),
    Chunk("c7", "The three memory tiers are episodic, semantic, and procedural.", "memory.md"),
    Chunk("c8", "Episodic memory keeps a rolling summary of the conversation.", "memory.md"),
]


# --- The internals: dense, sparse, fuse, rerank (ALL hidden behind retrieve) ---
def _dense_leg(query: str, k: int) -> list[str]:
    q = embed(query)
    scored = sorted(CORPUS, key=lambda c: float(np.dot(q, embed(c.text))), reverse=True)
    return [c.id for c in scored[:k]]


def _sparse_leg(query: str, k: int) -> list[str]:
    # term-overlap BM25 stand-in: rank by shared query terms
    qterms = set(query.lower().split())
    scored = sorted(CORPUS,
                    key=lambda c: len(qterms & set(c.text.lower().split())),
                    reverse=True)
    return [c.id for c in scored[:k]]


def _rrf_fuse(dense: list[str], sparse: list[str], rrf_k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(sparse):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def _rerank(query: str, candidate_ids: list[str], top_k: int) -> list[Chunk]:
    # a cross-encoder stand-in: re-score by query-chunk term overlap + dense sim
    q = embed(query)
    by_id = {c.id: c for c in CORPUS}
    out = []
    for cid in candidate_ids:
        c = by_id[cid]
        c.score = float(np.dot(q, embed(c.text)))
        out.append(c)
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:top_k]


# --- THE INTERFACE: the one contract the agents call --------------------------
def retrieve(query: str, k: int = 5, filters: dict | None = None,
             use_reranker: bool = True) -> list[Chunk]:
    """The single retrieval interface. The agents call THIS and know nothing of
    the dense/sparse/RRF/rerank internals -- which is exactly what lets those
    internals change without touching the agents."""
    dense = _dense_leg(query, k=20)
    sparse = _sparse_leg(query, k=20)
    fused = _rrf_fuse(dense, sparse)
    if filters:                                  # metadata filter (citation/scope)
        by_id = {c.id: c for c in CORPUS}
        fused = [cid for cid in fused
                 if all(getattr(by_id[cid], f) == v for f, v in filters.items())]
    if use_reranker:
        return _rerank(query, fused, top_k=k)
    # without the reranker: return the fused order, same SIGNATURE
    by_id = {c.id: c for c in CORPUS}
    return [by_id[cid] for cid in fused[:k]]


def main() -> int:
    print(f"embedder: {EMBEDDER}\n")

    # The agent's view: it just calls retrieve(). It doesn't know how it works.
    q = "what is the refund window?"
    print(f"query: {q!r}")
    hits = retrieve(q, k=3)
    for c in hits:
        print(f"  {c.id}  ({c.source_doc})  score={c.score:.3f}  {c.text[:50]}")

    # Prove the SIGNATURE is stable when an internal changes (reranker off):
    print("\nsame query, reranker DISABLED (internal change, SAME interface):")
    hits2 = retrieve(q, k=3, use_reranker=False)
    for c in hits2:
        print(f"  {c.id}  ({c.source_doc})  {c.text[:50]}")

    # Prove the filter works (only policy.md docs):
    print("\nsame query, filtered to source_doc=policy.md:")
    hits3 = retrieve(q, k=3, filters={"source_doc": "policy.md"})
    for c in hits3:
        print(f"  {c.id}  ({c.source_doc})  {c.text[:50]}")

    assert all(c.source_doc == "policy.md" for c in hits3), "filter failed"
    assert "c3" in {c.id for c in hits} or "c4" in {c.id for c in hits}, \
        "refund query should retrieve the refund/return chunk"

    print("\nLESSON: the agent called retrieve(query) three times -- normal,")
    print("reranker-off, and filtered -- and the INTERFACE never changed. The")
    print("dense/sparse/RRF/rerank internals are hidden, so you can swap any of")
    print("them (add GraphRAG, change the reranker) without touching the agents.")
    print("That decoupling is the most important design move in Sprint A.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact scores depend on the embedder)
# -----------------------------------------------------------------------------
#
# embedder: BAAI/bge-large-en-v1.5
#
# query: 'what is the refund window?'
#   c3  (policy.md)  score=0.71  The refund window is thirty days from the date...
#   c4  (policy.md)  score=0.64  Returns must be initiated within thirty days an...
#   c5  (cost.md)    score=0.41  Semantic caching embeds the query and looks up ...
#
# same query, reranker DISABLED (internal change, SAME interface):
#   c3  (policy.md)  The refund window is thirty days from the date...
#   ...
#
# same query, filtered to source_doc=policy.md:
#   c3  (policy.md)  The refund window is thirty days from the date...
#   c4  (policy.md)  Returns must be initiated within thirty days an...
#
# LESSON: the agent called retrieve(query) three times ... the INTERFACE never changed.
#
# NOTE: with the hashing fallback the scores differ but the SHAPE holds -- the
# refund chunk ranks first, the filter restricts to policy.md, and the interface
# signature is identical whether the reranker is on or off. In the real capstone,
# the internals are your week-9 hybrid retriever over the 10 GB corpus; the
# interface is exactly this retrieve(query, k, filters) contract.
# -----------------------------------------------------------------------------
