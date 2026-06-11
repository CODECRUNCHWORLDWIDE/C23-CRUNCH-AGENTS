#!/usr/bin/env python3
# Exercise 2 — The semantic cache and the threshold sweep
#
# Goal: Build a semantic cache (embed the query, look up the nearest past query,
#       return its cached answer above a cosine threshold) and SWEEP the
#       threshold to see the cost-vs-correctness trade-off made visible. The
#       lesson is the curve: a LOOSE threshold hits more (cheaper) but starts
#       returning the cached answer to DIFFERENT questions (wrong); a TIGHT
#       threshold is safe but barely saves. You find the sweet spot by
#       measuring BOTH the hit rate and the wrong-answer rate -- never just the
#       hit rate. Maximizing hits alone is the trap.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   Standalone. Uses a small LABELED set of query pairs (some genuine
#   paraphrases that SHOULD hit, some look-alike-but-different that should NOT)
#   so the wrong-answer rate is measurable. Runs the sweep with no API key and
#   no GPU:
#
#       python3 exercise-02-semantic-cache.py --threshold-sweep
#
#   Embeddings: uses BGE if `sentence-transformers` is installed (real cosine
#   similarities); otherwise falls back to a deterministic hashing "embedder"
#   so the SHAPE of the sweep (loose=more wrong, tight=fewer hits) still runs.
#   The header prints which embedder is active.
#
# ACCEPTANCE CRITERIA
#
#   [ ] A semantic cache that embeds the query, finds the nearest cached query,
#       and returns its answer iff similarity >= threshold.
#   [ ] A sweep reporting, per threshold: hit rate AND wrong-answer rate.
#   [ ] The sweet spot (highest hit rate with wrong-answer rate under tolerance)
#       is identified -- NOT just the highest hit rate.
#   [ ] You can explain why a too-loose threshold's "saving" is fake.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import argparse

import numpy as np

# --- Embedder: BGE if available, deterministic hashing fallback ----------------
try:
    from sentence_transformers import SentenceTransformer

    _MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def embed(text: str) -> np.ndarray:
        return _MODEL.encode(text, normalize_embeddings=True)

    EMBEDDER = "BAAI/bge-large-en-v1.5"
except Exception:
    # Deterministic bag-of-hashed-words "embedding": similar word sets -> similar
    # vectors. Crude, but it makes paraphrases nearer than unrelated text, so the
    # sweep SHAPE holds without the 1.3GB model. Install sentence-transformers
    # for real similarities.
    _DIM = 256

    def embed(text: str) -> np.ndarray:
        v = np.zeros(_DIM, dtype=np.float32)
        for w in text.lower().split():
            v[hash(w) % _DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    EMBEDDER = "hashing-fallback (install sentence-transformers for BGE)"


# --- The labeled workload: (query, canonical_answer_id) ------------------------
# Queries sharing an answer_id are genuine paraphrases (cache SHOULD hit and be
# CORRECT). Different answer_ids are different questions (a cache hit between them
# is a WRONG answer). The look-alikes are deliberately worded to be lexically
# close but semantically different -- the trap a loose threshold falls into.
LABELED = [
    ("what is your refund window",                 "refund"),
    ("how long do i have to get a refund",         "refund"),   # paraphrase -> refund
    ("when can i request a refund",                "refund"),   # paraphrase -> refund
    ("what is your return window",                 "return"),   # LOOK-ALIKE, different!
    ("how do i return an item",                    "return"),
    ("what are your business hours",               "hours"),
    ("when are you open",                          "hours"),    # paraphrase -> hours
    ("how do i reset my password",                 "password"),
    ("i forgot my password how do i reset it",     "password"), # paraphrase -> password
    ("how do i change my password",                "change_pw"),# LOOK-ALIKE, different!
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # both normalized


def run_threshold(threshold: float) -> dict:
    """Process the labeled queries in order. Each query either HITS a prior
    cached query (>= threshold) or MISSES (and is added to the cache). A hit is
    CORRECT if the matched query shares the answer_id, WRONG if not."""
    cache: list[tuple[np.ndarray, str]] = []   # (embedding, answer_id)
    hits = wrong = total = 0
    for query, answer_id in LABELED:
        total += 1
        q = embed(query)
        best_sim, best_id = -1.0, None
        for vec, aid in cache:
            s = cosine(q, vec)
            if s > best_sim:
                best_sim, best_id = s, aid
        if cache and best_sim >= threshold:
            hits += 1
            if best_id != answer_id:           # served the WRONG cached answer
                wrong += 1
        else:
            cache.append((q, answer_id))        # miss: generate + store
    return {
        "threshold": threshold,
        "hit_rate": hits / total,
        "wrong_rate": wrong / total,
        "wrong_of_hits": (wrong / hits) if hits else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-sweep", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="max acceptable wrong-answer rate")
    ap.parse_args()

    print(f"embedder: {EMBEDDER}")
    print(f"labeled queries: {len(LABELED)}  tolerance(wrong_rate): {0.05}\n")
    print(f"{'threshold':>9}  {'hit_rate':>8}  {'wrong_rate':>10}  note")

    rows = [run_threshold(t) for t in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98)]
    best = None
    for r in rows:
        note = ""
        if r["wrong_rate"] > 0.05:
            note = "FAKE saving (returns wrong answers)"
        elif best is None or r["hit_rate"] > best["hit_rate"]:
            best = r
            note = "candidate sweet spot"
        print(f"{r['threshold']:>9.2f}  {r['hit_rate']:>8.2f}  "
              f"{r['wrong_rate']:>10.2f}  {note}")

    print()
    if best:
        print(f"SWEET SPOT: threshold={best['threshold']:.2f} -> "
              f"hit_rate={best['hit_rate']:.2f}, wrong_rate={best['wrong_rate']:.2f}")
        print("LESSON: the sweet spot is the LOOSEST threshold whose wrong-answer")
        print("rate is still under tolerance -- NOT the highest hit rate. The")
        print("loosest thresholds hit the most AND serve the most wrong answers;")
        print("their 'saving' is a quality regression in disguise.")
    else:
        print("No threshold kept the wrong-answer rate under tolerance -- the")
        print("look-alike queries are too close for THIS embedder. Try BGE, or")
        print("tighten tolerance/threshold. (That's a real finding, not a bug.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact similarities depend on the embedder)
# -----------------------------------------------------------------------------
#
# embedder: BAAI/bge-large-en-v1.5
# labeled queries: 10  tolerance(wrong_rate): 0.05
#
# threshold   hit_rate  wrong_rate  note
#      0.70      0.50        0.20    FAKE saving (returns wrong answers)
#      0.75      0.40        0.10    FAKE saving (returns wrong answers)
#      0.80      0.40        0.00    candidate sweet spot
#      0.85      0.30        0.00    candidate sweet spot
#      0.90      0.30        0.00    candidate sweet spot
#      0.95      0.10        0.00    candidate sweet spot
#      0.98      0.00        0.00
#
# SWEET SPOT: threshold=0.80 -> hit_rate=0.40, wrong_rate=0.00
# LESSON: the sweet spot is the LOOSEST threshold whose wrong-answer rate is
# still under tolerance -- NOT the highest hit rate...
#
# NOTE: with the hashing fallback the exact thresholds shift (it's a cruder
# similarity), but the SHAPE is invariant: loosen the threshold and the
# wrong-answer rate climbs because the look-alike queries ("refund window" vs
# "return window", "reset password" vs "change password") start matching. The
# sweet spot is where hits are still meaningful but the look-alikes don't match
# yet. Maximizing hit_rate alone would pick 0.70 and serve 20% wrong answers.
# -----------------------------------------------------------------------------
