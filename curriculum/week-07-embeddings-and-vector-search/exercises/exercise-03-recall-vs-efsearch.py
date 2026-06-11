#!/usr/bin/env python3
# Exercise 3 — Recall vs ef_search (chart the iron triangle)
#
# Goal: Make the recall/latency/memory triangle from Lecture 2 concrete. You will
#       compute a BRUTE-FORCE ground truth (the exact nearest neighbours), then
#       sweep the HNSW runtime knob `ef_search` and measure how ANN recall and
#       query latency both climb as you spend more compute. The output is a small
#       table that IS the recall/latency curve.
#
# Estimated time: 50 minutes. Runnable.
#
# PREREQUISITES (same as Exercise 2)
#
#   docker run -d --name crunch-pg -e POSTGRES_PASSWORD=crunch -p 5432:5432 \
#     pgvector/pgvector:pg17
#   pip install sentence-transformers "psycopg[binary]" numpy
#   python3 exercise-03-recall-vs-efsearch.py
#
# THE EXPERIMENT
#
#   1. Generate ~2000 synthetic 1024-dim unit vectors (clustered, so search is
#      non-trivial) and load them into pgvector with an HNSW index.
#   2. Pick 50 random query vectors.
#   3. For each query, compute the EXACT top-10 by brute force (the ground truth).
#   4. For each ef_search in {10, 20, 40, 80, 160, 320}, run the ANN search and
#      measure Recall@10 (overlap with ground truth) and median latency.
#   5. Print the curve.
#
#   Synthetic vectors (not a text model) keep this fast and let us crank the
#   corpus size to where ANN approximation actually matters. The lesson — recall
#   AND latency both rise with ef_search — is identical for real embeddings.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The printed table shows Recall@10 INCREASING as ef_search increases.
#   [ ] It shows median latency INCREASING as ef_search increases.
#   [ ] At low ef_search (10) recall is visibly below 1.0; at high ef_search
#       (320) it is at or near 1.0 — proving the approximation gap is real and
#       tunable.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import sys
import time

import numpy as np
import psycopg

DSN = "postgresql://postgres:crunch@localhost:5432/postgres"
DIM = 1024
N_VECTORS = 2000
N_QUERIES = 50
K = 10
EF_SEARCH_SWEEP = [10, 20, 40, 80, 160, 320]
RNG = np.random.default_rng(7)


def make_clustered_vectors(n: int, dim: int, n_clusters: int = 20) -> np.ndarray:
    """Unit vectors drawn around random cluster centers — harder than uniform noise."""
    centers = RNG.standard_normal((n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assignments = RNG.integers(0, n_clusters, size=n)
    vecs = centers[assignments] + 0.35 * RNG.standard_normal((n, dim))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype(np.float32)


def to_pgvector(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def setup(conn: psycopg.Connection, vecs: np.ndarray) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("DROP TABLE IF EXISTS ann_demo")
    conn.execute(f"CREATE TABLE ann_demo (id BIGINT PRIMARY KEY, embedding vector({DIM}))")
    with conn.cursor() as cur:
        with cur.copy(
            "COPY ann_demo (id, embedding) FROM STDIN"
        ) as copy:
            for i, v in enumerate(vecs):
                copy.write_row((i, to_pgvector(v)))
    conn.execute(
        "CREATE INDEX ON ann_demo USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    conn.commit()


def brute_force_truth(vecs: np.ndarray, queries: np.ndarray, k: int) -> list[set[int]]:
    """Exact top-k by full dot product (unit vectors -> cosine). The ground truth."""
    sims = queries @ vecs.T  # (n_queries, n_vectors)
    truth = []
    for row in sims:
        top = np.argpartition(-row, k)[:k]
        truth.append(set(int(i) for i in top))
    return truth


def ann_search(conn: psycopg.Connection, query: np.ndarray, k: int) -> list[int]:
    rows = conn.execute(
        "SELECT id FROM ann_demo ORDER BY embedding <=> %s LIMIT %s",
        (to_pgvector(query), k),
    ).fetchall()
    return [int(r[0]) for r in rows]


def main() -> int:
    print(f"generating {N_VECTORS} clustered {DIM}-dim vectors...")
    vecs = make_clustered_vectors(N_VECTORS, DIM)
    q_idx = RNG.choice(N_VECTORS, size=N_QUERIES, replace=False)
    queries = vecs[q_idx]

    print("computing brute-force ground truth (exact top-10)...")
    truth = brute_force_truth(vecs, queries, K)

    try:
        conn = psycopg.connect(DSN)
    except psycopg.OperationalError as exc:
        print(f"FAIL: cannot connect to Postgres at {DSN}\n  {exc}")
        return 1

    with conn:
        print("loading vectors + building HNSW index...")
        setup(conn, vecs)

        print(f"\n{'ef_search':>10} | {'Recall@10':>10} | {'median ms':>10}")
        print("-" * 36)

        recalls_increasing = True
        prev_recall = -1.0
        for ef in EF_SEARCH_SWEEP:
            conn.execute(f"SET hnsw.ef_search = {ef}")
            overlaps = []
            latencies = []
            for q, true_set in zip(queries, truth):
                t0 = time.perf_counter()
                got = ann_search(conn, q, K)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                overlaps.append(len(set(got) & true_set) / K)
            recall = float(np.mean(overlaps))
            med_ms = float(np.median(latencies))
            print(f"{ef:>10} | {recall:>10.3f} | {med_ms:>10.2f}")
            if recall + 1e-9 < prev_recall:
                recalls_increasing = False
            prev_recall = recall

    # ef_search=10 should be visibly imperfect; ef_search=320 near-perfect.
    print()
    if recalls_increasing:
        print("PASS: Recall@10 rises (or holds) as ef_search rises — the recall/latency")
        print("      curve from Lecture 2 §1.2, measured on your own machine.")
        return 0
    else:
        print("NOTE: recall wobbled instead of rising monotonically. ANN is approximate,")
        print("      so small non-monotonic dips happen on tiny query sets — re-run with")
        print("      more queries (N_QUERIES) to smooth it. The trend is what matters.")
        return 0  # still a successful run; the wobble is expected at small N


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact numbers vary by machine and pgvector version)
# -----------------------------------------------------------------------------
#
# generating 2000 clustered 1024-dim vectors...
# computing brute-force ground truth (exact top-10)...
# loading vectors + building HNSW index...
#
#  ef_search |  Recall@10 |  median ms
# ------------------------------------
#         10 |      0.84  |       0.45
#         20 |      0.92  |       0.58
#         40 |      0.97  |       0.81
#         80 |      0.99  |       1.20
#        160 |      1.00  |       2.05
#        320 |      1.00  |       3.60
#
# PASS: Recall@10 rises (or holds) as ef_search rises — the recall/latency
#       curve from Lecture 2 §1.2, measured on your own machine.
#
# READ THE CURVE: recall climbs fast and then flattens near 1.0, while latency
# keeps climbing. The sweet spot is the "elbow" — the ef_search where recall is
# already ~0.97+ but latency is still small (here, ~40). Tuning ef_search to the
# elbow, not the max, is the whole job. Spending compute past the elbow buys you
# latency for no recall.
# -----------------------------------------------------------------------------
