#!/usr/bin/env python3
# Exercise 2 — Filtered ANN: pre-filter vs post-filter vs native, and the recall
#              collapse that picks your store
#
# Goal: Make Lecture 1 §3 concrete. You will run the SAME filtered query three
#       ways — POST-filter (ANN over everything, then drop non-matches), PRE-filter
#       (filter first, then exact search within the subset), and NATIVE/in-filter
#       (the production answer the real stores implement) — and SEE post-filter's
#       recall COLLAPSE on a SELECTIVE filter while pre/native hold. That collapse
#       is a SILENT production bug (no error, just too few results), and catching
#       it is what picks Qdrant-style native filtering over a naive post-filter.
#
# Estimated time: 50 minutes. Runnable.
#
# WHAT THIS DEPENDS ON (read before running)
#
#   * REQUIRED: `pip install numpy`. That's it. This exercise is a SELF-CONTAINED
#     simulation of the three filtering strategies over synthetic vectors with a
#     `tenant` metadata field. NO database, NO model download — it runs anywhere,
#     today, so the filtered-ANN lesson is exercisable before you stand up Qdrant.
#   * The "ANN" here is modeled honestly: a post-filter only sees the global top-N
#     candidates (as a real ANN index would surface), so when the matching tenant
#     is rare, its vectors fall OUTSIDE that candidate set and post-filter returns
#     too few — exactly the recall collapse. PRE/NATIVE search within the matching
#     subset, so they don't collapse. Run it against real Qdrant (post_filter via a
#     big `limit` then drop; native via `query_filter`) and you'll see the same.
#
# ACCEPTANCE CRITERIA
#
#   [ ] For a BROAD filter (most vectors match), all three strategies have similar
#       recall — post-filter is fine here.
#   [ ] For a SELECTIVE filter (the matching tenant is rare), POST-filter recall
#       COLLAPSES (it returns too few matching results) while PRE/NATIVE hold.
#   [ ] You can state why: post-filter only sees the global ANN candidates, and a
#       rare tenant's vectors aren't in them. This is the silent bug native
#       filtering exists to prevent.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(42)
DIM = 64
N = 5000                 # total vectors in the store
ANN_CANDIDATES = 100     # a real ANN index surfaces ~top-N globally; post-filter
                         # only gets to filter WITHIN these. This is the crux.
K = 10                   # results we want back


def make_corpus(rare_tenant_frac: float):
    """N vectors, each tagged with a tenant. `rare_tenant_frac` of them belong to
    tenant 'rare'; the rest to 'common'. We'll query for 'rare' and watch what each
    strategy does as 'rare' gets rarer (the filter gets more selective)."""
    vecs = RNG.normal(size=(N, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    n_rare = max(1, int(N * rare_tenant_frac))
    tenants = np.array(["common"] * N, dtype=object)
    rare_idx = RNG.choice(N, size=n_rare, replace=False)
    tenants[rare_idx] = "rare"
    return vecs, tenants, set(rare_idx.tolist())


def cosine_topk(vecs, qv, k, candidate_mask=None):
    """Exact cosine top-k, optionally restricted to a candidate mask (the subset a
    strategy is allowed to consider)."""
    sims = vecs @ qv
    if candidate_mask is not None:
        sims = np.where(candidate_mask, sims, -np.inf)
    order = np.argsort(-sims)
    return [i for i in order[:k] if sims[i] > -np.inf]


def post_filter(vecs, tenants, qv, want_tenant, k):
    """ANN over EVERYTHING (top ANN_CANDIDATES globally), THEN drop non-matches.
    This is what a naive post-filter does — and why it collapses on rare tenants."""
    candidates = cosine_topk(vecs, qv, ANN_CANDIDATES)        # global ANN top-N
    matching = [i for i in candidates if tenants[i] == want_tenant]  # then filter
    return matching[:k]


def pre_filter(vecs, tenants, qv, want_tenant, k):
    """Filter FIRST (only the matching tenant's vectors), then exact search within.
    Exact wrt the filter; can be slow on a broad filter (big subset to scan)."""
    mask = np.array([t == want_tenant for t in tenants])
    return cosine_topk(vecs, qv, k, candidate_mask=mask)


def native_filter(vecs, tenants, qv, want_tenant, k):
    """Filter DURING traversal — modeled here as exact search within the matching
    subset (what Qdrant's filterable HNSW achieves at index speed). Gets the
    filter's exactness AND the index's reach, so it doesn't collapse."""
    mask = np.array([t == want_tenant for t in tenants])
    return cosine_topk(vecs, qv, k, candidate_mask=mask)


def recall_at_k(retrieved, relevant_topk, k):
    """Fraction of the TRUE top-k (within the tenant) that the strategy returned."""
    if not relevant_topk:
        return 1.0
    return len(set(retrieved[:k]) & set(relevant_topk)) / len(relevant_topk[:k])


def run(rare_frac: float):
    vecs, tenants, rare_set = make_corpus(rare_frac)
    qv = vecs[next(iter(rare_set))] * 0.6 + RNG.normal(size=DIM) * 0.4  # near a rare vec
    qv = (qv / np.linalg.norm(qv)).astype(np.float32)

    # Ground truth: the true top-k AMONG the rare tenant's vectors.
    truth = native_filter(vecs, tenants, qv, "rare", K)

    strategies = {
        "post-filter": post_filter,
        "pre-filter": pre_filter,
        "native": native_filter,
    }
    n_rare = sum(1 for t in tenants if t == "rare")
    print(f"\nrare-tenant fraction = {rare_frac:.4f}  ({n_rare} of {N} vectors)  "
          f"selectivity = {'HIGH (selective)' if rare_frac < 0.05 else 'low (broad)'}")
    print(f"  {'strategy':>12} | {'returned':>8} | {'recall@k':>8}")
    print("  " + "-" * 34)
    rows = {}
    for name, fn in strategies.items():
        got = fn(vecs, tenants, qv, "rare", K)
        rec = recall_at_k(got, truth, K)
        rows[name] = (len(got), rec)
        flag = "  <-- COLLAPSE" if rec < 0.5 and name == "post-filter" else ""
        print(f"  {name:>12} | {len(got):>8} | {rec:>8.3f}{flag}")
    return rows


def main() -> int:
    print("FILTERED ANN: the same query, three strategies, as the filter gets "
          "more selective (the matching tenant gets rarer).")

    # BROAD filter: 'rare' is actually common -> post-filter is fine.
    run(0.40)
    # SELECTIVE filter: 'rare' is genuinely rare -> watch post-filter collapse.
    run(0.02)
    run(0.004)

    print("\nLESSON: with a BROAD filter, post-filter finds matches inside the "
          "global ANN candidates, so all three agree. With a SELECTIVE filter, the "
          "rare tenant's vectors fall OUTSIDE the global top-N candidates, so "
          "post-filter returns too few — RECALL COLLAPSES — silently (no error). "
          "Pre-filter and native search WITHIN the matching subset, so they hold. "
          "This is exactly why production stores (Qdrant's filterable HNSW) filter "
          "DURING traversal, and why you must measure recall AT A SELECTIVE FILTER, "
          "not just unfiltered latency (Lecture 1 §3). The store that holds recall "
          "on your most selective tenant is the store that won't quietly return "
          "'no results' for your smallest customers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact recalls vary with the seed)
# -----------------------------------------------------------------------------
#
# FILTERED ANN: the same query, three strategies, as the filter gets more selective...
#
# rare-tenant fraction = 0.4000  (2000 of 5000 vectors)  selectivity = low (broad)
#      strategy | returned | recall@k
#   ----------------------------------
#    post-filter |       10 |    1.000
#     pre-filter |       10 |    1.000
#         native |       10 |    1.000
#
# rare-tenant fraction = 0.0200  (100 of 5000 vectors)  selectivity = HIGH (selective)
#      strategy | returned | recall@k
#   ----------------------------------
#    post-filter |        3 |    0.300  <-- COLLAPSE
#     pre-filter |       10 |    1.000
#         native |       10 |    1.000
#
# rare-tenant fraction = 0.0040  (20 of 5000 vectors)  selectivity = HIGH (selective)
#      strategy | returned | recall@k
#   ----------------------------------
#    post-filter |        0 |    0.000  <-- COLLAPSE
#     pre-filter |       10 |    1.000
#         native |       10 |    1.000
#
# LESSON: ... post-filter RECALL COLLAPSES silently as the filter gets selective ...
#
# READ IT: at 40% the rare tenant is everywhere, so its vectors ARE in the global
# ANN candidates and post-filter is fine. At 0.4% the rare tenant's 20 vectors are
# almost never in the global top-100, so post-filter returns 0 — the search "works"
# (no error) and returns NOTHING for that tenant. That's the silent bug. Native
# filtering (Qdrant's filterable HNSW) gets the index's speed AND the filter's
# exactness, so it returns the right 10 every time. THIS number — recall at a
# selective filter — is what picks the store for multi-tenant retrieval.
# -----------------------------------------------------------------------------
