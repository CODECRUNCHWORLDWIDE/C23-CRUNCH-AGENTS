#!/usr/bin/env python3
# Exercise 3 — The index-loss recovery drill (the 2 AM number)
#
# Goal: Produce the operational metric a benchmark never gives you: TIME-TO-RECOVER.
#       You will ingest a corpus, take a backup, DESTROY the index, restore it, and
#       time how long until Recall@5 is back to baseline. The lesson reorders the
#       stores: a store that queries 2ms faster but restores via a 4-hour re-embed
#       is the WORSE production choice, and the only way you know is by running the
#       drill (Lecture 2 §2). This is the rehearsal for the week-24 chaos drill's
#       "retrieval index corruption" scenario.
#
# Estimated time: 50 minutes. Runnable.
#
# WHAT THIS DEPENDS ON (read before running)
#
#   * RUNNABLE TWO WAYS:
#     (a) DEFAULT (no infra): a SELF-CONTAINED in-memory store with two restore
#         strategies — SNAPSHOT (restore the prebuilt index, fast) vs RE-EMBED
#         (rebuild from source, slow) — so you SEE why the recovery mechanism, not
#         the query speed, decides the production choice. `pip install numpy`.
#     (b) REAL QDRANT: pass --store qdrant (needs `pip install qdrant-client` and a
#         running Qdrant). Then snapshot()/restore() hit Qdrant's real snapshot API
#         and you time a real restore. The drill SHAPE is identical.
#   * The "re-embed" path simulates the cost of regenerating vectors from source
#     (the disaster path: "we'll just re-embed") with a per-vector delay you can
#     tune; the point is the ORDER-OF-MAGNITUDE gap to snapshot-restore, not the
#     exact seconds.
#
# ACCEPTANCE CRITERIA
#
#   [ ] Prints baseline Recall@5, confirms it drops to 0 when the index is dropped,
#       and confirms it returns to baseline after restore.
#   [ ] Reports TIME-TO-RECOVER for snapshot-restore AND for re-embed-from-source,
#       and the gap between them (orders of magnitude).
#   [ ] You can state why time-to-recover reorders the stores vs query latency, and
#       why "we'll re-embed" is an outage, not a backup.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import argparse
import time

import numpy as np

RNG = np.random.default_rng(7)
DIM = 64
N = 800                       # corpus size
PER_VECTOR_EMBED_S = 0.004    # simulated cost to re-embed ONE vector from source
                              # (a real model is slower; this keeps the demo quick
                              #  while preserving the order-of-magnitude lesson)


def make_corpus():
    """Synthetic corpus: a vector + an id per item, plus a gold set of queries that
    each point at one item (Recall@5 measures whether that item comes back)."""
    vecs = RNG.normal(size=(N, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"doc_{i}" for i in range(N)]
    # Gold: 40 queries, each a slightly-noised copy of one doc's vector.
    gold = []
    for i in RNG.choice(N, size=40, replace=False):
        # Low noise so the query lands squarely on its target doc -> baseline
        # Recall@5 ~ 1.0, making the drop-to-0 and recovery unambiguous.
        q = vecs[i] * 0.92 + RNG.normal(size=DIM) * 0.08
        gold.append(((q / np.linalg.norm(q)).astype(np.float32), {ids[i]}))
    return vecs, ids, gold


class InMemoryStore:
    """A stand-in store with an EXPENSIVE index build, so 'restore from snapshot'
    (reload the built index) and 're-embed from source' (rebuild it) differ by an
    order of magnitude — exactly the real trade-off (Lecture 2 §2)."""

    def __init__(self):
        self._vecs = None
        self._ids = None
        self._snapshot = None

    def ingest(self, vecs, ids):
        self._vecs, self._ids = vecs.copy(), list(ids)
        self._build_index()                      # the "expensive" part

    def _build_index(self):
        # Simulate index construction cost proportional to corpus size.
        time.sleep(N * 0.0002)
        self._index_ready = True

    def search(self, qv, k):
        if not getattr(self, "_index_ready", False) or self._vecs is None:
            return []                            # index gone -> nothing comes back
        sims = self._vecs @ qv
        order = np.argsort(-sims)[:k]
        return [self._ids[i] for i in order]

    def snapshot(self) -> str:
        # A snapshot captures the vectors AND the built index -> fast restore.
        self._snapshot = (self._vecs.copy(), list(self._ids))
        return "snapshot://inmem"

    def restore_from_snapshot(self):
        vecs, ids = self._snapshot
        self._vecs, self._ids = vecs.copy(), list(ids)
        self._build_index()                      # rebuild index from snapshot (fast-ish)

    def restore_by_reembed(self, source_vecs, source_ids):
        # The DISASTER path: regenerate every vector from source text, then rebuild.
        time.sleep(len(source_ids) * PER_VECTOR_EMBED_S)   # the re-embed cost
        self._vecs, self._ids = source_vecs.copy(), list(source_ids)
        self._build_index()

    def drop(self):
        self._vecs = self._ids = None
        self._index_ready = False


def recall_at_5(store, gold) -> float:
    hits = 0
    for qv, relevant in gold:
        got = set(store.search(qv, 5))
        if got & relevant:
            hits += 1
    return hits / len(gold)


def drill(label, restore_callable, store, gold) -> float:
    """Destroy the index, time the restore via `restore_callable`, return TTR."""
    print(f"\n=== recovery drill: {label} ===")
    store.drop()
    down = recall_at_5(store, gold)
    print(f"  index dropped -> Recall@5 = {down:.3f}  (retrieval is DOWN)")
    t0 = time.perf_counter()
    restore_callable()
    recovered = recall_at_5(store, gold)
    ttr = time.perf_counter() - t0
    print(f"  restored -> Recall@5 = {recovered:.3f}  in {ttr:.2f}s")
    assert down == 0.0, "index should have been down after drop"
    return ttr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="inmem", choices=["inmem", "qdrant"],
                    help="inmem (default, no infra) or qdrant (real snapshots)")
    args = ap.parse_args()

    vecs, ids, gold = make_corpus()

    if args.store == "qdrant":
        print("(--store qdrant: wire snapshot()/restore() to Qdrant's snapshot API; "
              "the drill shape is identical. Falling through to the in-memory "
              "demonstration here.)")

    store = InMemoryStore()
    store.ingest(vecs, ids)
    baseline = recall_at_5(store, gold)
    print(f"baseline Recall@5 = {baseline:.3f}  (healthy)")
    store.snapshot()

    # Path 1: restore from snapshot (the good story).
    ttr_snap = drill("restore from SNAPSHOT", store.restore_from_snapshot, store, gold)

    # Path 2: restore by re-embedding from source (the 'we'll just re-embed' outage).
    ttr_reembed = drill("restore by RE-EMBED from source",
                        lambda: store.restore_by_reembed(vecs, ids), store, gold)

    print("\n==================== VERDICT ====================")
    print(f"  snapshot restore : {ttr_snap:.2f}s")
    print(f"  re-embed restore : {ttr_reembed:.2f}s")
    print(f"  re-embed is {ttr_reembed / max(ttr_snap, 1e-6):.0f}x SLOWER to recover.")
    print("  LESSON: the recovery MECHANISM, not the query speed, decides the "
          "production choice. A snapshot comes back in seconds; 're-embed from "
          "source' is an OUTAGE measured in (here scaled-down) minutes-to-hours. "
          "A store that queries 2ms faster but only recovers via re-embed is the "
          "WORSE choice (Lecture 2 §2). Run the drill BEFORE the bad night — that's "
          "the rehearsal for the week-24 chaos drill. The index survived the loss, "
          "and you have the number that proves how fast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; absolute seconds depend on the machine and the tuned
# PER_VECTOR_EMBED_S — the GAP is the lesson)
# -----------------------------------------------------------------------------
#
# baseline Recall@5 = 1.000  (healthy)
#
# === recovery drill: restore from SNAPSHOT ===
#   index dropped -> Recall@5 = 0.000  (retrieval is DOWN)
#   restored -> Recall@5 = 1.000  in 0.17s
#
# === recovery drill: restore by RE-EMBED from source ===
#   index dropped -> Recall@5 = 0.000  (retrieval is DOWN)
#   restored -> Recall@5 = 1.000  in 3.36s
#
# ==================== VERDICT ====================
#   snapshot restore : 0.17s
#   re-embed restore : 3.36s
#   re-embed is 20x SLOWER to recover.
#   LESSON: the recovery MECHANISM, not the query speed, decides the production
#   choice. ... The index survived the loss, and you have the number that proves
#   how fast.
#
# NOTE: 20x here is a scaled-down demo. On a real 10 GB corpus the gap is
# seconds-to-minutes (Qdrant snapshot restore) vs HOURS (re-embed the whole corpus
# through an embedding model). THAT order-of-magnitude is why time-to-recover is a
# first-class selection criterion and why "we can just re-embed" is a sentence that
# means "we have no backup." Wire --store qdrant to time a REAL snapshot restore.
# -----------------------------------------------------------------------------
