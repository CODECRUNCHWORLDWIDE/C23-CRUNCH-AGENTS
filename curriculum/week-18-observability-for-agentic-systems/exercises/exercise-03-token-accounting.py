#!/usr/bin/env python3
# Exercise 3 — Token accounting + p50/p95/p99 from recorded spans
#
# Goal: Given a list of synthetic RECORDED spans (each carrying the real gen_ai.*
#       usage attributes + route/user/model metadata + a duration), compute the
#       two aggregates every observability dashboard is built from:
#         (1) token + COST rollups per route / per user / per model, priced with
#             REAL 2026 Anthropic rates (Lecture 1 §6), and
#         (2) p50/p95/p99 LATENCY per agent step (Lecture 2 §1.1).
#       This is the "build the metric before you trust the dashboard" exercise —
#       the same move as week 8, where you built Recall@5 before the harness.
#
# Estimated time: 50 minutes. Runnable.
#
# WHAT THIS DEPENDS ON (read before running)
#
#   * REQUIRED: `pip install numpy`. Everything else is the standard library.
#   * NO backend, NO network, NO API key. The "spans" are a fixed in-file list, so
#     the rollups and percentiles are FULLY DETERMINISTIC and reproduce exactly.
#   * In production you would not write this loop — Langfuse and Phoenix compute
#     these rollups FROM the same gen_ai.usage.* attributes. You build it by hand
#     here so you know precisely what the dashboard is doing (Lecture 1 §6).
#
# ACCEPTANCE CRITERIA
#
#   [ ] Prints a per-route, per-user, AND per-model table of input tokens, output
#       tokens, and USD cost.
#   [ ] Prints p50/p95/p99 latency per agent step (grouped by operation/step).
#   [ ] Costs use the real 2026 Anthropic prices below; a self-hosted open model
#       costs $0 (you own the GPU, not a per-token bill).
#   [ ] You can read the p50/p99 spread and say "tail problem" vs "systemic slow."
#
# Expected output is at the bottom of the file.

from __future__ import annotations

from collections import defaultdict

import numpy as np

# --- REAL 2026 Anthropic prices, USD per token (input, output) = $/M / 1e6 ------
# A self-hosted open model is $0/token: you pay for the GPU, not per token.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8":   (5.00 / 1e6, 25.00 / 1e6),
    "claude-sonnet-4-6": (3.00 / 1e6, 15.00 / 1e6),
    "claude-haiku-4-5":  (1.00 / 1e6,  5.00 / 1e6),
    "self-hosted-open":  (0.0,         0.0),       # you own the GPU
}


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    p_in, p_out = PRICES[model]
    return in_tok * p_in + out_tok * p_out


# --- The recorded spans: one dict per LLM-call/step span -----------------------
# Each carries the REAL gen_ai.* attribute names, plus crunch.* metadata (the
# slicing dimensions) and a duration_ms. This is the shape an OTLP/JSON export
# gives you — exactly what you'd read back from Langfuse/Phoenix.
SPANS: list[dict] = [
    # route=summarize: a cheap haiku plan + an expensive opus generation, per call
    {"name": "chat plan",     "gen_ai.operation.name": "chat",
     "gen_ai.request.model": "claude-haiku-4-5",
     "gen_ai.usage.input_tokens": 612,  "gen_ai.usage.output_tokens": 28,
     "crunch.route": "summarize", "crunch.user_id": "u_17", "duration_ms": 190},
    {"name": "chat generate", "gen_ai.operation.name": "chat",
     "gen_ai.request.model": "claude-opus-4-8",
     "gen_ai.usage.input_tokens": 1843, "gen_ai.usage.output_tokens": 412,
     "crunch.route": "summarize", "crunch.user_id": "u_17", "duration_ms": 2100},
    {"name": "chat plan",     "gen_ai.operation.name": "chat",
     "gen_ai.request.model": "claude-haiku-4-5",
     "gen_ai.usage.input_tokens": 590,  "gen_ai.usage.output_tokens": 25,
     "crunch.route": "summarize", "crunch.user_id": "u_42", "duration_ms": 205},
    {"name": "chat generate", "gen_ai.operation.name": "chat",
     "gen_ai.request.model": "claude-opus-4-8",
     "gen_ai.usage.input_tokens": 1990, "gen_ai.usage.output_tokens": 455,
     "crunch.route": "summarize", "crunch.user_id": "u_42", "duration_ms": 9200},  # tail!

    # route=search: haiku plan + sonnet generation; cheaper, snappier
    {"name": "chat plan",     "gen_ai.operation.name": "chat",
     "gen_ai.request.model": "claude-haiku-4-5",
     "gen_ai.usage.input_tokens": 480,  "gen_ai.usage.output_tokens": 22,
     "crunch.route": "search", "crunch.user_id": "u_17", "duration_ms": 180},
    {"name": "chat generate", "gen_ai.operation.name": "chat",
     "gen_ai.request.model": "claude-sonnet-4-6",
     "gen_ai.usage.input_tokens": 1100, "gen_ai.usage.output_tokens": 240,
     "crunch.route": "search", "crunch.user_id": "u_17", "duration_ms": 1300},
    {"name": "chat generate", "gen_ai.operation.name": "chat",
     "gen_ai.request.model": "claude-sonnet-4-6",
     "gen_ai.usage.input_tokens": 1250, "gen_ai.usage.output_tokens": 300,
     "crunch.route": "search", "crunch.user_id": "u_88", "duration_ms": 1450},

    # route=classify: a single self-hosted open model call; $0 cost
    {"name": "chat generate", "gen_ai.operation.name": "chat",
     "gen_ai.request.model": "self-hosted-open",
     "gen_ai.usage.input_tokens": 800,  "gen_ai.usage.output_tokens": 12,
     "crunch.route": "classify", "crunch.user_id": "u_88", "duration_ms": 95},

    # NON-chat steps: retrieval/tool spans carry duration but NO token usage.
    # They count toward per-step latency, NOT toward token/cost rollups.
    {"name": "embeddings query", "gen_ai.operation.name": "embeddings",
     "crunch.route": "search", "crunch.user_id": "u_17", "duration_ms": 60},
    {"name": "vector_search",    "gen_ai.operation.name": "execute_tool",
     "gen_ai.tool.name": "vector_search", "crunch.route": "search",
     "crunch.user_id": "u_17", "duration_ms": 240},
    {"name": "vector_search",    "gen_ai.operation.name": "execute_tool",
     "gen_ai.tool.name": "vector_search", "crunch.route": "summarize",
     "crunch.user_id": "u_42", "duration_ms": 8700},                       # tail!
    {"name": "vector_search",    "gen_ai.operation.name": "execute_tool",
     "gen_ai.tool.name": "vector_search", "crunch.route": "search",
     "crunch.user_id": "u_88", "duration_ms": 270},
]


# --- (1) Token + cost rollups: group LLM-call spans by an attribute key --------
def rollup(spans: list[dict], key: str) -> dict[str, dict]:
    """Group `chat` spans by `key` and sum input/output tokens + USD cost.
    Only chat spans have token usage; everything else is skipped (Lecture 1 §6)."""
    out: dict[str, dict] = defaultdict(lambda: {"in": 0, "out": 0, "usd": 0.0})
    for s in spans:
        if s.get("gen_ai.operation.name") != "chat":
            continue
        in_tok = int(s.get("gen_ai.usage.input_tokens", 0))
        out_tok = int(s.get("gen_ai.usage.output_tokens", 0))
        bucket = out[str(s[key])]
        bucket["in"] += in_tok
        bucket["out"] += out_tok
        bucket["usd"] += cost_usd(str(s["gen_ai.request.model"]), in_tok, out_tok)
    return dict(out)


def print_rollup(title: str, table: dict[str, dict]) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'key':<18} {'in_tok':>8} {'out_tok':>8} {'cost_usd':>10}")
    print("  " + "-" * 46)
    total_in = total_out = 0
    total_usd = 0.0
    for k in sorted(table):
        b = table[k]
        total_in += b["in"]; total_out += b["out"]; total_usd += b["usd"]
        print(f"  {k:<18} {b['in']:>8} {b['out']:>8} {b['usd']:>10.4f}")
    print("  " + "-" * 46)
    print(f"  {'TOTAL':<18} {total_in:>8} {total_out:>8} {total_usd:>10.4f}")


# --- (2) p50/p95/p99 latency per step -----------------------------------------
def percentiles(durations_ms: list[float]) -> dict[str, float]:
    """p50/p95/p99 from a list of span durations (ms). A sort-and-index, not an
    average and not a max (Lecture 2 §1.1)."""
    if not durations_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
    arr = np.asarray(durations_ms, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "n": len(arr),
    }


def latency_per_step(spans: list[dict]) -> dict[str, dict]:
    """Group durations by step (we use the span name) and percentile each group."""
    by_step: dict[str, list[float]] = defaultdict(list)
    for s in spans:
        by_step[s["name"]].append(float(s["duration_ms"]))
    return {step: percentiles(durs) for step, durs in by_step.items()}


def print_latency(table: dict[str, dict]) -> None:
    print("\n=== p50/p95/p99 latency per step (ms) ===")
    print(f"  {'step':<18} {'n':>3} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
    print("  " + "-" * 56)
    for step in sorted(table):
        p = table[step]
        print(f"  {step:<18} {p['n']:>3} {p['mean']:>8.0f} {p['p50']:>8.0f} "
              f"{p['p95']:>8.0f} {p['p99']:>8.0f}")


def main() -> int:
    print(f"recorded spans: {len(SPANS)} "
          f"({sum(1 for s in SPANS if s.get('gen_ai.operation.name') == 'chat')} "
          f"chat, the rest non-billed steps)")

    # (1) the three cost rollups — same function, different group-by key.
    print_rollup("tokens + cost per ROUTE",  rollup(SPANS, "crunch.route"))
    print_rollup("tokens + cost per USER",   rollup(SPANS, "crunch.user_id"))
    print_rollup("tokens + cost per MODEL",  rollup(SPANS, "gen_ai.request.model"))

    # (2) per-step latency percentiles.
    lat = latency_per_step(SPANS)
    print_latency(lat)

    # Read the tail: vector_search has a 8700ms outlier among ~250ms calls.
    vs = lat["vector_search"]
    print("\n==================== READING THE TAIL ====================")
    print(f"  vector_search: p50={vs['p50']:.0f}ms  p99={vs['p99']:.0f}ms  "
          f"(mean={vs['mean']:.0f}ms)")
    if vs["p99"] > 5 * vs["p50"]:
        print("  p99 is many times p50 -> a TAIL problem: almost every call is")
        print("  fast, but a rare one (a retry / cold dependency) stalls for")
        print("  seconds. Open ONE of those slow traces and find the dominant")
        print("  span (Lecture 2 §3.2) — do NOT 'optimize the average', the")
        print("  average is a fiction the outlier poisoned.")
    print("\n  NOTE: the mean COST per route hides the same asymmetry as latency —")
    print("  'summarize' costs far more per call than 'classify' (opus vs a")
    print("  self-hosted $0 model). Both facts came straight off the spans.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact numbers are deterministic from the SPANS above)
# -----------------------------------------------------------------------------
#
# recorded spans: 12 (8 chat, the rest non-billed steps)
#
# === tokens + cost per ROUTE ===
#   key                  in_tok  out_tok   cost_usd
#   ----------------------------------------------
#   classify                800       12     0.0000      <-- self-hosted, $0
#   search                 2830      562     0.0157
#   summarize              5035      920     0.0423
#   ----------------------------------------------
#   TOTAL                  8665     1494     0.0580
#
# === tokens + cost per USER ===
#   key                  in_tok  out_tok   cost_usd
#   ----------------------------------------------
#   u_17                   4035      702     0.0278
#   u_42                   2580      480     0.0220
#   u_88                   2050      312     0.0083
#   ----------------------------------------------
#   TOTAL                  8665     1494     0.0580
#
# === tokens + cost per MODEL ===
#   key                  in_tok  out_tok   cost_usd
#   ----------------------------------------------
#   claude-haiku-4-5       1682       75     0.0021
#   claude-opus-4-8        3833      867     0.0408       <-- most of the spend
#   claude-sonnet-4-6      2350      540     0.0152
#   self-hosted-open        800       12     0.0000
#   ----------------------------------------------
#   TOTAL                  8665     1494     0.0580
#
# === p50/p95/p99 latency per step (ms) ===
#   step                 n     mean      p50      p95      p99
#   --------------------------------------------------------
#   chat generate        5     2829     1450     7780     8916
#   chat plan            3      192      190      204      205
#   embeddings query     1       60       60       60       60
#   vector_search        3     3070      270     7857     8531
#
# ==================== READING THE TAIL ====================
#   vector_search: p50=270ms  p99=8531ms  (mean=3070ms)
#   p99 is many times p50 -> a TAIL problem: ... open ONE slow trace.
#
# WHY THIS MATTERS: every number above is a GROUP-BY over span attributes — tokens
# summed by a slicing key, durations percentiled by step. Put the slicing
# dimension (crunch.route, crunch.user_id) on the span at emit time and these
# questions have exact answers; omit it and the call is invisible to the rollup.
# This is the mechanism Langfuse/Phoenix implement for you — built once by hand.
# -----------------------------------------------------------------------------
