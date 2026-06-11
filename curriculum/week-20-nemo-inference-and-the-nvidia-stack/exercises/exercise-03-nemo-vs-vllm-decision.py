#!/usr/bin/env python3
# Exercise 3 — NeMo-vs-vLLM decision (a weighted decision matrix, not a vibe)
#
# Goal: Turn "which serving stack?" from an argument into a measured decision.
#       Take recorded (or synthetic) throughput/latency numbers for a NeMo-
#       TensorRT-LLM deploy vs the week-19 vLLM deploy, PLUS the qualitative axes
#       from Lecture 1 (kernel perf, policy tooling, OSS velocity, operational
#       simplicity, lock-in), score them on a WEIGHTED decision matrix, and print
#       the matrix + a recommendation WITH REASONS. The whole point: the answer
#       depends on YOUR weights (your constraints), so the matrix makes the
#       trade-off explicit instead of inheriting it.
#
# Estimated time: 50 minutes. Runnable. NO GPU. Pure stdlib + numpy.
#
# HOW TO USE THIS FILE
#
#   Standalone:
#
#       python3 exercise-03-nemo-vs-vllm-decision.py
#
#   It ships with SYNTHETIC numbers (clearly labeled) so it runs anywhere. Replace
#   them with your OWN measured numbers from the challenge:
#     - PERF: throughput (tok/s) and p99 latency (ms) for each stack, same H100,
#       same Qwen2.5-14B, same workload (week 19 for vLLM, this week for NeMo).
#     - QUAL: 1-5 scores for the qualitative axes (Lecture 1 §6).
#     - WEIGHTS: how much each axis matters to YOUR deployment. This is the lever.
#
#   It also does real 2026 cost math (H100 ~ $2-3/hr) to convert throughput into
#   cost-per-million-tokens, because a serving decision that ignores the bill is
#   incomplete.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The script prints a weighted decision matrix with a per-axis score for
#       each stack and a weighted total.
#   [ ] It prints a recommendation naming the winner, the margin, and the 2-3
#       axes that drove it.
#   [ ] Changing WEIGHTS (e.g. crank operational simplicity up, lock-in tolerance
#       down) can FLIP the recommendation - proving the decision is constraint-
#       dependent, which is the whole lesson.
#   [ ] The cost-per-Mtok numbers are derived from real $/hr and measured tok/s.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# 1) PERF numbers. SYNTHETIC placeholders — replace with your measured numbers.
#    Same model (Qwen2.5-14B), same H100, same workload. NeMo's edge here is the
#    compiled FP8/BF16 engine + in-flight batching (Lecture 1 §2); vLLM is strong
#    but a runtime, not a compiler. These illustrative numbers give NeMo a ~20%
#    throughput edge — YOUR benchmark may differ, which is the point.
# ---------------------------------------------------------------------------
@dataclass
class Perf:
    name: str
    throughput_tok_s: float   # aggregate output tokens/sec under load
    p99_latency_ms: float     # 99th-percentile per-request latency
    rail_overhead_ms: float = 0.0  # extra per-request latency from Guardrails (Lecture 2 §6)


NEMO = Perf(name="NeMo / TensorRT-LLM / Triton",
            throughput_tok_s=4800.0, p99_latency_ms=620.0, rail_overhead_ms=180.0)
VLLM = Perf(name="vLLM",
            throughput_tok_s=4000.0, p99_latency_ms=700.0, rail_overhead_ms=0.0)

# Real 2026 H100 rental price. Both stacks run on the SAME H100, so the $/hr is
# identical; the cost-per-token difference comes entirely from throughput.
H100_USD_PER_HR = 2.50


# ---------------------------------------------------------------------------
# 2) QUALITATIVE axes (1-5, higher = better) from Lecture 1 §6. These are
#    judgments you defend in the memo, not measurements. Defaults reflect the
#    lecture's honest table; adjust to YOUR experience.
# ---------------------------------------------------------------------------
@dataclass
class Stack:
    name: str
    perf: Perf
    qual: dict = field(default_factory=dict)


AXES = [
    "kernel_perf",           # raw throughput/latency on NVIDIA HW
    "policy_tooling",        # NeMo Guardrails integration (Lecture 2)
    "flexibility",           # model coverage, no build step, change config & go
    "oss_velocity",          # community pace, new-model support speed
    "operational_simplicity",# how easy to run in prod (procs, ports, layers)
    "lock_in_freedom",       # higher = LESS locked in (portability)
]

nemo = Stack(name=NEMO.name, perf=NEMO, qual={
    "kernel_perf": 5,
    "policy_tooling": 5,
    "flexibility": 2,
    "oss_velocity": 3,
    "operational_simplicity": 2,
    "lock_in_freedom": 1,
})
vllm = Stack(name=VLLM.name, perf=VLLM, qual={
    "kernel_perf": 4,
    "policy_tooling": 2,
    "flexibility": 5,
    "oss_velocity": 5,
    "operational_simplicity": 5,
    "lock_in_freedom": 5,
})

# ---------------------------------------------------------------------------
# 3) WEIGHTS — the lever. How much does each axis matter to YOUR deployment?
#    These must sum to ~1.0. The DEFAULT below is a balanced profile. Change them
#    to model your real constraints, and watch the recommendation move.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "kernel_perf": 0.25,
    "policy_tooling": 0.20,
    "flexibility": 0.15,
    "oss_velocity": 0.10,
    "operational_simplicity": 0.20,
    "lock_in_freedom": 0.10,
}


# ---------------------------------------------------------------------------
# Cost math: convert throughput into $ per million output tokens.
# ---------------------------------------------------------------------------
def cost_per_mtok(perf: Perf, usd_per_hr: float) -> float:
    tok_per_hr = perf.throughput_tok_s * 3600.0
    usd_per_tok = usd_per_hr / tok_per_hr
    return usd_per_tok * 1_000_000.0


def effective_p99(perf: Perf) -> float:
    """p99 INCLUDING the rail overhead — the honest apples-to-apples number
    (Lecture 2 §6: you cannot compare bare vLLM to NeMo+rails and call it fair)."""
    return perf.p99_latency_ms + perf.rail_overhead_ms


def weighted_score(stack: Stack, weights: dict) -> float:
    return sum(weights[a] * stack.qual[a] for a in AXES)


def main() -> int:
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "WEIGHTS must sum to 1.0"

    stacks = [nemo, vllm]

    print("=" * 72)
    print("PERFORMANCE (same Qwen2.5-14B, same H100, same workload)")
    print("=" * 72)
    print(f"{'stack':<34} {'tok/s':>8} {'p99(ms)':>9} {'+rail':>7} {'$/Mtok':>9}")
    print("-" * 72)
    for s in stacks:
        print(f"{s.name:<34} {s.perf.throughput_tok_s:>8.0f} "
              f"{s.perf.p99_latency_ms:>9.0f} {effective_p99(s.perf):>7.0f} "
              f"{cost_per_mtok(s.perf, H100_USD_PER_HR):>9.4f}")
    print(f"(H100 @ ${H100_USD_PER_HR:.2f}/hr, 2026; same hardware so $/Mtok tracks throughput)")
    print("NOTE: '+rail' is p99 INCLUDING Guardrails overhead. Comparing bare vLLM")
    print("      p99 to NeMo+rail p99 is the apples-to-apples mistake (Lecture 2 §6).")

    print("\n" + "=" * 72)
    print("WEIGHTED DECISION MATRIX (qualitative axes, 1-5; higher better)")
    print("=" * 72)
    header = f"{'axis':<24} {'weight':>7} " + " ".join(f"{s.name.split(' /')[0]:>12}" for s in stacks)
    print(header)
    print("-" * 72)
    for a in AXES:
        row = f"{a:<24} {WEIGHTS[a]:>7.2f} "
        row += " ".join(f"{s.qual[a]:>12d}" for s in stacks)
        print(row)
    print("-" * 72)
    totals = {s.name: weighted_score(s, WEIGHTS) for s in stacks}
    row = f"{'WEIGHTED TOTAL':<24} {sum(WEIGHTS.values()):>7.2f} "
    row += " ".join(f"{totals[s.name]:>12.3f}" for s in stacks)
    print(row)

    # ---- Recommendation -------------------------------------------------
    print("\n" + "=" * 72)
    print("RECOMMENDATION")
    print("=" * 72)
    ranked = sorted(stacks, key=lambda s: totals[s.name], reverse=True)
    winner, runner = ranked[0], ranked[1]
    margin = totals[winner.name] - totals[runner.name]

    # The axes that drove the win: where the winner out-scored the runner-up,
    # weighted by importance.
    contribs = []
    for a in AXES:
        delta = (winner.qual[a] - runner.qual[a]) * WEIGHTS[a]
        contribs.append((a, delta))
    drivers = [a for a, d in sorted(contribs, key=lambda x: -x[1]) if d > 0][:3]

    print(f"WINNER: {winner.name}")
    print(f"  weighted total {totals[winner.name]:.3f} vs {totals[runner.name]:.3f} "
          f"(margin {margin:.3f})")
    print(f"  driven by: {', '.join(drivers)}")
    print(f"  throughput: {winner.perf.throughput_tok_s:.0f} tok/s, "
          f"${cost_per_mtok(winner.perf, H100_USD_PER_HR):.4f}/Mtok")

    if margin < 0.25:
        print("  CLOSE CALL: the margin is thin. The qualitative tie-breakers and YOUR")
        print("  constraints (the WEIGHTS) decide it - this is exactly the call the memo")
        print("  must justify, not a runaway winner.")

    print("\n--- sensitivity: flip the weights, flip the decision ---")
    # Demonstrate constraint-dependence: a vLLM-favoring profile (simplicity +
    # portability matter most) and a NeMo-favoring one (perf + policy matter most).
    profiles = {
        "ops-first (simplicity + portability)": {
            "kernel_perf": 0.10, "policy_tooling": 0.10, "flexibility": 0.20,
            "oss_velocity": 0.15, "operational_simplicity": 0.30, "lock_in_freedom": 0.15,
        },
        "perf+policy-first (NVIDIA shop)": {
            "kernel_perf": 0.35, "policy_tooling": 0.40, "flexibility": 0.05,
            "oss_velocity": 0.05, "operational_simplicity": 0.10, "lock_in_freedom": 0.05,
        },
    }
    for label, w in profiles.items():
        assert abs(sum(w.values()) - 1.0) < 1e-6
        t = {s.name: weighted_score(s, w) for s in stacks}
        win = max(stacks, key=lambda s: t[s.name])
        print(f"  {label:<40} -> {win.name.split(' /')[0]} "
              f"({t[win.name]:.3f} vs {min(t.values()):.3f})")
    print("  The recommendation MOVES with the weights. That is the lesson: there is")
    print("  no universal best stack - there is the best stack for YOUR constraints,")
    print("  and the matrix makes the trade-off explicit instead of inherited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact numbers depend on the inputs above)
# -----------------------------------------------------------------------------
#
# ========================================================================
# PERFORMANCE (same Qwen2.5-14B, same H100, same workload)
# ========================================================================
# stack                                 tok/s   p99(ms)   +rail    $/Mtok
# ------------------------------------------------------------------------
# NeMo / TensorRT-LLM / Triton           4800       620     800    0.1447
# vLLM                                   4000       700     700    0.1736
# (H100 @ $2.50/hr, 2026; same hardware so $/Mtok tracks throughput)
# NOTE: '+rail' is p99 INCLUDING Guardrails overhead. ...
#
# ========================================================================
# WEIGHTED DECISION MATRIX (qualitative axes, 1-5; higher better)
# ========================================================================
# axis                      weight         NeMo         vLLM
# ------------------------------------------------------------------------
# kernel_perf                 0.25            5            4
# policy_tooling              0.20            5            2
# flexibility                 0.15            2            5
# oss_velocity                0.10            3            5
# operational_simplicity      0.20            2            5
# lock_in_freedom             0.10            1            5
# ------------------------------------------------------------------------
# WEIGHTED TOTAL              1.00        3.300        4.150
#
# ========================================================================
# RECOMMENDATION
# ========================================================================
# WINNER: vLLM
#   weighted total 4.150 vs 3.300 (margin 0.850)
#   driven by: operational_simplicity, flexibility, lock_in_freedom
#   throughput: 4000 tok/s, $0.1736/Mtok
#
# --- sensitivity: flip the weights, flip the decision ---
#   ops-first (simplicity + portability)     -> vLLM (...)
#   perf+policy-first (NVIDIA shop)           -> NeMo (...)
#   The recommendation MOVES with the weights. ...
#
# READ IT: with the DEFAULT balanced weights, vLLM's flexibility + simplicity +
# portability outweigh NeMo's kernel-perf + policy edge — even though NeMo is
# faster and cheaper per token. Crank perf+policy (an NVIDIA shop) and NeMo wins.
# The numbers don't decide; YOUR weights do, and the memo defends the weights.
# -----------------------------------------------------------------------------
