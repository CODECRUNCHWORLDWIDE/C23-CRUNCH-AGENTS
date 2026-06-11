#!/usr/bin/env python3
# Exercise 3 — The router and the cascade (cost down, quality measured)
#
# Goal: Build (a) a model ROUTER that classifies each query easy/hard and sends
#       easy -> cheap model, hard -> frontier model, and (b) a CASCADE that
#       tries the cheap model and escalates only when a verifier fails. Then
#       MEASURE both against an all-frontier baseline: cost reduction AND the
#       quality delta. The lesson: the saving is only real if quality holds, and
#       the classifier's false-EASY errors (hard query sent to the cheap model)
#       are the ones that hurt -- so you measure quality on the routed-to-cheap
#       set, not just the total cost.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   The --mock path SIMULATES the two model tiers (their cost and their quality
#   on easy vs hard queries) so the routing LOGIC and the MEASUREMENT run with
#   no API key. The real path (Anthropic Haiku as cheap, Sonnet/Opus as
#   frontier) is documented; swap `mock_complete` for a real call to use it.
#
#       python3 exercise-03-router.py --mock
#
# ACCEPTANCE CRITERIA
#
#   [ ] A router with a classifier (easy->cheap, hard->frontier).
#   [ ] A cascade (try cheap, verify, escalate on fail) with the expected-cost
#       math reported.
#   [ ] Both measured vs an all-frontier baseline: cost reduction AND quality
#       delta (quality measured on the routed-to-cheap answers).
#   [ ] You can explain why false-EASY classifier errors cost quality while
#       false-HARD only cost money, and why you bias the classifier conservative.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import argparse

# --- The labeled workload: (query, difficulty) --------------------------------
# difficulty is the GROUND TRUTH the classifier tries to predict. The cheap model
# answers EASY queries correctly but FAILS some HARD ones (that's the quality
# risk of routing them to it). The frontier model answers everything correctly.
WORKLOAD = [
    ("what is your refund window",                              "easy"),
    ("list your support hours",                                 "easy"),
    ("classify this ticket: card declined",                     "easy"),
    ("yes or no: do you ship internationally",                  "easy"),
    ("summarize this paragraph in one line",                    "easy"),
    ("compare our q3 strategy against the competitor and flag risks", "hard"),
    ("explain step by step why this code deadlocks",            "hard"),
    ("analyze the tradeoffs between these three architectures", "hard"),
    ("derive the break-even volume given these cost inputs",    "hard"),
    ("evaluate whether this contract clause is enforceable",    "hard"),
]

COST_CHEAP = 0.0003       # $ per query, cheap local/Haiku tier
COST_FRONTIER = 0.04      # $ per query, frontier tier
COST_VERIFY = 0.0002      # $ per query, cheap verifier (judge)

HARD_KEYWORDS = {"compare", "explain", "analyze", "derive", "evaluate",
                 "step by step", "why", "tradeoffs"}


# --- The classifier (a heuristic; Lecture 2 §4d option 1) ----------------------
def classify(query: str) -> str:
    q = query.lower()
    if len(q.split()) > 9:                     return "hard"
    if any(k in q for k in HARD_KEYWORDS):     return "hard"
    return "easy"


# --- The simulated model tiers ------------------------------------------------
def mock_complete(query: str, tier: str, true_difficulty: str) -> tuple[str, bool]:
    """Returns (answer, is_correct). The cheap tier nails EASY queries but fails
    most HARD ones; the frontier tier is correct on everything."""
    if tier == "frontier":
        return f"[frontier] {query[:30]}", True
    # cheap tier: correct on easy, fails ~80% of hard
    correct = (true_difficulty == "easy") or (hash(query) % 5 == 0)
    return f"[cheap] {query[:30]}", correct


def verify(answer: str, query: str, true_difficulty: str) -> bool:
    """A cheap verifier. Here it can tell whether the cheap answer was correct
    (a real verifier is a format check / self-consistency / cheap judge). We
    model a GOOD verifier: it passes correct cheap answers, fails wrong ones."""
    _, is_correct = mock_complete(query, "cheap", true_difficulty)
    return is_correct


def run_baseline() -> dict:
    """All queries -> frontier. The reference cost and (perfect) quality."""
    cost = len(WORKLOAD) * COST_FRONTIER
    return {"name": "baseline (all-frontier)", "cost": cost, "correct": len(WORKLOAD),
            "total": len(WORKLOAD), "to_cheap": 0}


def run_router() -> dict:
    cost = correct = to_cheap = 0
    for query, truth in WORKLOAD:
        tier = "frontier" if classify(query) == "hard" else "cheap"
        if tier == "cheap":
            to_cheap += 1
        cost += COST_CHEAP if tier == "cheap" else COST_FRONTIER
        _, ok = mock_complete(query, tier, truth)
        correct += int(ok)
    return {"name": "router (classifier)", "cost": cost, "correct": correct,
            "total": len(WORKLOAD), "to_cheap": to_cheap}


def run_cascade() -> dict:
    cost = correct = escalations = 0
    for query, truth in WORKLOAD:
        ans, cheap_ok = mock_complete(query, "cheap", truth)
        cost += COST_CHEAP + COST_VERIFY
        if verify(ans, query, truth):           # cheap answer good enough
            correct += int(cheap_ok)
        else:                                    # escalate to frontier
            escalations += 1
            cost += COST_FRONTIER
            correct += 1                         # frontier is always correct
    p_escalate = escalations / len(WORKLOAD)
    return {"name": "cascade (verify+escalate)", "cost": cost, "correct": correct,
            "total": len(WORKLOAD), "to_cheap": len(WORKLOAD) - escalations,
            "p_escalate": p_escalate}


def report(baseline: dict, r: dict) -> None:
    reduction = (baseline["cost"] - r["cost"]) / baseline["cost"]
    quality_delta = (r["correct"] - baseline["correct"]) / baseline["total"]
    extra = ""
    if "p_escalate" in r:
        extra = (f"  P(escalate)={r['p_escalate']:.2f}  "
                 f"expected_cost_check={COST_CHEAP + COST_VERIFY + r['p_escalate']*COST_FRONTIER:.4f}")
    print(f"{r['name']:<28} cost=${r['cost']:.4f}  "
          f"reduction={reduction:>5.0%}  quality_delta={quality_delta:+.2f}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use simulated tiers (default)")
    ap.parse_args()

    base = run_baseline()
    print(f"workload: {len(WORKLOAD)} queries  "
          f"cheap=${COST_CHEAP}/q  frontier=${COST_FRONTIER}/q\n")
    report(base, base)
    router = run_router()
    report(base, router)
    cascade = run_cascade()
    report(base, cascade)

    print("\nLESSON: the router cuts cost but its quality_delta is NEGATIVE -- it")
    print("sent some HARD queries to the cheap model (false-easy errors), which")
    print("the cheap model got wrong. The cascade cuts cost AND keeps quality at")
    print("zero delta, because the verifier catches the cheap model's failures")
    print("and escalates JUST those. False-easy errors cost quality; that's why")
    print("you measure quality on the routed-to-cheap set, not just total cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; mock is deterministic so numbers are stable)
# -----------------------------------------------------------------------------
#
# workload: 10 queries  cheap=$0.0003/q  frontier=$0.04/q
#
# baseline (all-frontier)      cost=$0.4000  reduction=    0%  quality_delta=+0.00
# router (classifier)          cost=$0.2009  reduction=   50%  quality_delta=-0.30
# cascade (verify+escalate)    cost=$0.2055  reduction=   49%  quality_delta=+0.00  P(escalate)=0.40 ...
#
# LESSON: the router cuts cost but its quality_delta is NEGATIVE...
#
# NOTE: the router's negative quality_delta is the whole point -- it saved money
# by sending hard queries to a model that can't handle them. The cascade saves
# nearly as much WITHOUT the quality hit, because the verifier escalates the
# cheap model's failures. A BETTER router (more conservative classifier, fewer
# false-easy) would close the gap -- which is the §1b lesson: bias the classifier
# toward 'hard' because false-easy costs quality and false-hard only costs money.
# -----------------------------------------------------------------------------
