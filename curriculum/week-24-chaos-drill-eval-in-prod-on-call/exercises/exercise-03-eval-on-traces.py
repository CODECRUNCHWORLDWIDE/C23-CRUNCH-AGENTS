#!/usr/bin/env python3
# Exercise 3 — Eval on traces (gate a deploy on production reality)
#
# Goal: Build the eval-in-prod gate that REPLAYS production traces through a
#       candidate version and BLOCKS the deploy if the candidate regresses on the
#       real query distribution — catching the regression on your desk, on real
#       data, before any user sees it.
#
# The lesson: the offline gold-set gate (Sprint B) tells you the system was good
# on the gold set last week. Trace replay tells you a candidate is good (or worse)
# on REAL traffic right now. Only the second claim keeps you out of trouble.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   This file ships a tiny set of SIMULATED production traces (real-shaped:
#   query + retrieved contexts + the answer that shipped) and two candidate
#   answer-generators (a good one and a regressed one), so the GATE LOGIC runs
#   with no live system and no judge spend. In the mini-project you swap the
#   simulated traces for real ones from Langfuse/Phoenix and the simulated
#   generator for your candidate writing-agent.
#
#       python3 exercise-03-eval-on-traces.py
#
#   It replays the traces through both candidates, scores each with a (mock)
#   faithfulness metric, compares to the incumbent baseline, and prints PASS for
#   the good candidate and FAIL (blocked deploy) for the regressed one.
#
# ACCEPTANCE CRITERIA
#
#   [ ] replay() runs each trace's query+contexts through a candidate generator.
#   [ ] score() returns a faithfulness number per candidate over the REAL traces.
#   [ ] gate() BLOCKS (returns False) when the candidate scores below the
#       incumbent baseline by more than a tolerance — a regression on real data.
#   [ ] The good candidate PASSES; the regressed candidate FAILS.
#   [ ] The output names WHICH traces regressed (so you know what broke).
#
# Expected output (shape) is at the bottom of the file.

from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Simulated production traces: real-shaped records captured by OTel tracing.
# Each is a real query + the contexts retrieved + the answer that shipped.
# --------------------------------------------------------------------------- #
@dataclass
class Trace:
    trace_id: str
    query: str
    contexts: list[str]
    shipped_answer: str


PROD_TRACES = [
    Trace("tr01", "What is the indemnity cap for data-breach claims?",
          ["The indemnity cap for data-breach claims is $2,000,000 per incident."],
          "The indemnity cap for data-breach claims is $2,000,000 per incident."),
    Trace("tr02", "How long must confidential information be protected?",
          ["Confidential information must be protected for five years after termination."],
          "Five years after termination."),
    Trace("tr03", "What is the payment term?",
          ["Invoices are net-30; late payments accrue interest at 1.5% per month."],
          "Net-30, with 1.5% monthly interest on late payments."),
    Trace("tr04", "What notice is required to terminate?",
          ["The agreement renews annually unless terminated with ninety days notice."],
          "Ninety days notice."),
]


# --------------------------------------------------------------------------- #
# Two candidate generators. The GOOD one grounds its answer in the context; the
# REGRESSED one ignores the context and confabulates (lower faithfulness).
# In the mini-project, both are replaced by your real candidate writing-agent
# (and the judge call is real — claude-opus-4-8, thinking adaptive).
# --------------------------------------------------------------------------- #
def good_candidate(query: str, contexts: list[str]) -> str:
    # Grounded: pull the answer from the context.
    return contexts[0]


def regressed_candidate(query: str, contexts: list[str]) -> str:
    # Regression: ignores context, makes something up.
    return "I believe the answer is approximately several million dollars, but the "\
           "exact figure depends on jurisdiction."


def mock_faithfulness(answer: str, contexts: list[str]) -> float:
    """Stand-in for Ragas faithfulness / a calibrated judge: 1.0 if the answer's
    key content is supported by the context, lower otherwise. Deterministic so the
    gate logic is testable. In the mini-project, call your real judge here."""
    ctx = " ".join(contexts).lower()
    ans_words = {w.strip(".,$%") for w in answer.lower().split() if len(w) > 3}
    ctx_words = {w.strip(".,$%") for w in ctx.split() if len(w) > 3}
    if not ans_words:
        return 0.0
    return len(ans_words & ctx_words) / len(ans_words)


# --------------------------------------------------------------------------- #
# The eval-in-prod gate.
# --------------------------------------------------------------------------- #
def replay(traces: list[Trace], candidate) -> list[tuple[str, float]]:
    """Run each trace's query+contexts through `candidate`, score faithfulness.
    Returns [(trace_id, faithfulness)]."""
    # TODO 1: for each trace, generate the candidate's answer from
    #   trace.query + trace.contexts, score it with mock_faithfulness against the
    #   trace's contexts, and collect (trace.trace_id, score).
    raise NotImplementedError("implement replay (TODO 1)")


def gate(baseline: float, scored: list[tuple[str, float]], tol: float = 0.05) -> bool:
    """Block the deploy if the candidate's mean faithfulness on REAL traces is
    below the incumbent baseline by more than `tol`. Name the regressed traces."""
    mean = sum(s for _, s in scored) / len(scored)
    regressed = [tid for tid, s in scored if s < baseline - tol]
    print(f"  candidate mean faithfulness on prod traces: {mean:.2f} "
          f"(baseline {baseline:.2f})")
    if regressed:
        print(f"  REGRESSED traces: {', '.join(regressed)}")
    # TODO 2: return True (ship) only if mean >= baseline - tol AND there are no
    #   regressed traces; otherwise return False (block the deploy).
    raise NotImplementedError("implement gate (TODO 2)")


def evaluate_candidate(name: str, candidate, baseline: float) -> None:
    print(f"=== candidate: {name} ===")
    scored = replay(PROD_TRACES, candidate)
    shipped = gate(baseline, scored)
    print(f"  GATE: {'PASS (ship)' if shipped else 'FAIL (deploy blocked)'}\n")


if __name__ == "__main__":
    # The incumbent's measured faithfulness on these prod traces (the baseline
    # the candidate must not regress below). The good candidate matches it; the
    # regressed one falls well under.
    INCUMBENT_BASELINE = 1.00
    evaluate_candidate("writing_agent_v4 (good)", good_candidate, INCUMBENT_BASELINE)
    evaluate_candidate("writing_agent_v5 (regressed)", regressed_candidate,
                       INCUMBENT_BASELINE)

# --------------------------------------------------------------------------- #
# EXPECTED OUTPUT (shape)
# --------------------------------------------------------------------------- #
#
# === candidate: writing_agent_v4 (good) ===
#   candidate mean faithfulness on prod traces: 1.00 (baseline 1.00)
#   GATE: PASS (ship)
#
# === candidate: writing_agent_v5 (regressed) ===
#   candidate mean faithfulness on prod traces: 0.10 (baseline 1.00)
#   REGRESSED traces: tr01, tr02, tr03, tr04
#   GATE: FAIL (deploy blocked)
#
# The point: the regressed candidate would have passed a casual demo (it returns
# fluent text), but replaying it through REAL traces shows it confabulates — and
# the gate BLOCKS the deploy before any user sees it. That is eval-in-prod: catch
# the regression on your desk, on real data, not in production at 3 AM.
#
# In the mini-project: pull real traces from Langfuse/Phoenix, replace the mock
# faithfulness with your calibrated claude-opus-4-8 judge, and wire this gate
# into the deploy pipeline so a regressed candidate cannot ship.
# --------------------------------------------------------------------------- #
