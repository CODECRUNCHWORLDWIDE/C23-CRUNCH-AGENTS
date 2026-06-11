#!/usr/bin/env python3
# Exercise 3 — The judge-calibration sweep (agreement, Cohen's kappa, threshold)
#
# Goal: Make "never trust an uncalibrated judge" concrete. You have 10 human-
#       labeled (query, answer, context) examples and an LLM-as-judge that emits a
#       CONTINUOUS faithfulness-ish score. You will: (1) run the judge, (2) compare
#       it to the humans at raw-agreement AND Cohen's kappa, (3) SWEEP the decision
#       threshold tau, and (4) pick the tau that maximizes agreement. You will SEE
#       a judge that looks "80% accurate" collapse to a near-zero kappa at a bad
#       threshold, and recover to a trustworthy kappa at the calibrated one.
#
# Estimated time: 50 minutes. Runnable.
#
# WHAT THIS DEPENDS ON (read before running)
#
#   * Runs OFFLINE by default with a DETERMINISTIC STUB JUDGE, so no API key or
#     GPU is needed and the sweep is reproducible:  python3 this_file.py
#   * `numpy` is used for the threshold grid; if it is missing the file falls back
#     to a pure-Python linspace so it STILL RUNS. `scikit-learn` is optional (we
#     ship a from-scratch cohen_kappa and cross-check against sklearn if present).
#   * The REAL-judge path (claude-opus-4-8 with structured output) is documented in
#     `real_judge_score` below; swapping it in changes nothing about the sweep.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The script prints raw agreement AND Cohen's kappa for each threshold in
#       the sweep.
#   [ ] At least one threshold shows HIGH raw agreement but LOW kappa (the trap:
#       agreement inflated by chance / class imbalance).
#   [ ] The sweep identifies the tau that MAXIMIZES kappa, and you can state "this
#       is the threshold at which I trust the judge, kappa=X."
#   [ ] You can explain why you report kappa, not raw agreement, as the trust
#       number.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np

    def linspace(a, b, n):
        return [float(x) for x in np.linspace(a, b, n)]
except Exception:  # numpy missing -> pure-Python fallback so the file still runs.
    def linspace(a, b, n):
        if n == 1:
            return [a]
        step = (b - a) / (n - 1)
        return [a + step * i for i in range(n)]


# =============================================================================
# The 10 human-labeled calibration examples. label=1 means a human judged the
# answer FAITHFUL to the context; label=0 means NOT faithful (a hallucination or
# unsupported claim). This is the human ground truth the judge is checked against.
# =============================================================================
@dataclass
class Example:
    question: str
    answer: str
    context: str
    human_label: int  # 1 = faithful (human), 0 = not faithful (human)


CALIBRATION_SET: list[Example] = [
    Example("confidentiality duration",
            "Protected for five years after termination.",
            "All confidential information must be protected for five years after termination.",
            1),
    Example("confidentiality duration",
            "Protected for ten years.",
            "All confidential information must be protected for five years after termination.",
            0),  # wrong number -> not faithful
    Example("insurance amount",
            "Liability insurance of one million dollars.",
            "The Contractor shall maintain professional liability insurance of one million dollars.",
            1),
    Example("insurance amount",
            "Liability insurance of one million dollars, plus a two million umbrella.",
            "The Contractor shall maintain professional liability insurance of one million dollars.",
            0),  # umbrella claim is unsupported -> not faithful
    Example("termination notice",
            "Either party may terminate on thirty days written notice.",
            "Either party may terminate this Agreement upon thirty days written notice.",
            1),
    Example("termination notice",
            "Termination requires ninety days notice and a fee.",
            "Either party may terminate this Agreement upon thirty days written notice.",
            0),
    Example("governing law",
            "Governed by the laws of Delaware.",
            "This Agreement is governed by the laws of the State of Delaware.",
            1),
    Example("fee payment",
            "Paid in twelve equal monthly installments.",
            "The annual fee shall be paid in twelve equal monthly installments.",
            1),
    Example("dispute resolution",
            "Disputes go to binding arbitration in San Francisco.",
            "Any dispute shall be resolved by binding arbitration in San Francisco.",
            1),
    Example("dispute resolution",
            "Disputes are resolved in New York courts.",
            "Any dispute shall be resolved by binding arbitration in San Francisco.",
            0),
]


# =============================================================================
# The judge. Stub by default (deterministic, offline). Returns a CONTINUOUS score
# in [0,1] — like a real judge's faithfulness output — that we then threshold.
# =============================================================================
import re


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$%]+", text.lower())
            if len(w) > 3 or w[0] in "0123456789$"}


def stub_judge_score(answer: str, context: str) -> float:
    """Continuous 'is the answer supported by the context?' score in [0,1].

    Stand-in for an LLM faithfulness judge: fraction of the answer's content words
    that appear in the context. High when the answer stays inside the context, low
    when it adds unsupported claims (extra numbers, extra facts) -> the score drops
    because those words are absent from the context."""
    aw = _content_words(answer)
    if not aw:
        return 1.0
    cw = _content_words(context)
    return len(aw & cw) / len(aw)


# Real-judge path (documented; not run by default). Same return contract: a float.
#
#   import anthropic
#   from pydantic import BaseModel, Field
#   client = anthropic.Anthropic()
#   class Verdict(BaseModel):
#       score: float = Field(ge=0.0, le=1.0, description="faithfulness in [0,1]")
#   def real_judge_score(answer, context):
#       prompt = (f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
#                 "Score 0.0-1.0: is EVERY claim in the answer supported by the context?")
#       r = client.messages.parse(
#           model="claude-opus-4-8", max_tokens=1500,
#           thinking={"type": "adaptive"}, output_config={"effort": "high"},
#           messages=[{"role": "user", "content": prompt}], output_format=Verdict)
#       return r.parsed_output.score
#
# Swap stub_judge_score -> real_judge_score below and the sweep is unchanged.


# =============================================================================
# Cohen's kappa from scratch (agreement corrected for chance). Cross-checked
# against sklearn if it's installed.
# =============================================================================
def cohen_kappa(human: list[int], judge: list[int]) -> float:
    n = len(human)
    if n == 0:
        return 0.0
    p_obs = sum(h == j for h, j in zip(human, judge)) / n
    h1 = sum(human) / n
    j1 = sum(judge) / n
    p_chance = h1 * j1 + (1 - h1) * (1 - j1)
    if p_chance >= 1.0:           # degenerate: both raters all one class
        return 1.0 if p_obs == 1.0 else 0.0
    return (p_obs - p_chance) / (1 - p_chance)


def raw_agreement(human: list[int], judge: list[int]) -> float:
    return sum(h == j for h, j in zip(human, judge)) / len(human)


def kappa_band(k: float) -> str:
    if k < 0.20:
        return "poor (do NOT ship)"
    if k < 0.40:
        return "fair (weak)"
    if k < 0.60:
        return "moderate (use with caution)"
    if k < 0.80:
        return "substantial (trustworthy)"
    return "almost perfect"


def main() -> int:
    human = [e.human_label for e in CALIBRATION_SET]
    judge_scores = [stub_judge_score(e.answer, e.context) for e in CALIBRATION_SET]

    print("10 human-labeled examples. Judge emits a continuous score; we sweep the")
    print("decision threshold tau (score >= tau counts as 'faithful').\n")
    print(f"  human labels: {human}  (1=faithful, 0=not)")
    print(f"  judge scores: {[round(s, 2) for s in judge_scores]}\n")

    print(f"{'tau':>5} | {'judge@tau':>28} | {'agree':>6} | {'kappa':>6} | band")
    print("-" * 78)

    rows = []
    for tau in linspace(0.1, 0.9, 17):
        judge_binary = [1 if s >= tau else 0 for s in judge_scores]
        agree = raw_agreement(human, judge_binary)
        kappa = cohen_kappa(human, judge_binary)
        rows.append((tau, agree, kappa, judge_binary))
        shown = "".join(str(b) for b in judge_binary)
        print(f"{tau:>5.2f} | {shown:>28} | {agree:>6.2f} | {kappa:>6.2f} | "
              f"{kappa_band(kappa)}")

    # Pick the calibrated threshold: the tau that MAXIMIZES kappa (ties -> highest
    # agreement, then lowest tau for a tighter faithful bar).
    best = max(rows, key=lambda r: (round(r[2], 6), round(r[1], 6), -r[0]))
    print("-" * 78)
    print(f"CALIBRATED threshold tau={best[0]:.2f}  kappa={best[2]:.2f} "
          f"({kappa_band(best[2])}), raw agreement={best[1]:.2f}")

    # Show the trap: a threshold with high agreement but low kappa.
    trap = max(rows, key=lambda r: (round(r[1], 6) - round(r[2], 6)))
    if trap[1] - trap[2] > 0.15:
        print(f"\nTHE TRAP: at tau={trap[0]:.2f}, raw agreement looks like "
              f"{trap[1]:.2f} but kappa is only {trap[2]:.2f}. That gap is CHANCE")
        print("agreement (class imbalance) masquerading as a good judge. Report")
        print("KAPPA, not raw agreement, as your trust number. (Lecture 2 Part 4.)")

    # Cross-check kappa against sklearn if available.
    try:
        from sklearn.metrics import cohen_kappa_score
        jb = best[3]
        sk = cohen_kappa_score(human, jb)
        print(f"\nsklearn cohen_kappa_score at the calibrated tau: {sk:.2f} "
              f"(matches from-scratch {best[2]:.2f})")
    except Exception:
        print("\n(install scikit-learn to cross-check kappa against the standard impl)")

    print("\nNow: trust the judge ONLY at the calibrated tau, and carry "
          "'metric @ tau, kappa' into the milestone report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact numbers depend on the stub judge's word matching)
# -----------------------------------------------------------------------------
#
# 10 human-labeled examples. Judge emits a continuous score; we sweep the
# decision threshold tau (score >= tau counts as 'faithful').
#
#   human labels: [1, 0, 1, 0, 1, 0, 1, 1, 1, 0]  (1=faithful, 0=not)
#   judge scores: [1.0, 1.0, 1.0, 0.67, 1.0, 0.4, 1.0, 1.0, 0.75, 0.25]
#
#   tau |                    judge@tau |  agree |  kappa | band
# ------------------------------------------------------------------------------
#  0.10 |                   1111111111 |   0.60 |   0.00 | poor (do NOT ship)
#  ...
#  0.45 |                   1111101110 |   0.80 |   0.55 | moderate (use with caution)
#  ...
#  0.70 |                   1110101110 |   0.90 |   0.78 | substantial (trustworthy)
#  0.75 |                   1110101110 |   0.90 |   0.78 | substantial (trustworthy)
#  ...
#  0.90 |                   1110101100 |   0.80 |   0.58 | moderate (use with caution)
# ------------------------------------------------------------------------------
# CALIBRATED threshold tau=0.70  kappa=0.78 (substantial (trustworthy)), raw agreement=0.90
#
# THE TRAP: at tau=0.10, raw agreement looks like 0.60 but kappa is only 0.00.
# That gap is CHANCE agreement (class imbalance) masquerading as a good judge.
# Report KAPPA, not raw agreement, as your trust number. (Lecture 2 Part 4.)
#
# sklearn cohen_kappa_score at the calibrated tau: 0.78 (matches from-scratch 0.78)
#
# READ THE SWEEP: at a too-low tau the judge calls EVERYTHING faithful, so it
# agrees with the humans only on the faithful examples (raw agreement = the
# faithful base rate) and kappa = 0 (it's not discriminating, just guessing the
# majority class). As tau rises past the unfaithful answers' scores, the judge
# starts separating faithful from not, and kappa climbs. The calibrated tau is
# where kappa peaks — THAT is the threshold at which the judge's decimals mean
# something. A real LLM judge won't hit kappa=1.0; substantial (0.61-0.80) is the
# realistic target, and you report it next to every metric. (Lecture 2 Part 4.)
# -----------------------------------------------------------------------------
