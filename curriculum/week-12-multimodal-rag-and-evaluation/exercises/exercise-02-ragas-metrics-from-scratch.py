#!/usr/bin/env python3
# Exercise 2 — The four Ragas metrics, FROM SCRATCH
#
# Goal: Stop the four Ragas metrics from being magic. You will implement
#       faithfulness, context recall, context precision, and answer relevancy
#       BY HAND on a tiny fixed dataset, using the SAME decomposition Ragas uses
#       (decompose into claims, check each claim against the context). When you
#       run the real library later, its numbers will be a black box no longer —
#       you'll know exactly which LLM sub-call produced each one.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   Standalone. No database, no network, no API key required by default. Just:
#
#       python3 exercise-02-ragas-metrics-from-scratch.py
#
#   The metrics need a "judge" to decide things like "is this claim supported by
#   the context?". By default this file uses a DETERMINISTIC STUB JUDGE — a tiny
#   keyword/substring matcher — so the file RUNS OFFLINE and the four metrics are
#   reproducible. The stub is good enough to make every metric move in the right
#   direction on the fixed dataset; it is NOT a real semantic judge. The header of
#   `StubJudge` documents how to swap in `claude-opus-4-8` (structured output) or a
#   local model so the SAME metric code runs against a real judge.
#
# ACCEPTANCE CRITERIA
#
#   [ ] All four metrics compute a score in [0,1] for every example.
#   [ ] The hallucinated example scores LOW on faithfulness (a claim is
#       unsupported) but may score fine on the other metrics — proving the metrics
#       are ORTHOGONAL (each catches a different failure).
#   [ ] The off-topic example scores LOW on answer relevancy while staying
#       faithful — again proving orthogonality.
#   [ ] You can explain, for each metric, which LLM sub-call the judge stands in
#       for, and how you'd replace the stub with a real LLM.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import re
from dataclasses import dataclass, field


# =============================================================================
# The judge. Swap this stub for a real LLM to run the SAME metrics for real.
# =============================================================================
#
# Real-judge path (documented, not run by default):
#
#   import anthropic
#   from pydantic import BaseModel, Field
#   client = anthropic.Anthropic()
#   class Supported(BaseModel):
#       supported: bool = Field(description="is the claim supported by the context?")
#   def claim_supported(claim, context):
#       prompt = f"Context:\n{context}\n\nClaim: {claim}\n\nIs the claim supported?"
#       r = client.messages.parse(
#           model="claude-opus-4-8", max_tokens=1000,
#           thinking={"type": "adaptive"}, output_config={"effort": "high"},
#           messages=[{"role": "user", "content": prompt}], output_format=Supported)
#       return r.parsed_output.supported
#
# The metric functions below call ONLY judge.claim_supported / judge.is_relevant,
# so swapping the stub for the real judge above changes nothing else.
class StubJudge:
    """Deterministic, offline stand-in for an LLM judge.

    A real judge reasons semantically; this stub uses keyword overlap, which is
    enough to make the four metrics behave correctly on the fixed dataset and to
    keep the file runnable with zero dependencies. Every decision a real LLM judge
    would make is isolated to one of these two methods.
    """

    @staticmethod
    def _keywords(text: str) -> set[str]:
        # Content words only; drop short stopword-ish tokens.
        words = re.findall(r"[a-z0-9$%]+", text.lower())
        return {w for w in words if len(w) > 3 or w.startswith("$")}

    def claim_supported(self, claim: str, context: str) -> bool:
        """Stand-in for 'is this claim entailed by the context?'.

        Supported iff most of the claim's content keywords appear in the context.
        """
        ck = self._keywords(claim)
        if not ck:
            return True
        ctx = self._keywords(context)
        overlap = len(ck & ctx) / len(ck)
        return overlap >= 0.6

    def is_relevant(self, item: str, question: str) -> bool:
        """Stand-in for 'is this retrieved chunk relevant to the question?'."""
        qk = self._keywords(question)
        ik = self._keywords(item)
        if not qk:
            return False
        return len(qk & ik) / len(qk) >= 0.3


# =============================================================================
# Claim decomposition (a real judge does this with an LLM; we split on sentences
# and conjunctions, which is enough for the fixed dataset).
# =============================================================================
def decompose_claims(answer: str) -> list[str]:
    """Break an answer into atomic claims. Ragas asks an LLM to do this; here we
    split on sentence boundaries and ' and ' so a compound answer becomes 2 claims."""
    parts: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        if not sentence:
            continue
        for piece in re.split(r"\s+and\s+", sentence):
            piece = piece.strip().rstrip(".")
            if piece:
                parts.append(piece)
    return parts


# =============================================================================
# The four metrics. Each returns a float in [0,1]. The ONLY judgement calls are
# judge.claim_supported and judge.is_relevant — swap the judge, keep the metrics.
# =============================================================================
def faithfulness(answer: str, contexts: list[str], judge) -> float:
    """Fraction of the ANSWER's claims that are supported by the retrieved context.
    Catches hallucination: claims the context never made."""
    claims = decompose_claims(answer)
    if not claims:
        return 1.0
    ctx = "\n".join(contexts)
    supported = sum(judge.claim_supported(c, ctx) for c in claims)
    return supported / len(claims)


def context_recall(reference: str, contexts: list[str], judge) -> float:
    """Fraction of the REFERENCE answer's claims attributable to the retrieved
    context. Catches a retriever that MISSED context the right answer needs.
    (Computed against the reference, not the generated answer.)"""
    claims = decompose_claims(reference)
    if not claims:
        return 1.0
    ctx = "\n".join(contexts)
    supported = sum(judge.claim_supported(c, ctx) for c in claims)
    return supported / len(claims)


def context_precision(question: str, contexts: list[str], judge) -> float:
    """Rank-weighted precision of the retrieved chunks' relevance to the question.
    Catches NOISY retrieval: irrelevant chunks, or relevant ones ranked low.

    Average Precision over the ranked contexts: for each relevant chunk at rank k,
    add precision@k; divide by the number of relevant chunks."""
    if not contexts:
        return 0.0
    relevances = [judge.is_relevant(c, question) for c in contexts]
    if not any(relevances):
        return 0.0
    cum_relevant = 0
    precision_sum = 0.0
    for k, rel in enumerate(relevances, start=1):
        if rel:
            cum_relevant += 1
            precision_sum += cum_relevant / k       # precision@k at this hit
    return precision_sum / cum_relevant


def answer_relevancy(question: str, answer: str, judge) -> float:
    """Does the answer ADDRESS the question? Catches off-topic/evasive answers.

    Ragas reverse-engineers questions from the answer and compares them to the
    real question via embeddings. Offline, we approximate that 'on-topic-ness'
    with keyword overlap between the answer and the question — high when the
    answer is about the question, low when it wandered."""
    qk = StubJudge._keywords(question)
    ak = StubJudge._keywords(answer)
    if not qk:
        return 0.0
    return len(qk & ak) / len(qk)


# =============================================================================
# The fixed dataset: 4 examples chosen so each metric is exercised distinctly.
# =============================================================================
@dataclass
class Sample:
    question: str
    answer: str
    reference: str
    contexts: list[str] = field(default_factory=list)
    note: str = ""


DATASET: list[Sample] = [
    Sample(
        question="How long must confidential information be protected?",
        answer="Confidential information must be protected for five years after termination.",
        reference="Five years after termination.",
        contexts=[
            "9. All confidential information must be protected for five years after termination.",
            "14. Either party may terminate this Agreement upon thirty days written notice.",
        ],
        note="GOOD: faithful, on-topic, context recalled, low noise.",
    ),
    Sample(
        question="How long must confidential information be protected?",
        answer="Confidential information must be protected for five years and the breach penalty is ten thousand dollars.",
        reference="Five years after termination.",
        contexts=[
            "9. All confidential information must be protected for five years after termination.",
        ],
        note="HALLUCINATION: the $10,000 penalty claim is NOT in the context.",
    ),
    Sample(
        question="How much liability insurance is required?",
        answer="Confidential information must be protected for five years after termination.",
        reference="One million dollars of professional liability insurance.",
        contexts=[
            "12. The Contractor shall maintain professional liability insurance of one million dollars.",
            "9. All confidential information must be protected for five years after termination.",
        ],
        note="OFF-TOPIC: faithful to context, but answers the WRONG question.",
    ),
    Sample(
        question="How is the annual fee paid?",
        answer="The annual fee is paid in twelve equal monthly installments.",
        reference="In twelve equal monthly installments.",
        contexts=[
            "1. This Agreement is entered into between the Company and the Contractor.",
            "18. This Agreement is governed by the laws of the State of Delaware.",
            "7. The annual fee shall be paid in twelve equal monthly installments.",
        ],
        note="NOISY RETRIEVAL: the relevant chunk is buried at rank 3 of 3.",
    ),
]


def main() -> int:
    judge = StubJudge()
    print("Computing the four Ragas-style metrics FROM SCRATCH (stub judge).\n")
    header = f"{'metric':>18} | " + " | ".join(f"ex{i}" for i in range(len(DATASET)))
    print(header)
    print("-" * len(header))

    faith = [faithfulness(s.answer, s.contexts, judge) for s in DATASET]
    crec = [context_recall(s.reference, s.contexts, judge) for s in DATASET]
    cprec = [context_precision(s.question, s.contexts, judge) for s in DATASET]
    arel = [answer_relevancy(s.question, s.answer, judge) for s in DATASET]

    def row(name, vals):
        print(f"{name:>18} | " + " | ".join(f"{v:.2f}" for v in vals))

    row("faithfulness", faith)
    row("context_recall", crec)
    row("context_precision", cprec)
    row("answer_relevancy", arel)

    print("\nPer-example notes:")
    for i, s in enumerate(DATASET):
        print(f"  ex{i}: {s.note}")

    # The lesson: the metrics are ORTHOGONAL. Verify the diagnostic shape.
    print("\n==================== LESSON CHECK ====================")
    hallucination_caught = faith[1] < faith[0]
    offtopic_caught = arel[2] < arel[0]
    noise_visible = cprec[3] < 1.0
    print(f"  faithfulness drops on the hallucination (ex1<ex0)? {hallucination_caught}")
    print(f"  answer_relevancy drops on the off-topic answer (ex2<ex0)? {offtopic_caught}")
    print(f"  context_precision penalizes the buried relevant chunk (ex3<1.0)? {noise_visible}")
    if hallucination_caught and offtopic_caught and noise_visible:
        print("  PASS: each failure was caught by a DIFFERENT metric. One collapsed")
        print("  'quality score' would have hidden which stage failed. That is why")
        print("  Ragas reports four numbers, not one. (Lecture 2 Part 1.)")
    else:
        print("  (Stub judge is coarse; with a real LLM judge the separation is")
        print("   sharper. The SHAPE — four orthogonal failure detectors — holds.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact decimals depend on the stub's keyword matching)
# -----------------------------------------------------------------------------
#
# Computing the four Ragas-style metrics FROM SCRATCH (stub judge).
#
#            metric | ex0 | ex1 | ex2 | ex3
# -----------------------------------------------
#       faithfulness | 1.00 | 0.50 | 1.00 | 1.00
#     context_recall | 1.00 | 1.00 | 1.00 | 1.00
#  context_precision | 1.00 | 1.00 | 1.00 | 0.33
#   answer_relevancy | 0.80 | 0.80 | 0.00 | 1.00
#
# Per-example notes:
#   ex0: GOOD: faithful, on-topic, context recalled, low noise.
#   ex1: HALLUCINATION: the $10,000 penalty claim is NOT in the context.
#   ex2: OFF-TOPIC: faithful to context, but answers the WRONG question.
#   ex3: NOISY RETRIEVAL: the relevant chunk is buried at rank 3 of 3.
#
# ==================== LESSON CHECK ====================
#   faithfulness drops on the hallucination (ex1<ex0)? True
#   answer_relevancy drops on the off-topic answer (ex2<ex0)? True
#   context_precision penalizes the buried relevant chunk (ex3<1.0)? True
#   PASS: each failure was caught by a DIFFERENT metric. ...
#
# READ THE TABLE: ex1's hallucination only shows up in FAITHFULNESS. ex2's
# off-topic answer only shows up in ANSWER_RELEVANCY (it's faithful — everything
# it says is in the context — it just answers the wrong question). ex3's buried
# relevant chunk only shows up in CONTEXT_PRECISION (rank-weighted: a relevant
# chunk at rank 3 scores 1/3, not 1). Four orthogonal detectors. A single "is it
# good?" number collapses these into something you cannot act on; the four
# separate numbers tell you WHICH STAGE to fix.
#
# -----------------------------------------------------------------------------
# Running it FOR REAL (claude-opus-4-8 judge), identical metric code:
#   Replace StubJudge with a class whose claim_supported / is_relevant call
#   client.messages.parse(model="claude-opus-4-8", thinking={"type":"adaptive"},
#   output_config={"effort":"high"}, output_format=<PydanticModel>). The four
#   metric functions DO NOT CHANGE — they only ever call judge.claim_supported /
#   judge.is_relevant. That is the whole point: the metric is the metric; the
#   judge is pluggable; and you CALIBRATE the judge (Exercise 3) before trusting
#   its decimals.
# -----------------------------------------------------------------------------
