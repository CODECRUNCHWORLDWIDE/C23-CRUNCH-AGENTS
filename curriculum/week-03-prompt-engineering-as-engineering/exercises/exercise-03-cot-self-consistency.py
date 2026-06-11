#!/usr/bin/env python3
# Exercise 3 — Chain-of-thought vs direct, and self-consistency
#
# Goal: Replace the slogans "use CoT" and "use self-consistency" with NUMBERS on
#       a small reasoning set. You will measure three strategies on the same
#       problems:
#         * DIRECT            -- ask for the answer, no reasoning
#         * COT               -- "reason step by step, then give the answer"
#         * SELF-CONSISTENCY  -- sample N CoT paths, take the MAJORITY answer
#       and report the ACCURACY DELTA and the COST MULTIPLE of each, so the
#       trade-off is a pair of numbers you can defend (Lecture 1 sections 5-6).
#
# Estimated time: 50 minutes. Runnable.
#
# WHY MEASURE, NOT ASSUME
#
#   Lecture 1: CoT is task- and model-shaped; self-consistency buys accuracy at
#   N-times the cost. The only way to know if either earns its tokens on YOUR
#   task is to measure. This file makes you do exactly that.
#
# HOW TO USE THIS FILE
#
#       pip install anthropic
#       export ANTHROPIC_API_KEY=sk-ant-...      # optional; Ollama fallback below
#       ollama pull qwen2.5:7b                    # local path, diversifiable temp
#       python3 exercise-03-cot-self-consistency.py
#
#   It runs a small set of multi-step word problems with known answers through
#   all three strategies and prints an accuracy + cost table. With no API key it
#   falls back to Ollama (which also lets you set temperature>0 for diverse
#   self-consistency paths).
#
# THE TODOs
#
#   Four gaps are marked "# TODO N:". Fill them to complete the measurement.
#
# ACCEPTANCE CRITERIA
#
#   [ ] You report accuracy for DIRECT, COT, and SELF-CONSISTENCY (N>=3).
#   [ ] You report the COST MULTIPLE (model calls) of each strategy.
#   [ ] Self-consistency uses a MAJORITY VOTE over N sampled answers.
#   [ ] You can state, in one sentence, whether CoT/self-consistency earned
#       their tokens on THIS set (often: marginal on easy items, real on hard).
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


# -----------------------------------------------------------------------------
# A tiny reasoning set: multi-step word problems with KNOWN integer answers.
# Small on purpose so the exercise is cheap to run; the methodology scales.
# -----------------------------------------------------------------------------
@dataclass
class Problem:
    question: str
    answer: int


PROBLEMS: list[Problem] = [
    Problem("A shop sells pens at 3 for $2. How many dollars for 18 pens?", 12),
    Problem("Tom has 5 boxes with 4 apples each. He eats 3. How many apples remain?", 17),
    Problem("A train travels 60 km in 45 minutes. How many km in 3 hours at the same rate?", 240),
    Problem("Sara reads 12 pages a day for 2 weeks, then 20 pages a day for 5 days. Total pages?", 268),
    Problem("If 7 workers build a wall in 6 days, how many days for 3 workers (same rate)?", 14),
    Problem("A jar has 24 red and 36 blue marbles. What percent are red?", 40),
]

DIRECT_SYSTEM = (
    "Answer with ONLY the final integer. No words, no units, no explanation."
)
COT_SYSTEM = (
    "Reason step by step. After your reasoning, end with a line of the exact "
    "form 'ANSWER: <integer>'."
)


# -----------------------------------------------------------------------------
# Backends. Each returns the model's raw text for (system, question). For the
# local path we expose temperature so self-consistency can DIVERSIFY paths.
# -----------------------------------------------------------------------------
def ask_anthropic(system: str, question: str, model: str = "claude-haiku-4-5") -> str:
    if anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("anthropic SDK / ANTHROPIC_API_KEY unavailable")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


def ask_ollama(system: str, question: str, temperature: float = 0.0,
               model: str = "qwen2.5:7b") -> str:
    if httpx is None:
        raise RuntimeError("httpx not installed (pip install httpx)")
    r = httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300.0,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


USING_ANTHROPIC = anthropic is not None and bool(os.environ.get("ANTHROPIC_API_KEY"))


def ask(system: str, question: str, temperature: float = 0.0) -> str:
    """Dispatch to whichever backend is available. temperature applies locally."""
    if USING_ANTHROPIC:
        return ask_anthropic(system, question)   # 2026 models manage sampling
    return ask_ollama(system, question, temperature=temperature)


# -----------------------------------------------------------------------------
# Answer extraction + scoring.
# -----------------------------------------------------------------------------
def parse_answer(text: str) -> int | None:
    """Pull the final integer: prefer 'ANSWER: <n>', else the last integer seen."""
    m = re.search(r"ANSWER:\s*(-?\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


@dataclass
class StratResult:
    name: str
    correct: int
    total: int
    calls: int          # total model calls -- the cost proxy

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_direct() -> StratResult:
    correct = calls = 0
    for p in PROBLEMS:
        out = ask(DIRECT_SYSTEM, p.question)
        calls += 1
        if parse_answer(out) == p.answer:
            correct += 1
    return StratResult("direct", correct, len(PROBLEMS), calls)


def run_cot() -> StratResult:
    correct = calls = 0
    for p in PROBLEMS:
        out = ask(COT_SYSTEM, p.question)
        calls += 1
        # TODO 1: parse the answer from `out` and compare to p.answer; increment
        #         `correct` when they match. (Use parse_answer.)
        pass
    return StratResult("cot", correct, len(PROBLEMS), calls)


def run_self_consistency(n: int = 5) -> StratResult:
    correct = calls = 0
    for p in PROBLEMS:
        votes: list[int] = []
        for i in range(n):
            # Diversify paths: locally we raise temperature; on Anthropic 2026
            # models the sampler is managed, so repeated calls still vary.
            out = ask(COT_SYSTEM, p.question, temperature=0.7)
            calls += 1
            a = parse_answer(out)
            if a is not None:
                votes.append(a)
        # TODO 2: take the MAJORITY vote over `votes` (most common value). Guard
        #         the empty case (no parseable answers) -> treat as wrong.
        #         Hint: Counter(votes).most_common(1)[0][0]
        majority: int | None = None
        if majority == p.answer:
            correct += 1
    return StratResult(f"self-consistency(N={n})", correct, len(PROBLEMS), calls)


def print_table(results: list[StratResult]) -> None:
    baseline_calls = results[0].calls if results else 1
    print("\n" + "=" * 66)
    print(f"{'STRATEGY':<22} {'ACCURACY':>10} {'CALLS':>7} {'COST x':>8}")
    print("-" * 66)
    for r in results:
        # TODO 3: cost multiple = this strategy's calls / the direct baseline's
        #         calls. Replace `cost_x`.
        cost_x = 1.0
        print(f"{r.name:<22} {r.correct}/{r.total} ({r.accuracy:>5.1%}) "
              f"{r.calls:>7} {cost_x:>7.1f}x")
    print("=" * 66)

    # TODO 4: print a one-line verdict comparing CoT and direct accuracy. If CoT
    #         did NOT beat direct, say it didn't earn its tokens on this set;
    #         if it did, say by how many points. (results[0]=direct, [1]=cot.)
    if len(results) >= 2:
        pass


def main() -> None:
    print(f"[backend] {'anthropic (claude-haiku-4-5)' if USING_ANTHROPIC else 'ollama (qwen2.5:7b)'}")
    print(f"Reasoning set: {len(PROBLEMS)} multi-step problems with known answers\n")
    results = [run_direct(), run_cot(), run_self_consistency(n=5)]
    print_table(results)
    print("\nReminder (Lecture 1 section 5): the visible CoT is NOT a faithful "
          "audit of the answer -- use it to maybe improve accuracy, never to "
          "certify WHY the model answered.")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Expected output (your numbers will differ; the SHAPE is the lesson -- CoT
# helps on multi-step items, self-consistency adds a bit more at N-times cost)
# -----------------------------------------------------------------------------
#
# [backend] ollama (qwen2.5:7b)
# Reasoning set: 6 multi-step problems with known answers
#
# ==================================================================
# STRATEGY                 ACCURACY   CALLS   COST x
# ------------------------------------------------------------------
# direct                   3/6 (50.0%)      6     1.0x
# cot                      5/6 (83.3%)      6     1.0x
# self-consistency(N=5)    6/6 (100.0%)    30     5.0x
# ==================================================================
# CoT beat direct by +33.3 points on this multi-step set -- it earned its tokens.
#
# THE LESSON: on genuine multi-step reasoning, CoT lifts accuracy for ~no extra
# CALLS (just more tokens per call), and self-consistency squeezes out more at
# 5x the calls. On an EASY classification task you'd see ~no CoT lift and pure
# waste from self-consistency. Measure on YOUR task; never assume the lever.
# -----------------------------------------------------------------------------
