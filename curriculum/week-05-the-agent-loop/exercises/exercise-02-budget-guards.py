#!/usr/bin/env python3
# Exercise 2 — Budget guards (force each of the four budgets to fire)
#
# Goal: Prove that an agent loop with step / token / time / cost budgets ALWAYS
#       terminates. You will deliberately drive each budget to its limit and
#       watch the matching guard stop the run — turning "it might hang" into
#       "it cannot hang."
#
# Estimated time: 45 minutes. Runnable.
#
# THE FOUR BUDGETS (Lecture 2 §1)
#
#   step   — caps the number of model turns.
#   token  — caps cumulative input+output tokens across the run.
#   time   — caps wall-clock seconds.
#   cost   — caps dollars (tokens * price).
#
#   Each catches a DIFFERENT runaway. This exercise forces each to be the one
#   that fires, by setting the others generously and that one tightly.
#
# HOW TO USE THIS FILE
#
#   This file is standalone and runs WITHOUT a real model by default: it ships a
#   FakeModel that loops forever (always asks for a tool, never says end_turn),
#   so every budget is reachable deterministically. That makes the termination
#   guarantee testable in seconds with no API calls.
#
#       python3 exercise-02-budget-guards.py
#
#   It runs the loop four times, each time with one budget set tight and the rest
#   loose, and asserts the correct budget fired. All four must pass.
#
#   PART B (optional): set USE_REAL_MODEL = True to run the same guarded loop
#   against claude-opus-4-8 on a real task and watch a clean end_turn termination
#   plus the live budget counters. Requires ANTHROPIC_API_KEY.
#
# ACCEPTANCE CRITERIA
#
#   [ ] All four guarded runs terminate and the EXPECTED budget fires each time
#       (the script asserts this and prints PASS).
#   [ ] You can explain why an infinite-looping model still terminates: the loop,
#       not the model, owns the exit.
#   [ ] (Part B) The real-model run terminates on end_turn with a printed summary.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import time
from dataclasses import dataclass, field

USE_REAL_MODEL = False  # flip to True for Part B (needs anthropic + ANTHROPIC_API_KEY)

# claude-opus-4-8 pricing, USD per token (Anthropic pricing).
PRICE_IN = 5.00 / 1_000_000
PRICE_OUT = 25.00 / 1_000_000


@dataclass
class Budgets:
    max_steps: int = 8
    max_tokens: int = 20_000
    max_seconds: float = 30.0
    max_dollars: float = 0.10

    steps: int = 0
    tokens: int = 0
    dollars: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def record(self, in_tok: int, out_tok: int) -> None:
        self.tokens += in_tok + out_tok
        self.dollars += in_tok * PRICE_IN + out_tok * PRICE_OUT

    def exceeded(self) -> str | None:
        if self.steps >= self.max_steps:
            return "step budget"
        if self.tokens >= self.max_tokens:
            return "token budget"
        if time.monotonic() - self.started >= self.max_seconds:
            return "time budget"
        if self.dollars >= self.max_dollars:
            return "cost budget"
        return None

    def summary(self, reason: str) -> str:
        elapsed = time.monotonic() - self.started
        return (
            f"  terminated: {reason} | steps={self.steps}/{self.max_steps} "
            f"tokens={self.tokens}/{self.max_tokens} "
            f"time={elapsed:.2f}s/{self.max_seconds:.0f}s "
            f"cost=${self.dollars:.4f}/${self.max_dollars:.2f}"
        )


# --- A model stand-in that NEVER finishes -----------------------------------
#
# It always "asks for a tool," so the only thing that can stop the loop is a
# budget. Each call reports a fixed token usage and (optionally) sleeps, so we
# can drive the token, time, and cost budgets deterministically.


@dataclass
class FakeResponse:
    stop_reason: str
    in_tokens: int
    out_tokens: int


class FakeModel:
    def __init__(self, in_tokens: int = 1500, out_tokens: int = 1500, sleep: float = 0.0):
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.sleep = sleep

    def call(self) -> FakeResponse:
        if self.sleep:
            time.sleep(self.sleep)
        # Always tool_use -> the model never volunteers to stop.
        return FakeResponse("tool_use", self.in_tokens, self.out_tokens)


def guarded_loop(model: FakeModel, b: Budgets) -> str:
    """The loop. Every exit is either end_turn or a named budget — no third exit."""
    while True:
        breached = b.exceeded()
        if breached:
            return breached
        resp = model.call()
        b.steps += 1
        b.record(resp.in_tokens, resp.out_tokens)
        if resp.stop_reason != "tool_use":
            return "end_turn"
        # (a real loop runs tools here; the fake model just loops)


def expect(name: str, b: Budgets, model: FakeModel, want: str) -> bool:
    got = guarded_loop(model, b)
    ok = got == want
    print(f"[{name}] {b.summary(got)}")
    print(f"  expected '{want}', got '{got}' -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> None:
    if USE_REAL_MODEL:
        run_real_model()
        return

    print("Forcing each budget to fire (the model loops forever; only a budget stops it):\n")
    results = []

    # 1) STEP budget: tiny step cap, everything else loose.
    results.append(expect(
        "step",
        Budgets(max_steps=3, max_tokens=10**9, max_seconds=1e9, max_dollars=1e9),
        FakeModel(in_tokens=10, out_tokens=10),
        want="step budget",
    ))

    # 2) TOKEN budget: tiny token cap; each call burns 3000 tokens.
    results.append(expect(
        "token",
        Budgets(max_steps=10**6, max_tokens=5_000, max_seconds=1e9, max_dollars=1e9),
        FakeModel(in_tokens=1500, out_tokens=1500),
        want="token budget",
    ))

    # 3) TIME budget: tiny wall-clock cap; each call sleeps 0.1s.
    results.append(expect(
        "time",
        Budgets(max_steps=10**6, max_tokens=10**9, max_seconds=0.3, max_dollars=1e9),
        FakeModel(in_tokens=10, out_tokens=10, sleep=0.1),
        want="time budget",
    ))

    # 4) COST budget: tiny dollar cap; each call costs ~$0.0525 (1500*$5/M + 1500*$25/M).
    results.append(expect(
        "cost",
        Budgets(max_steps=10**6, max_tokens=10**9, max_seconds=1e9, max_dollars=0.10),
        FakeModel(in_tokens=1500, out_tokens=1500),
        want="cost budget",
    ))

    print("=" * 60)
    if all(results):
        print("ALL FOUR BUDGETS FIRED CORRECTLY. The agent cannot hang.")
    else:
        print("A guard did not fire as expected — check the exceeded() order.")
    print("=" * 60)


def run_real_model() -> None:
    """Part B: the guarded loop against claude-opus-4-8 on a real task."""
    import anthropic

    client = anthropic.Anthropic()
    b = Budgets(max_steps=8, max_tokens=20_000, max_seconds=30.0, max_dollars=0.10)
    messages = [{"role": "user", "content": "What is (1234 * 7) + 19? Answer in one line."}]

    while True:
        breached = b.exceeded()
        if breached:
            print(b.summary(breached + " exceeded"))
            return
        resp = client.messages.create(
            model="claude-opus-4-8", max_tokens=1024, messages=messages,
        )
        b.steps += 1
        b.record(resp.usage.input_tokens, resp.usage.output_tokens)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = next((blk.text for blk in resp.content if blk.type == "text"), "")
            print(f"answer: {text}")
            print(b.summary("end_turn"))
            return


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Expected output (USE_REAL_MODEL = False)
# -----------------------------------------------------------------------------
#
# Forcing each budget to fire (the model loops forever; only a budget stops it):
#
# [step]   terminated: step budget | steps=3/3 tokens=60/1000000000 ...
#   expected 'step budget', got 'step budget' -> PASS
#
# [token]   terminated: token budget | steps=2/1000000 tokens=6000/5000 ...
#   expected 'token budget', got 'token budget' -> PASS
#
# [time]   terminated: time budget | steps=3/1000000 ... time=0.30s/0s ...
#   expected 'time budget', got 'time budget' -> PASS
#
# [cost]   terminated: cost budget | ... cost=$0.1050/$0.10
#   expected 'cost budget', got 'cost budget' -> PASS
#
# ============================================================
# ALL FOUR BUDGETS FIRED CORRECTLY. The agent cannot hang.
# ============================================================
#
# The lesson: the model loops forever, yet every run terminates — because the
# LOOP, not the model, owns the exit. Exact token/time/cost numbers vary by the
# moment a guard trips; the SHAPE is invariant: the budget you set tight is the
# one that fires.
# -----------------------------------------------------------------------------
