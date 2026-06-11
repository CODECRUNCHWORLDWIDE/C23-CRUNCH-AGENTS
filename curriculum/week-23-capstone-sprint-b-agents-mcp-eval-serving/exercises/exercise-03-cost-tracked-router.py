#!/usr/bin/env python3
# Exercise 3 — The cost-tracked router (easy local, hard vendor)
#
# Goal: Build the easy-vs-hard classifier that decides which tier serves a query,
#       route easy queries to the local 7B and hard queries to claude-opus-4-8,
#       and ACCOUNT FOR per-request cost so the capstone's cost report (deliverable
#       5: median/p95/p99 per query by route) is real numbers, not a guess.
#
# The lesson is the week-21 mantra: the cheapest token is the one you don't
# generate; the second cheapest is the one a 7B handles instead of a frontier
# model. Routing easy queries local is the second-cheapest token at scale.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   The classifier calls claude-opus-4-8 (cheap: one structured call per query).
#   The "served" model call is MOCKED so you can run this WITHOUT a vLLM cluster
#   or vendor inference spend on the answers — only the classifier calls the API.
#   In the capstone, replace serve_mock() with a real LiteLLM call.
#
#       pip install anthropic
#       export ANTHROPIC_API_KEY=sk-ant-...
#       python3 exercise-03-cost-tracked-router.py
#
# ACCEPTANCE CRITERIA
#
#   [ ] classify_difficulty() returns 'easy' or 'hard' via a STRUCTURED call to
#       claude-opus-4-8 (an enum field, not parsed prose). No budget_tokens, no
#       temperature (both 400 on this model); use thinking + effort.
#   [ ] route_model() maps 'easy' -> local tier, 'hard' -> vendor tier.
#   [ ] Per-request cost is computed from token usage x the per-model rate.
#   [ ] The cost table prints per-query cost and the route, then a summary with
#       median/p95 cost and the local-vs-vendor split — the cost report skeleton.
#
# Expected output (shape) is at the bottom of the file.

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass

from anthropic import Anthropic

client = Anthropic()  # ANTHROPIC_API_KEY from the environment

# --------------------------------------------------------------------------- #
# Per-model pricing, $ per 1M tokens (input, output). The local tier is priced
# at an amortized self-host estimate; the vendor tier is the published rate.
# --------------------------------------------------------------------------- #
RATES = {
    # model name -> (input $/1M, output $/1M)
    "local-fast": (0.10, 0.10),         # amortized 7B on rented L4/A10
    "vendor-hard": (5.00, 25.00),       # claude-opus-4-8 published rate
}


def cost_dollars(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one served request, in dollars."""
    in_rate, out_rate = RATES[model]
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


CLASSIFY_SYSTEM = """You classify research queries by difficulty so a router can
send EASY ones to a small local model and HARD ones to a frontier model.

easy: a single fact lookup, a definition, a yes/no, or a simple extraction that a
      7B model handles reliably.
hard: multi-hop reasoning, synthesis across several sources, nuanced judgment,
      or anything where a wrong nuance is costly.

Output only a JSON object {"difficulty": "easy"|"hard", "reason": "<one line>"}."""


def classify_difficulty(query: str) -> dict:
    """Return {'difficulty': 'easy'|'hard', 'reason': ...} via a structured call."""
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "low",            # classification is cheap; low effort suffices
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "difficulty": {"type": "string",
                                       "enum": ["easy", "hard"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["difficulty", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        system=CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


def route_model(query: str) -> tuple[str, str]:
    """Return (model_name, reason). easy -> local-fast, hard -> vendor-hard."""
    # TODO 1: call classify_difficulty(query); map 'easy' -> "local-fast" and
    #   'hard' -> "vendor-hard". Return (model_name, the classifier's reason).
    raise NotImplementedError("implement route_model (TODO 1)")


# --------------------------------------------------------------------------- #
# Mocked served call. In the capstone this is a LiteLLM call to the routed model;
# here it returns deterministic token counts so the cost arithmetic is testable
# without a GPU or vendor answer spend. Hard queries cost more output tokens.
# --------------------------------------------------------------------------- #
@dataclass
class Served:
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


def serve_mock(model: str, query: str) -> Served:
    """Stand-in for a real LiteLLM call. Deterministic token counts."""
    input_tokens = 400 + len(query)                 # prompt + context
    output_tokens = 600 if model == "vendor-hard" else 200
    return Served(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost_dollars(model, input_tokens, output_tokens),
    )


def run_router(queries: list[str]) -> list[Served]:
    served = []
    print(f"{'QUERY':<46} {'ROUTE':<12} {'COST $':>9}")
    print("-" * 70)
    for q in queries:
        model, reason = route_model(q)
        s = serve_mock(model, q)
        served.append(s)
        print(f"{q[:44]:<46} {model:<12} {s.cost:>9.5f}")
    return served


def cost_report(served: list[Served]) -> None:
    """Print the median/p95 cost and the local-vs-vendor split — the deliverable-5
    skeleton. (median/p99 over 100 questions in the real capstone run.)"""
    # TODO 2: compute and print:
    #   - median cost across all requests (statistics.median)
    #   - p95 cost (sort, index at int(0.95 * (n-1)))
    #   - the fraction of requests served local vs vendor
    #   - the total cost, and the total cost of a vendor-ONLY baseline (route every
    #     query to vendor-hard) so you can report the savings from routing.
    raise NotImplementedError("implement cost_report (TODO 2)")


if __name__ == "__main__":
    queries = [
        "What is the net payment term in the agreement?",            # easy
        "What is the confidentiality duration after termination?",   # easy
        "Reconcile the indemnity cap against the liability-limit "
        "clause and explain which controls a data-breach claim.",    # hard
        "Summarize the renewal mechanics and flag any conflict with "
        "the termination-notice window across all three exhibits.",  # hard
        "What interest rate applies to late invoices?",              # easy
    ]
    served = run_router(queries)
    print()
    cost_report(served)

# --------------------------------------------------------------------------- #
# EXPECTED OUTPUT (shape — exact $ depend on the deterministic token counts;
# the easy/hard routing depends on the classifier but should match the comments)
# --------------------------------------------------------------------------- #
#
# QUERY                                          ROUTE        COST $
# ----------------------------------------------------------------------
# What is the net payment term in the agreemen   local-fast   0.00006
# What is the confidentiality duration after t   local-fast   0.00006
# Reconcile the indemnity cap against the liab   vendor-hard  0.01730
# Summarize the renewal mechanics and flag any   vendor-hard  0.01760
# What interest rate applies to late invoices?   local-fast   0.00006
#
# median cost: 0.00006   p95 cost: 0.01760
# routed local: 3/5 (60%)   vendor: 2/5 (40%)
# total: 0.03508   vendor-only baseline: 0.08650   savings: 59.4%
#
# (Your absolute numbers and the savings % will differ with the token counts and
# the classifier's calls; the SHAPE — easy local, hard vendor, real savings — is
# what you're proving. In the capstone, run this over the 100-question gold set.)
# --------------------------------------------------------------------------- #
