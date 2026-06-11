#!/usr/bin/env python3
# Exercise 2 — The uniform client (one interface over a hosted model and a local model)
#
# Goal: Wrap two completely different transports — the Anthropic SDK (hosted
#       frontier model) and the Ollama HTTP API (local open-weights model) —
#       behind ONE interface: complete(prompt) -> Completion. Every backend
#       returns the same dataclass: text, tokens_in, tokens_out, latency_s,
#       and a cost estimate. This is the seam the whole course is built on:
#       swap the model, not the architecture.
#
# Estimated time: 45 minutes. Runnable.
#
# WHY THIS MATTERS
#
#   Lecture 1 said an LLM is a function `prompt -> text` with measurable
#   token counts and latency, regardless of who serves it. This file makes
#   that literal. Once complete() is uniform, the mini-project (llmpick) is
#   just "call complete() on N backends in parallel and compare the numbers."
#
# HOW TO USE THIS FILE
#
#   Standalone. Activate a venv with `anthropic` and `httpx`, have Ollama
#   running, then:
#
#       pip install anthropic httpx
#       ollama pull qwen2.5:7b
#       export ANTHROPIC_API_KEY=sk-ant-...      # optional; falls back gracefully
#       python3 exercise-02-uniform-client.py
#
#   It runs the SAME prompt through both backends and prints a comparison
#   table. If ANTHROPIC_API_KEY is unset or the network is down, the Anthropic
#   row prints "unavailable" and the Ollama row still carries the lesson.
#
# THE TODOs
#
#   Three small gaps are marked "# TODO N:". Fill them to complete the
#   uniform interface. Everything else is done for you.
#
# ACCEPTANCE CRITERIA
#
#   [ ] Both backends return a Completion with the SAME fields populated.
#   [ ] tokens_in / tokens_out are the REAL counts from each backend's own
#       tokenizer (not estimated, not from a different model's tokenizer).
#   [ ] The Anthropic and Ollama token counts for the SAME prompt DIFFER —
#       and you can say why (different tokenizers).
#   [ ] Cost is computed from real token counts and a per-model price table.
#   [ ] With no API key, the program still runs and the Ollama row is complete.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

try:
    import anthropic
except ImportError:  # the local path still works without the SDK installed
    anthropic = None  # type: ignore


PROMPT = (
    "You are a systems engineer. In exactly two sentences, explain what a "
    "KV cache is and why it makes token generation after the first token fast."
)

# Per-MTok pricing (USD per 1,000,000 tokens). Hosted prices are list prices;
# the local model's marginal token cost is 0 (you pay for the box, not tokens).
# Keep these current against the vendor's pricing page.
PRICES = {
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-opus-4-8": {"in": 5.00, "out": 25.00},
    "ollama-local": {"in": 0.0, "out": 0.0},
}


@dataclass
class Completion:
    """The uniform result every backend returns. THIS is the interface."""

    backend: str
    model: str
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float

    def cost_usd(self) -> float:
        """Cost from real token counts and the price table (per-MTok -> per-token)."""
        price = PRICES.get(self.model, PRICES["ollama-local"])
        # TODO 1: compute cost. Prices are per 1,000,000 tokens. Multiply
        #         tokens_in by the input price and tokens_out by the output
        #         price, each divided by 1_000_000, and sum them.
        #         Replace the line below.
        return 0.0


def complete_anthropic(prompt: str, model: str = "claude-haiku-4-5") -> Completion:
    """Hosted frontier backend via the Anthropic SDK."""
    if anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("anthropic SDK or ANTHROPIC_API_KEY not available")

    client = anthropic.Anthropic()
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0
    text = next((b.text for b in msg.content if b.type == "text"), "")
    # msg.usage gives the REAL token counts from the model's own tokenizer.
    return Completion(
        backend="anthropic",
        model=model,
        text=text,
        tokens_in=msg.usage.input_tokens,
        tokens_out=msg.usage.output_tokens,
        latency_s=latency,
    )


def complete_ollama(prompt: str, model: str = "qwen2.5:7b") -> Completion:
    """Local open-weights backend via the Ollama HTTP API."""
    t0 = time.perf_counter()
    r = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180.0,
    )
    r.raise_for_status()
    data = r.json()
    latency = time.perf_counter() - t0
    # Ollama reports prompt_eval_count (prefill tokens = tokens_in) and
    # eval_count (decode tokens = tokens_out), from ITS OWN tokenizer.
    return Completion(
        backend="ollama",
        model="ollama-local",   # for pricing; the real tag is `model`
        # TODO 2: fill text, tokens_in, tokens_out from `data`.
        #   text       <- data["response"]
        #   tokens_in  <- data["prompt_eval_count"]
        #   tokens_out <- data["eval_count"]
        text="",
        tokens_in=0,
        tokens_out=0,
        latency_s=latency,
    )


def run_all(prompt: str) -> list[Completion]:
    """Call every available backend through the SAME interface."""
    results: list[Completion] = []
    backends = [
        ("anthropic", lambda: complete_anthropic(prompt)),
        ("ollama", lambda: complete_ollama(prompt)),
    ]
    for name, fn in backends:
        try:
            results.append(fn())
        except Exception as e:  # report, don't crash — a backend may be down
            print(f"[{name}] unavailable: {e}")
    return results


def print_table(results: list[Completion]) -> None:
    print("\n" + "=" * 78)
    print(f"{'BACKEND':<10} {'MODEL':<18} {'IN':>5} {'OUT':>5} "
          f"{'LATENCY':>9} {'COST(USD)':>11}")
    print("-" * 78)
    for c in results:
        print(f"{c.backend:<10} {c.model:<18} {c.tokens_in:>5} {c.tokens_out:>5} "
              f"{c.latency_s:>8.2f}s {c.cost_usd():>11.6f}")
    print("=" * 78)

    # TODO 3: if BOTH an anthropic and an ollama result exist, print a one-line
    #         observation comparing their tokens_in for the SAME prompt, e.g.:
    #         "tokens_in differ (anthropic=NN, ollama=MM) — different tokenizers."
    #         This is the load-bearing lesson; make the program state it.


def main() -> None:
    print(f"Prompt:\n  {PROMPT}\n")
    results = run_all(PROMPT)
    if not results:
        print("No backend available. Start Ollama and/or set ANTHROPIC_API_KEY.")
        return
    for c in results:
        print(f"\n[{c.backend}] {c.text.strip()[:240]}")
    print_table(results)


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Expected output (both backends available; your numbers will differ)
# -----------------------------------------------------------------------------
#
# Prompt:
#   You are a systems engineer. In exactly two sentences, explain ...
#
# [anthropic] A KV cache stores the attention keys and values computed for ...
# [ollama] The KV cache saves the key and value tensors from earlier tokens ...
#
# ==============================================================================
# BACKEND    MODEL                 IN   OUT   LATENCY   COST(USD)
# ------------------------------------------------------------------------------
# anthropic  claude-haiku-4-5      48    61      1.04s    0.000353
# ollama     ollama-local         44    66      2.71s    0.000000
# ==============================================================================
# tokens_in differ (anthropic=48, ollama=44) — different tokenizers.
#
# THE LESSON: one interface, two transports. The token counts differ because
# the two model families tokenize the SAME text differently — which is exactly
# why you must never estimate one model's cost with another's tokenizer.
#
# Expected output (no API key)
# -----------------------------------------------------------------------------
#
# [anthropic] unavailable: anthropic SDK or ANTHROPIC_API_KEY not available
# ... (ollama row complete; the lesson still lands on the local path)
# -----------------------------------------------------------------------------
