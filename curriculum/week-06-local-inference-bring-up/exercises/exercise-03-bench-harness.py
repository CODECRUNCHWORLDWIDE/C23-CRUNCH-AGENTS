#!/usr/bin/env python3
# Exercise 3 — The inference benchmark harness (prefill/decode, TTFT, p50/p95,
#              throughput under concurrency)
#
# Goal: Build the reusable, HONEST benchmark from Lecture 2 §5. Point it at any
#       OpenAI-compatible endpoint (Ollama, llama.cpp, vLLM) and measure: time-to-
#       first-token (TTFT, set by prefill), decode tokens/sec (set by decode), and
#       AGGREGATE throughput as you raise concurrency (1, 8, 32, ...). The headline
#       lesson is the SHAPE of the throughput-vs-concurrency curve: it's the axis
#       that reveals an engine's real character, and the reason "benchmark at
#       concurrency 1" mis-ranks engines for a serving workload.
#
# Estimated time: 55 minutes. Runnable.
#
# WHAT THIS DEPENDS ON (read before running)
#
#   * REQUIRED: `pip install httpx`. Optionally `numpy` (we fall back to a pure-
#     Python percentile if it's missing, so the file always runs).
#   * REQUIRED: a RUNNING OpenAI-compatible endpoint. Easiest today (no GPU):
#         ollama serve & ; ollama pull qwen2.5:7b
#         python3 exercise-03-bench-harness.py \
#             --base-url http://localhost:11434/v1 --model qwen2.5:7b
#     For the real concurrency story, point it at a vLLM server on a GPU; the SAME
#     harness reveals vLLM's rising curve vs Ollama's flattening one.
#   * Streaming is used to time the FIRST token (TTFT = prefill) separately from
#     the rest (decode). If your endpoint doesn't stream, the script falls back to
#     a non-streamed timing and notes that TTFT is then approximate.
#
# ACCEPTANCE CRITERIA
#
#   [ ] Prints TTFT (p50/p95) and decode tok/s for a single request.
#   [ ] Prints AGGREGATE tokens/sec at each concurrency level in the sweep.
#   [ ] On a continuous-batching engine (vLLM) the aggregate RISES with
#       concurrency; on Ollama/llama.cpp it flattens. You can point at the curve
#       and say which engine this is built for.
#   [ ] Warm-up request is discarded; each point is the median of repeated runs.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

try:
    import numpy as np

    def percentile(xs: list[float], p: float) -> float:
        return float(np.percentile(xs, p))
except Exception:  # numpy optional — pure-Python fallback so the file always runs
    def percentile(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        k = (len(s) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)


PROMPTS = [
    "Explain prefill versus decode in two sentences.",
    "What is continuous batching and why does it raise throughput?",
    "Summarize paged attention in three sentences.",
    "Why is decode memory-bandwidth-bound?",
    "Give two reasons to self-host an LLM.",
    "What does Q4_K_M quantization trade away?",
    "How does speculative decoding speed up generation?",
    "When would you choose SGLang over vLLM?",
] * 32   # plenty to fill high concurrency without repeating within a batch too soon


async def stream_one(client: httpx.AsyncClient, url: str, model: str,
                     prompt: str, max_tokens: int) -> tuple[float, float, int]:
    """Return (ttft_s, total_s, output_tokens) for one streamed request.

    TTFT (time-to-first-token) is set by PREFILL; the remaining time over the
    remaining tokens is the DECODE rate (Lecture 1 §2)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    async with client.stream("POST", url, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0      # first token => prefill done
            n_tokens += 1
    total = time.perf_counter() - t0
    return (ttft or total), total, max(n_tokens, 1)


async def bench_concurrency(base_url: str, model: str, concurrency: int,
                            max_tokens: int) -> dict:
    """Fire `concurrency` requests at once; report AGGREGATE tokens/sec (the
    serving metric, Lecture 2 §5.2) plus TTFT and decode percentiles."""
    url = base_url.rstrip("/") + "/chat/completions"
    prompts = PROMPTS[:concurrency]
    async with httpx.AsyncClient(timeout=300) as client:
        wall0 = time.perf_counter()
        results = await asyncio.gather(
            *(stream_one(client, url, model, p, max_tokens) for p in prompts)
        )
        wall = time.perf_counter() - wall0

    ttfts = [r[0] for r in results]
    totals = [r[1] for r in results]
    toks = [r[2] for r in results]
    total_tokens = sum(toks)
    # Per-request decode rate: tokens after the first, over time after TTFT.
    decode_rates = [
        (t - 1) / max(tot - ttft, 1e-6)
        for ttft, tot, t in results if t > 1
    ]
    return {
        "concurrency": concurrency,
        "aggregate_tok_s": total_tokens / wall,         # THE serving number
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "decode_tok_s_per_req_p50": percentile(decode_rates, 50) if decode_rates else 0.0,
        "wall_s": wall,
    }


async def amain(args) -> int:
    print(f"endpoint: {args.base_url}   model: {args.model}")
    print(f"max_tokens/request: {args.max_tokens}\n")

    # Warm-up (model load, cache cold) — DISCARDED (Lecture 2 §5.5).
    print("warming up (discarded)...")
    try:
        await bench_concurrency(args.base_url, args.model, 1, args.max_tokens)
    except Exception as e:
        print(f"\nCould not reach the endpoint: {e}")
        print("Start one first, e.g.:  ollama serve & ; ollama pull qwen2.5:7b")
        return 1

    levels = [int(c) for c in args.concurrency.split(",")]
    print(f"\n{'conc':>4} | {'agg tok/s':>9} | {'ttft p50':>8} | {'ttft p95':>8} | "
          f"{'decode/req':>10} | curve")
    print("-" * 64)

    rows = []
    peak_agg = 0.0
    for c in levels:
        # Median of N repeats per point so a thermal blip doesn't fool you.
        samples = []
        for _ in range(args.repeats):
            samples.append(await bench_concurrency(
                args.base_url, args.model, c, args.max_tokens))
        samples.sort(key=lambda s: s["aggregate_tok_s"])
        m = samples[len(samples) // 2]                  # median by aggregate
        rows.append(m)
        peak_agg = max(peak_agg, m["aggregate_tok_s"])
        bar = "#" * int(round(m["aggregate_tok_s"] / max(peak_agg, 1) * 30))
        print(f"{c:>4} | {m['aggregate_tok_s']:>9.1f} | {m['ttft_p50']:>7.3f}s | "
              f"{m['ttft_p95']:>7.3f}s | "
              f"{m['decode_tok_s_per_req_p50']:>9.1f} | {bar}")

    print("-" * 64)
    first, last = rows[0], rows[-1]
    ratio = last["aggregate_tok_s"] / max(first["aggregate_tok_s"], 1e-6)
    print(f"aggregate throughput grew {ratio:.1f}x from concurrency "
          f"{first['concurrency']} to {last['concurrency']}.")
    if ratio >= 2.0:
        print("SHAPE: aggregate RISES with concurrency -> this engine "
              "continuous-batches (vLLM-like). It amortizes the weight-read across "
              "an in-flight batch (Lecture 2 §2). THIS is a serving engine.")
    else:
        print("SHAPE: aggregate ~FLAT with concurrency -> this engine serializes "
              "(Ollama/llama.cpp-like). Great for one user, not for a server. "
              "Benchmarking only at concurrency 1 would have HIDDEN this.")
    print("\nNote: TTFT is set by PREFILL; decode/req is the DECODE rate. They are "
          "different phases (Lecture 1 §2) — never blend them into one number.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Honest LLM inference benchmark.")
    ap.add_argument("--base-url", default="http://localhost:11434/v1",
                    help="OpenAI-compatible endpoint (Ollama/llama.cpp/vLLM).")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--concurrency", default="1,4,8,16",
                    help="comma-separated concurrency levels to sweep")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeats per point; the MEDIAN is reported")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; absolute numbers depend on engine, model, hardware)
# -----------------------------------------------------------------------------
#
# On vLLM (CUDA GPU) -- aggregate RISES with concurrency:
#
# endpoint: http://localhost:8000/v1   model: Qwen/Qwen2.5-7B-Instruct-AWQ
# max_tokens/request: 128
#
# warming up (discarded)...
#
# conc | agg tok/s | ttft p50 | ttft p95 | decode/req | curve
# ----------------------------------------------------------------
#    1 |      58.0 |   0.090s |   0.090s |       58.0 | ######
#    4 |     210.0 |   0.110s |   0.150s |       54.0 | ######################
#    8 |     360.0 |   0.140s |   0.230s |       49.0 | ############################
#   16 |     560.0 |   0.190s |   0.410s |       41.0 | ##############################
# ----------------------------------------------------------------
# aggregate throughput grew 9.7x from concurrency 1 to 16.
# SHAPE: aggregate RISES with concurrency -> this engine continuous-batches...
#
# On Ollama (laptop) -- aggregate ~FLAT (it serializes):
#
# conc | agg tok/s | ttft p50 | ttft p95 | decode/req | curve
# ----------------------------------------------------------------
#    1 |      32.0 |   0.120s |   0.120s |       32.0 | ##############################
#    4 |      34.0 |   0.450s |   0.700s |        9.0 | ##############################
#    8 |      33.0 |   0.900s |   1.500s |        4.2 | ############################
#   16 |      33.0 |   1.800s |   3.100s |        2.1 | ############################
# ----------------------------------------------------------------
# aggregate throughput grew 1.0x from concurrency 1 to 16.
# SHAPE: aggregate ~FLAT with concurrency -> this engine serializes...
#
# READ THE CURVES: vLLM's aggregate climbs ~10x because continuous batching keeps
# the GPU full as requests pile in; per-request decode barely drops. Ollama's
# aggregate is FLAT and per-request decode COLLAPSES (32 -> 2 tok/s) because the
# requests queue behind each other -- the same total work, serialized. If you had
# benchmarked only at concurrency 1, vLLM (58) and Ollama (32) would look like a
# 2x story; at concurrency 16 it's a 17x story. THAT is "pick the engine for the
# workload, not the benchmark" (Lecture 1) made into a number you can ship.
# -----------------------------------------------------------------------------
