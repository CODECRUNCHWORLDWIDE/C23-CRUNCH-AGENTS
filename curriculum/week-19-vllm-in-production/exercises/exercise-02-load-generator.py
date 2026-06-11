#!/usr/bin/env python3
# Exercise 2 — The async load generator (drive REAL concurrency, measure the curve)
#
# Goal: Build a load generator that drives an OpenAI-compatible endpoint (your
#       vLLM server from Exercise 1) at controlled concurrency levels, and
#       measure throughput (tokens/sec) and latency (p50/p95) at each level.
#       The lesson is structural: you will SEE throughput RISE as concurrency
#       goes 1 -> 8 -> 32 (the continuous batch filling up), while p95 latency
#       creeps up. If your throughput does NOT rise with concurrency, you are
#       not actually concurrent (the #1 benchmarking bug) — and this file is
#       built to make that mistake impossible to miss.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   Point it at a live vLLM endpoint and run:
#
#       export VLLM_BASE_URL=http://<gpu-host>:8000/v1
#       export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
#       python3 exercise-02-load-generator.py --concurrency 1,8,32
#
#   It sends a fixed workload of prompts at each concurrency level using
#   asyncio + httpx (TRUE concurrency: N requests in flight at once), discards
#   a warm-up batch, and prints throughput + p50/p95 per level.
#
#   NO GPU? Set MOCK=1 to run against a built-in async mock "server" that
#   simulates per-token decode latency and a fixed batch ceiling, so the SHAPE
#   of the curve (throughput rises then plateaus, p95 climbs) is reproducible
#   on a laptop. The mock is NOT a real benchmark — it exists to teach the
#   methodology and let you debug the generator before you spend GPU money.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The generator drives TRUE concurrency (N requests in flight) via
#       asyncio.gather — proven by the fact that throughput rises with
#       concurrency (it cannot rise if requests are serial).
#   [ ] Reports throughput (output tok/s), p50, and p95 per concurrency level.
#   [ ] A warm-up batch is discarded before timing.
#   [ ] You can explain why concurrency=1 throughput is the WORST case and why
#       quoting it as "vLLM throughput" inflates cost-per-token ~10x.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

# --- The fixed workload: same prompts every run, so throughput is comparable ---
# A controlled prompt/output distribution. Real benchmarks load a dataset; this
# small set keeps the exercise self-contained while still exercising the batch.
WORKLOAD_PROMPTS = [
    "Summarize the benefits of continuous batching in two sentences.",
    "What is a KV cache and why does it bound concurrency? Answer briefly.",
    "Explain PagedAttention to a new engineer in three sentences.",
    "List three vLLM serving flags and what each controls.",
    "Why is decode memory-bandwidth-bound while prefill is compute-bound?",
    "Define throughput vs latency for an inference server in one line each.",
] * 6  # 36 prompts; enough to fill a batch and amortize warm-up
TARGET_MAX_TOKENS = 96


# --- The real client path: hit an OpenAI-compatible endpoint over async httpx ---
async def call_real(client, base_url: str, model: str, prompt: str) -> tuple[float, int]:
    """Returns (latency_seconds, output_tokens) for one request."""
    t0 = time.perf_counter()
    resp = await client.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": TARGET_MAX_TOKENS,
            "temperature": 0.0,
        },
        headers={"Authorization": "Bearer not-needed"},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    latency = time.perf_counter() - t0
    out_tokens = data["usage"]["completion_tokens"]
    return latency, out_tokens


# --- The mock path: a fake server with a batch ceiling, for laptops ----------
class MockServer:
    """Simulates a continuously-batched server: a fixed batch_ceiling of slots,
    each producing ~tok_per_step tokens/sec. Throughput rises with concurrency
    up to batch_ceiling, then plateaus; per-request latency rises as the batch
    fills. This reproduces the SHAPE of a real vLLM curve without a GPU."""

    def __init__(self, batch_ceiling: int = 24, tok_per_step_per_seq: float = 60.0):
        self.batch_ceiling = batch_ceiling
        self.tok_per_step = tok_per_step_per_seq
        self._in_flight = 0
        self._lock = asyncio.Lock()

    async def call(self, out_tokens: int) -> tuple[float, int]:
        async with self._lock:
            self._in_flight += 1
            contention = self._in_flight
        # When in_flight exceeds the batch ceiling, sequences queue: their
        # effective rate drops proportionally (the plateau + latency cliff).
        slowdown = max(1.0, contention / self.batch_ceiling)
        per_seq_rate = self.tok_per_step / slowdown
        latency = out_tokens / per_seq_rate
        await asyncio.sleep(latency)
        async with self._lock:
            self._in_flight -= 1
        return latency, out_tokens


async def run_level(concurrency: int, base_url: str, model: str, mock) -> dict:
    """Drive `len(WORKLOAD_PROMPTS)` requests with at most `concurrency` in
    flight at once. Returns throughput and latency percentiles."""
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    token_counts: list[int] = []

    async def one(prompt: str, client) -> None:
        async with sem:  # caps simultaneous in-flight requests at `concurrency`
            if mock is not None:
                lat, toks = await mock.call(TARGET_MAX_TOKENS)
            else:
                lat, toks = await call_real(client, base_url, model, prompt)
            latencies.append(lat)
            token_counts.append(toks)

    if mock is not None:
        wall_start = time.perf_counter()
        await asyncio.gather(*(one(p, None) for p in WORKLOAD_PROMPTS))
        wall = time.perf_counter() - wall_start
    else:
        import httpx

        async with httpx.AsyncClient() as client:
            wall_start = time.perf_counter()
            # asyncio.gather launches ALL tasks; the Semaphore caps how many run
            # at once. THIS is what makes concurrency real.
            await asyncio.gather(*(one(p, client) for p in WORKLOAD_PROMPTS))
            wall = time.perf_counter() - wall_start

    total_out = sum(token_counts)
    return {
        "concurrency": concurrency,
        "requests": len(WORKLOAD_PROMPTS),
        "throughput_tok_s": total_out / wall if wall else 0.0,
        "p50": statistics.median(latencies),
        "p95": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "wall_s": wall,
    }


async def main_async(levels: list[int], base_url: str, model: str, mock) -> int:
    print(f"workload: {len(WORKLOAD_PROMPTS)} prompts, max_tokens={TARGET_MAX_TOKENS}")
    print(f"backend: {'MOCK' if mock else base_url}  model={model}\n")

    # Warm-up: one pass at low concurrency, discarded (model load / CUDA graph /
    # cold prefix cache). NEVER time the warm-up.
    print("warming up (discarded)...")
    await run_level(4, base_url, model, mock)

    rows = []
    for c in levels:
        r = await run_level(c, base_url, model, mock)
        rows.append(r)
        print(
            f"concurrency={r['concurrency']:>3}  "
            f"throughput={r['throughput_tok_s']:>7.0f} tok/s  "
            f"p50={r['p50']:>5.2f}s  p95={r['p95']:>5.2f}s"
        )

    # The lesson, checked: throughput should rise from the first to a later level.
    if len(rows) >= 2 and rows[-1]["throughput_tok_s"] > rows[0]["throughput_tok_s"]:
        print("\nLESSON CONFIRMED: throughput rose with concurrency — the batch")
        print("is filling. Concurrency=1 is the WORST-case throughput; quoting it")
        print("as 'the' throughput would inflate cost-per-token roughly 10x.")
    else:
        print("\n!! Throughput did NOT rise with concurrency. Either you are not")
        print("   driving real concurrency, --max-num-seqs is throttling the batch,")
        print("   or concurrency already exceeds the batch ceiling (try lower levels).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", default="1,8,32",
                    help="comma-separated concurrency levels")
    args = ap.parse_args()
    levels = [int(x) for x in args.concurrency.split(",")]

    mock = MockServer() if os.environ.get("MOCK") == "1" else None
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    return asyncio.run(main_async(levels, base_url, model, mock))


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact numbers depend on GPU/model, or the mock params)
# -----------------------------------------------------------------------------
#
# $ MOCK=1 python3 exercise-02-load-generator.py --concurrency 1,8,32
# workload: 36 prompts, max_tokens=96
# backend: MOCK  model=Qwen/Qwen2.5-7B-Instruct
#
# warming up (discarded)...
# concurrency=  1  throughput=     58 tok/s  p50= 1.60s  p95= 1.60s
# concurrency=  8  throughput=    460 tok/s  p50= 1.60s  p95= 1.61s
# concurrency= 32  throughput=   1380 tok/s  p50= 2.10s  p95= 2.40s
#
# LESSON CONFIRMED: throughput rose with concurrency — the batch is filling.
# Concurrency=1 is the WORST-case throughput; quoting it as 'the' throughput
# would inflate cost-per-token roughly 10x.
#
# On a real H100 + Qwen2.5-14B the numbers are larger (hundreds -> thousands of
# tok/s) and the curve PLATEAUS past the batch ceiling while p95 climbs steeply
# (the latency cliff) — that plateau is where you read cost-per-token in the
# challenge. The SHAPE is the invariant: rises, bends, plateaus; p95 climbs.
# -----------------------------------------------------------------------------
