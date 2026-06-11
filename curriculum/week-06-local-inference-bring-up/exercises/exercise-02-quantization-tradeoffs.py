#!/usr/bin/env python3
# Exercise 2 — Quantization trade-offs (VRAM / decode speed / quality) and the curve
#
# Goal: Make "Q4_K_M sits at the knee of the quality/size curve" concrete and
#       quantitative. You will compute, for one fixed 7B model, the VRAM footprint
#       and the (memory-bandwidth-limited) decode speedup of each quant level, and
#       plot a perplexity-vs-size curve so you can SEE the knee where quality stops
#       being free. The lesson: quantization level is a tuned choice with numbers
#       behind it, not "I quantized it, seems fine."
#
# Estimated time: 45 minutes. Runnable.
#
# WHAT THIS DEPENDS ON (read before running)
#
#   * REQUIRED: `pip install numpy`. That's it. This exercise is pure arithmetic
#     plus an ASCII chart — NO GPU, NO model download. It runs anywhere, today,
#     so the quant intuition is exercisable before you touch a GPU.
#   * The decode-speed model here is the FIRST-PRINCIPLES one from Lecture 1 §2:
#     decode is memory-bandwidth-bound, so decode tok/s scales (to first order)
#     with 1 / bytes_read_per_token = 1 / model_size. We compute the RELATIVE
#     speedup that bandwidth bound predicts; real numbers track this closely on
#     single-stream decode and you confirm it in Exercise 3 / llama-bench.
#   * The perplexity numbers are REPRESENTATIVE values for a 7B model (the SHAPE
#     of the curve is what matters and is stable across models): perplexity barely
#     moves from FP16 down to ~Q4, then climbs sharply below Q4. Swap in your own
#     measured perplexities (llama.cpp `llama-perplexity`) for the stretch goal.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The script prints, per quant level: bytes/param, model size (GB), predicted
#       VRAM (weights + a KV-cache estimate), and the bandwidth-predicted decode
#       speedup vs FP16.
#   [ ] The perplexity-vs-size chart shows the KNEE: near-flat from FP16 to Q4_K_M,
#       then rising below it. You can point at Q4_K_M and say "this is the knee."
#   [ ] You can state the trade for Q4_K_M in one sentence with numbers
#       (e.g. "~3.4x smaller, ~3.3x faster decode, perplexity within ~2% of FP16").
#
# Expected output is at the bottom of the file.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- The fixed model: a 7B. Only the QUANT varies. --------------------------------
N_PARAMS = 7.0e9          # 7 billion parameters (Qwen2.5-7B / Llama-3.1-8B class)
N_LAYERS = 28             # representative for a 7B
HIDDEN = 3584             # representative hidden dim for a 7B
CONTEXT = 8192            # tokens of KV cache we budget for
BATCH = 1                 # single-stream; KV cache grows with batch (Lecture 2 §4)


@dataclass
class Quant:
    name: str
    bits_per_param: float   # effective bits/param (K-quants are not exactly N bits)
    perplexity: float       # representative wikitext-style perplexity for a 7B
    runs_on: str            # which engine family loads it


# Effective bits/param are approximate: K-quants carry some higher-precision parts,
# so e.g. Q4_K_M is ~4.8 effective bits, not exactly 4. Perplexities are
# representative; the SHAPE (knee at ~Q4) is the durable lesson.
QUANTS = [
    Quant("FP16",   16.0, 5.80, "everywhere (baseline)"),
    Quant("Q8_0",    8.5, 5.81, "GGUF (llama.cpp/Ollama)"),
    Quant("Q6_K",    6.6, 5.83, "GGUF"),
    Quant("Q5_K_M",  5.7, 5.86, "GGUF"),
    Quant("Q4_K_M",  4.8, 5.94, "GGUF  <- the knee"),
    Quant("AWQ-4",   4.2, 5.99, "vLLM (GPU 4-bit)"),
    Quant("Q3_K_M",  3.9, 6.32, "GGUF"),
    Quant("Q2_K",    2.6, 7.45, "GGUF (quality cliff)"),
]


def model_size_gb(bits_per_param: float) -> float:
    """Weights only, in GB. bits/8 = bytes/param."""
    return N_PARAMS * (bits_per_param / 8.0) / 1e9


def kv_cache_gb() -> float:
    """KV-cache VRAM (Lecture 2 §4): 2 (K&V) x layers x context x hidden x batch
    x 2 bytes (fp16 cache). Independent of weight quant — a fixed add-on you must
    not forget, or you'll OOM (Lecture 2 §1.3)."""
    bytes_ = 2 * N_LAYERS * CONTEXT * HIDDEN * BATCH * 2
    return bytes_ / 1e9


def decode_speedup_vs_fp16(bits_per_param: float) -> float:
    """Decode is memory-bandwidth-bound: tok/s ~ 1 / bytes_read_per_token, and the
    bytes read per token are dominated by the weights. So speedup vs FP16 is
    (FP16 size) / (this size) to first order (Lecture 1 §2)."""
    return model_size_gb(16.0) / model_size_gb(bits_per_param)


def main() -> int:
    kv = kv_cache_gb()
    fp16_ppl = QUANTS[0].perplexity

    print(f"Fixed model: 7B, {N_LAYERS} layers, hidden={HIDDEN}, "
          f"context={CONTEXT}, batch={BATCH}")
    print(f"KV-cache VRAM (same for every quant): {kv:.2f} GB  "
          f"(this is the cost people forget -> OOM)\n")

    header = (f"{'quant':>8} | {'bits':>4} | {'size GB':>7} | {'VRAM GB':>7} | "
              f"{'decode x':>8} | {'ppl':>5} | {'ppl Δ%':>6} | runs on")
    print(header)
    print("-" * len(header))

    for q in QUANTS:
        size = model_size_gb(q.bits_per_param)
        vram = size + kv                      # weights + KV cache (Lecture 2 §4)
        speedup = decode_speedup_vs_fp16(q.bits_per_param)
        ppl_delta = 100.0 * (q.perplexity - fp16_ppl) / fp16_ppl
        print(f"{q.name:>8} | {q.bits_per_param:>4.1f} | {size:>7.2f} | "
              f"{vram:>7.2f} | {speedup:>7.2f}x | {q.perplexity:>5.2f} | "
              f"{ppl_delta:>5.1f}% | {q.runs_on}")

    # --- The quality/size curve: perplexity (y) vs size (x), ASCII ----------------
    print("\nperplexity vs model size  (the KNEE is where quality stops being free)")
    sizes = [model_size_gb(q.bits_per_param) for q in QUANTS]
    ppls = [q.perplexity for q in QUANTS]
    pmin, pmax = min(ppls), max(ppls)
    smin, smax = min(sizes), max(sizes)
    WIDTH, HEIGHT = 52, 12
    grid = [[" "] * WIDTH for _ in range(HEIGHT)]
    for q, s, p in zip(QUANTS, sizes, ppls):
        col = int((s - smin) / (smax - smin) * (WIDTH - 1))
        # higher perplexity -> higher up the chart (row 0 is top)
        row = int((pmax - p) / (pmax - pmin) * (HEIGHT - 1))
        grid[row][col] = "*"
    for r, line in enumerate(grid):
        # annotate the top (worst ppl) and bottom (best ppl) rows
        tag = ""
        if r == 0:
            tag = f"  <- ppl {pmax:.2f} (worst: Q2_K, the cliff)"
        if r == HEIGHT - 1:
            tag = f"  <- ppl {pmin:.2f} (best: FP16)"
        print("  " + "".join(line) + tag)
    print("  " + "small <" + "-" * (WIDTH - 14) + "> large  (model size GB)")

    # --- Name the knee ------------------------------------------------------------
    knee = next(q for q in QUANTS if q.name == "Q4_K_M")
    knee_size = model_size_gb(knee.bits_per_param)
    knee_speed = decode_speedup_vs_fp16(knee.bits_per_param)
    knee_dppl = 100.0 * (knee.perplexity - fp16_ppl) / fp16_ppl
    print(f"\nKNEE = Q4_K_M: {model_size_gb(16.0)/knee_size:.2f}x smaller, "
          f"~{knee_speed:.2f}x faster decode, perplexity +{knee_dppl:.1f}% vs FP16.")
    print("DECISION: start at Q4_K_M. Go UP (Q5/Q8/FP16) only if you MEASURE a "
          "quality regression you can't accept. Below Q4 you pay real quality for "
          "marginal size (the cliff). That is the quant choice, defended with "
          "numbers — not 'I quantized it, seems fine.'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; representative perplexities — swap in your measured ones)
# -----------------------------------------------------------------------------
#
# Fixed model: 7B, 28 layers, hidden=3584, context=8192, batch=1
# KV-cache VRAM (same for every quant): 3.29 GB  (this is the cost people forget...)
#
#    quant | bits | size GB | VRAM GB | decode x |   ppl | ppl Δ% | runs on
# -----------------------------------------------------------------------------
#     FP16 | 16.0 |   14.00 |   17.29 |    1.00x |  5.80 |   0.0% | everywhere (baseline)
#     Q8_0 |  8.5 |    7.44 |   10.73 |    1.88x |  5.81 |   0.2% | GGUF (llama.cpp/Ollama)
#     Q6_K |  6.6 |    5.78 |    9.06 |    2.42x |  5.83 |   0.5% | GGUF
#   Q5_K_M |  5.7 |    4.99 |    8.27 |    2.81x |  5.86 |   1.0% | GGUF
#   Q4_K_M |  4.8 |    4.20 |    7.49 |    3.33x |  5.94 |   2.4% | GGUF  <- the knee
#    AWQ-4 |  4.2 |    3.68 |    6.96 |    3.81x |  5.99 |   3.3% | vLLM (GPU 4-bit)
#   Q3_K_M |  3.9 |    3.41 |    6.70 |    4.10x |  6.32 |   9.0% | GGUF
#     Q2_K |  2.6 |    2.28 |    5.56 |    6.15x |  7.45 |  28.4% | GGUF (quality cliff)
#
# perplexity vs model size  (the KNEE is where quality stops being free)
#   *                                                     <- ppl 7.45 (worst: Q2_K...)
#       *
#         (... points rising as size shrinks below Q4 ...)
#                                              * * * *  *  <- ppl 5.80 (best: FP16)
#   small <--------------------------------------> large  (model size GB)
#
# KNEE = Q4_K_M: 3.33x smaller, ~3.33x faster decode, perplexity +2.4% vs FP16.
# DECISION: start at Q4_K_M. ...
#
# READ THE TABLE: from FP16 down to Q4_K_M, perplexity moves ~2% while size drops
# 3.3x and decode speeds up 3.3x -- nearly free. Below Q4 (Q3, Q2) perplexity jumps
# 9%, 28% -- you're paying real quality for little size. That is WHY Q4_K_M is the
# default: it's the knee. And note VRAM is weights + 3.29 GB of KV cache that's the
# SAME at every quant -- the cost people forget when they say "the 4-bit model is
# only 4 GB" (it's 7.5 GB once the cache is counted). Raise BATCH or CONTEXT and
# watch that 3.29 GB grow (Lecture 2 §4) -- that's the OOM nobody predicted.
# -----------------------------------------------------------------------------
