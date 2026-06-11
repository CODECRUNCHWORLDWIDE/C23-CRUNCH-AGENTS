#!/usr/bin/env python3
# Exercise 3 — LoRA/QLoRA memory budget + reading the loss curve
#
# Goal: Make two abstract things concrete. (1) The MEMORY MATH that explains why
#       a 7B fits in 24 GB under QLoRA but not under full fine-tuning — you
#       compute the budget for full FT, LoRA (16-bit base), and QLoRA (4-bit
#       base) and SEE which fit. (2) The three LOSS-CURVE SHAPES — learning,
#       memorizing/overfitting, diverging — simulated so you can recognize a real
#       one when your training run prints it.
#
# Estimated time: 50 minutes. Runnable. CPU-only, no ML deps (pure arithmetic +
# a numpy-free loss simulation).
#
# HOW TO USE THIS FILE
#
#       python3 exercise-03-lora-memory-and-loss.py
#
#   Part A prints a memory table for a 7B model under three regimes and marks
#   which fit on a 24 GB GPU. Part B prints three ASCII loss curves and labels
#   what each one means and what to do about it.
#
# ACCEPTANCE CRITERIA
#
#   [ ] Part A shows full fine-tuning does NOT fit in 24 GB, LoRA (16-bit base)
#       is borderline, and QLoRA (4-bit base) fits comfortably — with the
#       arithmetic, not just the verdict.
#   [ ] Part B prints three distinct curves (learning / memorizing / diverging)
#       and you can state, for each, what it means and the fix.
#   [ ] You can explain WHY QLoRA's 4-bit frozen base barely hurts quality (the
#       base isn't trained; only the higher-precision adapter learns).
#
# Expected output (shape) is at the bottom of the file.

from __future__ import annotations

import math

GB = 1024 ** 3
PARAMS_7B = 7.6e9            # ~7.6B params for Qwen2.5-7B
ADAPTER_FRACTION = 0.005    # LoRA trains ~0.5% of params at rank 16


# =============================================================================
# PART A — the memory budget
# =============================================================================
def bytes_to_gb(b: float) -> float:
    return b / GB


def memory_report() -> bool:
    print("=" * 64)
    print("PART A — Can a 7B (Qwen2.5-7B) be fine-tuned on a 24 GB GPU?")
    print("=" * 64)
    budget = 24.0

    # --- Full fine-tuning: 16-bit weights + Adam optimizer (2 state tensors) +
    #     gradients, all at full param count. The optimizer state is the killer.
    w = PARAMS_7B * 2                       # 16-bit weights: 2 bytes/param
    grads = PARAMS_7B * 2                    # gradients: 2 bytes/param
    optim = PARAMS_7B * 2 * 2 * 2            # Adam: 2 states, fp32 (4 bytes) each ~ approx
    full = w + grads + optim
    print(f"\nFull fine-tuning:")
    print(f"  weights (16-bit)        {bytes_to_gb(w):6.1f} GB")
    print(f"  gradients               {bytes_to_gb(grads):6.1f} GB")
    print(f"  Adam optimizer state    {bytes_to_gb(optim):6.1f} GB")
    print(f"  --------------------------------")
    print(f"  total (approx)          {bytes_to_gb(full):6.1f} GB   "
          f"{'FITS' if bytes_to_gb(full) <= budget else 'DOES NOT FIT in 24 GB'}")

    # --- LoRA on a 16-bit frozen base: full base stays in memory, but only the
    #     tiny adapter has gradients + optimizer state.
    base16 = PARAMS_7B * 2                   # frozen base, 16-bit
    adapter_params = PARAMS_7B * ADAPTER_FRACTION
    lora_grads = adapter_params * 2
    lora_optim = adapter_params * 4 * 2      # Adam on the adapter only
    lora = base16 + lora_grads + lora_optim
    print(f"\nLoRA (16-bit frozen base):")
    print(f"  frozen base (16-bit)    {bytes_to_gb(base16):6.1f} GB")
    print(f"  adapter grads + optim   {bytes_to_gb(lora_grads + lora_optim):6.1f} GB")
    print(f"  --------------------------------")
    print(f"  total (approx)          {bytes_to_gb(lora):6.1f} GB   "
          f"{'FITS' if bytes_to_gb(lora) <= budget else 'BORDERLINE / TIGHT in 24 GB'}")

    # --- QLoRA: 4-bit frozen base (the enabler) + tiny adapter. The base shrinks
    #     ~4x; the adapter is unchanged.
    base4 = PARAMS_7B * 0.5                  # 4-bit ~ 0.5 bytes/param
    qlora = base4 + lora_grads + lora_optim
    print(f"\nQLoRA (4-bit frozen base):")
    print(f"  frozen base (4-bit)     {bytes_to_gb(base4):6.1f} GB   <- the enabler")
    print(f"  adapter grads + optim   {bytes_to_gb(lora_grads + lora_optim):6.1f} GB")
    print(f"  --------------------------------")
    print(f"  total (approx)          {bytes_to_gb(qlora):6.1f} GB   "
          f"{'FITS comfortably' if bytes_to_gb(qlora) <= budget else 'DOES NOT FIT'}")

    print("\nWHY QLoRA's 4-bit base barely hurts quality: the base is FROZEN — it's")
    print("never updated — so quantizing it loses a little precision in weights")
    print("that aren't learning anyway. The ADAPTER (kept higher precision) learns")
    print("around that. That's the QLoRA paper's headline result.")

    fits_correct = (bytes_to_gb(full) > budget and bytes_to_gb(qlora) <= budget)
    return fits_correct


# =============================================================================
# PART B — reading the loss curve
# =============================================================================
def _plot(values: list[float], label: str, lo: float = 0.0, hi: float = 3.0) -> None:
    print(f"\n  {label}")
    width = 40
    for i, v in enumerate(values):
        filled = int((v - lo) / (hi - lo) * width)
        filled = max(0, min(width, filled))
        bar = "#" * filled
        print(f"   step {i:2d} |{bar:<{width}}| {v:.2f}")


def loss_curves() -> None:
    print("\n" + "=" * 64)
    print("PART B — the three loss-curve shapes")
    print("=" * 64)
    steps = list(range(12))

    # Learning: smooth descent that flattens. THE GOOD SHAPE.
    learning = [2.6 * math.exp(-0.35 * s) + 0.35 for s in steps]
    _plot(learning, "LEARNING (good): smooth descent, then flattens at a floor.")
    print("   -> The model converged. This is what you want. Stop near the flat.")

    # Memorizing/overfitting: training loss keeps dropping toward ~0.
    memorizing = [2.6 * math.exp(-0.5 * s) for s in steps]
    _plot(memorizing, "MEMORIZING (overfit): training loss dives toward ~0.")
    print("   -> Training loss looks 'great' but held-out loss would be RISING.")
    print("      Fix: fewer epochs, more data, or lower rank. You ONLY catch this")
    print("      with a held-out eval — training loss alone lies.")

    # Diverging: loss spikes / oscillates.
    diverging = [2.6 * math.exp(-0.3 * s) + 0.4 + (0.9 if s in (5, 8, 11) else 0)
                 + (s * 0.08 if s > 6 else 0) for s in steps]
    _plot(diverging, "DIVERGING (bad): loss spikes / climbs / oscillates.")
    print("   -> Learning rate too high (or data/template broken). Fix: lower LR;")
    print("      if it persists, check the chat template and the data.")


def main() -> int:
    a_ok = memory_report()
    loss_curves()
    print("\n" + "=" * 64)
    if a_ok:
        print("PASS: you computed why full FT blows past 24 GB while QLoRA fits, and")
        print("you can name the three loss-curve shapes and their fixes. Now you can")
        print("read the real loss curve your Unsloth run prints in the challenge.")
        return 0
    print("CHECK: the memory arithmetic should show full FT > 24 GB and QLoRA <<")
    print("24 GB. Re-read Part A's optimizer-state line — that's what kills full FT.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact GB depend on the approximations above)
# -----------------------------------------------------------------------------
#
# PART A — Can a 7B be fine-tuned on a 24 GB GPU?
#   Full fine-tuning:        total (approx) ~89 GB   DOES NOT FIT in 24 GB
#   LoRA (16-bit base):      total (approx) ~15 GB   FITS (but base alone is big)
#   QLoRA (4-bit base):      total (approx) ~4 GB    FITS comfortably  <- the enabler
#
#   WHY QLoRA's 4-bit base barely hurts quality: the base is FROZEN ...
#
# PART B — the three loss-curve shapes
#   LEARNING (good): smooth descent, then flattens ...
#   MEMORIZING (overfit): training loss dives toward ~0 ...  (held-out would RISE)
#   DIVERGING (bad): loss spikes / climbs / oscillates ...   (LR too high)
#
# PASS: you computed why full FT blows past 24 GB while QLoRA fits, ...
#
# NOTE: the GB numbers are approximations to build INTUITION, not a precise VRAM
# predictor (real usage includes activations, which depend on sequence length and
# batch size). The SHAPE of the conclusion is what's exact: full FT optimizer
# state dwarfs everything, and QLoRA's 4-bit frozen base is the single change
# that brings a 7B into 24 GB. That's the memory story of the whole week.
# -----------------------------------------------------------------------------
