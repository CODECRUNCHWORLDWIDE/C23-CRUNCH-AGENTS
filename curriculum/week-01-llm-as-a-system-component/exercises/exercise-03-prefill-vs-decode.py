#!/usr/bin/env python3
# Exercise 3 — Prefill vs decode (measure TTFT and TPOT, and explain the gap)
#
# Goal: Measure, against a LOCAL model, the two latency numbers that are
#       actually different physical things:
#         * TTFT  (time-to-first-token)  -- dominated by PREFILL (compute-bound)
#         * TPOT  (time-per-output-token) -- dominated by DECODE  (bandwidth-bound)
#       Then run the SAME generation under a SHORT prompt and a LONG prompt and
#       watch TTFT balloon with prompt length while TPOT stays roughly flat.
#       That single comparison IS Lecture 1 §4.
#
# Estimated time: 45 minutes. Runnable.
#
# WHY LOCAL, NOT HOSTED
#
#   Ollama returns the prefill duration (prompt_eval_duration) and the decode
#   duration (eval_duration) separately and honestly, so you can SEE the two
#   phases. Hosted APIs hide this behind one wall-clock number. To learn the
#   distinction you measure it where it's visible.
#
# HOW TO USE THIS FILE
#
#       ollama pull qwen2.5:7b
#       python3 exercise-03-prefill-vs-decode.py
#
#   It streams a response so we can timestamp the FIRST token (real TTFT),
#   then reads Ollama's reported prefill/decode durations for ground truth.
#   It does this for a short prompt and a long prompt and prints both.
#
# THE TODOs
#
#   Two gaps are marked "# TODO N:". Fill them to compute TTFT and TPOT.
#
# ACCEPTANCE CRITERIA
#
#   [ ] You print a measured TTFT (wall-clock to first streamed token) AND
#       Ollama's reported prefill/decode durations for both prompts.
#   [ ] TTFT for the LONG prompt is clearly larger than for the SHORT prompt.
#   [ ] TPOT (ms/token) is roughly SIMILAR across the two prompts.
#   [ ] You can state, in one sentence, why: prefill scales with prompt length
#       (longer prompt -> slower first token); decode is per-token and roughly
#       prompt-length-independent.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

SHORT_PROMPT = "Explain what a KV cache is in one sentence."

# A long prompt: same task, but preceded by a big block of context so PREFILL
# has a lot of tokens to chew through before the first output token.
LONG_PREFIX = (
    "Here is some background reading you do not need to summarize. " * 220
)
LONG_PROMPT = LONG_PREFIX + "\n\nNow: explain what a KV cache is in one sentence."


@dataclass
class Measure:
    label: str
    prompt_tokens: int
    output_tokens: int
    ttft_s: float           # wall-clock to the first STREAMED token
    prefill_s: float        # Ollama-reported prompt_eval_duration
    decode_s: float         # Ollama-reported eval_duration

    def tpot_ms(self) -> float:
        """Time per output token, in milliseconds, from the DECODE phase."""
        # TODO 1: return decode_s / output_tokens * 1000, guarding against
        #         output_tokens == 0 (return 0.0 in that case).
        return 0.0


def measure(label: str, prompt: str) -> Measure:
    """Stream a generation, timestamp the first token, then read final stats."""
    t0 = time.perf_counter()
    ttft: float | None = None
    final: dict = {}

    with httpx.stream(
        "POST",
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": True},
        timeout=300.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            # The first chunk carrying actual generated text marks first token.
            if ttft is None and chunk.get("response"):
                # TODO 2: set ttft to the elapsed wall-clock since t0
                #         (time.perf_counter() - t0). This is the REAL TTFT.
                ttft = 0.0
            if chunk.get("done"):
                final = chunk  # the terminal chunk carries the duration stats

    # Ollama durations are nanoseconds. prompt_eval = prefill; eval = decode.
    prefill_s = final.get("prompt_eval_duration", 0) / 1e9
    decode_s = final.get("eval_duration", 0) / 1e9
    return Measure(
        label=label,
        prompt_tokens=final.get("prompt_eval_count", 0),
        output_tokens=final.get("eval_count", 0),
        ttft_s=ttft if ttft is not None else 0.0,
        prefill_s=prefill_s,
        decode_s=decode_s,
    )


def print_row(m: Measure) -> None:
    print(f"{m.label:<14} {m.prompt_tokens:>6} {m.output_tokens:>6} "
          f"{m.ttft_s:>8.3f}s {m.prefill_s:>9.3f}s {m.decode_s:>8.3f}s "
          f"{m.tpot_ms():>8.1f}")


def main() -> None:
    print(f"Model: {MODEL}\n")
    print(f"{'PROMPT':<14} {'P-TOK':>6} {'O-TOK':>6} "
          f"{'TTFT':>9} {'PREFILL':>10} {'DECODE':>9} {'TPOT(ms)':>9}")
    print("-" * 74)

    short = measure("short", SHORT_PROMPT)
    print_row(short)
    long_ = measure("long", LONG_PROMPT)
    print_row(long_)

    print("\n--- diagnosis ---")
    print(f"TTFT  short={short.ttft_s:.3f}s   long={long_.ttft_s:.3f}s   "
          f"(long should be LARGER -> prefill scales with prompt length)")
    print(f"TPOT  short={short.tpot_ms():.1f}ms  long={long_.tpot_ms():.1f}ms  "
          f"(should be SIMILAR -> decode is per-token, ~prompt-independent)")

    if long_.ttft_s > short.ttft_s and abs(long_.tpot_ms() - short.tpot_ms()) < (
        0.5 * max(short.tpot_ms(), 1.0)
    ):
        print("\nMATCH: longer prompt -> slower first token (prefill), but the "
              "per-token streaming rate (decode) held roughly steady. That is "
              "the prefill/decode split, measured on your own hardware.")
    else:
        print("\nNOTE: the SHAPE may be noisy on a loaded machine. Re-run on an "
              "idle box; the prefill-grows / decode-flat pattern is the lesson.")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Expected output (your numbers will differ; the SHAPE is the point)
# -----------------------------------------------------------------------------
#
# Model: qwen2.5:7b
#
# PROMPT          P-TOK  O-TOK      TTFT    PREFILL    DECODE  TPOT(ms)
# --------------------------------------------------------------------------
# short              14     38     0.181s     0.142s    1.520s      40.0
# long             2204     41     1.094s     1.043s    1.640s      40.0
# (the long row's TTFT/prefill are ~5-10x the short row's; TPOT ~unchanged)
#
# --- diagnosis ---
# TTFT  short=0.181s   long=1.094s   (long should be LARGER -> prefill ...)
# TPOT  short=40.0ms  long=40.0ms  (should be SIMILAR -> decode is per-token ...)
#
# MATCH: longer prompt -> slower first token (prefill), but the per-token
# streaming rate (decode) held roughly steady. ...
#
# THE LESSON: "latency" is TWO numbers. Prefill (compute-bound) sets TTFT and
# grows with prompt length; decode (bandwidth-bound) sets TPOT and is roughly
# prompt-length-independent. This is why a 100k-token prompt feels slow to
# START but still streams at a normal rate once it gets going.
# -----------------------------------------------------------------------------
