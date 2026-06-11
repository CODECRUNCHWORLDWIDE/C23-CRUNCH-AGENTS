# Challenge 1 — The Three-Engine Bakeoff

**Time estimate:** ~150 minutes (plus ~$1 of rented GPU for the vLLM leg, if you don't have a local CUDA card).

## Problem statement

You have one model and a serving decision to make. Four reasonable people on your team each swear by a different engine. You are going to end the argument the only way it can honestly end: run the *same* model on each engine, through the *same* benchmark, at the *same* concurrency points, and let the numbers pick the serving engine. Then you write down *why* it won — because a winner you can't explain is a winner you got lucky with.

This is the syllabus hands-on lab in committed form. The output is a decision: one serving engine, with prefill/decode tokens/sec, TTFT, p50/p95, VRAM, and the aggregate-throughput-vs-concurrency curve, and a paragraph of reasons grounded in *your* workload.

## The three engines

Run exactly these three, same model on each:

1. **Ollama** — `ollama serve`, the friction-free runtime (Q4_K_M GGUF). The iteration baseline.
2. **llama.cpp** — `llama-server` on the same Q4_K_M GGUF. The portable substrate; also your honest CPU baseline if you run it CPU-only.
3. **vLLM** — `vllm serve` the AWQ build on a CUDA GPU (local or rented L4/A10). The throughput engine.

> **The one honest asymmetry:** Ollama and llama.cpp serve a **GGUF Q4_K_M** quant; vLLM serves an **AWQ-4** quant. Both are 4-bit but they're *different* 4-bit formats, so a vLLM-vs-llama.cpp delta is partly engine and partly quant. Call this out in the memo — exactly like late-chunking's model swap in week 8. Where you can, note the FP16 baseline numbers too so the quant effect is visible.

## What is fixed (do not let these vary)

- **Model:** Qwen2.5-7B-Instruct (or Llama-3.1-8B) — the *same* model on every engine. This is the whole validity of the bakeoff.
- **Prompt set:** the same 100 prompts (same input lengths) for every engine. (Generate or reuse a fixed set; a mix of short and RAG-length prompts is realistic.)
- **`max_tokens`:** the same output cap (e.g. 128) for every engine, so decode work is equal.
- **Concurrency points:** the same sweep (1, 8, 32, 128) for every engine — and the *curve* reported, not a single point.
- **Measurement:** prefill vs decode separated; TTFT p50/p95; aggregate tokens/sec per concurrency; VRAM at the serving concurrency; warm-ups discarded; median of repeats.

## The harness approach

Reuse your Exercise 3 benchmark; it already hits any OpenAI-compatible endpoint by `base_url`. The whole bakeoff reduces to: point the *same* harness at each engine's URL, sweep concurrency, collect the curves.

```python
ENGINES = {
    "ollama":   ("http://localhost:11434/v1", "qwen2.5:7b",                    "gguf-q4km"),
    "llamacpp": ("http://localhost:8080/v1",  "qwen2.5-7b",                    "gguf-q4km"),
    "vllm":     ("http://localhost:8000/v1",  "Qwen/Qwen2.5-7B-Instruct-AWQ", "awq-4"),
}

for engine, (url, model, quant) in ENGINES.items():
    for c in (1, 8, 32, 128):
        row = await bench_concurrency(url, model, concurrency=c, max_tokens=128)
        record(engine, quant, c, row)   # aggregate tok/s, ttft p50/p95, decode/req
    record_vram(engine, nvidia_smi_used_gb())   # at the HIGHEST concurrency you ran
```

That identical harness call per engine is the whole point: you changed only the engine (modulo the noted quant asymmetry). The curve is the result.

## Acceptance criteria

- [ ] A `challenge-01/` directory with a runnable `bakeoff.py` (built on the Exercise 3 harness) that benchmarks all three engines at the same concurrency sweep and prints a comparison table.
- [ ] The table reports, per engine: **prefill/decode tokens/sec, TTFT p50/p95, aggregate tokens/sec at each concurrency, and VRAM** at the serving concurrency.
- [ ] The model, prompt set, and `max_tokens` are demonstrably **identical** across engines; the GGUF-vs-AWQ quant asymmetry is explicitly noted.
- [ ] The **aggregate-throughput-vs-concurrency curve** is plotted (ASCII bar or a chart) for each engine, and the difference in *shape* (vLLM rising, Ollama/llama.cpp flattening) is visible.
- [ ] A one-page `bakeoff-memo.md` that names the **serving engine you'd ship**, gives its numbers, and explains in a paragraph **why it won for this workload** (a concurrent server vs a single-user tool) — not in general.
- [ ] At least one **promise-format line** for the winner, e.g. `engine=vllm c=32 -> aggregate 1950 tok/s, p95 0.41s, VRAM 8.9 GB — served on YOUR endpoint`, plus a counter-example concurrency where a single-stream engine's per-request decode collapsed.

## The trap (read after a first attempt)

The trap is **benchmarking at concurrency 1 and ranking the engines on it.** At concurrency 1, vLLM and llama.cpp look close — sometimes vLLM looks *slower* (heavier startup, no batch to amortize). If you stop there, you'll conclude "llama.cpp is as fast as vLLM" and ship the wrong engine for a server. The whole character of a serving engine is the *shape of the curve as concurrency rises* — vLLM's aggregate climbs because continuous batching keeps the GPU full; llama.cpp's flattens because it serializes. **You must sweep concurrency, not measure one point.** Measure only concurrency 1 and your bakeoff measured iteration speed, not serving capacity — and those are different jobs (Lecture 1 §5, Lecture 2 §2).

A second, subtler trap: **crediting a quant difference to the engine.** vLLM-AWQ beating llama.cpp-GGUF is two changes (engine + quant format). To isolate the engine, also note the FP16 numbers where you can run them, or at least flag in the memo that the comparison includes a quant-format change — so nobody mistakes "AWQ is leaner than this GGUF" for "vLLM's engine is faster." (Same discipline as Lecture 2 §5.5's fair-comparison checklist.)

## Stretch goals

- **Add the FP16 control.** Run one engine (vLLM) at FP16 and at AWQ on the same prompts. The FP16-vs-AWQ gap *at the same engine* is the pure quant effect, engine held constant — the honest isolation.
- **Speculative decoding on the winner.** Turn on a 0.5B draft model in vLLM (`--speculative-config ...`) and re-measure decode tok/s. Predict the lift from the acceptance rate first, then measure (Lecture 2 §3.1).
- **CPU baseline.** Run llama.cpp CPU-only (no GPU) and record how many tokens/sec a 7B does on pure CPU. That's the floor — the "no GPU at all" number every cost discussion needs.
- **Prefix-cache test.** Send 32 requests that share a 1,000-token system prompt to vLLM with automatic prefix caching on, then off. Measure the TTFT difference (Lecture 2 §3.2).

## Why this matters

In the Phase I milestone you serve your week-5 ReAct agent on a *local* model and, in the capstone, you serve the local tier on a vLLM cluster (week 19) with a vendor fallback for hard routes. The reviewer will not ask you to recite the three engines — they'll point at your serving choice and ask "why *that* engine, and how do you know it holds up at the concurrency you'll actually see?" This challenge *is* that conversation, rehearsed: you ran the alternatives, you have the curves, you can name the engine and the number that justifies it. Every LLM you serve after this runs on *some* engine whether you chose it deliberately or not — the engineer who *chose* it, with a measured concurrency curve behind the choice, is the one whose server doesn't fall over at 2 AM when the traffic arrives. You served it yourself, and you can prove it survives load.
