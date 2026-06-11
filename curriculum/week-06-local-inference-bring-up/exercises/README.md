# Week 6 — Exercises

Three focused drills that take you from "I call a model over an API" to "I serve a model myself and I measured exactly what it costs." Each takes 30–60 minutes. Do them in order — exercise 3 (the benchmark) measures the engines you brought up in exercise 1, and the quantization intuition from exercise 2 explains the VRAM and tokens/sec numbers you'll see.

## Index

1. **[Exercise 1 — Bring up three engines](exercise-01-bring-up-three-engines.md)** — stand the *same* 7B up on Ollama, llama.cpp, and vLLM, and hit each through its OpenAI-compatible endpoint with one client. (~50 min, guided)
2. **[Exercise 2 — Quantization trade-offs](exercise-02-quantization-tradeoffs.py)** — compute the VRAM/decode-speed/quality trade-off across quant formats (FP16/Q8_0/Q5_K_M/Q4_K_M/AWQ) and chart the quality-vs-size curve so "Q4_K_M sits at the knee" stops being a slogan. (~45 min, runnable)
3. **[Exercise 3 — The benchmark harness](exercise-03-bench-harness.py)** — a reusable inference benchmark: prefill vs decode tokens/sec, TTFT, p50/p95, and aggregate throughput under rising concurrency. (~55 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install the deps as each exercise needs them: `pip install openai httpx numpy`. The vLLM leg needs `pip install vllm` and a CUDA GPU (or a rented one); the llama.cpp leg needs the `llama.cpp` build; Ollama needs the Ollama install. Each exercise header lists exactly what it needs and its fallback.
- **Pick one 7B model and hold it fixed.** `Qwen2.5-7B-Instruct` (or Llama-3.1-8B) on all three engines, so the *engine* is the only variable. Changing the model mid-week is how you get numbers you can't compare.
- **Name the phase before you read the number.** Every tokens/sec figure is *either* prefill *or* decode (Lecture 1 §2). A blended number tells you nothing; separate them.
- **Measure VRAM at the concurrency you serve, not at concurrency 1** (Lecture 2 §4). The KV cache only shows up under load.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

Exercises 2 and 3 are standalone. Exercise 2 is pure arithmetic + a chart (no GPU, no model download — it runs anywhere). Exercise 3 hits a *running* engine's endpoint; its header documents how to point it at Ollama (laptop, no GPU) if you don't have vLLM up yet, so the harness logic is exercisable today and the full concurrency story when you have the GPU.

```bash
# Exercise 2 needs nothing but numpy:
python3 exercise-02-quantization-tradeoffs.py

# Exercise 3 needs a running endpoint. Easiest: Ollama on your laptop.
ollama serve &           # exposes :11434
ollama pull qwen2.5:7b
python3 exercise-03-bench-harness.py --base-url http://localhost:11434/v1 --model qwen2.5:7b
```

The first model pull downloads several GB. Do it on good wifi, not five minutes before a deadline.

## A note on hardware

- **Ollama and llama.cpp run on your laptop** (16 GB Mac or Linux box). The whole CPU/Metal half of the week needs no GPU.
- **vLLM needs a CUDA GPU.** If you don't have one, the vLLM leg runs on a **rented L4/A10 for ~$1** (1–2 hours; see resources.md for the recipe and the spending cap). The CPU fallback (llama.cpp) carries the rest, and Exercise 3 still runs fully against Ollama so you exercise the harness now and re-run the concurrency curve on the GPU when you have it.
- **No lab is gated on personal hardware.** Every header has a fallback.

## A note on determinism

Inference throughput is *not* perfectly deterministic — it depends on thermal state, background load, driver version, and (for ANN-free generation) sampling. So run each measurement **3 times, discard the warm-up, and report the median**, exactly as the benchmark harness does. If your tokens/sec swings 30% run-to-run, something else is using the GPU (or your laptop is thermal-throttling) — find it before you trust the number.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-06` to compare.
