# Week 6 — Local Inference Bring-Up

Welcome to the week you stop renting a brain. For five weeks you've called a model that lives on someone else's GPU behind an API key — `claude-sonnet-4-6`, a local Ollama model you treated as a black box, whatever answered your `tool_use` calls. This week you open the black box and stand the inference up *yourself*: the same 7B model on three different engines, on hardware you understand, with a benchmark that tells you — in tokens per second, in p95 latency, in gigabytes of VRAM — exactly what each choice costs. By Friday you will never again say "the model is slow" without being able to say *which part* is slow and *why*.

This is the last week of **Phase I — Foundations**, and it closes the loop the phase opened. Week 5 gave you a ReAct agent. This week gives you the *substrate that agent runs on* when there's no vendor in the loop — and the phase-I capstone milestone is exactly that: a working ReAct agent on a **local** 7B model, with a measured benchmark score. The agent you hand-rolled in week 5 stops being a thing that talks to Anthropic and becomes a thing that talks to a model *you* are serving. That's the difference between an engineer who can prototype and one who can ship without a credit card on file.

The one sentence to internalize before you read another line:

> **The fastest token is the one you do not generate.** Quantization, batching, and KV-cache reuse are not optimizations bolted onto a working system — they *are* the product. A self-hosted LLM that ignores them is a space heater.

Here's why that's not hyperbole. A 7B model in FP16 is ~14 GB of weights; every token you generate reads all 14 GB out of VRAM. Memory bandwidth — not compute — is the wall you hit first in single-stream decoding. Quantize to 4-bit and the same token reads ~4 GB: roughly 3.5× less memory traffic, roughly 3.5× more tokens per second, on the *same* GPU, for a few points of quality. Batch eight requests and the weights are read *once* for all eight, not eight times. Reuse the KV cache across a multi-turn conversation and you skip re-encoding the prompt every turn. None of these is exotic. All of them are the difference between a GPU that serves one user and a GPU that serves a hundred. This week is where you learn to see them.

There's a corollary worth taping to your monitor:

> **Pick the engine for the workload, not the benchmark.** Ollama wins on friction. llama.cpp wins on portability. vLLM wins on throughput under concurrency. The "fastest" engine on a single-stream microbenchmark can be the *slowest* choice for a server taking 50 concurrent requests — because the benchmark measured the wrong thing.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** the local-inference stack of 2026 — **Ollama** for fast iteration, **llama.cpp** for portable CPU/Metal inference, **vLLM** for high-throughput GPU serving, with **SGLang**, **TensorRT-LLM**, **TGI**, and **Apple MLX** placed by the workload each wins — and pick one for a given job with reasons, not vibes.
- **Bring up** the same 7B model (Qwen2.5-7B / Llama-3.1-8B class) on Ollama, llama.cpp, and vLLM, and hit each with the same prompt through each one's API.
- **Reason** about the two phases of inference — **prefill** (compute-bound, processes the whole prompt at once) and **decode** (memory-bandwidth-bound, generates one token at a time) — and explain why they have completely different performance characteristics.
- **Choose** a quantization format — **GGUF** (Q4_K_M and friends, the llama.cpp family), **AWQ** and **GPTQ** (GPU-side 4-bit for vLLM), **bitsandbytes** (on-the-fly), **FP16** (the unquantized baseline) — and articulate the quality/speed/VRAM trade-off each makes.
- **Measure** an inference engine honestly: tokens/sec (prefill vs decode separately), p50/p95 latency, time-to-first-token, VRAM, and throughput under concurrency — and read the resulting chart without fooling yourself.
- **Understand** the throughput multipliers — **continuous batching**, **paged attention**, **KV-cache reuse**, and **speculative decoding** — well enough to explain *why* vLLM serves many concurrent users on the same GPU where llama.cpp serves one.
- **Stand up** a real serving endpoint (vLLM's OpenAI-compatible API) and point your week-5 agent at it, completing the Phase I milestone: a ReAct agent on a model *you* serve.
- **Cost** a self-hosted deployment against a vendor API and state the break-even — when serving your own 7B is cheaper than paying per token, and when it isn't.

## Prerequisites

This week assumes you have completed **C23 weeks 1–5**, or have equivalent fluency. Specifically:

- You finished **week 5** and have a hand-rolled ReAct agent (~150 lines, no framework) that runs against a model behind an API. **This week you point that agent at a local endpoint** — if it only works against Anthropic's API and you hard-coded the client, refactor it to take a base URL first.
- You're comfortable from **week 1** calling a model via Ollama and reading a model card; from **week 2** you know what a token, a context window, and a KV cache *are* conceptually. This week makes the KV cache a thing you can see in a VRAM number.
- Python 3.12 on Linux, macOS (Apple Silicon), or WSL2; `docker` and `docker compose`; a shell you're not afraid of. Comfort reading `nvidia-smi` output helps but we teach it.

**Hardware reality, stated plainly.** The Ollama and llama.cpp legs run on a **laptop** — a 16 GB Mac or a 16 GB Linux box does the whole CPU/Metal half of the week. The vLLM leg wants a **CUDA GPU**: a local 24 GB card (RTX 3090/4090) is ideal, but the lab is written to run on a **rented L4 or A10 for ~$1 of compute** (~1–2 hours at ~$0.50–$1.00/h on a spot instance). Every lab has a CPU-only or smaller-model fallback documented in its header, and the vLLM leg has a "rent it for an hour" recipe with a cost ceiling. **You are not blocked without a GPU** — you'll run vLLM on rented compute for one short session and reason about the numbers.

## Topics covered

- **Why local inference at all:** the vendor-lock failure mode, the data-residency case, the cost case, and the latency case — when self-hosting is the right call and when it's premature.
- **The two phases of inference:** prefill (compute-bound, the whole prompt in parallel) vs decode (memory-bandwidth-bound, one token at a time), and why every performance number splits along this seam.
- **Ollama:** the lowest-friction local runtime; the model registry; `ollama run` / `ollama serve`; the OpenAI-compatible endpoint; why it's the right tool for iteration and the wrong tool for a production server.
- **llama.cpp:** the portable C/C++ substrate; GGUF; CPU and Metal and CUDA backends; `llama-server`; the engine that runs *anywhere*, from a Raspberry Pi to an H100.
- **vLLM:** the high-throughput serving engine; **continuous batching** and **paged attention**; the OpenAI-compatible server; why it's the open serving primitive for concurrency.
- **The rest of the stack, placed:** **SGLang** (structured/grammar-heavy workloads), **TensorRT-LLM** (NVIDIA-optimized kernels), **TGI** (the HF ecosystem), **Apple MLX** (Mac-native) — named, contrasted, and slotted by workload.
- **Quantization:** GGUF (Q4_K_M, Q5_K_M, Q8_0), AWQ, GPTQ, bitsandbytes, FP16; what bits you trade for what speed and VRAM; the perplexity-vs-size curve; why Q4_K_M is the workhorse default.
- **Speculative decoding and KV-cache reuse:** the draft-model trick that generates several tokens per big-model forward pass; prefix caching that skips re-encoding a shared prompt.
- **Measuring inference honestly:** tokens/sec, TTFT, p50/p95, VRAM, throughput-under-concurrency; the benchmark that compares engines fairly and the microbenchmark that lies.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                          | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|----------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Why local; prefill vs decode; Ollama + llama.cpp bring-up      |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Quantization formats; GGUF/AWQ/GPTQ/FP16; the quant exercise   |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | vLLM; continuous batching; paged attention; the serving leg    |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | The benchmark harness; measuring tokens/sec, p95, VRAM         |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The three-engine bakeoff + memo; agent-on-local-endpoint       |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                          |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                       |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                                | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The engine docs (Ollama/llama.cpp/vLLM), quantization references, the speculative-decoding and paged-attention papers, model cards |
| [lecture-notes/01-the-local-inference-stack.md](./lecture-notes/01-the-local-inference-stack.md) | Why local, prefill vs decode, the three engines (Ollama/llama.cpp/vLLM) and the rest of the stack placed by workload |
| [lecture-notes/02-quantization-batching-and-serving.md](./lecture-notes/02-quantization-batching-and-serving.md) | Quantization formats, continuous batching + paged attention, speculative decoding, and the honest benchmark |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-bring-up-three-engines.md](./exercises/exercise-01-bring-up-three-engines.md) | Bring the same 7B up on Ollama, llama.cpp, and vLLM and hit each through its API |
| [exercises/exercise-02-quantization-tradeoffs.py](./exercises/exercise-02-quantization-tradeoffs.py) | Compute the VRAM/speed/quality trade-off across quant formats and chart it |
| [exercises/exercise-03-bench-harness.py](./exercises/exercise-03-bench-harness.py) | A reusable inference benchmark: tokens/sec, TTFT, p50/p95, throughput under concurrency |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-three-engine-bakeoff.md](./challenges/challenge-01-three-engine-bakeoff.md) | The full bakeoff: one model, three engines, the 100-prompt benchmark, a winner memo |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the engine-selection memo and the agent-on-local-endpoint task |
| [mini-project/README.md](./mini-project/README.md) | The `crunchserve` benchmark harness — pluggable engines, one report, a defended recommendation |

## The "served it yourself" promise

C23 uses a recurring marker for every exercise that ends in inference actually running *on hardware you understand*, with a number that proves it:

```
$ python bench.py --engine vllm --model qwen2.5-7b --quant awq --concurrency 32
engine=vllm  model=qwen2.5-7b-awq  concurrency=32
  prefill: 4200 tok/s   decode: 61 tok/s/req   aggregate: 1950 tok/s
  TTFT p50: 0.18s   p95: 0.41s   VRAM: 8.9 GB
  100/100 prompts OK — served on YOUR endpoint, no vendor key
```

That aggregate-1950-tokens/sec line is the whole point of the week. The *same* model on llama.cpp at concurrency 32 would serialize the requests and crawl; vLLM's continuous batching keeps the GPU full and the throughput multiplies. The number isn't a brag — it's the evidence behind a deployment decision you'll defend in the milestone memo.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **PagedAttention / vLLM paper** (Kwon et al., 2023, arXiv 2309.06180) until you can explain why fragmentation of the KV cache wastes VRAM and how paging fixes it: <https://arxiv.org/abs/2309.06180>. Then watch your vLLM VRAM usage as you raise `--max-num-seqs` and see paging in action.
- Turn on **speculative decoding** in vLLM with a small draft model (e.g. a 0.5B drafting for a 7B) and measure the decode-speed lift on your benchmark. Predict the lift first (it depends on draft acceptance rate), then measure.
- Run the same model through **Apple MLX** on a Mac and compare tokens/sec to llama.cpp's Metal backend. MLX is the Mac-native path; see where it wins.
- Quantize a model to **three** GGUF levels yourself (Q4_K_M, Q5_K_M, Q8_0) with `llama-quantize`, measure perplexity on a fixed text, and plot perplexity vs file size — the quality/size curve, by your own hand.

## Up next

Week 6 closes Phase I. You now have local inference under your fingers and a ReAct agent running on a model you serve — the Phase I capstone milestone. **Week 7 opens Phase II — RAG & Memory Systems** and changes the question from "how do I run the model" to "how do I give the model knowledge it didn't have at training time." You'll embed a corpus, index it in pgvector, and measure retrieval — and the local-embedding and local-reranker models you run there will lean on exactly the bring-up skills you built this week. Push your benchmark harness before you start week 7; the discipline of measuring inference honestly is the same discipline you'll point at retrieval.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
