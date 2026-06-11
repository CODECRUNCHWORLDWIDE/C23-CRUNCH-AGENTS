# Week 6 — Local Inference Bring-Up

Welcome to the week you stop being a tenant on someone else's GPU. By Friday you will be able to take the same 7B model, stand it up on **Ollama**, **llama.cpp**, and **vLLM**, run a fixed benchmark against each, and read the resulting tokens/sec, latency, and VRAM chart honestly — and you will be able to quantize that model three ways and say, with a number, what you traded for what. By the end of the week your Week 5 ReAct agent runs entirely on a server *you* brought up, and that is the Phase I milestone.

We assume you finished Week 5 and have a `crunch_agent` loop with a `provider="qwen"` path that already talks to Ollama's OpenAI-compatible endpoint. This week's labs reuse that loop as a *client*: the agent doesn't change; the *server under it* does, three times. If your Week 5 agent isn't a clean `from crunch_agent.loop import run`, fix that first — every benchmark this week points it at a different local server.

The one thing to internalize before you read another line: **the fastest token is the one you do not generate, and the second fastest is the one a quantized 7B generates on hardware you already own.** Local inference is not a hobbyist's compromise — it is the difference between a product whose unit economics work and one that hemorrhages money to a vendor API on every call. Quantization, batching, and KV-cache reuse are not optimizations you bolt on at the end; on a self-hosted stack they *are* the product. The engineer who can pick the right runtime and the right quantization for a workload — and prove the choice with a perf chart — is the one who gets to ship a self-hosted tier at all.

This week is where you stop being vendor-locked.

## Learning objectives

By the end of this week, you will be able to:

- **Map** the 2026 local-inference landscape — **Ollama** (fast iteration), **llama.cpp** (portable CPU/Metal), **vLLM** (high-throughput CUDA), **SGLang** (structured workloads), **TensorRT-LLM** (NVIDIA-optimized), **TGI** (HF ecosystem), **Apple MLX** (Mac-native) — and say which runtime fits which job and why.
- **Bring up** the same 7B model on Ollama, llama.cpp, and vLLM, confirm each is serving, and call all three from one client with the same prompt.
- **Explain** the inference loop at a systems level — prefill vs decode, the KV cache and why it makes streaming feel fast, why the first token is slow and the rest are fast — without the math.
- **Distinguish** the quantization formats — **GGUF** (llama.cpp/Ollama), **AWQ** and **GPTQ** (GPU weight-only), **bitsandbytes** (on-the-fly) — and state what each trades and where it runs.
- **Benchmark** a runtime properly: a fixed prompt set, tokens/sec, p50/p95 latency, time-to-first-token, and VRAM, with the measurement method documented so the numbers are defensible.
- **Reason** about throughput levers — continuous batching, paged attention, prefix caching, speculative decoding, KV-cache reuse — and say which runtime gives you which.
- **Choose** a quantization for a job by reading a trade-off chart (quality vs VRAM vs tokens/sec) rather than by reputation.
- **Serve** your Week 5 agent entirely on local inference and report its benchmark score on a self-hosted tier.

## Prerequisites

This week assumes you have completed **C23 weeks 1–5**, or have equivalent fluency. Specifically:

- The Week 5 **`crunch_agent` loop** importable, with a working Ollama/`qwen2.5:7b-instruct` path.
- **Ollama** installed and serving (`ollama list` works; `ollama run qwen2.5:7b-instruct "hi"` replies). On Apple Silicon this is your Metal path; on Linux+NVIDIA it uses CUDA.
- Comfort on a Linux shell and with Docker (`docker run`, port mapping, `--gpus all`). The vLLM lab runs in a container.
- Python 3.12, the `openai` client (for OpenAI-compatible local endpoints) and `anthropic` (for the frontier comparison baseline).
- A machine that can run a 7B: a 16 GB laptop carries the Ollama/llama.cpp labs (CPU/Metal); the vLLM lab wants a 24 GB GPU **or** a rented L4/A10 (~$0.50–$1.00/h — the whole lab is ~$1 of compute). A CPU-only fallback is documented for every lab; you'll be slower but unblocked.

You do **not** need prior CUDA or kernel experience. We stay at the *systems* level — prefill/decode, KV cache, batching, quantization formats — and treat the kernels as a black box whose behavior you measure. If you have only ever called a vendor API, this is the week local inference becomes load-bearing.

## Topics covered

- **The runtime landscape**: Ollama (lowest-friction iteration, great on Apple Silicon), llama.cpp (the portable GGUF substrate — runs anywhere, CPU and Metal), vLLM (continuous batching + paged attention, the open serving primitive on CUDA), SGLang (wins on grammar/structured workloads), TensorRT-LLM (every microsecond on H100/H200), TGI (pragmatic in the HF ecosystem), Apple MLX (first-class Apple Silicon). Which to reach for, and why.
- **The inference loop at a systems level**: tokenize → **prefill** (process the whole prompt, fill the KV cache) → **decode** (generate one token at a time, reusing the cache). Why time-to-first-token is dominated by prefill and the rest is decode-bound; why the KV cache is the thing streaming feels.
- **Quantization formats**: **GGUF** (the llama.cpp/Ollama format; `Q4_K_M`, `Q5_K_M`, `Q8_0` and what the suffixes mean), **AWQ** and **GPTQ** (GPU weight-only 4-bit, vLLM/TGI-friendly), **bitsandbytes** (on-the-fly 8/4-bit in PyTorch), **FP16/BF16** (the unquantized baseline). What each trades (quality, VRAM, speed) and where each runs.
- **Throughput levers**: **continuous batching** (the vLLM multiplier — interleave many requests so the GPU never idles), **paged attention** (KV cache as paged memory, no fragmentation), **prefix caching** (reuse a shared prompt prefix across requests), **speculative decoding** (a small draft model proposes, the big model verifies), **KV-cache reuse**. Which runtime gives you which.
- **Benchmarking like an engineer**: a fixed prompt set, tokens/sec (decode), time-to-first-token (prefill), p50/p95 end-to-end latency, VRAM, and *concurrency* (throughput at 1/8/32 concurrent requests — where continuous batching shows up). Documenting the method so the chart is defensible.
- **The cost story**: tokens/sec × utilization vs the equivalent vendor API price; when self-hosting breaks even; why a quantized 7B on a rented L4 can beat a frontier API on unit cost for the easy half of a workload.
- **Pointing your agent at local inference**: the Week 5 loop, unchanged, served by the local server you brought up — and its benchmark score on a self-hosted tier.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                              | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|----------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | The runtime landscape; the inference loop          |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Ollama + llama.cpp bring-up; GGUF quantization     |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | vLLM; continuous batching; the throughput levers   |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Benchmarking method; quantization trade-offs       |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The cost story; point the agent at local serving    |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                             |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, perf-chart write-up polish           |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                    | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The Ollama / llama.cpp / vLLM docs, quantization references, and the talks worth your time |
| [lecture-notes/01-the-inference-stack-and-quantization.md](./lecture-notes/01-the-inference-stack-and-quantization.md) | The runtime landscape, the prefill/decode loop, the KV cache, and the quantization formats |
| [lecture-notes/02-serving-batching-and-the-benchmark.md](./lecture-notes/02-serving-batching-and-the-benchmark.md) | Bringing up vLLM, continuous batching and the throughput levers, benchmarking method, and the cost story |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-bring-up-three-runtimes.md](./exercises/exercise-01-bring-up-three-runtimes.md) | Serve the same 7B on Ollama, llama.cpp, and vLLM; call all three from one client; confirm they agree |
| [exercises/exercise-02-benchmark-harness.py](./exercises/exercise-02-benchmark-harness.py) | A reusable benchmark harness: tokens/sec, TTFT, p50/p95, against any OpenAI-compatible local endpoint |
| [exercises/exercise-03-quantization-sweep.py](./exercises/exercise-03-quantization-sweep.py) | Sweep one model across GGUF quants (Q4_K_M / Q5_K_M / Q8_0 / FP16); chart quality vs VRAM vs speed |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-concurrency-and-batching.md](./challenges/challenge-01-concurrency-and-batching.md) | Find continuous batching: benchmark vLLM at concurrency 1/8/32/128 and explain the throughput curve |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the three-runtime perf-chart write-up |
| [mini-project/README.md](./mini-project/README.md) | The `crunch_serve` bench harness + your Week 5 agent on a self-hosted tier, with a reproducible perf report |

## The "the server is serving" promise

C23 uses a recurring marker for every exercise that ends in a local model actually responding over its serving endpoint:

```text
$ curl -s http://localhost:8000/v1/models | jq -r '.data[0].id'
Qwen/Qwen2.5-7B-Instruct
$ # first token in 180ms, then 62 tokens/sec decode, 14.2 GB VRAM
```

If `/v1/models` 404s or hangs, the server isn't up — check the container logs, the port mapping, and whether the model finished loading (a 7B can take 30–90s to load weights into VRAM). A server that "started" but whose `/v1/models` is empty is the canonical local-inference false start. The point of Week 6 is to make that serving line ordinary — and to make a failed bring-up *loud* (a clear log line) instead of a silent hang.

## Stretch goals

If you finish the regular work early and want to push further:

- Bring up the same 7B on **SGLang** and run a grammar-constrained generation (a strict JSON schema). Compare its structured-output throughput to vLLM's. SGLang wins on grammar workloads — measure by how much.
- Enable **speculative decoding** in vLLM (a small draft model + the 7B as verifier) and measure the tokens/sec lift on your benchmark. Note where it helps (predictable text) and where it doesn't (high-entropy output).
- On Apple Silicon, run the same model through **MLX** and compare tokens/sec to llama.cpp's Metal path on the identical prompt set. The Mac-native path sometimes wins; measure it.
- Quantize a model to **AWQ** and serve it on vLLM; compare tokens/sec and VRAM to the FP16 baseline and to the GGUF `Q4_K_M` on llama.cpp. Now you have a cross-runtime, cross-format chart — the real engineering artifact.

## Up next

This closes **Phase I**. You now have a working ReAct agent on a local 7B with a tool surface and a measured benchmark score — served on inference you brought up yourself, with a quantization you chose for reasons. Push your mini-project; Phase II (Week 7) opens RAG by giving that agent knowledge it didn't have at training time, starting with embeddings and vector search. The local-serving muscle you built this week returns in Week 19, when you scale vLLM to a production cluster.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
