# Week 6 — Resources

Every resource here is **free** or has a free tier. The inference engines (Ollama, llama.cpp, vLLM, SGLang, TGI, MLX) are open source. Model weights live on Hugging Face under open licenses (Qwen, Llama, Mistral). The papers are on arXiv. The only thing that costs money is the rented GPU for the vLLM leg — and that's ~$1 of compute for the whole week, with a CPU/smaller-model fallback documented for every lab.

Engine names, flags, and quant formats move every cohort — the *concepts* (prefill vs decode, continuous batching, paged attention, the quality/size quant curve, throughput-under-concurrency) are stable. When a specific flag 404s or a CLI option is renamed, search the engine's docs for the concept name.

## Required reading (work it into your week)

- **vLLM documentation — quickstart and the OpenAI-compatible server.** The serving leg of the whole week. Read the quickstart and the "OpenAI Compatible Server" page until you can start a server and curl `/v1/chat/completions`:
  <https://docs.vllm.ai/en/latest/>
- **llama.cpp README.** The portable substrate. Read the build instructions for your platform (Metal on Mac, CUDA on Linux) and the `llama-server` section:
  <https://github.com/ggml-org/llama.cpp>
- **Ollama documentation.** The lowest-friction runtime. Read `ollama run`, `ollama serve`, the Modelfile basics, and the OpenAI-compatibility note:
  <https://github.com/ollama/ollama/blob/main/docs/README.md>
- **PagedAttention / vLLM paper** — Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023). The mechanism that makes continuous batching memory-efficient. Read §3–4:
  <https://arxiv.org/abs/2309.06180>

## The engines (have these open all week)

- **Ollama** — `curl -fsSL https://ollama.com/install.sh | sh` (Linux) or the Mac app. Model registry: `ollama pull qwen2.5:7b`. OpenAI-compatible endpoint at `http://localhost:11434/v1`:
  <https://github.com/ollama/ollama>
- **llama.cpp** — build from source or `brew install llama.cpp`. Runs GGUF on CPU/Metal/CUDA/Vulkan. `llama-server -m model.gguf` exposes an OpenAI-compatible endpoint on `:8080`:
  <https://github.com/ggml-org/llama.cpp>
- **vLLM** — `pip install vllm` (CUDA) . `vllm serve Qwen/Qwen2.5-7B-Instruct` starts an OpenAI-compatible server on `:8000`. The high-throughput engine:
  <https://github.com/vllm-project/vllm>
- **SGLang** — `pip install "sglang[all]"`. A fast serving engine that wins on structured/grammar-constrained and complex prompting workloads (RadixAttention prefix caching):
  <https://github.com/sgl-project/sglang>
- **Text Generation Inference (TGI)** — Hugging Face's production server; pragmatic when you live in the HF ecosystem:
  <https://github.com/huggingface/text-generation-inference>
- **Apple MLX / mlx-lm** — `pip install mlx-lm`. The Mac-native inference path; `mlx_lm.generate --model ...`. First-class Apple Silicon:
  <https://github.com/ml-explore/mlx-lm>

## Quantization references

- **llama.cpp quantization (GGUF) docs.** What Q4_K_M / Q5_K_M / Q8_0 mean, how `llama-quantize` works, and the quality/size table:
  <https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md>
- **AWQ — Activation-aware Weight Quantization** (Lin et al., 2023). The 4-bit GPU quant vLLM loads; protects salient weights by activation scale:
  <https://arxiv.org/abs/2306.00978>
- **GPTQ — Accurate Post-Training Quantization** (Frantar et al., 2022). The other GPU 4-bit method; one-shot, layer-wise:
  <https://arxiv.org/abs/2210.17323>
- **bitsandbytes** — on-the-fly 8-bit and 4-bit (NF4) quantization in the HF/transformers ecosystem; the easiest path, not the fastest:
  <https://github.com/bitsandbytes-foundation/bitsandbytes>
- **`GGUF` quant types, in plain language** (the llama.cpp wiki / community tables) — the practical "which Q-level for which VRAM budget" cheat sheet. Search for the current table; Q4_K_M is the standard default.

## Serving and throughput

- **Continuous batching, explained** — the Anyscale write-up on how continuous (a.k.a. in-flight) batching multiplies throughput vs static batching. The clearest intuition for the week's headline number:
  <https://www.anyscale.com/blog/continuous-batching-llm-inference>
- **vLLM — speculative decoding docs.** Turn on a draft model and measure the decode-speed lift; the flags and the acceptance-rate intuition:
  <https://docs.vllm.ai/en/latest/features/spec_decode.html>
- **Speculative decoding paper** — Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2022). The draft-then-verify mechanism:
  <https://arxiv.org/abs/2211.17192>

## Measuring inference

- **`nvidia-smi`** — your VRAM and GPU-utilization meter. `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1` polls once a second. On Mac, use `asitop` / Activity Monitor's GPU history.
- **vLLM benchmarks** — vLLM ships `benchmarks/benchmark_serving.py` and `benchmark_throughput.py`; the canonical way to load-test a server at a chosen concurrency. Study its request-rate and percentile reporting:
  <https://github.com/vllm-project/vllm/tree/main/benchmarks>
- **`llama-bench`** — llama.cpp's built-in micro-benchmark for prefill (`pp`) and decode (`tg`) tokens/sec across quant levels and thread counts:
  <https://github.com/ggml-org/llama.cpp/tree/master/examples/llama-bench>

## Models you'll use this week

- **`Qwen/Qwen2.5-7B-Instruct`** — the default 7B for the bakeoff: strong, openly licensed (Apache-2.0), available as GGUF, AWQ, and FP16. Same model on all three engines so the *engine* is the only variable:
  <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- **`Qwen/Qwen2.5-7B-Instruct-AWQ`** — the AWQ 4-bit build for the vLLM leg:
  <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ>
- **`bartowski/Qwen2.5-7B-Instruct-GGUF`** (or the official GGUF) — the GGUF builds (Q4_K_M, Q5_K_M, Q8_0) for the llama.cpp and Ollama legs:
  <https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF>
- **`meta-llama/Llama-3.1-8B-Instruct`** — the alternative 7B-class model if you prefer Llama; gated on a license click. Either model works; pick one and hold it fixed:
  <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
- **`Qwen/Qwen2.5-0.5B-Instruct`** — the tiny draft model for the speculative-decoding stretch goal:
  <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>

## Renting a GPU for the vLLM leg (~$1)

- **RunPod / Vast.ai / Lambda** — spot/community L4 or A10 (24 GB) at ~$0.40–$1.00/h. You need it for ~1–2 hours total. Use a container image with CUDA 12.x and `pip install vllm`; tear it down when done. **Always set a spending cap and stop the pod.**
- **The cost ceiling for the whole week is ~$1–$3.** If you have a local 24 GB GPU (3090/4090), you need *nothing* rented. If you have neither, the vLLM leg runs on a rented L4 for one short session, and the CPU fallback (llama.cpp on your laptop) carries the rest.

## Tools you'll use this week

- **`ollama`** — the runtime + registry; `ollama serve` exposes `:11434/v1`.
- **`llama.cpp` / `llama-server` / `llama-quantize` / `llama-bench`** — the portable substrate, server, quantizer, and benchmark.
- **`vllm`** — `pip install vllm`; `vllm serve <model>` exposes `:8000/v1`.
- **`openai` Python client** — point its `base_url` at any of the three engines' OpenAI-compatible endpoints; one client, three backends.
- **`httpx` / `asyncio`** — for the concurrency leg of the benchmark (fire N requests at once and measure aggregate throughput).
- **`nvidia-smi` / `asitop`** — VRAM and utilization.
- **`week-5` agent** — your hand-rolled ReAct loop; this week it gets a `--base-url` and runs against a local endpoint.

## A note on the model

The whole week holds **one model fixed** — `Qwen2.5-7B-Instruct` (or Llama-3.1-8B if you prefer) — so the *engine* and the *quantization* are the only variables. This is the same one-variable-at-a-time discipline you'll use for the chunking A/B in week 8 and the vector-store bakeoff in week 10: if you change the model *and* the engine, you've learned nothing, because you can't attribute the delta. Pick your 7B on Monday and don't change it until the bakeoff is done.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Prefill** | Processing the whole input prompt in one parallel forward pass; compute-bound. Sets time-to-first-token. |
| **Decode** | Generating output one token at a time, each reading the whole model from memory; memory-bandwidth-bound. Sets tokens/sec. |
| **TTFT** | Time To First Token — how long until the user sees the first output character. Dominated by prefill. |
| **KV cache** | The stored keys/values for past tokens, so decode doesn't re-process the prompt every step. Grows with context length. |
| **Continuous batching** | Adding/removing requests from the running batch token-by-token so the GPU never idles; vLLM's throughput multiplier. |
| **Paged attention** | Storing the KV cache in fixed-size pages (like OS virtual memory) so it doesn't fragment VRAM; lets many sequences share a GPU. |
| **Quantization** | Storing weights in fewer bits (4/8 instead of 16) to cut VRAM and memory traffic, trading a little quality. |
| **GGUF** | llama.cpp's quantized model file format; Q4_K_M / Q5_K_M / Q8_0 are common levels. |
| **AWQ / GPTQ** | 4-bit GPU-side post-training quantization methods vLLM can load. |
| **Speculative decoding** | A small draft model proposes several tokens; the big model verifies them in one pass — more tokens per big-model step. |
| **Throughput** | Aggregate tokens/sec across all concurrent requests — the server metric. Distinct from single-stream tokens/sec. |
| **p50 / p95 latency** | Median and 95th-percentile request latency; p95 is what your slowest 1-in-20 users feel. |

---

*If a link 404s, please open an issue so we can replace it.*
