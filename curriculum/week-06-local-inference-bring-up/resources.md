# Week 6 — Resources

Every resource here is **free** and current for 2026. Ollama, llama.cpp, vLLM, SGLang, and MLX are all open-source with public docs. The model weights we use (Qwen 2.5 7B) are open-weights. No paywalled books are linked.

Model and runtime versions move fast; the *concepts* — prefill/decode, the KV cache, continuous batching, quantization formats — are stable across versions. Where a doc is versioned, the current URL is given; swap for your installed version if it has moved.

## Required reading (work it into your week)

- **vLLM — *Easy, Fast, and Cheap LLM Serving with PagedAttention*** (Kwon et al., 2023) — the paper behind continuous batching and paged attention; read it Wednesday:
  <https://arxiv.org/abs/2309.06180>
- **Ollama — README and quickstart** — pull, run, and the OpenAI-compatible `/v1` endpoint your agent already uses:
  <https://github.com/ollama/ollama>
- **llama.cpp — README** — the portable GGUF substrate; build, the `llama-server`, and the GGUF format:
  <https://github.com/ggml-org/llama.cpp>
- **vLLM — quickstart and OpenAI-compatible server** — `vllm serve`, the API surface, and serving config:
  <https://docs.vllm.ai/en/latest/getting_started/quickstart.html>
- **GGUF and quantization types** — what `Q4_K_M`, `Q5_K_M`, `Q8_0` actually mean:
  <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>

## The runtime docs (the ones you'll have open all week)

- **vLLM docs** — serving, quantization (AWQ/GPTQ), speculative decoding, prefix caching:
  <https://docs.vllm.ai/>
- **SGLang** — the structured-output-heavy runtime; grammar-constrained decoding at throughput:
  <https://github.com/sgl-project/sglang>
- **Hugging Face TGI (Text Generation Inference)** — the HF-ecosystem serving option:
  <https://github.com/huggingface/text-generation-inference>
- **Apple MLX** and **mlx-lm** — the Mac-native inference path:
  <https://github.com/ml-explore/mlx-lm>
- **NVIDIA TensorRT-LLM** — the NVIDIA-optimized kernel path for H100/H200:
  <https://github.com/NVIDIA/TensorRT-LLM>

## Quantization references

- **AWQ — *Activation-aware Weight Quantization*** (Lin et al., 2023) — the 4-bit GPU weight-only method vLLM supports:
  <https://arxiv.org/abs/2306.00978>
- **GPTQ — *Accurate Post-Training Quantization*** (Frantar et al., 2022) — the other common 4-bit GPU format:
  <https://arxiv.org/abs/2210.17323>
- **bitsandbytes** — on-the-fly 8-bit/4-bit quantization in PyTorch (the Transformers `load_in_4bit` path):
  <https://github.com/bitsandbytes-foundation/bitsandbytes>
- **The llama.cpp quantization table** — every GGUF quant type with its bits-per-weight and quality note (search the repo's `quantize` tool output and the k-quant discussion):
  <https://github.com/ggml-org/llama.cpp/discussions/2094>

## How-to and benchmarking

- **vLLM — benchmarking guide** — `benchmark_serving.py`, the throughput/latency methodology:
  <https://docs.vllm.ai/en/latest/contributing/profiling/profiling_index.html>
- **vLLM — continuous batching explained** — the blog that made the concept legible:
  <https://www.anyscale.com/blog/continuous-batching-llm-inference>
- **Ollama — the Modelfile and model import** — pulling a specific quant, importing a GGUF:
  <https://github.com/ollama/ollama/blob/main/docs/modelfile.md>
- **Renting a GPU cheaply** — for the vLLM lab, any spot L4/A10 works; pick a provider you trust and watch the per-hour rate. The lab is ~$1 of compute; don't leave the box running.

## Talks worth your time (free, no signup)

- **vLLM Office Hours / meetups** — the maintainers walk continuous batching, paged attention, and speculative decoding; posted free:
  <https://www.youtube.com/@vllm-project>
- **GPU MODE (formerly CUDA MODE)** — community talks on inference kernels and serving internals:
  <https://www.youtube.com/@GPUMODE>
- **Ollama and llama.cpp community channels** — quantization and Apple Silicon walkthroughs from the maintainers and community.

## Tools you'll use this week

- **`ollama`** — `ollama pull qwen2.5:7b-instruct`, `ollama serve`, `ollama run`. Lowest friction; Metal on Mac, CUDA on Linux.
- **`llama-server`** (llama.cpp) — `llama-server -m model.gguf --port 8080`. The portable OpenAI-compatible server over a GGUF file.
- **`vllm serve`** — `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`. Continuous batching, paged attention; needs CUDA.
- **`nvidia-smi`** — VRAM and GPU utilization. Your VRAM column comes from here (`nvidia-smi --query-gpu=memory.used --format=csv`).
- **`curl` + `jq`** — `curl localhost:8000/v1/models | jq` to confirm a server is up before you benchmark it.
- **The `openai` Python client** — point `base_url` at any of the three local servers; the API is OpenAI-compatible across all of them.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Prefill** | Processing the whole input prompt in one pass, filling the KV cache. Dominates time-to-first-token. |
| **Decode** | Generating output one token at a time, reusing the KV cache. Dominates tokens/sec. |
| **KV cache** | Cached key/value tensors from prior tokens so each new token doesn't reprocess the whole sequence. Why streaming feels fast. |
| **TTFT** | Time-to-first-token — how long until the first output token. Prefill-bound. |
| **tokens/sec** | Decode throughput — output tokens per second after the first. Decode-bound. |
| **Continuous batching** | Interleaving many requests at the token level so the GPU never idles between sequences. vLLM's throughput multiplier. |
| **Paged attention** | KV cache stored as fixed-size pages (like OS virtual memory) — no fragmentation, higher batch sizes. |
| **Prefix caching** | Reusing the KV cache for a shared prompt prefix across requests. |
| **Speculative decoding** | A small draft model proposes several tokens; the big model verifies them in one pass. Faster on predictable text. |
| **GGUF** | The llama.cpp/Ollama model file format; carries the quantization (e.g. `Q4_K_M`). |
| **AWQ / GPTQ** | 4-bit GPU weight-only quantization formats; run on vLLM/TGI. |
| **bitsandbytes** | On-the-fly 8/4-bit quantization in PyTorch (`load_in_4bit`). |
| **Q4_K_M** | A GGUF 4-bit k-quant (medium) — the common "good enough, small" default for a 7B. |
| **FP16 / BF16** | 16-bit unquantized weights — the quality baseline, ~2 bytes/param (a 7B ≈ 14 GB). |
| **VRAM** | GPU memory. Weights + KV cache + activations must fit. The quantization you pick is mostly a VRAM decision. |

---

*If a link 404s, please open an issue so we can replace it.*
