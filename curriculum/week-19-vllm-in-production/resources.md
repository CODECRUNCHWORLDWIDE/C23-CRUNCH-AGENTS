# Week 19 — Resources

Every resource here is **free** or has a free tier. vLLM, LiteLLM, and the benchmarking tools are open source. The PagedAttention paper is on arXiv. The only thing that costs money is the GPU rental for the headline lab — budget ~$12–$18 for ~6 hours on a rented H100, or less on an A10/L4 with a smaller model. Vendor APIs (Anthropic Claude, used in the LiteLLM-fallback leg) have a free tier or a few cents of usage.

Library names, flags, and the exact model SKUs move every cohort — the *concepts* (PagedAttention, continuous batching, the OpenAI-compatible surface, the cost-per-million break-even) are stable. When a specific flag is renamed or a page 404s, search the vLLM docs for the function/flag name.

This week sits on top of week 6 (you brought up vLLM once there) and feeds weeks 20, 23, and 24. The benchmark harness you build here is reused by week 20 (point it at NeMo/Triton) and the LiteLLM router is the capstone's router.

## Required reading (work it into your week)

- **vLLM documentation — Quickstart and the OpenAI-Compatible Server** — the canonical reference for `vllm serve`, the endpoints, and the client. Read the serving args page twice:
  <https://docs.vllm.ai/en/latest/>
- **PagedAttention / vLLM paper** — Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023). The mechanism behind the whole week — the KV cache as virtual memory:
  <https://arxiv.org/abs/2309.06180>
- **LiteLLM documentation — Proxy Server** — the unified OpenAI-compatible router; read the `config.yaml` model-list, routing, and fallbacks pages:
  <https://docs.litellm.ai/docs/simple_proxy>
- **vLLM benchmarking** — `vllm bench serve` (the built-in serving benchmark) and the methodology for reading throughput vs latency under concurrency:
  <https://docs.vllm.ai/en/latest/contributing/benchmarks.html>

## The vLLM internals references

- **vLLM — Continuous batching / scheduling** — how the scheduler admits and evicts sequences each step; the difference between static and in-flight batching:
  <https://docs.vllm.ai/en/latest/design/arch_overview.html>
- **Anyscale — "How continuous batching enables 23x throughput"** — the canonical blog post that makes the static-vs-continuous batching difference concrete with numbers:
  <https://www.anyscale.com/blog/continuous-batching-llm-inference>
- **vLLM — Automatic Prefix Caching** — `--enable-prefix-caching`; reuse the KV cache of a shared prompt prefix across requests:
  <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>
- **vLLM — Speculative Decoding** — `--speculative-config`; the draft-model and n-gram variants, accept rate, and the throughput/latency trade-off:
  <https://docs.vllm.ai/en/latest/features/spec_decode.html>
- **vLLM — Distributed Serving (tensor / pipeline parallel)** — `--tensor-parallel-size`, `--pipeline-parallel-size`; sharding a model that doesn't fit on one GPU:
  <https://docs.vllm.ai/en/latest/serving/distributed_serving.html>

## Routing and the OpenAI-compatible surface

- **LiteLLM — Router / fallbacks** — model aliases, weighted routing, health checks, and `fallbacks` (the local→vendor failover the capstone uses):
  <https://docs.litellm.ai/docs/routing>
- **OpenAI Python SDK** — the client you point at `http://localhost:8000/v1`; the *same* `chat.completions.create` shape you used for vendors. Set `base_url` and a dummy `api_key`:
  <https://github.com/openai/openai-python>
- **`httpx` async client** — the load generator's HTTP layer; `AsyncClient` + `asyncio.gather` to drive true concurrency:
  <https://www.python-httpx.org/async/>

## GPU rental (pick one, rent only for the labs)

- **RunPod** — per-second GPU pods, H100/A100/L40S/A10; the friendliest for a 6-hour lab. Spin up, SSH in, `pip install vllm`, tear down:
  <https://www.runpod.io/>
- **Lambda Cloud** — on-demand H100/A100 instances; clean Ubuntu + CUDA images:
  <https://lambda.ai/>
- **Modal** — serverless GPU; run vLLM as a function, good if you prefer Python-defined infra to SSH:
  <https://modal.com/>
- **vast.ai** — cheapest spot GPUs (community hosts); good for budget, less predictable. Read the host's reliability score:
  <https://vast.ai/>

> **Rent discipline:** set a teardown alarm. A forgotten H100 at $2.50/h is $60 overnight. The labs are designed to run in one ~6-hour sitting; do the reading and write the load generator *before* you spin up the GPU, then run the sweep and tear down.

## Models you'll use this week

- **`Qwen/Qwen2.5-14B-Instruct`** — the headline model for the H100 lab; fits comfortably in FP16 on one H100 (80 GB) with room for a large KV cache. The break-even math compares its self-hosted cost against a vendor:
  <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct>
- **`Qwen/Qwen2.5-7B-Instruct`** — the A10/L4 fallback model if you don't rent an H100; smaller, fits on 24 GB, same concepts:
  <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- **`meta-llama/Llama-3.1-8B-Instruct`** — an alternative 8B for the smaller-GPU path (gated; accept the license on HF first):
  <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
- **The vendor fallback** — Anthropic **`claude-haiku-4-5`** (cheap, fast) or **`claude-sonnet-4-6`** for the LiteLLM `fallbacks` leg. You only spend a few cents proving failover works.

## Tools you'll use this week

- **`vllm`** — `pip install vllm` (on the GPU box, in a fresh venv with matching CUDA). `vllm serve <model>` brings up the OpenAI-compatible server on `:8000`.
- **`litellm[proxy]`** — `pip install "litellm[proxy]"`. `litellm --config config.yaml` runs the router on `:4000`.
- **`openai`** — `pip install openai`. The client; point `base_url` at vLLM or at LiteLLM.
- **`httpx`** — `pip install httpx`. The async HTTP layer for the load generator.
- **`vllm bench serve`** — ships with vLLM; the built-in serving benchmark you cross-check your hand-rolled load generator against.
- **`nvidia-smi` / `nvtop`** — watch GPU utilization and VRAM live while the sweep runs; this is how you *see* the batch fill up.

## A note on the workload

The benchmark and the cost math run against a **fixed synthetic chat workload** — a set of ~200 prompts with a controlled prompt-length and output-length distribution, so "throughput" and "cost-per-million" mean something reproducible. Real traffic has a different shape (longer prompts, more variance), and the lecture is explicit that the break-even number moves with the workload — which is exactly why you measure *your* workload, not a vendor's benchmark. The mini-project's harness takes the workload as a config so you can re-run the sweep against the capstone's real prompt distribution in week 23.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Prefill** | The forward pass over the whole prompt at once; compute-bound; produces the first token + the prompt's KV cache. |
| **Decode** | Generating one output token per step; memory-bandwidth-bound; where batching pays off. |
| **KV cache** | The stored keys/values for every token so far, so you don't recompute attention; the real VRAM bottleneck. |
| **PagedAttention** | Storing the KV cache in fixed-size non-contiguous blocks (like OS paging) so you waste almost no VRAM. |
| **Continuous batching** | Admitting/evicting sequences every scheduler step instead of every request; the 20–30× throughput win. |
| **`--gpu-memory-utilization`** | Fraction of VRAM vLLM claims; more → bigger KV cache → higher batch ceiling. |
| **`--max-num-seqs`** | The maximum number of sequences in a batch; the batch-size ceiling. |
| **Tensor parallel** | Sharding one model's weights across N GPUs so a too-big model fits; `--tensor-parallel-size`. |
| **Prefix caching** | Reusing the KV cache of a shared prompt prefix across requests; `--enable-prefix-caching`. |
| **Speculative decoding** | A small draft model proposes tokens; the big model verifies them in one pass; a latency lever. |
| **Throughput** | Total tokens/sec across all concurrent requests; what self-hosting economics turn on. |
| **p50 / p95 latency** | Median / 95th-percentile per-request time; the user-facing number that batching trades against throughput. |
| **Break-even volume** | The token volume above which self-hosting (fixed GPU $/h) beats a vendor (zero fixed, per-token price). |
| **LiteLLM** | An OpenAI-compatible proxy/router in front of N backends (self-hosted + vendor) with fallback. |

---

*If a link 404s, please open an issue so we can replace it.*
