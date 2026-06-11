# Week 19 — vLLM in Production

Welcome to the week your self-hosting stops being a science project and starts being an economic argument. For eighteen weeks you have called models — local Ollama models in week 6, vendor APIs everywhere else — and mostly you have not had to care what a GPU was doing while you waited for tokens. This week you care. You stand up **vLLM** on a real GPU, you serve a 14B model behind an OpenAI-compatible endpoint, you put a **LiteLLM** router in front of it, and you push concurrency from 1 to 128 until the numbers tell you whether self-hosting beats paying a vendor for *your* workload. By Friday you can look at any inference deployment and say, with a chart, what its throughput is, where it saturates, and what a million tokens actually cost.

This is week 1 of **Phase IV — Production AI & Capstone**, and it is the week the capstone serving tier is born. The syllabus capstone serves its local 7B/13B tier on "a self-hosted vLLM cluster with continuous batching." This week is where you learn what that sentence costs and how to make it true. Everything here feeds week 23 (you deploy this cluster for real) and week 24 (you kill a replica on purpose and watch the router fail over).

The one sentence to internalize before you read another line:

> **Continuous batching is the throughput multiplier that makes self-hosting feasible. Without it, you are paying for idle GPU.**

Here is why that is not marketing. A GPU serving one request at a time is a Ferrari in a school zone — the hardware can do thousands of token-generations in parallel, but a naive server makes it wait for one user's sequence to finish before it starts the next. The GPU sits 90% idle, you pay for 100% of it, and your cost-per-token is terrible. vLLM's two core ideas — **PagedAttention** (store the KV cache in non-contiguous pages, like virtual memory, so you waste almost no VRAM) and **continuous batching** (swap finished sequences out and new ones in *every step*, not every request) — are what turn that idle GPU into a server that does 20–30× the throughput of the naive version. The whole week is making that multiplier real and measuring it.

There is a corollary worth taping next to it:

> **You do not self-host to save money on a low-volume workload — the break-even is a volume number, and you must compute it before you commit a GPU.** A vendor API at $3/$15 per million tokens has zero fixed cost; an H100 has a fixed hourly burn whether it serves one request or a thousand. Self-hosting wins *above* a crossover volume, and "above" is something you calculate, not something you assume.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** vLLM's architecture from the outside — the prefill/decode split, why decode is memory-bandwidth-bound, PagedAttention's page table for the KV cache, and continuous (a.k.a. in-flight) batching that swaps sequences in and out every scheduler step.
- **Stand up** a vLLM OpenAI-compatible server on a rented GPU, serving an open-weights model, and call it with the unmodified `openai` Python client — same code that talks to a vendor.
- **Configure** the serving knobs that matter: `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, `--tensor-parallel-size`, `--enable-prefix-caching`, and the speculative-decoding flags — and predict what each does to throughput, latency, and VRAM.
- **Put LiteLLM in front** of one or more vLLM replicas as a unified OpenAI-compatible proxy, with model aliases, multiple backends, and fallback routing — the same router the capstone uses to fail over from local to vendor.
- **Benchmark** throughput and latency at concurrency 1, 8, 32, and 128 with `vllm bench serve` (or a hand-rolled async load generator), and read the throughput-vs-concurrency curve honestly — where it scales linearly, where it saturates, where p95 latency falls off a cliff.
- **Compute** the self-hosted-vs-vendor break-even: amortize the GPU hourly cost over measured throughput to get cost-per-million-tokens, compare against the vendor's published price, and write the break-even volume memo the syllabus asks for.
- **Reason** about prefix caching and speculative decoding as throughput/latency levers — when each helps, when each is a trap, and how to measure the lift instead of assuming it.

## Prerequisites

This week assumes you have completed **C23 weeks 1–18**, or have equivalent fluency. Specifically:

- You finished **week 6** and have brought up a local model on Ollama / llama.cpp / vLLM at least once. You know what a quantization is, what VRAM is, and roughly how big a 7B vs a 14B model is in FP16 vs a 4-bit quant. This week goes deep on the vLLM path you touched there.
- You are comfortable on Linux with Docker and the NVIDIA Container Toolkit, or you are comfortable renting a GPU instance (RunPod, Lambda, Modal, vast.ai) and SSHing in. The labs run on a rented H100 for ~6 hours; budget ~$12–$18.
- You can read and write async Python (`asyncio`, `httpx.AsyncClient`) — the load generator is async, because that is the only honest way to drive concurrency.
- You know the `openai` Python SDK's `chat.completions.create` shape from earlier weeks. vLLM speaks that exact API, so your client code does not change when you point it at a self-hosted endpoint.

You **do** need GPU access this week — a rented H100 (~$2–$3/h) for the headline lab, or an A10/L4 (~$0.50/h) for a smaller model if budget is tight. The README's lab notes give a CPU-and-small-model fallback so the *concepts* (continuous batching, the throughput curve, the break-even math) are reachable even without an H100, but you should rent the GPU for at least the headline benchmark — reading a real saturation curve is the point.

## Topics covered

- **The inference two-step:** prefill (compute-bound, processes the whole prompt in one forward pass) vs decode (memory-bandwidth-bound, one token per step), and why the decode phase is where batching pays off.
- **PagedAttention:** the KV cache as the real bottleneck; why contiguous KV allocation wastes 60–80% of VRAM to fragmentation and over-reservation; how paging the cache (fixed-size blocks, a block table per sequence) recovers it and enables higher batch sizes.
- **Continuous batching:** the scheduler that admits and evicts sequences every step instead of every request; why it beats static batching by 20–30× on real traffic with mixed sequence lengths.
- **The OpenAI-compatible server:** `vllm serve`, the `/v1/chat/completions` and `/v1/completions` endpoints, the `openai` client pointed at `localhost:8000`, and what's the same vs different from a vendor.
- **Serving configuration:** `--gpu-memory-utilization` (how much VRAM to claim for the KV cache), `--max-model-len` (context budget), `--max-num-seqs` (the batch-size ceiling), and the trade-offs each forces.
- **Tensor and pipeline parallelism:** `--tensor-parallel-size` to shard one model across GPUs when it doesn't fit on one; pipeline parallel for multi-node; when you actually need either.
- **Prefix caching:** `--enable-prefix-caching` to reuse the KV cache of a shared prompt prefix (system prompt, few-shot block) across requests — the self-hosted analog of vendor prompt caching.
- **Speculative decoding:** a small draft model proposes tokens, the big model verifies them in one pass; the throughput lever and its memory cost; n-gram and EAGLE variants; when the accept rate makes it worth it.
- **Multi-replica + LiteLLM:** N vLLM replicas behind one LiteLLM proxy; model aliases, weighted routing, health checks, and vendor fallback — the capstone's router.
- **The economics:** GPU hourly cost ÷ measured throughput = cost-per-million-tokens; the break-even volume vs a vendor; the carbon and operational-overhead footnotes.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                          | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|----------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | PagedAttention + continuous batching; the inference two-step   |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | `vllm serve`; the OpenAI-compatible endpoint; serving knobs    |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | LiteLLM router; multi-replica; fallback; prefix caching        |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Benchmarking methodology; the load generator; building the lab |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The concurrency sweep + break-even memo; serving clinic        |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                          |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                       |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                                | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The vLLM docs, the PagedAttention paper, LiteLLM docs, benchmarking guides, GPU-rental notes |
| [lecture-notes/01-paged-attention-and-continuous-batching.md](./lecture-notes/01-paged-attention-and-continuous-batching.md) | The inference two-step, the KV-cache bottleneck, PagedAttention, continuous batching, the serving knobs |
| [lecture-notes/02-litellm-routing-and-self-hosted-economics.md](./lecture-notes/02-litellm-routing-and-self-hosted-economics.md) | The OpenAI-compatible server, LiteLLM routing + fallback, prefix caching, speculative decoding, the break-even math |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-stand-up-vllm.md](./exercises/exercise-01-stand-up-vllm.md) | Bring up `vllm serve`, hit it with the `openai` client, read the startup log for KV-cache blocks |
| [exercises/exercise-02-load-generator.py](./exercises/exercise-02-load-generator.py) | Build an async load generator and measure throughput + p50/p95 at concurrency 1/8/32 |
| [exercises/exercise-03-litellm-router.py](./exercises/exercise-03-litellm-router.py) | Put LiteLLM in front of a backend, add a vendor fallback, prove failover works |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-concurrency-sweep.md](./challenges/challenge-01-concurrency-sweep.md) | The full concurrency sweep: 1/8/32/128, the throughput curve, cost-per-million, the break-even memo |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the one-page break-even memo |
| [mini-project/README.md](./mini-project/README.md) | The `crunchserve` benchmark harness — pluggable backends + the sweep + cost reporting |

## The "the GPU is actually busy" promise

C23 uses a recurring marker for every exercise that ends in a self-hosted server doing real, measured work *because* batching was on:

```
$ python sweep.py --backend vllm --model Qwen/Qwen2.5-14B-Instruct --concurrency 32
backend=vllm model=Qwen2.5-14B concurrency=32
  throughput: 2840 tok/s   p50: 1.9s   p95: 3.4s
  GPU util (measured): 94%   KV-cache blocks in use: 7120/7680
  cost: $2.50/h ÷ 2840 tok/s  ->  $0.244 / 1M output tokens  ✓  (vendor: $15.00/1M)
```

If that GPU-util line reads 30% instead of 94%, your batching is off (or your concurrency is too low to fill the batch), and your cost-per-million is 3× worse than it should be — you are paying for idle silicon. The point of week 19 is to make the GPU *busy* — to fill the continuous batch — and to prove it with a measured throughput-vs-concurrency curve and a cost number, not a vibe about how "vLLM is fast."

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **PagedAttention paper** (Kwon et al., 2023) until you can explain why a 4-byte-per-token KV cache fragmented across contiguous allocations wastes the majority of VRAM, and why fixed-size blocks fix it: <https://arxiv.org/abs/2309.06180>. Then find the "KV cache blocks" line in your own vLLM startup log and compute how many concurrent sequences your config can actually hold.
- Turn on **`--enable-prefix-caching`** and re-run the sweep with a long shared system prompt across all requests. Measure the throughput lift from KV reuse — and notice it only helps when the prefix is actually shared.
- Stand up **two vLLM replicas** of a smaller model and put LiteLLM in front with weighted round-robin. Kill one replica mid-sweep (this is a dry run for the week-24 chaos drill) and watch the router route around it. Measure the user-visible blip.
- Run **speculative decoding** (`--speculative-config` with an n-gram or a small draft model) and measure the accept rate and the latency change. Notice that it helps single-stream latency but can *hurt* throughput at high batch sizes — measure, don't assume.

## Up next

Week 20 takes the vLLM cluster you built here and puts the NVIDIA enterprise stack — **NeMo Inference, TensorRT-LLM, Triton, and NeMo Guardrails** — next to it on the same H100, so you can decide which serving story your capstone signs up for. You keep the LiteLLM router and the benchmark harness; week 20 swaps the backend behind them and re-runs your sweep. Push your `crunchserve` harness before you start it — week 20 points the same sweep at a NeMo/Triton backend and compares the curves.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
