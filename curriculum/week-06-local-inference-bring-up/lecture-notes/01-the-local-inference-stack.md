# Lecture 1 — The Local Inference Stack: Why, Prefill vs Decode, and the Three Engines

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can say why you'd self-host an LLM at all, explain the two phases of inference (prefill and decode) and why they have opposite performance characteristics, name the local-inference stack of 2026 (Ollama, llama.cpp, vLLM, plus SGLang/TensorRT-LLM/TGI/MLX placed by workload), and choose the right engine for a given job with reasons instead of vibes.

If you remember one sentence from this entire week, remember this one:

> **The fastest token is the one you do not generate.** Quantization, batching, and KV-cache reuse are not optimizations bolted on afterward — they are the product. A self-hosted LLM that ignores them is a space heater that occasionally answers questions.

There's a corollary you should tape next to it:

> **Pick the engine for the workload, not the benchmark.** The "fastest" engine on a single-stream microbenchmark can be the worst choice for a server taking fifty concurrent requests. The benchmark measured the wrong thing.

For five weeks you've called a model that lives somewhere else. This week you bring it home. Everything that follows is in service of one shift in stance: from "the model is a thing I call" to "the model is a thing I *run*, on hardware I understand, with numbers I can read."

---

## 1. Why local inference at all

Self-hosting an LLM is real work. Before you sign up for it, be honest about *why*. There are four good reasons and one bad one.

**Reason 1 — vendor independence.** The syllabus names this failure mode explicitly: the "vendor-locked graduate" who cannot operate without an OpenAI or Anthropic key. If your product dies when a vendor changes pricing, rate-limits you, deprecates a model, or has an outage, you don't have a product — you have a dependency. Local inference is the insurance. You may *choose* to run on a vendor for the frontier capability (that's a sound architecture — local for the easy 80%, vendor for the hard 20%, which is week 21's routing lesson), but you should be *able* to run without one.

**Reason 2 — data residency and privacy.** Some data cannot leave your network: medical records under HIPAA, legal documents under privilege, source code under an NDA, anything under GDPR data-residency rules. "We send it to a third-party API" is a non-starter for these. A model on your own hardware keeps the data on your own hardware. For a large class of enterprise buyers this is the *only* thing that matters, and it's why self-hosting is a growing market, not a hobbyist's corner.

**Reason 3 — cost at volume.** Vendor APIs charge per token. At low volume that's wonderful — you pay cents, no ops. At high *sustained* volume the arithmetic flips: a rented or owned GPU has a fixed hourly cost, and if you keep it busy, your cost-per-million-tokens can drop below the vendor's. The break-even depends on your utilization (an idle GPU is pure waste) and is the subject of week 19's break-even memo. The point this week: you can't *reason* about that break-even until you can measure your own throughput, which is what the benchmark harness gives you.

**Reason 4 — latency and control.** A local model on a local GPU has no network round-trip to a vendor, no queue behind other tenants, no surprise rate limit at 2 AM. You control the batching, the quantization, the context length. For latency-sensitive paths (a voice agent, an autocomplete) that control is worth a lot.

**The bad reason — "it's cooler."** Self-hosting because it feels more hardcore is how you end up paying for an idle A100 to serve five requests a day, when a vendor API would have cost you a dollar a month and zero ops. Self-host when one of the four real reasons applies. Otherwise, call the API and spend your engineering on the product. The senior move is knowing *which* situation you're in.

> **The decision, in one line:** self-host for independence, residency, volume, or latency — not for vibes. And whichever you choose, be *able* to do the other.

---

## 2. The two phases of inference — and why every number splits along this seam

Here is the single most clarifying idea in the week. Inference is not one thing; it's two phases with opposite performance characteristics. Confuse them and every benchmark number you read will mislead you.

**Phase 1 — prefill.** When you send a prompt, the model processes *the entire prompt at once*, in parallel, in one big forward pass. It computes the attention and feed-forward activations for all N prompt tokens simultaneously and builds the **KV cache** (the keys and values for every prompt token, which decode will reuse). Prefill is **compute-bound**: the GPU's matrix-multiply units are the bottleneck, and you're doing a lot of FLOPs over a lot of tokens at once. Prefill sets your **time-to-first-token (TTFT)** — the user sees nothing until prefill finishes. A 4,000-token prompt has a noticeably longer prefill than a 100-token one; that's why stuffing a huge context makes the *first* token slow.

**Phase 2 — decode.** After prefill, the model generates output **one token at a time**. To produce each new token, it does a forward pass over a *single* new token — but to do that pass, it must read *the entire model's weights* out of memory. A 7B model in FP16 is ~14 GB; every single decoded token reads ~14 GB from VRAM. Decode is therefore **memory-bandwidth-bound**: the bottleneck is how fast the GPU can stream its weights, not how many FLOPs it can do. Decode sets your **tokens-per-second** — the speed of the streaming output.

Why this matters, concretely:

- **Quantization helps decode the most.** Going FP16 → 4-bit cuts the weights from ~14 GB to ~4 GB, so each decoded token reads ~3.5× less memory — roughly 3.5× faster decode. It helps prefill less (prefill is compute-bound, and 4-bit matmuls aren't 3.5× faster than 16-bit on most hardware). So when someone says "quantizing made it 3× faster," ask: *the decode or the prefill?* It's almost always the decode.
- **Batching helps decode by amortizing the weight read.** In decode, reading the 14 GB of weights is the cost. If you decode one token for *one* request, you read 14 GB to produce 1 token. If you decode one token for *eight* requests in a batch, you read 14 GB *once* and produce 8 tokens. The weight read is amortized across the batch. This is why concurrency multiplies throughput — and it's the entire reason vLLM exists (§5, and Lecture 2).
- **A long prompt punishes TTFT (prefill); a long *output* punishes total latency (decode).** They're different costs. A RAG prompt with 6,000 tokens of retrieved context has a heavy prefill (slow first token) but if it answers in 50 tokens, the decode is cheap. A short-prompt creative-writing request has a fast first token but a long decode. Knowing which phase dominates *your* workload tells you which optimization to reach for.

> **The seam:** prefill is compute-bound and sets TTFT; decode is memory-bandwidth-bound and sets tokens/sec. Every performance lever in this week pushes on one phase or the other. Name the phase before you reason about the number.

A quick mental model you'll use all week: imagine the GPU as a worker who has to walk to a giant filing cabinet (VRAM) to read the model's weights. **Prefill** is "do a huge pile of math on the desk" — the desk (compute) is the limit. **Decode** is "walk to the cabinet, read everything, walk back, write one word, repeat" — the *walk* (memory bandwidth) is the limit, and the word you write is tiny compared to the walk. Quantization makes the cabinet smaller (less to read per walk). Batching means writing eight words per walk instead of one. That picture predicts every result in the bakeoff.

---

## 3. Ollama — the lowest-friction local runtime

**Ollama** is where everyone starts, and for good reason: it has the least friction of anything in the stack. Install it, run one command, and you have a model serving locally.

```bash
# Install (Linux); macOS has a .app. Then pull and run a model:
ollama pull qwen2.5:7b
ollama run qwen2.5:7b "Explain prefill vs decode in two sentences."
```

Under the hood Ollama is a wrapper around **llama.cpp** (§4) with a model registry, automatic GGUF download, sensible defaults, and a daemon. Its three superpowers:

1. **The registry.** `ollama pull qwen2.5:7b` fetches a curated, quantized GGUF without you hunting on Hugging Face for the right file. It picks a reasonable quant (usually Q4_K_M) by default. This is the "it just works" that makes Ollama the right first tool.
2. **The Modelfile.** A `Modelfile` lets you bake a system prompt, parameters, and a base model into a named model — `ollama create my-agent -f Modelfile` — so your team shares one reproducible config.
3. **The OpenAI-compatible endpoint.** `ollama serve` exposes an API at `http://localhost:11434`, including an OpenAI-compatible surface at `/v1`. So you can point the *same* `openai` Python client at Ollama that you point at vendors:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key is ignored
resp = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Say hi in five words."}],
)
print(resp.choices[0].message.content)
```

That `base_url` swap is the thread through the whole week: one client, three engines. Your week-5 agent gets a `--base-url` and runs against any of them.

**Where Ollama is the right tool:** local iteration, prototyping, your laptop, a single-user demo, Apple Silicon (it uses Metal well). It's the fastest path from zero to a talking model.

**Where Ollama is the *wrong* tool:** a production server taking concurrent traffic. Ollama serializes (or only lightly batches) requests by default — it's optimized for one user at a time, not a hundred. For throughput under concurrency you reach for vLLM. Using Ollama as your production serving layer is the classic "it worked on my laptop, it falls over at ten users" mistake. **Ollama is for iteration; vLLM is for serving.** Hold that distinction.

---

## 4. llama.cpp — the portable substrate that runs anywhere

**llama.cpp** (Georgi Gerganov's project, the `ggml-org/llama.cpp` repo) is the C/C++ inference engine underneath half the local-LLM world, Ollama included. Its defining virtue is **portability**: it runs on CPU, on Apple Metal, on CUDA, on Vulkan, on ROCm — on a Raspberry Pi, a MacBook, a Linux server, an H100. If a device can do arithmetic, llama.cpp can probably run a model on it. There is no other engine with this reach.

It runs models in the **GGUF** format (its own quantized file format, covered in Lecture 2). You either download a GGUF from Hugging Face or convert and quantize one yourself with the bundled tools.

```bash
# Build with Metal (Mac) or CUDA (Linux); then run the server:
# (build: see the README for your platform)
llama-server \
  -m ./qwen2.5-7b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 8192            # context length (sizes the KV cache)
```

`llama-server` exposes an OpenAI-compatible endpoint on `:8080/v1`, so — again — the same client talks to it:

```python
client = OpenAI(base_url="http://localhost:8080/v1", api_key="llamacpp")
```

llama.cpp also ships the tools you'll use directly this week:

- **`llama-quantize`** — convert an FP16 GGUF to Q4_K_M / Q5_K_M / Q8_0 yourself (the quant stretch goal).
- **`llama-bench`** — micro-benchmark prefill (`pp`, prompt-processing) and decode (`tg`, text-generation) tokens/sec across quant levels and thread counts. This is the cleanest way to *see* the prefill/decode split in numbers, and you'll use it in the exercises.

**Where llama.cpp is the right tool:** anywhere portability matters — CPU-only boxes, Macs, edge devices, mixed fleets, or just "I want one binary that runs the model with no Python and no CUDA." It's also the *honest CPU baseline*: when you want to know "how slow is this model with no GPU at all," llama.cpp on CPU is the answer.

**Where llama.cpp is weak:** like Ollama (which wraps it), single-llama.cpp-process serving is not built for high *concurrency*. It batches modestly but it is not vLLM. For one user or a small fleet it's superb; for a high-QPS server you move to vLLM. The mental model: **llama.cpp is the portable substrate; vLLM is the throughput engine.**

---

## 5. vLLM — the high-throughput serving engine

**vLLM** is the open serving primitive for *concurrency*. Where Ollama and llama.cpp shine at one-user-at-a-time, vLLM is built to keep a GPU saturated with *many* concurrent requests — which is what a real server does. Its two headline innovations (detailed in Lecture 2) are **continuous batching** and **paged attention**; together they let one GPU serve dozens of concurrent users at throughput that single-stream engines can't touch.

```bash
# CUDA required. One command starts an OpenAI-compatible server on :8000.
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

And, predictably, the same client:

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="vllm")
```

The thing to internalize now (the *why* is Lecture 2): on a **single-stream** benchmark — one request at a time — vLLM may look only comparable to llama.cpp, sometimes slower to start (it has heavier initialization). But raise the concurrency to 8, 32, 128 and vLLM's aggregate throughput climbs while llama.cpp's flattens, because vLLM keeps adding requests to the running batch and amortizing the weight read (§2), while the single-stream engines serialize. **The whole point of vLLM is the shape of the throughput-vs-concurrency curve, not the single-stream number.** This is exactly why "pick the engine for the workload, not the benchmark" is the corollary of the week: if you benchmark at concurrency 1, you will mis-rank the engines for a serving workload.

**Where vLLM is the right tool:** a serving endpoint, a production API, anything taking concurrent traffic, the capstone's local tier. It's the answer when "how many users can this GPU serve?" is the question.

**Where vLLM is the wrong tool:** your laptop with no NVIDIA GPU (it's CUDA-first; CPU and other backends exist but are not its strength), or a quick one-off prompt where Ollama's friction-free `ollama run` is faster to reach for. You don't `vllm serve` to ask one question; you `vllm serve` to *serve*.

> **The three-engine mental model, locked in:**
> - **Ollama** — friction-free iteration. One user. Your laptop. The first thing you reach for.
> - **llama.cpp** — portable substrate. Runs anywhere. The honest CPU baseline and the edge story.
> - **vLLM** — throughput under concurrency. The serving primitive. The production answer.

---

## 6. The rest of the stack, placed by workload

The three above are the spine of the week. The syllabus names four more; you should be able to *place* each one — know the workload it wins — without necessarily running it this week.

**SGLang.** A fast serving engine (like vLLM) that wins specifically on **structured-output-heavy and complex-prompting workloads**. Its RadixAttention aggressively caches shared prefixes, so when you have many requests that share a long system prompt or a tree of related prompts (agents, structured extraction, grammar-constrained decoding), SGLang's prefix reuse pulls ahead. Reach for it when your workload is grammar/JSON-heavy or has lots of shared-prefix structure (which ties back to week 2's grammar-constrained decoding).

**TensorRT-LLM.** NVIDIA's library for compiling models into **optimized CUDA kernels** for NVIDIA GPUs. It wins on raw per-GPU latency and throughput on H100/H200 when you've invested in NVIDIA-specific engine builds — the last microseconds. The cost is build complexity and NVIDIA lock-in; it's the most opinionated, highest-ceiling option. You meet it properly in week 20 (the NVIDIA stack); this week, place it as "when you need every microsecond on NVIDIA hardware and will pay in build complexity to get it."

**TGI (Text Generation Inference).** Hugging Face's production server. Pragmatic when you already **live in the HF ecosystem** — your models, tokenizers, and tooling are HF-native and you want a serving layer that matches. Feature-competitive with vLLM for many workloads; the choice is often about which ecosystem you're standing in.

**Apple MLX.** Apple's array framework with `mlx-lm` for inference. The **Mac-native** path: it targets Apple Silicon's unified memory directly and can beat llama.cpp's Metal backend on some Mac workloads. If your deployment target *is* a Mac (a desktop app, on-device inference on Apple hardware), MLX is the first-class path. On a Mac, it's a real alternative to llama.cpp-Metal — measure both.

> **The placement table (commit this):**
> - **SGLang** → structured/grammar/shared-prefix workloads (RadixAttention).
> - **TensorRT-LLM** → every-microsecond NVIDIA serving, at the cost of build complexity (week 20).
> - **TGI** → you live in the Hugging Face ecosystem.
> - **MLX** → your target is Apple Silicon.

You are not asked to run all seven engines this week. You're asked to run *three* (Ollama, llama.cpp, vLLM) deeply and *place* the other four correctly. An engineer who can say "this is a high-concurrency NVIDIA serving job, so vLLM or TensorRT-LLM; it's structured-output-heavy, so look hard at SGLang; the target is a Mac app, so MLX" is reasoning about the stack, not reciting it.

---

## 7. One model, three engines — the discipline

The exercises and the bakeoff hold **one model fixed** (Qwen2.5-7B-Instruct, or Llama-3.1-8B if you prefer) and vary only the *engine* (and, in Lecture 2, the *quantization*). This is the same one-variable-at-a-time discipline that runs through all of C23 — the chunking A/B in week 8, the vector-store bakeoff in week 10. The reason is identical: if you change the model *and* the engine, the throughput delta could be either, and you've learned nothing you can defend.

So: pick your 7B on Monday. Pull it as a GGUF for Ollama and llama.cpp, and as the FP16 or AWQ build for vLLM. Hit all three with the *same* prompt set through the *same* OpenAI client (just a different `base_url`). Now the only thing that differs is the engine, and the numbers mean something.

```python
ENGINES = {
    "ollama":   "http://localhost:11434/v1",
    "llamacpp": "http://localhost:8080/v1",
    "vllm":     "http://localhost:8000/v1",
}

def client_for(engine: str) -> OpenAI:
    return OpenAI(base_url=ENGINES[engine], api_key=engine)  # key ignored locally
```

That dictionary is the whole bring-up, abstracted. Three engines, one model, one client, one prompt set — and the benchmark (Lecture 2 §6, Exercise 3) reads the difference in tokens/sec, TTFT, p95, and VRAM. The "served it yourself" promise is exactly this: the same model talking back to you from three engines you stood up, with numbers that tell you which to ship.

---

## 8. GPU memory math — does the model fit, and on what?

Before you `vllm serve` anything you should be able to answer, on paper, "will this fit, and on which GPU?" The arithmetic is short and you should be able to do it in your head. VRAM has three consumers:

**1. Weights.** `params × bytes_per_param`. At FP16 that's 2 bytes; at 8-bit, 1 byte; at 4-bit, ~0.5 bytes (plus a little for the quant's scales/zero-points). So a 7B model:

- **FP16:** 7e9 × 2 ≈ **14 GB**.
- **8-bit:** 7e9 × 1 ≈ **7 GB**.
- **4-bit:** 7e9 × 0.5 ≈ **3.5–4 GB** (call it ~4 GB with overhead).

**2. KV cache.** The hidden VRAM cost from Lecture 2 §4 — `2 × layers × context × hidden_dim × batch × bytes`. At long context and high concurrency it can equal or dwarf the weights. You budget it at the concurrency and context you'll actually serve, not at one stream.

**3. Activations + overhead.** The transient tensors of a forward pass, the CUDA context, the framework's own footprint, and (for vLLM) the pre-reserved paging pool. A practical rule: leave **~2 GB of headroom** on top of weights + KV cache, and never plan to fill 100% of VRAM — `--gpu-memory-utilization 0.90` exists because the last 10% is the safety margin between "serving" and "OOM at peak."

**Worked example — a 7B, FP16 vs 4-bit, on a 24 GB card (L4/A10/RTX 4090).**

```
FP16:  weights 14 GB + KV cache (8k ctx, modest batch) ~4 GB + overhead ~2 GB ≈ 20 GB
       → fits on 24 GB, but little room to raise concurrency before OOM.
4-bit: weights  4 GB + KV cache ~4 GB + overhead ~2 GB ≈ 10 GB
       → fits with 14 GB to spare → you can serve far more concurrent sequences
         on the SAME card, because the freed VRAM becomes KV-cache room.
```

That second line is the whole argument for quantizing a *server*, not just a laptop: the bytes you save on weights don't vanish, they **convert into concurrency** (more KV-cache pages → more in-flight requests → higher aggregate throughput, §5 and Lecture 2 §2).

> **Pick-a-GPU table (2026, single-GPU sizing):**
> - **L4 24 GB / A10 24 GB** — the efficient serving workhorses; a 7B at 4-bit with healthy concurrency, or FP16 with modest concurrency. Cheap to rent, low power.
> - **RTX 3090 / 4090 24 GB** — the same 24 GB as consumer cards; great for dev and single-box serving (no data-center NVLink, but plenty for a 7B–14B).
> - **A100 40 GB** — a 7B FP16 with real concurrency, or a 13B–14B comfortably; the older data-center default.
> - **A100 80 GB / H100 80 GB** — a 70B at 4-bit (~40 GB weights) with room for KV cache, or a 7B–34B with very high concurrency. The H100's higher memory bandwidth directly buys **decode** speed (§2: decode is bandwidth-bound).

Two corollaries you should internalize. First, a **70B at FP16 is ~140 GB of weights** — it does not fit on a single 80 GB GPU, so you either quantize it (4-bit ≈ 40 GB, fits on one H100) or shard it across GPUs (tensor parallelism, `--tensor-parallel-size`). Second, when decode feels slow on a card that "fits," the bottleneck is usually **memory bandwidth**, not capacity — an H100 decodes faster than an A100 holding the *same* model because it streams the weights faster, exactly as §2 predicts.

---

## 9. The OpenAI-compatible API — the load-bearing standardization

You've now seen the same three lines four times: point an `openai` client at a `base_url`, get tokens back. That repetition is not a convenience — it is the single most important piece of glue in the local stack, and it deserves to be named as a design principle.

Every serious engine — Ollama, llama.cpp's `llama-server`, vLLM, SGLang, TGI — exposes the **same** `/v1/chat/completions` (and `/v1/completions`, `/v1/embeddings`, `/v1/models`) surface that OpenAI's API defined. Because the *wire format* is shared, your client code is engine-agnostic:

```python
# The ONLY thing that changes between a vendor and three local engines is base_url.
client = OpenAI(base_url=ENGINES[engine], api_key="local")
resp = client.chat.completions.create(model=MODEL, messages=msgs, stream=True)
```

This is what makes "one model, three engines" (§7) *mechanically* possible, and it is what lets **your week-5 agent run locally with no rewrite**: the agent was written against the OpenAI client; give it `--base-url http://localhost:8000/v1` and it now drives a vLLM-served Qwen2.5-7B instead of a frontier model. The abstraction held. That is the "served it yourself" promise cashed out in code.

**What the standard covers well:** the chat/completion request shape, **streaming** (server-sent events, token-by-token), `temperature`/`top_p`/`max_tokens`, stop sequences, and `usage` token accounting. Swap `base_url` and these all behave.

**Where the abstraction leaks — and you must test, not assume:**

- **Tool / function calling.** The *schema* (`tools=[...]`, `tool_choice`) is standardized, but **support and reliability are not uniform**. A frontier model emits clean, well-formed tool calls almost every time. A local **7B** is weaker: it may mis-format arguments, hallucinate tool names, or ignore `tool_choice="required"` — and the *engine* matters too, because tool-call parsing is engine-side. vLLM needs an explicit `--enable-auto-tool-choice` and a `--tool-call-parser` matched to the model's chat template (e.g. the Hermes or Mistral parser); llama.cpp and Ollama have their own grammar/template handling. The lesson: **tool use is the first thing that breaks when you swap a frontier model for a local 7B**, and it breaks at *both* the model and the engine layer. Test your agent's actual tool calls against the local model before you trust them — this is exactly where grammar-constrained decoding (week 2, and SGLang §6) earns its keep, forcing valid JSON the small model couldn't reliably produce on its own.
- **Structured outputs / JSON mode.** Supported, but via engine-specific flags (vLLM's `guided_json`/outlines backend, llama.cpp's GBNF grammars). The OpenAI `response_format` field is increasingly honored, but verify it on *your* engine.
- **Model names and capabilities.** `model="qwen2.5:7b"` (Ollama) vs `model="Qwen/Qwen2.5-7B-Instruct"` (vLLM) — the *string* differs per engine even for the same weights, so your config maps a logical name to each engine's identifier.

> **The standardization, stated plainly:** the OpenAI-compatible surface means *one client, any backend* — which is what makes the bakeoff fair and the agent portable. But "compatible" covers the happy path (chat + streaming) far better than it covers **tool use and structured output**, which depend on both the model's competence and the engine's parser. Standardize your client; *verify* your tool calls.

---

## 10. Model formats and where your disk went

A practical aside that saves a lot of confused `du -sh`: the same model exists in several **formats**, and they live in different places on disk.

- **safetensors (the HF format).** The modern default for an unquantized model on Hugging Face: a directory of `*.safetensors` weight shards plus `config.json`, `tokenizer.json`, and the chat template. "Safe" because it's a flat tensor dump with no arbitrary-code-execution risk (unlike old PyTorch `.bin` pickles). This is what **vLLM, TGI, and transformers** load, and what AWQ/GPTQ quants are also packaged as (quantized safetensors + a quant config).
- **GGUF.** llama.cpp's single-file format — *one* file that bundles weights, tokenizer, metadata, and the quant scheme (Q4_K_M, etc.). This is what **llama.cpp and Ollama** run. Self-contained and portable, which is why it's the edge/CPU/Mac format.

**How the tools connect the formats:**

```bash
# llama.cpp: convert an HF safetensors model to GGUF, then quantize it.
python convert_hf_to_gguf.py ./Qwen2.5-7B-Instruct --outfile qwen-f16.gguf
llama-quantize qwen-f16.gguf qwen-q4_k_m.gguf Q4_K_M
```

```dockerfile
# Ollama: a Modelfile points at a (GGUF) base and bakes in config.
FROM ./qwen-q4_k_m.gguf
PARAMETER temperature 0.7
SYSTEM "You are a terse coding assistant."
# ollama create my-coder -f Modelfile  →  a named, reproducible model.
```

**Where the bytes actually live** — the answer to "where did my 14 GB go?":

- **The Hugging Face cache:** `~/.cache/huggingface/hub/` (overridable with `HF_HOME`). Every `from_pretrained` / `vllm serve <hf-id>` downloads shards here once and reuses them. This is usually the biggest directory on an inference box, and the first place to look when the disk fills.
- **The Ollama blob store:** `~/.ollama/models/` — a content-addressed store of `blobs/` (the GGUF layers, deduplicated by SHA) and `manifests/`. Two models that share a base layer share the blob; that's why pulling a second quant of the same model is smaller than the first.

> **The operational reality:** a single 7B can occupy disk **three times** — safetensors in the HF cache (~14 GB), a GGUF you converted (~14 GB f16, or ~4 GB quantized), and an Ollama blob (~4 GB). On a dev box doing the bakeoff this adds up fast. Know the two cache locations (`~/.cache/huggingface`, `~/.ollama/models`), point `HF_HOME` at a big disk, and `ollama rm` / prune the HF cache when the bring-up is done. "Where did my disk go" is always one of these two directories.

---

## 11. CPU and Apple-Silicon inference — the honest floor

vLLM assumes a CUDA GPU, but a huge amount of real local inference happens with *no* NVIDIA GPU at all — on a server CPU, or on a Mac. You should know what to actually expect, because the honest number is humbling and the right deployments are real.

**Why memory bandwidth dominates (not cores).** Decode is bandwidth-bound (§2): every token streams the whole model from memory. A CPU's problem isn't FLOPs — it's that DDR5 system RAM delivers maybe **50–100 GB/s**, while a data-center GPU's HBM delivers **2,000–3,000+ GB/s**. That ~20–40× bandwidth gap is *directly* the decode-speed gap. Adding CPU cores barely helps decode, because you're waiting on memory, not math. This is why "my 64-core server should be fast" is wrong: it's the RAM bandwidth, not the core count, that sets your tokens/sec.

**What to actually expect (order-of-magnitude, llama.cpp, a 7B at Q4_K_M):**

- **CPU-only (modern server/desktop, DDR5):** roughly **5–15 tokens/sec** decode — readable, but slow, and it craters under any concurrency. This is the *honest floor*: the "how slow with no GPU at all" baseline llama.cpp gives you (§4).
- **Apple Silicon (M-series, Metal/MLX):** roughly **20–60+ tokens/sec** for a 7B at 4-bit on an M3/M4 Pro/Max, because Apple's **unified memory** has far higher bandwidth (hundreds of GB/s, ~400–500+ GB/s on the Max tiers) than commodity DDR5 — and the GPU and CPU share it with no copy. That bandwidth is exactly why a Mac punches well above a same-priced CPU box for inference.

On the Mac, two paths (§6): **llama.cpp's Metal backend** (also what Ollama uses) and **Apple's MLX / `mlx-lm`**, which targets unified memory directly and can edge out llama.cpp-Metal on some Mac workloads:

```bash
# Apple-native inference with mlx-lm (4-bit, MLX-format weights):
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --prompt "Explain prefill vs decode in two sentences." --max-tokens 128
```

**When CPU/Mac inference is genuinely the right call** — and it often is:

- **On-device / edge:** the model ships *with* the app (a desktop tool, a kiosk, a phone), runs offline, no server bill, no network. GGUF-on-CPU and MLX-on-Mac own this.
- **Privacy / data residency (§1, Reason 2):** the strongest version of "the data never leaves" is "there is no server" — it runs on the user's own machine. Nothing beats on-device for this.
- **No-GPU development:** every engineer can run a 7B on their laptop for iteration before any GPU is rented. The bakeoff's CPU baseline isn't a toy — it's the floor every other number is measured against.
- **Low, bursty volume:** if you serve a handful of requests an hour, a CPU that costs nothing when idle beats a GPU you're paying for 24/7 to sit idle (§1, the "idle A100" trap).

> **The CPU/Mac rule:** don't expect GPU throughput — expect single-digit-to-low-double-digit tokens/sec on CPU, low-double-to-mid on Apple Silicon — and remember **bandwidth, not cores**, sets the number. But for **edge, on-device, privacy, and no-GPU dev**, CPU and Mac inference isn't a compromise, it's the *correct* deployment. "The fastest token is the one you don't generate" has an edge-case sibling: the cheapest token is the one you generate on hardware you already own.

---

## 12. Recap

You should now be able to:

- State **why you'd self-host** — vendor independence, data residency, cost at sustained volume, latency/control — and recognize the bad reason ("it's cooler") that leads to an idle GPU.
- Explain the **two phases of inference**: prefill (compute-bound, parallel over the whole prompt, sets TTFT) and decode (memory-bandwidth-bound, one token at a time, reads all the weights per token, sets tokens/sec) — and predict which optimization pushes on which phase.
- Place the **three core engines**: Ollama (friction-free iteration, one user), llama.cpp (portable substrate, the CPU baseline, the edge), vLLM (throughput under concurrency, the serving primitive) — and reach for the right one for a stated job.
- **Place the rest of the stack** by workload: SGLang (structured/shared-prefix), TensorRT-LLM (NVIDIA microseconds), TGI (HF ecosystem), MLX (Apple Silicon).
- Hold **one model fixed and vary only the engine**, through one OpenAI client with a swapped `base_url`, so the bakeoff measures the engine and nothing else.
- Do the **GPU memory math** — weights (`params × bytes/param`) + KV cache + ~2 GB overhead — to decide whether a 7B fits FP16 (~20 GB) or 4-bit (~10 GB) on a 24 GB card, and pick a GPU (L4/A10/3090/4090 24 GB, A100 40/80 GB, H100 80 GB) for the job; know that quantization's freed VRAM *converts into concurrency*.
- Treat the **OpenAI-compatible API** as the load-bearing standardization that makes one client drive any engine and the week-5 agent run locally — while *verifying* the parts that leak across engines (tool calls, structured output), which break first when a local 7B replaces a frontier model.
- Locate models on disk: **safetensors** (HF cache, `~/.cache/huggingface`) for vLLM/TGI vs **GGUF** (Ollama blob store, `~/.ollama/models`) for llama.cpp/Ollama, and how `convert_hf_to_gguf` + `llama-quantize` + the Modelfile connect them — so "where did my disk go" has an answer.
- Set honest expectations for **CPU and Apple-Silicon** inference (single-digit-to-low-double tok/s on CPU, low-double-to-mid on Apple Silicon), know that **memory bandwidth, not core count**, sets the number, and recognize when edge/on-device/privacy/no-GPU-dev make CPU or Mac the *right* deployment.

Next: the levers that make these engines fast — quantization (the bits-for-speed trade), continuous batching and paged attention (the concurrency multiplier), speculative decoding (more tokens per big-model pass) — and the honest benchmark that compares engines without lying to you. Continue to [Lecture 2 — Quantization, Batching, and Serving](./02-quantization-batching-and-serving.md).

---

## References

- *Efficient Memory Management for LLM Serving with PagedAttention* — Kwon et al., 2023 (the vLLM paper): <https://arxiv.org/abs/2309.06180>
- *vLLM documentation* (the OpenAI-compatible server, the serving primitive): <https://docs.vllm.ai/en/latest/>
- *llama.cpp* (the portable substrate, `llama-server`, `llama-bench`, `llama-quantize`): <https://github.com/ggml-org/llama.cpp>
- *Ollama* (the friction-free runtime + registry): <https://github.com/ollama/ollama>
- *SGLang* (structured-workload serving, RadixAttention): <https://github.com/sgl-project/sglang>
- *Text Generation Inference (TGI)* (HF-ecosystem serving): <https://github.com/huggingface/text-generation-inference>
- *Apple MLX / mlx-lm* (Apple Silicon native inference): <https://github.com/ml-explore/mlx-lm>
- *Continuous batching, explained* (Anyscale): <https://www.anyscale.com/blog/continuous-batching-llm-inference>
- *vLLM — OpenAI-compatible server and tool calling* (`--enable-auto-tool-choice`, `--tool-call-parser`): <https://docs.vllm.ai/en/latest/features/tool_calling.html>
- *Hugging Face — the safetensors format* (the safe weight-serialization standard): <https://github.com/huggingface/safetensors>
- *llama.cpp — `convert_hf_to_gguf.py` and the GGUF format* (HF → GGUF conversion): <https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py>
- *Ollama — Modelfile reference* (baking a base model, system prompt, and parameters into a named model): <https://github.com/ollama/ollama/blob/main/docs/modelfile.md>
