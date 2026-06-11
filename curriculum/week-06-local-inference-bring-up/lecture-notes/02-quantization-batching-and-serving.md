# Lecture 2 — Quantization, Batching, and the Honest Benchmark

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can choose a quantization format (GGUF/AWQ/GPTQ/bitsandbytes/FP16) and state the quality/speed/VRAM trade-off it makes, explain continuous batching and paged attention well enough to say *why* vLLM serves many users on one GPU, describe speculative decoding and KV-cache reuse, and run a benchmark that compares engines honestly — measuring prefill and decode separately, p50/p95, VRAM, and throughput under concurrency, without fooling yourself.

Lecture 1 placed the engines. This lecture is about the **levers** that make them fast and the **measurement** that tells you whether a lever actually helped. The two have to be learned together: a lever you can't measure is a lever you can't tune.

> **The fastest token is the one you do not generate** — and the second fastest is the one a *quantized* model generates while *batched* with seven others, reusing a *cached* prefix. This lecture is those three clauses, in order.

---

## Part 1 — Quantization: trading bits for VRAM and speed

A model's weights are numbers. Train them and they're 16-bit floats (FP16 or BF16) — two bytes each. A 7B-parameter model is therefore ~14 GB of weights. **Quantization** stores those numbers in *fewer bits* — 8-bit, 4-bit, sometimes lower — to shrink the model. The two things you buy:

1. **Less VRAM.** A 4-bit 7B is ~4 GB instead of ~14 GB. That's the difference between fitting on a 6 GB laptop GPU and not fitting at all, or between one model per GPU and three.
2. **Faster decode.** Recall from Lecture 1 §2 that decode is *memory-bandwidth-bound*: every token reads the whole model. Read 4 GB instead of 14 GB and you read ~3.5× less per token, so decode runs ~3.5× faster — on the *same* GPU. This is the single biggest lever for single-stream speed.

And the one thing you pay: **a little quality.** Rounding 16-bit weights to 4-bit loses information. The miracle of modern quantization is *how little* you lose: a good 4-bit quant (Q4_K_M, AWQ) costs a few percent on most tasks — often within the noise of a benchmark — for a 3.5× decode speedup and a 3.5× VRAM cut. That trade is so favorable that **4-bit is the default**, not the exception, for self-hosted inference in 2026. You start at 4-bit and only go up (to 8-bit or FP16) if you measure a quality regression you can't accept.

### 1.1 The formats, placed

| Format | Where it runs | Bits | What it's for |
|---|---|---|---|
| **FP16 / BF16** | everywhere (GPU) | 16 | The unquantized baseline. Max quality, max VRAM. Your reference point. |
| **GGUF** (Q4_K_M, Q5_K_M, Q8_0, …) | llama.cpp / Ollama | 2–8 | The llama.cpp family. CPU+GPU+Metal. The portable quant. Q4_K_M is the workhorse. |
| **AWQ** | vLLM / GPU | 4 | Activation-aware: protects the weights that activations are most sensitive to. Strong 4-bit for GPU serving. |
| **GPTQ** | vLLM / GPU | 4 (3/8) | Layer-wise one-shot quant. The other common GPU 4-bit; AWQ often edges it on quality. |
| **bitsandbytes** (NF4/INT8) | transformers / GPU | 4/8 | On-the-fly quant at load time. Easiest to apply (one flag), not the fastest to serve. |

The mental map:

- **You're on llama.cpp/Ollama/CPU/Mac** → **GGUF**. Pick **Q4_K_M** as the default; go to **Q5_K_M** or **Q8_0** if you measure a quality regression and have the VRAM; the K-quants (the `_K_` ones) are the modern, better-than-legacy variants.
- **You're serving on vLLM/GPU** → **AWQ** (or GPTQ). These are GPU-native 4-bit; vLLM loads them directly and serves them fast.
- **You want the absolute baseline quality** → **FP16/BF16**, and pay the VRAM. Always benchmark *against* FP16 so you know exactly what quality you traded.
- **You're prototyping in plain transformers and just want it to fit** → **bitsandbytes** (`load_in_4bit=True`). Easy; not your production serving path.

### 1.2 The quality/size curve

The relationship between quantization level and quality is a curve you should be able to draw. Measured as **perplexity** (lower = better; how surprised the model is by held-out text) against **file size**:

```
perplexity
   ^
   |  *  Q2_K          (tiny, noticeably worse — the quality falls off a cliff here)
   |    *  Q3_K
   |        *  Q4_K_M  <- the knee: near-FP16 quality, ~1/3.5 the size. SHIP THIS.
   |           *  Q5_K_M
   |              *  Q6_K
   |                 *  Q8_0   (almost FP16 quality, ~half the size)
   |                    * FP16 (baseline)
   +-------------------------------------> file size
```

The shape is the lesson: from FP16 down to ~Q4, perplexity barely moves while size drops fast — that's free money. Below Q4 (Q3, Q2), perplexity starts climbing sharply — you're now paying real quality for marginal size. **Q4_K_M sits at the knee**: the point where you've captured almost all the size savings before the quality cost accelerates. That's why it's the default. You'll *draw this curve yourself* in Exercise 2 (estimated) and, as a stretch, *measure* it with `llama-quantize` + perplexity.

> **The discipline:** never report "I quantized it" without saying *to what* and *what quality you measured*. "Q4_K_M, perplexity within 2% of FP16, 3.4× smaller" is an engineering statement. "I quantized it, seems fine" is a vibe. The quant level is a tuned choice with a number behind it, exactly like chunk size in week 8.

### 1.3 The trap: quantizing the wrong thing, or comparing across models

Two common errors:

- **Comparing a quantized small model to an FP16 large model and crediting the quant.** If Qwen-7B-Q4 beats Llama-3B-FP16, that's a *model* difference, not a quant lesson. Hold the model fixed; vary only the quant. (One-variable-at-a-time, again.)
- **Forgetting the KV cache also takes VRAM.** Your VRAM budget isn't just weights. The KV cache grows with `context_length × batch_size` and can be gigabytes on its own at long context and high concurrency. A 4 GB quantized model can still OOM if you ask for 32 concurrent 32k-token sequences, because the *cache* blew the budget. This is exactly the problem paged attention (Part 2) solves — and the reason your VRAM measurement must include the cache, not just the weights.

---

## Part 2 — Continuous batching and paged attention: the concurrency multiplier

This is *why vLLM exists* and *why it serves many users where llama.cpp serves one*. Both ideas attack the same fact from Lecture 1 §2: in decode, reading the weights is the cost, so you want to produce as many tokens as possible per weight-read.

### 2.1 Static batching and its waste

The naive way to batch: collect N requests, run them together until *all* finish, then take the next N. The problem is that requests finish at *different times* — one wants 10 tokens, another wants 500. With static batching, the 10-token request's slot sits *idle* for 490 steps waiting for the 500-token request, because the batch can't move on until everyone's done. The GPU is reading weights for a batch that's mostly finished. Utilization tanks. This is the classic LLM-serving waste.

### 2.2 Continuous (in-flight) batching

**Continuous batching** (a.k.a. in-flight batching) fixes it: instead of a fixed batch that starts and ends together, the engine manages a *rolling* batch where requests are added and removed **token-by-token**. The instant a request finishes, its slot is freed and a *waiting* request takes its place — the batch never drains, the GPU never idles. New arrivals join the in-flight batch immediately rather than waiting for the current batch to clear.

The effect on the number that matters: aggregate throughput climbs with concurrency because the weight-read (the decode cost) is shared across however many requests are in flight *right now*, and that number stays high because finished requests are instantly replaced. This is the entire reason the vLLM throughput-vs-concurrency curve *keeps rising* where the single-stream engines flatten. The Anyscale write-up in resources.md visualizes it; the one sentence to keep:

> **Continuous batching keeps the GPU busy by replacing finished requests token-by-token, so the expensive weight-read is amortized across a batch that never drains.** That's the throughput multiplier.

### 2.3 Paged attention

Continuous batching needs somewhere to put each in-flight request's **KV cache**, and that's where **paged attention** comes in. Naively, you'd reserve a contiguous block of VRAM per request, sized for its *maximum possible* length — wildly wasteful, because most requests are far shorter, and the reserved-but-unused space fragments the GPU so you can't fit as many sequences as the VRAM should allow.

Paged attention borrows the operating-system trick of **virtual memory**: it stores each sequence's KV cache in fixed-size **pages** that don't have to be contiguous, allocated on demand as the sequence grows. No request reserves more than it's using; pages are handed out and reclaimed dynamically; fragmentation nearly vanishes. The payoff: you fit *many more* concurrent sequences in the same VRAM, which is exactly what continuous batching needs to keep the batch full. The two innovations are a pair — paging makes room, continuous batching uses it.

This is also why your VRAM measurement is subtle (Part 1 §1.3): with paged attention, KV-cache VRAM scales with *actual* concurrent sequence lengths, and vLLM pre-reserves a pool (`--gpu-memory-utilization`) to page within. Watch `nvidia-smi` as you raise `--max-num-seqs`: you'll see the cache pool fill — paging in action.

> **The pair:** paged attention stops the KV cache from fragmenting VRAM (OS-style paging); continuous batching keeps the GPU's decode busy by filling those pages with a never-draining batch. Together they are the difference between "one user per GPU" and "a hundred." (vLLM paper, arXiv 2309.06180.)

---

## Part 3 — Speculative decoding and KV-cache reuse

Two more levers, both about generating tokens more cheaply.

### 3.1 Speculative decoding

Decode is one-token-at-a-time, and each token costs a full big-model forward pass (a full weight-read). **Speculative decoding** breaks that one-per-pass limit: a small, fast **draft model** proposes the *next several* tokens cheaply; then the big model verifies all of them in a *single* forward pass. The big model checks "would I have produced these K tokens?" — for the prefix it agrees with, it accepts them all at once; at the first disagreement, it corrects and continues. If the draft is right most of the time, you get several tokens per big-model pass instead of one, and decode speeds up — *with no quality loss*, because the big model still has the final say on every token.

The lever is the **acceptance rate**: how often the big model agrees with the draft. A draft that's well-matched to the target (same family, e.g. Qwen-0.5B drafting for Qwen-7B) gets high acceptance and a real speedup; a poorly matched draft gets rejected often, and you've paid for the draft passes with little gain. So speculative decoding helps most on *predictable* text (code, structured output) and a *well-chosen* draft. vLLM supports it with a flag; the stretch goal has you turn it on, predict the lift from the acceptance rate, and measure.

### 3.2 KV-cache reuse (prefix caching)

When many requests share a **prefix** — the same long system prompt, the same few-shot examples, the same retrieved-context preamble — re-running prefill over that shared prefix for every request is wasted compute. **Prefix caching** (vLLM's automatic prefix caching; SGLang's RadixAttention) stores the KV cache for a shared prefix and *reuses* it across requests, so the second request that shares the prefix skips re-encoding it and jumps straight to its unique suffix. The TTFT for shared-prefix requests drops sharply.

This is why SGLang (Lecture 1 §6) wins on structured/agentic workloads: those workloads are *full* of shared prefixes (the same tool schema, the same system prompt, a tree of related sub-prompts), and aggressive prefix caching turns that redundancy into a speedup. It's also a reason to *design* your prompts with a stable prefix — put the constant system prompt and examples *first*, the variable user content *last*, so the cache hits.

> **Both levers reduce generated-token cost:** speculative decoding produces several tokens per big-model pass; prefix caching skips re-encoding shared context. "The fastest token is the one you don't generate" — prefix caching literally doesn't re-generate the shared prefix.

---

## Part 4 — KV cache and context length: the hidden VRAM cost

A short but load-bearing aside, because it's the most common "why did it OOM?" surprise. The KV cache stores keys and values for *every token in the context*, for *every layer*, for *every concurrent sequence*. Its size scales roughly as:

```
KV-cache bytes  ≈  2 (K and V) × layers × context_length × hidden_dim × batch_size × bytes_per_element
```

The two terms you control at serve time are **context_length** and **batch_size** (concurrency). Double the context, double the cache. Double the concurrency, double the cache. At long context *and* high concurrency, the cache can dwarf the (quantized) weights — a 4 GB model serving 64 concurrent 16k-token sequences can need *more* VRAM for the cache than for the weights. This is why:

- `--max-model-len` (context cap) and `--max-num-seqs` (concurrency cap) are your VRAM knobs in vLLM, and
- paged attention (Part 2) exists at all — to use that cache VRAM efficiently instead of reserving worst-case-contiguous blocks.

When you measure VRAM in the benchmark, measure it *at the concurrency and context you'll actually serve*, not at concurrency 1 — because the cache cost only shows up under load. A VRAM number from a single-stream test is a number that will betray you in production.

---

## Part 5 — The honest benchmark

Now the measurement, because everything above is only as good as your ability to tell whether it helped. A serving benchmark that lies is worse than no benchmark — it gives you false confidence in the wrong engine. Here's how to measure honestly.

### 5.1 Measure prefill and decode separately

From Lecture 1 §2: prefill and decode are different phases with different bottlenecks. A single "tokens/sec" number that blends them tells you nothing actionable. Report them apart:

- **Prefill throughput** (tokens/sec processing the prompt) and the resulting **TTFT** (time to first token).
- **Decode throughput** (tokens/sec generating the output), per request.

`llama-bench` does this natively (`pp` = prompt-processing/prefill, `tg` = text-generation/decode). For server engines, you derive TTFT from the first-token timestamp and decode rate from the inter-token timestamps.

### 5.2 Measure throughput *under concurrency*, not single-stream

This is the one that separates engineers from benchmarkers. The serving metric is **aggregate throughput** — total tokens/sec across *all* concurrent requests — at the concurrency you'll actually serve. Run the benchmark at concurrency 1, 8, 32, 128 and plot the curve. You'll see:

- **llama.cpp / Ollama**: aggregate throughput rises a little then *flattens* — they don't continuous-batch, so more concurrency just queues.
- **vLLM**: aggregate throughput keeps *climbing* with concurrency (until the GPU saturates), because continuous batching keeps adding requests to the in-flight batch.

If you benchmarked only at concurrency 1, you'd rank them wrong for a serving job. **Concurrency is the axis that reveals the engine's real character.**

```python
import asyncio, time
import httpx

async def one_request(client, url, prompt):
    t0 = time.perf_counter()
    r = await client.post(url, json={
        "model": "qwen2.5-7b", "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
    })
    dt = time.perf_counter() - t0
    out_tokens = r.json()["usage"]["completion_tokens"]
    return dt, out_tokens

async def bench_concurrency(url, prompts, concurrency):
    """Fire `concurrency` requests at once; report aggregate tokens/sec."""
    async with httpx.AsyncClient(timeout=120) as client:
        t0 = time.perf_counter()
        tasks = [one_request(client, url, p) for p in prompts[:concurrency]]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    total_tokens = sum(tok for _, tok in results)
    latencies = sorted(dt for dt, _ in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    return {
        "concurrency": concurrency,
        "aggregate_tok_s": total_tokens / wall,   # the serving metric
        "p50_s": p50, "p95_s": p95,
    }
```

That `aggregate_tok_s` at rising concurrency *is* the bakeoff's headline. The same code hits any of the three engines (just change `url`) — one harness, three backends, fair comparison.

### 5.3 Report p50 and p95, not just the mean

Latency is a distribution, not a number. The **mean** hides the tail; your slowest 1-in-20 users live at **p95**, and they're often the ones who complain or time out. A server with a great mean and a terrible p95 has a real problem — usually queueing or a too-large batch starving individual requests. Always report **p50** (the typical user) *and* **p95** (the unlucky user). A single mean is a benchmark hiding its tail.

### 5.4 Report VRAM at the serving point

Per Part 4: measure VRAM (`nvidia-smi`) *at the concurrency and context you serve*, so the KV-cache cost is included. A VRAM number from concurrency 1 understates the real footprint, and "it fit in single-stream" is how you OOM in production.

### 5.5 The fair-comparison checklist

Before you trust a cross-engine number, confirm:

- [ ] **Same model**, same quantization level where the format allows (and where it doesn't — GGUF vs AWQ — *say so*, because that's a confound, exactly like late-chunking's model swap in week 8).
- [ ] **Same prompt set** (same input lengths) and **same `max_tokens`** (same output lengths) for every engine.
- [ ] **Same concurrency points** (1/8/32/128), and the curve reported, not one point.
- [ ] **Prefill and decode separated**; **p50 and p95** both reported; **VRAM at serving concurrency**.
- [ ] **Warm-up runs discarded** (the first request pays for model load, JIT, cache cold — don't count it).

Tick those and your bakeoff is honest. Skip one and your "vLLM is 5× faster" might be "vLLM had a warmer cache and a different quant," which is a confession, not a result.

> **The "served it yourself" promise, made measurable:**
> ```
> engine=vllm quant=awq concurrency=32
>   prefill 4200 tok/s   decode 61 tok/s/req   aggregate 1950 tok/s
>   TTFT p50 0.18s p95 0.41s   VRAM 8.9 GB (incl. KV cache at c=32)
> ```
> Every number names its phase and its load. That's a deployment decision you can defend, not a vibe about which engine "felt fast."

---

## Part 6 — Putting it together: the deployment decision

The whole week converges on a decision you'll write in the milestone memo. Given a workload, you now reason:

1. **What's the workload?** One user iterating (→ Ollama), an edge/CPU/Mac target (→ llama.cpp/MLX), a concurrent server (→ vLLM), structured/shared-prefix-heavy (→ SGLang), every-microsecond NVIDIA (→ TensorRT-LLM, week 20).
2. **What quantization?** Start at 4-bit (GGUF Q4_K_M for llama.cpp, AWQ for vLLM); benchmark against FP16; go up only if you measure an unacceptable regression.
3. **What's the VRAM budget — *with the KV cache* at your real concurrency and context?** Cap `--max-model-len` and `--max-num-seqs` to fit.
4. **What's the throughput-vs-concurrency curve, and the p95 at your target QPS?** That's the number that says how many users one GPU serves — and feeds the cost-vs-vendor break-even (week 19).

Answer those four with *measured numbers from your benchmark* and you've done the engineering the week is built to teach: not "I ran a model locally," but "I served *this* model on *this* engine at *this* quant, and here is the throughput, the tail latency, and the VRAM that justify shipping it." The clause survived — the *token* survived — and you can prove it.

---

## Part 7 — How quantization actually works (under the table)

Part 1 told you *which* format to pick. This part tells you *why* 4-bit costs so little quality — because once you see the mechanism, the formats table stops being a list to memorize and becomes a set of choices you can reason about.

Start with the naive thing and watch it fail. A weight is an FP16 number; you want it in 4 bits, which can represent only **16 distinct levels** (2⁴). The simplest scheme: find the largest-magnitude weight in the whole tensor, divide the range into 16 even steps, and round every weight to the nearest step. Store a single **scale** (the step size) plus 4 bits per weight. To use a weight, you read its 4-bit level and multiply by the scale — *dequantize* back to roughly the original float.

The problem is **outliers**. Neural-net weight distributions are spiky: most weights cluster near zero, but a few are large. If one giant weight sets the scale for the whole tensor, then your 16 levels are spread across a huge range, and the *typical* near-zero weights — the vast majority — all collapse onto the same two or three levels. You've spent your precision representing a handful of outliers and starved everyone else. One global scale is why naive 4-bit is bad.

### 7.1 Per-block (group) scales — the K-quants' core idea

The fix is to **stop using one scale.** Chop the tensor into small **blocks** (groups) of, say, 32 or 64 contiguous weights, and give *each block its own scale* (and often a **zero-point**, an integer offset so the 16 levels can straddle an asymmetric range instead of being forced symmetric about zero). Now an outlier only blows up the scale of *its own* 32-weight block; every other block keeps a tight scale matched to its own near-zero weights, so its 16 levels land exactly where the mass of the distribution is. Precision goes where the weights actually are.

That is precisely what **GGUF's K-quants** do — the `_K_` in `Q4_K_M` is "block-wise (K-quant) with per-block scales," and the `_M` is a size/quality tier (the K-quants even quantize the *scales* themselves and mix a couple of bit-widths across a tensor to spend bits where they matter). **AWQ** and **GPTQ** likewise use **per-group / per-channel** scales rather than one global scale. The few extra bytes you spend on per-block scale metadata buy back almost all the quality the global scale threw away — that is the whole reason a good 4-bit quant lands within a couple of percent of FP16 instead of falling off a cliff.

> **The one-sentence mechanism:** 4-bit is cheap because the levels are placed *per small block*, so outliers can't ruin the precision of the many near-zero weights — one global scale fails; thousands of local scales succeed.

### 7.2 Activation-aware quantization — why AWQ protects salient weights

K-quants treat every block by the same rule. **AWQ** (Activation-aware Weight Quantization) adds a second insight: **not all weights matter equally**, and which ones matter is determined by the *activations* that flow through them, not by the weight magnitudes alone. A small fraction of weight channels — the ones multiplied by consistently large activations — dominate the layer's output; quantization error in *those* channels hurts far more than error in the rest.

AWQ finds those **salient channels** (by observing activation statistics) and scales them up before quantizing so they land on the fine part of the grid — effectively giving the important weights more effective precision — then scales the activations down to compensate, leaving the math equivalent. The result: the few channels that drive the output are protected, the error concentrates in channels that barely matter, and 4-bit AWQ holds quality better than a magnitude-only scheme at the same bit-width. That is why the formats table calls AWQ "activation-aware" and why it's the GPU-serving default — it spends its limited precision where the *forward pass* will actually feel it.

---

## Part 8 — Calibration and the data-dependence of PTQ

All of these are **post-training quantization (PTQ)**: you quantize an already-trained model, no retraining. But "activation-aware" (AWQ) and GPTQ's error-correction both need to *observe activations* — and activations only exist when you run data through the model. So these methods require a **calibration dataset**: a small sample (often a few hundred sequences) that you push through the model to measure which channels are salient (AWQ) or to compute the error each layer's quantization introduces so the next weights can compensate (GPTQ).

This is a real dependency with a real failure mode: **calibration mismatch.** If you calibrate AWQ/GPTQ on generic English web text and then serve a model that does, say, Python code or clinical notes or Japanese, the activation statistics at serve time differ from the ones you calibrated on. The method protected the channels that were salient *for the calibration data*, which may not be the channels salient for *your* domain — and you can eat a quality hit that never shows up if you only evaluate on data resembling the calibration set. The lesson: when you grab a pre-quantized AWQ/GPTQ build off Hugging Face, you've inherited *someone else's* calibration set, which may not match your workload.

**GGUF K-quants are calibration-free.** They use only the weights themselves (per-block min/max to set scales) — no data, no forward pass, no calibration set to mismatch. That's a genuine practical advantage and it shapes which quant to reach for:

- **GGUF Q4_K_M** — calibration-free, deterministic, domain-agnostic. The safe default, especially for an unusual or private domain where you can't easily assemble a representative calibration set. Nothing to get wrong.
- **AWQ / GPTQ** — typically a touch better at the same 4 bits *when the calibration data matches your workload*, and the GPU-serving format vLLM loads natively. Worth it for serving, but if you're quantizing yourself, **calibrate on data that looks like your traffic** — and if you're downloading a pre-quant, check what it was calibrated on, or measure on *your* domain before trusting it.

> **The practical rule:** unusual domain or no representative calibration data → reach for **calibration-free GGUF K-quants**. GPU serving with a matching calibration set → **AWQ**. Either way, the honest-benchmark discipline (Part 5) is what catches a calibration mismatch — *measure quality on your own data*, not on the quantizer's.

---

## Part 9 — When the model doesn't fit on one GPU: tensor and pipeline parallelism

Everything so far assumed the model fits on a single GPU. It often won't. An FP16 70B is ~140 GB of weights — more than an 80 GB H100 or A100 holds — and even quantized, the biggest open models plus their KV cache can overflow one card. When the model exceeds one GPU's VRAM, you **shard it across several**, and there are two ways to cut it. (This is mostly week 19's multi-GPU topic, but the concept belongs next to quantization because it's the *other* answer to "it won't fit.")

**Tensor parallelism (TP)** splits *each layer* across GPUs: a single matrix-multiply is partitioned column-wise (or row-wise) so every GPU does a *slice* of every layer, and they combine partial results with an **all-reduce** after each split op. Because all GPUs work on the *same* token at the *same* time, TP cuts per-token **latency** as well as fitting the model — but the all-reduce happens *many times per forward pass*, so the GPUs must talk constantly. That communication only stays cheap on a **fast interconnect**: **NVLink** between GPUs in one box, ideally. Run TP across GPUs connected only by slower PCIe (or across machines) and the all-reduce traffic becomes the bottleneck, eating the speedup. **TP is latency-sensitive and interconnect-hungry; keep it inside one NVLink-connected node.**

**Pipeline parallelism (PP)** splits *whole layers* across GPUs: GPU 0 holds layers 1–20, GPU 1 holds 21–40, and so on. A request flows through GPU 0, then its output is handed to GPU 1, like stations on an assembly line. The communication is tiny — you pass *activations* between stages once per boundary, not an all-reduce per op — so PP tolerates *slower* links (even across machines). The catch is the **pipeline bubble**: while GPU 0 works on the first token, GPUs 1–3 sit idle until work reaches them, so a single request doesn't go faster — you hide the bubble by keeping the pipeline *full* with many in-flight requests. **PP is throughput-friendly and interconnect-tolerant, but only pays off under concurrency.**

vLLM exposes both. The common knob is tensor parallelism:

```bash
# Shard Qwen2.5-72B across 4 GPUs in one NVLink-connected node (tensor-parallel):
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
# For multi-node, add --pipeline-parallel-size N to split layers across boxes.
```

The reasoning for which to use:

- **Model fits on one GPU** → neither; don't pay parallelism's overhead for nothing.
- **Doesn't fit, GPUs share NVLink in one box, latency matters** → `--tensor-parallel-size` (2, 4, 8 — usually a divisor of the attention-head count).
- **Spanning multiple nodes, or links between GPUs are slow** → add `--pipeline-parallel-size` so the cross-node hop carries cheap activations, not constant all-reduces. Production multi-node serving often combines them: TP *within* each node, PP *across* nodes.

> **The split rule:** tensor-parallel splits each layer (latency-sensitive, demands NVLink, stays in one node); pipeline-parallel splits the layer stack (throughput-friendly, tolerates slow links, needs concurrency to fill the bubble). First try to make it fit on one GPU with quantization — parallelism is what you reach for when even a 4-bit quant won't fit.

---

## Part 10 — A worked example: reading a real benchmark table

The checklist in Part 5 is abstract until you read numbers with it. Here's a small bakeoff: the **same** Qwen2.5-7B-Instruct, the **same** 512-token-prompt / 128-token-output workload, run on two engines across three concurrency points, on one A100. (Aggregate throughput is total tokens/sec across all concurrent requests; TTFT and decode are per-request.)

```text
                 concurrency=1        concurrency=8         concurrency=32
engine   quant   agg t/s  ttft p95   agg t/s  ttft p95    agg t/s  ttft p95   VRAM@32
------   -----   -------  --------   -------  ---------   -------  --------   -------
llamacpp Q4_K_M     58     0.09s       96      0.51s        102     2.40s      6.1 GB
vLLM     AWQ        49     0.14s      210      0.22s        540     0.39s      9.4 GB
```

Walk it the way the checklist asks. First, **is the comparison fair?** Same model, same prompt/output lengths, same concurrency points, VRAM reported at the serving concurrency, p95 (not mean) — good. The one confound the checklist flags out loud: the **quant differs** (GGUF Q4_K_M vs AWQ). Both are 4-bit, but they are *different* 4-bit schemes, so a small quality or speed gap could be the quant, not the engine — you note it rather than hide it.

Now read the curve, which is the whole point:

- **At concurrency 1, llama.cpp wins** (58 vs 49 t/s, and a faster TTFT). If you stopped here you'd conclude "llama.cpp is faster" — and you'd ship the wrong engine for a server. This is the single-stream trap from Lecture 1 §5, in numbers.
- **From 1 → 32, the engines diverge.** llama.cpp's aggregate barely moves (58 → 102) — it doesn't continuous-batch, so extra requests mostly queue. vLLM's *climbs* (49 → 540, over 5×) — continuous batching keeps filling the in-flight batch (Part 2). The *shape* of the two curves, not any single cell, is the result.
- **Watch the p95 tail, not just throughput.** llama.cpp's TTFT p95 explodes to **2.40s** at concurrency 32 — that's the queueing tail: the unlucky request waited behind a full serial pipeline. vLLM holds **0.39s**. A throughput-only table would have hidden this; p95 is what exposes the user who's timing out.
- **VRAM at concurrency 32**, not at 1: vLLM's 9.4 GB includes its KV-cache pool under load (Part 4) — the honest footprint you must budget for, well above the ~4 GB of weights alone.

The **right conclusion**: *for single-user / edge / CPU, llama.cpp is excellent and even wins at concurrency 1; for a concurrent server, vLLM is decisively better — ~5× the aggregate throughput at concurrency 32 with a far tighter tail latency — and the gap widens as concurrency rises.* The **wrong conclusions** to avoid: "llama.cpp is faster" (true only at c=1, the wrong axis for serving); "vLLM is 11× faster" (it's ~5× here — quote the number at *your* target concurrency, not the most flattering one); and "vLLM uses more VRAM so it's worse" (the extra VRAM is the KV-cache pool *buying* the 5× throughput — that's the trade, not waste). Every one of those wrong conclusions comes from reading a single cell instead of the curve. Read the curve.

---

## Part 11 — Recap

You should now be able to:

- **Choose a quantization format** (FP16 baseline, GGUF/Q4_K_M for llama.cpp, AWQ/GPTQ for vLLM, bitsandbytes for easy on-the-fly) and state its quality/VRAM/decode-speed trade — and place Q4_K_M at the knee of the quality/size curve, defended with perplexity, not vibes.
- **Explain why 4-bit costs so little quality**: per-block (group) scales place the 16 levels where each small block's weights actually live, so outliers can't starve the near-zero majority (the K-quants), and activation-aware methods (AWQ) protect the salient channels the forward pass depends on — one global scale fails, thousands of local scales succeed.
- **Reason about calibration**: AWQ/GPTQ need a calibration set and can suffer a domain mismatch (you inherit someone else's calibration when you download a pre-quant), while GGUF K-quants are calibration-free — reach for K-quants on an unusual or private domain, AWQ when serving with matching calibration data.
- Explain **continuous batching** (replace finished requests token-by-token so the GPU never idles) and **paged attention** (OS-style paging of the KV cache so it doesn't fragment VRAM) as the pair that lets vLLM serve many users on one GPU.
- Describe **speculative decoding** (a draft model proposes K tokens, the big model verifies in one pass — more tokens per weight-read, no quality loss, lift set by acceptance rate) and **prefix caching** (reuse the KV cache of a shared prefix — the SGLang/structured-workload win).
- Account for the **KV cache's VRAM cost**, which scales with context × concurrency and can dwarf the quantized weights — and measure VRAM at the real serving point.
- **Shard a model too big for one GPU**: tensor-parallel splits each layer (latency-sensitive, needs NVLink, stays in one node, `--tensor-parallel-size`) vs pipeline-parallel splits the layer stack (throughput-friendly, tolerates slow links, needs concurrency) — but try a quant first.
- **Run an honest benchmark and read it**: prefill and decode separated, throughput under rising concurrency (the curve, not one point), p50 *and* p95, VRAM at serving concurrency, warm-ups discarded, and a fair-comparison checklist that flags confounds like a quant mismatch across engines — then draw the right conclusion from the *curve*, not a single flattering cell.

Next: the exercises put this on real hardware — bring the same 7B up on three engines, chart the quantization trade-offs, and build the benchmark harness that produces the bakeoff's numbers. Continue to [the exercises](../exercises/README.md).

---

## References

- *Efficient Memory Management for LLM Serving with PagedAttention* — Kwon et al., 2023: <https://arxiv.org/abs/2309.06180>
- *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2022: <https://arxiv.org/abs/2211.17192>
- *AWQ: Activation-aware Weight Quantization* — Lin et al., 2023: <https://arxiv.org/abs/2306.00978>
- *GPTQ: Accurate Post-Training Quantization* — Frantar et al., 2022: <https://arxiv.org/abs/2210.17323>
- *k-quants* (GGUF block-wise K-quantization design) — llama.cpp PR #1684: <https://github.com/ggml-org/llama.cpp/pull/1684>
- *vLLM — distributed inference and serving* (`--tensor-parallel-size`, `--pipeline-parallel-size`): <https://docs.vllm.ai/en/latest/serving/distributed_serving.html>
- *Megatron-LM* (tensor & pipeline parallelism for transformers) — Shoeybi et al., 2019: <https://arxiv.org/abs/1909.08053>
- *Continuous batching, explained* (Anyscale): <https://www.anyscale.com/blog/continuous-batching-llm-inference>
- *vLLM — speculative decoding docs*: <https://docs.vllm.ai/en/latest/features/spec_decode.html>
- *llama.cpp quantization (GGUF) docs*: <https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md>
- *vLLM benchmarks (benchmark_serving.py)*: <https://github.com/vllm-project/vllm/tree/main/benchmarks>
