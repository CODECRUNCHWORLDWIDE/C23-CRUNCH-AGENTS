# Lecture 1 — PagedAttention and Continuous Batching: Making the GPU Busy

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can explain the inference two-step (prefill vs decode), state why the KV cache — not the model weights — is the binding VRAM constraint at serving time, describe PagedAttention as virtual memory for the KV cache, describe continuous batching as a per-step scheduler, and predict what each of vLLM's serving knobs does to throughput, latency, and VRAM. You can read your own vLLM startup log and say how many concurrent sequences your config can hold.

If you remember one sentence from this entire week, remember this one:

> **Continuous batching is the throughput multiplier that makes self-hosting feasible. Without it, you are paying for idle GPU.**

There's a corollary you should tape next to it:

> **The KV cache, not the weights, is what runs out first.** Loading a 14B model in FP16 costs ~28 GB of VRAM and never changes. The KV cache grows with every token of every concurrent request, and it is what decides how many users you can serve at once. Manage the cache and you manage the server.

And a third, because the economics ride on it:

> **A self-hosted GPU's cost-per-token is a function of how busy you keep it.** The same H100 at the same hourly rate produces a $0.24/million-token cost when its batch is full and a $3.50/million-token cost when it serves one request at a time. The hardware didn't change; your utilization did. Everything in this lecture is in service of raising that utilization, because utilization *is* the cost.

For eighteen weeks you treated "call the model" as an atomic operation — a function from a prompt to tokens. This week you pry it open at the systems level, because the gap between a naive server and vLLM is the gap between a GPU that's 30% busy and one that's 94% busy, and that gap *is* the economics of self-hosting. Everything that follows is in service of one measurable question: **how do you keep the GPU full so your cost-per-token is low?**

---

## 1. The inference two-step: prefill and decode

A generation request runs in two distinct phases, and they have opposite performance characteristics. Understanding this split is the whole foundation.

**Phase 1 — prefill.** You hand the model the entire prompt. It runs one forward pass over *all* the prompt tokens at once, in parallel, and produces (a) the first output token and (b) the **KV cache** for the prompt — the keys and values for every attention layer for every prompt token. Prefill is **compute-bound**: it's a big matrix multiply over the whole prompt, and the GPU's compute units are the bottleneck. A 2000-token prompt prefills in one shot.

**Phase 2 — decode.** Now the model generates output tokens one at a time. Each step takes the last token, runs *one* forward pass, attends to the entire KV cache built so far, and emits one new token (whose K and V get appended to the cache). Decode is **memory-bandwidth-bound**: each step does very little compute (one token) but must read the *entire* KV cache and the model weights from VRAM. The GPU's compute units sit mostly idle; the bottleneck is how fast you can move bytes.

This asymmetry is the key to everything:

> **Prefill saturates compute. Decode wastes compute and is gated on memory bandwidth.** A single decoding request leaves the GPU's arithmetic units almost idle — you're paying for thousands of cores to compute one token. That idle compute is exactly what batching reclaims: run *many* sequences' decode steps together, and the model-weight read (the expensive part) is shared across all of them.

Here's the intuition that makes it click. Reading the 14B model's weights from VRAM costs the same whether you're decoding 1 sequence or 100 — it's one read of ~28 GB. If you decode one sequence, that read produces one token. If you batch 100 sequences and decode them together, the *same* weight read produces 100 tokens. The weight read amortizes across the batch. That's the entire throughput argument for batching, and it's why decode — the memory-bound phase — is where batching pays off most.

One more way to see the memory-bound claim, in numbers. An H100 has roughly 3.35 TB/s of memory bandwidth and ~1000 TFLOP/s of FP16 compute. Decoding one token of a 14B model reads ~28 GB of weights and does ~28 GFLOP of math (one token through the network). The time to *read* 28 GB at 3.35 TB/s is ~8.4 ms; the time to *compute* 28 GFLOP at 1000 TFLOP/s is ~0.028 ms — three hundred times faster. So a single-sequence decode step spends ~8.4 ms moving bytes and ~0.03 ms computing: the GPU's arithmetic units are idle 99.7% of the step, waiting on memory. Batch 100 sequences and you still read the 28 GB once (~8.4 ms) but now do 100× the compute (~2.8 ms) — the compute is still hidden under the memory read, and you got 100 tokens for nearly the price of one. *That* is why decode is memory-bound and why batching is close to free until the compute finally catches up with the memory read at large batch sizes (the point where throughput plateaus). The whole game is filling the dead time during the weight read with more sequences' worth of useful work.

---

## 2. Why the KV cache is the real bottleneck

People new to serving assume the model weights are the constraint. They are not — weights are a fixed cost you pay once at load. The constraint is the **KV cache**, and it's worth doing the arithmetic so the number lands.

For each token, each layer stores a key vector and a value vector. The per-token KV size is:

```
kv_bytes_per_token = 2 (K and V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

For a 14B-class model in FP16 that's roughly **~100–200 KB per token** depending on architecture (and grouped-query attention shrinks it by sharing KV heads). Multiply by sequence length and by the number of concurrent sequences:

```
total_kv = kv_bytes_per_token × sum(seq_len for each active sequence)
```

A single 4000-token sequence might use a few hundred MB of KV. Forty concurrent sequences each at a few thousand tokens, and you're into tens of GB — on a GPU where the weights already took 28 GB. **The KV cache is what fills the remaining VRAM, and when it fills, you can't admit another request.** The number of concurrent sequences you can serve is, to a first approximation:

```
max_concurrent ≈ (VRAM − weights − overhead) / (kv_bytes_per_token × avg_seq_len)
```

That is the equation the whole week turns on. Make the KV cache cheaper (paging, quantization, shorter context) and `max_concurrent` goes up. A bigger batch means more sequences sharing each weight read, which means higher throughput, which means lower cost-per-token. The chain — **cache efficiency → batch size → throughput → cost** — is the spine of vLLM's design.

> **The serving mantra:** weights are a fixed tax; the KV cache is the variable cost that decides your concurrency. Every vLLM knob that matters is really a knob on the KV cache.

Let's make the arithmetic concrete with a worked example, because the number is more persuasive than the formula. Take an H100 with 80 GB of VRAM serving Qwen2.5-14B in FP16:

```
VRAM total                = 80 GB
weights (14B × 2 bytes)   = ~28 GB
activation + overhead     = ~6 GB
VRAM left for KV cache    = ~46 GB
```

Now suppose your workload averages 2000 tokens per sequence (prompt + output) and the model's per-token KV cost is ~140 KB (FP16, with grouped-query attention shrinking the KV heads). Each sequence costs `2000 × 140 KB ≈ 280 MB` of KV. So:

```
max_concurrent ≈ 46 GB / 280 MB ≈ 164 sequences
```

That's the headline number: this GPU, this model, this workload → ~164 concurrent sequences before the KV cache is full. Change any input and the number moves. Halve the average sequence length (1000 tokens) and you double concurrency to ~328. Switch to a model without grouped-query attention (each token costs 4× the KV) and concurrency drops to ~40. **Concurrency is not a fixed property of the GPU — it's a function of the model architecture and your workload's sequence-length distribution, computed through the KV cache.** This is why two teams running "the same H100" report wildly different throughput: they have different models and different traffic, so they have different KV budgets.

The practical upshot for the sweep: when you read the "# GPU blocks" line in the vLLM startup log (Exercise 1), you can hand-compute exactly this number — blocks × tokens-per-block ÷ your average sequence length — and predict where your server will saturate *before* you drive a single request at it. An engineer who can do that prediction, and then watch the sweep confirm it, understands the server. One who's surprised by where it saturates does not.

---

## 3. The naive server and where it bleeds

Before PagedAttention, the standard approach allocated the KV cache as one **contiguous** block per sequence, sized for the *maximum* possible length. This bleeds VRAM in three ways, and naming them is how you appreciate the fix.

**Internal fragmentation — over-reservation.** You don't know how long a response will be, so you reserve `max_model_len` (say 4096 tokens) of KV space per sequence up front. A request that generates 200 tokens uses 200 tokens of that reservation and wastes the other 3896. Across a batch, you're reserving for the worst case and using a fraction of it.

**External fragmentation.** Sequences finish at different times, freeing contiguous blocks of different sizes scattered through VRAM. A new sequence needs a contiguous block big enough; even if total free VRAM is plenty, no single hole is large enough. You're VRAM-rich and allocation-poor.

**No sharing.** Two requests with the same system prompt each store their own copy of that prompt's KV cache. Identical bytes, stored twice (or 32 times, for 32 concurrent requests with the same preamble).

The measured result, reported in the vLLM paper: contiguous KV allocation wastes **60–80%** of the KV memory to fragmentation and over-reservation. You bought 80 GB of H100 and you're effectively serving with 20. Your batch size is small, your GPU is 30% busy, and your cost-per-token is three times what it should be. This is the problem PagedAttention solves.

---

## 4. PagedAttention: the KV cache as virtual memory

The insight is borrowed directly from operating systems. An OS doesn't give a process one giant contiguous block of physical RAM — it gives the process a *virtual* address space backed by fixed-size **pages** that can live anywhere in physical RAM, mapped through a page table. PagedAttention does exactly this for the KV cache.

The mechanism:

1. **Divide VRAM into fixed-size KV blocks.** Each block holds the KV for a fixed number of tokens (e.g. 16). These blocks are the "physical pages."
2. **Give each sequence a block table.** Instead of one contiguous allocation, a sequence's KV cache is a *list of block indices* — logically contiguous, physically scattered. As the sequence grows token by token, you allocate one more block only when the current one fills. This is the "virtual address space."
3. **Allocate on demand.** A sequence that generates 200 tokens uses ~13 blocks (200/16) and not one block more. The over-reservation waste is gone — you allocate as you go, in small increments.
4. **No external fragmentation.** All blocks are the same fixed size, so any free block fits any sequence. The scattered-holes problem disappears because every hole is exactly one block.
5. **Copy-on-write sharing.** Two sequences with the same prompt prefix can *share* the same physical blocks for that prefix — both block tables point at the same blocks. When one sequence diverges (generates a different token), it copies just that block and writes there. This is the foundation of prefix caching (§7) and of efficient parallel sampling.

```text
Sequence A block table:  [12, 47, 3,  88]   →  physical blocks scattered in VRAM
Sequence B block table:  [12, 47, 91]       →  shares blocks 12, 47 with A (same prefix!)

VRAM blocks:  [...][blk3][...][blk12: shared][...][blk47: shared][...][blk88][...][blk91]...
```

The payoff: KV-memory waste drops from 60–80% to a few percent (just the partial last block per sequence). That recovered VRAM goes straight into a bigger KV cache, which means a bigger batch, which means higher throughput and lower cost. PagedAttention is not a micro-optimization — it's what makes the whole continuous-batching scheme have enough memory to work with.

> **The mental model:** the KV cache is a heap, and PagedAttention is `malloc` with fixed-size pages and a page table per sequence. Once you see it as virtual memory, the block-table line in the vLLM startup log stops being mysterious — it's telling you how many pages of KV heap you have.

---

## 5. Continuous batching: scheduling every step, not every request

PagedAttention gives you the memory to hold a big batch. Continuous batching is the *scheduler* that keeps that batch full. Contrast it with the naive approach.

**Static (request-level) batching.** You collect N requests, run them as a batch until *all* N finish, then start the next batch. The problem: requests have wildly different output lengths. One request generates 20 tokens, another generates 800. The whole batch runs for 800 steps, and for steps 21–800 the first request's slot is *idle* — it finished, but the batch can't release it until everyone's done. You padded to the longest sequence and burned the difference. With real traffic (high variance in output length), static batching leaves the GPU half-idle.

**Continuous (in-flight) batching.** vLLM's scheduler runs at the granularity of a single decode *step*, not a whole request. Every step:

1. Run one decode step for every sequence currently in the batch.
2. Any sequence that just emitted its stop token (EOS, stop sequence, max tokens) is **evicted** — its KV blocks are freed immediately.
3. Any waiting request is **admitted** into the now-free slot — its prefill runs and it joins the decode batch.

So the batch is a living thing: sequences flow in and out continuously, the freed slot from a finished short request is instantly reused by a waiting one, and the GPU never sits idle waiting for the slowest sequence. On real traffic with mixed lengths, this is the **20–30× throughput** improvement over static batching that the Anyscale post measures.

```text
Static batching (8 slots, wait for all):
  step:  1   2   3 ... 20  21 ... 800
  seq0:  ●   ●   ●     ●   (done, slot IDLE) .......... 779 wasted steps
  seq1:  ●   ●   ●     ●   ●  ●  ●  ... ●  (runs to 800)
  ...batch can't refill until step 800.

Continuous batching (8 slots, refill every step):
  step:  1   2   3 ... 20  21        22 ...
  seq0:  ●   ●   ●     ●   (done→evicted)
  slot:                   seq8 admitted →  ●   ●  ...   (waiting request fills the slot)
  ...GPU stays full.
```

There's a scheduling subtlety worth naming: **prefill and decode compete for the GPU.** Admitting a new request means running its prefill, which is a compute-heavy burst that can stall the ongoing decode steps of everyone else (a latency spike for active users). vLLM has policies for this (chunked prefill, which splits a long prefill into pieces interleaved with decode steps, so one giant prompt doesn't freeze the batch). You don't have to tune this on day one, but you should know it's why a burst of long-prompt requests can briefly spike p95 latency even when throughput is fine.

> **The discipline:** throughput is a fleet-level number (total tokens/sec across the batch); latency is a per-request number (how long *one* user waits). Continuous batching maximizes the first. The two trade off — a bigger batch raises throughput and raises per-request latency — and the concurrency sweep (the challenge) is how you find where that trade-off lands for your workload.

To make the eviction-and-admission rhythm tangible, walk one concrete scenario. Eight requests arrive at once; the batch has 4 slots. The scheduler admits the first 4 (A, B, C, D) and queues E, F, G, H. Now the decode loop runs:

```
step  | slot0 slot1 slot2 slot3 | queue           | event
------+-------------------------+-----------------+--------------------------------
 1    |  A     B     C     D    | E F G H         | all four decode one token
 ...  |  A     B     C     D    | E F G H         | ...
 12   |  A*    B     C     D    | E F G H         | A emits EOS -> EVICTED, KV freed
 13   |  E     B     C     D    | F G H           | E admitted: its prefill runs,
      |                         |                 |   then E joins the decode batch
 ...  |  E     B     C     D    | F G H           |
 40   |  E     B*    C     D    | F G H           | B done -> evicted
 41   |  E     F     C     D    | G H             | F admitted into B's freed slot
```

The slot that A vacated at step 12 does not sit idle waiting for B, C, and D to finish — E moves in at step 13 and the GPU keeps doing 4 sequences' worth of work every step. Under *static* batching, that slot would stay empty from step 12 until the whole batch (the longest of A/B/C/D) finished, then a fresh batch of 4 would start. The wasted steps — slot0 idle from step 13 onward in the static world — are exactly the throughput continuous batching reclaims. On a workload where output lengths range from 20 to 800 tokens, those wasted steps are the majority of the GPU's time under static batching, which is where the 20–30× figure comes from.

Notice the cost hidden at step 13: admitting E means running its **prefill** (a compute burst over E's whole prompt) in the middle of A/B/C/D's decode steps. For one short prompt this is invisible; for a 4000-token prompt it stalls the decode of everyone else for the duration of that prefill, and the active users see a latency hiccup. That is the prefill-vs-decode contention from §5's preview, and it's why **chunked prefill** exists — splitting E's prefill into smaller pieces interleaved with decode steps so no single admission freezes the batch. You don't tune this on day one, but when your p95 latency spikes on a burst of long-prompt traffic while throughput looks fine, this is the mechanism to suspect.

---

## 6. The serving knobs that matter

vLLM has dozens of flags; six of them decide your throughput, latency, and VRAM. Know what each does to the KV-cache equation from §2.

**`--gpu-memory-utilization` (default 0.9).** The fraction of total VRAM vLLM is allowed to claim. After loading weights, vLLM uses the rest (up to this fraction) for the KV cache. Higher → bigger KV cache → bigger batch → higher throughput. Set too high and you OOM on a spike or leave no room for activation memory. 0.9 is a sane default; 0.95 squeezes more batch if you're confident in your length distribution.

**`--max-model-len`.** The maximum context (prompt + output) length. This caps the per-sequence KV size and, with `--gpu-memory-utilization`, determines how many sequences fit. Setting it *lower* than the model's native context (e.g. 8192 instead of 32768) is a legitimate throughput lever: if your workload never exceeds 8K tokens, capping at 8K lets vLLM pack more sequences into the same VRAM. Don't reserve context you'll never use.

**`--max-num-seqs` (the batch-size ceiling).** The maximum number of sequences in a single batch. This is the hard cap on concurrency at the model level. If your KV cache could hold 200 sequences but `--max-num-seqs` is 64, you top out at 64. Raise it to fill the batch; but a huge value with insufficient KV memory just means requests queue (PENDING) rather than run. Tune it against your measured KV-block count.

**`--tensor-parallel-size` (TP).** Shards the model's weights across N GPUs. Use it when the model doesn't fit on one GPU (a 70B in FP16 needs ~140 GB → two 80 GB H100s at TP=2) or when you want to split the memory-bandwidth load. TP has communication overhead (the GPUs exchange activations every layer over NVLink), so TP=2 is not 2× the throughput of one GPU — it's how you run a model that otherwise wouldn't fit. Don't reach for it on a model that fits on one card.

**`--pipeline-parallel-size` (PP).** Splits the *layers* across GPUs/nodes (GPU 0 runs layers 0–N, GPU 1 runs N–2N). Used for multi-node serving of very large models. More complex than TP and usually a later concern; know it exists.

**`--enable-prefix-caching`.** Turns on KV reuse for shared prompt prefixes (see §7). Free throughput when your requests share a prefix; near-zero cost when they don't. Almost always worth turning on for chat/agent workloads with a fixed system prompt.

The interaction to internalize: **`--gpu-memory-utilization` and `--max-model-len` together set how big your KV cache is; `--max-num-seqs` caps how much of it you use as concurrent sequences.** If your GPU util is low under load, the usual cause is `--max-num-seqs` too small (the batch can't fill) or concurrency too low (not enough requests arriving to fill the batch). The fix is to raise the ceiling and drive more concurrency — which is exactly what the sweep does.

---

## 7. Prefix caching: free throughput when prompts share a head

§4's copy-on-write sharing has a direct application. In a chat or agent workload, every request usually begins with the *same* large prefix: a system prompt, a tool schema, a few-shot block. With `--enable-prefix-caching`, vLLM hashes prompt prefixes and, when a new request's prefix matches a cached one, **reuses the existing KV blocks** instead of recomputing the prefill for that prefix.

The win is twofold: you skip the prefill compute for the shared part (lower latency, less compute), and the shared blocks are stored once not N times (more KV memory free for the batch). On a workload where 2000 tokens of system prompt precede a 50-token user question, prefix caching skips ~97% of the prefill on every request after the first. This is the self-hosted analog of vendor prompt caching from your earlier weeks — same idea (reuse the KV of a stable prefix), implemented in your own server.

The catch — and it's the same catch as vendor prompt caching: **the prefix must be byte-identical and stable.** Interpolate a timestamp or a per-request ID into the front of the prompt and the prefix changes every request, the hash never matches, and prefix caching does nothing. Put the volatile content *after* the stable prefix. The stretch goal in the README is to measure the lift with a genuinely shared prefix and watch it evaporate when you inject a per-request token into the head — which is the lesson made visible.

> **Rule of thumb:** turn on `--enable-prefix-caching` for any workload with a fixed system prompt or tool schema. Then keep the front of your prompt frozen — exactly the discipline from prompt caching in earlier weeks, now enforced by your own server's hash.

---

## 8. Speculative decoding: a latency lever with a throughput tax

Decode is memory-bound (§1) — each step reads the whole model to produce one token. Speculative decoding attacks this: a small, fast **draft model** proposes several tokens ahead (cheap, because it's small), then the big **target model** verifies all of them in a *single* forward pass (one expensive weight read produces multiple accepted tokens). If the draft was right, you got K tokens for the price of one big-model step. If wrong, you fall back and lose a little.

The lever is the **accept rate** — how often the draft model's proposals match what the target would have produced. High accept rate (the draft is good at this domain) → big latency win. Low accept rate → you wasted the draft compute and the extra memory for little gain. Variants:

- **Draft-model spec decode:** a small model (e.g. a 1B) drafts for a big one (the 14B). Needs VRAM for the second model.
- **N-gram / prompt-lookup:** no draft model — propose tokens by matching n-grams from the prompt itself (great for tasks that echo the input, like code editing or summarization).
- **EAGLE / Medusa:** trained lightweight heads on the target model that predict several tokens; higher accept rates than a generic draft.

The honest catch: **speculative decoding helps single-stream latency but can *hurt* throughput at high batch sizes.** When the batch is already full, the GPU is compute-saturated by the batch itself — there's no idle compute for the draft model to exploit, and the verification overhead becomes pure cost. Spec decode shines at *low* concurrency (a single user wants their answer fast) and fades or reverses at *high* concurrency (the batch already keeps the GPU busy). This is the canonical "measure, don't assume" case: the README stretch goal is to turn it on, measure the accept rate and the latency change at concurrency 1 *and* at concurrency 64, and see the curves cross.

---

## 9. The throughput-vs-concurrency curve (and what it tells you)

Everything above produces one artifact: the curve of throughput (and latency) as you raise concurrency. It has a predictable shape, and reading it honestly is the week's core skill.

- **Low concurrency (1–8):** throughput rises roughly linearly with concurrency. Each new request fills an empty batch slot; the GPU was idle and now isn't. Latency is low and barely moves — the batch isn't contended.
- **Mid concurrency (the sweet spot):** throughput keeps climbing but starts to bend. The batch is filling up; per-request latency creeps up because each request waits its turn through a fuller batch.
- **Saturation (high concurrency):** throughput **plateaus** — the GPU is compute- or memory-bound and can't do more total work no matter how many requests you add. Beyond this point, adding concurrency only adds *queueing*: requests wait in the PENDING queue, p95 latency climbs steeply (the "cliff"), and throughput is flat.

```text
throughput                            p95 latency
   │            ___________                │                    /
   │          /              (plateau)     │                   /
   │        /                              │              ____/  (cliff)
   │      /                                │   _________/
   │    /                                  │__/
   └────────────────── concurrency        └────────────────── concurrency
        1   8   32   128                        1   8   32   128
```

The operating point you *want* is just before the cliff: high throughput (cheap cost-per-token) without p95 latency violating your SLO. **Throughput tells you the cost; latency tells you the user experience; the sweep finds the concurrency where both are acceptable.** A server tuned to the plateau is cheap but slow for users; a server run at concurrency 1 is fast but ruinously expensive per token. The whole point of measuring the curve is to choose your spot on it deliberately.

This is also where cost-per-million-tokens comes from. At your chosen operating point you have a measured throughput (tok/s). The GPU costs a fixed $/hour. So:

```
cost_per_1M_tokens = (gpu_cost_per_hour / 3600) / throughput_tok_per_s × 1_000_000
```

A $2.50/h H100 at 2840 tok/s costs `(2.50/3600)/2840 × 1e6 ≈ $0.24` per million output tokens. Compare that to a vendor at $15/million and self-hosting looks like a steal — *at that throughput*. Drop to concurrency 1 (single user, 200 tok/s) and the same GPU costs `(2.50/3600)/200 × 1e6 ≈ $3.47` per million, and the picture is murkier. **The cost depends entirely on how busy you keep the GPU**, which is why batching is the product and the sweep is the proof.

---

## 10. The break-even, and the rest of the cost story

Lecture 2 does the full break-even math, but here's the shape so the economics land now. A vendor API has **zero fixed cost** and a per-token price. A self-hosted GPU has a **fixed hourly cost** and a near-zero marginal per-token cost (electricity). Plot total cost vs monthly token volume:

```text
cost                                          vendor (slope = price/token, intercept 0)
  │                                       /
  │                                    /
  │                            ____ /  ← break-even volume
  │ ___________________ /___/             self-hosted (high fixed, ~flat slope)
  │/         (fixed GPU $/month)
  └──────────────────────────────── monthly token volume
```

Below the break-even volume, the vendor is cheaper (you don't pay for an idle GPU). Above it, self-hosting wins (the GPU is busy enough that its fixed cost spreads thin). The break-even volume is where the lines cross — and it depends on your throughput (how cheap your GPU's tokens are) and the vendor's price. **You compute this number; you don't guess it.** The challenge's memo is exactly this calculation for your measured throughput.

And there are costs the simple math omits, which Lecture 2 makes you name: **operational overhead** (someone is on-call for the GPU at 2 AM — the vendor handles that for you), **utilization risk** (you pay for the GPU 24/7 but traffic is bursty, so your *effective* throughput is lower than your peak), and the **carbon** footprint (a busy GPU is more carbon-efficient per token than an idle one, which is another argument for keeping it full). Self-hosting is not free money above the break-even — it's a trade of vendor margin for operational burden, and the memo should say so.

---

## 10b. Quantization, the KV cache, and the levers that compound

Week 6 taught quantization as a way to fit a model on a smaller GPU. At serving time it does something subtler that's worth naming, because it interacts with everything above. Quantizing the *weights* (AWQ, GPTQ, FP8) shrinks the 28 GB weight footprint — freeing VRAM that goes straight into a bigger KV cache, which raises concurrency, which raises throughput. So weight quantization is not just "fits on a smaller card"; on the *same* card it's a throughput lever, because the recovered VRAM becomes batch capacity.

There's a second, distinct knob: **KV-cache quantization** (`--kv-cache-dtype fp8`). This halves the per-token KV cost, which directly doubles `max_concurrent` from §2's equation — the most direct throughput lever there is, because the KV cache *is* the binding constraint. The catch is the usual one: lower-precision KV can cost a little accuracy, so you measure it (does your eval score hold?) rather than assuming it's free. The pattern repeats across the whole week — every knob is a trade you measure, not a default you trust.

The levers compound, and seeing how is the systems-thinking payoff:

```
FP8 weights      → smaller weight footprint → bigger KV cache → higher concurrency
FP8 KV cache     → half the KV per token    → 2× concurrency at the same VRAM
prefix caching   → shared prefix stored once → more KV free for distinct sequences
lower max-model-len (to the real P99 length) → less KV reserved per seq → more seqs
```

Stack them and you can take a server from ~40 concurrent sequences to a few hundred on the same hardware — which is the difference between a cost-per-token that loses to the vendor and one that beats it tenfold. None of these is magic; each is a measured adjustment to the KV-cache budget. The engineer who sees them as one coherent system — "everything is a knob on the cache, and the cache is concurrency, and concurrency is cost" — is the one who can tune a deployment instead of cargo-culting flags from a blog post.

## 10c. The failure modes you'll actually hit

Before the recap, name the four ways this goes wrong in practice, because you'll see all of them:

- **OOM at startup.** You asked for a bigger KV cache (`--gpu-memory-utilization` too high, or `--max-model-len` too long) than the card has after loading weights. The server dies at launch. Fix: lower one of the two until the startup log's "# GPU blocks" is positive and comfortable.
- **Throughput won't climb past a low ceiling.** You raise concurrency and tokens/sec flatlines early. Usual cause: `--max-num-seqs` is throttling the batch below what the KV cache could hold, or you're not actually driving concurrent requests (the benchmark bug from Lecture 2 §4). Raise the ceiling, verify true concurrency.
- **p95 latency spikes on bursts.** A burst of long-prompt requests arrives; their prefills contend with everyone's decode (§5). Throughput looks fine; tail latency is ugly. Mitigation: chunked prefill, and sizing `--max-num-seqs` so admission bursts don't overwhelm the decode loop.
- **Prefix caching does nothing.** You turned it on and saw no lift. Almost always: the prefix isn't byte-stable (a timestamp or per-request ID at the front), so the hash never matches. Freeze the prefix; put volatile content last.

Each of these is diagnosable from the startup log and the sweep — which is why reading both is a core skill, not a formality.

## 10d. A worked batching walkthrough — the 20–30× in numbers

The "20–30×" figure is easy to assert and worth deriving once so it stops being a slogan. Take eight requests with decode lengths `20, 800, 30, 30, 40, 25, 600, 35` tokens, a per-slot decode rate of ~50 tok/s, and a 4-slot batch.

**Static batching** runs each batch until its *longest* member finishes. Batch 1 = `[20, 800, 30, 30]` runs until the 800-token request is done — 800/50 = **16 s** — with three slots idle for ~15 s each after their short requests finished. Batch 2 = `[40, 25, 600, 35]` runs until the 600-token request finishes — **12 s**. Total: ~28 s for 1,580 tokens → **~56 tok/s**, barely above one slot's rate, because three of four slots were idle most of the time.

**Continuous batching** refills a slot the instant a sequence finishes, so the short requests churn through and get replaced while the two long ones hold their slots. The wall-clock is bounded by the long pole (800/50 = **16 s**), and 1,580 tokens in 16 s → **~99 tok/s** with all four slots busy. On this tiny queue that's ~1.8×; on a *full* queue of hundreds of mixed-length requests, continuous batching pins all four slots at ~50 tok/s (~200 tok/s) while static stays stuck near one slot — and *that* gap is the 20–30× the Anyscale post measures. The lesson in one line: **static batching's throughput collapses toward a single slot whenever lengths vary; continuous batching scales with slot count.**

## 11. Recap

You should now be able to:

- Explain the **inference two-step** — compute-bound prefill, memory-bandwidth-bound decode — and why decode is where batching reclaims wasted compute.
- State why the **KV cache, not the weights, is the binding VRAM constraint**, and write the `max_concurrent` equation that connects cache size to the number of users you can serve.
- Name the three ways naive contiguous KV allocation **bleeds 60–80% of VRAM**, and explain **PagedAttention** as virtual memory for the cache (fixed-size blocks, per-sequence block tables, copy-on-write sharing).
- Describe **continuous batching** as a per-step scheduler that evicts finished sequences and admits waiting ones every decode step, and why it beats static batching 20–30× on mixed-length traffic.
- Predict what **`--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`, `--tensor-parallel-size`, `--enable-prefix-caching`** each do to throughput, latency, and VRAM.
- Explain **prefix caching** (free throughput on shared prefixes; dies if the prefix isn't byte-stable) and **speculative decoding** (a low-concurrency latency lever with a high-concurrency throughput tax — measure both).
- Read a **throughput-vs-concurrency curve** — linear region, sweet spot, plateau, latency cliff — and turn the operating-point throughput into a **cost-per-million-tokens** number.

Next: how to actually stand the server up, put a LiteLLM router in front, and turn the throughput number into the self-hosted-vs-vendor break-even memo. Continue to [Lecture 2 — LiteLLM Routing and Self-Hosted Economics](./02-litellm-routing-and-self-hosted-economics.md).

---

## References

- *Efficient Memory Management for Large Language Model Serving with PagedAttention* — Kwon et al., 2023: <https://arxiv.org/abs/2309.06180>
- *vLLM documentation (architecture, scheduling, serving args)*: <https://docs.vllm.ai/en/latest/>
- *How continuous batching enables 23x throughput in LLM inference* — Anyscale: <https://www.anyscale.com/blog/continuous-batching-llm-inference>
- *vLLM — Automatic Prefix Caching*: <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>
- *vLLM — Speculative Decoding*: <https://docs.vllm.ai/en/latest/features/spec_decode.html>
- *vLLM — Distributed Serving (tensor / pipeline parallel)*: <https://docs.vllm.ai/en/latest/serving/distributed_serving.html>
