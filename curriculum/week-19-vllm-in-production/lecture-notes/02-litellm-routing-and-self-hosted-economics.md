# Lecture 2 — LiteLLM Routing and Self-Hosted Economics

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can stand up a vLLM OpenAI-compatible server and call it with the unmodified `openai` client; put a LiteLLM proxy in front of one or more backends with model aliases and a vendor fallback; run a disciplined concurrency benchmark and read the result; and compute the self-hosted-vs-vendor break-even volume from a measured throughput number, with the operational and carbon caveats stated honestly.

Lecture 1 explained *why* vLLM is fast. This lecture is *how you run it, route it, and pay for it.* The throughline:

> **vLLM speaks the OpenAI API, so self-hosting is a `base_url` change, not a rewrite — and that is the whole point of LiteLLM: one client, many backends, with failover.** Your application code does not learn that the model moved in-house. It calls the same `/v1/chat/completions`; the router decides whether that lands on your H100 or a vendor.

---

## 1. The OpenAI-compatible server: `vllm serve`

The single most important operational fact about vLLM: it ships an **OpenAI-compatible HTTP server**. One command:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --port 8000
```

This downloads the weights (first run), loads them onto the GPU, sizes the KV cache against your `--gpu-memory-utilization` and `--max-model-len`, and exposes `/v1/chat/completions`, `/v1/completions`, and `/v1/models` on port 8000 — the exact endpoints the vendor APIs expose. The startup log is worth reading line by line the first time; the line that matters most is the KV-cache report:

```
INFO ... # GPU blocks: 7680, # CPU blocks: 512
INFO ... Maximum concurrency for 8192 tokens per request: 15.00x
```

That "# GPU blocks" is your KV-cache capacity in PagedAttention pages (Lecture 1 §4). Multiply by the tokens-per-block (16) to get total cacheable tokens, and you can hand-check the `max_concurrent` equation from Lecture 1 §2. The "Maximum concurrency" line is vLLM telling you, given your `--max-model-len`, how many full-length sequences it can hold at once. If that number is smaller than the concurrency you intend to drive, requests will queue — raise `--gpu-memory-utilization` or lower `--max-model-len`.

Now call it. Here's the punchline — **your client code is unchanged from the vendor weeks**, you just point `base_url` at localhost and pass any non-empty `api_key` (vLLM ignores it unless you set `--api-key`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-14B-Instruct",   # the model name vLLM was launched with
    messages=[{"role": "user", "content": "Explain continuous batching in one sentence."}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
print(resp.usage)   # prompt_tokens, completion_tokens — your cost accounting hook
```

That `resp.usage` block is the same shape you've metered vendor calls with. It's how the benchmark counts tokens and how the cost math gets its numerator. The whole serving stack is "the API you already know, hosted on a GPU you control."

> **The migration is trivial *because* of the standard.** vLLM, SGLang, TGI, and most self-hosted servers speak the OpenAI API on purpose — it's the lingua franca of inference. That's what makes the routing layer in §3 possible: every backend, vendor or self-hosted, looks the same to the caller.

---

## 2. Docker, and the one-line container path

You can `pip install vllm` on a GPU box, but the reproducible path is the official container, which bundles the matching CUDA:

```bash
docker run --gpus all -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-14B-Instruct \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enable-prefix-caching
```

Two flags people forget: `--ipc=host` (vLLM uses shared memory between processes; without it you get cryptic crashes under load) and the HF cache volume mount (so you don't re-download 28 GB every container restart). Mount the cache, set `--ipc=host`, expose 8000, and the container behaves exactly like the bare-metal `vllm serve`. This is the form the capstone deploys.

---

## 3. LiteLLM: one client, many backends, with fallback

You now have one vLLM endpoint. Production wants more: multiple replicas for capacity and redundancy, a vendor model for the "hard" routes, model aliases so callers don't hard-code backend URLs, and — critically — **fallback** so that if your GPU dies, traffic spills to the vendor instead of erroring. That's LiteLLM's job. It's an OpenAI-compatible **proxy** that sits in front of N backends.

A `config.yaml`:

```yaml
model_list:
  # Two replicas of the self-hosted model, same alias -> LiteLLM load-balances across them.
  - model_name: local-14b
    litellm_params:
      model: openai/Qwen/Qwen2.5-14B-Instruct
      api_base: http://vllm-replica-a:8000/v1
      api_key: "none"
  - model_name: local-14b
    litellm_params:
      model: openai/Qwen/Qwen2.5-14B-Instruct
      api_base: http://vllm-replica-b:8000/v1
      api_key: "none"
  # A vendor model for hard routes and as the fallback target.
  - model_name: frontier
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  # If every local-14b replica fails, spill to the vendor. This is the capstone's failover.
  fallbacks: [{"local-14b": ["frontier"]}]
  num_retries: 2
```

Run it: `litellm --config config.yaml --port 4000`. Now your application talks to **LiteLLM**, not to vLLM directly:

```python
client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-litellm-key")
resp = client.chat.completions.create(model="local-14b", messages=[...])
```

`model="local-14b"` is an *alias*. LiteLLM picks one of the two vLLM replicas (round-robin / weighted / least-busy, configurable). If both replicas are down, the `fallbacks` rule sends the request to `frontier` (Claude) instead — and the caller never knows. That is the exact mechanism the syllabus capstone uses ("LiteLLM router fails over to the remaining replicas and to the vendor fallback") and the exact thing the week-24 chaos drill tests by killing a replica.

Three LiteLLM features the capstone leans on:

- **Model aliases** decouple the caller from the backend. `local-14b` can be re-pointed from vLLM to NeMo (week 20) without touching application code.
- **Fallback** is the resilience story — local first (cheap), vendor on failure (reliable). Combined with week 21's *routing* (easy→local, hard→vendor *by choice*, not by failure), this becomes the full cost-and-resilience layer.
- **Unified usage logging** — LiteLLM logs `usage` per request per model, which is the raw material for the per-route cost dashboards in weeks 21 and 24.

> **The architectural point:** LiteLLM turns "which GPU/vendor serves this" into a routing decision made *outside* the application. The app asks for `local-14b`; the platform decides where that lands and what happens when it can't. This separation is why self-hosting can be added (or removed) without a rewrite.

### 3b. Routing by choice vs fallback by failure — two different mechanisms

It's worth drawing a line now between two things LiteLLM does that are easy to conflate, because week 21 builds entirely on the first and week 24 stress-tests the second.

**Fallback (by failure)** is what §3's `fallbacks` rule does: a request goes to its intended backend, and *only if that backend errors* does it spill to the alternative. It's a resilience mechanism — the vendor is a safety net, not the plan. The user gets the vendor's answer (and its cost) only because the local tier was unavailable. This is reactive.

**Routing (by choice)** is the opposite: you *decide up front*, per request, which backend should handle it — easy questions to the cheap local 7B, hard questions to the frontier vendor model — based on the request's content, not on any failure. The vendor is the plan for *those* requests, deliberately, to get better answers where it matters while keeping the cheap path for everything else. This is proactive, and it's the entire subject of week 21 (model routing and cost engineering).

LiteLLM supports both, and a production system uses both at once: a classifier (week 21) *routes* each request to the cheapest backend that can handle it, and a `fallbacks` rule (this week) catches *failures* of whatever backend was chosen. The two compose cleanly — routing picks the intended target, fallback handles the case where the target is down — and together they form the cost-and-resilience layer the capstone runs on. For now, this week, you build only the fallback half (resilience); week 21 adds the routing half (cost). Knowing they're distinct keeps you from the common confusion of "the router sent it to the vendor" — was that a routing *choice* (week 21) or a fallback *failure* (this week)? They look the same in the logs but mean opposite things about your system's health.

---

## 4. Benchmarking: the discipline, not just the number

"vLLM is fast" is a vibe. "vLLM serves Qwen2.5-14B at 2840 tok/s at concurrency 32 with p95 3.4s on an H100" is an engineering claim. Getting from the first to the second requires methodology, and sloppy benchmarking is the most common way to get a wrong cost number.

The rules:

1. **Fix the workload.** Use the same set of prompts with a controlled prompt-length and output-length distribution every run. If your prompts change between runs, your throughput isn't comparable. (The mini-project takes the workload as config exactly so it's frozen.)
2. **Warm up.** The first few requests pay model-load and CUDA-graph-capture costs; the prefix cache is cold. Discard a warm-up batch before you start timing.
3. **Drive true concurrency.** You must have N requests *in flight simultaneously* to fill the continuous batch. A loop that sends one request, waits, sends the next is concurrency 1 no matter how fast the loop — you'll measure single-stream throughput and conclude (wrongly) that vLLM is slow. Use async (`asyncio.gather` over `httpx.AsyncClient`) or vLLM's built-in benchmark.
4. **Measure the right things:** total output tokens / wall-clock = throughput (tok/s); per-request latencies → p50 and p95; and watch `nvidia-smi`/`nvtop` for GPU utilization (the proof the batch is full).
5. **Sweep concurrency:** 1, 8, 32, 128. One number is a point; the curve is the answer (Lecture 1 §9).

vLLM ships a benchmark you should cross-check against your hand-rolled one:

```bash
vllm bench serve \
  --model Qwen/Qwen2.5-14B-Instruct \
  --base-url http://localhost:8000 \
  --dataset-name random --num-prompts 200 \
  --max-concurrency 32
```

It reports request throughput, output-token throughput, and TTFT/TPOT/latency percentiles. Your exercise-02 load generator should land in the same ballpark; if it's wildly different, your generator isn't driving real concurrency (rule 3) — the classic bug.

> **The trap:** measuring concurrency 1 and calling it "throughput." Single-stream throughput on a 14B is a few hundred tok/s and makes self-hosting look terrible. The whole value of continuous batching shows up only when many requests are in flight. If your benchmark sends requests serially, you have measured the *worst* case and will compute a cost-per-million 10× too high. True concurrency is non-negotiable.

A short checklist of the *other* ways benchmarks lie, so you can audit your own:

- **Counting input tokens as throughput.** Prefill processes the whole prompt in one shot, so folding prompt tokens into your tok/s inflates the number and hides the decode bottleneck. Report *output* throughput for the cost math; report prompt-token rate separately if at all.
- **Estimating tokens instead of reading `usage`.** `len(text.split())` is not the token count — it's off by 20–40% and varies by content. The response's `usage.completion_tokens` is ground truth; use it always. An estimated denominator corrupts every cost number downstream.
- **Forgetting the warm-up.** The first requests pay model-load and CUDA-graph-capture costs and hit a cold prefix cache. Including them drags throughput down and latency up. Discard a warm-up batch before timing.
- **A workload that drifts.** If the prompts or `max_tokens` change between the "before" and "after" runs, the comparison is meaningless. Freeze the workload (the mini-project takes it as config for exactly this reason) so the only thing that varies is the server config you're testing.
- **One run, no variance.** A single sweep is a point estimate; GPUs have noisy neighbors and thermal throttling. Run the sweep 2–3 times and at least confirm the *shape* reproduces.

Every one of these flips the break-even decision by inflating or deflating the cost number. The discipline is the rest-of-the-course discipline: a measured claim with a stated method beats a confident number with no method.

---

## 5. From throughput to cost-per-million-tokens

Once the sweep gives you a throughput at your chosen operating point, the cost math is arithmetic (Lecture 1 §9, restated as a recipe):

```python
gpu_cost_per_hour = 2.50          # rented H100, your actual rate
throughput_tok_s  = 2840          # measured output-token throughput at the operating point

cost_per_1M = (gpu_cost_per_hour / 3600) / throughput_tok_s * 1_000_000
# = (2.50/3600) / 2840 * 1e6 ≈ $0.244 per 1M output tokens
```

A few honesty checks that separate a real number from a fantasy:

- **Output tokens vs total tokens.** Vendors price input and output separately (input is cheaper). Your self-hosted cost should be computed on the metric you'll compare — usually output-token throughput, since decode is the bottleneck. Be explicit about which you're using on both sides.
- **Utilization, not peak.** That $0.244 assumes the GPU runs at the operating-point throughput *all the time*. Real traffic is bursty: peaks at noon, near-idle at 3 AM. Your *effective* cost is the GPU cost divided by your *average* throughput, which is lower than peak. A GPU that's 40% utilized over a day costs ~2.5× more per token than its peak-throughput number suggests. **The peak number is the best case; the utilization-weighted number is the truth.**
- **On-demand vs spot vs owned.** $2.50/h is on-demand rental. Spot is cheaper but can be reclaimed (a chaos-drill scenario). Owned hardware has capex amortization and a power bill. The break-even moves with which you assume; state it.

---

## 6. The break-even: when self-hosting wins

Now the headline deliverable's math (Lecture 1 §10, made concrete). Two cost functions of monthly token volume `V` (in millions):

```
vendor_cost(V)      = vendor_price_per_M × V                       # zero fixed cost
selfhost_cost(V)    = gpu_cost_per_month + tiny_marginal × V       # high fixed, ~flat
```

`gpu_cost_per_month = gpu_cost_per_hour × 24 × 30` (if you run it 24/7). The marginal per-token cost of self-hosting is electricity — negligible next to the fixed GPU cost — so `selfhost_cost` is essentially a flat line at the monthly GPU cost. Set the two equal and solve for the break-even volume:

```python
gpu_cost_per_month = 2.50 * 24 * 30          # ≈ $1800/month for a 24/7 H100
vendor_price_per_M = 15.00                    # vendor output price, $/1M tokens

breakeven_M = gpu_cost_per_month / vendor_price_per_M
# = 1800 / 15 = 120 million output tokens / month
```

So: if you serve **fewer than ~120M output tokens/month**, the vendor is cheaper (you'd be paying $1800 for a GPU you barely use). **Above ~120M/month**, self-hosting wins, and the more you serve the bigger the win — because the GPU's fixed cost spreads over more tokens while the vendor charges linearly forever.

But notice what this number hides, and what the memo must surface:

- **The break-even assumes you can keep the GPU busy enough to hit 120M.** 120M tokens/month at 2840 tok/s requires the GPU running at that throughput ~12 hours a day. If your traffic can't fill the batch (utilization risk), your *effective* break-even volume is higher — you need *more* monthly volume to justify the GPU, because you're wasting capacity.
- **Operational cost is real and unpriced here.** Someone maintains the GPU box, patches CUDA, gets paged when it OOMs at 2 AM. The vendor bundles all that into the per-token price. A fair memo adds an estimate of operational FTE cost to the self-hosted side, which pushes the break-even up.
- **Reliability and bursting.** The vendor absorbs your traffic spikes elastically; your single GPU does not. The capstone's *hybrid* answer — self-host the steady base load, fall over / burst to the vendor for spikes and for the "hard" routes (week 21) — is usually the right architecture precisely because neither pure option is best.

> **The discipline:** never claim "self-hosting is cheaper" without the break-even volume and your actual monthly volume next to it. "We serve 400M tokens/month, break-even is 120M, so self-hosting saves ~$3600/month before ops cost" is an answer. "vLLM is cheaper than the API" is a confession that you didn't do the arithmetic.

Here is a worked memo's worth of numbers, end to end, so the whole chain is visible in one place. Suppose your sweep settled on concurrency 64 as the operating point (good throughput, p95 still under your 4s SLO):

```
measured throughput at concurrency 64   = 2400 output tok/s
GPU rental                              = $2.50 / hour  (on-demand H100)
vendor (Claude Haiku 4.5) output price  = $5.00 / 1M output tokens

self-hosted $/1M output                 = (2.50/3600) / 2400 × 1e6  = $0.29 / 1M
fixed monthly GPU cost (24/7)           = 2.50 × 24 × 30             = $1,800 / month
break-even volume vs Haiku              = 1,800 / 5.00              = 360 M tok/month
```

Now the decision lands on where your expected volume sits:

- **Expected 50M tokens/month?** Below break-even. The vendor costs `50 × $5 = $250/month`; the GPU costs $1,800/month for capacity you barely touch. **Use the vendor.** Self-hosting here is paying $1,800 to save $250 — a $1,550/month mistake.
- **Expected 1,000M tokens/month?** Far above break-even. The vendor costs `1000 × $5 = $5,000/month`; the GPU costs $1,800. **Self-host** — you save ~$3,200/month before ops cost, and the gap widens with volume.
- **Expected 360M tokens/month?** You're at the line. The two are a wash on raw cost, so the *other* factors decide: operational burden tips toward the vendor; data-locality or latency requirements might tip toward self-hosting. At the break-even point, the spreadsheet abstains and judgment takes over — which is exactly why the memo must name the volume and the caveats, not just a winner.

And re-run that middle row with the utilization caveat applied. If your traffic only fills the batch ~30% of the day (bursty, daytime-heavy), your *effective* throughput averaged over the month is closer to 720 tok/s, not 2400. Recompute: `(2.50/3600)/720 × 1e6 = $0.96/1M` effective, and the break-even volume rises to ~`1,800/5.00`... no — the fixed cost is unchanged, but you now need *more* monthly volume to keep the GPU justified, because much of it sits idle. The honest framing: **the GPU costs $1,800 whether you serve 50M or 360M tokens; self-hosting only pays off if you can actually route enough traffic through it to clear break-even.** A team that self-hosts for a workload that can't fill the batch is buying an expensive idle GPU and calling it savings.

---

## 7. Prefix caching and spec decoding, measured (not assumed)

Lecture 1 §7–8 described these as levers. In the serving context, the discipline is the same as everywhere in this course: **measure the lift on your workload, don't trust the brochure.**

**Prefix caching in the benchmark.** Run the sweep twice: once with `--enable-prefix-caching` off, once on, with a workload that has a long *shared* system prompt across all requests. You should see throughput rise (skipped prefill compute) and latency drop (no re-prefill of the shared head). Then run a third time where each prompt has a *unique* prefix (a per-request ID at the front) and watch the lift vanish — the cache never hits. That third run is the lesson: prefix caching is free money *only* when the prefix is genuinely shared and byte-stable.

**Speculative decoding in the benchmark.** Turn it on and sweep concurrency. At concurrency 1, you should see lower per-request latency (the draft model fills idle compute). At concurrency 64, you may see throughput *drop* — the batch already saturates the GPU, the draft model's compute is no longer free, and verification is pure overhead. The two curves cross somewhere in the middle. The takeaway: spec decode is a *latency* optimization for low-concurrency / single-user paths, not a throughput optimization for a saturated server. Putting it on a high-throughput batch server can make things worse.

Both are in the README stretch goals because the *measurement* is the lesson — a flag you turned on without measuring its effect is a flag you're cargo-culting.

---

## 8. Multi-replica, health, and the failover dry run

The capstone runs *multiple* vLLM replicas behind LiteLLM, and week 24 kills one on purpose. The setup, conceptually:

1. **Two (or more) replicas** of the same model, each a `vllm serve` on its own GPU/port, registered under the *same* LiteLLM `model_name` alias. LiteLLM load-balances across them (round-robin or least-busy).
2. **Health checks** — LiteLLM probes each backend; a replica that fails its health check is taken out of rotation so traffic stops landing on a dead GPU.
3. **Fallback to vendor** — if *all* replicas are unhealthy, the `fallbacks` rule (§3) spills to the vendor. The user sees higher latency / vendor cost, but not an error.

The exercise-03 + challenge stretch is to run two replicas, start a sweep, kill one replica mid-sweep, and measure the user-visible blip: how many requests error before LiteLLM notices, how latency moves while running on one replica, and whether the vendor fallback kicks in if you kill both. This is a controlled rehearsal for the week-24 chaos drill ("kill one vLLM replica; verify the LiteLLM router fails over"). Doing it now, with a benchmark running, means in week 24 you already know what healthy failover looks like.

---

## 8b. TTFT, TPOT, and why "latency" is two numbers

So far "latency" has meant one number — how long a request takes end to end. For a streaming chat UI, that's the wrong granularity, and the benchmark reports two finer metrics you should understand because they map to two different parts of the inference two-step (Lecture 1 §1).

**TTFT — time to first token.** How long from sending the request until the *first* output token streams back. This is dominated by **prefill**: the model has to process the whole prompt before it can emit token one. A long prompt → long TTFT. Prefix caching (Lecture 1 §7) attacks exactly this — a cached prefix is already prefilled, so TTFT drops to near the cost of prefilling just the novel suffix. For a chat UI, TTFT is the "is it thinking or is it broken?" number: under ~1s feels responsive, several seconds feels dead.

**TPOT — time per output token** (a.k.a. inter-token latency). Once generation starts, how fast do subsequent tokens arrive? This is the **decode** rate, and it's what determines whether the streamed text reads at a comfortable pace. It's gated by the batch: a fuller batch shares each weight-read across more sequences (good for throughput) but each individual sequence's decode steps are interleaved with others, so per-sequence TPOT *rises* as the batch fills. This is the throughput-vs-latency trade-off (Lecture 1 §9) seen at the token level.

```
total request latency  ≈  TTFT  +  (output_tokens − 1) × TPOT
                          └prefill┘   └────── decode ──────┘
```

Why this matters for the sweep and the SLO: a server tuned for maximum throughput (big batch) can have great aggregate tokens/sec but a TPOT so high that streaming feels sluggish, *and* a TTFT spike whenever a long-prompt admission stalls the decode loop. Your SLO is usually expressed in these terms — "TTFT p95 < 800ms, TPOT p95 < 50ms" — not in aggregate throughput. So the operating point you choose on the concurrency curve isn't just "where throughput plateaus"; it's "the highest throughput where TTFT and TPOT still meet the SLO." The `vllm bench serve` output reports all of these, and a complete benchmark records them — quoting only aggregate throughput hides whether the server is actually usable for a streaming UI.

> **The discipline:** for an interactive product, measure TTFT and TPOT, not just throughput. Aggregate tokens/sec sells the GPU; TTFT and TPOT are what the user feels. The cheap-but-sluggish operating point and the fast-but-expensive one are both real choices on the same curve — pick deliberately.

## 9. The carbon footnote

The syllabus mentions the carbon story, and it's a one-paragraph point worth making honestly. A GPU draws roughly the same power whether it's serving one request or a full batch — the dynamic range between idle and busy is real but not huge. So an *underutilized* GPU is carbon-inefficient: you burn ~the same watts to produce a fraction of the tokens. **Keeping the GPU busy (continuous batching, high utilization) is the most carbon-efficient thing you can do per token.** This aligns the carbon argument with the cost argument: the same discipline — fill the batch, keep utilization high — minimizes both dollars-per-token and grams-of-CO₂-per-token. A near-idle GPU is the worst of both. The vendor, by pooling many customers onto shared, highly-utilized hardware, often has a better per-token carbon profile than a lightly-loaded private GPU — another reason the break-even volume matters beyond dollars.

---

## 9b. A worked break-even, with the vendor table

Make the break-even concrete so the number stops being abstract. Fix the hardware: one H100 at **$2.50/hr** is `2.50 × 24 × 30 = $1,800/month` of fixed cost, paid whether you serve one token or a trillion. The 2026 Claude prices, blended at a 1:1 input:output chat mix, are the prices to beat:

| Vendor tier | Input / Output $/MTok | Blended (1:1) $/MTok | Break-even = $1,800 ÷ blended |
|---|---|---:|---:|
| `claude-haiku-4-5` | $1 / $5 | $3 | **600M tokens/month** |
| `claude-sonnet-4-6` | $3 / $15 | $9 | **200M tokens/month** |
| `claude-opus-4-8` | $5 / $25 | $15 | **120M tokens/month** |

Read the right column: against Haiku you must clear **600M tokens/month** before the H100 is worth it; against Opus, only **120M**, because the thing you're beating is five times pricier so the fixed rental amortizes far sooner. Now layer in throughput: at concurrency 32 your H100 serves ~2,840 tok/s → `(2.50/3600)/2840 × 1e6 ≈ $0.24/MTok`, which undercuts even Haiku's $3 by ~12×. So *above* break-even the win is enormous; the entire question is whether your real volume clears the crossover.

```text
                              $1,800/mo  vendor bill at volume V (blended $/MTok)
volume (tokens/month)         crosses    Haiku $3      Sonnet $9     Opus $15
  100M                        below      $300          $900          $1,500
  600M  ← Haiku break-even    at         $1,800        $5,400        $9,000
  1B                          above      $3,000        $9,000        $15,000
```

At 1B tokens/month you self-host and save $1,200/month vs Haiku (and $13,200 vs Opus) — *if* you keep the GPU busy enough to sustain that throughput. At 100M tokens/month every vendor is cheaper, because $1,800 of mostly-idle GPU beats a $300 Haiku bill. **The decision is: where does your expected volume land in this table, and can you sustain the utilization that makes the $0.24/MTok real?** That's the memo, and it's why you measure your own volume and throughput rather than quoting a vendor benchmark.

And note the answer is rarely all-or-nothing. The same LiteLLM router that gives you a vendor *fallback* lets you run a **hybrid**: self-host the steady base load (where you clear break-even and the GPU stays busy) and route the spiky overflow — and the hard prompts that need a frontier model — to the vendor. That keeps your self-hosted utilization high (you only send it the volume it can keep busy) while the vendor absorbs the bursts you'd otherwise overprovision a second GPU for. The break-even table tells you *which* traffic belongs on which tier: base load above the crossover → local; overflow and quality-sensitive routes → vendor. The memo's strongest form isn't "self-host vs vendor" but "here's the volume split, and here's the router config that implements it."

---

## 10. Recap

You should now be able to:

- Stand up **`vllm serve`** (or the Docker container with `--ipc=host` and the HF cache mount), read the KV-block line in the startup log, and call it with the **unmodified `openai` client** by changing only `base_url`.
- Put **LiteLLM** in front of one or more backends with **model aliases**, multi-replica load balancing, and a **vendor `fallbacks`** rule — the capstone's router and failover.
- Run a **disciplined benchmark**: fixed workload, warm-up, *true* concurrency via async, measure throughput + p50/p95 + GPU util, and sweep 1/8/32/128 — and recognize the concurrency-1 trap that inflates cost-per-token.
- Convert a measured **throughput into cost-per-million-tokens**, and distinguish the peak number from the utilization-weighted truth.
- Compute the **break-even volume** against a vendor price, and state the caveats — utilization risk, operational cost, bursting — that make the hybrid (self-host base load, vendor for spikes/hard routes) the usual right answer.
- Treat **prefix caching and speculative decoding** as measured levers (prefix caching: free on shared stable prefixes; spec decode: a low-concurrency latency win, a high-concurrency throughput tax), and explain the **carbon** alignment with the cost argument.

Next week stands the NVIDIA enterprise stack — NeMo Inference, TensorRT-LLM, Triton, NeMo Guardrails — next to this vLLM deployment on the same H100, and re-runs your sweep so you can decide which serving story the capstone signs up for. Push the `crunchserve` harness; week 20 points it at a NeMo/Triton backend.

---

## References

- *vLLM documentation — OpenAI-Compatible Server*: <https://docs.vllm.ai/en/latest/>
- *LiteLLM — Proxy Server & Routing*: <https://docs.litellm.ai/docs/simple_proxy>
- *LiteLLM — Fallbacks / Reliability*: <https://docs.litellm.ai/docs/routing>
- *vLLM benchmarking (`vllm bench serve`)*: <https://docs.vllm.ai/en/latest/contributing/benchmarks.html>
- *PagedAttention paper* — Kwon et al., 2023: <https://arxiv.org/abs/2309.06180>
- *Anthropic model pricing (for the vendor side of the break-even)*: <https://platform.claude.com/docs/en/pricing>
