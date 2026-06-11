# Challenge 1 — The vLLM + LiteLLM Break-Even Lab

**Time estimate:** ~150 minutes (plus the GPU rental window — budget ~6 hours of H100 time end to end, and **destroy the instance the moment you have your numbers**).

## Problem statement

Your team needs a local serving tier for the capstone, and someone has to answer the question that decides whether it exists: **at our expected token volume, is self-hosting Qwen2.5-14B on a rented H100 cheaper than calling a vendor API — and by how much?** You will end the argument the only honest way: stand up the server, put a router in front of it with a vendor fallback, drive it under real concurrent load, measure tokens/sec, compute cost-per-million-tokens, find the break-even volume, and write a memo that commits to a decision with the number behind it.

This is the syllabus self-hosted-economics lab. The output is a **serving decision** — self-host or vendor — justified by a measured throughput curve and a break-even volume, not by a vibe about which "feels" cheaper.

## What's fixed (do not let these vary)

- **The model:** `Qwen/Qwen2.5-14B-Instruct` (bf16 on one H100 80GB; the resources recipe). On a no-GPU box, substitute `Qwen/Qwen2.5-0.5B-Instruct --device cpu` so the mechanics are real, or use the `--simulate` path for the numbers.
- **The workload:** a fixed, repeatable chat workload — one shared system prompt (so prefix caching has something to cache) plus short user prompts with a fixed `max_tokens`. Fixing the workload is what makes the concurrency sweep comparable run-to-run and the cost-per-token honest.
- **The metric:** aggregate **tokens/sec** at each concurrency level (the denominator of the cost formula), plus p50/p95 latency and req/sec. Cost-per-million-tokens and the break-even volume follow from tokens/sec by the Lecture 2 §5 formula.
- **The concurrency sweep:** 1, 8, 32, 128. The whole point is to see the curve, so you must hit all four.

## The harness approach

Three pieces: the server, the router, the benchmark.

**1. The vLLM server** (Lecture 1's launch command):

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
    --gpu-memory-utilization 0.90 --max-model-len 8192 \
    --max-num-seqs 256 --enable-prefix-caching --port 8000
```

**2. LiteLLM in front, with a vendor fallback** — `config.yaml`:

```yaml
model_list:
  - model_name: qwen-local                       # public name clients request
    litellm_params:
      model: openai/Qwen/Qwen2.5-14B-Instruct     # "openai/" => OpenAI-compatible (vLLM)
      api_base: http://localhost:8000/v1
      api_key: EMPTY                              # vLLM ignores it; must be non-empty
  - model_name: claude-fallback
    litellm_params:
      model: anthropic/claude-haiku-4-5           # the vendor fallback target
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  routing_strategy: least-busy
  num_retries: 2
  fallbacks:
    - qwen-local: ["claude-fallback"]             # qwen-local fails -> retry on Claude
```

```bash
litellm --config config.yaml      # serves an OpenAI-compatible proxy on :4000
```

**3. The benchmark** — drive concurrent requests and read tokens from `usage` (Exercise 2's logic). Sketch:

```python
import asyncio, time, statistics, httpx, numpy as np

BASE = "http://localhost:4000"   # the LiteLLM proxy (or :8000/v1 to hit vLLM directly)

async def one(client):
    t0 = time.perf_counter()
    r = await client.post(f"{BASE}/chat/completions", json={
        "model": "qwen-local",
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": "Name one benefit of batching."}],
        "max_tokens": 64,
    }, headers={"Authorization": "Bearer sk-anything"}, timeout=120)
    out = r.json()["usage"]["completion_tokens"]          # EXACT token count
    return time.perf_counter() - t0, out

async def level(concurrency, rounds=3):
    lat, toks, wall0 = [], 0, time.perf_counter()
    async with httpx.AsyncClient() as c:
        for _ in range(rounds):
            for l, t in await asyncio.gather(*[one(c) for _ in range(concurrency)]):
                lat.append(l); toks += t
    wall = time.perf_counter() - wall0
    return {"tps": toks / wall, "p50": statistics.median(lat),
            "p95": float(np.percentile(lat, 95)), "rps": concurrency*rounds/wall}

for c in (1, 8, 32, 128):
    print(c, asyncio.run(level(c)))
```

Then compute the cost (Exercise 3): `cost_per_MTok = gpu_$hr / (tps * 3600) * 1e6`, and `break_even = gpu_$hr * 720 / vendor_$per_token`.

## Acceptance criteria

- [ ] A `challenge-01/` directory with the vLLM launch command, the LiteLLM `config.yaml` (with the `fallbacks` rule), and a runnable `benchmark.py`.
- [ ] A throughput table reporting **tokens/sec, p50, p95, and req/sec** at concurrency **1, 8, 32, 128**, with tokens/sec demonstrably climbing as concurrency rises.
- [ ] Cost-per-million-tokens computed at your production concurrency, and the **break-even monthly volume** vs `claude-haiku-4-5` (blended).
- [ ] LiteLLM is actually in the path: at least one request routed through the proxy's `qwen-local` name (not vLLM directly), and the `fallbacks` rule is present in the config.
- [ ] A one-page `serving-memo.md` that names the **decision** (self-host or vendor) for a stated expected volume, gives the tokens/sec and $/MTok behind it, and states the utilization assumption it depends on.
- [ ] One **promise-format trace** showing the curve held: `concurrency 128 still served, tokens/sec up ~Nx over concurrency 1`.

## The trap (read after a first attempt)

The trap is **measuring throughput at concurrency 1 and concluding self-hosting is uneconomical.** At concurrency 1 the GPU is memory-bound and mostly idle (Lecture 1 §1): you'll see ~40 tokens/sec and ~$17/MTok — *worse* than every vendor tier — and conclude, wrongly, that self-hosting makes no sense. **You measured an unloaded server.** The entire point of vLLM is the *loaded* server; you must drive concurrency 32/128 to see the $0.32/MTok number that actually decides the question. If your table only has a concurrency-1 row, you fell in the trap.

A second, subtler trap: **forgetting that the idle GPU dominates cost, so utilization is the real lever.** Your benchmark measures *peak* tokens/sec at full concurrency, but you pay the H100's rental **24/7 whether or not you keep it busy.** If your real traffic averages a few requests at a time, your GPU is idle most hours, your *effective* (month-averaged) tokens/sec is far below the peak, and your *effective* $/MTok is far worse than the benchmark's best case. The memo must state the utilization assumption — "this $/MTok holds only if we sustain ~X concurrency" — or it's quoting a peak number to justify an average bill.

## Stretch goals

- **Add speculative decoding and re-measure.** Restart vLLM with `--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'` and re-run the sweep. Measure the latency change at concurrency 1 *and* at 128 — does the win survive on a saturated batch? (Usually it doesn't: the batch is already full, so there's no idle compute for speculation to soak up — Lecture 2 §4.3.) Report the acceptance behavior you observe.
- **Add a second replica behind LiteLLM and load-balance.** Launch a second `vllm serve` on `:8001`, add a second `qwen-local` `model_list` entry pointing at it, and watch `least-busy` spread load. Then **kill one replica mid-benchmark** and confirm the survivor keeps serving (and that the `fallbacks` rule catches requests if you kill *both*). That's week 24's chaos drill, rehearsed.

## Why this matters

The capstone (weeks 22–24) serves its **local tier from exactly this pattern**: vLLM replicas behind LiteLLM with a vendor fallback. The break-even memo you write here is the artifact that justifies that tier existing — a reviewer will point at your local serving and ask "why self-host, and how do you know it beats the vendor?", and this lab *is* that conversation, rehearsed with numbers. And the fallback you configure here is the safety net **week 24's chaos drill kills a replica to verify**: it kills a `qwen-local` backend and checks that LiteLLM rides over the loss and, if the whole pool dies, fails over to Claude — with the user never seeing an error. The throughput climbed, the cost fell, and you can prove which side of break-even your volume lands on.
