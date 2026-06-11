# Week 19 — Exercises

Three focused drills that take you from "I have a GPU" to "I measured throughput and proved failover." Each takes 30–60 minutes. Do them in order — exercise 2's load generator drives the server you stand up in exercise 1, and exercise 3 puts a router in front of it.

## Index

1. **[Exercise 1 — Stand up vLLM](exercise-01-stand-up-vllm.md)** — bring up `vllm serve` on your GPU, hit it with the unmodified `openai` client, and read the startup log for the KV-cache block count. (~45 min, guided)
2. **[Exercise 2 — The load generator](exercise-02-load-generator.py)** — build an async load generator and measure throughput + p50/p95 latency at concurrency 1, 8, and 32 against your server. (~50 min, runnable)
3. **[Exercise 3 — The LiteLLM router](exercise-03-litellm-router.py)** — put LiteLLM in front of your backend, add a vendor fallback, and prove failover works by pointing it at a dead backend. (~50 min, runnable)

## How to work the exercises

- **Rent the GPU once, for the whole session.** Exercises 1 and 2 need a live vLLM server. Spin up a rented H100 (or A10/L4 for a 7B), do exercises 1–2 in one sitting, then tear down. Exercise 3's router logic you can develop locally against a mock and only test against the real backend while the GPU is up.
- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install per exercise: on the GPU box `pip install vllm`; on your client `pip install openai httpx "litellm[proxy]"`.
- **Drive true concurrency or measure nothing.** Exercise 2's whole point is that a serial loop measures concurrency 1 no matter how fast it runs. The continuous batch only fills when N requests are in flight at once — that's what `asyncio.gather` is for. If your concurrency-32 throughput looks identical to concurrency-1, you're not actually concurrent.
- **Watch `nvtop` while the load runs.** Open a second SSH session and run `nvtop` (or `watch -n1 nvidia-smi`). Seeing GPU utilization climb from 30% to 90%+ as you raise concurrency is the lesson made visible — the batch filling up.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The `.py` files are standalone. Exercise 2 needs a live vLLM endpoint (from exercise 1); exercise 3 needs LiteLLM installed and, for the failover test, a vendor API key.

```bash
# On the GPU box (exercise 1):
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9 --max-model-len 8192 --enable-prefix-caching

# On your client (exercises 2 and 3), with the venv active:
export VLLM_BASE_URL=http://<gpu-host>:8000/v1
python3 exercise-02-load-generator.py --concurrency 1,8,32
export ANTHROPIC_API_KEY=sk-ant-...   # for the exercise-3 fallback leg
python3 exercise-03-litellm-router.py
```

The first `vllm serve` call downloads the model weights (7B ≈ 15 GB, 14B ≈ 28 GB). Do it on the GPU box's fast disk, not five minutes before your rental clock matters.

## A note on reproducibility

Throughput numbers depend on the GPU, the model, the CUDA/vLLM version, and the workload shape — so your *absolute* tok/s will differ from the expected-output examples. What's reproducible is the *shape*: throughput rises with concurrency then plateaus, p95 latency climbs, and GPU utilization goes up as the batch fills. If raising concurrency does *not* raise throughput, something is wrong (you're not driving real concurrency, or `--max-num-seqs` is throttling the batch) — and that's worth finding before you trust any cost number.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-19` to compare.
