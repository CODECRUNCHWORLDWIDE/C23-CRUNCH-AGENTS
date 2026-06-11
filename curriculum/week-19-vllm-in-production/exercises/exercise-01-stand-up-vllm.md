# Exercise 1 — Stand up vLLM and read its KV-cache log

**Time estimate:** ~45 minutes (plus model-download time). Guided.

## Goal

Bring up a vLLM OpenAI-compatible server on a real GPU, call it with the *unmodified* `openai` client, and read the startup log to find your KV-cache capacity — the number that, per Lecture 1 §2, decides how many concurrent sequences you can serve. By the end you will have proven that self-hosting is a `base_url` change, not a rewrite, and you'll be able to hand-compute your server's max concurrency from its own log.

## Prerequisites

- A GPU box: a rented H100 (for the 14B) or A10/L4 (for the 7B). See [resources.md](../resources.md) for rental options. **Set a teardown alarm.**
- On the GPU box: a fresh venv and `pip install vllm`. (Or use the `vllm/vllm-openai` Docker image — see Lecture 2 §2.)
- On your client (can be the same box): `pip install openai`.

## Steps

### 1. Launch the server

On the GPU box, in your venv:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --port 8000
```

(Use `Qwen/Qwen2.5-14B-Instruct` if you're on an H100.) The first run downloads the weights. Wait for the line `Application startup complete` / `Uvicorn running on http://0.0.0.0:8000`.

### 2. Read the KV-cache report in the startup log

Scroll back through the log and find lines like:

```
INFO ... GPU KV cache size: 122,880 tokens
INFO ... # GPU blocks: 7680, # CPU blocks: 4096
INFO ... Maximum concurrency for 8192 tokens per request: 15.00x
```

Write down three numbers in `exercise-01-notes.md`:

- **# GPU blocks** — your KV-cache capacity in PagedAttention pages.
- **GPU KV cache size (tokens)** — total tokens you can cache at once (≈ blocks × 16).
- **Maximum concurrency** — vLLM's own estimate of how many full-`max-model-len` sequences fit.

Then **hand-check it**: `max_concurrent ≈ total_kv_tokens / max_model_len`. With 122,880 cached tokens and `--max-model-len 8192`, that's ~15 — which should match the "Maximum concurrency" line. You just verified Lecture 1 §2's equation against a real server.

### 3. Call it with the OpenAI client (unchanged code)

On your client, set the base URL and call it exactly like a vendor:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain PagedAttention in two sentences."}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
print("usage:", resp.usage)   # prompt_tokens / completion_tokens — your cost hook
```

Confirm you get a coherent answer and a populated `usage` block. **The only thing that changed from your vendor code is `base_url`** — note that in your write-up.

### 4. Re-launch with a smaller KV cache and watch the number drop

Stop the server and relaunch with `--gpu-memory-utilization 0.50`. Read the new "# GPU blocks" / "Maximum concurrency" line. It should be roughly half. You just observed that **the KV cache — not the weights — is the lever on concurrency**: same model, same VRAM, half the cache budget → half the concurrent sequences. Record both numbers.

## Acceptance criteria

- [ ] `vllm serve` is running and `/v1/chat/completions` returns a coherent response to the `openai` client.
- [ ] `exercise-01-notes.md` records the **# GPU blocks**, **KV cache tokens**, and **Maximum concurrency** from the startup log.
- [ ] You hand-checked `total_kv_tokens / max_model_len ≈ Maximum concurrency` and it matched (within rounding).
- [ ] You re-launched at `--gpu-memory-utilization 0.50` and recorded that the concurrency estimate roughly halved.
- [ ] Your write-up notes that the *only* client change from a vendor call was `base_url`.

## Hint

If `/v1/chat/completions` refuses with a model-not-found error, the `model=` string in your client must exactly match the model name vLLM was launched with (check `GET /v1/models`). If the server OOMs at startup, lower `--gpu-memory-utilization` or `--max-model-len` — you asked for a bigger KV cache than the card has after loading weights. If you can't rent a GPU at all, you can run a *tiny* model on CPU (`vllm serve facebook/opt-125m --device cpu`) just to see the API surface and the (small) KV-cache log — the concepts transfer, only the numbers shrink.

## Why this matters

Every serving decision this week — batch size, concurrency, cost-per-token — traces back to the KV-cache capacity you just read off the log. An engineer who can launch the server and recite its concurrency ceiling from the startup output is an engineer who can size a deployment. The capstone serves on exactly this server; you just met it.
