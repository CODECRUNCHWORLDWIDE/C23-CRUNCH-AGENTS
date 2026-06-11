# Exercise 1 — Bring Up the Same 7B on Three Engines

**Goal:** Stand the *same* 7B model up on **Ollama**, **llama.cpp**, and **vLLM**, and hit each one through its OpenAI-compatible endpoint with a *single* Python client — proving the "one client, three backends, swap the `base_url`" pattern that is the spine of the whole week. You will train the most important reflex of local inference: **the engine is a variable you can swap, and the model is a constant you hold fixed.**

**Estimated time:** 50 minutes. Guided.

---

## Setup

Pick one model and hold it fixed for the whole week. We use **Qwen2.5-7B-Instruct** (Apache-2.0, available as GGUF, AWQ, and FP16). Llama-3.1-8B works too if you prefer — pick one.

```bash
pip install openai          # one client for all three engines
```

You'll bring up three endpoints. Two run on your laptop; the third needs a CUDA GPU (local or rented — see resources.md for the ~$1 rental recipe). **You can do the Ollama and llama.cpp legs today and the vLLM leg on a rented GPU later** — the point of the exercise is the *pattern*, and two engines already prove it.

---

## Step 1 — Ollama (the friction-free leg)

```bash
# Install: curl -fsSL https://ollama.com/install.sh | sh   (Linux) or the Mac app.
ollama pull qwen2.5:7b      # downloads a curated Q4_K_M GGUF
ollama serve &              # exposes the API on :11434, OpenAI-compatible at /v1
```

Confirm it's up:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b","messages":[{"role":"user","content":"Say hi in five words."}]}'
```

You should get a JSON response with a `choices[0].message.content`. That's Ollama serving an OpenAI-compatible endpoint — the same shape as a vendor API.

> Note what Ollama did *for* you: it found the right GGUF, picked a quant (Q4_K_M), and exposed a standard endpoint, all from one `pull`. That friction-freeness is exactly why Ollama is the iteration tool (Lecture 1 §3).

---

## Step 2 — llama.cpp (the portable substrate leg)

Build llama.cpp for your platform (Metal on Mac, CUDA on Linux — see the README), then download a GGUF and serve it:

```bash
# Get a GGUF (the same model, Q4_K_M, so only the ENGINE differs from Ollama):
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
  Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir ./models

llama-server \
  -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 8192                       # context length -> sizes the KV cache
```

Confirm:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"Say hi in five words."}]}'
```

Same JSON shape, different port. You're now running the *same* Q4_K_M weights through llama.cpp directly instead of through Ollama's wrapper — which is a useful thing to confirm: Ollama *is* llama.cpp underneath, so these two should be close in speed and identical in output distribution. Any large gap means a config difference (quant, threads, context), and finding it is the lesson.

> **Stretch — `llama-bench`.** Run `llama-bench -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` and read the `pp` (prefill) and `tg` (decode) tokens/sec. This is the cleanest place to *see* the prefill/decode split (Lecture 1 §2) in real numbers on your hardware.

---

## Step 3 — vLLM (the throughput leg)

This leg needs a CUDA GPU. On a local 24 GB card or a rented L4/A10:

```bash
pip install vllm
# Serve the AWQ 4-bit build (GPU-native 4-bit; the vLLM-appropriate quant):
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
# starts an OpenAI-compatible server on :8000
```

Confirm:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct-AWQ","messages":[{"role":"user","content":"Say hi in five words."}]}'
```

> **Honest confound, noted now:** vLLM is serving an **AWQ** quant and Ollama/llama.cpp are serving a **GGUF Q4_K_M** quant. Both are 4-bit, but they're *different* 4-bit formats — so a vLLM-vs-llama.cpp speed difference is partly engine and partly quant. That's a real confound (exactly like late-chunking's model swap in week 8); in the bakeoff you note it, and where you can, you hold the quant format as close as the engines allow.

---

## Step 4 — One client, three backends

Now the payoff. The *same* Python client hits all three by swapping `base_url`:

```python
from openai import OpenAI

ENGINES = {
    "ollama":   ("http://localhost:11434/v1", "qwen2.5:7b"),
    "llamacpp": ("http://localhost:8080/v1",  "qwen2.5-7b"),
    "vllm":     ("http://localhost:8000/v1",  "Qwen/Qwen2.5-7B-Instruct-AWQ"),
}

PROMPT = "In one sentence: why is decode memory-bandwidth-bound?"

for engine, (base_url, model) in ENGINES.items():
    try:
        client = OpenAI(base_url=base_url, api_key=engine)   # key ignored locally
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=80,
        )
        print(f"\n=== {engine} ===")
        print(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"\n=== {engine} === (not running: {e})")
```

Run it. You should get three answers — same model, same prompt, three engines you stood up yourself. They'll be *similar* but not identical (sampling differs, quant differs slightly), and that's expected. The point is the **pattern**: your code doesn't care which engine answers; it points at a URL. Your week-5 agent gets exactly this treatment (homework) — a `--base-url` and it runs against any local engine.

---

## Step 5 — Write down what you found

Build a small table in `notes/week-06/bring-up.md`:

| Engine | Quant | Started in (s) | First answer looked right? | VRAM (nvidia-smi / asitop) | When I'd use it |
|---|---|---|---|---|---|

Fill one row per engine you brought up. The "When I'd use it" column is the point: you're building the Lecture 1 §5 placement (Ollama=iteration, llama.cpp=portable, vLLM=serving) from your own observation, not from the lecture's table.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] At least **two** engines (Ollama + llama.cpp) serve the same 7B and return a sensible answer to the same prompt; the **vLLM** leg done now or scheduled on a rented GPU.
- [ ] The single Python client hits each running engine by **only changing `base_url`** (and the model name) — no per-engine SDK.
- [ ] You confirmed Ollama and llama.cpp (same Q4_K_M weights) give **close** behavior, and you can explain why (Ollama wraps llama.cpp).
- [ ] You noted the **quant confound** for vLLM (AWQ vs GGUF) in your notes.
- [ ] `notes/week-06/bring-up.md` has one row per engine with the "when I'd use it" column filled from your own observation.

---

## Stretch

- Run `llama-bench` and `ollama` on the *same* GGUF and confirm their prefill/decode tokens/sec match within a few percent (they should — same engine underneath). A big gap means a thread-count or context-length difference; find it.
- Bring the model up a *fourth* way via **MLX** on a Mac (`pip install mlx-lm; mlx_lm.server --model Qwen/Qwen2.5-7B-Instruct-4bit`) and compare its Metal tokens/sec to llama.cpp-Metal. On Apple Silicon, MLX is a real contender (Lecture 1 §6).
- Point your **week-5 ReAct agent** at the Ollama endpoint (give it a `--base-url`). Watch it run a tool-using loop on a model *you* serve — that's the Phase I milestone taking shape (and Homework Problem 4).

---

When this feels comfortable, move to [Exercise 2 — Quantization trade-offs](exercise-02-quantization-tradeoffs.py).
