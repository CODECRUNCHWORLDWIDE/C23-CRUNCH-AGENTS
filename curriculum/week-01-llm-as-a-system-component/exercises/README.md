# Week 1 — Exercises

Three focused drills that turn the lectures into muscle memory. Each takes 30–60 minutes. Do them in order — exercise 3 reuses the uniform-client mental model you build in exercise 2, and both lean on the "model as a function" picture from Lecture 1.

## Index

1. **[Exercise 1 — Read three model cards](exercise-01-read-three-model-cards.md)** — read the Llama 4, Qwen 3, and a frontier-model card; extract the six load-bearing facts into a comparison table; write a one-line "can we ship?" license verdict for each. (~45 min, guided)
2. **[Exercise 2 — The uniform client](exercise-02-uniform-client.py)** — wrap a hosted frontier model (Anthropic SDK) and a local open model (Ollama) behind one `complete(prompt) -> Completion` interface, instrumented for tokens-in, tokens-out, and latency. (~45 min, runnable)
3. **[Exercise 3 — Prefill vs decode](exercise-03-prefill-vs-decode.py)** — measure time-to-first-token vs time-per-output-token against a local model under a short and a long prompt, and explain the gap with the prefill/decode model. (~45 min, runnable)

## How to work the exercises

- **Have your environment ready before you start.** `ANTHROPIC_API_KEY` exported (or accept the local-only fallback), Ollama running with `qwen2.5:7b` pulled, and a venv with `anthropic` and `httpx` installed:

  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install anthropic httpx
  ollama pull qwen2.5:7b   # or llama3.2:3b on a ≤16GB machine
  ```

- **Read the model as a function as you go.** Every time you see a token count, ask: which tokenizer produced it? Every time you see a latency, ask: prefill or decode? The exercises are designed to make those questions reflexive.
- Each runnable exercise (`.py`) ends with an **expected output** block. Your exact numbers will differ (your hardware, your network, the model's nondeterminism) but the *shape* must match. If it doesn't, you're not done.
- **No API key? No problem.** Exercises 2 and 3 both run end-to-end against Ollama alone. The Anthropic path degrades gracefully to an "unavailable" line; the local path carries the lesson.

## Running the Python exercises

The two `.py` files are standalone — no package, no framework. Activate your venv and run them directly:

```bash
source .venv/bin/activate
python3 exercise-02-uniform-client.py
python3 exercise-03-prefill-vs-decode.py
```

Each file's header documents how to use it, the acceptance criteria, and the expected output shape. Read the header before you run.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-01` to compare.
