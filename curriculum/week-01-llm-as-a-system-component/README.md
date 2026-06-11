# Week 1 — The LLM as a System Component

Welcome to the first week of C23. By Friday you will be able to look at any large language model — open-weights or behind a vendor API — and describe it the way a systems engineer describes a database or a load balancer: as a component with a known interface, known failure modes, a cost curve, and a license you have actually read. You will stop saying "the AI" and start saying "a function from a token sequence to a probability distribution over the next token, wrapped in sampling and scheduling."

This is the mental model the entire course is built on. Everything downstream — retrieval, agents, serving, evaluation — is engineering *around* that function. If you internalize the function this week, the next twenty-three weeks are about the wrapper.

The one sentence to carry out of this week: **an LLM is a deterministic function `tokens → logits`; everything you experience as "intelligence," "creativity," or "randomness" is what your code does with those logits afterward.** The model does not "decide" to be creative. Your sampler does. The model does not "remember" your conversation. Your context-assembly code does. The model does not "know" today's date. Your prompt does, or it doesn't. Once you see the seam between the function and the wrapper, you can debug the wrapper — and almost every production LLM bug lives in the wrapper, not the model.

We assume nothing about prior LLM experience beyond having called an API once and felt the shape of a product underneath. We assume Python fluency, Linux comfort, and the willingness to run Ollama on your laptop. We do **not** assume you have read *Attention Is All You Need*; we re-derive the parts that matter for systems work and skip the parts that matter for training a model from scratch (that is C5's job).

## Learning objectives

By the end of this week, you will be able to:

- **Describe** the external interface of an LLM end-to-end — tokenizer → context window → forward pass → logits → sampling → detokenizer — and name which stage owns which observable behavior (latency, cost, determinism, truncation).
- **Explain** the decoder-only transformer at a systems level: what the forward pass computes, why it is `O(n²)` in sequence length, what the KV cache buys you, and why the first token is slow and the rest are fast — without writing a single line of attention math.
- **Enumerate** the 2026 model landscape — open-weights families (Llama 4, Qwen 3, Mistral, Gemma 3, DeepSeek) versus closed-weights frontier tiers (Claude 4 class such as `claude-opus-4-8`, GPT-5 class, Gemini 2.5 class) — and place a given model on the open/closed and capability/cost axes.
- **Read** a model card without falling for the leaderboard: locate the license, the context window, the training-cutoff date, the intended-use and out-of-scope sections, and the benchmark caveats, and separate "leaderboard score" from "fit for my job."
- **Reason** about model licensing as an engineering constraint: distinguish a true open-source license (Apache-2.0, MIT) from a source-available community license (Llama, some Qwen/Gemma variants), and identify the commercial-use, royalty-threshold, and derivative-works clauses that decide whether you can ship.
- **Call** the same prompt against multiple models — a hosted frontier model via the Anthropic SDK and a local open-weights model via Ollama — through one uniform interface, and measure latency, tokens-in, tokens-out, and cost per request.
- **Build** a CLI tool, `llmpick`, that queries N models in parallel for a prompt under a budget and a latency target, and recommends one with reasons you can defend in a review.

## Prerequisites

This week assumes you have met the course-level prerequisites (C1 Python fluency, Linux comfort, Docker basics) and, specifically:

- **Python 3.12** (the Ubuntu 24.04 / current-macOS default). `python3 --version` works; you can create a venv and `pip install`.
- An **Anthropic API key** in your environment as `ANTHROPIC_API_KEY`. The hands-on lab calls a hosted frontier model; the free-tier credit covers the week's traffic many times over. If you cannot get a key, every lab has a local-only fallback path documented inline.
- **Ollama** installed and a small model pulled: `ollama pull qwen2.5:7b` (or `llama3.2:3b` if you have ≤16 GB RAM). `ollama run qwen2.5:7b "hello"` returns text.
- You can read a JSON response and a `pyproject.toml`, and you are comfortable with `async`/`await` at the "I have seen `asyncio.gather` before" level — we re-explain it where it appears.

You do **not** need a GPU this week. A 16 GB laptop carries every exercise; the local model runs on CPU/Metal. You do **not** need to understand backpropagation, attention math, or PyTorch — we stay strictly at the interface.

## Topics covered

- **The model as a function.** The precise external contract: input is a sequence of integer token IDs; output is a vector of logits, one per vocabulary entry, for the *next* token. Everything else — chat templates, system prompts, "roles" — is encoding that collapses to token IDs before the model sees it.
- **The five stages.** Tokenizer (text → IDs), context window (the bounded buffer of IDs the model conditions on), forward pass (IDs → logits, the expensive part), sampling (logits → one chosen ID), detokenizer (IDs → text). Which stage owns latency, which owns cost, which owns determinism.
- **The decoder-only transformer at a systems level.** Embeddings, stacked attention+MLP blocks, the final unembedding to vocabulary logits. Why attention is quadratic in sequence length, why that makes long context expensive, and what "the model is autoregressive" means for how you stream output.
- **Prefill versus decode.** The two phases of generation: prefill (process the whole prompt at once, fill the KV cache — compute-bound) and decode (generate one token at a time, reuse the cache — memory-bandwidth-bound). Why time-to-first-token and time-per-output-token are *different* numbers you must measure separately.
- **The 2026 model landscape.** Open-weights families and what each is good at; closed-weights frontier tiers and when their capability premium is worth the lock-in. The honest map: capability, cost, context window, license.
- **Reading a model card.** Where the load-bearing information lives — license, context window, cutoff date, intended use, out-of-scope use, eval caveats — and the gap between a leaderboard rank and product fit.
- **Licensing as engineering.** Apache-2.0/MIT (true open source) versus community/source-available licenses (Llama's acceptable-use + 700M-MAU clause, Gemma's prohibited-use policy, Qwen's mostly-Apache-with-exceptions). Commercial use, royalty thresholds, derivative-works and distribution obligations.
- **The uniform client.** Wrapping heterogeneous providers (Anthropic SDK, Ollama HTTP) behind one `complete(prompt) -> Completion` interface so the rest of the course can swap models with one config change — the "the course is the engineering, not the import" principle in code.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                       | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | The model as a function; the five stages; first API call   |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Decoder-only transformer at a systems level; prefill/decode|    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | The 2026 landscape; reading a model card; the multi-model lab |  2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Licensing as engineering; cost & latency measurement       |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The uniform client; `llmpick` design; studio              |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                      |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                   |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                            | **6h**   | **7h**    | **3h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | Model cards, the SDK docs, the licenses, the papers, and the talks worth your time |
| [lecture-notes/01-the-model-is-a-function.md](./lecture-notes/01-the-model-is-a-function.md) | The five stages, the decoder-only transformer at a systems level, prefill vs decode, the KV cache |
| [lecture-notes/02-the-landscape-cards-and-licenses.md](./lecture-notes/02-the-landscape-cards-and-licenses.md) | The 2026 model landscape, reading a model card, licensing as an engineering constraint |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-read-three-model-cards.md](./exercises/exercise-01-read-three-model-cards.md) | Read Llama 4, Qwen 3, and a frontier model card; build a comparison table and a license verdict |
| [exercises/exercise-02-uniform-client.py](./exercises/exercise-02-uniform-client.py) | A `complete()` interface over Anthropic + Ollama with token and latency instrumentation |
| [exercises/exercise-03-prefill-vs-decode.py](./exercises/exercise-03-prefill-vs-decode.py) | Measure time-to-first-token vs time-per-output-token against a local model and explain the gap |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-model-selection-memo.md](./challenges/challenge-01-model-selection-memo.md) | Pick a model for three contrasting jobs and defend each in a one-page memo with measured numbers |
| [quiz.md](./quiz.md) | 14 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the model-selection memo |
| [mini-project/README.md](./mini-project/README.md) | `llmpick` — the parallel multi-model recommender CLI |

## The "I can defend this number" promise

C23 has a recurring marker for every artifact that ends in a decision: a number you can defend in a review. This week it looks like this:

```
$ llmpick --prompt "Summarize this 800-word incident report in 3 bullets." \
          --budget 0.002 --latency-target 2.0

RECOMMENDATION: claude-haiku-4-5
  reason: met the 2.0s p50 latency target (measured 1.3s) and the $0.002
          budget (measured $0.00041/call); quality tier sufficient for
          extractive summarization.
  runner-up: qwen2.5:7b (local, $0.00 marginal) — 0.9s but lower
             instruction-following on the held-out check.
  rejected: claude-opus-4-8 — over budget at $0.0073/call for this task class.
```

If your tool prints a recommendation with no measured latency, no measured cost, and no stated reason, you are not done. A recommendation without numbers is a vibe, and the rubric fails vibes. The point of Week 1 is to make "I measured it, here is the number, here is the trade-off" the ordinary way you talk about models.

## Stretch goals

If you finish the regular work early and want to push further:

- Read **§3.1–3.2 of *Attention Is All You Need*** (the scaled dot-product and multi-head attention definitions) until you can state, in one sentence each, what the query/key/value projections are *for* — not the math, the purpose. You will not need the math; you will need the vocabulary in week 13.
- Pull a **second local model** (`gemma3:4b` or `llama3.2:3b`) and add it to your uniform client. Run the same prompt across three local models and one hosted model; eyeball where the open models keep up and where the frontier model pulls away.
- Instrument your uniform client to **log `usage.cache_read_input_tokens`** on the Anthropic path. Send the same long system prompt twice and watch the second call read from cache. You will formalize this in week 21; seeing it now plants the seed.
- Find the **exact clause** in the Llama 4 community license that triggers the 700-million-monthly-active-user threshold, and write two sentences on what it would mean for a startup that succeeds wildly. This is the kind of thing a founding engineer is expected to know off the top of their head.

## Up next

Week 2 zooms into the first and fourth stages of the function — **tokenization and sampling**. You will build a tokenization explorer across three open tokenizers, instrument token counts and timings, and write a grammar-constrained decoder that emits schema-valid JSON. The uniform client you build this week becomes the harness you instrument next week. Push your `llmpick` repo before you start it.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
