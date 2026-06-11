# Week 1 — Resources

Every resource here is **free** to read. The model cards, the SDK docs, and the open-weights licenses are all published openly. The two papers are on arXiv. No paywalled books are linked. Where a link is to a vendor's docs that move over time, the title is given so you can re-find it if the URL drifts.

Pin yourself to **2026-current** model IDs when you write code: the hosted frontier model in this course is Anthropic Claude — `claude-opus-4-8` (most capable), `claude-sonnet-4-6` (balanced), `claude-haiku-4-5` (fast/cheap) — and the local model is whatever Ollama tag you pulled (`qwen2.5:7b`, `llama3.2:3b`, `gemma3:4b`). Do not copy older model strings from blog posts; they may be retired.

## Required reading (work it into your week)

- **The Anthropic Messages API — Models overview.** The canonical list of current model IDs, context windows, and pricing. Read it Monday and keep it open all week:
  <https://docs.claude.com/en/docs/about-claude/models/overview>
- **Llama 4 model card** (Meta). The flagship open-weights release of the era; read the license section twice:
  <https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md>
- **Qwen 3 model card / blog** (Alibaba). The other major open-weights family; note the per-variant license differences:
  <https://qwenlm.github.io/blog/qwen3/>
- **Ollama — README and model library.** How the local-inference path you use all course actually works:
  <https://github.com/ollama/ollama> · library: <https://ollama.com/library>

## The papers (read for shape, not for math)

You will not implement these. You read them so the vocabulary is yours when it reappears in weeks 5, 13, and 19.

- **Attention Is All You Need** (Vaswani et al., 2017). One paragraph per day this week. By Friday you can say what query/key/value are *for*. Skip the positional-encoding math:
  <https://arxiv.org/abs/1706.03762>
- **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023 — the vLLM paper). Read only the introduction and §2 this week — it is where "prefill vs decode" and "KV cache" are explained more rigorously than any blog:
  <https://arxiv.org/abs/2309.06180>

## The licenses (the parts that decide whether you can ship)

Read these as an engineer reads a contract — Ctrl-F for "commercial," "monthly active users," "derivative," "distribute," "trademark."

- **Apache License 2.0** — the gold standard for "you can do almost anything." Mistral and several Qwen variants ship under it:
  <https://www.apache.org/licenses/LICENSE-2.0>
- **Llama 4 Community License Agreement** — *source-available*, not OSI-open. Note the 700M-MAU clause and the acceptable-use policy:
  <https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE>
- **Gemma Terms of Use + Prohibited Use Policy** (Google) — permissive for commercial use but with an enforceable use policy attached:
  <https://ai.google.dev/gemma/terms>

## API & SDK references (open all week)

- **Anthropic Python SDK** — `client.messages.create`, `client.messages.count_tokens`, streaming, usage accounting:
  <https://github.com/anthropics/anthropic-sdk-python>
- **Anthropic — Token counting** — the correct way to count tokens for a hosted model (never `tiktoken` for a non-OpenAI model):
  <https://docs.claude.com/en/docs/build-with-claude/token-counting>
- **Ollama API reference** — the `/api/generate` and `/api/chat` HTTP endpoints, the `eval_count` / `prompt_eval_count` / `*_duration` fields you will measure:
  <https://github.com/ollama/ollama/blob/main/docs/api.md>
- **Hugging Face `tokenizers` / `transformers.AutoTokenizer`** — for the local tokenizers you will inspect this week and explore in depth next week:
  <https://huggingface.co/docs/transformers/main_classes/tokenizer>

## Landscape maps (read skeptically)

Leaderboards are useful for orientation and dangerous for decisions. Read these to know what exists, not to pick.

- **LMArena (formerly Chatbot Arena) leaderboard** — human-preference Elo across models. Good for "which models are even in the conversation"; bad for "which one fits my extraction job":
  <https://lmarena.ai/leaderboard>
- **Hugging Face Open LLM Leaderboard (archive)** — academic benchmarks across open-weights models; read the "why this can mislead" caveats:
  <https://huggingface.co/open-llm-leaderboard>
- **Artificial Analysis — models comparison** — a cost/latency/quality cross-section that is closer to a product-fit lens than a pure-capability leaderboard:
  <https://artificialanalysis.ai/models>

## Talks worth your time (free, no signup)

- **"Intro to Large Language Models"** (Andrej Karpathy, 1 hour). The clearest "what an LLM is from the outside" talk that exists. Watch it Monday night:
  <https://www.youtube.com/watch?v=zjkBMFhNj_g>
- **"Let's build the GPT Tokenizer"** (Karpathy). Optional this week, required intuition for week 2:
  <https://www.youtube.com/watch?v=zduSFxRajkE>

## Tools you'll use this week

- **`ollama`** — `ollama pull qwen2.5:7b`, `ollama run`, and the HTTP API on `localhost:11434`. Your local model for the whole course.
- **`anthropic` (Python SDK)** — `pip install anthropic`. The hosted frontier path.
- **`httpx`** — `pip install httpx`. Async HTTP for talking to Ollama and for the parallel `llmpick` queries.
- **`time.perf_counter()`** — your latency measurement primitive. Use it, not `time.time()`, for durations.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Token** | The unit the model actually reads/writes — a sub-word integer ID, not a character and not a word. |
| **Tokenizer** | The reversible map between text and token IDs. Different model families have different tokenizers. |
| **Context window** | The maximum number of tokens (prompt + generated) the model can condition on at once. |
| **Logits** | The raw, unnormalized scores the model emits — one per vocabulary entry — for the next token. |
| **Forward pass** | One run of the model that turns a token sequence into next-token logits. The expensive part. |
| **Autoregressive** | Generates one token at a time, feeding each output back in as input for the next. |
| **Prefill** | Processing the whole prompt at once to fill the KV cache. Compute-bound; sets time-to-first-token. |
| **Decode** | Generating output tokens one at a time, reusing the cache. Memory-bandwidth-bound; sets time-per-token. |
| **KV cache** | The stored keys/values from prior tokens so each new token doesn't reprocess the whole sequence. |
| **TTFT** | Time-to-first-token. Dominated by prefill (and queueing). What "it feels slow to start" measures. |
| **TPOT** | Time-per-output-token. Dominated by decode. What "it streams fast/slow" measures. |
| **Open-weights** | The weights are downloadable; the *license* governs what you may do (which can still be restrictive). |
| **Source-available** | Downloadable but under a non-OSI license with conditions (e.g., Llama's MAU clause). Not "open source." |
| **Model card** | The document shipped with a model: license, context window, cutoff, intended use, eval caveats. |
| **Training cutoff** | The date past which the model has no parametric knowledge. Anything newer must come from context. |

---

*If a link 404s, please open an issue so we can replace it.*
