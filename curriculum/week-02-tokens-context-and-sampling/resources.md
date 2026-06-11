# Week 2 — Resources

Every resource here is **free** to read. The tokenizer libraries, the sampling and constrained-decoding papers, and the structured-output tool docs are all published openly; the papers are on arXiv. Where a link is to docs that move over time, the title is given so you can re-find it if the URL drifts.

Pin yourself to **2026-current** model IDs and tokenizers when you write code: the hosted frontier model is Anthropic Claude — `claude-opus-4-8` (most capable), `claude-sonnet-4-6` (balanced), `claude-haiku-4-5` (fast/cheap) — counted with `client.messages.count_tokens`, **never** `tiktoken`. The local models are Ollama tags (`qwen2.5:7b`, `llama3.2:3b`, `gemma3:4b`), and the tokenizer-explorer work uses Hugging Face `AutoTokenizer` against open checkpoints (Llama 4, Qwen 3, and friends).

## Required reading (work it into your week)

- **Anthropic — Token counting.** The correct way to count tokens for a hosted model. Read it Monday; it is the ground truth for every cost estimate you make this week:
  <https://docs.claude.com/en/docs/build-with-claude/token-counting>
- **Anthropic — Pricing & models overview.** Per-MTok input/output rates and context windows. Your cost formula's price table lives here:
  <https://docs.claude.com/en/docs/about-claude/models/overview>
- **Hugging Face — Tokenizers / `AutoTokenizer`.** The tool you use for the side-by-side tokenizer explorer. Skim the `transformers` tokenizer page and the `tokenizers` summary:
  <https://huggingface.co/docs/transformers/main_classes/tokenizer> · <https://huggingface.co/docs/tokenizers/index>
- **`outlines` documentation.** The grammar-constrained decoding library you use in `exercise-03` and the mini-project. Read the "JSON schema" and "regex" pages:
  <https://dottxt-ai.github.io/outlines/>

## Tokenization (read for the algorithm, not the math)

You implement BPE *encoding* this week (the stretch goal), so read these for the mechanism — how merges are applied — not the training proofs.

- **Neural Machine Translation of Rare Words with Subword Units** (Sennrich et al., 2016) — the paper that brought BPE to NLP. Read §3 (the algorithm). It is short:
  <https://arxiv.org/abs/1508.07909>
- **SentencePiece: A simple and language-independent subword tokenizer** (Kudo & Richardson, 2018) — the `▁`-space marker and the unigram option behind Llama-style tokenizers:
  <https://arxiv.org/abs/1808.06226>
- **Subword Regularization (Unigram LM)** (Kudo, 2018) — the probabilistic-segmentation alternative to greedy BPE:
  <https://arxiv.org/abs/1804.10959>
- **`tiktoken`** (OpenAI's BPE tokenizer). Read it to understand byte-level BPE — and remember: it is the *wrong* ruler for any non-OpenAI model, which is the cost-estimation trap of the week:
  <https://github.com/openai/tiktoken>

## Context & long-context behavior

- **Lost in the Middle: How Language Models Use Long Contexts** (Liu et al., 2023). The U-shaped position effect. Read the figures; they are the whole lesson on *where* a token sits:
  <https://arxiv.org/abs/2307.03172>
- **Efficient Memory Management for LLM Serving with PagedAttention** (Kwon et al., 2023 — vLLM). §2 again, now for the KV cache / context-cost callback:
  <https://arxiv.org/abs/2309.06180>
- **Anthropic — Prompt caching.** The vendor feature that persists the prefill KV cache for a shared prefix; the biggest cost lever for a fixed-system-prompt pipeline:
  <https://docs.claude.com/en/docs/build-with-claude/prompt-caching>

## Sampling (the distribution-transform papers)

- **The Curious Case of Neural Text Degeneration** (Holtzman et al., 2020) — introduces nucleus (top-p) sampling and the case against maximum-probability decoding (i.e. against beam search for open text). The single most useful sampling paper:
  <https://arxiv.org/abs/1904.09751>
- **Turning Up the Heat: Min-p Sampling** (Nguyen et al., 2024) — min-p as confidence-adaptive truncation; the newest of the common knobs:
  <https://arxiv.org/abs/2407.01082>

## Structured output & grammar-constrained decoding (the 2026 toolset)

- **Efficient Guided Generation for Large Language Models** (Willard & Louf, 2023 — the `outlines` paper). FSM-constrained decoding: compile a regex/schema to a state machine and mask the logits. The mechanism behind "provably valid":
  <https://arxiv.org/abs/2307.09702>
- **`outlines`** (GitHub + docs) — JSON-schema- and regex-constrained generation against local models:
  <https://github.com/dottxt-ai/outlines>
- **`guidance`** — templating-style constrained generation (interleave fixed text, constrained regions, free generation):
  <https://github.com/guidance-ai/guidance>
- **`xgrammar`** — a fast grammar engine focused on low per-token overhead, integrated into serving stacks. Read when you care about the *cost* of constraint:
  <https://github.com/mlc-ai/xgrammar> · paper: <https://arxiv.org/abs/2411.15100>
- **`jsonschema`** (Python) — validate the constrained output to *prove* the schema holds, not just assume it:
  <https://python-jsonschema.readthedocs.io/>

## Talks worth your time (free, no signup)

- **"Let's build the GPT Tokenizer"** (Andrej Karpathy, ~2h). Builds byte-level BPE from scratch on screen. This is the required intuition for Lecture 1; watch it Monday night:
  <https://www.youtube.com/watch?v=zduSFxRajkE>
- **"Intro to Large Language Models"** (Karpathy, ~1h). Re-watch the sampling segment with this week's eyes:
  <https://www.youtube.com/watch?v=zjkBMFhNj_g>

## Interactive tools (poke at tokenizers in the browser)

- **Tiktokenizer** — paste text, see the tokens for several tokenizers side by side. A fast way to build the "code and CJK tokenize weirdly" intuition before you do `exercise-01`:
  <https://tiktokenizer.vercel.app/>
- **Hugging Face tokenizer playground** — inspect any open model's tokenizer on the Hub. Useful for spot-checking the worst-case-disagreement stretch goal:
  <https://huggingface.co/spaces>

## Tools you'll use this week

- **`transformers` + `tokenizers`** — `pip install transformers tokenizers`. `AutoTokenizer.from_pretrained(...)` for exact open-model token counts.
- **`numpy`** — `pip install numpy`. The sampler exercise is ~40 lines of it; `np.exp`, `np.argsort`, `np.cumsum`, `np.partition`, boolean masking.
- **`outlines`** — `pip install outlines`. Grammar-constrained decoding against a tiny local model (`Qwen/Qwen2.5-0.5B-Instruct`).
- **`jsonschema`** — `pip install jsonschema`. Prove the constrained output is schema-valid across the fuzz set.
- **`anthropic`** — `pip install anthropic`. `client.messages.count_tokens(...)` for the hosted-tokenizer comparisons.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Token** | The sub-word integer the model reads/writes. The unit you pay for. |
| **BPE** | Byte-Pair Encoding: start from bytes, greedily apply the highest-priority adjacent merge until none apply. |
| **Merge list** | The ordered table of token pairs to combine. Its *order* is the tokenizer's algorithm. |
| **SentencePiece** | A tokenizer library/format (BPE or unigram) that encodes spaces explicitly (`▁`) and is language-agnostic. |
| **Unigram** | A tokenizer model that picks the segmentation maximizing total token probability (not greedy merges). |
| **`count_tokens`** | Anthropic's endpoint returning the *real* input token count from the model's own tokenizer. |
| **Logits** | The raw unnormalized next-token scores the forward pass emits — one per vocab entry. |
| **Softmax** | `exp(zᵢ)/Σexp(zⱼ)` — turns logits into a probability distribution; the exponential amplifies the lead. |
| **Temperature** | Scales logits before softmax: `<1` sharpens, `>1` flattens, `→0` is greedy. Not "creativity." |
| **Top-k** | Keep the `k` highest-logit tokens; fixed count. |
| **Top-p (nucleus)** | Keep the smallest set whose cumulative probability ≥ `p`; adaptive to mass. Not "diversity." |
| **Min-p** | Keep tokens with prob ≥ `min_p × p_max`; adaptive to the model's confidence at this position. |
| **Repetition penalty** | Down-weights already-seen tokens to fight loops. |
| **Grammar-constrained decoding** | Mask out (set to `−∞`) every token that would violate the grammar, each decode step. Makes invalid output *unreachable*. |
| **Beam search** | Keep the top-`b` partial sequences by cumulative probability. Great for translation, bland for open text — unused for LLM generation. |
| **Lost in the middle** | Models attend unevenly across long context; facts in the middle are used less reliably than at the edges (U-shape). |

---

*If a link 404s, please open an issue so we can replace it.*
