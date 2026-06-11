# Week 2 — Tokens, Context, and Sampling

Last week you learned that an LLM is a function from token IDs to next-token logits, wrapped in five stages. This week you zoom into the first and fourth stages — **tokenization** and **sampling** — because that is where two of the most expensive and most misunderstood behaviors in the whole stack live: what you pay (tokens) and what you get (the sampler's choice). By Friday you will be able to look at any piece of text and estimate its token cost to within a few percent for a given model, instrument a running model for per-request token accounting, and write a decoder that is *guaranteed* to emit schema-valid JSON because you constrained the sampler, not because you asked nicely.

The one sentence to carry out of this week:

> **Temperature is not creativity and top-p is not diversity. Both are knobs on a probability distribution over tokens — and the most reliable way to get structured output from a model is not a better prompt, it is constraining which tokens the sampler is even allowed to choose.**

This is the week where "the model gave me broken JSON again" stops being a prompt-engineering problem and becomes a *decoding* problem with a real fix. It is also the week where your cost estimates stop being guesses. A senior AI engineer can glance at a 12-page document and a model name and tell you, roughly, what it costs to summarize — because they understand tokenization at the level we build it here.

We assume you finished Week 1: you have a uniform `complete()` client over Anthropic + Ollama, you can read prefill/decode latency, and you internalized that sampling (Stage 4) — not the weights — owns determinism and randomness. We build directly on that. The `llmpick` harness you wrote becomes the thing you instrument this week.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** how sub-word tokenization works — BPE and SentencePiece at the algorithm level — and why the same text is a different number of tokens across the Llama, Qwen, and tiktoken tokenizers.
- **Estimate** the token cost of a piece of text for a given model to within a few percent, using the model's *own* tokenizer, and explain why estimating with the wrong tokenizer (e.g. tiktoken for a non-OpenAI model) is a real and costly error.
- **Reason** about the context window as a budget you spend: the linear token cost, the super-linear attention cost of long context, and the "lost in the middle" position effect that makes *where* a token sits matter.
- **Instrument** a running model session for per-request token accounting — prompt tokens, output tokens, and timing — and roll those up into a cost report you can put in front of a budget owner.
- **Define** the sampling parameters precisely — temperature, top-p (nucleus), top-k, min-p, repetition penalty — as transformations on the logit distribution, and predict the qualitative effect of each on output.
- **Implement** a sampler from raw logits in NumPy — softmax with temperature, then top-k / top-p / min-p truncation — so you understand exactly what the knobs do rather than treating them as magic.
- **Produce** structured, schema-valid output reliably using grammar-constrained decoding with `outlines` (and know where `guidance` and `xgrammar` fit), and assert the schema holds — turning "usually valid JSON" into "provably valid JSON."
- **State** why beam search exists, why it was once standard, and why almost nobody uses it for open-ended LLM generation in 2026.

## Prerequisites

This week assumes you completed **Week 1** (or have equivalent fluency) and, specifically:

- You have the **Week 1 uniform client** (`complete()` over Anthropic + Ollama) working, or can rebuild it in an hour. Several exercises instrument it.
- **Python 3.12** with a venv. New this week: `pip install transformers tokenizers numpy outlines`. (`outlines` pulls a local model backend; the grammar-constrained lab runs against a small local model so you don't pay per token to iterate.)
- **Ollama** running with a small model pulled (`qwen2.5:7b` or `llama3.2:3b`). The token-accounting and grammar-constrained labs run locally.
- An **Anthropic API key** (`ANTHROPIC_API_KEY`) for the hosted-tokenizer comparisons. Every lab has a local-only fallback if you don't have one.
- Comfort with **NumPy at the "I can do `np.exp`, `np.argsort`, and boolean masking" level** — the sampler exercise is ~40 lines of NumPy and we re-explain the array ops inline.

You do **not** need a GPU. A 16 GB laptop runs every lab; the local models run on CPU/Metal. You do **not** need to understand the BPE *training* algorithm (how the merges are learned) — only how a trained tokenizer *applies* merges to encode text, which we build from scratch.

## Topics covered

- **Tokenization mechanics.** Byte-pair encoding (BPE): start from bytes/characters, greedily merge the most frequent adjacent pair, repeat — and at inference, apply the learned merges to turn text into IDs. SentencePiece and the unigram alternative. The Llama tokenizer, the Qwen tokenizer, `tiktoken`, and why their vocabularies and merge rules differ.
- **Token-accurate cost estimation.** Counting tokens with the model's own tokenizer (`count_tokens` for hosted, `AutoTokenizer` / Ollama counts for local); why character-counts and word-counts are wrong; the systematic gap between tokenizers on code, whitespace, and non-English text.
- **Context windows and the price of long context.** The linear token cost; the super-linear attention cost (Week 1's `O(n²)`) that makes a long prompt cost more *per token*; the "lost in the middle" effect; budgeting the window like a cache you can't afford to fill carelessly.
- **The KV cache and why streaming feels fast** (a Week-1 callback, now made precise): why the first token is slow (prefill) and subsequent tokens stream (decode reusing the cache), and how that shapes the latency you measure.
- **Sampling parameters as distribution transforms.** Temperature (scales logits before softmax — flattens or sharpens), top-k (keep the k highest-probability tokens), top-p / nucleus (keep the smallest set whose cumulative probability ≥ p), min-p (keep tokens above a fraction of the top token's probability), repetition penalty (down-weight already-seen tokens). What each does to the distribution and to the output.
- **Writing a sampler from logits.** Softmax-with-temperature, then truncation (top-k / top-p / min-p), then a draw — in NumPy, so the knobs stop being magic.
- **Structured outputs and grammar-constrained decoding.** JSON-mode vs grammar-constrained decoding; `outlines` (regex/JSON-schema-constrained generation), `guidance`, and `xgrammar` as the 2026 toolset; the key idea — at each decode step, *mask out* tokens that would violate the grammar, so invalid output is structurally impossible.
- **Beam search and why nobody uses it.** What beam search does (keep the top-`b` partial sequences by cumulative probability), why it was standard for machine translation, and why it produces bland, repetitive text for open-ended generation — and so lost to sampling.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                       | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | BPE & SentencePiece; tokenizer differences; cost estimation|    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Context as a budget; the tokenizer explorer exercise       |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Sampling as distribution transforms; the NumPy sampler     |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Grammar-constrained decoding; structured-output engineering|    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | Token-accounting instrumentation; mini-project; studio     |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                      |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, report polish                                |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                            | **6h**   | **7h**    | **3h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | Tokenizer docs, the sampling references, the `outlines`/`guidance`/`xgrammar` docs, and the talks |
| [lecture-notes/01-tokenization-and-cost.md](./lecture-notes/01-tokenization-and-cost.md) | BPE/SentencePiece, tokenizer differences, token-accurate cost, context as a budget |
| [lecture-notes/02-sampling-and-structured-output.md](./lecture-notes/02-sampling-and-structured-output.md) | Sampling parameters as distribution transforms, the NumPy sampler, grammar-constrained decoding, beam search |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-tokenizer-explorer.md](./exercises/exercise-01-tokenizer-explorer.md) | A side-by-side tokenization explorer across three open tokenizers; explain the differences |
| [exercises/exercise-02-sampler-from-logits.py](./exercises/exercise-02-sampler-from-logits.py) | Implement temperature / top-k / top-p / min-p sampling from raw logits in NumPy and verify each knob's effect |
| [exercises/exercise-03-constrained-json.py](./exercises/exercise-03-constrained-json.py) | Grammar-constrained decoding with `outlines`; assert the output is schema-valid 100% of the time |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-token-budget-audit.md](./challenges/challenge-01-token-budget-audit.md) | Audit a real document-processing pipeline's token budget and cut its cost without losing quality |
| [quiz.md](./quiz.md) | 14 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the token-budget cost report |
| [mini-project/README.md](./mini-project/README.md) | `toklab` — a tokenization explorer + per-request token-accounting instrument + a schema-constrained JSON generator |

## The "provably valid, not probably valid" promise

C23's recurring marker for this week is the assertion that *holds*. By Thursday your structured-output code looks like this:

```python
import json
from outlines import models, generate

model = models.transformers("Qwen/Qwen2.5-0.5B-Instruct")  # tiny, runs on CPU
generator = generate.json(model, schema)   # the schema CONSTRAINS the sampler

for prompt in fuzz_prompts:                # 100 adversarial prompts
    out = generator(prompt)
    json.loads(out)                        # never raises — it CANNOT be invalid
    jsonschema.validate(out, schema)       # always passes — by construction
```

If your JSON generator "usually" produces valid output and you handle the occasional break with a retry, you are not done. Retry-on-broken-JSON is the symptom of asking the model nicely instead of constraining the sampler. The point of Week 2 is to make schema-valid output a *structural guarantee*, not a probability you babysit with try/except.

## Stretch goals

If you finish the regular work early and want to push further:

- **Implement BPE encoding from scratch.** Given a tiny learned merge list, write the encoder that turns a string into token IDs by greedily applying merges. ~30 lines. You will never again be confused about what a tokenizer "is."
- **Find the worst-case tokenizer disagreement.** Search for a text snippet where two tokenizers' counts differ by the largest ratio you can find (hint: code with lots of whitespace, or a non-Latin script, or emoji). Report the ratio. This is the cost-estimation error in its most dramatic form.
- **Temperature sweep on a real distribution.** Pull the logits from a local model for one position (via the transformers API), then run your NumPy sampler at temperature 0.1, 0.7, 1.0, and 1.5 and plot how the chosen-token distribution spreads. Seeing the distribution flatten is worth a thousand words.
- **Compare `outlines` vs `xgrammar` latency.** Generate the same constrained JSON with both and measure the per-token overhead each adds. Grammar-constrained decoding is not free; knowing the cost is part of using it well.

## Up next

Week 3 takes the precision you built here — exact token counts, controlled sampling, schema-valid output — and applies it to **prompt engineering as engineering**: versioning prompts, diffing them, and writing regression tests so a "better" prompt is a measured claim, not a vibe. The token-accounting instrument you build this week becomes part of how you measure a prompt's cost there. Push your `toklab` repo before you start it.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
