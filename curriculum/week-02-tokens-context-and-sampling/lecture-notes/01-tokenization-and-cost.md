# Lecture 1 — Tokenization and Cost: What You Pay For, and Why Your Estimate Is Wrong

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can describe BPE and SentencePiece at the algorithm level (how a *trained* tokenizer applies merges to encode text), explain why the same string is a different number of tokens on the Llama, Qwen, and `tiktoken` tokenizers, count tokens for a hosted model with its *own* tokenizer to within a percent, and reason about the context window as a budget you spend — with a linear token cost, a super-linear attention cost, and a position effect ("lost in the middle") that makes *where* a token sits matter.

If you remember one sentence from this lecture, remember this one:

> **The token is the unit you pay for, and the only correct token count comes from the model's own tokenizer. Estimating cost with a character count, a word count, or — worst of all — a different model's tokenizer is not "close enough"; it is a systematic error that compounds across every request you ever make.**

Last week you learned the model is a function from token IDs to logits, wrapped in five stages. Stage 1 (the tokenizer) decides how many integers your text becomes, and Stage 2 (the context window) is the bounded buffer those integers live in. This lecture lives entirely inside those two stages, because that is where the money is. A senior AI engineer can look at a 12-page PDF and a model name and tell you, to within a few percent, what it costs to summarize. By the end of this lecture, so can you — not by guessing, but because you understand the tokenizer well enough to build one.

---

## 1. The tokenizer is a learned compression scheme

Start from the contract. The tokenizer is a deterministic, reversible map:

```
encode : str        ->  List[int]   (text  -> token IDs)
decode : List[int]  ->  str         (token IDs -> text)
```

It is **not** a word splitter (`"unbelievable"` is not one token in most vocabularies) and **not** a character splitter (`" the"` is one token, not four). It sits in between, on **sub-word** units. The whole point of sub-word tokenization is a trade-off:

- **Character-level** tokenizers have a tiny vocabulary (~256 byte values) but make every sequence very long — `"hello"` is five tokens, and the model has to learn that `h-e-l-l-o` means a greeting. Long sequences are expensive (attention is `O(n²)`) and the model wastes capacity re-learning spelling.
- **Word-level** tokenizers make sequences short but need an enormous vocabulary and still choke on any word they never saw (every typo, every proper noun, every `kubectl` is an out-of-vocabulary token).
- **Sub-word** tokenizers split the difference: common words are a single token, rare words decompose into a few sub-word pieces, and *nothing* is out-of-vocabulary because the fallback is bytes. `"tokenizer"` might be one token; `"antidisestablishmentarianism"` is several; a never-before-seen string of emoji is still encodable, byte by byte.

This is, at heart, a **compression** problem: assign short codes (single tokens) to frequent strings, longer codes (multiple tokens) to rare ones. That framing is exactly right and it tells you where the cost surprises come from: text that looked frequent to the *training* corpus is cheap; text that didn't is expensive. English prose is cheap. Dense code, unusual whitespace, and non-Latin scripts are expensive — because the tokenizer's merge rules were learned on a corpus where those were comparatively rare.

> **The grain of the lens.** Last week we said the tokenizer is the lens and the lens has a grain — which is why a model "can't spell strawberry backwards." Same grain, money version: the lens also decides how many tokens (= how many dollars) a given string costs. The grain is uneven, and the unevenness is the whole subject of §3.

---

## 2. Byte-Pair Encoding (BPE), at the algorithm level

We do not need the *training* algorithm in depth (how the merges are learned), but you must understand how a *trained* tokenizer **applies** its merges to encode text — because that is the encoder, and once you can run it by hand you will never again be confused about what a tokenizer "is."

### 2.1 The two artifacts a trained BPE tokenizer ships

A trained BPE tokenizer is just two tables:

1. **A vocabulary** — a map from token string to integer ID. Size `V` ranges from ~32k (Llama 2-era) to ~128k–256k (Llama 4, Qwen 3, frontier models). Bigger vocab → fewer tokens per text (better compression) but a bigger embedding/unembedding matrix.
2. **An ordered list of merges** — pairs of tokens that should be combined, *in priority order*. Merge `0` is the highest-priority merge the training saw most; merge `49999` is a rare one. The order is the entire algorithm.

That's it. Encoding is "apply the merges in order until none apply."

### 2.2 The encode algorithm (this is the whole thing)

To encode a string with BPE:

1. **Pre-tokenize** into chunks (usually on whitespace/punctuation boundaries, via a regex). Each chunk is encoded independently — this is why a leading space matters (`"the"` vs `" the"` are different tokens; the space is part of the chunk).
2. **Start from bytes/characters.** Split the chunk into its smallest units (single bytes in byte-level BPE — which is why nothing is ever out-of-vocabulary).
3. **Repeatedly apply the highest-priority merge that is currently possible.** Look at every adjacent pair in the current sequence; find the pair whose merge has the *lowest rank* (highest priority) in the merge list; merge it. Repeat.
4. **Stop when no adjacent pair appears in the merge list.** Map the resulting token strings to IDs via the vocab.

Here is the encoder, in ~25 lines of real Python. This is the stretch goal from the week README, done for you so you can read it; doing it from scratch on a tiny merge list is the exercise of understanding:

```python
def bpe_encode(text: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    """Greedy BPE: repeatedly merge the highest-priority adjacent pair.

    `ranks` maps a pair like ("t","h") to its merge priority (lower = merge first).
    Returns the list of token strings (map to IDs via the vocab separately).
    """
    tokens = list(text)  # start from characters (bytes in real byte-level BPE)
    while True:
        # Find the adjacent pair with the BEST (lowest) merge rank.
        best_pair = None
        best_rank = None
        for a, b in zip(tokens, tokens[1:]):
            rank = ranks.get((a, b))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank, best_pair = rank, (a, b)
        if best_pair is None:
            break  # no adjacent pair is mergeable -> done
        # Merge every occurrence of best_pair, left to right.
        a, b = best_pair
        merged, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        tokens = merged
    return tokens
```

Trace it on `"lower"` with a merge list that has `("l","o") -> 0`, `("lo","w") -> 1`, `("e","r") -> 2`:

```
['l','o','w','e','r']  -> apply ("l","o")  -> ['lo','w','e','r']
['lo','w','e','r']      -> apply ("lo","w") -> ['low','e','r']
['low','e','r']         -> apply ("e","r")  -> ['low','er']
no more merges -> ['low', 'er']  -> two tokens
```

Two facts fall out of this that matter for the rest of the course:

- **Encoding is greedy and order-dependent.** The merge list's *order* is the model. Swap two merges and you get a different tokenization of the same string. This is why you cannot mix a Llama merge list with a Qwen vocab and expect anything sensible.
- **The cost of a string is "how few merges compress it."** A string the training corpus saw a lot compresses to few tokens (cheap). A string it rarely saw stays near its byte length (expensive). The next section is just this fact, made concrete across real tokenizers.

### 2.3 Byte-level BPE and why nothing is out-of-vocabulary

Real BPE (GPT-2 onward, and most modern tokenizers) operates on **bytes**, not Unicode characters. Every string is first encoded to UTF-8 bytes; the base vocabulary is the 256 byte values; merges combine byte sequences. The payoff: *any* string is encodable, because the worst case is "fall all the way back to raw bytes." An emoji that the merges never combined is just its 3–4 UTF-8 bytes, i.e. 3–4 tokens. A CJK character is its bytes. This is the mechanism behind the cost surprises in §3 — rare scripts get no merge help and pay near their byte length.

---

## 3. SentencePiece, the unigram alternative, and why tokenizers disagree

BPE is one family. The other you will meet constantly is **SentencePiece**, which is a *library/format* (used by Llama, T5, many multilingual models) that can run either a BPE model or a **unigram** model.

### 3.1 SentencePiece's two ideas

- **It treats the input as a raw stream and encodes spaces explicitly.** SentencePiece replaces spaces with a visible marker (`▁`, "lower one-eighth block") and tokenizes the whole thing, so detokenization is exactly reversible with no language-specific whitespace rules. This is why Llama tokens often look like `▁the`, `▁context`. It makes the tokenizer language-agnostic — no assumption that words are space-separated, which matters for Japanese, Chinese, Thai.
- **The unigram model picks a segmentation probabilistically, not greedily.** Instead of "apply merges in order," a unigram tokenizer has a vocabulary where each token carries a probability, and encoding chooses the segmentation that *maximizes the total probability* (via a Viterbi-style search). Different mechanism, same goal: turn text into a short-ish sequence of sub-word IDs.

For systems purposes you do not need to implement unigram. You need to know it exists, that it is what many SentencePiece models use, and that it produces **different** tokenizations than BPE for the same text — which is the next point.

### 3.2 Why Llama, Qwen, and `tiktoken` give different counts

Three tokenizers, three different maps, because they differ on every axis that matters:

| Axis | Llama (SentencePiece/BPE) | Qwen (BPE) | `tiktoken` (OpenAI BPE) |
|---|---|---|---|
| Vocabulary size | ~128k (Llama 3/4) | ~150k | ~100k (`cl100k`) / ~200k (`o200k`) |
| Training corpus | Meta's mix | Alibaba's mix (heavy CJK) | OpenAI's mix |
| Whitespace handling | `▁` marker | byte-level | byte-level |
| Merge rules | its own | its own | its own |

Because the **merge rules and vocabulary are learned on different corpora**, the same string compresses differently:

- **English prose:** the three tend to land within ~10% of each other — everyone trained heavily on English, so the merges are similar.
- **Code:** counts diverge more. Whitespace runs (indentation), `snake_case`, and symbols like `::`, `=>`, `</>` are tokenized very differently depending on whether the corpus had a lot of that exact code.
- **Non-English / CJK:** counts diverge *a lot*. Qwen, trained heavily on Chinese, tokenizes Chinese far more efficiently (fewer tokens per character) than a tokenizer that saw little of it. The same Chinese paragraph can be 1.5–3× more tokens on the wrong tokenizer.

This is not a curiosity. It is the **cost-estimation error in its most dangerous form**:

> **If you estimate a Llama or Anthropic model's token cost using `tiktoken` (OpenAI's tokenizer), you will be wrong — often by 10–30% on code and by *much* more on non-English text — and you will be wrong in the same direction on every single request, so the error does not average out. It compounds. A 20% under-count on a pipeline doing 10M tokens/day is a 20% under-budget that shows up as a surprise invoice.**

The exercise this week (`exercise-01`) makes you see this with your own eyes across three open tokenizers. The lesson lands harder when the numbers are yours.

---

## 4. Counting tokens correctly, for hosted and local models

The rule is simple and absolute: **count with the model's own tokenizer.** The mechanism differs by deployment.

### 4.1 Hosted (Anthropic): use the counting endpoint

You do not have Anthropic's tokenizer as a local file, and you must never approximate it with `tiktoken`. The SDK gives you the real count from the real tokenizer:

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

resp = client.messages.count_tokens(
    model="claude-opus-4-8",
    messages=[{"role": "user", "content": "Summarize the Q3 incident report."}],
)
print(resp.input_tokens)  # the REAL number, from Anthropic's own tokenizer
```

`count_tokens` is the same tokenizer the billing path uses, so it is ground truth for the *input* side. (Output tokens you only know after generation, from `msg.usage.output_tokens`.) Note: `count_tokens` counts the input including the chat-template overhead — roles, formatting — not just your raw string, which is exactly what you want, because that overhead is real tokens you pay for.

> **Never `tiktoken` for a non-OpenAI model.** This is the single most common cost-estimation bug in the wild. `tiktoken` is OpenAI's tokenizer; using it to estimate Anthropic, Llama, or Qwen tokens is using the wrong ruler. It is wrong on English and badly wrong on code and non-English text. The correct ruler is the model's own tokenizer — `count_tokens` for Anthropic, `AutoTokenizer` for an open model, `prompt_eval_count` for Ollama.

### 4.2 Local open-weights: `AutoTokenizer` (exact) or Ollama counts (operational)

For an open model you have the tokenizer itself. Hugging Face's `AutoTokenizer` loads the exact tokenizer files shipped with the model:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
ids = tok.encode("def add(a, b):\n    return a + b\n")
print(len(ids), ids[:8])   # exact token count + the first few IDs
print(tok.convert_ids_to_tokens(ids[:8]))  # see the sub-word pieces
```

This is the *exact* tokenizer the model uses, so `len(tok.encode(text))` is the true token count for that model — perfect for the explorer exercise. Operationally, when you are actually *running* the model through Ollama, you read the count off the response instead (it is the same tokenizer, reported after the fact):

```python
import httpx

d = httpx.post("http://localhost:11434/api/generate",
               json={"model": "qwen2.5:7b", "prompt": "...", "stream": False},
               timeout=120.0).json()
print(d["prompt_eval_count"], d["eval_count"])  # tokens_in, tokens_out
```

`AutoTokenizer` answers "how many tokens *will* this be" before you spend anything; Ollama's counts answer "how many tokens *was* it" after a run. Use the first to budget, the second to reconcile.

### 4.3 The cost formula, made precise

Pricing is per **million tokens** (MTok), split into input and output rates. The cost of one request is:

```
cost = tokens_in  * price_in  / 1_000_000
     + tokens_out * price_out / 1_000_000
```

Two non-obvious facts a senior engineer holds:

- **Output is usually 3–5× the price of input.** Decode is the expensive phase (Lecture 1, Week 1: bandwidth-bound, serial), and the pricing reflects it. So a job that reads a lot and writes a little (classification, extraction) is cheap; a job that writes a lot (long generation, verbose reasoning) is where the bill lives. This flips the intuition that "long documents are expensive" — a long *input* is linear and cheap-ish; a long *output* is what hurts.
- **You pay for the chat-template overhead and the system prompt on every call.** A 2,000-token system prompt re-sent on every turn of a 50-turn conversation is 100,000 input tokens you paid for the system prompt alone. This is why prompt caching (a TTFT *and* cost optimization, Week 1 §4) exists, and why Week 3's prompt discipline is a cost lever, not just a quality one.

Worked example. Summarize an 1,800-word incident report (~2,400 input tokens by `count_tokens`) into a 120-token summary on `claude-haiku-4-5` (input $1.00/MTok, output $5.00/MTok):

```
cost = 2400 * 1.00 / 1_000_000  +  120 * 5.00 / 1_000_000
     = 0.0024                    +  0.0006
     = $0.0030 per call
```

At 50,000 reports/day that is **$150/day, ~$4,500/month** — a number a budget owner cares about, derived from a real token count and a real price table. That calculation, done before you ship, is the difference between an engineer and someone who finds out from the invoice.

---

## 5. The context window is a budget, not a bucket

Stage 2 is the bounded buffer: the maximum number of tokens (prompt + generated) the model conditions on at once. In 2026 this ranges from ~8k on small local models to **1M tokens** on frontier models like `claude-opus-4-8`. The beginner mistake is to treat the window as a bucket you fill until it's full and free until then. It is a **budget** you spend, and it costs you in three distinct currencies.

### 5.1 The linear cost: tokens are money

The obvious one. Every token in the window is a token you pay for (input rate). Stuffing 200k tokens of "just in case" context into a window to answer a question that needed 2k of it is paying 100× for the same answer. "It fits in the window" and "it is cheap" are different claims — a 1M-token window is an invitation to overspend, not a license to.

### 5.2 The super-linear cost: attention is `O(n²)`

From Week 1 §3: attention work grows with the *square* of sequence length. Doubling the prompt roughly quadruples the attention compute. Two consequences:

- **A long prompt costs more *per token* to process**, not just more tokens. The first-token latency (prefill, Week 1 §4) balloons with prompt length — you measured this in `exercise-03` last week. A 100k-token prompt has a laggy first token even on fast hardware.
- **This is the real reason retrieval exists.** Weeks 8–11 (Phase II) are, in large part, the discipline of putting the *right* 4k tokens in the window instead of *all* 400k. Not because the big window doesn't fit it — because filling it is slow and expensive and, as the next point shows, often *worse* for quality.

### 5.3 The quality cost: lost in the middle

Here is the one that surprises people. Models do **not** attend uniformly across a long context. The well-documented **"lost in the middle"** effect: when a relevant fact is placed in the *middle* of a long context, models retrieve and use it less reliably than when the same fact is at the *beginning* or the *end*. Accuracy as a function of the fact's position looks like a **U-shape** — strong at the edges, sagging in the middle.

```
accuracy
  high |■                                   ■
       | ■                                 ■
       |   ■                             ■
       |      ■                       ■
   low |          ■    ■    ■    ■   (the "middle" sag)
       +----------------------------------------
        start          middle          end   (position of the relevant fact)
```

The systems consequences are concrete and you will act on them all course:

- **Position is a design variable.** *Where* you place the retrieved chunk, the instruction, and the question in the prompt changes the answer quality, independent of the content. Put the most important context at the **start or the end**, not buried in the middle of a 50k-token dump.
- **More context is not more better.** Padding the window with marginally-relevant chunks can *lower* quality by pushing the relevant one into the sagging middle and by diluting attention. This is the empirical backbone of "retrieve precisely, don't dump" (Week 11).
- **"It's in the context, why didn't it use it?" is a position bug, not a model failure.** When a fact is present but ignored, check where it sits before you blame the model. This is a new row for the Week 1 diagnosis table.

### 5.4 The window as a cache you can't afford to fill carelessly

Put the three costs together and the mental model is: the context window is a **cache with a price per slot and a non-uniform hit rate**. You budget it — spend tokens where they buy the most answer quality, leave the rest empty, and place the high-value tokens at positions the model actually attends to. That sentence is the entire job of Phase II, previewed. This week you just internalize that the window is spent, not filled.

---

## 6. KV cache callback: why the *first* token of a long prompt is the expensive one

A precise callback to Week 1 §4, now with the cost lens. When you send a long prompt:

- **Prefill** processes all `n` prompt tokens at once, computing and storing the attention keys/values for every token and every layer into the **KV cache**. This is compute-bound and its duration *is* your time-to-first-token. A long prompt = a long prefill = a slow first token, and it scales with prompt length.
- **Decode** then reuses that cache: each new token attends to the cached keys/values and only its own key/value is computed. Bandwidth-bound, per-token, roughly prompt-length-independent — it sets your streaming rate (TPOT).

The cost lens this lecture adds:

- **Prompt caching is a budget tool, not just a latency tool.** When a vendor lets you cache a long shared prefix (a big system prompt, a fixed document), they persist its *prefill KV cache* across requests. Reuse it and you skip re-prefilling — lower TTFT *and* a discounted token rate on the cached prefix. For a pipeline that re-sends the same 5k-token instruction block on every one of a million calls, prompt caching is the single biggest cost lever you have. Week 21 measures it; recognize now that it is the same KV cache from Week 1, persisted.
- **The KV cache grows with sequence length and competes for memory.** On a local GPU it fights the weights for VRAM; this is why a long context is expensive in *memory*, not just compute. PagedAttention (Week 19) exists to manage exactly this. For now: long context spends three budgets — money, latency, and memory.

> **The thread tying §4–6 together:** tokens cost money (linear), attention costs compute (quadratic, paid as TTFT), and the KV cache costs memory (linear in length, but it competes with the weights). "Long context" spends all three. A budget you can't afford to fill carelessly.

---

## 7. Recap

You should now be able to:

- Describe the tokenizer as a learned compression scheme — sub-word units chosen to give frequent strings short codes — and state why character- and word-level tokenization both lose.
- Run BPE encoding by hand and in ~25 lines of Python: start from bytes/characters, greedily apply the highest-priority adjacent merge until none apply, map to IDs. Explain why encoding is greedy, order-dependent, and never out-of-vocabulary (byte fallback).
- Distinguish BPE from SentencePiece/unigram (explicit-space marker, probabilistic segmentation) and explain why Llama, Qwen, and `tiktoken` give different counts for the same string — different corpora → different merges → different compression, worst on code and non-English text.
- Count tokens correctly: `count_tokens` for Anthropic (never `tiktoken`), `AutoTokenizer` for an open model (exact, pre-spend), Ollama's `prompt_eval_count`/`eval_count` (operational, post-run); and compute cost with the per-MTok formula, knowing output is the expensive side.
- Reason about the context window as a budget with three currencies — linear token cost, super-linear (`O(n²)`) attention cost paid as TTFT, and the "lost in the middle" quality cost — and treat position as a design variable.

Next: the *other* end of the pipeline. The forward pass hands you logits; **how you turn logits into a chosen token** — temperature, top-k, top-p, min-p, repetition penalty — is sampling, and getting *guaranteed* structured output is constraining the sampler, not asking nicely. Continue to [Lecture 2 — Sampling and Structured Output](./02-sampling-and-structured-output.md).

---

## References

- *Neural Machine Translation of Rare Words with Subword Units* (Sennrich et al., 2016) — the BPE-for-NLP paper: <https://arxiv.org/abs/1508.07909>
- *SentencePiece: A simple and language independent subword tokenizer* (Kudo & Richardson, 2018): <https://arxiv.org/abs/1808.06226>
- *Subword Regularization / Unigram LM* (Kudo, 2018): <https://arxiv.org/abs/1804.10959>
- *Lost in the Middle: How Language Models Use Long Contexts* (Liu et al., 2023): <https://arxiv.org/abs/2307.03172>
- *Anthropic — Token counting* (count with the model's own tokenizer): <https://docs.claude.com/en/docs/build-with-claude/token-counting>
- *Anthropic — Pricing and models overview* (per-MTok rates, context windows): <https://docs.claude.com/en/docs/about-claude/models/overview>
- *Hugging Face `transformers.AutoTokenizer`*: <https://huggingface.co/docs/transformers/main_classes/tokenizer>
- *Let's build the GPT Tokenizer* (Karpathy) — builds BPE from scratch on video: <https://www.youtube.com/watch?v=zduSFxRajkE>
