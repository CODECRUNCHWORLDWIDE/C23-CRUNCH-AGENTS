# Exercise 1 — The Tokenizer Explorer

**Time estimate:** ~60 minutes. Guided.

## Goal

See, with your own eyes, that **the same string is a different number of tokens on different tokenizers** — and that the disagreement is small on English, larger on code, and dramatic on non-Latin scripts and emoji. By the end you will have a side-by-side counts table across three open tokenizers and a written explanation, grounded in the actual sub-word *pieces*, of why they disagree. This is the cost-estimation lesson of Lecture 1 made concrete: estimating with the wrong tokenizer is not "close enough."

## Why this matters

Lecture 1 claimed that BPE/SentencePiece tokenizers, trained on different corpora, compress the same text differently — and that using `tiktoken` (or any other model's tokenizer) to estimate a Llama or Anthropic cost is a systematic error. A table you built yourself, showing a Chinese paragraph at 2–3× the tokens on the wrong tokenizer, is worth more than any assertion. You will reach for this intuition every time you budget a job.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install transformers tokenizers
# (optional, for the hosted comparison at the end) pip install anthropic
```

The first `AutoTokenizer.from_pretrained(...)` for each model downloads only the tokenizer files (a few MB), **not** the weights — so this exercise is fast and runs on any machine, no GPU.

## The three tokenizers

Use these three open tokenizers (all loadable with `AutoTokenizer`):

| Label | `from_pretrained` argument | Family / notes |
|---|---|---|
| **Qwen** | `Qwen/Qwen2.5-7B-Instruct` | BPE, ~150k vocab, heavy CJK training |
| **Llama** | `meta-llama/Llama-3.2-3B-Instruct` | SentencePiece/BPE, ~128k vocab |
| **GPT (tiktoken-style)** | `Xenova/gpt-4o` | OpenAI `o200k`-style BPE, ~200k vocab |

> **Gated models.** `meta-llama/...` may require accepting Meta's license on the Hub and a `huggingface-cli login`. If you don't want to gate, substitute `mistralai/Mistral-7B-Instruct-v0.3` (Apache-2.0, ungated) for the Llama row. The lesson is identical; just label the row with whatever you actually loaded.

## The five probe strings

Tokenize **each** of these on **all three** tokenizers:

```python
PROBES = {
    "english":   "The quick brown fox jumps over the lazy dog near the riverbank.",
    "code":      "def merge(a, b):\n    return {**a, **b}  # shallow merge\n",
    "chinese":   "今天天气很好，我们一起去公园散步吧。",
    "emoji":     "Shipping it 🚀🚀🚀 — looks great 🎉 ✅",
    "whitespace":"x = 1\n\n\n        y = 2\n\t\treturn x + y",
}
```

Each probe is chosen to stress a different part of the tokenizer: plain English (everyone trained on it → small disagreement), code (symbols + `snake_case`), Chinese (CJK → big disagreement), emoji (multi-byte, often unmerged), and pathological whitespace (indentation runs).

## Step 1 — Load the tokenizers and count

```python
from transformers import AutoTokenizer

TOKENIZERS = {
    "Qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "Llama": "meta-llama/Llama-3.2-3B-Instruct",   # or mistralai/Mistral-7B-Instruct-v0.3
    "GPT":   "Xenova/gpt-4o",
}

PROBES = {
    "english":   "The quick brown fox jumps over the lazy dog near the riverbank.",
    "code":      "def merge(a, b):\n    return {**a, **b}  # shallow merge\n",
    "chinese":   "今天天气很好，我们一起去公园散步吧。",
    "emoji":     "Shipping it 🚀🚀🚀 — looks great 🎉 ✅",
    "whitespace":"x = 1\n\n\n        y = 2\n\t\treturn x + y",
}

tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in TOKENIZERS.items()}

# Count: len(tok.encode(text)) is the exact token count for that model.
# add_special_tokens=False so we compare the RAW text, not the chat-template wrapper.
counts = {}
for pname, text in PROBES.items():
    counts[pname] = {
        tname: len(tok.encode(text, add_special_tokens=False))
        for tname, tok in tokenizers.items()
    }
```

> **Why `add_special_tokens=False`.** Each tokenizer would otherwise prepend/append its own BOS/EOS markers, adding a constant that muddies the comparison. We want the count of the *content*. (In production you DO pay for the special tokens and the chat template — but here we isolate the encoding of the text itself.)

## Step 2 — Print the table and the disagreement ratio

```python
print(f"{'probe':<11} {'Qwen':>6} {'Llama':>6} {'GPT':>6}   ratio(max/min)")
print("-" * 48)
for pname, row in counts.items():
    vals = list(row.values())
    ratio = max(vals) / max(min(vals), 1)
    print(f"{pname:<11} {row['Qwen']:>6} {row['Llama']:>6} {row['GPT']:>6}   {ratio:>5.2f}x")
```

The `ratio` column is the headline: how many times more tokens the *worst* tokenizer needs than the *best* for that string. On English it will be near `1.0x`; on Chinese it will be the largest number in the table — that ratio *is* the cost-estimation error, in its most dramatic form.

## Step 3 — Look at the actual pieces (this is where it clicks)

Counts tell you *that* they disagree; the pieces tell you *why*. For the Chinese and code probes, print what each tokenizer actually produced:

```python
for pname in ("chinese", "code"):
    print(f"\n=== pieces for probe '{pname}' ===")
    for tname, tok in tokenizers.items():
        ids = tok.encode(PROBES[pname], add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        print(f"{tname:>6}: {pieces}")
```

Read the output. You will typically see:

- On **Chinese**, the CJK-heavy tokenizer (Qwen) emits roughly one token per character or fewer, while a tokenizer that saw little Chinese falls back toward *bytes* — multiple tokens per character — because the merges that would combine those bytes were never learned.
- On **code**, the tokenizers split `def`, `merge`, `{**a`, `# shallow` differently. Watch how indentation and the `**` operator are handled — whitespace runs are a classic source of disagreement.

## Step 4 (optional) — Add the hosted ground truth

If you have an API key, add Anthropic's real count for the English probe and notice it is *yet another* number — and the only correct one for that model:

```python
import anthropic, os
if os.environ.get("ANTHROPIC_API_KEY"):
    client = anthropic.Anthropic()
    r = client.messages.count_tokens(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": PROBES["english"]}],
    )
    print(f"\nAnthropic count_tokens (english, claude-opus-4-8): {r.input_tokens}")
    print("note: this includes the chat-template overhead, so it's a bit higher.")
```

## Deliverable

Write `notes/week-02/tokenizer-explorer.md` containing:

1. The **counts table** (5 probes × 3 tokenizers) and the **ratio** column.
2. The **pieces** for the Chinese and code probes (paste the printed lists).
3. A **3–5 sentence explanation** of *why* the Chinese ratio is the largest — tie it to "different training corpora → different merges → byte fallback for unseen scripts."
4. One sentence stating the **cost consequence**: if you budgeted a Chinese-heavy pipeline using the wrong tokenizer's count, by roughly what factor would you be off, and in which direction?

## Acceptance criteria

- [ ] `notes/week-02/tokenizer-explorer.md` exists with the counts table for all 5 probes × 3 tokenizers.
- [ ] The three counts **differ** for at least the code, Chinese, and emoji probes.
- [ ] The **Chinese probe has the largest max/min ratio** of the five (it should — that's the point).
- [ ] The pieces for the Chinese and code probes are pasted, and your explanation references **byte fallback** and **different training corpora**.
- [ ] You state the cost consequence with a rough factor and a direction (under- or over-count).
- [ ] Committed.

## Expected output shape

Your exact numbers depend on the tokenizer versions, but the **shape** must match: English ratio near 1.0, Chinese ratio the largest, code and whitespace in between.

```
probe         Qwen  Llama    GPT   ratio(max/min)
------------------------------------------------
english         13     14     13    1.08x
code            19     22     18    1.22x
chinese         16     38     34    2.38x      <- the largest ratio
emoji           11     17     14    1.55x
whitespace      13     21     12    1.75x

=== pieces for probe 'chinese' ===
  Qwen: ['今天', '天气', '很', '好', '，', '我们', ...]            <- whole words/chars
 Llama: ['今', '天', '天', '气', ...]  (more, finer pieces)        <- falls toward bytes
   GPT: ['今', '天', '天', '气', ...]
```

> **The lesson:** the same Chinese sentence is ~2.4× more tokens — and therefore ~2.4× the input cost — if you tokenize it with the wrong tokenizer. That error doesn't average out across requests; it's in the same direction every time. Estimate with the model's *own* tokenizer, always.

## Stretch

- **Find the worst-case disagreement** (the week's stretch goal). Search for a snippet that maximizes the max/min ratio — try deeply nested code with tabs, a rare script (Thai, Devanagari), or long emoji sequences with skin-tone modifiers. Report the ratio you found; it's the cost-estimation error in its most extreme form.
- **Quantify the whitespace tax.** Take a real 100-line Python file and count it on all three. The tokenizer that handles indentation runs best can be 20–40% cheaper on code — a real lever if your pipeline is code-heavy.
