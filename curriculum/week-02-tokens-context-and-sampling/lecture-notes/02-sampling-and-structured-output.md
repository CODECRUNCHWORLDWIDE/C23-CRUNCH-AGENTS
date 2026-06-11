# Lecture 2 — Sampling and Structured Output: Turning Logits Into Tokens, On Purpose

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can define temperature, top-k, top-p (nucleus), min-p, and repetition penalty *precisely* as transformations on the logit distribution and predict each one's effect; write a sampler from raw logits in NumPy (softmax-with-temperature, then truncation, then a draw); explain grammar-constrained decoding as token masking and why it makes schema-valid output a structural guarantee rather than a probability you babysit; and state why beam search exists and why almost nobody uses it for open-ended LLM generation in 2026.

If you remember one sentence from this lecture, remember the week's promise:

> **Temperature is not creativity and top-p is not diversity. Every sampling knob is a transformation on a probability distribution over the next token — and the most reliable way to get structured output is not a better prompt, it is constraining which tokens the sampler is even *allowed* to choose.**

Lecture 1 lived in Stage 1 (tokenizer) and Stage 2 (context). This lecture lives in **Stage 4 — sampling**, the stage you own. The forward pass (Stage 3) hands you a vector of `V` logits — one score per vocabulary entry — and refuses to pick. *You* pick. Everything you experience as the model being "creative," "random," "deterministic," "repetitive," or "structured" is a property of how your code, or your inference server's config, turns that logit vector into one chosen token ID. We are going to make every one of those knobs concrete by building the sampler.

---

## 1. From logits to a probability distribution: softmax

The forward pass gives you logits `z = [z₀, z₁, …, z_{V-1}]` — unnormalized real scores. They are not probabilities (they don't sum to 1, can be negative). To turn them into a distribution you apply the **softmax**:

```
p_i = exp(z_i) / Σ_j exp(z_j)
```

Each `p_i` is now in `[0, 1]` and they sum to 1. The exponential is the important part: it **amplifies differences**. A logit that is 2 larger than another becomes `e² ≈ 7.4×` more probable, not 2× more. This is why the model usually has a clear "favorite" next token even when several are plausible — softmax exaggerates the lead.

In NumPy, with the standard numerical-stability trick (subtract the max before exponentiating, so you never `exp()` a large positive number and overflow):

```python
import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)        # shift for numerical stability; result unchanged
    e = np.exp(z)
    return e / e.sum()
```

The shift is mathematically a no-op (`exp(z - c)/Σexp(z - c) = exp(z)/Σexp(z)`) but it keeps the exponentials in a safe range. Always do it. Every sampler in this course starts here.

---

## 2. The knobs, as distribution transforms

Now the knobs. Each one is a *transformation applied before the draw*. Think of it as a pipeline: `logits → (temperature) → (truncation: top-k / top-p / min-p) → (penalties) → softmax → draw`. We take them one at a time and say exactly what each does to the distribution.

### 2.1 Temperature — scale the logits before softmax

Temperature `T` divides the logits before softmax:

```
p_i = softmax(z / T)
```

- **`T = 1`**: the distribution is unchanged (the model's "native" distribution).
- **`T < 1`** (e.g. 0.2): divides logits by a small number → *amplifies* the gaps → the distribution **sharpens** toward the top token. `T → 0` approaches **greedy** (argmax): the single most probable token, deterministically.
- **`T > 1`** (e.g. 1.5): divides logits by a big number → *shrinks* the gaps → the distribution **flattens**, giving low-probability tokens more chance. Higher `T` = more "surprising" output.

> **Temperature is not creativity.** It is a flatten/sharpen dial on a fixed distribution. High `T` does not make the model *think* of new ideas; it makes the sampler more willing to pick tokens the model already scored as unlikely. Sometimes that reads as creative; often it reads as incoherent, because the long tail of low-probability tokens is mostly noise. "Turn up the temperature for creativity" is folklore; what you are actually doing is widening the draw.

```python
def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:                 # T->0 is greedy; handle as a special case
        out = np.full_like(logits, -np.inf)
        out[np.argmax(logits)] = 0.0     # one survivor -> argmax after softmax
        return out
    return logits / temperature
```

### 2.2 Top-k — keep the k highest-probability tokens

Top-k truncation: keep only the `k` tokens with the highest logits, set the rest to `-inf` (so softmax gives them probability 0), renormalize, draw.

```python
def apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= logits.size:
        return logits
    kth = np.partition(logits, -k)[-k]   # the k-th largest logit value
    out = logits.copy()
    out[out < kth] = -np.inf             # drop everything below the cutoff
    return out
```

Effect: hard cap on how many tokens are even in play. `k = 1` is greedy. `k = 50` is a common default. The weakness: `k` is *fixed* regardless of how confident the model is. When the model is very sure (one token at 0.95), `k = 50` still keeps 49 near-zero tokens; when the model is unsure (a flat distribution), `k = 50` may chop off plausible tokens. That fixed-ness is exactly what top-p fixes.

### 2.3 Top-p (nucleus) — keep the smallest set whose cumulative probability ≥ p

Top-p, a.k.a. **nucleus sampling**: sort tokens by probability descending, accumulate until the running sum reaches `p`, keep exactly that set ("the nucleus"), drop the rest.

```python
def apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    if not (0.0 < p < 1.0):
        return logits
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]          # indices, most-probable first
    cumulative = np.cumsum(probs[order])
    # Keep tokens up to and INCLUDING the one that crosses p (so the set is non-empty).
    cutoff = np.searchsorted(cumulative, p) + 1
    keep = order[:cutoff]
    out = np.full_like(logits, -np.inf)
    out[keep] = logits[keep]
    return out
```

Effect: the cutoff is **adaptive**. When the model is confident (one token at 0.95), the nucleus is tiny — maybe one or two tokens. When the model is unsure (probability spread thin), the nucleus is large. This is why top-p is usually preferred over top-k: it keeps "the tokens that matter" rather than "a fixed number of tokens." A common default is `p = 0.9` or `0.95`.

> **Top-p is not diversity.** It is an adaptive truncation: how much of the probability mass you let into the draw. It interacts with temperature — temperature reshapes the distribution, then top-p truncates the reshaped distribution. Tuning both at once is why sampling "feels like alchemy" to people who never built the pipeline. You are building it, so it won't.

### 2.4 Min-p — keep tokens above a fraction of the top token's probability

Min-p is the newest of the common knobs and the most intuitive once you have top-p. Set a threshold *relative to the most probable token*: keep every token whose probability is at least `min_p × p_max`, where `p_max` is the top token's probability.

```python
def apply_min_p(logits: np.ndarray, min_p: float) -> np.ndarray:
    if not (0.0 < min_p < 1.0):
        return logits
    probs = softmax(logits)
    threshold = min_p * probs.max()      # scaled by the top token's confidence
    out = logits.copy()
    out[probs < threshold] = -np.inf
    return out
```

Effect: when the model is **confident** (a tall peak), the threshold is high, so only a few tokens survive — tight, focused output. When the model is **unsure** (a flat distribution, low `p_max`), the threshold is low, so many tokens survive — more exploration where the model itself is uncertain. Min-p adapts to the model's *confidence at this position*, which is a cleaner signal than a fixed `k` or a cumulative-mass `p`. A common value is `min_p = 0.05`–`0.1`. Many practitioners now prefer min-p as the single truncation knob, often with a higher temperature, because it lets temperature explore the *plausible* tail without admitting the noise tail.

### 2.5 Repetition penalty — down-weight already-seen tokens

The other family of knobs penalizes tokens that already appeared, to fight loops ("the the the") and verbatim repetition. A repetition penalty divides (or subtracts from) the logit of any token already in the sequence:

```python
def apply_repetition_penalty(logits: np.ndarray, generated_ids: list[int],
                             penalty: float) -> np.ndarray:
    if penalty == 1.0 or not generated_ids:
        return logits
    out = logits.copy()
    for tid in set(generated_ids):
        # Standard formulation: positive logits divided, negative logits multiplied,
        # so the penalty always pushes the token DOWN regardless of sign.
        out[tid] = out[tid] / penalty if out[tid] > 0 else out[tid] * penalty
    return out
```

Effect: makes the model less likely to reuse tokens it just produced. `penalty = 1.0` is off; `1.1`–`1.3` is a typical range. Overdo it and the model avoids necessary repetition (it'll refuse to say "the" twice, producing stilted text). There are variants (frequency penalty: scale by how *often* a token appeared; presence penalty: a flat penalty for appearing at all) — same idea, different bookkeeping.

### 2.6 The summary table

| Knob | What it does to the distribution | Adaptive? | Typical value |
|---|---|---|---|
| **Temperature** | Scales logits before softmax: flattens (`>1`) or sharpens (`<1`) | No (global) | `0.0`–`1.0` |
| **Top-k** | Keep the `k` highest-logit tokens, drop the rest | No (fixed count) | `40`–`50` |
| **Top-p (nucleus)** | Keep smallest set with cumulative prob ≥ `p` | Yes (to mass) | `0.9`–`0.95` |
| **Min-p** | Keep tokens with prob ≥ `min_p × p_max` | Yes (to confidence) | `0.05`–`0.1` |
| **Repetition penalty** | Down-weight already-seen tokens | Per-sequence | `1.0`–`1.3` |

> **Who owns these knobs?** On open-weights models served by Ollama/vLLM, all of these are yours. On some 2026 hosted frontier models (including the current Anthropic Opus tier, `claude-opus-4-8`), the user-facing `temperature`/`top_p`/`top_k` knobs have been **removed** — the vendor manages sampling internally and you steer behavior with the prompt. Sending those params is an error there. Knowing *who owns Stage 4* for a given deployment is part of reading the model as a component (Week 1 §2.4). You build the sampler this week against open models precisely because that is where the knobs live and where you can *see* what they do.

---

## 3. Writing the sampler: putting the pipeline together

Now assemble the full sampler. The pipeline order matters: penalties and temperature reshape the logits, *then* truncation drops tokens, *then* softmax-and-draw. Here is the whole thing — this is the spine of `exercise-02`, where you fill the truncation functions yourself:

```python
import numpy as np

def sample_next(logits: np.ndarray, *, temperature: float = 1.0,
                top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0,
                generated_ids: list[int] | None = None,
                repetition_penalty: float = 1.0,
                rng: np.random.Generator) -> int:
    """Turn one logit vector into one chosen token ID, applying the knobs in order."""
    z = logits.astype(np.float64)
    if generated_ids:
        z = apply_repetition_penalty(z, generated_ids, repetition_penalty)
    z = apply_temperature(z, temperature)      # reshape
    if top_k:   z = apply_top_k(z, top_k)       # truncate (fixed count)
    if top_p < 1.0: z = apply_top_p(z, top_p)   # truncate (cumulative mass)
    if min_p > 0.0: z = apply_min_p(z, min_p)   # truncate (relative to peak)
    probs = softmax(z)                          # back to a distribution
    return int(rng.choice(len(probs), p=probs)) # the draw
```

Two things to internalize from this code:

- **Greedy is just `temperature → 0`** (or `top_k = 1`). There is no separate "greedy mode"; it falls out of the same pipeline as a limiting case. Determinism is a sampler setting, not a model property — exactly Week 1's claim, now in code you can run.
- **The `rng` is explicit.** Seed it and the output is reproducible; the "randomness" is entirely in this one `rng.choice` call. When someone says "the model is non-deterministic," point at this line. The weights are deterministic (Week 1 §1); the draw is where the dice are.

The exercise has you pull real logits from a local model for one position, run this sampler at `T = 0.1, 0.7, 1.0, 1.5`, and watch the chosen-token distribution spread as temperature rises. Seeing the histogram flatten is worth a thousand words about what temperature "is."

---

## 4. Structured output: from "probably valid" to "provably valid"

Here is the part that changes how you build. You need the model to emit JSON matching a schema. The amateur approach is to *ask*: "Respond with valid JSON matching this schema, no prose." It works… most of the time. Then 2% of the time you get a trailing comma, a missing brace, a `"true"` where you wanted `true`, or a chatty `Sure! Here's your JSON:` preamble, and your `json.loads` throws. So you add a retry. Now you are babysitting a probability with try/except.

> **The week's promise, restated: if your JSON generator "usually" produces valid output and you handle breaks with a retry, you are not done. Retry-on-broken-JSON is the symptom of asking the model nicely instead of constraining the sampler. The fix is to make invalid output structurally impossible.**

### 4.1 JSON mode vs grammar-constrained decoding

There are two levels of "structured output," and they are not the same:

- **JSON mode** (a vendor feature): the model is biased/instructed toward JSON, often with server-side validation and retry. Better than nothing. Still fundamentally "the model tries hard to produce JSON." It can still fail the *schema* (right syntax, wrong shape).
- **Grammar-constrained decoding** (what we build understanding of): at **every decode step**, before the draw, *mask out every token that would make the output violate the grammar*. The sampler is only allowed to choose among tokens that keep the output on a valid path through the grammar. Invalid output is not "unlikely" — it is *unreachable*.

### 4.2 The mechanism: masking the logits against a grammar

Recall the sampler: it draws from a distribution over `V` tokens. Grammar-constrained decoding inserts one more transform — a **mask** — right before the draw:

```
logits → (temperature/truncation) → MASK illegal tokens to -inf → softmax → draw
```

The mask comes from a state machine (compiled from a regex or JSON schema) that tracks "given what we've emitted so far, which next tokens keep us on a valid path?" After `{"name":` the grammar knows the next token must begin a string; every token that isn't `"` (or whitespace) gets masked to `-inf`. The model literally *cannot* emit a syntax error, because the tokens that would cause one have probability 0.

```python
# Conceptual sketch (the real engines — outlines, xgrammar — do this efficiently):
def masked_sample(logits, grammar_state, rng):
    allowed = grammar_state.allowed_token_ids()   # set of legal next tokens
    mask = np.full_like(logits, -np.inf)
    mask[list(allowed)] = 0.0                      # 0 for legal, -inf for illegal
    probs = softmax(logits + mask)                 # illegal tokens -> probability 0
    tok = int(rng.choice(len(probs), p=probs))
    grammar_state.advance(tok)                     # move the state machine forward
    return tok
```

This is the same logit-masking pattern as top-k/top-p — you have been doing "set illegal tokens to `-inf`" all lecture. Grammar-constrained decoding is that operation driven by a grammar instead of a probability threshold.

### 4.3 The 2026 toolset: `outlines`, `guidance`, `xgrammar`

You do not build the state machine yourself in production — you use a library:

- **`outlines`** — the one you use this week. Compiles a regex or a JSON schema into a finite-state machine and constrains a local model's decoding to it. `generate.json(model, schema)` gives you a generator whose every output is schema-valid by construction.
- **`guidance`** — a templating-style API for interleaving fixed text, constrained regions, and free generation (`{{gen 'name' pattern='...'}}`). Same masking idea, ergonomics tuned for "fill in this template."
- **`xgrammar`** — a fast grammar engine focused on *low per-token overhead*, integrated into serving stacks (vLLM). When you care about the latency cost of constraint (and you should — masking is not free), this is the performance-minded option.

The point this week is the *idea*, demonstrated with `outlines`:

```python
import json, jsonschema
from outlines import models, generate

model = models.transformers("Qwen/Qwen2.5-0.5B-Instruct")  # tiny, runs on CPU
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "confidence"],
}
generator = generate.json(model, json.dumps(schema))   # the schema CONSTRAINS the sampler

for prompt in fuzz_prompts:           # 100 adversarial prompts, including hostile ones
    out = generator(prompt)
    obj = json.loads(out)             # never raises — it CANNOT be invalid JSON
    jsonschema.validate(obj, schema)  # always passes — by construction, not by luck
```

`exercise-03` is exactly this against a fuzz set, asserting 100% schema validity. When the assertion holds across every adversarial prompt — including prompts engineered to make the model chatty or to inject extra fields — you have *proven* the property, not observed it. That is the week's promise made executable.

### 4.4 The cost of constraint, and when not to use it

Grammar-constrained decoding is not free. Masking adds per-token overhead (computing the allowed set and the mask each step), and an over-tight grammar can *hurt* quality by forbidding tokens the model wanted for good reasons (forcing a field the model has no information for makes it hallucinate a value). Two engineering rules:

- **Constrain the structure, not the content.** Enforce "this is an object with a `sentiment` enum and a numeric `confidence`"; don't over-specify a regex so tight the model can only produce one string. The grammar guarantees *shape*; the model still owns *substance*.
- **Measure the overhead.** The stretch goal compares `outlines` vs `xgrammar` per-token latency. Knowing the cost is part of using constraint well — you are trading a little decode speed for a hard guarantee, and that's usually a great trade, but it's a trade.

---

## 5. Beam search, and why nobody uses it

You will see beam search in older papers and translation systems and wonder why your LLM doesn't use it. Here is the whole story.

### 5.1 What beam search does

Beam search is a **search**, not a sampler. Instead of drawing one token at a time, it keeps the `b` most probable *partial sequences* ("beams") at each step. For each beam it expands all next tokens, scores every resulting sequence by cumulative (log-)probability, and keeps the top `b` overall. At the end it returns the highest-probability *complete sequence* found.

```
beam width b = 2:
step 1:  "The"(−0.2)   "A"(−0.9)            keep top 2
step 2:  "The cat"(−0.5)  "The dog"(−0.7)   "A cat"(−1.4) ...  keep top 2
step 3:  ... expand each, keep top 2 by cumulative log-prob ...
return the highest-scoring COMPLETE sequence
```

It approximates "find the single most probable output sequence" better than greedy (which is myopic — it can't undo an early high-probability choice that leads to a dead end).

### 5.2 Why it was standard, and why it lost

Beam search was the default for **machine translation** and summarization, where there is roughly *one correct output* and "most probable sequence" is close to "best sequence." For those closed-ended tasks it beat greedy and sampling.

For **open-ended generation** (chat, writing, reasoning) it fails, for a sharp reason:

- **The most probable sequence is bland and repetitive.** Maximizing cumulative probability drives the output toward the safest, most generic continuation at every step — "the the the," looping phrases, dull text. Human-like text is *not* the maximum-probability sequence; it has surprise in it. This is the same observation that motivated nucleus sampling: high-quality open-ended text lives in the *typical* region of the distribution, not at its peak.
- **It's expensive.** `b` beams means `b×` the compute, and the KV-cache bookkeeping for parallel hypotheses is awkward.
- **Sampling is simply better here.** Temperature + nucleus/min-p sampling produces more natural, more diverse, more human text for open-ended tasks, at `1×` cost.

> **The verdict:** beam search optimizes for "the most probable sequence," which is the *right* target for closed-ended tasks (translation) and the *wrong* target for open-ended generation (chat, writing) — where the most probable sequence is bland. That mismatch, plus its cost, is why almost nobody uses beam search for LLM generation in 2026. Know it exists, know what it does, and know why you reach for sampling instead.

---

## 6. Recap

You should now be able to:

- Apply softmax (with the subtract-the-max stability trick) to turn logits into a distribution, and explain why the exponential amplifies the model's favorite.
- Define each sampling knob as a *transform on the distribution*: temperature (scale before softmax — flatten/sharpen), top-k (fixed-count truncation), top-p/nucleus (cumulative-mass truncation, adaptive), min-p (truncation relative to the peak, confidence-adaptive), repetition penalty (down-weight seen tokens) — and predict each one's qualitative effect.
- Write the full sampler in NumPy: penalties → temperature → truncation → softmax → seeded draw, with greedy as the `T→0` limiting case, and state that determinism is a sampler setting, not a model property.
- Explain grammar-constrained decoding as per-step logit masking against a state machine compiled from a schema/regex, name the 2026 toolset (`outlines`, `guidance`, `xgrammar`), and articulate why it converts "probably valid" into "provably valid" — while respecting its per-token cost and the "constrain structure, not content" rule.
- State what beam search does (keep the top-`b` partial sequences by cumulative probability), why it suited translation, and why its "most probable sequence" target makes it bland and unused for open-ended LLM generation.

That closes Week 2. You can now estimate cost to a few percent (Lecture 1), control the sampler instead of fearing it (this lecture §1–3), and make structured output a guarantee (§4). Next week takes this precision into **prompt engineering as engineering** — versioning, diffing, and regression-testing prompts so "better" is a measured claim. Your token-accounting instrument from the mini-project becomes part of how you measure a prompt's cost there.

---

## References

- *The Curious Case of Neural Text Degeneration* (Holtzman et al., 2020) — introduces nucleus (top-p) sampling and the case against maximum-probability decoding: <https://arxiv.org/abs/1904.09751>
- *Min-p sampling* — *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs* (Nguyen et al., 2024): <https://arxiv.org/abs/2407.01082>
- *Efficient Guided Generation for LLMs* (Willard & Louf, 2023 — the `outlines` paper, FSM-constrained decoding): <https://arxiv.org/abs/2307.09702>
- *XGrammar: Flexible and Efficient Structured Generation* (2024): <https://arxiv.org/abs/2411.15100>
- *`outlines` documentation* (regex/JSON-schema-constrained generation): <https://dottxt-ai.github.io/outlines/>
- *`guidance` documentation*: <https://github.com/guidance-ai/guidance>
- *Anthropic — Messages API / models overview* (which sampling knobs the frontier tier exposes): <https://docs.claude.com/en/docs/about-claude/models/overview>
- *The Curious Case ... §3* on why beam search degenerates for open-ended generation (same Holtzman paper above).
