# Exercise 1 — Token accounting: find where the money goes

**Time estimate:** ~40 minutes. Guided.

## Goal

Meter a real (or simulated) workload, attribute its cost *per route and per feature* from the `usage` block, and build the cost table that tells you which lever to pull. By the end you'll have done the cost-detective work from Lecture 1 §1 — turning a single "we spent $X" number into "$Y of it is feature Z doing a 7B's job on Opus," which is the difference between knowing your bill and knowing your fix.

## Prerequisites

- `pip install anthropic`. An `ANTHROPIC_API_KEY` if you want real numbers; otherwise use the published prices and a few hand-built `usage` records to do the accounting on paper.
- The current Claude prices (resources.md): Haiku $1/$5, Sonnet $3/$15, Opus $5/$25 per million input/output tokens.

## Steps

### 1. Make a few real (or representative) calls and capture `usage`

Send three kinds of request and capture the `usage` block from each:

```python
from anthropic import Anthropic
client = Anthropic()

def cost_of(usage, model_prices):
    return (usage.input_tokens  * model_prices["in"]  / 1e6
          + usage.output_tokens * model_prices["out"] / 1e6
          + getattr(usage, "cache_read_input_tokens", 0) * model_prices["in"] * 0.1 / 1e6)

PRICES = {"claude-haiku-4-5": {"in": 1.0, "out": 5.0},
          "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
          "claude-opus-4-8":  {"in": 5.0, "out": 25.0}}

resp = client.messages.create(
    model="claude-sonnet-4-6", max_tokens=512,
    thinking={"type": "adaptive"}, output_config={"effort": "medium"},
    messages=[{"role": "user", "content": "Summarize the key idea of continuous batching."}],
)
print(resp.usage, "->", cost_of(resp.usage, PRICES["claude-sonnet-4-6"]))
```

(No key? Hand-build `usage`-like records with realistic token counts per the table in Lecture 1 §1's worked example, and do the accounting on those.)

### 2. Tag each call with its route and feature, and aggregate

Build a list of call records, each tagged `{feature, model, input_tokens, output_tokens, cost}`. Cover at least three features at different models, including one *deliberately wasteful* one — a trivial classification task running on Opus. Aggregate cost per feature and per model into a table:

```
feature     model           calls   avg_in   avg_out   $/period
chat        claude-sonnet-4-6  ...     ...      ...       ...
summarize   claude-opus-4-8    ...     ...      ...       ...
classify    claude-opus-4-8    ...     ...      ...       ...   <- the leak
```

### 3. Read the table like a detective and name the fix

Write `exercise-01-notes.md` answering: which feature is the biggest spender, and is that *justified* (high-value user work) or *wasteful* (a frontier model on a trivial task)? For the wasteful one, name the lever (route to a cheap model; cache; compress) and estimate the saving. The point is the *diagnosis* — the table doesn't just total the bill, it points at the leak.

### 4. Add prompt caching to one feature and re-meter

Take a feature that sends a repeated large prefix (a long system prompt). Add `cache_control: {"type": "ephemeral"}` to the prefix, call it twice, and capture `cache_read_input_tokens` on the second call. Compute the cost with and without the cache read. Record the discount you measured.

## Acceptance criteria

- [ ] A per-feature, per-model cost table built from `usage` (real or representative), with at least three features.
- [ ] One feature identified as a *leak* (a frontier model doing a cheap-model job), with the lever and estimated saving named.
- [ ] Prompt caching applied to one repeated-prefix feature, with `cache_read_input_tokens > 0` on the second call and the measured discount recorded.
- [ ] `exercise-01-notes.md` reads as a diagnosis (where the money goes, what to fix), not just a total.

## Hint

Output tokens dominate cost (4–5× input), so a verbose feature is more expensive than a long-prompt one — check `output_tokens`, not just `input_tokens`, when hunting the leak. The classic leak is a high-*volume* trivial task on a frontier model: 50,000 classify calls/day on Opus costs more than a few hundred chat calls, even though each classify call is tiny. Volume × model-price is the thing to scan for.

## Why this matters

The capstone requires a cost report broken down by route. This exercise is that report's foundation — the per-route attribution from `usage` that every later lever (caching, routing, the cost dashboard) is measured against. An engineer who can read a cost table and point at the leak is the one who cuts the bill; one who sees only the total is guessing.
