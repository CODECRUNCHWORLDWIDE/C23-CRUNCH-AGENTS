# Mini-Project — `toklab`: A Tokenization, Cost, and Constrained-Output Lab

> Build a CLI tool `toklab` with three subcommands: `tokens` (explore how text tokenizes across models and show the disagreement), `cost` (instrument a request for per-call token accounting and project the bill), and `json` (generate schema-constrained JSON that is *provably* valid across a fuzz set). Three subcommands, one thesis: **you measure tokens with the right tokenizer, you account for every one, and you make structure a constraint — not a hope.**

This is the artifact that turns Week 2's whole argument into a tool you'll actually reach for. By the end you have a command that answers, for any text and model, "how many tokens is this, what does it cost, and how do I get guaranteed-valid structured output?" — with real numbers from real tokenizers and a validity rate that is exactly 100%, not "usually."

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** The token-accounting instrument (`cost`) becomes the measurement core of the **Week 3 prompt-engineering harness** (a "better" prompt is a measured cost-and-quality claim) and the **Week 21 cost-engineering/routing layer**. The constrained-JSON generator (`json`) is the structured-output backbone every tool-using agent in Phase III relies on. Build it well now; you'll extend it for the rest of the course.

---

## What you will build

A small Python package `toklab` with three deliverables, one per subcommand:

1. **`toklab/tokens.py`** — the tokenization explorer. Given text and a list of models, encode it with each model's own tokenizer (HuggingFace `AutoTokenizer` for open models, `count_tokens` for Anthropic), and report the per-model token count, the max/min disagreement ratio, and (on request) the actual sub-word pieces. This is `exercise-01` promoted to a reusable module.
2. **`toklab/cost.py`** — the per-request token-accounting instrument. Given a prompt (and optional system prompt) and a model, record `tokens_in` (broken down: system / user / overhead where available), `tokens_out`, and `cost` from **real** counts, and project a monthly bill at a stated volume across tiers. This is the headline instrument.
3. **`toklab/jsongen.py`** — the schema-constrained generator. Given a JSON schema and a prompt, produce output that is schema-valid **by construction** via `outlines`, and a `--fuzz` mode that runs a fuzz set and asserts 100% validity with `jsonschema`. This is `exercise-03` promoted to a module.

A `toklab/cli.py` wires the three behind one `argparse` entry point. By the end you have a public repo of ~300–450 lines of Python (excluding tests) you can run against any text, prompt, or schema.

---

## Why these three, together

The three subcommands are the three stages this week owns, end to end:

- **`tokens`** is Stage 1 (the tokenizer): *what* you pay for, measured with the right ruler.
- **`cost`** is Stage 1 + Stage 2 (tokenizer + context budget): *how much* you pay, accounted per request and projected to a bill.
- **`json`** is Stage 4 (sampling): *what you get*, made a structural guarantee by constraining the sampler.

Two design rules are non-negotiable, because they're the lessons of the week:

- **Always count with the model's own tokenizer.** `count_tokens` for Anthropic, `AutoTokenizer` for open models, `prompt_eval_count` for Ollama. **Never `tiktoken` for a non-OpenAI model** — `toklab` exists partly to make that mistake impossible to make accidentally.
- **Structure is a constraint, not a prompt.** The `json` subcommand must use grammar-constrained decoding and must *assert* validity, not validate-and-retry. A generator that "usually" works is a fail on this project's central axis.

---

## Package layout

```
toklab/
├── pyproject.toml
├── toklab/
│   ├── __init__.py
│   ├── tokens.py        # AutoTokenizer / count_tokens explorer + disagreement ratio
│   ├── cost.py          # per-request token accounting + monthly projection
│   ├── jsongen.py       # outlines schema-constrained generation + fuzz assertion
│   └── cli.py           # argparse: `toklab tokens|cost|json ...`
└── tests/
    ├── test_tokens.py   # counts differ across tokenizers; ratio computed right
    ├── test_cost.py     # cost formula correct for hosted + local price tables
    └── test_jsongen.py  # is_schema_valid() accepts good JSON, rejects bad
```

---

## Deliverable 1 — `tokens.py` (the explorer)

Promote `exercise-01` into a module. It must:

- Provide `count(text: str, model: str) -> int` that dispatches to the right tokenizer: a `claude-*` model → `client.messages.count_tokens`; an open-model HF id → `AutoTokenizer.from_pretrained(...).encode(...)`.
- Provide `compare(text: str, models: list[str]) -> dict[str, int]` returning per-model counts, plus a `ratio(counts) -> float` helper for max/min disagreement.
- Optionally return the sub-word **pieces** (`--pieces`) so the user can *see* why counts differ.
- Cache loaded `AutoTokenizer`s so repeated calls don't re-load.

Here is the spine to start from; fill in the rest yourself:

```python
"""toklab.tokens — count and compare tokenizations with each model's OWN tokenizer."""
from __future__ import annotations

import os
from functools import lru_cache

_HF = {  # friendly label -> HuggingFace tokenizer id (open models)
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "gpt": "Xenova/gpt-4o",
}


@lru_cache(maxsize=None)
def _hf_tokenizer(hf_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_id)


def count(text: str, model: str) -> int:
    """Token count from the MODEL'S OWN tokenizer. Never tiktoken for non-OpenAI."""
    if model.startswith("claude-"):
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"{model}: ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic()
        r = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}])
        return r.input_tokens
    hf_id = _HF.get(model, model)  # allow a raw HF id too
    return len(_hf_tokenizer(hf_id).encode(text, add_special_tokens=False))


def compare(text: str, models: list[str]) -> dict[str, int]:
    return {m: count(text, m) for m in models}


def ratio(counts: dict[str, int]) -> float:
    vals = [v for v in counts.values() if v > 0]
    return max(vals) / min(vals) if vals else 1.0
```

---

## Deliverable 2 — `cost.py` (the per-request instrument)

This is the headline instrument and the piece that compounds forward. Given a prompt (and optional system prompt) and a model, it must:

1. **Record real token counts**: `tokens_in` from `count_tokens` (or the open tokenizer), `tokens_out` from an actual generation's `usage` (or a stated assumption for a dry estimate). Where the API exposes it, **break `tokens_in` down** by system prompt vs user content vs chat-template overhead.
2. **Compute cost** with the per-MTok formula: `cost = tokens_in*price_in/1e6 + tokens_out*price_out/1e6`. Output is priced 3–5× input — your report must surface that.
3. **Project a monthly bill** at a stated `--calls-per-day`, and compute the same projection for the other tiers in the price table so the user sees the cost gradient.
4. **Flag the hidden cost**: if a system prompt is present and re-sent every call, note its per-call and monthly contribution and that prompt caching would discount it.

The price table and the pure cost function must be **unit-testable without any API call**:

```python
"""toklab.cost — per-request token accounting and monthly projection."""
from __future__ import annotations

from dataclasses import dataclass

PRICES = {  # USD per 1,000,000 tokens
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-opus-4-8": {"in": 5.00, "out": 25.00},
}
LOCAL_PRICE = {"in": 0.0, "out": 0.0}


@dataclass
class CallCost:
    model: str
    tokens_in: int
    tokens_out: int

    def cost_usd(self) -> float:
        p = PRICES.get(self.model, LOCAL_PRICE)
        return (self.tokens_in * p["in"] + self.tokens_out * p["out"]) / 1_000_000

    def monthly_usd(self, calls_per_day: int, days: int = 30) -> float:
        return self.cost_usd() * calls_per_day * days


def project_all_tiers(tokens_in: int, tokens_out: int,
                      calls_per_day: int) -> dict[str, float]:
    """Same token shape, every tier -> the cost gradient a budget owner wants."""
    return {
        model: CallCost(model, tokens_in, tokens_out).monthly_usd(calls_per_day)
        for model in PRICES
    }
```

The I/O part (actually calling the model to get `usage`, or `count_tokens` for a dry run) wraps this pure core. Keep them separate so `test_cost.py` can test the math with hand-built numbers.

---

## Deliverable 3 — `jsongen.py` (the provable structured-output generator)

Promote `exercise-03` into a module. It must:

- Build a schema-constrained generator from a JSON schema via `outlines.generate.json` over a small local model.
- Provide `generate(prompt, schema) -> dict` returning a parsed, schema-valid object.
- Provide `fuzz(schema, prompts) -> FuzzResult` that runs every prompt through the constrained generator and **asserts** with `jsonschema` that each output is valid — returning a 100%-or-it-raised result, not a "validity rate" you eyeball.
- Provide `is_schema_valid(raw, schema) -> tuple[bool, str]` (parse + validate) as the reusable check, unit-tested directly.

```python
"""toklab.jsongen — schema-constrained generation that is PROVABLY valid."""
from __future__ import annotations

import json
from dataclasses import dataclass

import jsonschema


def is_schema_valid(raw: str, schema: dict) -> tuple[bool, str]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"not JSON: {e}"
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as e:
        return False, f"schema: {e.message}"
    return True, "ok"


@dataclass
class FuzzResult:
    total: int
    valid: int   # invariant: under the constraint, valid == total, always

    @property
    def rate(self) -> float:
        return 100.0 * self.valid / self.total if self.total else 0.0


def build(schema: dict, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    from outlines import models, generate
    model = models.transformers(model_id)
    return generate.json(model, json.dumps(schema))


def fuzz(schema: dict, prompts: list[str]) -> FuzzResult:
    gen = build(schema)
    valid = 0
    for p in prompts:
        out = gen(p)
        raw = out if isinstance(out, str) else json.dumps(out)
        ok, reason = is_schema_valid(raw, schema)
        assert ok, f"constraint failed on {p!r}: {reason}"  # MUST NOT fire
        valid += 1
    return FuzzResult(total=len(prompts), valid=valid)
```

---

## Deliverable 4 — `cli.py` (the command)

An `argparse` entry point with three subcommands. Examples of the required output shape:

```
$ toklab tokens --text "今天天气很好" --models qwen,llama,gpt --pieces
qwen   : 4 tokens   ['今天', '天气', '很', '好']
llama  : 10 tokens  ['今', '天', '天', '气', ...]
gpt    : 9 tokens   ['今', '天', ...]
disagreement ratio (max/min): 2.50x  <- the cost-estimation error, made visible

$ toklab cost --prompt-file report.txt --system-file sys.txt \
              --model claude-haiku-4-5 --calls-per-day 20000
tokens_in : 2412 (system 1402 / user 985 / overhead 25)
tokens_out: 118
cost/call : $0.003002
monthly @ 20000/day:
  claude-haiku-4-5   $1,801   <- chosen
  claude-sonnet-4-6  $5,404
  claude-opus-4-8    $9,007
note: the 1402-token system prompt is $841/mo of that bill — prompt-cacheable.

$ toklab json --schema schema.json --fuzz fuzz_prompts.txt
constrained validity: 25/25 (100.0%)  PASS — schema-valid by construction
```

---

## Rules

- **You must** count tokens with each model's own tokenizer; **never** `tiktoken` for a non-OpenAI model, never characters/words. A cross-tokenizer count is an automatic fail on the "measurement discipline" axis.
- **You must** make `json --fuzz` *assert* validity (not validate-and-retry), and it must report 100%. If it can ever print less than 100%, the constraint isn't wired up.
- **You must** keep the pure cost math and the pure validity check testable without any network/model call.
- **You must not** crash when a backend is unavailable (no key, no local model) — degrade to the paths that still work and say which were skipped.
- Python 3.12; dependencies limited to `transformers`, `tokenizers`, `numpy`, `outlines`, `jsonschema`, `anthropic`, plus `pytest` for tests.

---

## Milestones

1. **Thursday — `tokens` + `cost` core.** `toklab tokens` compares ≥3 tokenizers and prints the ratio; `toklab cost` prints a per-call breakdown and a 3-tier monthly projection from real counts. Pure cost math unit-tested.
2. **Friday — `json` + assertions.** `toklab json --fuzz` runs a fuzz set through `outlines` and asserts 100% schema validity; `is_schema_valid` unit-tested on good and bad inputs.
3. **Saturday — polish + report.** CLI ergonomics, the README with the documented price table and one "this surprised me" paragraph, all tests green, repo pushed.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-02-toklab-<yourhandle>`.
- [ ] `pip install -e .` succeeds; `toklab --help` lists the three subcommands.
- [ ] `toklab tokens` reports per-model counts from each model's **own** tokenizer and the max/min ratio; `--pieces` shows sub-word pieces.
- [ ] `toklab cost` reports `tokens_in` (broken down where possible), `tokens_out`, cost/call, and a **3-tier monthly projection** — all from **real** token counts, and flags a re-sent system prompt as cacheable.
- [ ] `toklab json --fuzz` runs a fuzz set and **asserts** 100% schema validity (the assertion is in the code), printing the rate.
- [ ] `pytest` passes, with at least:
  - `test_tokens.py`: counts differ across tokenizers for a code/CJK probe; `ratio()` correct.
  - `test_cost.py`: `cost_usd()` and `monthly_usd()` correct for a hosted and a local price table.
  - `test_jsongen.py`: `is_schema_valid()` accepts valid JSON and rejects malformed / schema-violating JSON.
- [ ] A `README.md` with run commands, the **documented price table**, and one paragraph on a result that surprised you (e.g. "the Chinese probe was 2.5× more tokens on the wrong tokenizer — that's a 150% budget error").
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Right-tokenizer discipline** | 20 | Counts always from the model's own tokenizer; no `tiktoken` for non-OpenAI; dispatch is correct. |
| **Cost accounting** | 25 | Real `tokens_in`/`tokens_out`; per-call breakdown; 3-tier monthly projection; output-priced-higher reflected; system-prompt flag. |
| **Provable structured output** | 25 | `outlines`-constrained generation; `--fuzz` *asserts* 100% validity (not validate-and-retry); `jsonschema` used. |
| **Pure cores + tests** | 20 | Cost math and validity check are pure and tested without I/O; `pytest` green; tokenizer-disagreement tested. |
| **CLI & docs** | 10 | Three working subcommands; documented price table; one honest "surprised me" paragraph; no secrets committed. |

**90+** is portfolio-grade and ready to grow into the Week-3 prompt harness and the Week-21 router. **70–89** works but estimates somewhere it should measure, or validate-and-retries where it should constrain. **Below 70** either counts with the wrong tokenizer or "usually" produces valid JSON — both are the exact mistakes this week argues against; fix those first.

---

## Stretch goals

- **Worst-case-disagreement finder.** Add `toklab tokens --worst` that searches a corpus for the snippet with the largest max/min tokenizer ratio. The number it surfaces is the cost-estimation error in its most extreme form.
- **Prompt-cache modeling.** Add a `--cache-system` flag to `cost` that recomputes the monthly bill assuming the system prompt is a cached prefix (discounted). For a fixed system prompt at scale, the delta is the single biggest cost lever — quantify it.
- **`outlines` vs `xgrammar` overhead.** Add a `--bench` mode to `json` that generates the same constrained output with two engines and reports the per-token overhead each adds. Constraint is not free; knowing the cost is part of using it well.
- **Sampler visualizer.** Wire your `exercise-02` NumPy sampler into `toklab` and add `toklab sample --temperature ... --top-p ...` that prints the surviving-token distribution after each transform — a teaching tool that shows the knobs reshaping the distribution live.

---

## How this connects to the rest of C23

- **Week 3 (prompt engineering as engineering)** uses your `cost` instrument to make a "better prompt" a *measured* cost-and-quality claim, and versions/diffs prompts against it.
- **Phase III (agents & tools)** relies on your `json` generator: every tool call an agent makes needs schema-valid arguments, and "provably valid" is what keeps an agent loop from derailing on a malformed call.
- **Week 21 (cost engineering & routing)** is `toklab cost` grown up: instead of accounting one call, it routes a stream of calls to the cheapest tier that clears a quality bar, with prompt caching in front.

When you've finished, push the repo and take the [quiz](../quiz.md).
