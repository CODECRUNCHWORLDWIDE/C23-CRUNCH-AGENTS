# Mini-Project — `crunchtune`: A Fine-Tune-or-Not Pipeline

> Build a reusable fine-tuning pipeline — dataset prep, LoRA training, held-out evaluation, and a worth-it-or-not verdict — so "should we fine-tune this, and did it help?" becomes `python -m crunchtune run --task dsl` and a memo with a number, not a reflex and a vibe.

This is the artifact that turns fine-tuning from a hopeful experiment into a measured decision. After this week, deciding to fine-tune is "build the dataset, train the adapter, run the held-out eval, read the verdict" — not "fine-tune and hope it's better." The pipeline is task-pluggable, eval-honest, and produces the one deliverable that matters: a base-vs-fine-tune comparison on data the model never saw, and a defensible "ship it / skip it."

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule. The actual GPU training is ~15–30 minutes of that; the rest is the dataset, the eval harness, and the verdict.

**Compounds forward:** This pipeline produces the **Phase III milestone's fine-tune-or-not decision document** (end of week 18 requires exactly that for the capstone's domain). It's also the template you'd reuse any time the capstone or a future project faces a "should we fine-tune?" question. The syllabus's week-16 deliverable is "decide whether the fine-tune was worth it"; *this pipeline is how you decide, repeatably.*

---

## What you will build

A small Python package `crunchtune` with five deliverables:

1. **`crunchtune/data.py`** — dataset prep: load raw NL→DSL pairs, validate every target (parses + canonical), apply the chat template, de-duplicate, and split into train/test with a fixed seed. The dataset-quality firewall from Lecture 2, in code.
2. **`crunchtune/train.py`** — the LoRA/QLoRA training loop: 4-bit-load the base, attach a LoRA adapter, train with `SFTTrainer`, save the adapter, and capture the loss curve. The single source of truth for "how the adapter is trained."
3. **`crunchtune/eval.py`** — the honest evaluator: run *base* (best prompt) and *fine-tune* over the held-out test set, score exact-match + valid-DSL + latency, return the comparison. The part that's the actual deliverable.
4. **`crunchtune/verdict.py`** — the decision logic: given the base and fine-tune scores and the cost, render "worth it / not worth it" with a stated threshold, and emit the memo skeleton.
5. **`crunchtune/cli.py`** — `data`, `train`, `eval`, and `run` (the whole pipeline) commands.

By the end you have a public repo of ~450–550 lines of Python that any future "should we fine-tune?" question can be pointed at, producing a number instead of an argument.

---

## Why a pipeline and not a notebook

You could do this in a Colab notebook. Don't — not as the artifact. A pipeline gives you:

- **Reuse.** The Phase III milestone wants a fine-tune-or-not decision for the *capstone's* domain; swap the dataset, re-run the pipeline, get the verdict. A notebook gets copy-pasted and the eval drifts.
- **An honest eval, in code.** The held-out split, the base-vs-fine-tune comparison, the "never evaluate on training data" rule — these live in version-controlled code, not in a cell you might accidentally point at the wrong file. "Did the fine-tune help?" is answered by re-running `eval.py`, not by eyeballing a notebook.
- **A repeatable verdict.** `run --task dsl` produces the same comparison every time (fixed seeds), so two training runs are *comparable*. A notebook with a re-shuffled split every run can't compare anything.
- **A CLI.** `crunchtune eval --base ... --adapter ...` is scriptable and CI-able. A notebook cell is none of those.

Notebooks are great for *exploring* a loss curve or sanity-checking a few generations by eye. The thing you ship and base a decision on is a pipeline. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchtune/
├── pyproject.toml
├── README.md                     # the results table + the worth-it memo
├── data/
│   ├── raw_pairs.jsonl           # NL->DSL pairs before validation/split
│   ├── train.jsonl               # produced by `crunchtune data`
│   └── test.jsonl                # the held-out firewall
├── crunchtune/
│   ├── __init__.py
│   ├── data.py                   # validate + template + dedup + split
│   ├── train.py                  # QLoRA + LoRA + SFTTrainer
│   ├── eval.py                   # base-vs-fine-tune on held-out
│   ├── verdict.py                # worth-it/not logic + memo skeleton
│   └── cli.py                    # data / train / eval / run
└── tests/
    ├── test_data.py              # validation, canonical-form, no-leakage
    └── test_eval.py              # exact-match + valid-DSL metrics correct
```

---

## Deliverable 1 — `data.py` (the dataset-quality firewall)

Everything that decides the fine-tune's quality lives here. No raw pair reaches training without passing validation, templating, and the no-leakage split.

```python
"""crunchtune.data — dataset prep with the quality firewall.

The dataset IS the fine-tune. Nothing trains on data that isn't valid, canonical,
de-duplicated, chat-templated, and split with the test set held out.
"""
from __future__ import annotations

import json
import re

from datasets import Dataset
from transformers import AutoTokenizer

_TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def validate(output: str) -> bool:
    """Target must parse as DSL AND be in canonical form (no format drift)."""
    # TODO 1: structural parse (matches the DSL grammar) AND canonical-form check
    #   (single-quoted state codes, trailing ;, single spaces, no double quotes).
    ...


def to_text(instruction: str, output: str) -> str:
    """Apply the model's chat template — NEVER hand-concatenate."""
    return _TOK.apply_chat_template(
        [{"role": "user", "content": instruction},
         {"role": "assistant", "content": output}],
        tokenize=False,
    )


def prepare(raw_path: str, test_size: float = 0.1, seed: int = 42):
    """Load raw pairs -> validate -> dedup -> template -> split. Returns (train, test)."""
    rows = [json.loads(line) for line in open(raw_path)]
    clean = {r["instruction"]: r for r in rows if validate(r["output"])}  # dedup + validate
    ds = Dataset.from_list(list(clean.values())).shuffle(seed=seed)
    ds = ds.map(lambda r: {"text": to_text(r["instruction"], r["output"])})
    split = ds.train_test_split(test_size=test_size, seed=seed)   # the firewall
    # TODO 2: assert NO test instruction (normalized) appears in train (leakage),
    #   and raise if it does — a leak invalidates the whole eval.
    return split["train"], split["test"]
```

> **The rule the project enforces:** no example trains unless it passed `validate()`, and the test set is *never* in the train set. If `test_data.py`'s leakage assertion ever fails, the eval downstream is meaningless — so the firewall is a hard gate, not a warning.

---

## Deliverable 2 — `train.py` (QLoRA + LoRA + SFT)

The training loop, with the loss curve captured.

```python
def train(train_ds, out_dir: str, r: int = 16, alpha: int = 32, epochs: int = 2):
    """4-bit load + LoRA attach + SFT. Saves the adapter; returns the loss log."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    model, tok = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", max_seq_length=2048, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, r=r, lora_alpha=alpha, lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    # TODO 3: print the trainable-param count and assert it's <1% of total (it's PEFT).
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=train_ds,
        args=SFTConfig(num_train_epochs=epochs, learning_rate=2e-4, logging_steps=1,
                       per_device_train_batch_size=2, gradient_accumulation_steps=4,
                       output_dir=out_dir, seed=42))
    result = trainer.train()
    model.save_pretrained(out_dir)
    return trainer.state.log_history   # the loss curve, for the memo
```

The non-negotiables `train.py` enforces:

- **QLoRA, not full fine-tuning** — 4-bit base, so it fits in 24 GB (the Lecture 1 memory math).
- **The adapter is <1% of params** — print and assert it. If you're training the whole model, you've lost PEFT.
- **The chat template was applied upstream** (in `data.py`) — `train.py` trusts that `train_ds["text"]` is correctly templated.

---

## Deliverable 3 — `eval.py` (the honest evaluator — the real deliverable)

The part that decides everything. Both models, same held-out set, never the training data.

```python
def evaluate_one(model, tok, test_ds) -> dict:
    exact = valid = 0
    latencies = []
    for ex in test_ds:
        out, dt = timed_generate(model, tok, ex["instruction"])
        exact += (out.strip() == ex["output"].strip())   # exact-match
        valid += dsl_parses(out)                          # structural validity
        latencies.append(dt)
    n = len(test_ds)
    return {"exact_match": exact/n, "valid_dsl": valid/n, "avg_latency": mean(latencies)}


def compare(base_model, ft_model, tok, test_ds) -> dict:
    """The deliverable: base (best prompt) vs fine-tune, on the held-out set."""
    # TODO 4: score base with your BEST prompt (few-shot + format instructions),
    #   score fine-tune, return {"base": ..., "fine_tuned": ..., "delta": ...}.
    ...
```

The non-negotiables `eval.py` enforces:

- **Held-out only** — `eval.py` reads `test.jsonl`, never `train.jsonl`. Evaluating on training data measures memorization, not learning.
- **A strong baseline** — the base model is prompted *as well as you can* (few-shot, format instructions), because that's the alternative you'd ship if you didn't fine-tune. Beating a strawman prompt proves nothing.
- **The same test set for both** — apples to apples, or the comparison is meaningless.

---

## Deliverable 4 — `verdict.py` (the decision)

```python
def verdict(comparison: dict, training_cost_usd: float, delta_threshold: float = 0.10) -> str:
    """Render WORTH IT / NOT WORTH IT with a stated rule."""
    delta = comparison["fine_tuned"]["exact_match"] - comparison["base"]["exact_match"]
    worth = delta >= delta_threshold
    # TODO 5: return a one-paragraph verdict naming the delta, the cost, and the
    #   decision. A small delta for real cost+a-new-artifact-to-serve is NOT WORTH IT.
    ...
```

The point: the verdict is *rule-based and stated*, not a gut call. A +0.44 exact-match gain for $3 clears any reasonable threshold (worth it); a +0.03 gain for $3 *plus* a new artifact to version and serve does not (not worth it — ship the prompt). The threshold is a parameter you justify, not a vibe.

---

## Deliverable 5 — `cli.py` (data / train / eval / run)

```bash
# Each stage:
python -m crunchtune data  --raw data/raw_pairs.jsonl     # validate + split
python -m crunchtune train --train data/train.jsonl --out ./out/dsl-lora
python -m crunchtune eval  --base Qwen2.5-7B --adapter ./out/dsl-lora --test data/test.jsonl

# Or the whole pipeline + verdict:
python -m crunchtune run --task dsl --cost 3.00
```

`run` should print the promise marker:

```
                         exact_match   valid_dsl   avg_latency
base (best prompt)           0.42        0.71         1.9s
fine-tuned (LoRA)            0.86        0.98         1.9s
--------------------------------------------------------------
loss curve: LEARNING (smooth descent, flattened at ~0.34)
verdict: WORTH IT — +0.44 exact-match on held-out, valid-DSL 0.71->0.98,
         same latency, ~$3 training. Prompt ceiling 0.42 cleared. -> ship the adapter.
```

The point is that "should we fine-tune, and did it help?" is a command with a printed, defensible answer — including, when the numbers say so, a confident "no."

---

## Rules

- **You may** read the papers, the Unsloth/TRL docs, the lecture notes, and your exercise code.
- **You must not** evaluate on training data — `eval.py` reads the held-out `test.jsonl` only. This is the firewall against fooling yourself.
- **You must not** compare the fine-tune to a weak prompt — the base baseline uses your best prompt, because that's the real alternative.
- **You must** use QLoRA (4-bit base) + LoRA, with the adapter <1% of params — full fine-tuning a 7B doesn't fit the week's hardware and isn't the lesson.
- **You must** apply the chat template via `apply_chat_template` — hand-concatenation is the #1 silent failure.
- Python 3.12, `unsloth`, `trl`, `peft`, `transformers`, `datasets`, plus `pytest`. The GPU run needs a 24 GB card or a rented A10/L4 (~$3). The data/eval-metric tests run on CPU.
- Be **just as ready to ship a "NOT worth it" verdict** as a "worth it" one. A measured negative result is a passing deliverable.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-16-crunchtune-<yourhandle>`.
- [ ] `crunchtune data` produces `train.jsonl` + `test.jsonl` with every target validated, the chat template applied, and a tested no-leakage split.
- [ ] `crunchtune train` runs a **QLoRA + LoRA** SFT on Qwen2.5-7B, saves the adapter, prints the trainable-param count (<1%), and captures the loss curve.
- [ ] `crunchtune eval` scores **base (best prompt)** and **fine-tune** on the **held-out** set (exact-match, valid-DSL, latency).
- [ ] `crunchtune run` prints the full comparison + the loss-curve shape + a rule-based verdict (the promise marker).
- [ ] `pytest` passes, with at least:
  - `test_data.py`: validation rejects invalid/non-canonical targets; the split has no leakage.
  - `test_eval.py`: exact-match and valid-DSL metrics compute correctly on a tiny fixture.
- [ ] A `README.md` with the results table, the loss curve (or its shape), and the **one-page worth-it memo** (decision, table, mechanism, cost, honest verdict — positive or negative).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Dataset quality** | 20 | Every target validated + canonical; chat template applied via the tokenizer; de-duplicated; train/test split with no leakage (tested). |
| **Training correctness** | 20 | QLoRA (4-bit base) + LoRA with stated rank/alpha; adapter <1% of params; loss curve captured and its shape named. |
| **Honest eval** | 30 | Held-out test set only; **strong-prompt** base baseline (not a strawman); same set for both; exact-match + valid-DSL + latency reported. |
| **The verdict** | 20 | A rule-based worth-it/not decision with a stated threshold, the delta, and the cost — and the willingness to render an honest negative. |
| **Tests & hygiene** | 10 | `test_data` (validation + leakage), `test_eval` (metrics); `pytest` green; no checkpoints/`__pycache__`/`.venv` committed. |

**90+** is portfolio-grade and ready to be the Phase III milestone's fine-tune-or-not document. **70–89** works but has a soft eval (evaluated on training data, or a weak baseline) or a forked-from-full-fine-tuning train. **Below 70** means the verdict isn't trustworthy — fix the eval first, because an untrustworthy verdict is worse than no fine-tune.

---

## Stretch goals

- **The capstone-domain run.** Re-run the whole pipeline on a task from *your* capstone domain (or a candidate one), producing the actual Phase III fine-tune-or-not document. This is the milestone deliverable, done early.
- **A DPO leg.** Add `crunchtune dpo` that runs a preference pass on top of the SFT adapter, and measure whether it beat SFT alone. Report whether the preference-data work was worth it.
- **Quantize-and-serve verify.** Add a `crunchtune serve-eval` that merges the adapter, quantizes to GGUF, serves via Ollama, and re-runs `eval` against the served model — proving the fine-tuned behavior survives quantization.
- **CI.** A GitHub Actions workflow that runs the CPU-side tests (`test_data`, `test_eval`) on every push, so the dataset firewall and the eval metrics can't silently regress. (The GPU train stays manual — CI doesn't rent A10s.)

---

## How this connects to the rest of C23

- **Week 2 (tokens/templates)** gave you the chat template this pipeline applies; week 3 (prompt-as-code) is the *baseline* the fine-tune must beat.
- **Weeks 8 & 12 (eval discipline)** gave you the held-out-split, measure-honestly habit this pipeline applies to a generation task.
- **Week 17 (safety)** shares this week's spine — measurement-driven honesty: week 16 measures whether a fine-tune helped; week 17 measures whether a defense holds.
- **Week 18 (Phase III milestone)** requires a 1-page fine-tune-or-not decision document for the capstone's domain — *this pipeline produces exactly that*, repeatably.
- **Week 19 (vLLM serving)** is where a *worth-it* adapter gets served — your serving stack loads the LoRA you trained here.

When you've finished, push the repo and take the [quiz](../quiz.md).
