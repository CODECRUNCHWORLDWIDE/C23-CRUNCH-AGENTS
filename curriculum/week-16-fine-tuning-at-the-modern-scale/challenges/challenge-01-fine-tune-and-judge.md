# Challenge 1 — Fine-Tune and Judge

**Time estimate:** ~150 minutes (plus ~10–20 minutes of GPU training time). Needs a 24 GB GPU (RTX 3090/4090) or a rented A10/L4 at ~$0.50–$1.00/h — budget **~$3 of compute**.

## Problem statement

You've decided (Exercise 1's discipline) that a narrow NL→DSL task is a legitimate fine-tune: it's an output-format problem (legitimate reason #1), and a well-prompted base model tops out below your target. Now you do the real thing. You'll SFT a LoRA adapter on **Qwen2.5-7B-Instruct** with **Unsloth**, on the 500-example dataset you built in Exercise 2, and then — the part that's the actual deliverable — you'll evaluate the fine-tuned model *against the base model* on a held-out test set, and render a verdict: was it worth it?

The training run is the easy 20 minutes. The judgment is the challenge. You will produce a number that says "the prompt ceiling was X, the fine-tune cleared it to Y" — or, just as validly, "the fine-tune barely moved the number, ship the prompt." Either way, you decide *with evidence*.

## What you build

A training + eval pipeline and a verdict:

1. **`train.py`** — load `Qwen2.5-7B-Instruct` in 4-bit (QLoRA), attach a LoRA adapter (rank 16, alpha 32), train on `train.jsonl` for 1–3 epochs with `SFTTrainer`, save the adapter to `./out/dsl-lora`. Log loss every step so you can read the curve.
2. **`eval.py`** — run *both* the base model (best prompt you can write) and the fine-tuned model over `test.jsonl` (the held-out 50). Score **exact-match** and **valid-DSL** (does it parse?), and measure latency. Print the comparison table.
3. **`fine-tune-memo.md`** — the verdict: the decision, the table, why it landed where it did, the cost, and the honest "worth it / not worth it."

## The harness approach

The whole pipeline reduces to: QLoRA-load the base, attach a LoRA, train, then eval *both models on the same held-out set*.

```python
# train.py — the QLoRA + LoRA + SFT path (Unsloth)
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

ds = load_dataset("json", data_files="train.jsonl")["train"]
ds = ds.map(lambda e: {"text": tok.apply_chat_template(
    [{"role":"user","content":e["instruction"]},
     {"role":"assistant","content":e["output"]}], tokenize=False)})

SFTTrainer(model=model, tokenizer=tok, train_dataset=ds,
           args=SFTConfig(num_train_epochs=2, learning_rate=2e-4,
                          per_device_train_batch_size=2, gradient_accumulation_steps=4,
                          logging_steps=1, output_dir="./out/dsl-lora", seed=42)).train()
model.save_pretrained("./out/dsl-lora")   # saves the ADAPTER, ~tens of MB
```

```python
# eval.py — the part that IS the deliverable: base vs fine-tune, held-out
def score(model, tok, test):
    exact = valid = 0
    for ex in test:
        out = generate(model, tok, ex["instruction"]).strip()
        exact += (out == ex["output"].strip())
        valid += dsl_parses(out)
    return {"exact_match": exact/len(test), "valid_dsl": valid/len(test)}

base = score(base_model, tok, test)        # best-prompt baseline — the ceiling to beat
ft   = score(ft_model,   tok, test)        # the fine-tune
```

Run order: `train.py` (rent the GPU, ~$3, ~15 min) → `eval.py` (compare) → write the memo. The base model's score is the ceiling you're trying to beat; the fine-tune's score is whether you beat it.

## Acceptance criteria

- [ ] A `challenge-01/` directory with `train.py`, `eval.py`, and `fine-tune-memo.md`, all runnable.
- [ ] The fine-tune uses **QLoRA** (4-bit base) + **LoRA** (rank/alpha set with a stated reason) — trainable params are <1% of the model (print the count).
- [ ] Training uses the **chat template** (`apply_chat_template`) — not hand-concatenated strings — and the **held-out `test.jsonl`** is never seen in training.
- [ ] `eval.py` scores **both** base (best prompt) and fine-tune on the **same held-out set**, reporting exact-match, valid-DSL, and latency.
- [ ] The **loss curve** is captured (the per-step log) and you state which shape it is (learning / memorizing / diverging) in the memo.
- [ ] A one-page `fine-tune-memo.md` with: the decision (one sentence), the base-vs-fine-tune table, *why* it landed there (the mechanism), the cost (~$, training time, the new artifact), and the honest **worth-it / not-worth-it** verdict.
- [ ] At least one **promise-format result** showing the held-out comparison:
  `base exact=0.42 / ft exact=0.86 / +0.44 / verdict: WORTH IT ✓` (or the honest negative).

## The trap (read after a first attempt)

The trap is **evaluating on the training data.** It is *so* tempting to run the fine-tuned model over a few examples it trained on, see it nail them, and declare victory. That measures *memorization*, which a fine-tune can always do — it tells you nothing about whether the model learned the *task*. The only honest measurement is on the **held-out test set** the model never saw. If your `eval.py` reads `train.jsonl` instead of `test.jsonl`, or if a test example leaked into training (Exercise 2's leakage check exists for this), your "0.95 exact-match!" is a lie. Evaluate on held-out data, or don't claim a number.

A second, subtler trap: **comparing the fine-tune to a weak prompt.** If your "base model" baseline uses a lazy one-line prompt, the fine-tune will look amazing by comparison — but you haven't beaten the *real* alternative, which is the base model with a *good* prompt (few-shot, format instructions, the works). The honest baseline is the best prompt you can write, because that's the thing you'd ship if you *didn't* fine-tune. Beat the strong baseline, or your "+0.44" is measuring the gap between a fine-tune and a strawman.

## Stretch goals

- **The negative-result version.** Pick a *different* task where the base model already does well with a strong prompt (high prompt ceiling), fine-tune anyway, and write the honest "NOT worth it" memo with the small delta. A clean negative result is a real engineering artifact — and practicing it makes you trust your own positive results more.
- **A DPO pass.** On top of your SFT adapter, build 50 preference pairs (a preferred DSL style vs a clunky-but-valid one) and run a `DPOTrainer` pass. Measure whether DPO moved the eval beyond SFT — and whether the extra preference-data work was worth it.
- **Quantize and serve.** Merge the adapter into the base, quantize to GGUF (week 6), and serve via Ollama. Confirm your fine-tuned behavior *survives* quantization (re-run `eval.py` against the served model) and chart the quality/size trade-off.
- **The regression check.** Run a handful of *general* queries (unrelated to the DSL) through base and fine-tune. Did the fine-tune degrade general capability (catastrophic forgetting)? Report it — a fine-tune that wins the narrow task but loses general ability is a *trade*, and the memo should name it.

## Why this matters

The Phase III milestone (end of week 18) requires a **1-page fine-tune-or-not decision document for the capstone's domain.** This challenge *is* that document, rehearsed: you built the dataset, ran the adapter, measured against the baseline, and rendered a verdict you can defend. Every team you'll ever work on has someone who wants to fine-tune; the engineer who can run the experiment *and honestly judge it* — including saying "no, the prompt is fine" with a number — is the one whose team doesn't waste a sprint and a serving slot on a fine-tune that never beat the prompt. The fine-tune cleared the ceiling, or it didn't, and you can prove which.
