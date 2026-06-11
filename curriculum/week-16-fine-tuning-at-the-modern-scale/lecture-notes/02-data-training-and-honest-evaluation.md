# Lecture 2 — Data, Training, and Honest Evaluation: The Parts That Decide Whether the Fine-Tune Was Worth It

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can engineer an SFT dataset (instruction format, chat template, quality over quantity, a clean train/test split) and avoid its failure modes (leakage, format drift, length bias), run a LoRA training loop with Unsloth and read the loss curve (learning vs memorizing vs diverging), evaluate the fine-tune honestly against the base model on a held-out set, and place Unsloth/Axolotl/NeMo/TRL on the tooling map.

Lecture 1 was *whether* to fine-tune and *how* (PEFT). This lecture is *the parts that actually decide the outcome* — because the training loop is the easy bit, and the fine-tune is made or broken by the dataset that feeds it and the eval that judges it.

> **Garbage data in, confidently-wrong weights out.** A fine-tune doesn't tell you when it learned the wrong thing — it just gets confidently good at reproducing whatever you trained it on, including your mistakes. The dataset *is* the fine-tune. The eval is the only thing standing between "it worked" and "it looked like it worked."

---

## Part 1 — Dataset engineering: the part that decides everything

### 1.1 The instruction/response format

SFT trains on input→output demonstrations. The canonical shape is an instruction-tuning record:

```json
{"instruction": "List all contracts signed after 2024 in Delaware.",
 "input": "",
 "output": "SELECT * FROM contracts WHERE signed_year > 2024 AND state = 'DE';"}
```

For a chat model (Qwen2.5-7B-Instruct), this becomes a user turn (the instruction) and an assistant turn (the output). The training signal is: *given this user message, produce this assistant message.* The loss is computed on the **response tokens only** — you don't train the model to predict the user's input, you train it to produce the right output *given* the input. (Unsloth and TRL handle this masking for you, but know it's happening: training on the prompt tokens too would teach the model to generate *questions*, not answers.)

### 1.2 Apply the chat template — the #1 silent bug

This is the single most common way a fine-tune silently fails. A chat model expects its input in an *exact* format — specific role markers, special tokens, a precise layout — defined by its **chat template.** If you train on raw `"instruction\noutput"` strings instead of the templated format the model was instruction-tuned with, you've trained it on a distribution it will never see at inference, and your eval will be baffling.

```python
# Apply the model's OWN chat template — never hand-format the string.
def format_example(example, tokenizer):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    # This produces the EXACT token layout Qwen2.5-Instruct expects, with the
    # right special tokens (<|im_start|>, role markers, <|im_end|>). Getting this
    # wrong is the classic silent fine-tune failure: training succeeds, loss
    # drops, and inference is garbage because the format doesn't match.
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}
```

The rule: **always format with `apply_chat_template` using the target model's tokenizer.** If your fine-tune "trained fine but produces nonsense," check the template *first* — it's the bug nine times out of ten.

### 1.3 Quality over quantity — 500 clean beats 5,000 noisy

The instinct is "more data is better." For SFT on a narrow task, it's wrong past a point. **A few hundred clean, correct, consistent examples beat thousands of noisy ones.** Why: the model learns the *distribution* of your demonstrations, including their errors. If 10% of your 5,000 examples have a wrong DSL output, you've taught the model to be wrong 10% of the time — confidently. A curated 500 with near-zero label noise teaches a sharper, more correct distribution. This week's headline lab uses **500 examples** deliberately: enough to learn a narrow task, small enough to *curate by hand* so the label quality is real.

The corollary: **spend your time on data quality, not data volume.** Every hour cleaning your 500 examples buys more than an hour scraping 5,000 more. Check them by eye. Run the targets through your DSL parser to confirm they're all valid. Look for the duplicates, the near-duplicates, the ones where the input and output don't actually match.

### 1.4 The train/test split — the firewall

You will evaluate the fine-tune on a **held-out test set** — examples the model *never trained on*. This is the same firewall you built in weeks 8 and 12, and it's non-negotiable: a fine-tune evaluated on its training data tells you nothing except "the model can memorize," which it can. Split before you train, and never let test examples leak into training:

```python
from datasets import Dataset

ds = Dataset.from_list(examples).shuffle(seed=42)
split = ds.train_test_split(test_size=0.1, seed=42)   # 90/10, fixed seed
train_ds, test_ds = split["train"], split["test"]
# train_ds trains the adapter; test_ds is NEVER seen during training.
# The fixed seed makes the split reproducible — same examples held out every run.
```

For a 500-example set, a 90/10 split gives ~50 held-out examples — enough to estimate exact-match with meaningful resolution. The fixed seed matters: if the split changes every run, you can't compare two training runs fairly, because they were evaluated on different test sets.

### 1.5 The data failure modes

Three ways the data quietly sabotages the fine-tune, each worth a checklist item:

- **Leakage.** A test example (or a near-duplicate) sneaks into training, so the held-out score is inflated — the model "knew the answer." De-duplicate *across* the split, not just within it. The most insidious form is near-duplicates: the same query with a synonym swapped.
- **Format drift.** Inconsistent output formatting in the training data (sometimes `state = 'DE'`, sometimes `state="Delaware"`, sometimes `STATE = de`) teaches the model an incoherent target distribution, and it'll produce all three at random. **Normalize your outputs** to one canonical form before training.
- **Length / position bias.** If every training example is short, the model learns short; if the correct answer is always the first option in a list, the model learns "pick the first." Watch for accidental regularities in your data that aren't part of the real task. Shuffle, vary length, and check the distribution.

---

## Part 2 — The training run

With a clean dataset, the training loop is genuinely the easy part. Unsloth wraps TRL's `SFTTrainer` and handles the QLoRA setup, the masking, and the speed optimizations.

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,                          # the QLoRA-prepped model from Lecture 1 §4.3
    tokenizer=tokenizer,
    train_dataset=train_ds,               # the "text" field from apply_chat_template
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,    # effective batch size 8
        warmup_steps=5,
        num_train_epochs=2,               # 1–3 for a small set; more risks overfitting
        learning_rate=2e-4,               # LoRA tolerates higher LR than full FT
        logging_steps=1,                  # log loss every step so you can SEE the curve
        output_dir="./out/dsl-lora",
        seed=42,
    ),
)
trainer.train()
model.save_pretrained("./out/dsl-lora")   # saves the ADAPTER (small), not the base
```

The hyperparameters that matter, and how to set them with reasons:

- **Epochs (1–3).** How many times the model sees the dataset. For a small clean set, 1–3 is the range. Too few and it hasn't learned; too many and it *memorizes* (overfits) the training examples and generalizes worse — visible as a training loss that keeps dropping while held-out performance plateaus or declines. Start at 2.
- **Learning rate (~2e-4).** LoRA tolerates a higher learning rate than full fine-tuning because you're only moving a small adapter. 1e-4 to 3e-4 is the typical band. Too high and the loss diverges (spikes up); too low and it crawls.
- **Batch size (effective).** `per_device_batch_size × gradient_accumulation_steps`. Larger is more stable but uses more memory; gradient accumulation lets you simulate a big batch on a small GPU. 8–16 effective is fine for this scale.
- **Rank/alpha** — set in Lecture 1's `get_peft_model` (16/32 for this task).

### 2.1 Reading the loss curve

`logging_steps=1` exists so you can *watch the loss*, and reading it is a core skill. The training loss over steps tells you which of three things is happening:

- **Learning (the good shape):** loss starts high and decreases smoothly, then flattens as the model converges. A gentle downward curve that levels off is exactly what you want. The flattening means the model has extracted what it can from the data.
- **Memorizing / overfitting:** training loss keeps dropping toward zero, but if you tracked held-out (eval) loss it would *rise* — the model is memorizing training examples rather than learning the task. The fix is fewer epochs, more data, or a lower rank. *You only catch this with a held-out eval* — training loss alone looks great while the model overfits.
- **Diverging (the bad shape):** loss spikes upward or oscillates wildly. The learning rate is too high (or the data is broken). Lower the LR; if it persists, check the data and the chat template.

A useful instinct: a *too-perfect* training loss (smoothly to near-zero) on a tiny dataset is a memorization warning, not a victory. The held-out eval is what tells you the difference, which is the whole point of Part 3.

### 2.2 The CPU / mechanics path

You can run this *entire* loop on CPU with a tiny model (a 0.5B Qwen) for a handful of steps — not to get a useful fine-tune, but to see the loss move, confirm the format is right, and learn the mechanics before you rent a GPU. The headline run needs a 24 GB GPU (or rented A10/L4, ~$0.50–$1.00/h), but you should *mechanically* understand the pipeline on CPU first so you're not debugging the basics on rented time. Exercise 2 and 3 are CPU-runnable on purpose.

---

## Part 3 — Honest evaluation: the actual deliverable

The fine-tune is not the deliverable. **The verdict is** — and the verdict comes from comparing the fine-tune to the base model on a held-out set, with a metric, and concluding "worth it" or "not worth it" with a number.

### 3.1 The base-vs-fine-tune comparison

You evaluate *both* models on the *same* held-out test set: the base model (prompted as well as you can, since that's the baseline you're trying to beat) and the fine-tuned model. The comparison is the answer to "did the fine-tune help?"

```python
def evaluate(model, tokenizer, test_ds) -> dict:
    """Run the model over the held-out set; score against the targets."""
    exact, valid = 0, 0
    for ex in test_ds:
        out = generate(model, tokenizer, ex["instruction"])
        if out.strip() == ex["output"].strip():       # exact-match
            exact += 1
        if dsl_parses(out):                            # valid-DSL (parses ok)
            valid += 1
    n = len(test_ds)
    return {"exact_match": exact / n, "valid_dsl": valid / n}

base_score = evaluate(base_model, tok, test_ds)        # the baseline to beat
ft_score   = evaluate(ft_model, tok, test_ds)          # the fine-tune
```

For a task with a checkable answer (the DSL), use **exact-match** and a **structural validity** check (does it parse?). For open-ended outputs where there's no single right string, use an **LLM-as-judge** with the calibration discipline from week 12 (score against a few human labels first so the judge is trustworthy). The point is the same either way: *a number for the base, a number for the fine-tune, on data neither saw in training.*

### 3.2 The verdict — worth it or not

The deliverable is a decision with the numbers behind it. A fine-tune is "worth it" if it cleared the prompt ceiling by a margin that justifies its cost:

```
                         exact_match   valid_dsl   avg_latency
base (best prompt)           0.42        0.71         1.9s
fine-tuned (LoRA)            0.86        0.98         1.9s
--------------------------------------------------------------
verdict: WORTH IT — +0.44 exact-match, valid-DSL 0.71 -> 0.98, same latency,
         ~$3 training cost. Prompt ceiling was 0.42; fine-tune cleared it.
```

But the honest verdict is sometimes the *other* way, and a good engineer is just as proud of it:

```
verdict: NOT WORTH IT — +0.03 exact-match for $3 + a new artifact to version
         and serve. The prompt ceiling (0.83) was already near the fine-tune
         (0.86). Ship the prompt; skip the fine-tune.
```

That negative result is a *real engineering outcome* and the most common true one. The whole "was it worth it?" promise of the week is making this a measured decision: a fine-tune that helped is a number, a fine-tune that didn't is also a number, and "we didn't fine-tune, here's the number that says we shouldn't" is a deliverable to be proud of — it saved the cost of a wrong call.

### 3.3 Watch for regression

One more honesty check: the fine-tune might be great at the narrow task and *worse* at things you didn't measure (Lecture 1 §3's regression risk). If the model will do anything beyond the narrow task, spot-check a few general queries before and after. A fine-tune that hits 0.86 on the DSL but can no longer hold a normal conversation has *traded* capability — fine if you meant to, a problem if you didn't notice. The held-out eval measures the win; a regression spot-check measures the cost.

---

## Part 4 — The tooling roster

Where the libraries fit, so you reach for the right one:

| Tool | Role | Reach for it when |
|---|---|---|
| **Unsloth** | Single-GPU LoRA/QLoRA, 2× faster, low memory | You're on one GPU and want the friendliest fast path. **This week's choice.** |
| **Axolotl** | Config-file-driven multi-GPU SFT/DPO | You've outgrown one GPU; you want reproducible YAML configs; production-credible. |
| **NeMo Framework** | Serious training at scale (multi-node, full FT) | You have a cluster and need full fine-tuning or very large models (the NVIDIA-stack week). |
| **TRL (HuggingFace)** | The `SFTTrainer`/`DPOTrainer` building blocks | You want to understand (or customize) what Unsloth/Axolotl wrap. |

The progression is real: you *start* on Unsloth (single GPU, this week), graduate to **Axolotl** when you need multi-GPU and config-driven reproducibility, and reach **NeMo** only when you have a cluster and a full-fine-tuning or scale need. **TRL** is the layer underneath all of them — Unsloth and Axolotl both ultimately call TRL trainers, so understanding `SFTTrainer` is understanding the engine. Don't reach for NeMo to fine-tune a 7B LoRA; that's a cluster tool for a single-GPU job. Match the tool to the scale.

---

## Part 4.5 — A worked dataset, end to end

Let's walk one example all the way through the pipeline, because the abstract steps hide where the work actually is. Take the NL→DSL task: "List all contracts signed after 2024 in Delaware" → `SELECT * FROM contracts WHERE signed_year > 2024 AND state = 'DE';`.

**Generation.** You produce 500 such pairs. For a synthetic DSL task you can *generate* them programmatically (parameterize the year, the state, the query shape), which is the cheapest way to get clean, consistent data — and it sidesteps label noise because the target is constructed correctly by definition. For a *real* domain task you'd collect demonstrations from logs, human annotators, or a frontier model's outputs (distillation), and then the label-quality work is real.

**Validation.** Every generated target goes through a check: does it *parse* as valid DSL, and is it in your *canonical* form? This is where format drift gets caught before it poisons training. If your generator sometimes emits `state = 'DE'` and sometimes `state = "Delaware"`, the validator rejects (or normalizes) the non-canonical one. Run *every* example through the DSL parser; an unparseable target is a training example that teaches the model to produce unparseable output.

```python
# Validation gate: nothing trains unless it parses AND is canonical.
clean = [ex for ex in raw if dsl_parses(ex["output"]) and is_canonical(ex["output"])]
```

**Templating.** Each clean pair becomes a chat-templated string via `apply_chat_template` (§1.2). This is mechanical but non-negotiable — the templated form is what the model trains on, and it must match the model's instruction format exactly.

**De-duplication and split.** De-dup by instruction (and by near-duplicate, to catch synonym-swapped repeats), then `train_test_split(test_size=0.1, seed=42)`. The 50 held-out examples are now sealed; they will judge the fine-tune and they are never trained on.

The thing to notice: of the four steps, *generation* and *validation* are where the quality lives, and they're where you should spend your time. Templating and splitting are mechanical. A team that rushes generation (noisy targets) and skips validation (format drift, unparseable outputs) will train a fine-tune that confidently reproduces its own garbage — and the held-out eval will be baffling, because the *eval* data has the same problems as the train data. Clean data in is the whole game; this worked example is where you make it clean.

---

## Part 4.6 — Why a single-GPU LoRA run is fast and cheap

It's worth dwelling on *why* this week's fine-tune is a ~15-minute, ~$3 operation when "fine-tuning a 7B" sounds expensive — because the intuition matters when you're estimating the cost of a real fine-tune.

The cost of a training run scales with: (number of examples) × (number of epochs) × (tokens per example) × (cost per token processed). For this week's task:

- **500 examples × 2 epochs = 1,000 example-passes.** That's tiny. A pretraining run sees *trillions* of tokens; you're seeing a few hundred thousand.
- **Short examples.** An NL→DSL pair is maybe 50–100 tokens. 1,000 passes × ~80 tokens ≈ 80K tokens of *training* signal — a rounding error compared to anything at scale.
- **QLoRA keeps it on one cheap GPU.** No multi-node coordination, no gradient-sync-across-GPUs overhead. An A10/L4 at ~$0.50–$1.00/h, for ~15–30 minutes, is a few dollars.

The lesson generalizes: **a narrow SFT fine-tune on a curated dataset is a small, cheap operation** — minutes and dollars, not days and thousands. This is *also* an argument the decision ladder uses: if the fine-tune is cheap, why is it last-resort? Because the *training* is cheap but the *artifact lifecycle* (Lecture 1 §8) is not, and because the cheap-to-train fine-tune still has to *beat the free prompt*. A $3 training run that doesn't clear the prompt ceiling cost you $3 *plus* a new artifact to serve forever — which is why even a cheap fine-tune needs the held-out verdict before it ships.

The flip side: because it's cheap, you can *afford to run the experiment and find out.* You don't have to agonize over whether to fine-tune — you build the dataset, spend $3, run the eval, and let the number decide. The cheapness is what makes the "measure, don't guess" discipline practical: the measurement *is* a fine-tune run, and a fine-tune run is cheap. Use that.

---

## Part 4.7 — Choosing the metric for the eval

The eval (Part 3) compares base and fine-tune on a held-out set — but *with which metric?* The choice depends entirely on whether your task has a checkable answer, and getting it right is what makes the verdict trustworthy.

**Exact-match — when there's one right string.** For NL→DSL, the target is a specific query; the generated output either equals it or it doesn't. Exact-match (`output.strip() == target.strip()`) is the cleanest, most honest metric there is: no judgment, no fuzziness, fully automatable. Use it whenever the task has a canonical correct output. (Normalize whitespace and trivial formatting before comparing, or you'll count `state = 'DE'` and `state= 'DE'` as different — which is a *format* problem your dataset's canonicalization should already have prevented.)

**Structural validity — a useful companion to exact-match.** Even when the output isn't an exact match, "does it parse?" is informative. A fine-tune might produce a *valid* query that differs from the target (a different-but-correct phrasing), which exact-match counts as a miss but valid-DSL counts as a partial win. Reporting *both* exact-match and valid-DSL tells a richer story: a model at exact-match 0.86 / valid-DSL 0.98 is producing valid queries that are *almost always* exactly right; one at 0.86 / 0.88 is producing the right query when it's valid but often produces invalid output. The gap between the two metrics diagnoses *what kind* of error the model makes.

**LLM-as-judge — when there's no single right answer.** For open-ended outputs (a summary, a rewrite, a generated description) where many outputs are acceptable, there's no string to match. Here you use an LLM judge: "given the input and this output, is it correct/on-brand/helpful? Score 0–1." The catch — and it's the week-12 lesson — is that an *un-calibrated* judge is itself unreliable. You **calibrate** it against a handful of human-labeled examples first: score those 10 examples by hand, run the judge on them, confirm the judge agrees with the humans, and only then trust the judge on the rest. An LLM judge you didn't calibrate is just another model whose output you're trusting blindly — which is exactly the thing this week is teaching you not to do.

The meta-lesson: **prefer the most automatable metric your task allows.** Exact-match > structural validity > calibrated LLM-judge, in order of how little human judgment they require. A task you can measure with exact-match gives you a verdict you can re-run on every change with zero human cost — which is why the headline lab deliberately chose a DSL task (checkable) over an open-ended one (judge-dependent). When you face a real task, the first question for the eval is "what's the most automatable metric that honestly captures success?" — because that metric is the one you can afford to run often enough to trust.

---

## Part 4.8 — Common training questions, answered

Questions that come up the first time you run a real fine-tune:

**"My loss went to near-zero in one epoch — great, right?"** Probably not. On a small dataset, a loss diving to zero is a *memorization* warning, not a victory — the model is memorizing your examples rather than learning the task. The training loss alone can't tell you; only the held-out eval can. If train loss is ~0 but held-out exact-match is mediocre, you overfit — fewer epochs, more data, or lower rank (§2.1).

**"My fine-tune trained fine but generates garbage. What's wrong?"** Check the chat template first (§1.2). If you trained on hand-concatenated strings instead of `apply_chat_template` output, the model learned a format it never sees at inference. This is the #1 silent failure, and it presents *exactly* as "training looked fine, inference is nonsense." Fix the template before debugging anything else.

**"How long should the fine-tune take?"** For 500 short examples at 2 epochs on a 7B with QLoRA, ~15–30 minutes on an A10/L4. If it's taking *hours*, check your sequence length and batch size — you may be padding to a huge max length or running too many epochs. A narrow SFT run is a minutes-scale operation (Part 4.6).

**"How many epochs?"** Start at 2. 1 is often too few (under-learned); 3+ risks overfitting on a small set. Watch the held-out eval across epoch counts if you're unsure — the right number is the one where held-out performance peaks, not where training loss bottoms out (§2).

**"Can I trust an LLM-as-judge for the eval?"** Only if you *calibrate* it first against a few human labels (Part 4.7). An un-calibrated judge is just another model whose output you're trusting blindly — the exact thing this week teaches you not to do. For checkable tasks (the DSL), prefer exact-match and skip the judge entirely; it's more honest and fully automatable.

---

## Part 4.9 — What the worth-it memo looks like

The deliverable of the whole week is a one-page "was it worth it?" memo (the homework's headline), and seeing its shape makes the eval discipline concrete. Here's the skeleton, filled with example numbers:

> **Fine-tune decision — NL→CONTRACTQL on Qwen2.5-7B**
>
> **Decision.** Ship the fine-tuned adapter — it cleared the prompt ceiling on the held-out set (0.42 → 0.86 exact-match).
>
> **The table** (50 held-out examples, never trained on):
> | | exact_match | valid_dsl | avg_latency |
> |---|---|---|---|
> | base (best prompt, few-shot) | 0.42 | 0.71 | 1.9s |
> | fine-tuned (LoRA r=16) | 0.86 | 0.98 | 1.9s |
>
> **Why this winner, on this task.** The base model, even with few-shot examples, couldn't reliably hit the *canonical* DSL form — it produced valid-but-non-canonical queries 29% of the time (valid_dsl 0.71 means 29% didn't even parse). The 500 demonstrations taught the exact format; valid_dsl rose to 0.98 and exact-match more than doubled. It's a format/style problem (legitimate reason #1), and demonstrations are how you teach a format.
>
> **The trade-off accepted.** A new artifact (the LoRA adapter) to version against its base and serve via vLLM, plus ~$3 training cost. Latency is unchanged (LoRA adds negligible inference cost).
>
> **Loss curve.** LEARNING shape — smooth descent flattening at ~0.34; no overfitting signal (held-out tracked train).
>
> **One per-query trace.** `q: "contracts after 2024 in Delaware" → base: SELECT * FROM contracts WHERE year>2024 (non-canonical, invalid) / ft: SELECT * FROM contracts WHERE signed_year > 2024 AND state = 'DE'; ✓`

Notice every claim is a number or a trace — nothing is "it felt better." And the *negative* version of this memo is just as valid: same structure, but the table shows base 0.83 / fine-tune 0.86, the decision is "ship the prompt, skip the fine-tune," and the reasoning is "+0.03 doesn't justify the artifact lifecycle." Both memos are passing deliverables; the discipline is the *measured verdict*, whichever way it falls. This is the artifact the Phase III milestone wants (a fine-tune-or-not decision for the capstone domain), and it's what separates an engineer who *decided* from one who *hoped*.

---

## Part 4.95 — The whole loop in one breath

Step all the way back and see the week as one loop, because the pieces only matter as a sequence:

1. **Decide** (Lecture 1) — climb the ladder, confirm fine-tuning is the right rung, name the legitimate reason, commit to the eval.
2. **Build the data** (Part 1) — generate, validate, chat-template, de-dup, split. The dataset *is* the fine-tune; this is where the quality lives.
3. **Train** (Part 2) — QLoRA + LoRA, sane hyperparameters, read the loss curve. The easy part.
4. **Evaluate** (Part 3) — base (best prompt) vs fine-tune, held-out set, the most automatable honest metric. The part that decides everything.
5. **Verdict** (Part 4.9) — worth it or not, with the numbers. Ship the adapter, or ship the prompt — either is a valid, evidenced decision.

The whole loop is *cheap* (Part 4.6) — a few dollars and an afternoon — which is what makes "measure, don't guess" practical: you can afford to run the experiment and let the held-out number decide, rather than agonizing in the abstract. And the loop is *honest* only if every step holds: garbage data (step 2) poisons the eval (step 4); a weak baseline or a train-data eval (step 4) inflates the verdict (step 5); skipping the decision (step 1) means you fine-tuned a problem retrieval would have solved. The senior engineer runs the whole loop, trusts the number it produces, and is equally proud of "we fine-tuned and it cleared the ceiling" and "we measured and the prompt was already enough." That equanimity — caring about the *answer*, not about *having fine-tuned* — is the mark of someone who treats fine-tuning as the engineering tool it is, not as a trophy.

One closing caution that ties the loop together: **the eval is the load-bearing step, and it's the one people rush.** Everyone enjoys the training run — watching the loss drop feels like progress. Far fewer build a rigorous held-out eval with a strong baseline, because it's less fun and it might deliver bad news ("the fine-tune didn't help"). But the eval is the *only* step that produces the verdict, and a fine-tune without an honest verdict is just a hopeful artifact you can't defend. If you have limited time, spend it on the data (step 2) and the eval (step 4), not on hyperparameter-twiddling the training run (step 3) — those two steps are where the trustworthiness of the whole decision lives. The training run is the easy 20 minutes; the data and the eval are the engineering.

---

## Part 5 — Recap

You should now be able to:

- **Engineer an SFT dataset**: the instruction→response format, applying the model's **chat template** (the #1 silent-failure bug to check first), **quality over quantity** (500 clean beats 5,000 noisy), and a **train/test split** with a fixed seed as the firewall against memorization.
- **Avoid the data failure modes**: leakage (de-dup across the split), format drift (normalize outputs to one canonical form), and length/position bias (watch for accidental regularities).
- **Run a LoRA training loop** with Unsloth/TRL, set epochs/learning-rate/batch-size with reasons, and **read the loss curve** — learning (smooth descent that flattens), memorizing (training loss → 0 while held-out plateaus), diverging (spikes; LR too high).
- **Evaluate honestly**: compare base (best prompt) vs fine-tune on a **held-out set**, with exact-match + structural validity for checkable tasks or a calibrated LLM-as-judge for open ones, and produce a **worth-it-or-not verdict with the numbers** — being just as ready to ship a clean negative result.
- **Place the tooling**: Unsloth (single-GPU, this week), Axolotl (multi-GPU, production), NeMo (cluster scale), TRL (the building blocks underneath).

Next: the exercises put this on a real task — reason through the fine-tune-or-not decision on three scenarios, build and validate a 500-example SFT dataset, and compute the LoRA memory budget and read a loss curve. Continue to [the exercises](../exercises/README.md).

---

## References

- *Unsloth docs (the SFT/LoRA training path)*: <https://docs.unsloth.ai/>
- *TRL `SFTTrainer`*: <https://huggingface.co/docs/trl/sft_trainer>
- *Hugging Face chat templating (`apply_chat_template`)*: <https://huggingface.co/docs/transformers/chat_templating>
- *Hugging Face `datasets` (train_test_split)*: <https://huggingface.co/docs/datasets/process#split>
- *QLoRA (the 4-bit base that fits the 7B in 24 GB)*: <https://arxiv.org/abs/2305.14314>
- *Axolotl (multi-GPU, config-driven)*: <https://github.com/axolotl-ai-cloud/axolotl>
- *NVIDIA NeMo Framework (cluster-scale training)*: <https://github.com/NVIDIA/NeMo>
- *Qwen2.5-7B-Instruct model card*: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
