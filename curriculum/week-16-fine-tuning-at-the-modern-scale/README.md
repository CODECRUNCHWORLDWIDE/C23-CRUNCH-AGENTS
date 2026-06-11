# Week 16 — Fine-Tuning at the Modern Scale

Welcome to the week you learn when *not* to do the thing this week is named after. Fine-tuning is the most over-reached-for tool in the applied-LLM toolbox: the first instinct of an engineer whose prompt isn't working is "let's fine-tune," and nine times out of ten the right move was a better prompt, better retrieval, or a better eval. This week teaches you to fine-tune *well* — LoRA, QLoRA, SFT, the data engineering, the training loop, the honest eval — precisely so you can recognize the one time in ten it's the right call and execute it competently, and the nine times it isn't and say so with a number.

This is week 4 of **Phase III — Agents & Orchestration**, and it's a deliberate sidestep off the orchestration track. Weeks 13–15 stacked frameworks, graphs, and tool protocols on top of *frozen* models. This week asks the other question: when the frozen model isn't enough — when you need a specific output style, a domain vocabulary it doesn't know, or a latency profile a 7B can hit that a frontier model can't — do you change the *weights*? The headline lab is a real fine-tune: a 500-example SFT dataset for a narrow domain (natural language → a custom DSL), a LoRA adapter trained on **Qwen2.5-7B** with **Unsloth** on a single 24 GB GPU (or a rented A10), and a held-out eval against the base model that tells you, with evidence, whether the fine-tune was worth it.

The one sentence to internalize before you read another line:

> **Fine-tuning is a debugging tool of last resort. The model is rarely the problem — the prompt, the retrieval, or the eval is.**

Here's why that's not false modesty about a powerful technique. Fine-tuning is expensive (compute, data engineering, eval infrastructure), it's slow to iterate (a training run is minutes-to-hours, a prompt edit is seconds), it bakes your data into weights you then have to version and serve, and it can *regress* capabilities you didn't measure. A prompt change is reversible in one commit; a fine-tune is a new artifact with its own lifecycle. So the discipline of this week is the same as every other week in C23: **measure first.** Before you fine-tune, you prove that prompt-and-retrieve hit a ceiling. After you fine-tune, you prove — on a held-out set — that the new weights cleared that ceiling. No measurement, no fine-tune.

There's a corollary worth taping to your GPU:

> **A fine-tune is only as good as the eval that judges it and the data that trained it.** Garbage data in, confidently-wrong weights out — and you won't know unless your held-out eval is honest.

## Learning objectives

By the end of this week, you will be able to:

- **Decide** whether to fine-tune at all — the "try prompt, then retrieve, then fine-tune" ladder — and articulate the three legitimate reasons (output style, domain vocabulary, latency/cost) versus the many illegitimate ones.
- **Explain** parameter-efficient fine-tuning — full fine-tuning vs **LoRA** (low-rank adapters) vs **QLoRA** (4-bit base + LoRA) vs **DoRA** — and the memory math that makes a 7B trainable on a 24 GB GPU.
- **Distinguish** the post-training objectives — **SFT** (supervised fine-tuning on demonstrations), **DPO**/**ORPO**/**KTO** (preference alignment) — and know which one your problem actually needs (almost always SFT).
- **Engineer** an SFT dataset: the instruction/response format, the chat template, quality-over-quantity, train/test split discipline, and the failure modes (leakage, format drift, length bias).
- **Train** a LoRA adapter on **Qwen2.5-7B** with **Unsloth** on a single GPU, choosing rank, alpha, learning rate, and epochs with reasons, and reading the loss curve.
- **Evaluate** a fine-tune honestly against the base model on a held-out test set — exact-match for a DSL, an LLM-as-judge for open outputs — and produce a cost/benefit verdict.
- **Survey** the production tooling — **Unsloth** (single-GPU friendly), **Axolotl** (multi-GPU, production), **NeMo Framework** (scale), **TRL** (building blocks) — and **RLHF/RLAIF** at the depth of "know what they are, know you won't run one this week."
- **Build** the `crunchtune` pipeline: dataset prep → LoRA training → held-out eval → a worth-it-or-not decision document.

## Prerequisites

This week assumes you have completed **C23 weeks 1–15**, or have equivalent fluency. Specifically:

- You remember **week 2** (tokens, the tokenizer, the chat template) and **week 3** (the prompt is code; you exhausted the prompt before reaching for weights). Fine-tuning operates on tokenized, templated data — those weeks are the substrate.
- You can read an **eval harness**: a gold set, a metric, a held-out split. Weeks 8 and 12 drilled this; this week reuses the discipline on a *generation* task (does the output match the target?) rather than a retrieval one.
- Python 3.12 on Linux or WSL2 (Unsloth's fast path is CUDA-Linux; macOS/MLX is a documented secondary path); a virtualenv you can `pip install` into.
- Comfort renting a GPU. The headline lab needs a **24 GB GPU** (RTX 3090/4090) or a rented **A10/L4** at ~$0.50–$1.00/h. The full week's training fits in **~$5 of rented compute**. A CPU-only path (a tiny model, a few steps, just to see the loss move) is documented so you can complete the *mechanics* with no GPU, then rent for the real run.

You do **not** need C5 (Data Science), though it helps — this week re-derives the fine-tuning parts that matter from a systems perspective (what a LoRA adapter *is*, why QLoRA fits in 24 GB, what the loss curve means) without assuming you've trained a transformer from scratch.

## Topics covered

- **When not to fine-tune:** the prompt → retrieve → fine-tune ladder, the three legitimate triggers (style, vocabulary, latency), and the cost of a fine-tune (data, compute, versioning, regression risk) that makes it last-resort.
- **PEFT — the memory story:** full fine-tuning's prohibitive memory, **LoRA** (freeze the base, train low-rank update matrices), **QLoRA** (quantize the frozen base to 4-bit, train LoRA on top — the 24 GB enabler), **DoRA** as a refinement; rank/alpha as the capacity knobs.
- **Post-training objectives:** **SFT** (learn from input→output demonstrations — what you'll do), **DPO/ORPO/KTO** (learn from preference pairs — survey), and the honest note that **RLHF/RLAIF** are the alignment lineage you'll read about but not run.
- **Dataset engineering:** the instruction format, applying the chat template, quality > quantity (500 clean examples beat 5,000 noisy ones), the train/test split, and the data failure modes — leakage, format drift, length/position bias.
- **The training run:** Unsloth's `FastLanguageModel`, choosing LoRA rank/alpha/dropout, learning rate and epochs, the loss curve (what "learning" vs "memorizing" vs "diverging" looks like), and saving the adapter.
- **Honest evaluation:** held-out test set, base-vs-fine-tune comparison, exact-match for a DSL / LLM-as-judge for open text, and the cost/benefit verdict that's the actual deliverable.
- **The tooling roster:** Unsloth, Axolotl, NeMo Framework, TRL — what each is for and when to reach for it.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                          | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|---------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | When NOT to fine-tune; the ladder; PEFT and the memory math   |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | SFT vs DPO; dataset engineering; the dataset exercises        |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | The training run (Unsloth/LoRA); the loss curve; tooling      |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Honest eval (base vs fine-tune); building the pipeline        |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The held-out eval + worth-it memo; training clinic            |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work (train + eval)                         |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                     |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                               | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The PEFT/LoRA/QLoRA papers, the Unsloth/Axolotl/TRL docs, the dataset-format references, and the model cards |
| [lecture-notes/01-when-to-fine-tune-and-peft.md](./lecture-notes/01-when-to-fine-tune-and-peft.md) | When not to fine-tune, the decision ladder, PEFT (LoRA/QLoRA/DoRA), the memory math, and SFT vs preference objectives |
| [lecture-notes/02-data-training-and-honest-evaluation.md](./lecture-notes/02-data-training-and-honest-evaluation.md) | Dataset engineering, the Unsloth/LoRA training run, the loss curve, honest base-vs-fine-tune eval, and the tooling roster |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-decide-to-fine-tune.md](./exercises/exercise-01-decide-to-fine-tune.md) | Walk three scenarios through the prompt→retrieve→fine-tune ladder and justify each decision |
| [exercises/exercise-02-build-an-sft-dataset.py](./exercises/exercise-02-build-an-sft-dataset.py) | Generate, format, and validate a 500-example SFT dataset with a clean train/test split |
| [exercises/exercise-03-lora-memory-and-loss.py](./exercises/exercise-03-lora-memory-and-loss.py) | Compute the LoRA/QLoRA memory budget and simulate the loss curve so you can read a real one |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-fine-tune-and-judge.md](./challenges/challenge-01-fine-tune-and-judge.md) | The full fine-tune: SFT a Qwen2.5-7B LoRA on a DSL task, eval against base on a held-out set, decide if it was worth it |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the one-page fine-tune-or-not decision memo |
| [mini-project/README.md](./mini-project/README.md) | The `crunchtune` pipeline — dataset prep, LoRA training, held-out eval, and the worth-it verdict |

## The "was it worth it?" promise

C23 uses a recurring marker for every exercise that ends in a fine-tune *measured honestly against its baseline*:

```
$ python -m crunchtune eval --base Qwen2.5-7B --adapter ./out/dsl-lora --test test.jsonl
                         exact_match   valid_dsl   avg_latency
base (prompted)              0.42        0.71         1.9s
fine-tuned (LoRA)            0.86        0.98         1.9s
--------------------------------------------------------------
verdict: WORTH IT — +0.44 exact-match on the held-out set, valid-DSL near 1.0,
         same latency, ~$3 of training. The base model's prompt ceiling was 0.42;
         the fine-tune cleared it. See memo for the decision.
```

If that fine-tune had landed at +0.03 exact-match for $3 and a new artifact to version and serve, the honest verdict is **NOT worth it** — go back to the prompt. The point of week 16 is to make "should we fine-tune?" a *measured decision with a held-out number*, not a reflex — and to be just as proud of a "we didn't fine-tune, here's why" memo as of a successful training run.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **QLoRA paper** (Dettmers et al., 2023) until you can explain why a 4-bit-quantized frozen base plus a small LoRA adapter trains in a fraction of the memory of full fine-tuning, with negligible quality loss: <https://arxiv.org/abs/2305.14314>. Then re-run your training with and without 4-bit and compare VRAM and final loss.
- Run a **DPO** pass on top of your SFT adapter: build 50 preference pairs (preferred vs rejected DSL outputs) and align the model toward the preferred style. Measure whether DPO moved your eval beyond what SFT alone achieved — and whether it was worth the extra data work.
- **Quantize the merged model** to GGUF (week 6's territory) and serve it via Ollama. Confirm your fine-tuned behavior survives quantization, and chart the quality/size trade-off.
- **Build the negative result deliberately.** Take a task where the base model already does well with a good prompt, fine-tune anyway, and produce the honest memo showing the fine-tune *didn't* help. A clean negative result is a real engineering artifact — and the most common true outcome.

## Up next

Week 17 returns to the agent you built in week 15 and *attacks* it: prompt injection, jailbreaks, output filtering, red-teaming, the OWASP LLM Top 10. The connective tissue with this week is honesty under measurement — week 16 measured whether a fine-tune helped; week 17 measures whether a defense holds, with an attack-success-rate before and after. (And the fine-tune-or-not decision document you write this week is itself a Phase III milestone deliverable.) Push your `crunchtune` decision memo before you start; the Phase III milestone wants it.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
