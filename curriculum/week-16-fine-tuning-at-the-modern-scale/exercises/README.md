# Week 16 — Exercises

Three focused drills that take you from "should we even fine-tune?" to "I built the dataset and I can read a loss curve." Each takes 30–60 minutes. Do them in order — exercise 3 reasons about the data you build in exercise 2, which assumes the decision discipline from exercise 1.

## Index

1. **[Exercise 1 — Decide to fine-tune](exercise-01-decide-to-fine-tune.md)** — walk three real scenarios through the prompt→retrieve→fine-tune ladder and write the decision (and the legitimate reason, or the reason not to) for each. (~45 min, guided)
2. **[Exercise 2 — Build an SFT dataset](exercise-02-build-an-sft-dataset.py)** — generate, chat-template, validate, and split a 500-example NL→DSL SFT dataset, catching the leakage/format-drift failure modes. (~50 min, runnable)
3. **[Exercise 3 — LoRA memory and the loss curve](exercise-03-lora-memory-and-loss.py)** — compute the LoRA/QLoRA memory budget that fits a 7B in 24 GB, and simulate the three loss-curve shapes so you can read a real one. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install the lightweight deps for these exercises: `pip install datasets transformers numpy`. (You do **not** need a GPU, Unsloth, or `bitsandbytes` for the exercises — those are for the challenge's actual training run. Exercises 2 and 3 are pure data/arithmetic and run on any CPU.)
- **Climb the ladder before you reach for weights.** Exercise 1's whole point is that most "fine-tune it" instincts are prompt or retrieval problems. Write the cheaper option down before the expensive one.
- **The chat template is the bug.** Exercise 2 makes you apply `apply_chat_template`; the lesson is that hand-formatting the string is the #1 silent fine-tune failure. Use the tokenizer's template, always.
- **A held-out split is the firewall.** Exercise 2 splits before any training and never lets test leak into train. If a test example is in train, your score is a lie.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone and CPU-only.

```bash
python3 exercise-02-build-an-sft-dataset.py     # builds + validates + splits the dataset
python3 exercise-03-lora-memory-and-loss.py     # memory budget + loss-curve simulation
```

Exercise 2 uses the HF `transformers` tokenizer to apply a real chat template; the first run downloads a small tokenizer (~a few MB, not the full model). If you're fully offline, the script falls back to a documented manual template so the validation lessons still run.

## A note on the GPU

The *exercises* need no GPU — they're about the decision, the data, and the arithmetic, which are the parts you get wrong if you skip them. The **challenge** and **mini-project** run the real training, which wants a 24 GB GPU or a rented A10/L4 (~$5 for the week). Do the exercises CPU-side first so you arrive at the rented GPU knowing exactly what you're going to run — debugging the data format on rented time is how you burn your $5 budget.

## A note on honesty

The whole week's deliverable is a *verdict*, and the verdict can be "don't fine-tune." Exercise 1 deliberately includes a scenario where the right answer is "fix the prompt" and one where it's "use retrieval" — not every scenario fine-tunes. If you reach "fine-tune" on all three, re-read the ladder; the discipline is reaching for the cheap rung first, every time.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-16` to compare.
