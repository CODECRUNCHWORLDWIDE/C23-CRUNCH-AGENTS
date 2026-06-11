# Week 16 — Challenges

The exercises drill the judgment and the data — the decision ladder, the dataset, the memory math. **The challenge makes you the engineer who runs the real fine-tune and renders the verdict.** You train a LoRA adapter on a 7B model, evaluate it against the base on a held-out set, and commit to "worth it" or "not worth it" with a measured number — the way the decision actually gets made on a real project, where the eval is the deliverable and the training run is just the part in the middle.

## Index

1. **[Challenge 1 — Fine-tune and judge](challenge-01-fine-tune-and-judge.md)** — SFT a Qwen2.5-7B LoRA adapter on the NL→DSL task with Unsloth, evaluate base-vs-fine-tune on a held-out test set (exact-match + valid-DSL), and write the worth-it-or-not memo. (~150 min, needs a 24 GB GPU or a rented A10/L4, ~$3 of compute)

Challenges are optional for passing the week, but this one *is* the syllabus deliverable in lab form and the direct input to the **Phase III milestone** (a 1-page fine-tune-or-not decision document for the capstone's domain). Do it. The skill — building a clean dataset, training a real adapter, and then *honestly judging* whether it beat the prompt baseline — is what separates an engineer who "fine-tuned a model" (and can't say if it helped) from one who shipped a measured decision. And the most valuable version of that skill is being equally willing to write "we fine-tuned, it didn't help, here's the number, ship the prompt" — because that negative result is the one that saves a team from serving a pointless artifact.
