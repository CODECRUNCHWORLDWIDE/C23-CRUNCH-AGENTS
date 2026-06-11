# Week 2 — Exercises

Three focused drills that turn the lectures into muscle memory. Each takes 45–75 minutes. Do them in order — exercise 1 makes tokenizer differences visceral, exercise 2 builds the sampler the lecture described, and exercise 3 uses the sampler-masking idea to make schema-valid output a guarantee.

## Index

1. **[Exercise 1 — The tokenizer explorer](exercise-01-tokenizer-explorer.md)** — encode the same five strings (English, code, CJK, emoji, whitespace) with three open tokenizers via `AutoTokenizer`, build a counts table, and explain — with the actual token pieces — *why* they disagree. (~60 min, guided)
2. **[Exercise 2 — A sampler from logits](exercise-02-sampler-from-logits.py)** — implement temperature / top-k / top-p / min-p truncation from raw logits in NumPy, then run a verification harness that proves each knob does what the lecture said. (~60 min, runnable)
3. **[Exercise 3 — Provably-valid constrained JSON](exercise-03-constrained-json.py)** — use `outlines` to grammar-constrain a tiny local model, then assert with `jsonschema` that the output is schema-valid across a fuzz set of adversarial prompts — 100% of the time, by construction. (~60 min, runnable)

## How to work the exercises

- **Have your environment ready before you start.** A venv with the week's packages, and (for ex. 3) a tiny local model that `outlines` downloads on first run:

  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install transformers tokenizers numpy outlines jsonschema anthropic
  # ex.3 pulls Qwen/Qwen2.5-0.5B-Instruct automatically on first run (~1GB, CPU-fine)
  ```

- **Always ask "which tokenizer produced this count?"** Every number in exercise 1 is from a specific tokenizer; mixing them is the cost-estimation error of the week. Never reach for `tiktoken` to count a non-OpenAI model.
- Each runnable exercise (`.py`) ends with an **expected output** block. Your exact numbers will differ (model nondeterminism, hardware) but the *shape* must match — and for exercise 3 the validity rate must be exactly **100%**, not "usually." If it isn't, you're not done.
- **No API key? No problem.** All three exercises run fully offline. Exercise 1 uses local tokenizers; exercises 2 and 3 use a local model and pure NumPy. The Anthropic comparison in exercise 1 is optional and degrades to an "unavailable" line.

## Running the Python exercises

The two `.py` files are standalone — no package, no framework. Activate your venv and run them directly:

```bash
source .venv/bin/activate
python3 exercise-02-sampler-from-logits.py
python3 exercise-03-constrained-json.py
```

Each file's header documents how to use it, the acceptance criteria, and the expected-output shape. The `# TODO N:` markers are the spots you fill in; everything else is done. Read the header before you run.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-02` to compare.
