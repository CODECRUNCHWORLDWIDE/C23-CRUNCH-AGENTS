# Week 17 — Exercises

Three focused drills that take you from "what attacks exist" to "I measured a defense reducing attack success." Each takes 30–60 minutes. Do them in order — exercise 3 runs the attacks you taxonomize in exercise 1 through the filters you build in exercise 2.

## Index

1. **[Exercise 1 — Write attacks](exercise-01-write-attacks.md)** — write a structured taxonomy of 15 adversarial prompts (direct, indirect, tool-argument abuse), each with a checkable success criterion, and predict which will land. (~45 min, guided)
2. **[Exercise 2 — Build an injection filter](exercise-02-build-an-injection-filter.py)** — build input and output filters, measure precision/recall on a labeled set, and confront the false-positive trade-off (benign-pass-rate). (~50 min, runnable)
3. **[Exercise 3 — Measure attack-success-rate](exercise-03-measure-attack-success-rate.py)** — run an adversarial suite against a toy tool-agent, compute ASR, add defenses one layer at a time, and report the per-layer delta. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install the lightweight deps: `pip install` is barely needed — Exercises 2 and 3 are pure Python (filters + a toy agent + arithmetic) and run on any CPU. The optional classifier path (Llama Guard) needs `transformers` and a GPU/hosted endpoint, documented inline; the core lessons run without it.
- **A checkable success criterion makes ASR a number.** Exercise 1's whole point is that "the attack succeeded" must be *automatically checkable* (the canary leaked / the sandbox was escaped / the prompt was revealed), or you can't measure ASR. Vague success = no measurement.
- **Measure both axes.** Exercise 2 forces you to track benign-pass-rate alongside attack-catch-rate. A filter that catches every attack by blocking every request is a denial-of-service, not a defense.
- **The load-bearing layer is argument validation.** Exercise 3 shows that input filtering reduces *whether* the model is steered, but argument validation reduces *what a steered model can do* — and the latter holds even when the former fails. Watch which layer the path-traversal attacks bounce off.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone and CPU-only.

```bash
python3 exercise-02-build-an-injection-filter.py     # filters + precision/recall + benign-pass
python3 exercise-03-measure-attack-success-rate.py   # ASR before/after layered defenses
```

## A note on ethics and scope

Every attack you write here targets a *toy agent you own* with *planted canaries*. Red-teaming is a defensive discipline: you attack your own systems to harden them. Do not point these techniques at systems you don't own or aren't authorized to test. The framing throughout is defensive threat modeling — which is exactly what a security review wants.

## A note on the residual

The honest outcome of Exercise 3 is *not* ASR = 0. Some attacks survive all your defenses, and naming them is the deliverable, not a failure. If your ASR hits exactly zero on a non-trivial suite, suspect your success criteria are too weak (the attacks "succeed" too rarely to measure) before you celebrate. A real defense pipeline drops ASR a lot and leaves a named residual.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-17` to compare.
