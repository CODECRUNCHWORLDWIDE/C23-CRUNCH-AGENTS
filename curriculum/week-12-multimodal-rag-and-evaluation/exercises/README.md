# Week 12 — Exercises

Three focused drills that take you from "I sent a figure to a VLM" to "my judge agrees with humans, and I have the kappa to prove it." Each takes 30–60 minutes. Do them in order — exercise 3 (judge calibration) reuses the metric intuition from exercise 2 (the Ragas metrics from scratch), which builds on the multimodal answer step from exercise 1.

## Index

1. **[Exercise 1 — VLM vs. Claude over a figure](exercise-01-vlm-vs-claude-over-a-figure.md)** — render a PDF page to an image, ask an open VLM *and* Claude vision the same question over the figure, compare; then embed images with CLIP and run a text→image retrieval. (~45 min, guided)
2. **[Exercise 2 — The Ragas metrics from scratch](exercise-02-ragas-metrics-from-scratch.py)** — implement faithfulness, context recall, context precision, and answer relevancy from scratch on a fixed dataset, so the four metrics stop being magic. (~50 min, runnable)
3. **[Exercise 3 — The judge-calibration sweep](exercise-03-judge-calibration-sweep.py)** — run an LLM-as-judge against 10 human labels, compute agreement and Cohen's kappa, sweep the threshold, and pick the calibrated operating point. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install deps as each exercise needs them: `pip install pymupdf pillow sentence-transformers anthropic ragas numpy scikit-learn matplotlib`. The open-VLM leg of Exercise 1 also wants `transformers torch` (or Ollama); both have graceful fallbacks so the file still runs without a GPU.
- **A metric you can't reproduce by hand is a metric you don't understand.** Exercise 2's whole point is to compute faithfulness *the way Ragas does* — decompose into claims, check each — so the library's number stops being a black box.
- **Never trust a judge you didn't calibrate.** Exercise 3 makes that concrete: you'll see a judge that looks 80%-accurate collapse to a near-zero kappa once you correct for chance, and you'll find the threshold that fixes it.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone and run **offline by default** — they ship a deterministic stub judge so you never need an API key or a GPU to get the lesson. The header of each documents how to swap the stub for a real judge (`claude-opus-4-8` or a local model).

```bash
# with the venv active:
python3 exercise-02-ragas-metrics-from-scratch.py
python3 exercise-03-judge-calibration-sweep.py
```

For Exercise 1's real legs (a live VLM or Claude vision), set `ANTHROPIC_API_KEY` for the Claude path or have a local VLM (`ollama run qwen2.5vl`, or `transformers` + a GPU) for the open path. If you have neither, the exercise's fallback prints the *shape* of the comparison so the lesson still lands.

## A note on determinism and the judge

The from-scratch metrics (Exercise 2) are deterministic given a fixed judge — same dataset, same stub, same numbers every run. A *real* LLM judge is **not** perfectly deterministic, which is exactly why you calibrate it (Exercise 3) and report the kappa rather than treating its score as ground truth. The calibration sweep is reproducible with the stub judge; with a real judge, re-run it a few times and note the variance — that variance *is* the judge's measurement noise, and a judge whose verdict flips run-to-run is one whose threshold you can't trust.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-12` to compare.
