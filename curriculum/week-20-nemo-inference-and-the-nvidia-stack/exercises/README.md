# Week 20 — Exercises

Three drills that take you from "I read about the NVIDIA stack" to "I built an engine, blocked an injection, and scored the decision." Do them in order — Exercise 1 builds the serving intuition, Exercise 2 builds the policy you'll measure, and Exercise 3 turns both into a defensible choice.

## Index

1. **[Exercise 1 — Triton + TensorRT-LLM deploy](exercise-01-triton-trtllm-deploy.md)** — build a TensorRT-LLM engine, lay out a Triton model repository, launch `tritonserver`, and hit the OpenAI-compatible endpoint. (~60 min, guided. **GPU-gated** for the real build; a no-GPU conceptual walkthrough + small-model/NIM path is included.)
2. **[Exercise 2 — Guardrails injection block](exercise-02-guardrails-injection-block.py)** — build a NeMo Guardrails config in-code, fire benign + prompt-injection prompts, and prove the rail blocks the attacks while passing the benign ones — printing an ASR before/after table. (~50 min, runnable. **No GPU.**)
3. **[Exercise 3 — NeMo-vs-vLLM decision](exercise-03-nemo-vs-vllm-decision.py)** — score recorded/synthetic throughput-latency numbers plus the qualitative axes (kernel perf, policy tooling, OSS velocity, operational simplicity, lock-in) on a weighted decision matrix, and print a recommendation with reasons. (~50 min, runnable. **No GPU.**)

## Which parts are GPU-gated vs CPU-reachable

This is the most important thing to know before you start, so you don't burn a rented GPU-hour on something you could do on your laptop:

| Part | Needs a GPU? | Notes |
|---|---|---|
| **Exercise 1 — real engine build + Triton serving** | **Yes (Hopper H100 ideal)** | `trtllm-build` and the `tensorrtllm` backend need a real NVIDIA GPU. Rent at ~$2–3/hr, tear down after. |
| **Exercise 1 — no-GPU path** | **No** | Read the conceptual walkthrough; build a 1–2B small model on a cheap/free-tier GPU, or pull a NIM container, to learn the model-repo layout and `config.pbtxt` without the 14B run. |
| **Exercise 2 — Guardrails injection block** | **No** | Runs CPU-only against `claude-opus-4-8` (or any OpenAI-compatible endpoint), and degrades to a mock-LLM + heuristic rail if `nemoguardrails` or an API key is absent — so it *always* runs. |
| **Exercise 3 — decision matrix** | **No** | Pure stdlib + numpy. Runs anywhere. |

So: **Exercises 2 and 3 run on any machine.** Exercise 1 is the only GPU-gated drill, and it has a documented no-GPU path so you can learn the mechanics regardless. Don't rent a GPU until you're ready to do Exercise 1's real build in one sitting.

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- For the no-GPU drills: `pip install nemoguardrails anthropic numpy`. (Exercise 2 runs even if `nemoguardrails` fails to install — it falls back to a mock LLM and a heuristic rail — but install it if you can, to see the real rail.)
- For Exercise 1's real build, work *inside* the NVIDIA container (`nvcr.io/nvidia/tritonserver:<tag>-trtllm-python-py3`) on the rented GPU — installing TensorRT-LLM bare-metal is a known time-sink; the container is the supported path.
- **Set your API key once:** `export ANTHROPIC_API_KEY=...` if you want Exercise 2 to run the *real* rail over `claude-opus-4-8`. Without it, Exercise 2 still runs (mock path) and the lesson — a rail blocks the injection — is still visible.
- **Reuse week 19's vLLM benchmark and week 17's red-team prompts.** Exercise 3 wants throughput/latency numbers for the comparison (synthetic ones ship in the file so it runs without them); Exercise 2's attack list mirrors the week-17 injection class so the ASR is comparable.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone:

```bash
# Runs anywhere, no GPU. Real rail if nemoguardrails + ANTHROPIC_API_KEY are present;
# mock-LLM + heuristic-rail fallback otherwise. Either way it always runs.
python3 exercise-02-guardrails-injection-block.py

# Pure stdlib + numpy. Scores the decision matrix and prints a recommendation.
python3 exercise-03-nemo-vs-vllm-decision.py
```

## A note on cost discipline

The one expensive mistake this week is a forgotten H100. The real-build leg of Exercise 1 is the only thing that costs money, and it costs ~$2–3/hr. **Decide what you'll measure before you spin up the instance, run the build + Triton + benchmark in one focused session, capture the numbers, and tear it down.** Everything else — the policy, the decision, the memo — is free and runs on your laptop. Don't leave a GPU idle while you write the memo.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-20` to compare.
