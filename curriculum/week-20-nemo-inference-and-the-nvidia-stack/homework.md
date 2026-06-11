# Week 20 Homework

Six problems that revisit the week's topics and force NVIDIA-stack and policy literacy into your fingers. The full set should take about **5 hours**. Work in your Week 20 Git repository (the same workspace as the exercises and the `crunchnemo` mini-project) so every problem produces at least one commit you can point to at the capstone serving review.

The headline deliverable is **Problem 6 — the one-page NeMo-vs-vLLM production memo**, the artifact that decides which stack serves your capstone's local tier. Treat it as the document a tech lead reads before signing off on the cluster, not a journal entry.

Have your **week-19 vLLM benchmark numbers** available (you compare against them) and your **week-17 red-team prompts** importable (the Guardrails ASR measurement reuses them). The Guardrails problems run with **no GPU** — `pip install nemoguardrails` and an endpoint (a Triton/vLLM target, or `claude-opus-4-8` via the anthropic engine, or the exercise's mock-LLM fallback). The TensorRT-LLM/Triton problems are GPU-gated; a rented-H100 recipe is in the README, and a small-model / conceptual path is documented for each.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Lay out a Triton model repository for the engine

**Problem statement.** Without necessarily building the engine on a GPU, design the Triton model repository for serving Qwen2.5-14B through the TensorRT-LLM backend. Produce `notes/week-20/triton-repo.md` containing: the directory tree (`model_repository/<name>/1/`, the `config.pbtxt`), an annotated `config.pbtxt` sketch naming the backend, max batch size, and instance group, and the `tritonserver` launch command. If you have a GPU, actually build a small-model engine with `trtllm-build` and serve it; either way, explain what each piece does.

**Acceptance criteria.**

- A model-repository tree plus an annotated `config.pbtxt` naming the TensorRT-LLM backend and the key fields.
- The `trtllm-build` and `tritonserver` commands, with a one-line explanation of each.
- A sentence distinguishing what TensorRT-LLM produced (the engine) from what Triton does (serves it). Committed.

**Hint.** The engine is the *compiled* artifact; the repository is how Triton *finds and serves* it (Lecture 1 §2–§3). If you can't build on a GPU, document the layout and commands and note the GPU step — the *structure* is what you're proving.

**Estimated time.** 45 minutes.

---

## Problem 2 — Write a rail that blocks one injection class, measure ASR

**Problem statement.** Build a NeMo Guardrails config (Colang + YAML, in-code via `RailsConfig.from_content` or a config dir) whose `self check input` rail blocks one specific prompt-injection class from your week-17 red-team (e.g. "ignore previous instructions / print the system prompt"). Run your week-17 attack prompts and a set of benign prompts through the railed endpoint *with the rail off and with it on*, and produce `notes/week-20/rail-asr.md` with the ASR-off, ASR-on, and benign-pass-rate numbers, plus one blocked-attack trace in the promise format.

**Acceptance criteria.**

- A working rail (real `nemoguardrails`, or the exercise's mock-LLM fallback) that blocks the chosen injection class.
- ASR measured *both* with the rail off (baseline) and on, plus the benign pass-rate.
- One per-attack trace in the promise format (`atk_03 (...) -> BLOCKED by self_check_input rail ✓`). Committed.

**Hint.** Reuse `exercise-02-guardrails-injection-block.py`. The rail-off number is the baseline that makes the rail-on number mean something (Lecture 2 §6.5). Report *both* metrics — ASR alone is a half-truth (Lecture 2 §6).

**Estimated time.** 50 minutes.

---

## Problem 3 — Benchmark NeMo/Triton vs week-19 vLLM, apples-to-apples

**Problem statement.** Compare the throughput/latency of a NeMo-TensorRT-LLM/Triton deployment against your week-19 vLLM deployment of the *same* model on the *same* hardware (or, without a GPU, use recorded/synthetic numbers and reason about fairness). Produce `notes/week-20/benchmark.md` with a tokens/sec and p50/p95 table for both stacks at matched concurrency, and a paragraph on what you held fixed to keep the comparison fair.

**Acceptance criteria.**

- A side-by-side throughput + p50/p95 table for NeMo/Triton vs vLLM at matched concurrency.
- An explicit statement of what was held fixed (model, hardware, workload, batching settings) so the comparison is apples-to-apples.
- A note on whether Guardrails latency is included on the NeMo side and how you accounted for it. Committed.

**Hint.** The trap is comparing different engine builds or different batching settings (challenge trap 1) — and forgetting the rail's latency (Lecture 2 §6.5). Match the settings; report the rail cost separately.

**Estimated time.** 45 minutes.

---

## Problem 4 — The false-positive / output-rail analysis

**Problem statement.** Take your Problem-2 rail and stress its *benign* side. Write 10 tricky-but-legitimate prompts that *resemble* an attack (e.g. "what formatting instructions do you follow?") and measure how many the rail wrongly blocks. Then add an *output* rail (self-check-output) and re-measure ASR to show defense-in-depth. Produce `notes/week-20/false-positives.md` with the benign-block count, two example false positives (if any), and the ASR before/after adding the output rail.

**Acceptance criteria.**

- 10 tricky-benign prompts run through the rail with the false-positive count reported.
- An output rail added, with ASR re-measured (input-only vs input+output).
- One sentence on the latency cost of the second rail. Committed.

**Hint.** False positives are invisible in ASR and visible only in the benign pass-rate (Lecture 2 §6). The output rail is the README stretch goal — defense in depth, measured. Watch the added per-request latency (Lecture 2 §6.5).

**Estimated time.** 45 minutes.

---

## Problem 5 — Score the decision matrix

**Problem statement.** Using `exercise-03-nemo-vs-vllm-decision.py` (or your own), score NeMo/TensorRT-LLM vs vLLM across the real axes — kernel performance, policy tooling, OSS velocity, operational simplicity, lock-in, and your measured throughput/latency from Problem 3 — weighting the axes by *your capstone's* needs. Produce `notes/week-20/decision-matrix.md` with the weighted matrix, the winning stack, and a sentence on which axis decided it.

**Acceptance criteria.**

- A weighted decision matrix with both stacks scored on the named axes.
- Weights justified by the capstone's actual requirements (not generic).
- A clear winner and the single most decisive axis. Committed.

**Hint.** The policy axis favors NeMo but its *size* depends on how much policy you actually need (Lecture 2 §7). Score by *your* week-17 threat model, not the abstract strength of the tooling.

**Estimated time.** 40 minutes.

---

## Problem 6 — The one-page NeMo-vs-vLLM production memo (headline deliverable)

**Problem statement.** Write a **one-page** memo at `notes/week-20/serving-memo.md` against this template:

1. **Decision** — one sentence: which stack (NeMo/Triton or vLLM) serves your capstone's local tier, with the headline number that justifies it.
2. **The benchmark** — the matched throughput + p50/p95 for both stacks, with what you held fixed.
3. **The policy story** — your rail's ASR-off vs ASR-on and benign pass-rate, and whether integrated Guardrails was a deciding factor.
4. **The trade-off accepted** — what you gave up (lock-in / operational weight for NeMo, or weaker integrated policy for vLLM) for the win.
5. **The honesty line** — the rail's latency cost (so the comparison is fair) and any axis where the *losing* stack was actually better.
6. **One blocked-attack trace** — in the promise format, showing the injection bounced off the rail, plus a benign prompt that still got answered.

**Acceptance criteria.**

- `notes/week-20/serving-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- The decision is justified by a *measured* number on *your* workload and threat model, not "NeMo is faster."
- The rail's latency cost and both ASR + benign-pass numbers are stated honestly.
- At least one blocked-attack trace in the promise format. Committed.

**Hint.** This is the capstone serving decision in miniature — the reviewer will ask "why *that* stack, and how do you know it's safe and fast enough?" The memo *is* that answer, with the benchmark and the ASR/benign pair behind it.

**Estimated time.** 55 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Triton model repository layout | 45 min |
| 2 — Rail blocks an injection class + ASR | 50 min |
| 3 — NeMo vs vLLM apples-to-apples benchmark | 45 min |
| 4 — False-positive + output-rail analysis | 45 min |
| 5 — Score the decision matrix | 40 min |
| 6 — NeMo-vs-vLLM production memo (headline) | 55 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchnemo` [mini-project](./mini-project/README.md) is in the same workspace — Week 21 builds a cost-routing layer on top of the serving stack you chose here, and the capstone serves from it. Then take the [quiz](./quiz.md) with your notes closed.
