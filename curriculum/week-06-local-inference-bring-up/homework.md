# Week 6 Homework

Six problems that revisit the week's topics and force local-inference literacy into your fingers. The full set should take about **5 hours**. Work in your Week 6 Git repository (the same workspace as the exercises and the `crunchserve` mini-project) so every problem produces at least one commit you can point to at the Phase I milestone review.

The headline deliverable is **Problem 4 — the engine-selection memo**, and **Problem 5 — your week-5 agent on a local endpoint** is the one that completes the Phase I capstone milestone. Treat both as artifacts a reviewer reads, not journal entries.

Have your **week-5 ReAct agent** importable and at least one local engine (Ollama is enough) running (`ollama serve & ; ollama pull qwen2.5:7b`). The vLLM problems need a CUDA GPU (local or rented ~$1; see resources.md). If a GPU-only problem is blocked, do its CPU/Ollama fallback and note it.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Bring up two engines and prove one client serves both

**Problem statement.** Bring up the same 7B on **Ollama** and **llama.cpp** (same Q4_K_M GGUF), and write `notes/week-06/bring-up.md` showing the *same* Python OpenAI client getting a sensible answer from each by only changing `base_url`. Confirm the two (same weights, same quant) give close behavior and explain why.

**Acceptance criteria.**

- Both engines serve the 7B and answer the same prompt; output shown in the notes.
- One Python client hits both by swapping `base_url` only (no per-engine SDK).
- A one-sentence explanation of why Ollama and llama.cpp on the same GGUF behave alike (Ollama wraps llama.cpp).
- Committed.

**Hint.** Get the GGUF once and point both at it: Ollama via `ollama pull qwen2.5:7b`, llama.cpp via `llama-server -m <same-or-equivalent>.gguf`. The OpenAI client takes `base_url=` and an ignored `api_key=`.

**Estimated time.** 40 minutes.

---

## Problem 2 — Separate prefill from decode with `llama-bench`

**Problem statement.** Run `llama-bench` on your 7B GGUF and record the **prefill** (`pp`) and **decode** (`tg`) tokens/sec separately. Write one sentence on which is bigger and why (prefill processes the prompt in parallel; decode is one-token-at-a-time and memory-bandwidth-bound). Put it in `notes/week-06/prefill-decode.md`.

**Acceptance criteria.**

- `pp` (prefill) and `tg` (decode) tokens/sec recorded for your model on your hardware.
- A one-sentence explanation of the gap grounded in Lecture 1 §2.
- (Stretch) the same model at two quant levels, showing decode speeds up more than prefill when you quantize.
- Committed.

**Hint.** `llama-bench -m model.gguf` prints `pp` and `tg` rows. Decode (`tg`) is the memory-bandwidth-bound phase; quantizing helps it most. If you only have Ollama, use the Exercise 3 harness's TTFT (prefill proxy) and decode-rate numbers instead.

**Estimated time.** 35 minutes.

---

## Problem 3 — The quantization trade-off table for your VRAM budget

**Problem statement.** Using Exercise 2's logic (or your own arithmetic), produce `notes/week-06/quant-budget.md`: for your actual VRAM budget (your GPU's GB, or a target like 8 GB), state which quant levels of a 7B *fit* once you include the KV cache at your intended context and concurrency, and which you'd ship. Name the knee and the trade in numbers.

**Acceptance criteria.**

- A table of quant level → model size + KV-cache estimate → fits-in-budget? for your target VRAM.
- The chosen quant named with its trade ("Q4_K_M: ~3.4× smaller, ~3.3× faster decode, perplexity within ~2% of FP16").
- The KV-cache cost is **included**, not forgotten (Lecture 2 §1.3, §4).
- Committed.

**Hint.** Run `exercise-02-quantization-tradeoffs.py` and set `CONTEXT`/`BATCH` to your intended serving point — the KV-cache line is the part people forget. The VRAM that matters is weights + cache at *your* concurrency, not weights alone.

**Estimated time.** 40 minutes.

---

## Problem 4 — The engine-selection memo (headline deliverable)

**Problem statement.** Run the three-engine bakeoff from Challenge 1 (or at least Ollama + llama.cpp on a laptop, and vLLM on a rented GPU if you can) and write a **one-page** memo at `notes/week-06/engine-memo.md` against this template:

1. **Decision** — one sentence: which engine you'd serve a *concurrent* workload on, and its headline number (aggregate tokens/sec at your target concurrency).
2. **The table** — the engines with prefill/decode tokens/sec, TTFT p50/p95, aggregate throughput at 1/8/32(/128), VRAM, and each engine's quant label.
3. **Why this winner, for this workload** — the mechanism (e.g. "vLLM's aggregate rises 17× from c=1 to c=32 because continuous batching keeps the GPU full; Ollama's per-request decode collapses as requests queue"), not a general claim.
4. **The trade-off accepted** — what you gave up (e.g. vLLM's heavier startup, CUDA requirement) for the throughput.
5. **The quant caveat** — the GGUF-vs-AWQ asymmetry means the comparison includes a quant-format change; state how you'd isolate it (an FP16 control at the same engine).
6. **The workload note** — for a *single-user iteration* workload your pick would flip to Ollama; name the workload your decision is for.

**Acceptance criteria.**

- `notes/week-06/engine-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- The model, prompt set, and `max_tokens` are demonstrably identical across engines; the quant asymmetry is noted.
- The winner is justified by a **specific** mechanism (the concurrency curve), not "it felt faster."
- At least one promise-format line for the winner.
- Committed.

**Hint.** The concurrency *curve* is load-bearing: benchmark only at concurrency 1 and your memo measures iteration, not serving (the challenge's trap). If you have no GPU, run Ollama + llama.cpp fully and reason about vLLM from the paper + a short rented session; state which numbers are measured vs reasoned.

**Estimated time.** 1 hour.

---

## Problem 5 — Your week-5 agent on a local endpoint (Phase I milestone)

**Problem statement.** Take your **week-5 ReAct agent** and give it a `--base-url` so it runs against a *local* engine instead of a vendor API. Run it against (a slice of) your week-5 benchmark on a local 7B and record the result. This is the Phase I capstone milestone — a working ReAct agent on a model *you* serve.

**Acceptance criteria.**

- The agent takes a `--base-url` (and model name) and runs end-to-end against a local engine (Ollama is fine), completing at least a few benchmark tasks with tool calls.
- `notes/week-06/agent-on-local.md` records: the endpoint used, the model, the tasks attempted, and the pass rate — versus the vendor-API numbers from week 5 if you have them.
- One sentence on what got *worse* on the local 7B (smaller models are weaker at tool-call formatting) and how you'd mitigate it (better prompting, grammar-constrained decoding from week 2, or a bigger local model).
- Committed.

**Hint.** If your week-5 agent hard-coded the Anthropic client, refactor it to construct an OpenAI-compatible client from a base URL first — that one change makes it engine-agnostic. A local 7B is weaker at tool-call JSON than a frontier model; expect a lower pass rate and *measure it* rather than hiding it. That honest delta is the lesson.

**Estimated time.** 1 hour.

---

## Problem 6 — Predict and measure a concurrency curve

**Problem statement.** Before running, **predict** the shape of the aggregate-throughput-vs-concurrency curve for one engine you have (Ollama on a laptop is fine). Then run Exercise 3's harness at concurrency 1/4/8/16 and compare your prediction to the result. Record both in `notes/week-06/concurrency.md`.

**Acceptance criteria.**

- A written prediction (rising or flat, and why) *before* the run.
- The measured aggregate tokens/sec at each concurrency level, and the per-request decode rate.
- A one-sentence reconciliation: did the curve rise or flatten, and does that match the engine's batching behavior (Ollama serializes → flat aggregate, collapsing per-request decode)?
- Committed.

**Hint.** Ollama serializes, so predict a *flat* aggregate and a *collapsing* per-request decode rate as concurrency rises (the requests queue). vLLM would predict a *rising* aggregate. Predicting first, then measuring, is the habit — it turns "huh, interesting" into "as expected, because continuous batching."

**Estimated time.** 45 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Two engines, one client | 40 min |
| 2 — Prefill vs decode with llama-bench | 35 min |
| 3 — Quant trade-off for your VRAM budget | 40 min |
| 4 — Engine-selection memo (headline) | 1 h 0 min |
| 5 — Week-5 agent on a local endpoint (milestone) | 1 h 0 min |
| 6 — Predict and measure a concurrency curve | 45 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchserve` [mini-project](./mini-project/README.md) is in the same workspace. Then take the [quiz](./quiz.md) with your notes closed. Problem 5 completes the Phase I milestone — make sure you can narrate that agent trace step-by-step, because that's what the milestone review asks for.
