# Week 19 Homework

Six problems that revisit the week's topics and force self-hosted serving literacy into your fingers. The full set should take about **5 hours**. Work in your Week 19 Git repository (the same workspace as the exercises and the `crunchserve` mini-project) so every problem produces at least one commit you can point to when the capstone's serving tier gets built.

The headline deliverable is **Problem 6 — the one-page break-even / serving memo**, called out explicitly in the syllabus. Treat it as the artifact a reviewer reads to decide whether the local serving tier exists, not a journal entry.

Have a vLLM server reachable (the resources GPU recipe — rent one H100 for ~6 hours with a ~$12–15 ceiling, **destroy it the moment you're done, set an alarm** — or the no-GPU path: a tiny model on CPU, or the `--simulate` path each `.py` ships). This week builds on **week 6** (you've run a model locally before); here you run it under load and behind a router. If your benchmark or cost scripts from the exercises are broken, fix them first — the homework reuses them.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Stand up and query vLLM (or simulate)

**Problem statement.** Launch a vLLM OpenAI-compatible server (`vllm serve Qwen/Qwen2.5-14B-Instruct ...` on a rented GPU, or `Qwen/Qwen2.5-0.5B-Instruct --device cpu` with no GPU). Hit `/v1/models` to read the served model id, then send a chat completion with both `curl` and the `openai` Python client pointed at the local `base_url`. Produce `notes/week-19/serving.md` containing: the model id from `/v1/models`, one `curl` response (showing the `usage` block), and the `openai`-client snippet that worked. (No GPU at all? Document the `--simulate` path from Exercise 2 instead and note that the server steps require a server.)

**Acceptance criteria.**

- `notes/week-19/serving.md` shows the served model id read from `/v1/models` (not hardcoded).
- A chat completion response is captured with its `usage` block visible (`completion_tokens`).
- The `openai` client is pointed at the local `base_url` with a placeholder `api_key`, and returned a completion.
- Committed.

**Hint.** The `api_key` can be any non-empty string — vLLM doesn't authenticate by default, but the client library requires a value. Read the model id from `client.models.list().data[0].id`; don't hardcode it. If `curl` refuses the connection, the weights are still downloading — watch the server log for "running."

**Estimated time.** 40 minutes.

---

## Problem 2 — Run the concurrency sweep

**Problem statement.** Run the concurrency benchmark (Exercise 2's logic, or your `crunchserve/benchmark.py`) at concurrency **1, 8, 32, 128** against your server (or `--simulate`). Produce `notes/week-19/sweep.md` with the tokens/sec, p50, p95, and req/sec table and an ASCII curve, plus one sentence naming the tokens/sec at concurrency 128 and the factor by which it beats concurrency 1.

**Acceptance criteria.**

- A table of tokens/sec, p50, p95, and req/sec for all four concurrency levels.
- tokens/sec demonstrably **climbs** from concurrency 1 to 128 (the continuous-batching curve), then begins to plateau.
- You state the top-concurrency tokens/sec and the climb factor (e.g. "~10x over concurrency 1") — that number is the denominator Problem 5 uses.
- Committed.

**Hint.** Read tokens from the response `usage` block, never from `len(text.split())` — an estimated denominator corrupts every downstream cost number. If tokens/sec *doesn't* climb on a real server, suspect `--max-num-seqs` set too low (you throttled yourself) before anything else. In `--simulate` the curve always climbs.

**Estimated time.** 45 minutes.

---

## Problem 3 — Reason about a serving config

**Problem statement.** Write `notes/week-19/config-reasoning.md` answering, with the mechanism (not a guess): (a) you're serving a 14B model on one H100 80GB — do you set `--tensor-parallel-size` above 1, and why or why not? (b) your workload is all ≤4K-token chats but you set `--max-model-len 32768` — what does that cost you? (c) your tokens/sec plateaus at concurrency 32 even though the GPU has memory headroom — which flag is the likely culprit and which way do you move it?

**Acceptance criteria.**

- Part (a): states that a 14B in bf16 (~28 GB) fits on one 80GB H100, so TP=1 (no parallelism); raising TP adds an all-reduce every layer for no memory benefit.
- Part (b): states that an oversized `max_model_len` budgets more KV *per sequence*, so fewer sequences fit → lower concurrency → lower throughput, for capacity you never use; size it to the real P99 length.
- Part (c): identifies `--max-num-seqs` set too low (the explicit concurrency cap) and says to raise it until throughput plateaus, watching for preemption.
- Committed.

**Hint.** Each knob trades against the same finite KV budget (Lecture 1 §9). `gpu_memory_utilization` sets the budget; `max_model_len` sets KV-per-sequence; `max_num_seqs` caps how many sequences run. The "throughput won't climb past 32" symptom is the canonical `max_num_seqs`-too-low signature.

**Estimated time.** 45 minutes.

---

## Problem 4 — Compute cost-per-million-tokens

**Problem statement.** Take your measured tokens/sec from Problem 2 (or the default 2,180) and a GPU price of $2.50/hr, and compute self-hosted cost-per-million-tokens with the Lecture 2 §5 formula (Exercise 3's logic or your `crunchserve/cost.py`). Then compute it *again* at the concurrency-1 tokens/sec (~40). Produce `notes/week-19/cost.md` with both numbers and one sentence on why they differ by ~50×.

**Acceptance criteria.**

- `cost_per_MTok = $/hr ÷ (tokens/sec × 3600) × 1e6` computed at your production tokens/sec.
- The same formula computed at ~40 tokens/sec (concurrency 1), showing a far higher $/MTok.
- A sentence explaining the ~50× gap is entirely the tokens/sec denominator — i.e. continuous batching keeping the batch full — and that concurrency-1 is an idle GPU.
- Committed.

**Hint.** A token is a token to a rented GPU, so the self-hosted figure is one blended per-token cost (no input/output split). The denominator is the whole game: same $/hr, same model, same GPU — only how full you kept the batch changes the cost by an order of magnitude.

**Estimated time.** 35 minutes.

---

## Problem 5 — Configure a LiteLLM fallback

**Problem statement.** Write a LiteLLM `config.yaml` (or your `crunchserve/litellm_config.yaml`) with a `qwen-local` backend pointing at your vLLM server and a `claude-fallback` backend pointing at `anthropic/claude-haiku-4-5`, with a `router_settings.fallbacks` rule that retries `qwen-local` on `claude-fallback`. Start the proxy (`litellm --config config.yaml`) and send one request to the `qwen-local` public name through the proxy. Capture the config and the working request in `notes/week-19/litellm.md`, and explain in one sentence what week 24's chaos drill will use this rule to verify. (No GPU? Document the config and explain the fallback path; you can point `qwen-local` at the CPU/tiny-model server or note the simulate substitution.)

**Acceptance criteria.**

- A `config.yaml` with a `model_list` (a `qwen-local` vLLM backend + a `claude-fallback` vendor backend) and a `fallbacks: - qwen-local: ["claude-fallback"]` rule.
- The vendor key is read from the environment (`os.environ/ANTHROPIC_API_KEY`), not hardcoded.
- One request sent to the *public* `qwen-local` name through the proxy (`:4000`), captured.
- A sentence stating that week 24's chaos drill kills a replica/pool and verifies LiteLLM rides over the loss and falls over to the vendor instead of erroring.
- Committed.

**Hint.** The `model:` inside `litellm_params` is provider-prefixed: `openai/...` for the OpenAI-compatible vLLM backend, `anthropic/claude-haiku-4-5` for the vendor. The client hits the *public* `model_name` (`qwen-local`) and never knows which backend answered — that indirection is the whole point.

**Estimated time.** 45 minutes.

---

## Problem 6 — The one-page break-even / serving memo (headline deliverable)

**Problem statement.** This is the syllabus deliverable. Combining Problems 2, 4, and 5, write a **one-page** serving memo at `notes/week-19/serving-memo.md` that decides self-host-vs-vendor for the capstone's local tier, against this template:

1. **Decision** — one sentence: self-host on vLLM-behind-LiteLLM, or call the vendor, and the headline number.
2. **The throughput evidence** — the concurrency sweep's tokens/sec at 1/8/32/128, naming the production-concurrency number.
3. **The unit cost** — self-hosted cost-per-MTok at production concurrency vs the blended vendor price (`claude-haiku-4-5` at your input:output ratio), with the factor.
4. **The break-even volume** — the monthly token volume where the fixed GPU rental equals the vendor bill, and where your *expected* volume sits relative to it.
5. **The utilization caveat** — the peak $/MTok holds only if you keep the GPU busy; state the utilization assumption and what happens to the effective cost if real traffic is light (the idle-GPU trap).
6. **The resilience note** — the LiteLLM vendor fallback you configured (Problem 5), and that an outage degrades to the vendor instead of to an error (the chaos-drill safety net).

**Acceptance criteria.**

- `notes/week-19/serving-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- The decision is justified by the **measured** tokens/sec and the computed $/MTok and break-even volume — not "it felt cheaper."
- The utilization caveat is stated honestly (peak vs average; the idle-GPU trap).
- The concurrency-1 trap is acknowledged: the memo does not quote a concurrency-1 cost as the verdict.
- At least one promise-format trace: `concurrency 128 still served, tokens/sec up ~Nx over concurrency 1`.
- Committed.

**Hint.** The break-even volume is `fixed_monthly_GPU_cost ÷ vendor_$_per_token`. Locate your *expected* monthly volume on that curve: above it, self-host; below it, the vendor is cheaper because the idle GPU's fixed rental dominates. The two levers to name are tokens/sec (raises the denominator, cuts $/MTok) and utilization (whether the fixed rental is worth paying at all).

**Estimated time.** 1 hour 10 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Stand up + query vLLM (or simulate) | 40 min |
| 2 — Run the concurrency sweep | 45 min |
| 3 — Reason about a serving config | 45 min |
| 4 — Compute cost-per-million-tokens | 35 min |
| 5 — Configure a LiteLLM fallback | 45 min |
| 6 — Break-even / serving memo (headline) | 1 h 10 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchserve` [mini-project](./mini-project/README.md) is in the same workspace — the capstone serves its local tier from this pattern, and week 24's chaos drill stress-tests the fallback you configured. Then take the [quiz](./quiz.md) with your notes closed.

Up next: **Week 20 — the NVIDIA stack (NeMo Inference, TensorRT-LLM, Triton/Dynamo)**, the heavier-weight path to throughput when vLLM's defaults aren't enough — benchmarked against the same workload this week's harness defines.
