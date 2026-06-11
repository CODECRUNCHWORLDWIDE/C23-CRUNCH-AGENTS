# Week 19 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 20. Answer key is at the bottom — don't peek.

---

**Q1.** Why does continuous batching beat static (request-level) batching for throughput? Pick the *most complete* reason.

- A) It uses a bigger batch size than static batching can.
- B) It schedules at the iteration level — a finished sequence leaves the batch immediately and a queued request joins immediately every decode step — so there are no dead slots and no head-of-line blocking, whereas a static batch runs until its *longest* member finishes with short requests sitting idle.
- C) It disables the KV cache, which is the bottleneck.
- D) It only works at concurrency 1, where there's no contention.

---

**Q2.** What does PagedAttention actually fix, and how?

- A) It compresses the model weights so more fit in memory.
- B) It manages the KV cache as fixed-size blocks allocated on demand (with a per-sequence block table), eliminating the 60–80% internal/external fragmentation of reserving `max_model_len` contiguously per sequence — so far more sequences fit in the same memory.
- C) It removes the need for a KV cache entirely.
- D) It speeds up the matrix multiply for a single token.

---

**Q3.** A naive server reserves `max_model_len` (say 2,048 tokens) of KV per sequence up front, but the average request uses ~120 tokens. What's the consequence?

- A) Requests over 120 tokens are rejected.
- B) Each sequence reserves ~17× more KV than it uses, so the GPU fits a tiny fraction of the sequences it could — that reserved-but-empty memory is concurrency you can't use. PagedAttention's on-demand blocks reclaim it.
- C) Nothing — reservation is free.
- D) The model runs slower per token.

---

**Q4.** What does `--enable-prefix-caching` buy you, and when does it *not* help?

- A) It caches the model weights; it always helps.
- B) It hashes and reuses the KV blocks of a shared prompt prefix (e.g. a long system prompt) so that prefix is prefilled once and reused across requests — lowering TTFT and freeing throughput — but it only helps when prefixes actually *repeat*; a per-request timestamp at the top of every prompt defeats it.
- C) It caches the output tokens so repeated questions are free.
- D) It only works with speculative decoding enabled.

---

**Q5.** When do you reach for tensor parallelism vs pipeline parallelism?

- A) Tensor parallel across nodes; pipeline parallel within a node.
- B) Tensor parallel splits each layer's matrices *across* GPUs (heavy per-step all-reduce, use within a single NVLink-connected node when the model doesn't fit on one GPU or for single-node throughput); pipeline parallel splits the model into *layer-range stages* (lighter inter-stage communication, use across nodes for very large models). The week's 14B-on-one-H100 needs neither.
- C) They're interchangeable; pick either.
- D) Both are required for any model larger than 1B parameters.

---

**Q6.** vLLM serves the OpenAI-compatible surface. To point the `openai` Python client at your self-hosted server, you must:

- A) Rewrite your agent to use a vLLM-specific SDK.
- B) Set the client's `base_url` to the vLLM server (e.g. `http://localhost:8000/v1`) and pass any non-empty string as `api_key` (vLLM doesn't authenticate by default; the client library merely requires a non-empty value).
- C) Pass your real OpenAI API key so vLLM can validate it.
- D) Use `curl` only — the Python client can't target a custom host.

---

**Q7.** In a LiteLLM `config.yaml`, you give *two* `model_list` entries the **same** `model_name` but different `api_base` values. What does that do?

- A) It's a config error — names must be unique.
- B) It declares a load-balanced pool: LiteLLM routes requests for that public name across both backends (per the `routing_strategy`, e.g. `least-busy`), which is how you spread load across two vLLM replicas.
- C) It makes the second entry a fallback for the first automatically.
- D) It doubles the price charged per request.

---

**Q8.** What does a LiteLLM `fallbacks` rule like `- qwen-local: ["claude-fallback"]` do, and why does it matter for week 24?

- A) It load-balances qwen-local and claude-fallback evenly.
- B) If a request to `qwen-local` (the self-hosted pool) errors, LiteLLM transparently retries it on `claude-fallback` (the vendor) — same OpenAI response shape, no client change. Week 24's chaos drill kills a replica/pool and verifies this failover catches the requests instead of erroring.
- C) It caches Claude's responses for qwen-local.
- D) It disables qwen-local permanently.

---

**Q9.** Speculative decoding's payoff is governed by the **acceptance rate**. What is it, and what does low acceptance cause?

- A) The fraction of GPU memory used; low acceptance causes OOM.
- B) The fraction of speculatively-proposed tokens the target model accepts; low acceptance means most proposals are rejected, so you spend draft compute for wasted work and can end up *slower* than no speculation at all.
- C) The percentage of requests that succeed; low acceptance drops requests.
- D) The cache hit rate; low acceptance disables prefix caching.

---

**Q10.** Why does a speculative-decoding latency win measured at concurrency 1 often *disappear* at concurrency 128?

- A) Speculation is disabled above concurrency 32.
- B) At concurrency 1 the GPU has idle compute that speculation's verification work soaks up "for free"; at concurrency 128 continuous batching has already filled the compute, so the verification competes with other requests' decoding and rejected proposals waste a now-scarce resource — the batch is already saturated.
- C) The acceptance rate is always zero above concurrency 8.
- D) High concurrency forces the draft model offline.

---

**Q11.** Self-hosted cost-per-million-tokens is GPU $/hr ÷ (tokens/sec × 3600) × 1e6. Why is benchmarking at concurrency 1 misleading about the economics?

- A) Concurrency 1 over-counts tokens, inflating the denominator.
- B) At concurrency 1 the GPU is memory-bound and mostly idle, so tokens/sec is tiny (~40), making $/MTok huge (~$17) — *worse* than vendors. The loaded server (concurrency 32/128) produces an order-of-magnitude-larger tokens/sec and ~$0.32/MTok. You must measure the *loaded* server.
- C) The formula is wrong at concurrency 1.
- D) Concurrency 1 can't be measured.

---

**Q12.** Self-hosting one H100 at $2.50/hr costs $1,800/month fixed, and you compare against `claude-haiku-4-5` at a blended $3/MTok. The break-even monthly volume is ~600M tokens. What does "break-even" mean here?

- A) The point where the GPU is fully utilized.
- B) The monthly token volume at which the self-hosted *fixed* cost ($1,800) equals the vendor's *variable* bill (volume × $/token): `1,800 ÷ $3e-6 = 600M tokens`. Below it the vendor is cheaper (idle GPU dominates); above it self-hosting wins as the fixed cost amortizes.
- C) The volume at which speculative decoding pays off.
- D) The point where tokens/sec stops climbing.

---

**Q13.** `gpu_memory_utilization` is the fraction of GPU memory vLLM may claim, with the headroom after weights becoming the KV cache. What's the failure mode of setting it *too high* (e.g. 0.98)?

- A) The server refuses to start.
- B) A burst of long requests (or another process touching the GPU) tips you into a CUDA out-of-memory crash — the server dies rather than degrades. (Too *low*, by contrast, wastes KV cache and plateaus throughput.)
- C) The KV cache shrinks, lowering concurrency.
- D) Prefix caching is disabled.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Iteration-level scheduling: finished sequences leave and queued ones join every decode step, so no dead slots and no head-of-line blocking; static batching waits for its longest member. (Lecture 1 §2, §7.)
2. **B** — Paged, on-demand KV blocks with a per-sequence block table eliminate the 60–80% fragmentation of contiguous `max_model_len` reservation, fitting far more sequences. (Lecture 1 §3, §8.)
3. **B** — Reserving max length per sequence while using a fraction is ~17× over-reservation; that reserved-but-empty KV is unusable concurrency, which PagedAttention's on-demand blocks reclaim (~16× more sequences in the worked example). (Lecture 1 §8.)
4. **B** — Prefix caching hashes/reuses a shared prefix's KV blocks so it's prefilled once (lower TTFT, freed throughput), but only when prefixes repeat; a per-request timestamp at the top defeats it. (Lecture 1 §4.)
5. **B** — TP = wider (split each layer across GPUs, heavy all-reduce, single node, model-too-big / single-node throughput); PP = taller (layer-range stages, lighter comms, across nodes). The 14B-on-one-H100 needs neither. (Lecture 1 §5.4–5.5, §9.)
6. **B** — Set `base_url` to the vLLM server and pass any non-empty `api_key`; vLLM doesn't authenticate by default. (Lecture 2 §1.2.)
7. **B** — Two entries sharing a `model_name` form a load-balanced pool routed per `routing_strategy` (e.g. `least-busy`) — how you spread load across replicas. (Lecture 2 §3.1–3.2.)
8. **B** — On a `qwen-local` failure, LiteLLM retries on the vendor `claude-fallback` transparently (same response shape); week 24's drill kills a replica/pool to verify this failover. (Lecture 2 §3.3.)
9. **B** — Acceptance rate is the fraction of proposed tokens the target accepts; low acceptance wastes draft compute and can be slower than no speculation. (Lecture 2 §4.2.)
10. **B** — At concurrency 1 idle compute makes verification free; at 128 continuous batching has filled the compute, so verification competes and rejected proposals waste a scarce resource. (Lecture 2 §4.3.)
11. **B** — Concurrency 1 is an idle, memory-bound GPU: tiny tokens/sec, ~$17/MTok; the loaded server is ~$0.32/MTok. Measure loaded. (Lecture 2 §5.1, §5.4.)
12. **B** — Break-even is where fixed self-hosted cost equals the vendor's variable bill: `$1,800 ÷ $3e-6 = 600M tokens/month`; below → vendor, above → self-host. (Lecture 2 §5.3.)
13. **B** — Too-high `gpu_memory_utilization` risks an OOM crash on a long-request burst or a shared GPU; too-low wastes KV and plateaus throughput. (Lecture 1 §5.1, §9.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
