# Week 6 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 7. Answer key is at the bottom — don't peek.

---

**Q1.** What are the two phases of LLM inference, and what bounds each?

- A) Encode and decode; both are compute-bound.
- B) Prefill (processes the whole prompt in parallel; compute-bound; sets TTFT) and decode (generates one token at a time, reading all the weights per token; memory-bandwidth-bound; sets tokens/sec).
- C) Training and inference; both are memory-bound.
- D) Prompt and response; both are network-bound.

---

**Q2.** You quantize a 7B from FP16 to 4-bit and single-stream decode gets ~3.4× faster. Why does quantization help decode so much?

- A) 4-bit matmuls are 3.4× more FLOP-efficient.
- B) Decode is memory-bandwidth-bound — every token reads the whole model — so cutting the weights from ~14 GB to ~4 GB reads ~3.4× less per token, roughly 3.4× faster decode.
- C) Quantization removes the KV cache.
- D) It doesn't; quantization only helps prefill.

---

**Q3.** Which engine is the right *first* tool for friction-free local iteration on a laptop, and why?

- A) vLLM, because it has the highest throughput.
- B) Ollama, because `ollama pull` + `ollama run` gets a quantized model serving an OpenAI-compatible endpoint with the least friction; it wraps llama.cpp with a registry and sensible defaults.
- C) TensorRT-LLM, because it's the most optimized.
- D) TGI, because it's from Hugging Face.

---

**Q4.** Why is vLLM the right choice for a server taking concurrent traffic, where llama.cpp is not?

- A) vLLM has a nicer CLI.
- B) vLLM does continuous batching and paged attention, so it keeps the GPU saturated across many concurrent requests (aggregate throughput rises with concurrency); single-stream engines serialize and their aggregate flattens.
- C) llama.cpp can't load 7B models.
- D) vLLM uses less VRAM in every case.

---

**Q5.** What does *continuous* (in-flight) batching do that static batching does not?

- A) It trains the model between requests.
- B) It adds and removes requests from the running batch token-by-token, so a finished request's slot is immediately refilled and the GPU never idles waiting for the slowest request in a fixed batch.
- C) It increases the context window.
- D) It disables the KV cache.

---

**Q6.** What problem does paged attention solve?

- A) It makes the model smaller on disk.
- B) The KV cache, if reserved as a contiguous worst-case block per request, fragments VRAM and wastes it; paged attention stores the cache in fixed-size pages (OS-style virtual memory) allocated on demand, so many more sequences fit in the same VRAM.
- C) It removes the need for quantization.
- D) It speeds up prefill specifically.

---

**Q7.** On the perplexity-vs-size quantization curve, why is Q4_K_M called "the knee"?

- A) Because it's the smallest possible quant.
- B) Because from FP16 down to ~Q4 perplexity barely moves while size and memory traffic drop ~3.4×; below Q4 (Q3, Q2) perplexity climbs sharply — so Q4_K_M captures almost all the savings just before quality cost accelerates.
- C) Because it's the only quant llama.cpp supports.
- D) Because it's exactly 4 bits per parameter.

---

**Q8.** Your 4-bit 7B is "only ~4 GB," yet it OOMs serving 64 concurrent 16k-token requests. Why?

- A) 4-bit models secretly use 16-bit at runtime.
- B) The KV cache also consumes VRAM and scales with context × concurrency; at long context and high concurrency the cache can dwarf the quantized weights and blow the budget.
- C) vLLM has a memory leak.
- D) Postgres ran out of connections.

---

**Q9.** How does speculative decoding speed up generation without losing quality?

- A) It lowers the temperature.
- B) A small draft model proposes several tokens; the big model verifies them in one forward pass, accepting the prefix it agrees with — so you get multiple tokens per big-model pass, and the big model still has final say on every token.
- C) It skips attention for some tokens.
- D) It runs the model at lower precision for hard tokens.

---

**Q10.** You benchmark vLLM and llama.cpp at concurrency 1 and they're close, so you conclude llama.cpp is fine for your server. What's the error?

- A) None; concurrency 1 is the standard benchmark.
- B) A serving engine's character shows in the throughput-vs-concurrency *curve*: vLLM's aggregate rises with concurrency (continuous batching) while llama.cpp's flattens (it serializes). At concurrency 1 you measured iteration speed, not serving capacity.
- C) You should have used a larger model.
- D) llama.cpp can't serve at all.

---

**Q11.** You compare vLLM-AWQ against llama.cpp-GGUF-Q4_K_M and vLLM wins. What's the honest conclusion?

- A) vLLM's engine is definitively faster.
- B) You changed *two* things (engine *and* quant format), so you've shown "vLLM+AWQ beats llama.cpp+GGUF," not that vLLM's engine alone is faster — to isolate the engine you'd run the same quant on both, or run an FP16 control.
- C) AWQ is broken.
- D) GGUF can't be used on a GPU.

---

**Q12.** Which is the correct reason to self-host an LLM (as opposed to a bad one)?

- A) Self-hosting is always cheaper than a vendor API.
- B) Vendor independence, data residency/privacy, cost at *sustained high* utilization, or latency/control — and you should still be *able* to fall back to a vendor; "it's cooler" is the bad reason that leads to an idle GPU.
- C) Local models are always more accurate.
- D) Vendor APIs don't support streaming.

---

**Q13.** When would you reach for SGLang over vLLM?

- A) Never — vLLM is strictly better.
- B) On structured-output-heavy or shared-prefix-heavy workloads (lots of requests sharing a long system prompt or a tree of related prompts), where SGLang's aggressive prefix caching (RadixAttention) pulls ahead.
- C) Only when you have no GPU.
- D) When you need to train a model.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Prefill (compute-bound, parallel over the prompt, sets TTFT) and decode (memory-bandwidth-bound, one token at a time reading all weights, sets tokens/sec). (Lecture 1 §2.)
2. **B** — Decode is memory-bandwidth-bound; 4-bit reads ~3.4× fewer bytes per token. The speedup is the size ratio, not a FLOP effect. (Lecture 1 §2; Lecture 2 §1.)
3. **B** — Ollama: lowest friction, wraps llama.cpp with a registry and OpenAI-compatible endpoint; the iteration tool. (Lecture 1 §3.)
4. **B** — Continuous batching + paged attention keep the GPU saturated under concurrency; aggregate throughput rises where single-stream engines flatten. (Lecture 1 §5; Lecture 2 §2.)
5. **B** — Token-by-token add/remove of requests so finished slots refill immediately and the GPU never idles for the slowest fixed-batch member. (Lecture 2 §2.2.)
6. **B** — Paged, on-demand KV-cache allocation (OS-style virtual memory) stops fragmentation and lets many more sequences share VRAM. (Lecture 2 §2.3.)
7. **B** — The knee: near-flat quality from FP16 to ~Q4, sharp climb below Q4. Q4_K_M captures the savings before quality cost accelerates. (Lecture 2 §1.2.)
8. **B** — KV cache scales with context × concurrency and can exceed the quantized weights; that's the forgotten VRAM cost. (Lecture 2 §1.3, §4.)
9. **B** — Draft proposes K tokens, big model verifies in one pass and accepts the agreed prefix; multiple tokens per big-model pass, no quality loss. Lift set by acceptance rate. (Lecture 2 §3.1.)
10. **B** — The concurrency curve is the engine's character; concurrency 1 measures iteration, not serving. vLLM rises, llama.cpp flattens. (Lecture 1 §5; Lecture 2 §2, §5.2.)
11. **B** — Two changes (engine + quant). Isolate by holding the quant constant or running an FP16 control. (Lecture 2 §5.1, §5.5.)
12. **B** — Independence, residency, sustained-volume cost, latency/control — and stay *able* to use a vendor; "it's cooler" → idle GPU. (Lecture 1 §1.)
13. **B** — SGLang wins on structured/shared-prefix workloads via RadixAttention prefix caching. (Lecture 1 §6; Lecture 2 §3.2.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
