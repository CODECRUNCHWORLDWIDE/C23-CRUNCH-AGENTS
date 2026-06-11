# Week 20 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 21. Answer key is at the bottom — don't peek.

---

**Q1.** In the NVIDIA inference stack, what does **TensorRT-LLM** actually produce?

- A) A trained model from scratch.
- B) A compiled, hardware-specific inference *engine* — fused kernels plus a plan tuned for your GPU (e.g. an H100) — built from a model checkpoint with `trtllm-build`, which Triton then serves.
- C) A vector database for retrieval.
- D) An OpenTelemetry exporter.

---

**Q2.** What is **Triton Inference Server**'s role, distinct from TensorRT-LLM?

- A) Triton compiles the kernels; TensorRT-LLM serves them.
- B) Triton is the serving runtime — it loads models from a model repository (each with a `config.pbtxt`), runs mixed model fleets / ensembles, exposes HTTP/gRPC (and an OpenAI-compatible) frontend, and uses the TensorRT-LLM *backend* to run the compiled engine.
- C) Triton is a training framework.
- D) Triton is NVIDIA's name for vLLM.

---

**Q3.** TensorRT-LLM's **in-flight batching** is NVIDIA's name for which vLLM concept?

- A) PagedAttention.
- B) Continuous batching — iteration-level scheduling that adds and removes sequences from the running batch as they start and finish, instead of waiting for a full static batch.
- C) Tensor parallelism.
- D) Prefix caching.

---

**Q4.** Where does **FP8 quantization** give TensorRT-LLM its largest hardware-specific advantage?

- A) On CPU-only inference.
- B) On Hopper-class GPUs (H100/H200), whose tensor cores have native FP8 support, so the compiled engine runs FP8 matmuls at higher throughput than BF16 with little quality loss — a win vLLM can approach but that is most native here.
- C) Only during training.
- D) FP8 is unrelated to TensorRT-LLM.

---

**Q5.** **NeMo Framework** and **NeMo Inference** are:

- A) The same thing under two names.
- B) Different layers — NeMo *Framework* is the training/customization layer (pretraining, SFT, alignment at scale); NeMo *Inference* (and NIM, the packaged form) is the production *serving* layer. You can serve a model you never trained.
- C) Both training-only.
- D) Both observability tools.

---

**Q6.** In NeMo Guardrails, prompt injection is stopped at which rail type?

- A) The output rail.
- B) The **input** rail — it inspects the incoming user message (e.g. via a `self check input` rail) and can `stop` the flow before the message ever reaches the main model.
- C) The retrieval rail.
- D) The dialog rail.

---

**Q7.** What are `RailsConfig` and `LLMRails`?

- A) Two competing guardrail products.
- B) `RailsConfig` is the loaded configuration (Colang flows + YAML rails settings, via `from_content` or `from_path`); `LLMRails` is the runtime you call `generate()` / `generate_async()` on, which applies the configured rails around the model.
- C) Triton backends.
- D) TensorRT-LLM build flags.

---

**Q8.** Why is benchmarking **NeMo+Guardrails against bare vLLM** not automatically apples-to-apples?

- A) Because the two serve different models.
- B) Because the rail adds extra LLM round-trip(s) per request (the `self check input` call, and any output rail) that the bare-vLLM number doesn't include — so the railed latency is higher *by design*. You must either add an equivalent filter to the vLLM side or report the rail's latency cost explicitly.
- C) Because vLLM cannot serve Qwen2.5-14B.
- D) Because Guardrails only runs on a GPU.

---

**Q9.** Why measure the **benign pass-rate** alongside the attack-success-rate (ASR)?

- A) ASR already captures everything.
- B) ASR only counts attacks; a too-aggressive rail can also block *legitimate* messages that resemble an attack (false positives), which is invisible in ASR and visible only in the benign pass-rate. A rail that blocks attacks *and* benign traffic isn't a policy, it's an outage.
- C) The benign pass-rate is required by Triton.
- D) They always move together, so one suffices.

---

**Q10.** A production rail uses a **small `self_check` model** (e.g. a haiku-class model) separate from the big `main` answer model. Why?

- A) Small models are more accurate at everything.
- B) The checker only answers a yes/no ("is this an injection?"), which a small, fast, cheap model does well — so the rail adds a *cheap* extra round-trip instead of doubling the cost/latency of the expensive answer model.
- C) The big model cannot run rails.
- D) NeMo Guardrails forbids using the main model for checks.

---

**Q11.** State the honest **NeMo-vs-vLLM** trade-off.

- A) NeMo is strictly better.
- B) NeMo wins on NVIDIA-specific kernel performance (TensorRT-LLM/FP8) and integrated policy tooling (Guardrails); vLLM wins on flexibility, OSS velocity, and operational simplicity. Score it by *your* threat model and stack, not by abstract tooling strength.
- C) vLLM is strictly better.
- D) They are identical; the choice is cosmetic.

---

**Q12.** You run the ASR evaluation with the rail **off** and again with it **on**. What does the rail-off number give you?

- A) Nothing useful.
- B) The *baseline* ASR — how many attacks the bare model already lets through — so the rail-on-vs-off delta is the rail's measured value. Without the baseline you can't claim the rail did anything.
- C) The benign pass-rate.
- D) The model's training accuracy.

---

**Q13.** Where does **NeMo Guardrails** run relative to GPUs, and why does that matter for this week's lab?

- A) It requires an H100 like TensorRT-LLM.
- B) It's CPU-reachable — it's just policy logic plus LLM calls to a configurable endpoint (a Triton/vLLM target *or* `claude-opus-4-8` via the anthropic engine), so the Guardrails leg of the week runs with no GPU even though the TensorRT-LLM/Triton build is GPU-gated.
- C) It runs only inside Triton.
- D) It cannot call a remote model.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — TensorRT-LLM compiles a hardware-specific engine via `trtllm-build`; Triton serves it. (Lecture 1 §2.)
2. **B** — Triton is the serving runtime (model repository, `config.pbtxt`, backends, OpenAI-compatible frontend) that runs the TensorRT-LLM engine. (Lecture 1 §3.)
3. **B** — In-flight batching is NVIDIA's continuous batching: iteration-level scheduling. (Lecture 1 §2.)
4. **B** — FP8 is most native on Hopper tensor cores, where the compiled engine's advantage is largest. (Lecture 1 §2.)
5. **B** — Framework = training/customization; Inference/NIM = serving; you can serve a model you didn't train. (Lecture 1 §4, §5.)
6. **B** — The input rail (e.g. `self check input`) stops injection before the message reaches the model. (Lecture 2 §2, §3.)
7. **B** — `RailsConfig` is the loaded config; `LLMRails` is the runtime you call `generate()` on. (Lecture 2 §2.)
8. **B** — The rail's extra LLM call(s) raise railed latency by design; equalize or report it, or the comparison is dishonest. (Lecture 2 §6, §6.5.)
9. **B** — ASR misses false positives; only the benign pass-rate catches a rail that also blocks legitimate users. (Lecture 2 §6, §6.5.)
10. **B** — A small checker answers the yes/no cheaply, making the rail a cheap round-trip instead of doubling the expensive call. (Lecture 2 §6.5.)
11. **B** — NeMo: kernel perf + policy tooling; vLLM: flexibility + OSS velocity + operational simplicity; score by your threat model. (Lecture 1 §6; Lecture 2 §7.)
12. **B** — Rail-off is the baseline ASR; the on-vs-off delta is the rail's measured value. (Lecture 2 §6.5.)
13. **B** — Guardrails is CPU-reachable (policy + LLM calls to any endpoint, incl. `claude-opus-4-8`), unlike the GPU-gated TensorRT-LLM/Triton build. (Lecture 2 §1, §5.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
