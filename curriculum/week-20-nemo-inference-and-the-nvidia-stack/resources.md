# Week 20 — Resources

Every resource here is **free** or has a free tier. The NVIDIA stack (TensorRT-LLM, Triton Inference Server, NeMo Framework, NeMo Guardrails) is open source on GitHub; the docs live on `docs.nvidia.com`. NIM containers are pulled from NVIDIA NGC (free developer tier). NeMo Guardrails runs **CPU-only** against any OpenAI-compatible or Anthropic endpoint, so the policy half of the week needs no GPU and no NVIDIA account at all.

Versions in this stack move *fast* — TensorRT-LLM and Triton ship roughly monthly, and a `config.pbtxt` field or a `trtllm-build` flag that was current last quarter may be renamed. The *concepts* (compiled engines, in-flight batching, paged KV cache, the model repository, input/output rails, Colang flows) are stable. When a specific flag or page 404s, search the repo for the symbol name and read the version of the docs that matches your installed container tag.

This week sits on top of **week 19** (the vLLM baseline you compare against) and **week 17** (the prompt-injection threat model the Guardrails rail defends). The resources below assume you have both.

## Required reading (work it into your week)

- **Triton Inference Server documentation** — the model-repository layout, `config.pbtxt`, backends, ensembles, and the OpenAI-compatible frontend. The canonical serving-runtime reference:
  <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html>
- **Triton Inference Server (GitHub)** — the server, the `server` repo's `docs/`, and the example model repositories:
  <https://github.com/triton-inference-server/server>
- **TensorRT-LLM documentation** — `trtllm-build`, the LLM API, in-flight batching, paged KV cache, FP8/quantization on Hopper. The kernel/engine reference:
  <https://nvidia.github.io/TensorRT-LLM/>
- **TensorRT-LLM (GitHub)** — the source, the `examples/` directory (per-model build recipes, including Qwen), and the Triton `tensorrtllm_backend`:
  <https://github.com/NVIDIA/TensorRT-LLM>
- **NeMo Guardrails documentation** — `RailsConfig`, `LLMRails`, input/output/dialog/retrieval rails, Colang, the self-check-input and jailbreak rails. The policy-layer reference for this week:
  <https://docs.nvidia.com/nemo/guardrails/>
- **NeMo Guardrails (GitHub)** — the source, the example configs, and the Colang reference. Read the `nemoguardrails` API and the `docs/` example rails:
  <https://github.com/NVIDIA/NeMo-Guardrails>
- **NeMo Framework documentation** — the training/customization stack (survey depth this week); read the *overview* and the *export-to-inference* path so you understand the hand-off to TensorRT-LLM:
  <https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html>
- **NVIDIA NIM overview** — the packaged, container-shipped form of NeMo inference (engine build + Triton + OpenAI-compatible API in one image). When the deploy unit is a container, not a config:
  <https://docs.nvidia.com/nim/>

## The kernel-optimization references

- **TensorRT-LLM — in-flight batching** — NVIDIA's name for continuous batching; how requests join and leave a batch mid-flight. The throughput lever:
  <https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html>
- **TensorRT-LLM — paged KV cache** — the paged attention KV-cache manager (the same idea vLLM popularized, in NVIDIA's compiled implementation):
  <https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-management.html>
- **TensorRT-LLM — quantization (FP8 / INT4 / SmoothQuant)** — the Hopper FP8 path where the hardware-specific win is largest; read the FP8 section twice:
  <https://nvidia.github.io/TensorRT-LLM/reference/precision.html>
- **TensorRT-LLM — `trtllm-build` and the LLM API** — the two ways to compile an engine: the CLI (`trtllm-build`) and the modern Python `LLM` API. Know both:
  <https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html>
- **NVIDIA Hopper architecture (H100) whitepaper** — why FP8 and the Transformer Engine exist on Hopper, and why a kernel compiled for it is fast *there*:
  <https://resources.nvidia.com/en-us-tensor-core>

## The Guardrails / Colang references

- **NeMo Guardrails — Colang language guide** — flows, canonical forms (`define user ...`, `define bot ...`), and how a flow matches and acts. The rail-authoring language:
  <https://docs.nvidia.com/nemo/guardrails/latest/colang-language-syntax-guide.html>
- **NeMo Guardrails — input rails / self-check input** — the rail type you use to block prompt injection before it reaches the model; the `self check input` flow and its prompt:
  <https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-library.html>
- **NeMo Guardrails — jailbreak detection** — the heuristic + model-based jailbreak rail; a heavier alternative to a hand-written self-check rail:
  <https://docs.nvidia.com/nemo/guardrails/latest/user-guides/jailbreak-detection-deployment.html>
- **NeMo Guardrails — configuration (`config.yml` / `RailsConfig`)** — the YAML that wires models, rails, and flows; `RailsConfig.from_content` vs `from_path`:
  <https://docs.nvidia.com/nemo/guardrails/latest/user-guides/configuration-guide.html>
- **OWASP Top 10 for LLM Applications — LLM01: Prompt Injection** — the threat model from week 17 that this week's rail defends against; re-read LLM01 so the rail targets a real class:
  <https://genai.owasp.org/llmrisk/llm01-prompt-injection/>

## The honest NeMo-vs-vLLM comparison reading

- **vLLM documentation** — the baseline stack from week 19; you compare against its throughput, latency, and operational model. Read the *engine arguments* and *production* pages so the comparison is fair:
  <https://docs.vllm.ai/>
- **TensorRT-LLM vs vLLM — NVIDIA's own benchmarks** — read these *critically*: they are vendor numbers on vendor hardware with a vendor's engine build. Useful for the kernel-perf axis, not for the operational-simplicity axis:
  <https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/performance>
- **Triton vs a single-framework server** — the multi-model-fleet argument for Triton (one server, many backends, ensembles) vs vLLM's one-model-per-process simplicity:
  <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md>
- **A note on lock-in** — there is no single canonical paper; the honest framing is in the lecture: NeMo wins on NVIDIA-specific kernel perf and policy tooling, vLLM wins on flexibility, OSS velocity, and operational simplicity. Build your own table from the two stacks' docs, not from a vendor slide.

## Tools you'll use this week

- **`tensorrt-llm`** — `pip install tensorrt-llm` (Linux + NVIDIA GPU; the container `nvcr.io/nvidia/tritonserver:<tag>-trtllm-python-py3` is the reliable path). Provides `trtllm-build` and the `tensorrt_llm.LLM` API. **GPU-gated.**
- **Triton Inference Server** — pulled as the NGC container above; you run `tritonserver --model-repository=...`. **GPU-gated** for the `tensorrtllm` backend.
- **`tritonclient`** — `pip install tritonclient[all]`. The Python client for Triton's HTTP/gRPC endpoints; or just use the **`openai`** client against Triton's OpenAI-compatible frontend (or a NIM).
- **`nemoguardrails`** — `pip install nemoguardrails`. **CPU-only**, runs anywhere. `RailsConfig.from_content(...)` / `from_path(...)`, `LLMRails`, `rails.generate(...)` / `generate_async(...)`. The policy half of the week.
- **`anthropic`** — `pip install anthropic`. If you put `claude-opus-4-8` behind a rail, use it as an LLM-judge, or use it as the comparison endpoint. `client.messages.create(...)`, `thinking={"type":"adaptive"}`, `output_config={"effort":...}`.
- **`numpy`** — for the decision-matrix scoring in Exercise 3 and the mini-project. Pure stdlib otherwise.

## A note on the rented-GPU recipe + CPU-reachable Guardrails path

The GPU-gated legs (TensorRT-LLM engine build, Triton serving, the 14B benchmark) want a real Hopper H100. Rent one at **~$2–3/hr (2026)**, budget **3–4 hours** (~$8–12), and **tear it down the moment the benchmark is captured**. If you cannot rent a GPU, the engine-build and Triton mechanics are documented conceptually in Exercise 1 with a **small-model path** (a 1–2B model that builds in minutes and fits on a much cheaper or free-tier GPU) so the model-repository layout and `config.pbtxt` are still learnable.

The **Guardrails** half — Exercise 2, the rail in the mini-project, the ASR before/after — runs **CPU-only** against `claude-opus-4-8` (or any OpenAI-compatible endpoint), so it needs no GPU and no NVIDIA account. Exercise 2 also degrades to a **mock LLM + heuristic rail** if `nemoguardrails` or an API key isn't present, so it *always* runs. The decision tool (Exercise 3) is pure stdlib + numpy and runs anywhere. **You can complete the entire policy and decision spine of this week without renting a single GPU-hour** — the GPU is only for the kernel-perf numbers.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **TensorRT-LLM** | NVIDIA's LLM kernel/engine *compiler* — turns a model into an optimized engine for a specific GPU. |
| **Engine** | The compiled, GPU-specific artifact `trtllm-build` produces. Fast *here*; not portable. |
| **`trtllm-build`** | The CLI that compiles a checkpoint into a TensorRT-LLM engine (the LLM API is the Python equivalent). |
| **In-flight batching** | NVIDIA's name for continuous batching — requests join/leave the batch mid-generation. |
| **Paged KV cache** | Paged-attention KV-cache management (the idea vLLM popularized), in TensorRT-LLM's compiled form. |
| **FP8** | 8-bit float quantization; on Hopper (H100) it's where TensorRT-LLM's hardware-specific speedup is largest. |
| **Triton Inference Server** | NVIDIA's multi-model, multi-backend *serving runtime* — one server, many models, ensembles. |
| **Model repository** | Triton's directory of models; each has a `config.pbtxt` and a versioned model artifact. |
| **`config.pbtxt`** | The per-model Triton config: backend, inputs/outputs, batching, instance groups. |
| **`tensorrtllm` backend** | The Triton backend that serves a TensorRT-LLM engine. |
| **Ensemble** | A Triton model that chains other models (e.g. tokenizer → engine → detokenizer) into one endpoint. |
| **NeMo Framework** | NVIDIA's *training and customization* stack (pretraining, SFT, alignment); upstream of inference. |
| **NeMo Inference / NIM** | The *packaged production-serving* form — NIM ships engine + Triton + OpenAI API as one container. |
| **NeMo Guardrails** | The *policy* layer — programmable rails (input/output/dialog/retrieval) around the model. |
| **Colang** | The language you write rails in — flows, canonical forms (`define user`, `define bot`). |
| **`RailsConfig` / `LLMRails`** | The Python config object and the runtime that applies rails to a model. |
| **Input rail** | A rail that inspects/blocks the *user input* before it reaches the model (where injection is stopped). |
| **Output rail** | A rail that inspects/blocks the *model output* before it reaches the user (where leakage is stopped). |
| **Self-check input** | The built-in rail that asks an LLM "should this user message be allowed?" — the anti-injection workhorse. |
| **ASR (attack-success-rate)** | Fraction of attack prompts that succeed; you measure it before and after adding a rail. |
| **Lock-in** | The cost of the NVIDIA win: a compiled engine and an opinionated stack tie you to one vendor's silicon. |

---

*If a link 404s, please open an issue so we can replace it — and check your installed container tag, since this stack's docs are versioned and move fast.*
