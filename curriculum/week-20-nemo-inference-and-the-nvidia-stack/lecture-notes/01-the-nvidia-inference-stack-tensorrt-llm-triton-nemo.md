# Lecture 1 — The NVIDIA Inference Stack: TensorRT-LLM, Triton, and the Honest vLLM Trade-off

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can name the four layers of the NVIDIA inference stack and say what each one does; build a TensorRT-LLM engine with `trtllm-build` (or the LLM API) and explain *why* a compiled engine with in-flight batching, paged KV cache, and FP8 can beat vLLM on Hopper; lay out a Triton model repository and launch `tritonserver` with the OpenAI-compatible frontend; and state the honest NeMo-vs-vLLM trade-off — kernel perf and policy tooling vs flexibility, OSS velocity, and operational simplicity — without reciting a vendor slide.

If you remember one sentence from this entire week, remember this one:

> **NVIDIA's stack is the production answer if you are an NVIDIA shop. It is also the most opinionated. Know what you are signing up for.**

Last week you served Qwen2.5-14B with vLLM and tuned it until you had a number. This week you serve the *same* model on the *same* H100 the NVIDIA way and get a *different* number — usually a faster one, on this hardware, with more setup and more lock-in. The job of this lecture is to make you able to explain *both halves of that sentence*: why the NVIDIA stack is genuinely faster here, and exactly what you signed away to get the speed.

There's a corollary to tape next to last week's vLLM mantra:

> **A compiled kernel beats an interpreted one on the hardware it was compiled for — and only there.** TensorRT-LLM's win is real and it is NVIDIA-specific. The moment you might run on something else, that win becomes a lock.

---

## 1. The four layers, and how they fit

The biggest source of confusion about "the NVIDIA stack" is that people say it as if it's one thing. It's four things, stacked, and each has a different job. Get the layering wrong and you'll spend an afternoon trying to make Triton train a model (it doesn't) or asking TensorRT-LLM to serve HTTP (it doesn't). Here is the whole stack, bottom to top:

```
┌─────────────────────────────────────────────────────────────┐
│  NeMo Inference / NIM   ── packaged production serving        │  <- the deploy unit
│  (engine + Triton + OpenAI-compatible API, in one container)  │
├─────────────────────────────────────────────────────────────┤
│  Triton Inference Server ── the multi-model serving runtime   │  <- serves over HTTP/gRPC
│  (model repository, backends, ensembles, OpenAI frontend)     │
├─────────────────────────────────────────────────────────────┤
│  TensorRT-LLM           ── the kernel / engine COMPILER       │  <- makes the fast artifact
│  (trtllm-build -> engine; in-flight batching; paged KV; FP8)  │
├─────────────────────────────────────────────────────────────┤
│  NeMo Framework         ── training & customization           │  <- upstream of inference
│  (pretraining, SFT, alignment; export to a deployable ckpt)   │
└─────────────────────────────────────────────────────────────┘
```

Read it as a pipeline that runs once at *build* time and then serves forever:

- **NeMo Framework** is where a model is *trained or customized* — pretraining, supervised fine-tuning, alignment. You will mostly *not* run this layer this week (that's the survey-depth part), but you need to know it exists, because it is the upstream source of the checkpoint you compile. If your company actually fine-tunes Qwen on its own data, NeMo Framework (or an equivalent) is where that happens, and the output is a checkpoint.
- **TensorRT-LLM** takes a checkpoint and **compiles** it into an **engine** — a GPU-specific, optimized artifact. This is the kernel-optimization layer, and it's where the speed comes from. `trtllm-build` is the compiler; the engine is the binary.
- **Triton Inference Server** takes one or more engines (and other model artifacts) and **serves** them over HTTP/gRPC, with batching, multiple model versions, ensembles, and an OpenAI-compatible frontend. Triton is the *runtime*; TensorRT-LLM is the *compiler*. Triton can serve a TensorRT-LLM engine, a PyTorch model, an ONNX model, a Python script — many backends, one server.
- **NeMo Inference / NIM** is the *packaged* form of all of the above: a NIM (NVIDIA Inference Microservice) is a container that already contains a compiled engine for a specific model + a configured Triton + an OpenAI-compatible API. You `docker run` it and you have a serving endpoint. NIM trades control (you don't pick the build flags) for convenience (you don't have to).

The seams matter. The hand-off from NeMo Framework to TensorRT-LLM is "here is a trained checkpoint, compile it." The hand-off from TensorRT-LLM to Triton is "here is an engine + a `config.pbtxt`, serve it." The hand-off from Triton to your application is "here is an HTTP endpoint." NIM collapses the middle two seams into one container. Every time you debug this stack, you are debugging *one* of these layers, and naming which one is half the battle.

> **The mental model:** TensorRT-LLM *compiles*, Triton *serves*, NeMo Framework *trains*, NIM *packages*. vLLM, by contrast, is one process that does the compile-equivalent (graph capture, kernel selection) and the serving in the same Python server. That single-process simplicity is exactly vLLM's operational advantage — and the four-layer split is exactly NVIDIA's flexibility advantage. Hold both.

---

## 2. TensorRT-LLM — the kernel optimization layer

This is where the NVIDIA stack earns its speed. The pitch is simple and it's true: instead of *interpreting* a model graph at run time (loading PyTorch ops, dispatching CUDA kernels op-by-op), TensorRT-LLM **compiles** the model into a fused, GPU-specific engine ahead of time. The compiler can do things an interpreter can't: fuse adjacent operations into one kernel, pick the optimal kernel for *this* GPU's tensor cores, bake in the sequence shapes, and use Hopper-only instructions like FP8 matmuls. The result is a binary that runs the model with far less per-op overhead.

### 2.1 The engine build

The build is the defining step. You take a checkpoint (HF weights or a NeMo checkpoint), convert it to TensorRT-LLM's checkpoint format, and compile it for a target GPU and a target *shape envelope* (max batch size, max input length, max output length). Two ways to do it.

**The CLI (`trtllm-build`)** — the classic, explicit path:

```bash
# 1. Convert HF weights to the TensorRT-LLM checkpoint format (per-model script).
#    For Qwen2.5, the converter lives in TensorRT-LLM/examples/qwen/.
python convert_checkpoint.py \
    --model_dir ./Qwen2.5-14B-Instruct \
    --output_dir ./qwen2.5-14b-ckpt \
    --dtype bfloat16

# 2. Compile the checkpoint into an engine for THIS GPU and THIS shape envelope.
trtllm-build \
    --checkpoint_dir ./qwen2.5-14b-ckpt \
    --output_dir ./qwen2.5-14b-engine \
    --gemm_plugin bfloat16 \
    --max_batch_size 256 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --use_paged_context_fmha enable
```

The flags *are* the lesson. `--max_batch_size`, `--max_input_len`, `--max_seq_len` define the envelope the engine is optimized for — the compiler bakes these in, which is why the engine is fast but inflexible: ask it to serve a 6000-token input when you built for 4096 and it will refuse or truncate. `--gemm_plugin bfloat16` selects the matmul kernels. `--use_paged_context_fmha enable` turns on the paged attention path. Every one of these is a build-time decision you cannot change without rebuilding — that's the trade for the speed.

**The modern LLM API** — the Pythonic path that hides the two-step build behind one object:

```python
from tensorrt_llm import LLM, SamplingParams

# The LLM API builds (or loads a cached) engine on construction, for THIS GPU.
llm = LLM(
    model="Qwen/Qwen2.5-14B-Instruct",
    tensor_parallel_size=1,          # one H100
    max_batch_size=256,
    max_seq_len=8192,
)

outputs = llm.generate(
    ["Explain paged KV cache in one sentence."],
    SamplingParams(max_tokens=128),   # NOTE: no temperature games — this is a kernel demo
)
print(outputs[0].outputs[0].text)
```

The LLM API is the right choice for prototyping and for the build-it-once-then-serve pattern; `trtllm-build` is the right choice when you want explicit control over every build flag and a reproducible engine artifact to ship. They produce the same kind of engine.

### 2.2 In-flight batching — NVIDIA's continuous batching

Week 19 taught you vLLM's **continuous batching**: instead of waiting for a fixed batch to finish, the server admits new requests and evicts finished ones token-by-token, keeping the GPU saturated. TensorRT-LLM has the same idea under a different name: **in-flight batching** (sometimes "in-flight sequence batching"). A request can join the running batch mid-generation and leave when it emits its stop token, without stalling its neighbors. This is *the* throughput lever for variable-length LLM workloads, and both stacks have it — so it is **not** where NVIDIA's advantage comes from. If you compare a TensorRT-LLM engine *without* in-flight batching against a vLLM server *with* continuous batching, you are measuring a configuration mistake, not a stack difference. (That trap is the headline-lab trap; remember it.)

### 2.3 Paged KV cache

Same story. vLLM popularized **paged attention**: the KV cache is stored in fixed-size pages, so memory isn't fragmented by variable sequence lengths and the server can pack far more concurrent sequences into the same VRAM. TensorRT-LLM has a **paged KV cache** too — its own compiled implementation of the same idea (`--use_paged_context_fmha`, the KV-cache manager). Again: both stacks have it. The KV cache is where most of your VRAM goes at serving time, and paging it is table stakes in 2026, not a differentiator. The differentiator is *below* this — in the kernels.

### 2.4 FP8 and quantization on Hopper — where the win actually is

Here is the real source of NVIDIA's advantage on the H100: **FP8**. Hopper has a Transformer Engine with native FP8 tensor cores — 8-bit floating-point matmuls at roughly double the throughput of BF16, with a quality loss small enough to be acceptable for most serving. TensorRT-LLM compiles directly to these FP8 instructions:

```bash
# Quantize the checkpoint to FP8, then build an FP8 engine for Hopper.
python quantize.py \
    --model_dir ./Qwen2.5-14B-Instruct \
    --output_dir ./qwen2.5-14b-fp8-ckpt \
    --dtype bfloat16 \
    --qformat fp8 \
    --kv_cache_dtype fp8

trtllm-build \
    --checkpoint_dir ./qwen2.5-14b-fp8-ckpt \
    --output_dir ./qwen2.5-14b-fp8-engine \
    --gemm_plugin fp8 \
    --max_batch_size 256
```

An FP8 engine on an H100 can serve meaningfully more tokens per second than a BF16 engine — that's the headroom that lets TensorRT-LLM out-throughput vLLM on the *same* H100. But read the trade carefully:

- **FP8 is Hopper-specific.** The whole point is the H100's FP8 tensor cores. Compile this engine and you have an artifact that is fast *on Hopper* and meaningless anywhere else. That is the corollary mantra made concrete: the kernel win is real and it is a lock.
- **FP8 trades a little quality for a lot of throughput.** You must *measure* the quality delta (perplexity, or task accuracy on your eval set), not assume it's free. For most chat/RAG serving it's fine; for a high-stakes task it might not be.
- **vLLM also supports FP8** — so this isn't a feature vLLM lacks. The difference is that TensorRT-LLM's *ahead-of-time compiled* FP8 path tends to squeeze more out of the hardware than vLLM's runtime path, on Hopper, today. That gap narrows every release. Which is exactly why you *measure it this week* instead of trusting a number from last quarter.

> **The honest version of "TensorRT-LLM is faster":** on NVIDIA Hopper, with a properly-built FP8 (or even BF16) engine and in-flight batching, TensorRT-LLM usually delivers higher throughput and lower latency than vLLM on the same GPU — because a compiler that targets the exact silicon beats a runtime that targets a family of silicon. That advantage is real, it is hardware-specific, and it costs you a build step, an opinionated stack, and portability. You don't take it on faith; you take it on a benchmark, which is the challenge.

---

## 3. Triton Inference Server — the serving runtime

TensorRT-LLM gives you an engine — a `.engine` file and some config. It does not give you an HTTP server, batching across requests, model versioning, or a way to serve five models from one process. That's Triton's job. **Triton Inference Server** is the runtime that turns engines (and PyTorch models, ONNX models, Python scripts) into a production endpoint.

### 3.1 The model repository

Triton is driven by a **model repository** — a directory with one subdirectory per model, each containing a `config.pbtxt` and one or more versioned model artifacts. For a TensorRT-LLM LLM you typically have an *ensemble* of a few models wired together. The layout:

```
model_repository/
├── tensorrt_llm/                 # the compiled engine, served by the tensorrtllm backend
│   ├── config.pbtxt
│   └── 1/                         # version 1
│       └── <the trtllm engine files>
├── preprocessing/                # tokenizer: text -> input_ids  (a Python backend model)
│   ├── config.pbtxt
│   └── 1/model.py
├── postprocessing/               # detokenizer: output_ids -> text (a Python backend model)
│   ├── config.pbtxt
│   └── 1/model.py
└── ensemble/                     # chains preprocessing -> tensorrt_llm -> postprocessing
    ├── config.pbtxt
    └── 1/
```

The `1/` directories are *version* directories — Triton serves versioned models, and you can have `1/`, `2/`, `3/` live at once and route between them. That versioning is one of Triton's quiet advantages over a single-model server: you can canary a new engine version under the same endpoint.

### 3.2 `config.pbtxt` — the per-model config

Each model declares its backend, inputs, outputs, and batching in a `config.pbtxt` (protobuf text). A sketch for the `tensorrt_llm` model:

```protobuf
name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: 256

model_transaction_policy {
  decoupled: true        # streaming: many output tokens per one input request
}

input [
  { name: "input_ids"      data_type: TYPE_INT32  dims: [ -1 ] },
  { name: "input_lengths"  data_type: TYPE_INT32  dims: [ 1 ] reshape: { shape: [ ] } },
  { name: "request_output_len" data_type: TYPE_INT32 dims: [ 1 ] reshape: { shape: [ ] } }
]
output [
  { name: "output_ids"  data_type: TYPE_INT32  dims: [ -1, -1 ] }
]

instance_group [
  { count: 1  kind: KIND_GPU }   # one engine instance, on the GPU
]

parameters: {
  key: "gpt_model_type"      value: { string_value: "inflight_fused_batching" }  # in-flight batching ON
}
parameters: {
  key: "gpt_model_path"      value: { string_value: "/models/tensorrt_llm/1" }
}
```

The load-bearing lines: `backend: "tensorrtllm"` selects the TensorRT-LLM backend; `decoupled: true` enables streaming (token-by-token output); `gpt_model_type: "inflight_fused_batching"` turns on in-flight batching at the *serving* layer (it must match what the engine supports); `instance_group` controls how many copies of the engine run and on which devices. This file is where you tune the serving-side behavior of the engine you compiled.

### 3.3 Launching the server and the OpenAI-compatible frontend

You point `tritonserver` at the repository and it loads every model:

```bash
tritonserver \
    --model-repository=/models \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002
```

Triton exposes its own HTTP/gRPC protocol, but in 2026 the path you'll actually use is the **OpenAI-compatible frontend** — Triton (and NIM) can expose a `/v1/chat/completions` endpoint that speaks the OpenAI wire format, so your application code doesn't change when you swap vLLM for Triton:

```python
# Same client code you used against vLLM in week 19 — only the base_url changed.
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="qwen2.5-14b",
    messages=[{"role": "user", "content": "Explain in-flight batching in one sentence."}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

That OpenAI compatibility is what makes the week-19-vs-week-20 comparison *fair and cheap*: you reuse the exact same load-generation client, point it at a different `base_url`, and measure. No application rewrite. (NIM exposes the same OpenAI-compatible endpoint, which is why a NIM is a drop-in if you don't want to build the engine and the model repo yourself.)

### 3.4 Mixed model fleets and ensembles — Triton's real argument

Here is where Triton stops being "a way to serve one engine" and becomes worth its complexity: **mixed model fleets**. One Triton server can serve your 14B LLM engine, a reranker, an embedding model, and a small classifier — different backends (TensorRT-LLM, ONNX, PyTorch, Python) — from one process, one port, one set of metrics. And it can wire them into an **ensemble**: a single logical model that internally runs tokenizer → LLM → detokenizer (or retriever → reranker → LLM) as one request. The `ensemble` model's `config.pbtxt` declares the data flow:

```protobuf
name: "ensemble"
platform: "ensemble"
max_batch_size: 256
ensemble_scheduling {
  step [
    { model_name: "preprocessing"  model_version: -1
      input_map  { key: "QUERY"      value: "text_input" }
      output_map { key: "input_ids"  value: "_input_ids" } },
    { model_name: "tensorrt_llm"   model_version: -1
      input_map  { key: "input_ids" value: "_input_ids" }
      output_map { key: "output_ids" value: "_output_ids" } },
    { model_name: "postprocessing" model_version: -1
      input_map  { key: "output_ids" value: "_output_ids" }
      output_map { key: "text_output" value: "text_output" } }
  ]
}
```

This is the multi-model-fleet advantage vLLM simply does not have: vLLM is *one model per process*. If you serve five models, that's five vLLM processes, five ports, five sets of ops. Triton is one server for all of them, with ensembles to chain them. For a *single* LLM, that flexibility is pure overhead — vLLM's one-process simplicity wins. For a *fleet*, Triton's architecture is the point. Which stack is "simpler" depends entirely on how many models you serve.

---

## 4. NeMo Framework — the training layer (survey depth)

We will not run a training job this week, but a serving engineer needs to know what the bottom layer is for. **NeMo Framework** is NVIDIA's stack for *building* models, not serving them: large-scale pretraining, supervised fine-tuning (SFT), parameter-efficient fine-tuning (LoRA/PEFT), and alignment (RLHF/DPO-style methods). It's built for multi-GPU, multi-node training with the parallelism (tensor, pipeline, sequence) that frontier-scale training needs.

For this week, the only thing you must understand is the **hand-off**: NeMo Framework produces a *checkpoint*, and that checkpoint is the input to the TensorRT-LLM engine build. The path is `NeMo Framework (train/customize) → export checkpoint → TensorRT-LLM (compile to engine) → Triton (serve) → NIM (package)`. If your organization fine-tunes Qwen on proprietary data, that's the NeMo-Framework layer; the inference stack you spend this week on is everything *downstream* of that. The reason NVIDIA can make a coherent end-to-end pitch — "train, optimize, and serve in one stack" — is precisely this vertical integration. The reason it's *opinionated* is the same vertical integration: it's smoothest when you use all of it, and rougher when you want to train in one ecosystem and serve in another.

> **Survey-depth takeaway:** NeMo Framework = training & customization; it is *upstream* of everything you serve this week. You don't run it, but you should be able to say what it does and where it hands off — because "we trained it in NeMo, so we serve it in NeMo" is a real (and real-sounding) argument you'll hear, and you should be able to weigh it instead of nodding.

---

## 5. NIM — the packaged form

A **NIM (NVIDIA Inference Microservice)** is the whole serving stack in one container: a pre-built, pre-optimized engine for a specific model + a configured Triton + an OpenAI-compatible API. You `docker run` a NIM and you have a `/v1/chat/completions` endpoint, no `trtllm-build`, no `config.pbtxt`, no model repository to lay out. NIM is to the NVIDIA stack what a managed database is to a self-hosted one: it trades control for convenience.

When NIM is the right call: you want NVIDIA-stack performance without owning the build, you're serving a model NVIDIA ships a NIM for, and you'd rather pin a container tag than maintain a TensorRT-LLM build pipeline. When it isn't: you need a custom-fine-tuned model NVIDIA doesn't ship, you need build flags NIM doesn't expose, or you can't run NVIDIA's container licensing. For this week, a NIM is the no-build escape hatch in Exercise 1 — if you can't build the engine yourself, pull a small-model NIM and hit its endpoint to learn the *serving* mechanics without the *build* mechanics.

The three ways to stand the same model up, on one line each, so you can place any deployment you meet:

| Path | What you own | What you give up |
|---|---|---|
| **`trtllm-build` + Triton (hand-rolled)** | Every build flag (FP8, batching, parallelism), the model repo, the Triton config — full control | You maintain the build pipeline and the config; most operational weight |
| **NIM (packaged)** | A container tag and an endpoint | Build flags NIM doesn't expose; you serve what NVIDIA ships; container licensing |
| **vLLM `serve` (next week's baseline)** | One command, OSS, simplest ops | NVIDIA-specific kernel wins TensorRT-LLM can extract; the integrated policy story |

Read the table as a control-vs-convenience axis: hand-rolled TensorRT-LLM is maximum control and maximum operational weight; NIM trades control for a container tag; vLLM trades the kernel ceiling for one-command simplicity. The same Qwen2.5-14B can be served by all three — the choice is *which costs you can live with at 2 AM*, which is exactly the decision the challenge memo forces. And note the asymmetry that makes this week's comparison fair: the model repository and `config.pbtxt` are a **deployment artifact** — versioned, reviewable, diffable — the same property week-18 prized in traces. vLLM's equivalent is a command line and flags; neither is wrong, but the NVIDIA stack pushes you toward declarative, checked-in serving config, which is a real operational-maturity point in its favor even as it's an operational-weight point against it.

---

## 6. The honest NeMo-vs-vLLM trade-off

Now the part that matters most for the capstone decision. Do not reduce this to "NeMo is faster" or "vLLM is simpler." Both are true and incomplete. Here is the honest, axis-by-axis trade, the one you'll score in the decision matrix:

| Axis | NeMo / TensorRT-LLM / Triton | vLLM |
|---|---|---|
| **Kernel perf on NVIDIA HW** | **Wins.** Compiled, FP8 on Hopper, ahead-of-time fused kernels. The throughput/latency edge is real on the H100. | Strong and improving, but a runtime targeting a family of GPUs rarely beats a compiler targeting the exact one. |
| **Policy / safety tooling** | **Wins.** NeMo Guardrails is a first-class, programmable policy layer (Lecture 2). Nothing in the vLLM ecosystem is as complete. | You bolt on your own filter / a separate Guardrails install / an external service. Doable, less integrated. |
| **Flexibility** | Opinionated. Build envelope baked in; rebuild to change shapes; tied to NVIDIA silicon. | **Wins.** Runs more model architectures faster after release, no build step, change config and restart. |
| **OSS velocity** | Fast, but vendor-paced and vendor-shaped. | **Wins.** Huge community, new models supported within days, public roadmap. |
| **Operational simplicity** | A four-layer stack (engine build + Triton repo + versioning + container). Powerful, more to operate. | **Wins.** One process, one port, `pip install vllm`, point your OpenAI client at it. |
| **Lock-in** | High. The engine is Hopper-specific; the stack assumes NVIDIA end-to-end. | Low. Portable across vendors as backends mature; no compiled artifact to strand you. |
| **Mixed model fleets** | **Wins.** One Triton serves many models + ensembles. | One model per process; a fleet is N processes. |

Read the table as a *decision*, not a scoreboard. If you are an NVIDIA shop, serving on H100s, you care about safety policy, and you run a fleet of models — NeMo's column is full of wins and the lock-in is a cost you were already paying. If you might run on multiple vendors, you ship fast, you serve one or two models, and a small team operates it — vLLM's column is full of wins and NeMo's perf edge isn't worth the operational weight. **There is no universal answer. There is the answer for *your* constraints, justified by *your* benchmark.** That's the memo you write this week.

> **The mantra, decoded:** "NVIDIA's stack is the production answer if you are an NVIDIA shop" — the perf and policy wins are concentrated exactly where you've committed to NVIDIA silicon. "It is also the most opinionated" — the four-layer split, the build envelope, the vertical integration are the price of those wins. "Know what you are signing up for" — measure the perf edge (don't trust the slide), price the lock-in (don't ignore it), and choose on purpose.

---

## 7. Recap

You should now be able to:

- Name the **four layers** — TensorRT-LLM (compiler), Triton (serving runtime), NeMo Framework (training), NeMo Inference / NIM (packaged serving) — and say where each hands off to the next.
- **Build a TensorRT-LLM engine** with `trtllm-build` (or the LLM API), read the build flags as the inflexibility-for-speed trade, and state *why* a compiled engine beats an interpreted runtime *on the hardware it targets*.
- Identify the real source of the win: **FP8 on Hopper** (with in-flight batching and paged KV cache as table-stakes both stacks share), and know that FP8 is Hopper-specific and quality-costing, so you *measure* it.
- **Lay out a Triton model repository** (`config.pbtxt`, the `tensorrtllm` backend, the ensemble), launch `tritonserver`, hit the **OpenAI-compatible frontend**, and explain the **mixed-model-fleet** advantage over vLLM's one-model-per-process model.
- State the **honest NeMo-vs-vLLM trade-off** axis by axis — kernel perf and policy tooling vs flexibility, OSS velocity, and operational simplicity — and frame it as a constraint-dependent decision, not a winner.

Next: the policy layer. NeMo Guardrails is the second half of NVIDIA's production pitch — the programmable safety rails that block a class of prompt injection *as policy*, the answer to week 17's threat model. Continue to [Lecture 2 — NeMo Guardrails as Policy](./02-nemo-guardrails-as-policy.md).

---

## References

- *Triton Inference Server documentation* (model repository, `config.pbtxt`, backends, ensembles): <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html>
- *TensorRT-LLM documentation* (`trtllm-build`, the LLM API, in-flight batching, paged KV): <https://nvidia.github.io/TensorRT-LLM/>
- *TensorRT-LLM (GitHub)* (the `examples/qwen/` build recipe, the Triton backend): <https://github.com/NVIDIA/TensorRT-LLM>
- *TensorRT-LLM — quantization / FP8 on Hopper*: <https://nvidia.github.io/TensorRT-LLM/reference/precision.html>
- *Triton Inference Server (GitHub)* (architecture, ensembles, the OpenAI-compatible frontend): <https://github.com/triton-inference-server/server>
- *NeMo Framework documentation* (training/customization, export to inference): <https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html>
- *NVIDIA NIM overview* (packaged engine + Triton + OpenAI API): <https://docs.nvidia.com/nim/>
- *vLLM documentation* (the week-19 baseline this stack is compared against): <https://docs.vllm.ai/>
