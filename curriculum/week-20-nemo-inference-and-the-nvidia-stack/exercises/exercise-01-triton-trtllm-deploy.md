# Exercise 1 — Triton + TensorRT-LLM Deploy

**Goal:** Build a **TensorRT-LLM engine** for a model, lay out a **Triton model repository**, launch `tritonserver`, and hit the **OpenAI-compatible endpoint** with the *same* client code you used against vLLM in week 19. You will learn the build-then-serve mechanics that make the NVIDIA stack fast — and you will see, in your own hands, the seams between the compiler (TensorRT-LLM) and the runtime (Triton).

**Estimated time:** 60 minutes. Guided.

> **GPU-gated, with a no-GPU path.** The real build needs an NVIDIA GPU. If you have one (rent a Hopper H100 at ~$2–3/hr), do Steps 1–5 for real. If you don't, read every step as a conceptual walkthrough, then do **Step 6** — the small-model / NIM path — to learn the model-repo layout and `config.pbtxt` without renting anything.

---

## Setup

Work *inside* the NVIDIA container — installing TensorRT-LLM bare-metal is a known time-sink, and the container ships a matched TensorRT-LLM + Triton + `tensorrtllm` backend:

```bash
# On the rented H100 instance, with the NVIDIA Container Toolkit installed.
# Pin a tag; this stack moves fast and a mismatched backend is a bad afternoon.
docker run --rm -it --gpus all \
    --shm-size=16g \
    -v $(pwd):/work -w /work \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 9000:9000 \
    nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 bash
```

Inside the container you have `trtllm-build`, `tritonserver`, and the TensorRT-LLM examples. Pick a model. For a *first* run, do **not** start with the 14B — start with a small model (1–2B) so the build is minutes, not tens of minutes, and the mechanics are the same. Save the 14B for the challenge.

> **Cost discipline:** decide what you'll measure *before* you start the GPU. Run Steps 1–5 in one focused session and tear the instance down the moment Step 5 works. A forgotten H100 is this week's only expensive mistake.

---

## Step 1 — Convert the checkpoint to TensorRT-LLM format

TensorRT-LLM doesn't compile HF weights directly; you first convert them to its checkpoint format. The converter is per-model and lives in the examples directory:

```bash
# For a Qwen-family model, the converter is in examples/qwen/.
cd /opt/tensorrt_llm/examples/qwen   # path varies by container tag; find it with `ls examples/`

python convert_checkpoint.py \
    --model_dir /work/models/Qwen2.5-1.5B-Instruct \
    --output_dir /work/ckpt/qwen-1.5b \
    --dtype bfloat16
```

Read the output: it writes a `config.json` + weight shards in TensorRT-LLM's checkpoint format. This is *not* the engine yet — it's the input to the compiler.

---

## Step 2 — Build the engine with `trtllm-build`

Now compile the checkpoint into an engine for *this* GPU and *this* shape envelope:

```bash
trtllm-build \
    --checkpoint_dir /work/ckpt/qwen-1.5b \
    --output_dir /work/engine/qwen-1.5b \
    --gemm_plugin bfloat16 \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --use_paged_context_fmha enable
```

The flags *are* the lesson (Lecture 1 §2.1): `--max_batch_size` / `--max_input_len` / `--max_seq_len` define the envelope the engine is optimized for and **cannot exceed without a rebuild**. `--use_paged_context_fmha enable` turns on the paged-attention path. The output `/work/engine/qwen-1.5b` is the compiled, GPU-specific artifact — fast *here*, portable nowhere.

**Confirm:** `ls /work/engine/qwen-1.5b` shows `config.json` and `rank0.engine` (the binary). That `.engine` file is what Triton serves.

---

## Step 3 — Lay out the Triton model repository

Triton needs a *model repository* — a directory of models, each with a `config.pbtxt` and a versioned artifact. The TensorRT-LLM backend ships an example repo you copy and edit:

```bash
# The all_models/inflight_batcher_llm template is the canonical starting point.
cp -r /opt/tensorrtllm_backend/all_models/inflight_batcher_llm /work/model_repo
```

The repo has four models (Lecture 1 §3.1): `preprocessing` (tokenizer), `tensorrt_llm` (the engine), `postprocessing` (detokenizer), and `ensemble` (chains them). You fill in two things in each `config.pbtxt`: the path to your tokenizer (for pre/post) and the path to your engine (for `tensorrt_llm`). The backend ships a `fill_template.py` to do this:

```bash
ENGINE=/work/engine/qwen-1.5b
TOKENIZER=/work/models/Qwen2.5-1.5B-Instruct

python /opt/tensorrtllm_backend/tools/fill_template.py -i \
    /work/model_repo/tensorrt_llm/config.pbtxt \
    triton_backend:tensorrtllm,engine_dir:${ENGINE},batching_strategy:inflight_fused_batching,max_batch_size:64

python /opt/tensorrtllm_backend/tools/fill_template.py -i \
    /work/model_repo/preprocessing/config.pbtxt \
    tokenizer_dir:${TOKENIZER},triton_max_batch_size:64

python /opt/tensorrtllm_backend/tools/fill_template.py -i \
    /work/model_repo/postprocessing/config.pbtxt \
    tokenizer_dir:${TOKENIZER},triton_max_batch_size:64

python /opt/tensorrtllm_backend/tools/fill_template.py -i \
    /work/model_repo/ensemble/config.pbtxt \
    triton_max_batch_size:64
```

The load-bearing setting is `batching_strategy:inflight_fused_batching` — that turns on **in-flight batching** at the serving layer (Lecture 1 §3.2). If you leave it off, you've crippled the very throughput lever you came here for — and any benchmark you run is meaningless. **Confirm it's set** before you serve.

---

## Step 4 — Launch `tritonserver`

Point the server at the repository:

```bash
tritonserver \
    --model-repository=/work/model_repo \
    --http-port=8000 --grpc-port=8001 --metrics-port=8002
```

Watch the load log: every model (`preprocessing`, `tensorrt_llm`, `postprocessing`, `ensemble`) should report `READY`. If `tensorrt_llm` fails to load, the usual culprits are a mismatched container tag vs engine build, or an `engine_dir` path that doesn't point at the `.engine`. **Confirm:** `curl localhost:8000/v2/health/ready` returns `200`.

---

## Step 5 — Hit the endpoint with the week-19 client

Triton's `tensorrtllm_backend` exposes an OpenAI-compatible frontend (or you launch the `openai_frontend` shim the backend ships). Either way, the point is that your week-19 client code is **unchanged but for the `base_url`**:

```python
# The SAME client you used against vLLM in week 19. Only base_url changed.
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="ensemble",   # the ensemble model name from the repo
    messages=[{"role": "user", "content": "Explain in-flight batching in one sentence."}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

If you don't have the OpenAI frontend wired, fall back to the Triton client directly to prove the engine serves:

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")
# (build the input_ids / request_output_len tensors per the ensemble's config.pbtxt,
#  then client.infer("ensemble", inputs) — the OpenAI frontend just wraps this.)
```

**You now have the NVIDIA serving path running.** Capture one latency number (time a single request) so you have a sanity check before the challenge's full benchmark. Then **tear the instance down.**

---

## Step 6 — The no-GPU path (do this if you skipped 1–5)

You can learn the *mechanics* — the model-repo layout, the `config.pbtxt`, the seam between compiler and runtime — without a GPU. Two ways:

**(a) Small-model build on a cheap/free-tier GPU.** A 1–2B model (Steps 1–2) builds in minutes and fits on a much smaller GPU than the 14B needs — a free-tier Colab T4 or a $0.30/hr instance is enough to run the *build* and see the engine appear, even if you don't benchmark it. The commands are identical; only the model is smaller.

**(b) A NIM container — no build at all.** A NIM packages the engine + Triton + OpenAI API in one image, so you skip Steps 1–4 entirely and learn the *serving* mechanics:

```bash
# Pull a small-model NIM from NGC (free developer tier; needs an NVIDIA NGC key).
# This is the "no build" escape hatch from Lecture 1 §5.
docker run --rm -it --gpus all \
    -e NGC_API_KEY=$NGC_API_KEY \
    -p 8000:8000 \
    nvcr.io/nim/<vendor>/<small-model>:<tag>
# Then hit http://localhost:8000/v1/chat/completions with the Step-5 client.
```

**(c) Pure paper exercise.** If you have no GPU access at all, write out — by hand — the four `config.pbtxt` files for an ensemble (`preprocessing` / `tensorrt_llm` / `postprocessing` / `ensemble`), and annotate each load-bearing line: which backend, where in-flight batching is enabled, how the ensemble wires the three models. You'll do this anyway in the challenge's config sketch; doing it now makes the repo layout stick.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] **(GPU path)** `trtllm-build` produced an engine (`rank0.engine`) for your chosen model, and you can state what the `--max_batch_size` / `--max_seq_len` flags baked in.
- [ ] **(GPU path)** `tritonserver` loads the `ensemble` (and the three sub-models) to `READY`, and `inflight_fused_batching` is confirmed set in the `tensorrt_llm` config.
- [ ] **(GPU path)** The OpenAI-compatible endpoint (or the Triton client) returns a real completion, using the *same* client code as week 19 with only `base_url` changed.
- [ ] **(No-GPU path)** You completed Step 6 (small-model build, a NIM, or the hand-written config sketch) and can explain the seam: TensorRT-LLM *compiles*, Triton *serves*, the ensemble *chains* tokenizer → engine → detokenizer.
- [ ] You can state, in one sentence, *why* the engine is fast here and useless elsewhere (compiled for this GPU's kernels; not portable).
- [ ] If you used a GPU, you **tore the instance down** and recorded the cost.

---

## Stretch

- **FP8 build.** Re-run Steps 1–2 with the FP8 quantization path (Lecture 1 §2.4) and diff the engine size and a single-request latency against BF16. FP8 on Hopper is where the hardware win is largest — see it.
- **Two models, one server.** Add a second small model to the repository and confirm `tritonserver` loads both. That's the mixed-model-fleet advantage (Lecture 1 §3.4) in miniature — one server, two models — that vLLM can't do in one process.
- **Break the envelope.** Send an input longer than `--max_input_len` and watch the engine reject or truncate it. That failure *is* the inflexibility-for-speed trade made visible.

---

When this feels comfortable, move to [Exercise 2 — Guardrails injection block](exercise-02-guardrails-injection-block.py).
