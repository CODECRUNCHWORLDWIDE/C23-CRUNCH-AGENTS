# Mini-Project — `crunchserve`: The Inference Benchmark Harness

> Build a reusable inference-benchmarking module that any team can point at any OpenAI-compatible endpoint to measure prefill/decode tokens/sec, TTFT, p50/p95 latency, VRAM, and aggregate throughput under concurrency — so "which engine, at what quant, and how do you know?" becomes a command, not an argument.

This is the artifact that turns local-inference choices from folklore into a measurement. After this week, picking a serving engine is `python -m crunchserve bench --engines ollama,llamacpp,vllm --concurrency 1,8,32,128` and reading a table — not copying a config from a blog post. The harness is engine-agnostic (anything OpenAI-compatible), measurement-honest (prefill and decode separated, the concurrency curve, p95 not just mean), and it produces the bakeoff memo you defend at the Phase I milestone.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This module is the measurement discipline you carry into **week 19 (vLLM in production)**, where you benchmark throughput at concurrency 1/8/32/128 on an H100 and write a cost-per-million-tokens break-even memo — using exactly this harness's numbers. The capstone serves its local tier on a vLLM cluster; *this* is how you'll know it can take the load. Build it well now; you'll lean on it for the rest of the course.

---

## What you will build

A small Python package `crunchserve` with four deliverables:

1. **`crunchserve/engines.py`** — a uniform `Engine` interface over the OpenAI-compatible endpoints (Ollama, llama.cpp, vLLM, and any other OpenAI-compatible server), so the rest of the code never has to remember which port or model name each one uses. One interface; the per-engine quirks (base URL, model id, quant label) live in the registry.
2. **`crunchserve/bench.py`** — the measurement core: stream a request, time TTFT (prefill) and decode rate separately, fire N concurrent requests, and compute aggregate tokens/sec, p50/p95 latency, and the concurrency curve. (Port and harden your Exercise 3 harness.)
3. **`crunchserve/report.py`** — the comparison: collect a row per (engine, concurrency), render the table and the ASCII throughput curve, and pick a winner by a chosen metric (default: aggregate throughput at the target concurrency), flagging quant asymmetries.
4. **`crunchserve/cli.py`** — a `bench` command that ties it together and prints the comparison table with a winner line and a serving recommendation.

By the end you have a public repo of ~400–500 lines of Python that any future serving decision can `python -m crunchserve bench ...` and stop guessing about engine and quant.

---

## Why a module and not a notebook

You could do all of this in a Jupyter notebook. Don't — not as the artifact. A module gives you:

- **Reuse.** Week 19 re-runs this exact benchmark on an H100; a notebook gets copy-pasted, drifts, and rots.
- **A fixed measurement.** The prompt set, the concurrency sweep, the warm-up-discard and median-of-repeats discipline live in code, version-controlled. "Is vLLM faster here?" is answered by re-running the *same* `bench.py`, not by eyeballing a new cell.
- **A CLI.** `bench --engines vllm,ollama --concurrency 1,8,32,128` is greppable, scriptable, and CI-able. A notebook cell is none of those.

Notebooks are great for *exploring* a single engine's behavior by eye. The thing you ship and depend on is a module. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchserve/
├── pyproject.toml
├── README.md                   # the bakeoff results table + the winner memo
├── prompts/
│   └── bench_prompts.jsonl     # the fixed 100-prompt set (short + RAG-length mix)
├── crunchserve/
│   ├── __init__.py
│   ├── engines.py              # the uniform Engine interface + registry
│   ├── bench.py                # TTFT/decode timing + concurrency sweep
│   ├── report.py               # table, ASCII curve, winner selection
│   └── cli.py                  # the `bench` command
└── tests/
    ├── test_metrics.py         # percentile/aggregate math is correct
    └── test_engine_registry.py # base-url/model resolution is correct
```

---

## Deliverable 1 — `engines.py` (the uniform interface)

Every engine is an OpenAI-compatible endpoint that differs only in base URL, model id, and (for honesty) a quant label. The registry hides that behind one interface so `bench.py` treats them identically.

```python
"""crunchserve.engines — one interface over OpenAI-compatible inference engines.

The per-engine quirks (port, model id, quant label) live HERE and nowhere else.
Callers just say `engines.get("vllm")` and benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Engine:
    name: str
    base_url: str
    model: str
    quant: str          # honest label: "gguf-q4km", "awq-4", "fp16" — flags confounds

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


_REGISTRY: dict[str, Engine] = {
    "ollama":   Engine("ollama",   "http://localhost:11434/v1",
                       "qwen2.5:7b", "gguf-q4km"),
    "llamacpp": Engine("llamacpp", "http://localhost:8080/v1",
                       "qwen2.5-7b", "gguf-q4km"),
    "vllm":     Engine("vllm",     "http://localhost:8000/v1",
                       "Qwen/Qwen2.5-7B-Instruct-AWQ", "awq-4"),
}


def get(name: str) -> Engine:
    if name not in _REGISTRY:
        raise ValueError(f"unknown engine: {name}; known: {list(_REGISTRY)}")
    return _REGISTRY[name]


# TODO 1: add a `register(engine: Engine)` so users can add their own endpoints
#   (a rented vLLM URL, an SGLang server) without editing this file.

# TODO 2: add an `fp16` variant of the vllm engine so the bakeoff can run the
#   FP16 control (Lecture 2 §5.1) and isolate the quant effect from the engine.
```

> **The rule the project enforces:** every engine carries a **quant label**, and the report *prints it*. If two engines you're comparing have different quant labels, the report flags it — so nobody mistakes a quant-format difference for an engine difference (Lecture 2 §5.5). Delete the label and you've reintroduced the confound.

---

## Deliverable 2 — `bench.py` (TTFT/decode timing + concurrency sweep)

The measurement core. It must:

- **Stream** each request so TTFT (prefill) is timed separately from decode (Lecture 1 §2).
- **Fire N concurrent requests** and compute **aggregate tokens/sec** (the serving metric), not just single-stream.
- Report **p50 and p95** latency, never just the mean (Lecture 2 §5.3).
- **Discard a warm-up** run and report the **median of repeats** per point (determinism, Lecture 2 §5.5).

```python
async def bench_point(engine, prompts, concurrency, max_tokens):
    """One concurrency point: fire `concurrency` requests, return the metrics row."""
    # TODO 3: gather `concurrency` streamed requests; for each, time TTFT (first
    #   token) and the decode rate (remaining tokens / remaining time). Return
    #   aggregate_tok_s (total tokens / wall), ttft p50/p95, decode/req p50.
    ...


async def sweep(engine, prompts, levels, max_tokens, repeats=3):
    """Warm up (discard), then for each concurrency level run `repeats` times and
    keep the MEDIAN by aggregate throughput. Returns one row per level."""
    # TODO 4: implement the warm-up discard + median-of-repeats discipline.
    ...
```

The non-negotiables `bench.py` enforces:

- **Prefill and decode are reported separately.** A single blended tokens/sec is forbidden — it hides which phase is the bottleneck.
- **The concurrency curve, not a point.** `sweep` always takes a list of levels; a one-point benchmark is the trap (challenge §trap).
- **VRAM is measured at the serving concurrency** (the highest level run), via `nvidia-smi` (or noted as N/A on CPU/Mac).

---

## Deliverable 3 — `report.py` (table, curve, winner)

```python
def render(rows_by_engine, target_concurrency, metric="aggregate_tok_s"):
    """rows_by_engine: {engine_name: [row per concurrency]}. Print the table, the
    ASCII throughput-vs-concurrency curve per engine, and the winner line."""
    # TODO 5: print one block per engine (its curve), then a winner line that
    #   picks by `metric` at `target_concurrency`, and a WARNING if the compared
    #   engines have different quant labels (the confound flag).
    ...
```

It should make the *shape* visible — vLLM's aggregate rising, Ollama/llama.cpp's flattening — because that shape is the decision (Lecture 2 §2).

---

## Deliverable 4 — `cli.py` (the `bench` command)

```bash
python -m crunchserve bench \
    --engines ollama,llamacpp,vllm \
    --prompts prompts/bench_prompts.jsonl \
    --concurrency 1,8,32,128 \
    --max-tokens 128 \
    --target-concurrency 32
```

It should run the sweep against each engine and print:

```
ENGINE     QUANT      C=1    C=8    C=32   C=128   TTFT_p95  VRAM_GB
ollama     gguf-q4km   32     34     33     33      3.1s      —
llamacpp   gguf-q4km   35     40     41     40      2.9s      —
vllm       awq-4       58    360   1950   3200      0.41s     8.9
--------------------------------------------------------------------
winner @ C=32 by aggregate_tok_s: vllm (1950)
  NOTE: vllm uses awq-4 while ollama/llamacpp use gguf-q4km — quant differs,
        see memo. For a single-user iteration tool, ollama is still the pick.
```

The winner line picks by the target-concurrency aggregate by default (a `--metric` flag switches it), and the memo (the README) makes the *judgment call* the table sets up: vLLM wins decisively for a concurrent server, but Ollama is still the right tool for one-user iteration — the point is a *decision for a stated workload*, printed and defended.

---

## Rules

- **You may** read the engine docs, the vLLM/PagedAttention paper, the lecture notes, and your Exercise 3 harness.
- **You must not** report a single blended tokens/sec — prefill and decode are separated, always.
- **You must not** benchmark at one concurrency point — the sweep is the measurement; one point is the trap.
- **You must** discard the warm-up and report the median of repeats — a cold-cache first run is not a measurement.
- **You must** print each engine's **quant label** and flag confounds when they differ.
- Python 3.12, `httpx`, `numpy`, plus `pytest`. The vLLM leg needs a CUDA GPU (local or rented ~$1); Ollama/llama.cpp run on a laptop, and the harness must run fully against Ollama alone for the no-GPU path.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-06-crunchserve-<yourhandle>`.
- [ ] `crunchserve` benchmarks any OpenAI-compatible endpoint via the `engines.py` registry; adding a new endpoint needs no core-code edit.
- [ ] `bench.py` streams to separate TTFT from decode, fires concurrent requests for aggregate throughput, reports p50/p95, and discards warm-ups / takes the median of repeats.
- [ ] `report.py` prints a per-engine throughput-vs-concurrency curve and a winner line, and flags quant-label confounds.
- [ ] `pytest` passes, with at least:
  - `test_metrics.py`: the percentile and aggregate-throughput math is correct on known inputs.
  - `test_engine_registry.py`: base-url/model/quant resolution is correct and unknown engines raise.
- [ ] `python -m crunchserve bench --engines ollama,llamacpp,vllm --concurrency 1,8,32,128` prints a comparison table with a winner line (the vLLM leg may be on a rented GPU; the run is documented).
- [ ] A `README.md` with the results table, the run commands, and the **one-page bakeoff memo** (the serving engine you'd ship for a concurrent workload, its numbers, why it won *for that workload*, the trade-off accepted, and the GGUF-vs-AWQ quant caveat).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Measurement honesty** | 25 | Prefill and decode separated; the concurrency *curve* (not a point); p50 AND p95; VRAM at serving concurrency; warm-ups discarded; median of repeats. |
| **Engine abstraction** | 20 | One `Engine` interface over all OpenAI-compatible endpoints; adding an endpoint needs no core edit; quant labels carried and printed. |
| **Concurrency story** | 20 | The aggregate-throughput-vs-concurrency curve is computed and rendered; the rising-vs-flat shape is visible and explained. |
| **Confound flagging** | 15 | The GGUF-vs-AWQ (and any FP16) quant asymmetry is labeled, not hidden; the report warns when compared engines differ in quant. |
| **Tests** | 10 | `test_metrics` and `test_engine_registry` green; the math is proven on known inputs. |
| **Docs & hygiene** | 10 | Clear README + bakeoff memo, no secrets, sensible commits, no `models/`/`.venv` checked in. |

**90+** is portfolio-grade and ready to re-run on an H100 in week 19. **70–89** works but has a soft confound or a missing percentile. **Below 70** means the benchmark isn't honest or reusable — fix that first, because week 19 re-runs it at scale and the capstone's serving decision rests on it.

---

## Stretch goals

- **FP16 control.** Add an `fp16` engine variant and run the FP16-vs-AWQ comparison at the *same* engine — isolating the pure quant effect (Lecture 2 §5.1).
- **Speculative-decoding measurement.** Add a flag that benchmarks vLLM with and without a 0.5B draft model; report the decode-speed lift and relate it to the acceptance rate (Lecture 2 §3.1).
- **Prefix-cache measurement.** Benchmark 32 requests sharing a long system prompt with vLLM prefix caching on vs off; report the TTFT delta (Lecture 2 §3.2).
- **CI.** A GitHub Actions workflow that runs `pytest` and a headless two-concurrency bench against an Ollama service container. Green check on every push.

---

## How this connects to the rest of C23

- **Week 5 (the agent loop)** gave you a ReAct agent; the Phase I milestone serves it on a *local* model — this harness is how you choose the engine that serves it.
- **Week 19 (vLLM in production)** re-runs this exact concurrency benchmark on an H100 and turns it into a cost-per-million-tokens break-even memo against a vendor API — your `bench.py` carries straight over.
- **Week 21 (cost engineering & routing)** routes "easy" queries to a local vLLM 7B and "hard" queries to a vendor frontier model; the local tier's throughput — measured *here* — is what makes that routing economical.

When you've finished, push the repo and take the [quiz](../quiz.md).
