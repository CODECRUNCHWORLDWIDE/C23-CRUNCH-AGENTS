# Mini-Project — `crunchserve`: The Self-Hosted Serving + Routing + Benchmarking Harness

> Build a reusable harness that fronts vLLM replicas with a LiteLLM router (plus a vendor fallback), sweeps concurrency to measure tokens/sec and p50/p95/p99 latency, computes self-hosted cost-per-million-tokens and the break-even volume, and emits a **serving decision memo** — so "self-host or vendor, and how do you know?" becomes a command, not an argument.

This is the artifact that turns self-hosting economics from folklore into a measurement. After this week, deciding your serving tier is `python -m crunchserve decide --gpu-usd-per-hr 2.50 --expected-mtok-per-month 1000` and reading a memo — not guessing from a blog post. The harness is workload-agnostic, replica-pluggable, and number-honest, and it produces the exact pattern the capstone serves from: vLLM behind LiteLLM with a vendor fallback.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This harness is what the **capstone (weeks 22–24)** serves its local tier from, and the LiteLLM fallback you build here is the safety net **week 24's chaos drill** kills a replica to verify. Week 20 (the NVIDIA stack) benchmarks against the *same* workload this harness defines, so the throughput numbers are comparable across serving backends. Build it well now; you'll lean on it for the rest of Phase IV and the whole capstone.

---

## What you will build

A small Python package `crunchserve` with four deliverables:

1. **`crunchserve/litellm_config.yaml`** — a LiteLLM proxy config: a `model_list` with one or more vLLM replicas (same `model_name` → load-balanced pool) and a vendor fallback, plus `router_settings` (`routing_strategy`, `num_retries`, `fallbacks`). The single source of truth for "how requests reach a backend and what happens when one fails."
2. **`crunchserve/benchmark.py`** — a reusable benchmark module that sweeps concurrency (1/8/32/128) against an OpenAI-compatible endpoint, records aggregate tokens/sec and p50/p95/**p99** latency and req/sec, and reads token counts from `usage` (never estimated). Ships a `--simulate` path so it runs with no GPU.
3. **`crunchserve/cost.py`** — a cost module computing self-hosted cost-per-million-tokens from tokens/sec + GPU $/hr, blended vendor prices, and the break-even monthly volume.
4. **`crunchserve/cli.py`** — a CLI (`benchmark` and `decide` commands) that ties it together and emits the **serving decision memo**.

By the end you have a public repo of ~300–450 lines of Python (plus the YAML) that any future serving project can `from crunchserve.cost import break_even` and stop guessing about self-host-vs-vendor.

---

## Why a module and not a notebook

You could do all of this in a Jupyter notebook. Don't — not as the artifact. A module gives you:

- **Reuse.** Week 20 imports your `benchmark.py` to compare vLLM against TensorRT-LLM on the *same* workload; the capstone imports your `cost.py` to justify its serving tier. A notebook gets copy-pasted, drifts, and rots.
- **A fixed measurement.** The workload, the metric, and the "read tokens from `usage`" discipline live in code, version-controlled. "Did this config change help?" is answered by re-running the *same* `benchmark.py`, not by eyeballing a new notebook cell.
- **A CLI.** `crunchserve decide --gpu-usd-per-hr 2.50` is greppable, scriptable, and CI-able. A notebook cell is none of those.

Notebooks are great for *exploring* a single server's behavior by eye. The thing you ship and depend on is a module. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchserve/
├── pyproject.toml
├── README.md                   # the throughput table + the serving memo
├── crunchserve/
│   ├── __init__.py
│   ├── litellm_config.yaml     # vLLM replicas (load-balanced) + vendor fallback
│   ├── workload.py             # the fixed chat workload (system + user prompts, max_tokens)
│   ├── benchmark.py            # concurrency sweep -> tokens/sec, p50/p95/p99, req/sec
│   ├── cost.py                 # cost-per-MTok, blended vendor prices, break-even
│   └── cli.py                  # the `benchmark` and `decide` commands
└── tests/
    ├── test_cost.py            # the cost + break-even math is correct
    └── test_benchmark.py       # the --simulate curve climbs; usage parsing is right
```

No external service is required to *develop* the harness — `benchmark.py --simulate` and `cost.py` run with stdlib + numpy. The real vLLM + LiteLLM path is what you point it at once you have a server.

---

## Deliverable 1 — `litellm_config.yaml` (routing + fallback)

The routing policy lives in YAML, not in application code. One or more vLLM replicas in a load-balanced pool, plus a vendor fallback:

```yaml
model_list:
  - model_name: qwen-local                        # public name clients request
    litellm_params:
      model: openai/Qwen/Qwen2.5-14B-Instruct      # "openai/" => OpenAI-compatible (vLLM)
      api_base: http://localhost:8000/v1           # replica A
      api_key: EMPTY                               # vLLM ignores it; must be non-empty
  # TODO 1: add a SECOND replica with the SAME model_name (api_base :8001) so
  #   LiteLLM load-balances across the pool. (Stretch: the chaos-drill rehearsal.)
  - model_name: claude-fallback
    litellm_params:
      model: anthropic/claude-haiku-4-5            # vendor fallback target (price to beat)
      api_key: os.environ/ANTHROPIC_API_KEY        # from env, never hardcoded

router_settings:
  routing_strategy: least-busy                     # keep every replica's batch full
  num_retries: 2
  fallbacks:
    - qwen-local: ["claude-fallback"]              # qwen-local fails -> retry on Claude
```

Run it with `litellm --config crunchserve/litellm_config.yaml` (serves an OpenAI-compatible proxy on `:4000`). The benchmark and any client hit `qwen-local`; LiteLLM decides which backend answers and falls over to the vendor on failure — the resilience pattern week 24's chaos drill verifies.

---

## Deliverable 2 — `benchmark.py` (the concurrency sweep)

The module that drives load and records the curve. It must:

- Define a **fixed workload** (`workload.py`: one shared system prompt + a short user prompt + a fixed `max_tokens`) so runs are comparable.
- Fire `concurrency` requests at a time at an OpenAI-compatible endpoint, for several rounds, at each of 1/8/32/128.
- Record aggregate **tokens/sec** (the cost denominator), **p50/p95/p99** latency, and **req/sec**.
- Read token counts from the response **`usage`** block — exact, never estimated.
- Ship a `--simulate` path (default-on when no endpoint is reachable) that models continuous-batching throughput so it runs with no GPU.

```python
import asyncio, time, statistics, numpy as np
from crunchserve.workload import SYSTEM_PROMPT, USER_PROMPT, MAX_TOKENS

async def _one(client, base_url, model_id):
    t0 = time.perf_counter()
    r = await client.post(f"{base_url}/chat/completions", json={
        "model": model_id,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": USER_PROMPT}],
        "max_tokens": MAX_TOKENS,
    }, headers={"Authorization": "Bearer EMPTY"}, timeout=120)
    out = int(r.json()["usage"]["completion_tokens"])     # EXACT token count
    return time.perf_counter() - t0, out

async def level(base_url, model_id, concurrency, rounds=3):
    # TODO 2: gather `concurrency` _one() calls per round, accumulate latencies +
    #   total tokens + wall time, and return tokens/sec, p50, p95, p99, req/sec.
    #   p99 = float(np.percentile(latencies, 99)).
    ...

def simulate_level(concurrency):
    # TODO 3: model aggregate tokens/sec as it climbs with concurrency toward a
    #   saturation ceiling (harmonic blend), so the curve runs offline. Return the
    #   SAME dict shape as level(). (Port Exercise 2's simulate model.)
    ...
```

The non-negotiable: **tokens come from `usage`, not estimates.** If you ever count tokens with `len(text.split())` you've corrupted the denominator that the entire cost model depends on.

---

## Deliverable 3 — `cost.py` (cost-per-MTok + break-even)

The module that turns tokens/sec into a dollar figure and a decision boundary:

```python
HOURS_PER_MONTH = 24 * 30  # 720

VENDOR_PRICES = {                                  # 2026 Claude prices, $/MTok in/out
    "claude-haiku-4-5":  {"in": 1.0, "out": 5.0},
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
    "claude-opus-4-8":   {"in": 5.0, "out": 25.0},
}

def cost_per_mtok(tokens_per_sec, gpu_usd_per_hr):
    """GPU $/hr / (tokens/sec * 3600) * 1e6 -- the self-hosted unit cost."""
    return gpu_usd_per_hr / (tokens_per_sec * 3600.0) * 1e6

def vendor_blended(model, io_ratio):
    # TODO 4: blend input/output price for an input:output ratio (1.0 = balanced).
    ...

def break_even_tokens(gpu_usd_per_hr, vendor_per_mtok):
    """Monthly volume where fixed GPU cost == vendor variable cost."""
    fixed_monthly = gpu_usd_per_hr * HOURS_PER_MONTH
    return fixed_monthly / (vendor_per_mtok / 1e6)
```

`cost.py` is pure arithmetic — no GPU, no network — so `test_cost.py` pins it down exactly (a known tokens/sec + $/hr → a known $/MTok and break-even volume).

---

## Deliverable 4 — `cli.py` (the `benchmark` and `decide` commands)

```bash
# Measure the curve (real server or --simulate):
python -m crunchserve benchmark --base-url http://localhost:4000 --concurrency 1,8,32,128

# Emit the serving decision memo:
python -m crunchserve decide \
    --tokens-per-sec 2180 --gpu-usd-per-hr 2.50 \
    --vendor claude-haiku-4-5 --expected-mtok-per-month 1000
```

`benchmark` prints the throughput table + an ASCII curve; `decide` prints the memo:

```
SERVING DECISION
  measured tokens/sec : 2,180 (concurrency 128)
  self-hosted $/MTok  : $0.32     vs claude-haiku-4-5 blended $3.00  (9.4x cheaper)
  fixed cost          : $1,800/month (H100 @ $2.50/hr, 24/7)
  break-even volume   : 600M tokens/month
  expected volume     : 1,000M tokens/month  -> ABOVE break-even
  -> SELF-HOST. Saves ~$1,200/month, PROVIDED we sustain the utilization that
     keeps tokens/sec near the measured peak. Configure the Claude fallback so an
     outage degrades to the vendor, not to an error.
```

The memo is the *judgment call* the numbers set up — and it states the utilization assumption explicitly, because the peak $/MTok only holds if the GPU stays busy.

---

## Rules

- **You may** read the vLLM/LiteLLM docs, the lecture notes, and your Exercise 2/3 code.
- **You must** measure tokens/sec or clearly run `--simulate` — **never fake the numbers.** A made-up tokens/sec produces a made-up decision; the whole value is that the memo is measured.
- **You must** read token counts from the response `usage` block, not estimate them.
- **You must not** hardcode the served model id — read it from `/v1/models`.
- **You must not** put secrets in the YAML — the vendor key is `os.environ/ANTHROPIC_API_KEY`.
- Any direct Claude call (the cost-comparison probe or a fallback sanity-check) uses the Anthropic SDK `client.messages.create(...)` with `thinking={"type": "adaptive"}` and `output_config={"effort": ...}` — never `budget_tokens` or `temperature`.
- Python 3.12, `numpy`, plus `httpx`/`openai` for the real path and `pytest`. The vLLM + LiteLLM path needs a server; the `--simulate` + cost path runs anywhere.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-19-crunchserve-<yourhandle>`.
- [ ] `litellm_config.yaml` defines a `qwen-local` pool and a `claude-fallback`, with a `fallbacks` rule from `qwen-local` to the vendor.
- [ ] `benchmark.py` sweeps concurrency 1/8/32/128 and reports tokens/sec, p50/p95/**p99**, and req/sec, reading tokens from `usage`; `--simulate` runs with no GPU and its curve climbs.
- [ ] `cost.py` computes cost-per-MTok and break-even volume, and `test_cost.py` proves the math against known inputs.
- [ ] `python -m crunchserve decide ...` emits a serving memo naming the decision, the tokens/sec and $/MTok behind it, the break-even volume, and the utilization assumption.
- [ ] `pytest` passes (`test_cost.py` + `test_benchmark.py` covering the simulate curve and `usage` parsing).
- [ ] A `README.md` with the throughput table and the one-page serving memo (self-host vs vendor, with the number).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Benchmark correctness** | 25 | Concurrency sweep hits 1/8/32/128; tokens/sec climbs then plateaus; tokens read from `usage`, never estimated; p50/p95/p99 reported; `--simulate` runs with no GPU. |
| **Cost & break-even math** | 25 | `cost_per_mtok` and `break_even_tokens` match the Lecture 2 formula; blended vendor prices correct; `test_cost.py` pins them to known inputs. |
| **Routing & fallback** | 20 | LiteLLM config has a load-balanced `qwen-local` pool and a working `fallbacks` rule to the vendor; the proxy is actually in the request path. |
| **The decision memo** | 15 | `decide` emits a memo that commits to self-host-or-vendor with the number, names the break-even volume, and states the utilization assumption (doesn't quote peak $/MTok to justify an average bill). |
| **Tests** | 10 | `pytest` green; cost math + simulate curve + `usage` parsing covered. |
| **Docs & hygiene** | 5 | Clear README + memo, no secrets in the YAML, sensible commits, no `__pycache__`/`.venv` checked in. |

**90+** is portfolio-grade and ready to drop into the capstone's serving tier. **70–89** works but has a soft mapping somewhere — a faked number, a concurrency-1-only table, or a memo with no utilization caveat. **Below 70** means the harness isn't a fair, reusable measurement — fix that first, because the capstone serves from this harness's pattern.

---

## Stretch goals

- **Multi-replica + failover.** Add the second `vllm serve` on `:8001` and the second `model_list` entry (TODO 1), watch `least-busy` load-balance, then kill one replica mid-benchmark and confirm the survivor serves; kill both and confirm the Claude fallback catches the requests. That's week 24's chaos drill, rehearsed.
- **Speculative-decoding lever.** Add a `--speculative` flag to your launch and re-run the sweep with n-gram speculation. Measure the latency change at concurrency 1 vs 128 and report whether the win survives a saturated batch (Lecture 2 §4.3).
- **Prefix-cache hit-rate measurement.** Send repeated requests with an identical long system prompt and measure the TTFT drop with `--enable-prefix-caching` on vs off — quantify the prefix-caching win on your workload (Lecture 1 §4).

---

## How this connects to the rest of C23

- **Week 6 (local inference)** got a model *running* — Ollama / llama.cpp / a quantized model on your laptop. `crunchserve` is the production-grade version: the same vLLM, now under load, behind a router, with the economics measured.
- **Capstone (weeks 22–24)** serves its **local tier from exactly this pattern** — vLLM replicas behind LiteLLM with a vendor fallback. Your `cost.py` is the artifact that justifies the tier existing; your `litellm_config.yaml` is the routing it runs on.
- **Week 24 (chaos drill)** **kills a replica** and verifies the failover you configured here: LiteLLM rides over the loss (load-balancing to the survivor) and, if the whole pool dies, falls over to the vendor — with the user never seeing an error. The fallback rule you wrote this week is the thing the drill stress-tests.

When you've finished, push the repo and take the [quiz](../quiz.md).
