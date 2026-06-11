# Mini-Project — `crunch_tools`: A Hardened Four-Tool Agent Surface, Benchmarked Frontier-vs-Local

> Build a reusable tool surface — calculator, file-read, web-fetch, Python sandbox — where every tool is validated and hardened against hostile arguments, exposed through one vendor-neutral registry, and measured on a fixed 50-task benchmark that runs identically against `claude-opus-4-8` and a local `qwen2.5:7b-instruct`. You will report tool-call accuracy, cost, and latency for both, with the method of measurement documented. Vibes do not count.

This is the artifact the syllabus names in the Week 4 hands-on: *"Build a 4-tool agent (calculator, file-read, web-fetch, Python sandbox) that runs against (a) Anthropic's API with `tool_use`, (b) a local Qwen 2.5 7B through Ollama with the same tool schema; measure tool-call accuracy on a fixed 50-task benchmark; write a defense for each tool against malicious arguments."* By the end you have a tool surface you trust against an untrusted model and a number that proves it works on both paths.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This registry becomes the tool surface your **Week 5 ReAct agent** loops over, your **Week 13 LangGraph** graph calls, and one tier of the **capstone**'s routed serving. The four hardened tools (calculator, file, web, Python) are the exact tool set the Phase I milestone requires (calculator + file-read + web-fetch at minimum). Build it well now; you'll import it for the next twenty weeks.

---

## What you will build

A small Python package `crunch_tools` with four deliverables:

1. **`crunch_tools/registry.py`** — the single source of truth. One `Tool` per capability, each carrying its name, description, JSON Schema, hardened implementation, and a validate-then-dispatch `run()`. No tool logic anywhere else.
2. **`crunch_tools/adapters.py`** — two thin adapters (Anthropic, Ollama) that translate the registry into each vendor's envelope and translate each vendor's tool call back into `(name, args)`. The *only* per-vendor code.
3. **`crunch_tools/agent.py`** — a vendor-agnostic run loop with explicit budgets (step, time) that takes an adapter + a question and returns a final answer plus a trace.
4. **`crunch_tools/benchmark.py`** — runs a fixed 50-task benchmark through both models and emits an accuracy/cost/latency report.

By the end you have a public repo of ~400–500 lines of Python (excluding tests) that any future crunch package can `from crunch_tools.registry import REGISTRY` and get a hardened, vendor-neutral tool surface.

---

## Why a registry and not per-vendor tool files

You could copy your tool definitions into an Anthropic file and an Ollama file. Don't. One registry gives you:

- **One place to harden.** The path-traversal guard, the SSRF guard, the sandbox isolation — each lives once. Duplicate them per vendor and you *will* fix a vulnerability in one copy and not the other.
- **One place to validate.** `Tool.run()` validates against the schema before dispatch, every time, for every vendor.
- **A weekend model swap.** When the frontier vendor changes — and per the README it will — you touch an adapter, not forty tool definitions.

The vendor envelope is the *only* thing that differs between models, so the vendor adapter is the *only* place vendor code belongs. That's the senior-shop convention in 2026.

---

## Package layout

```
crunch_tools/
├── pyproject.toml
├── crunch_tools/
│   ├── __init__.py
│   ├── registry.py          # the four Tools + Tool.run() (source of truth)
│   ├── tools/
│   │   ├── calculator.py     # AST-whitelist evaluator (no eval)
│   │   ├── file_read.py      # sandbox-confined (no traversal)
│   │   ├── web_fetch.py      # SSRF-guarded (no private IPs, no blind redirects)
│   │   └── python_sandbox.py # isolated execution (container or subprocess + limits)
│   ├── adapters.py           # to_anthropic / to_ollama + parse_calls
│   ├── agent.py              # vendor-agnostic loop with budgets
│   └── benchmark.py          # 50-task runner + report
├── benchmark/
│   └── tasks.jsonl           # 50 tasks, each {prompt, tool, expected}
└── test/
    ├── test_security.py      # the attacks from Exercise 3, as regression tests
    └── test_registry.py      # every schema is valid; every Tool.run validates
```

---

## Deliverable 1 — `registry.py` (the source of truth)

Define exactly four tools. Each must be **hardened** per Lecture 2 §3:

- **`calculator`** — AST-whitelist evaluator. **Never `eval()`.** Only `+ - * / ** ()` and numbers; anything else raises and returns an error.
- **`read_file`** — confined to a sandbox root via `os.path.realpath` + `startswith(SANDBOX + os.sep)`. Resolves `..` and symlinks before the check. Bounds the read size. No traversal escape.
- **`fetch_url`** — `http`/`https` only, resolves the host, refuses private/loopback/link-local/reserved IPs, `follow_redirects=False`, bounded timeout and response size. No SSRF.
- **`run_python`** — isolated execution. Prefer the **Anthropic code execution tool** (a managed sandbox) if you're on the frontier path; if you self-host, a container per call with `--network=none`, `--read-only`, a memory limit, and a wall-clock timeout that kills the container. **Never `exec()` in your own process.** Document your isolation choice in the README.

Each tool is a `Tool` with `run()` doing schema validation then dispatch, returning `(text, is_error)`:

```python
from dataclasses import dataclass
from typing import Callable
import jsonschema

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    impl: Callable[..., str]

    def run(self, args: dict) -> tuple[str, bool]:
        try:
            jsonschema.validate(args, self.input_schema)
        except jsonschema.ValidationError as e:
            return f"ERROR: invalid arguments: {e.message}", True
        try:
            return self.impl(**args), False
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}", True

REGISTRY: dict[str, Tool] = {t.name: t for t in (CALCULATOR, READ_FILE, FETCH_URL, RUN_PYTHON)}
```

> **Special case you must handle:** the Python sandbox is the one tool you cannot make safe by validating arguments — arbitrary code is arbitrary code. Its safety lives in *isolation*, not in `input_schema`. Document this distinction explicitly in your README: three tools are defended by argument validation; the fourth is defended by a sandbox. Getting this distinction right is the difference between "I validated the args" and "I understood the threat model."

---

## Deliverable 2 — `adapters.py` (the only vendor code)

Two adapters, identical interface:

```python
def to_anthropic_tools() -> list[dict]: ...
def parse_anthropic_calls(response) -> list[tuple[str, str, dict]]:  # (tool_use_id, name, args)
    ...
def to_ollama_tools() -> list[dict]: ...
def parse_ollama_calls(message) -> list[tuple[str, str, dict]]:      # (id, name, args)
    ...
```

The Ollama adapter must handle `arguments` being either a dict or a JSON string (the trap from the challenge). The Anthropic adapter must preserve the `tool_use_id` so results can be matched.

---

## Deliverable 3 — `agent.py` (vendor-agnostic loop with budgets)

A loop that takes an adapter and a question and runs until the model stops calling tools — **with explicit budgets**, because a small local model will otherwise loop forever:

- **Step budget** — max N model turns (default 8). Exceed it → return a budget-exceeded result, don't hang.
- **Time budget** — wall-clock ceiling. Exceed it → bail.

The loop body dispatches purely through `REGISTRY[name].run(args)` and contains **no tool names**. It returns the final answer plus a structured trace (every `tool_use` and `tool_result`) — that trace is what you'll narrate at the Phase I milestone.

---

## Deliverable 4 — `benchmark.py` (the number)

Fifty tasks in `benchmark/tasks.jsonl`, each `{"prompt": ..., "expected": ..., "class": ...}` where `class` is one of `calc`, `file`, `web`, `python`, `multi` (needs >1 tool), or `security` (must be refused). A task **passes** if the final answer contains `expected` (or, for `security` tasks, if the offending tool refused). Run all 50 through each model. Emit:

```
TOOL-CALL ACCURACY (50 tasks)
model                  passed   accuracy  avg_turns  avg_latency  est_cost
claude-opus-4-8        49/50    98.0%     2.3        1.1s         $0.04
qwen2.5:7b-instruct    41/50    82.0%     2.8        0.4s         $0.00

By class (qwen2.5:7b-instruct):
  calc      10/10   web       7/10    multi    5/10
  file       9/10   python    6/10    security 4/5
```

The per-class breakdown is the interesting part: it tells you *where* the local model is weak (usually multi-tool and Python), which is exactly the routing signal the capstone needs.

---

## Rules

- **You may** read the Anthropic docs, the MCP docs, the `outlines`/`jsonschema` source, and OWASP's SSRF/path-traversal cheat sheets.
- **You must not** call `eval()` or `exec()` on model-chosen input anywhere. If `grep -rn "eval(\|exec(" crunch_tools/` finds either on a model-supplied string, you've broken the project's reason to exist. (An `ast.parse(...)` is fine; `eval(...)` is not.)
- **You must not** duplicate tool logic per vendor. Tool implementations live only in `crunch_tools/tools/`. If `grep` finds tool logic in an adapter, fix the seam.
- **You must** bound every tool — read size, response size, timeout, recursion. An unbounded tool is a DoS on your own context and wallet.
- Python 3.12. `anthropic`, `ollama`, `pydantic`, `jsonschema`, `httpx`, and `pytest`. No third-party "agent framework" — this week is hand-rolled on purpose (Week 13 is when you graduate to a framework).

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-04-crunch-tools-<yourhandle>`.
- [ ] `pip install -e .` succeeds; `pytest` is green.
- [ ] All four tools are hardened: `test_security.py` reproduces the path-traversal and SSRF attacks from Exercise 3 and asserts they're refused; the calculator rejects `__import__`; the Python sandbox is isolated (document how).
- [ ] `grep -rn "eval(\|exec(" crunch_tools/` finds no `eval`/`exec` on model-supplied input.
- [ ] One registry drives both vendors; `agent.py` contains no tool names; the only vendor code is `adapters.py`.
- [ ] `python -m crunch_tools.benchmark` runs all 50 tasks against both models and prints the accuracy/cost/latency report with the per-class breakdown.
- [ ] A `README.md` in the repo root with: the benchmark table, the per-tool defense write-up (the malicious argument for each tool and how you defeat it), the three-tools-validated-vs-one-tool-isolated distinction, and one failure mode you observed on the local model.
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

This follows the canonical weekly four-axis rubric (Correctness 30 / Engineering 25 / Measurement 25 / Write-up 20), specialized to this project.

| Area | Points | What we look for |
|---|---:|---|
| **Tool correctness** | 30 | All four tools work on valid input; the registry drives both vendors with no per-vendor tool logic; budgets prevent runaway loops. CI green. |
| **Security hardening** | 25 | Path traversal, SSRF, and `eval`/`exec` are all defeated, proven by regression tests; the Python sandbox is genuinely isolated; the validated-vs-isolated distinction is articulated. |
| **Measurement** | 25 | The 50-task benchmark runs against both models; accuracy, cost, latency, and a per-class breakdown are reported with the grading method documented. A number, not a vibe. |
| **Write-up** | 20 | README explains the registry design, the per-tool defense for each malicious argument, the frontier-vs-local gap, and one observed failure mode. |

**90+** is portfolio-grade and ready to drop into Week 5's agent loop. **70–89** works but has a soft defense or a vibes-only measurement. **Below 70** means either a tool isn't actually hardened or the benchmark isn't actually measuring — fix that first. Graders are instructed to **fail vibes-only submissions**: a working demo with no benchmark number is not a "meets."

---

## Stretch goals

- **Strict mode A/B.** Turn on `"strict": True` for the Anthropic tools and re-run the benchmark. Report the accuracy and latency delta (first strict request pays a compile cost).
- **MCP-ify one tool.** Expose `read_file` as an MCP server over stdio (preview of Week 15) and consume it through the SDK's MCP helpers. Confirm the benchmark still passes through the MCP path.
- **Grammar-constrained local tools.** Drive the local model's tool-call *arguments* through `outlines`/`xgrammar` so malformed JSON becomes structurally impossible, and measure the local accuracy lift on the `multi` class.
- **CI gate.** A GitHub Actions workflow that runs `pytest` and the security regression tests on every push, plus the benchmark against the *local* model only (no API key in CI) so accuracy regressions are caught.

---

## How this connects to the rest of C23

- **Week 5 (the agent loop)** imports this exact `REGISTRY` and wraps it in a ReAct loop with step/token/time/cost budgets. Your `agent.py` budget logic is the seed of that.
- **Week 13 (LangGraph)** re-implements the loop as a state graph that calls the same tools — the registry doesn't change, only the orchestration around it.
- **Week 17 (safety)** red-teams this tool surface with prompt-injected documents that try to steer tool arguments. Your hardening is what holds (or doesn't).
- **Week 21 (routing) and the capstone** route easy tasks to the local model and hard ones to the frontier model — over this one tool surface. Your per-class benchmark breakdown is the routing signal.

When you've finished, push the repo and take the [quiz](../quiz.md).
