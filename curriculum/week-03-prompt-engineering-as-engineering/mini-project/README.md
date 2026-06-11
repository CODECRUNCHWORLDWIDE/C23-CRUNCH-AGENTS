# Mini-Project — `promptlab`: A Versioned Prompt Registry With a Regression Gate

> Build a CLI tool `promptlab` that treats prompts the way you treat code: a **versioned prompt registry** (each prompt has named versions in files), a **promptfoo-style regression runner** (run a version against a golden set, score the pass rate, catch regressions), and a **structured-review report generator** (a per-version score report plus a diff-and-checklist review you can put in front of a reviewer). By the end you have one command that answers "is this prompt version better, by how much, and did it break anything?" — with numbers and a SHA, not a vibe.

This is the artifact that turns Week 3's whole thesis — *if you cannot diff it and test it, it is not a prompt, it is a wish* — into a tool you'll actually reach for. It is the from-scratch version of what promptfoo and Langfuse productize, built so you understand the mechanism end to end.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** The regression runner you build here is the harness every later prompt-bearing lab uses. **Week 4** regression-tests tool-calling prompts with it (a tool call that "usually" fires correctly is a wish until you test it). The **Phase I capstone milestone** requires "prompts versioned in git; promptfoo regression tests committed" — `promptlab` is that requirement, owned. Build it well now; you'll extend it for twenty weeks.

---

## What you will build

A small Python package `promptlab` with three deliverables:

1. **`promptlab/registry.py`** — the versioned prompt registry. Prompts live as files (`prompts/<name>.v<N>.txt`); the registry lists versions, loads a specific version, diffs two versions, and resolves a `production` label to a version (a tiny local stand-in for Langfuse prompt management).
2. **`promptlab/runner.py`** — the regression runner. Load a golden set (`(input, assertion)` pairs from YAML/JSON), run a prompt version against a model, score each example with its assertion, compute the pass rate, and compare two versions to surface **regressions** (cases that passed before and fail now). This is `exercise-02` promoted to a real, testable module.
3. **`promptlab/report.py` + `promptlab/cli.py`** — the report generator and entry point. `promptlab eval`, `promptlab diff`, and `promptlab review` print the score table, the version diff, and the structured-review report.

By the end you have a public repo of ~300–450 lines of Python (excluding tests) that you can point at any prompt + golden set and get a defensible answer.

---

## Why a registry, a runner, and a report — and why measure-don't-vibe

Three design rules are non-negotiable, because they are the lessons of the week:

- **Versions are files, never string literals.** A prompt you can't `diff` is a prompt you can't review. The registry's whole job is to make every version a first-class, addressable thing with a SHA.
- **"Better" is computed against a fixed golden set.** The runner never asks you to read outputs and judge. It runs *every* version against *all* examples and reports a pass rate, so iteration is monotonic: a version ships only if it doesn't regress. This is the regression gate from Lecture 2 §5, in code you own.
- **The report is the deliverable a reviewer reads.** Not the raw outputs — the *table* (version, rate, delta, SHA) and the *diff + checklist*. The report is what makes the prompt change defensible in a review.

---

## Package layout

```
promptlab/
├── pyproject.toml
├── prompts/
│   ├── support-triage.v1.txt
│   ├── support-triage.v2.txt
│   └── ...                       # the versions you iterate
├── golden/
│   └── support-triage.yaml       # the golden example set
├── promptlab/
│   ├── __init__.py
│   ├── registry.py               # version list/load/diff/label resolution
│   ├── runner.py                 # load golden, run a version, score, compare
│   ├── report.py                 # score table + structured-review report
│   └── cli.py                    # argparse: eval / diff / review
└── tests/
    ├── test_runner.py            # pure scoring + regression detection (no model)
    └── test_registry.py          # version listing, diff, label resolution
```

---

## Deliverable 1 — `registry.py` (the versioned registry)

Promote the file-per-version idea into a module. It must:

- **List versions** of a named prompt by scanning `prompts/<name>.v*.txt` and returning them sorted.
- **Load** a specific version's text by name + version number.
- **Diff** two versions (use `difflib.unified_diff`) and return the diff as text — the unit of review.
- **Resolve a label** (`production`, `latest`) to a version. Store labels in a tiny `prompts/<name>.labels.json` so moving `production` from v6 back to v5 is a one-line change — a local stand-in for Langfuse's rollback.

Here is the spine to start from; fill in the rest yourself:

```python
"""promptlab.registry — prompts as versioned, diffable, labelled files."""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from dataclasses import dataclass

PROMPTS_DIR = Path("prompts")


@dataclass
class PromptVersion:
    name: str
    version: int
    text: str
    path: Path


def list_versions(name: str, root: Path = PROMPTS_DIR) -> list[PromptVersion]:
    """All versions of `name`, sorted ascending by version number."""
    out = []
    for p in sorted(root.glob(f"{name}.v*.txt")):
        m = re.search(rf"{re.escape(name)}\.v(\d+)\.txt$", p.name)
        if m:
            out.append(PromptVersion(name, int(m.group(1)), p.read_text(), p))
    return sorted(out, key=lambda pv: pv.version)


def load_version(name: str, version: int, root: Path = PROMPTS_DIR) -> PromptVersion:
    p = root / f"{name}.v{version}.txt"
    if not p.exists():
        raise FileNotFoundError(f"no such prompt version: {p}")
    return PromptVersion(name, version, p.read_text(), p)


def diff_versions(name: str, a: int, b: int, root: Path = PROMPTS_DIR) -> str:
    """Unified diff between two versions — what a reviewer actually reads."""
    va, vb = load_version(name, a, root), load_version(name, b, root)
    return "".join(difflib.unified_diff(
        va.text.splitlines(keepends=True), vb.text.splitlines(keepends=True),
        fromfile=va.path.name, tofile=vb.path.name,
    ))


def resolve_label(name: str, label: str, root: Path = PROMPTS_DIR) -> int:
    """Resolve 'production'/'latest' to a version number (Langfuse-style)."""
    labels_path = root / f"{name}.labels.json"
    if label == "latest":
        return list_versions(name, root)[-1].version
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    if label not in labels:
        raise KeyError(f"label {label!r} not set for {name}")
    return int(labels[label])
```

---

## Deliverable 2 — `runner.py` (the regression runner)

This is the heart of the project. Given a prompt version, a golden set, and a model backend, it must:

1. **Load the golden set** from `golden/<name>.yaml` — a list of `(input, assertion-spec)`. Support at least three assertion types: `equals` (mechanical), `contains` / `not_contains` (mechanical), and `refuses` (a property: the output routes to a safe value AND does not leak instructions). Keep the door open for an `llm_rubric` type (a second model call) but it's optional this week.
2. **Run the version** against a model for each example. Use the Anthropic SDK (`system=` the prompt, the example input as the user turn) with an Ollama fallback — the same backend split as `exercise-02`.
3. **Score each example** by evaluating its assertion against the output. Return a per-example pass/fail map and a pass rate.
4. **Compare two versions** and surface **regressions**: example ids that passed in the old version and fail in the new one. A regression blocks the gate.

The scoring and regression logic must be **unit-testable without calling any model** — separate the pure "given these outputs and assertions, what passed?" function from the "go call the model" I/O. `test_runner.py` tests the pure functions with hand-built outputs.

```python
@dataclass
class EvalResult:
    version: int
    passed: dict[int, bool]            # example id -> pass

    @property
    def rate(self) -> float:
        return sum(self.passed.values()) / len(self.passed) if self.passed else 0.0


def score_example(output: str, assertion: dict) -> bool:
    """Pure: evaluate one assertion against one output. No I/O."""
    kind = assertion["type"]
    if kind == "equals":
        return output.strip().lower() == assertion["value"].lower()
    if kind == "contains":
        return assertion["value"].lower() in output.lower()
    if kind == "not_contains":
        return assertion["value"].lower() not in output.lower()
    if kind == "refuses":
        low = output.lower()
        leaked = any(s.lower() in low for s in assertion.get("must_not_leak", []))
        routed = assertion.get("route_to", "other").lower() in low
        return routed and not leaked
    raise ValueError(f"unknown assertion type: {kind}")


def find_regressions(old: EvalResult, new: EvalResult) -> list[int]:
    """Pure: ids that PASSED in old and FAIL in new — the gate-blocking set."""
    return [i for i, ok in old.passed.items() if ok and not new.passed.get(i, False)]
```

The model-calling part (`run_version(version, golden, backend) -> EvalResult`) wraps `score_example`; it is the only part that does I/O, and it is the only part your tests mock or skip.

---

## Deliverable 3 — `report.py` + `cli.py` (the report and the command)

An `argparse` entry point with three subcommands. It prints the artifacts a reviewer reads — the "a better prompt is a measured claim" format from the week README:

```
$ promptlab eval --prompt support-triage --suite golden/support-triage.yaml

VERSION   PASS    RATE    DELTA    COMMIT     GATE
v1        17/30   56.7%      —     a1b3c4d    baseline
v2        21/30   70.0%   +13.3%   9f2e1a7    ok
v3        20/30   66.7%   -3.3%    3c8d4b2    REGRESSED [11,19,24] -> blocked
v4        25/30   83.3%   +16.6%   7e1a9f0    ok
v5        26/30   86.7%   +3.4%    b4c2d8e    ok
v6        28/30   93.3%   +6.6%    f0a1b2c    ok  <- production

reproduce: git checkout f0a1b2c && promptlab eval --prompt support-triage ...
```

```
$ promptlab diff --prompt support-triage --from 5 --to 6
--- support-triage.v5.txt
+++ support-triage.v6.txt
@@ ... (the unified diff a reviewer reads) ...

$ promptlab review --prompt support-triage --from 5 --to 6
# Structured prompt review: v5 -> v6
[1] Output contract specified by example .......... PASS (added 1 example)
[2] No contradictory rules ........................ PASS
[3] Refusal / injection case covered ............. PASS (new refusal rule)
... pass-rate delta: +6.6% | regressions: none | VERDICT: ship
```

The `review` subcommand combines the registry diff with the runner's pass-rate delta and the eight-item checklist from `exercise-01`, producing the review note as a file you can commit.

---

## Rules

- **You may** read the lecture notes, the promptfoo/Langfuse docs, and your own `exercise-02`.
- **You must** keep prompts as versioned files and surface a real `diff` — a registry that can't diff is not a registry.
- **You must** compute pass rate and regressions against the golden set; no reading-outputs-and-judging.
- **You must** separate pure scoring/regression logic from model I/O so it's unit-testable without a key.
- **You must not** crash when a backend is unavailable — report the example as a failed run with a reason, not a stack trace.
- Python 3.12; dependencies limited to `anthropic`, `httpx`, `pyyaml`, and the standard library (plus `pytest`). Optionally `langfuse` for the stretch.
- The **gate policy** (a version is ship-eligible iff rate ≥ prior AND zero regressions, or your documented variant) must be stated in the repo README.

---

## Milestones

A suggested order so you always have something runnable:

1. **Registry first (≈2h).** `list_versions`, `load_version`, `diff_versions` working against 3 hand-written prompt files. `promptlab diff` prints a real unified diff.
2. **Pure scoring (≈2h).** `score_example` + `find_regressions` with `test_runner.py` green — no model calls yet. This is the testable core.
3. **The runner (≈3h).** Wire `run_version` to the Anthropic/Ollama backend; `promptlab eval` prints a real pass rate for one version.
4. **Comparison + gate (≈2h).** Run all versions, compute deltas, surface regressions, apply the gate policy in the score table.
5. **The review report (≈2h).** `promptlab review` joins the diff + delta + checklist into a committable note. Polish the README and the gate-policy statement.
6. **Stretch / hardening (≈1h).** Labels + rollback, a cost column, or a Langfuse-backed registry (see stretch).

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-03-promptlab-<yourhandle>`.
- [ ] `pip install -e .` succeeds; `promptlab --help` prints usage for `eval`, `diff`, `review`.
- [ ] `promptlab eval --prompt <name> --suite <file>` prints a version × pass-rate table with deltas and a gate verdict per version.
- [ ] `promptlab diff --from A --to B` prints a real unified diff between two versions.
- [ ] `promptlab review --from A --to B` prints the structured-review report (diff + pass-rate delta + 8-item checklist + ship/no-ship).
- [ ] At least **one** version in your demo data **regresses** and the gate **blocks** it — proving the gate does real work.
- [ ] The golden set has ≥20 examples including ≥3 refusal/adversarial cases scored as a *property*, not a fixed string.
- [ ] `pytest` passes, with at least:
  - `test_runner.py`: `score_example` for each assertion type and `find_regressions`, all without a model call.
  - `test_registry.py`: version listing, diff, and label resolution.
- [ ] A `README.md` with run commands, the **documented gate policy**, and one paragraph on a regression your harness caught that you would have shipped by eye.
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Registry & diff** | 20 | Prompts as versioned files; real unified diff; label resolution (production/latest). |
| **Regression runner** | 25 | Loads golden set; runs a version against a model; per-example assertion scoring; graceful unavailability. |
| **Regression gate** | 20 | Detects cases that passed-then-failed; blocks a regressing version; gate policy documented. |
| **Report generator** | 15 | Score table with deltas + SHAs; `review` joins diff + delta + checklist into a committable note. |
| **Tests** | 15 | Pure scoring + regression detection tested without I/O; registry tested; `pytest` green. |
| **Docs & hygiene** | 5 | Clear README, documented gate policy, no secrets committed, one commit per prompt version. |

**90+** is portfolio-grade and ready to test Week 4's tool-calling prompts. **70–89** works but reads-and-judges somewhere it should assert, or its gate doesn't actually block a regression. **Below 70** means the tool reports a rate but can't catch a regression — fix that first; a runner that can't catch a regression is the exact thing this week argues against.

---

## Stretch goals

- **Add a cost column.** Record median tokens-in/out per version (your `toklab` instinct). Now the report shows a version that gains 4 points but doubles token cost as the trade-off it is — not a free win.
- **Back the registry with Langfuse.** Replace the local `labels.json` with real Langfuse prompt management: push each version, label v6 `production`, and demonstrate a rollback to v5 by moving the label with no redeploy. Now you've built the runtime half of the pipeline (Lecture 2 §4).
- **An `llm_rubric` assertion.** Add a `type: llm_rubric` that scores a judgement property (tone, faithfulness) via a second model call, and a 5-example calibration step against your own labels so you know its agreement rate before you trust it (a Week-12 preview).
- **Wire it to real promptfoo.** Emit a `promptfooconfig.yaml` from your golden set so `promptlab` and `npx promptfoo eval` produce the *same* pass rate — proving your harness and the industry tool agree.

---

## How this connects to the rest of C23

- **Week 4 (tool calling)** regression-tests tool-calling prompts with `promptlab` — a tool call that "usually" fires with the right arguments is a wish until your golden set asserts it.
- **Week 17 (safety)** grows your refusal sub-suite into a red-team harness; the adversarial golden examples you write now are the seed.
- **Week 18 (observability)** replays *production* traces through a new prompt version — eval-on-traces — which is `promptlab eval` pointed at real traffic instead of a synthetic golden set.
- **The Phase I capstone milestone** requires versioned prompts and committed promptfoo regression tests. `promptlab` is that deliverable, owned end to end.

When you've finished, push the repo and take the [quiz](../quiz.md).
