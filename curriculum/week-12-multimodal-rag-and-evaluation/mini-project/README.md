# Mini-Project — `crunchrag_eval`: The Calibrated RAG Evaluation Suite

> Build a reusable evaluation package that any RAG pipeline can import to score the four Ragas metrics behind a *calibrated* LLM-as-judge, over a multimodal-aware answerer, across pipeline variants — so "is this pipeline better, on what axis, and can I trust the number?" becomes a command, not an argument.

This is the artifact that turns RAG evaluation from a one-off notebook into a measurement you ship. After this week, grading a pipeline is `python -m crunchrag_eval report --variants baseline,reranker,hybrid` and reading a four-metric table with a kappa in the header — not eyeballing a demo. The package is corpus-agnostic, judge-pluggable (local model **or** Claude), multimodal-aware (it can answer over a figure when the answer is in a picture), and calibration-honest (no metric ships without its threshold and kappa). It reuses weeks 7–11's `evaluate()` and `store.py` **unchanged**, and the **Ragas report it produces is the Phase II milestone**.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This module is imported directly by **Phase III (week 13, LangGraph)**. An agent is a graph of LLM calls, and an agent without an eval harness is the same vibe-shipped liability as a RAG pipeline without Ragas — just with more moving parts and more places to hallucinate. Week 13 reuses your `crunchrag_eval` to grade *agent trajectories* (was each step's answer faithful to what the tool returned?), not just one-shot answers. The calibrated judge you build here is the trust anchor for the whole back half of the course. Build it well now.

---

## What you will build

A small Python package `crunchrag_eval` with five deliverables:

1. **`crunchrag_eval/metrics.py`** — a thin, uniform wrapper over the four Ragas metrics (faithfulness, context recall, context precision, answer relevancy) so the rest of the code calls `score(dataset, judge)` and gets four numbers, regardless of which judge backend is wired in.
2. **`crunchrag_eval/judge.py`** — the pluggable judge: a local OpenAI-compatible model (the open path) **or** `claude-opus-4-8` with structured output (the frontier path), plus the **calibration** machinery (10 human labels → agreement → Cohen's kappa → threshold sweep → calibrated `(tau, kappa)`).
3. **`crunchrag_eval/answerer.py`** — a multimodal-aware answer step: given retrieved contexts (text chunks *and/or* page images), generate an answer with a text LLM or a VLM (Claude vision / open VLM). This is the thing whose output the metrics score.
4. **`crunchrag_eval/report.py`** — the variant loop: for each pipeline variant, retrieve → answer → score the four metrics with the *calibrated* judge, collect the row, and emit the four-metrics × N-variants table and plot.
5. **`crunchrag_eval/cli.py`** — a `report` command (and a `calibrate` command) that ties it together and prints the comparison table with the calibrated-judge header.

By the end you have a public repo of ~450–550 lines of Python (excluding the corpus) that any future RAG or agent project can `from crunchrag_eval import score, Judge` and stop shipping on faith.

---

## Why a package and not a notebook

You could do all of this in a Jupyter notebook. Don't — not as the artifact. A package gives you:

- **Reuse.** Week 13 imports your judge and your `score()` to grade agent steps. A notebook gets copy-pasted, drifts, and rots.
- **A fixed, calibrated measurement.** The gold set, the four metrics, the calibrated threshold, and the "judge ≠ generator" rule live in code, version-controlled. "Did this change help?" is answered by re-running the *same* `report.py`, not by eyeballing a new cell with a possibly-different judge.
- **A CLI.** `report --variants baseline,reranker,hybrid` is greppable, scriptable, and CI-able. A notebook cell is none of those, and you cannot gate a deploy on a notebook.

Notebooks are great for *exploring* a single judge's verdicts by eye. The thing you ship and depend on is a package with a calibrated judge inside it. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchrag_eval/
├── pyproject.toml
├── docker-compose.yml          # pgvector/pgvector:pg17 on localhost:5432 (from week 10)
├── README.md                   # the Ragas report table + the milestone memo
├── corpus/
│   ├── legal_docs.jsonl        # multi-clause documents: {"doc_id": "...", "text": "..."}
│   ├── gold.json               # 40 queries: [{"question","reference","relevant":["clause_14"]}]
│   └── calibration.json        # 10 human-labeled examples for judge calibration
├── crunchrag_eval/
│   ├── __init__.py             # exposes score(), Judge, calibrate()
│   ├── metrics.py              # the four Ragas metrics behind one score()
│   ├── judge.py                # pluggable judge (local | claude) + calibration
│   ├── answerer.py             # multimodal-aware answer step (text LLM | VLM)
│   ├── report.py               # the variant loop + table + plot
│   └── cli.py                  # the `report` and `calibrate` commands
└── tests/
    ├── test_metrics.py         # each metric moves on the right failure (from Exercise 2)
    └── test_calibration.py     # kappa + threshold sweep are correct (from Exercise 3)
```

Your week-7 `crunchrag_embed` package is a dependency (installed editable or vendored); `report.py` imports `evaluate` (the retrieval metric) and `store` from it **unchanged** — Ragas's *answer* metrics layer on top of those *retrieval* metrics.

---

## Deliverable 1 — `metrics.py` (the four Ragas metrics, one entry point)

This wraps Ragas so the rest of the code never juggles metric objects. One function, four numbers.

```python
"""crunchrag_eval.metrics — the four Ragas metrics behind one score()."""
from __future__ import annotations

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference, ResponseRelevancy,
)

_METRICS = [Faithfulness(), LLMContextRecall(),
            LLMContextPrecisionWithReference(), ResponseRelevancy()]


def score(samples: list[dict], judge_llm, embeddings) -> dict[str, float]:
    """samples: [{user_input, response, retrieved_contexts, reference}].
    Returns {faithfulness, context_recall, context_precision, answer_relevancy}."""
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset=dataset, metrics=_METRICS,
                      llm=judge_llm, embeddings=embeddings)
    return {
        "faithfulness": float(result["faithfulness"]),
        "context_recall": float(result["context_recall"]),
        "context_precision": float(result["context_precision"]),
        "answer_relevancy": float(result["answer_relevancy"]),
    }

# TODO 1: add a `score_offline()` fallback that uses your Exercise-2 from-scratch
#   metrics + a stub judge, so the package's tests and a smoke run work with NO
#   network and NO API key (the open-by-default principle). report.py picks the
#   offline path when no judge is configured.
```

> **The rule the project enforces:** four numbers, never one. If a caller asks for a single "quality score," refuse — the whole point of Ragas is that the four metrics localize *which stage* failed. Collapsing them throws away the diagnosis.

---

## Deliverable 2 — `judge.py` (pluggable judge + calibration)

The heart of the trust story. A `Judge` that can be a local model or Claude, plus the calibration that earns the right to believe it.

```python
"""crunchrag_eval.judge — pluggable LLM-as-judge with calibration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Calibration:
    threshold: float        # tau: judge score >= tau counts as pass
    kappa: float            # Cohen's kappa vs the 10 human labels
    n_labels: int           # how many human labels backed the calibration


def cohen_kappa(human: list[int], judge: list[int]) -> float:
    """Agreement corrected for chance. (Port from Exercise 3.)"""
    n = len(human)
    p_obs = sum(h == j for h, j in zip(human, judge)) / n
    h1, j1 = sum(human) / n, sum(judge) / n
    p_chance = h1 * j1 + (1 - h1) * (1 - j1)
    return 1.0 if p_chance >= 1.0 else (p_obs - p_chance) / (1 - p_chance)


class Judge:
    """Frontier path: claude-opus-4-8 with structured output. Open path: a local
    OpenAI-compatible model. Both return a continuous faithfulness-ish score."""

    def __init__(self, backend: str = "claude", model: str = "claude-opus-4-8"):
        self.backend, self.model = backend, model
        self.calibration: Calibration | None = None

    def score_one(self, question: str, answer: str, contexts: list[str]) -> float:
        # TODO 2: implement the claude-opus-4-8 path with messages.parse and a
        #   Pydantic verdict (thinking={"type":"adaptive"}, output_config={"effort":
        #   "high"}, NO temperature). Implement the local path via an OpenAI-
        #   compatible client. Both return a float in [0,1].
        raise NotImplementedError

    def calibrate(self, labeled: list[dict]) -> Calibration:
        """labeled: [{question, answer, context, human_label(0/1)}] (>=10).
        Sweep tau to maximize Cohen's kappa; store and return the Calibration."""
        human = [ex["human_label"] for ex in labeled]
        scores = [self.score_one(ex["question"], ex["answer"], [ex["context"]])
                  for ex in labeled]
        best = None
        for i in range(1, 10):                  # tau in 0.1..0.9
            tau = i / 10
            binary = [1 if s >= tau else 0 for s in scores]
            k = cohen_kappa(human, binary)
            if best is None or k > best[1]:
                best = (tau, k)
        # TODO 3: store self.calibration = Calibration(best[0], best[1], len(labeled))
        #   and REFUSE to score a report (raise) if kappa < 0.4 — an untrustworthy
        #   judge must not silently produce confident decimals.
        self.calibration = Calibration(best[0], best[1], len(labeled))
        return self.calibration
```

> **The non-negotiable:** no report runs on an uncalibrated judge. `report.py` checks `judge.calibration is not None` and refuses otherwise. A confident decimal from an unchecked judge is the exact failure this whole package exists to prevent.

---

## Deliverable 3 — `answerer.py` (the multimodal-aware answer step)

The metrics score *answers*. This produces them — from text chunks, or from a page image when the answer lives in a figure (Lecture 1).

```python
"""crunchrag_eval.answerer — generate an answer from retrieved context.
Text path: a text LLM over text chunks. Multimodal path: a VLM over a page image."""
from __future__ import annotations


def answer_text(question: str, contexts: list[str], generate_fn) -> str:
    """Generate an answer from TEXT chunks. generate_fn is the FIXED generator
    (held constant across variants, and DIFFERENT from the judge model)."""
    prompt = ("Context:\n" + "\n---\n".join(contexts) +
              f"\n\nQuestion: {question}\nAnswer using only the context.")
    return generate_fn(prompt)


def answer_multimodal(question: str, page_image_b64: str) -> str:
    """Generate an answer from a PAGE IMAGE with a VLM (Claude vision shape).
    Used when the answer is in a figure the text pipeline can't ground on."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-8", max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": page_image_b64}},
            {"type": "text", "text": question},
        ]}],
    )
    return "".join(b.text for b in response.content if b.type == "text")

# TODO 4: add an OPEN-VLM path (Qwen2.5-VL via transformers or Ollama) so the
#   multimodal answerer has an open-weights option, not only the Claude path.
```

The generator is **held constant across variants** and is **a different model from the judge** — the two structural rules that keep a metric delta attributable to the *retriever* and free of self-preference bias.

---

## Deliverable 4 — `report.py` (the variant loop + table + plot)

The function that produces the milestone. For each variant: retrieve, answer, score with the calibrated judge, collect the row.

```python
from crunchrag_embed.eval import evaluate as retrieval_eval  # week 7, UNCHANGED
from crunchrag_eval import metrics


def run_report(variants, gold, generate_fn, judge, embeddings):
    if judge.calibration is None:
        raise RuntimeError("calibrate the judge (10 human labels) before reporting")
    rows = []
    for name, retrieve_fn in variants.items():
        samples = []
        for q in gold:
            contexts = retrieve_fn(q["question"])           # this variant's top-k
            response = generate_fn(q["question"], contexts) # the FIXED generator
            samples.append({"user_input": q["question"],
                            "retrieved_contexts": contexts,
                            "response": response,
                            "reference": q["reference"]})
        m = metrics.score(samples, judge.langchain_llm(), embeddings)  # 4 numbers
        rows.append({"variant": name, **m})
    return rows

# TODO 5: emit the matplotlib four-metrics × N-variants grouped bar chart
#   (ragas_report.png) and a markdown table with the calibrated (tau, kappa) in
#   the header, then identify which metric improved MOST between consecutive
#   variants (the syllabus finding: "which metric moved for which change").
```

The non-negotiables `report.py` enforces:

- **One generator, one judge, one gold set.** The retriever is the only thing that varies between variants; the four metrics and the calibrated judge are identical for every row.
- **The judge is calibrated** — `report.py` refuses to run otherwise — and the calibrated `(tau, kappa)` is printed in the report header so the reader knows how much to trust the table.
- **The generator is a different model from the judge** — no self-preference bias contaminating the absolute scores.

---

## Deliverable 5 — `cli.py` (the `report` and `calibrate` commands)

```bash
# Step 1: calibrate the judge once against the 10 human labels.
python -m crunchrag_eval calibrate \
    --labels corpus/calibration.json \
    --judge claude            # or: --judge local --base-url http://localhost:11434/v1

# Step 2: run the four-metric report across the three variants.
python -m crunchrag_eval report \
    --corpus corpus/legal_docs.jsonl \
    --gold corpus/gold.json \
    --variants baseline,reranker,hybrid \
    --judge claude
```

`report` should print:

```
judge=claude-opus-4-8  calibrated: tau=0.62  kappa=0.71 (substantial)  generator=qwen2.5:14b

VARIANT     FAITHFUL   CTX_RECALL   CTX_PRECISION   ANS_RELEVANCY
baseline      0.81        0.84          0.68            0.80
reranker      0.85        0.85          0.79            0.82
hybrid        0.88        0.91          0.78            0.84
-----------------------------------------------------------------
biggest move 1->2: context_precision +0.11 (reranker demoted irrelevant chunks)
biggest move 2->3: context_recall    +0.06 (hybrid caught lexical matches dense missed)
wrote ragas_report.png
```

The header carries the calibrated `(tau, kappa)` and names the generator (so the judge≠generator rule is auditable). The two "biggest move" lines are the syllabus finding — *which metric improved most for which change* — printed, not hand-waved.

---

## Rules

- **You may** read the Ragas docs, the lecture notes, your weeks-7–11 code, and the Exercise 2/3 implementations.
- **You must not** ship a metric without a calibrated judge behind it. `calibrate` runs before `report`, and `report` refuses an uncalibrated judge. A confident decimal from an unchecked judge is the failure this package prevents.
- **You must not** use the same model as generator and judge (self-preference bias). If you must share a family, label it and treat absolute numbers as suspect.
- **You must not** re-implement Recall@5/MRR — import them from `crunchrag_embed.eval` unchanged; Ragas's answer metrics sit on top of them.
- Python 3.12, `ragas`, `anthropic`, `pydantic`, `numpy`, `matplotlib`, plus `pytest`. The open path needs a local OpenAI-compatible endpoint (Ollama/vLLM); an open-only run must be possible with no API key.
- `evaluate()` (retrieval) must stay pure; you wrap it, you don't edit it.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-12-crunchrag-eval-<yourhandle>`.
- [ ] `docker compose up -d` brings up pgvector; the report runs against it.
- [ ] `metrics.py` computes the four Ragas metrics behind one `score()`, with an offline from-scratch fallback (TODO 1) so tests run with no network.
- [ ] `judge.py` implements both a Claude and a local backend, plus `calibrate()` with Cohen's kappa and a threshold sweep; it **refuses** to be used in a report with kappa < 0.4.
- [ ] `answerer.py` has a text path and a multimodal (VLM) path, with the generator held constant and different from the judge.
- [ ] `pytest` passes, with at least:
  - `test_metrics.py`: faithfulness drops on a hallucinated answer; answer_relevancy drops on an off-topic answer; context_precision penalizes a buried relevant chunk (the Exercise 2 orthogonality, as tests).
  - `test_calibration.py`: Cohen's kappa is correct on a known case; the threshold sweep picks the kappa-maximizing tau (the Exercise 3 logic, as tests).
- [ ] `python -m crunchrag_eval report --variants baseline,reranker,hybrid` prints a three-row, four-metric table with the calibrated `(tau, kappa)` in the header and the two "biggest move" lines.
- [ ] A `ragas_report.png` four-metrics × three-variants plot is generated.
- [ ] A `README.md` with the results table, the run commands, and the **one-page milestone memo** (the variant shipped, the four numbers, which metric moved most for which change with the mechanism, the calibrated kappa, and the judge≠generator note).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Metric correctness** | 20 | The four Ragas metrics computed correctly behind one `score()`; the offline fallback matches the from-scratch Exercise-2 behavior; four numbers, never one collapsed score. |
| **Judge calibration** | 25 | 10 human labels; Cohen's kappa computed correctly; threshold swept to maximize kappa; the report refuses an uncalibrated (or sub-0.4-kappa) judge; `(tau, kappa)` carried into every metric. |
| **Variant validity** | 20 | Generator and judge held constant across variants; the retriever is the only thing that varies; the generator is a *different model* from the judge (no self-preference). |
| **Multimodal answerer** | 10 | A working VLM path (Claude vision shape correct: image block, no temperature, claude-opus-4-8) plus an open-VLM option; used where the answer is in a figure. |
| **Report & finding** | 15 | The CLI runs all variants, prints the four-metric table + plot, and names which metric moved most for which change with a mechanism, not just a delta. |
| **Tests & hygiene** | 10 | `test_metrics` + `test_calibration` green; clear README + memo; no secrets committed; no `__pycache__`/`.venv`/`models/` checked in. |

**90+** is portfolio-grade and ready to drop into week 13's agent-eval harness. **70–89** works but has a soft calibration or an unlabeled judge≠generator gap. **Below 70** means the metrics aren't trustworthy — fix the calibration first, because the *entire* value of this package is a number you can defend.

---

## Stretch goals

- **The multimodal variant.** Add a fourth variant that retrieves figure images and answers with the VLM path; run the same four metrics and see whether faithfulness improves on figure questions the text pipeline couldn't ground.
- **Two-judge agreement.** Calibrate a second judge (a different model) and compute inter-judge kappa; flag the queries where the two judges disagree as your least-trustworthy metrics.
- **DeepEval CI gate.** Wrap your `score()` in a DeepEval `assert_test` and add a GitHub Actions workflow that fails the build if any variant's faithfulness regresses below the calibrated threshold — eval that gates deploys (Lecture 2 Part 6).
- **Verbosity-bias correction.** Probe the judge for length bias (pad correct answers), and if present, add a length-normalization step and re-run; report the before/after kappa.

---

## How this connects to the rest of C23

- **Weeks 7–11 (embeddings → chunking → reranking/hybrid → vector stores → memory)** built the retrieval pipeline and gave you `evaluate()` (Recall@5/MRR) and `store.py`; this suite imports both unchanged and layers the four *answer* metrics on top — retrieval metrics under answer metrics, exactly as Lecture 2 frames it.
- **This week (evaluation)** is the **Phase II capstone milestone**: the Ragas report across three variants, behind a calibrated judge, is the artifact you defend at the architecture review.
- **Week 13+ (Phase III — agents, starting with LangGraph)** consumes this suite to grade *agent trajectories*: an agent step's answer must be faithful to what its tool returned, and the calibrated judge you built here is the trust anchor for that. An agent without `crunchrag_eval` is a RAG-without-Ragas vibe with more moving parts.

When you've finished, push the repo and take the [quiz](../quiz.md).
