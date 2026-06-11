# Challenge 1 — The Ragas Report Across Three Variants

**Time estimate:** ~150 minutes.

## Problem statement

You have a RAG pipeline you have been building all of Phase II, and you have made it *better* three times — first a baseline dense retriever (week 7), then you added a reranker (week 9), then you fused in hybrid lexical search (week 9). Three variants, each supposedly an improvement. Four reasonable people on your team each have an opinion about which one to ship. You are going to end the argument the only way Phase II permits: run all three through a Ragas evaluation suite, score the **four answer-quality metrics** on a 40-question gold set, put a **calibrated** LLM-as-judge behind every number, and let the metrics name not just the winner but *which metric improved most for which change*. Then you write the report — the Phase II milestone — because a number from an uncalibrated judge is a vibe with a decimal point.

This is the syllabus deliverable in lab form. The output is a decision *and a diagnosis*: which variant you ship, and which of the four metrics each change actually moved, with a Cohen's kappa proving the judge was trustworthy.

## The three variants

Run exactly these three, and nothing else varies between them except the retrieval stage:

1. **Baseline dense** — your week-7/8 dense retriever (BGE-large, your winning chunking strategy from week 8), top-k by cosine. The control.
2. **+Reranker** — the same dense retrieval, with a cross-encoder reranker on top (week 9). Tests whether re-ordering the top-k improves the *answer*, not just the retrieval metric.
3. **+Hybrid** — dense + BM25 lexical fused (week 9), then reranked. Tests whether adding lexical recall improves the *answer*.

> **The one variable per step.** Variant 2 differs from variant 1 by *only* the reranker. Variant 3 differs from variant 2 by *only* the hybrid lexical leg. So a metric delta from 1→2 is attributable to the reranker, and a delta from 2→3 to hybrid. If you change two things between variants, you can't say which caused the metric to move — and "which metric moved for which change" is the entire deliverable. This is week 8's one-variable-at-a-time discipline, pointed at the retrieval stage and measured by *answer* metrics instead of Recall@5.

## What is fixed (do not let these vary)

- **Gold set:** the 40-query legal gold set from weeks 7–11, now with a **reference answer** and (per query) the **retrieved contexts** each variant produced. Ragas's answer metrics need question + answer + contexts + reference.
- **Generator:** one generator, held constant across all three variants — so a metric delta is the *retriever's* doing, not the generator's. (Use your week-6 local model or a fixed Claude model; just keep it the *same* across variants, and **different from the judge** to avoid self-preference.)
- **Judge:** one judge backend (`claude-opus-4-8` for the report, or a local model for the open path), **calibrated once** against 10 human labels, with a single tuned threshold and a reported kappa.
- **Metric suite:** the four Ragas metrics — faithfulness, context_recall, context_precision, answer_relevancy — computed identically for every variant.

## The harness approach

The whole report reduces to: for each variant, produce a Ragas `EvaluationDataset` (question, the variant's answer, the variant's retrieved contexts, the reference), call the *same* `evaluate()` with the *same calibrated judge*, collect the four numbers, and diff across variants.

```python
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference, ResponseRelevancy,
)

# Your week-7 retrieval harness, UNCHANGED, gives you the contexts per variant.
from crunchrag_embed import store                # week 7/10, unchanged
from crunchrag_embed.eval import evaluate as retrieval_eval  # week 7, unchanged

METRICS = [Faithfulness(), LLMContextRecall(),
           LLMContextPrecisionWithReference(), ResponseRelevancy()]


def build_dataset(gold, retrieve_fn, generate_fn):
    """For one variant: retrieve contexts, generate an answer, package for Ragas."""
    samples = []
    for q in gold:                                # q: {question, reference, ...}
        contexts = retrieve_fn(q["question"])     # this variant's top-k chunks
        answer = generate_fn(q["question"], contexts)  # the FIXED generator
        samples.append({
            "user_input": q["question"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": q["reference"],
        })
    return EvaluationDataset.from_list(samples)


def run_variant(name, gold, retrieve_fn, generate_fn, judge_llm, embeddings):
    ds = build_dataset(gold, retrieve_fn, generate_fn)
    result = evaluate(dataset=ds, metrics=METRICS, llm=judge_llm, embeddings=embeddings)
    return {                                       # the four numbers for this variant
        "variant": name,
        "faithfulness": result["faithfulness"],
        "context_recall": result["context_recall"],
        "context_precision": result["context_precision"],
        "answer_relevancy": result["answer_relevancy"],
    }


# The three variants share the SAME generator, judge, embeddings, and gold set.
rows = [
    run_variant("baseline", gold, dense_retrieve, generate, judge, emb),
    run_variant("reranker", gold, reranked_retrieve, generate, judge, emb),
    run_variant("hybrid",   gold, hybrid_retrieve, generate, judge, emb),
]
```

The **calibration** step (do it once, before the report) follows Lecture 2 Part 4 and Exercise 3: label 10 (question, answer, context) examples by hand, run the judge, sweep the threshold to maximize Cohen's kappa, and record `(tau, kappa)`. Every metric in the report is then reported *at that calibrated threshold*, with the kappa stated, so a reader knows how much to trust the decimals.

The **plot** is the four metrics on the x-axis (or as a grouped bar chart), three bars per metric (one per variant):

```python
import matplotlib.pyplot as plt
import numpy as np

metrics = ["faithfulness", "context_recall", "context_precision", "answer_relevancy"]
variants = [r["variant"] for r in rows]
x = np.arange(len(metrics))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
for i, r in enumerate(rows):
    ax.bar(x + (i - 1) * width, [r[m] for m in metrics], width, label=r["variant"])
ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=15)
ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.legend(title="variant")
ax.set_title("Four Ragas metrics × three pipeline variants (calibrated judge)")
fig.tight_layout(); fig.savefig("ragas_report.png", dpi=150)
```

## Acceptance criteria

- [ ] A `challenge-01/` directory with a runnable `ragas_report.py` that runs all three variants against the fixed generator + judge + gold set and prints a comparison table.
- [ ] The table reports **faithfulness, context_recall, context_precision, answer_relevancy** for all three variants, plus the variant's chunk/context count.
- [ ] The judge is **calibrated**: 10 human labels, a swept threshold, a reported Cohen's kappa, and the metrics reported *at* that threshold. The kappa is in the report, not just the code.
- [ ] The generator is held **constant** across variants and is a **different model from the judge** (no self-preference). The retrieval stage is the only thing that changes between variants.
- [ ] A **four-metrics × three-variants plot** (`ragas_report.png`) is generated.
- [ ] A one-page `ragas-report-memo.md` that names which metric **improved most for which change** (e.g. "the reranker moved context_precision +0.11 by demoting irrelevant chunks; hybrid moved context_recall +0.07 by catching lexical matches dense missed"), with the mechanism, not just the delta.
- [ ] At least one **per-query trace** in the promise format, e.g. `q12 ("five-year confidentiality") faithful=1.0 grounded in ctx_09 ✓`, for the winning variant — and one query where a losing variant hallucinated or missed.

## The trap (read after a first attempt)

The trap is **judging with the model that generated the answer.** If your generator is Claude and your judge is Claude, **self-preference bias** inflates every faithfulness and relevancy score — the judge rates its own family's output higher, and your beautiful 0.91 is partly the judge admiring itself. The fix is structural: generate with one model, judge with a *different* one (e.g. generate with a local Qwen, judge with `claude-opus-4-8`, or vice versa). If you must share a family, say so in the memo and treat the absolute numbers as suspect — the *relative* deltas between variants are still informative because the bias is constant, but the absolute level is not trustworthy.

A second, subtler trap: **reporting the metric without the kappa.** "Faithfulness 0.88" from an uncalibrated judge is a number you made up with extra steps. If you skip the 10-label calibration, you have no idea whether your judge agrees with a human at all — it could be at chance (kappa ≈ 0) while emitting confident decimals. Every metric in the report travels with its calibrated threshold and kappa, or it doesn't go in the report. (This is exactly the Exercise 3 sweep; if you delete it, you've fallen in the trap.)

A third trap, specific to this challenge: **letting the generator vary between variants.** If variant 2 also uses a better prompt or a bigger generator, then a faithfulness gain isn't the reranker's doing — it's two changes at once, and "which metric moved for which change" becomes unanswerable. Hold the generator fixed; vary only the retrieval stage.

## Stretch goals

- **The multimodal variant.** Add a fourth variant whose retrieval includes figure images (Lecture 1 Option B), with a VLM (`claude-opus-4-8` vision or Qwen2.5-VL) as the generator over the retrieved page image. Run the *same* four Ragas metrics on its answers. Does a multimodal pipeline improve faithfulness on the figure questions where the text pipeline had nothing to ground on?
- **Two judges, one kappa.** Calibrate a *second* judge (a different model) and compute the inter-judge kappa. Where the two judges disagree is where your metric is least trustworthy — flag those queries.
- **Verbosity-bias probe.** Take 10 answers, pad half with correct-but-redundant filler, and check whether the judge scores the padded ones higher. If it does, your judge has a length bias; note it and, ideally, correct for it (instruct the judge to ignore length) and re-run.
- **Cost-vs-judge sweep.** Run the report with `claude-haiku-4-5` as the judge and again with `claude-opus-4-8`. Calibrate both. Does the cheaper judge reach a substantial kappa? If so, you can run the *sweep* cheap and reserve Opus for the final report.

## Why this matters

This is the **end of Phase II**, and this report is the milestone. At the architecture review, the reviewer will not ask you to recite the four metrics — they'll point at your pipeline and ask "is it better than the baseline, on what axis, and how do you know your judge isn't lying to you?" This challenge *is* that conversation, rehearsed: you ran the variants, you have the four-metric table, you calibrated the judge and have the kappa, and you can say "the reranker bought us +0.11 context_precision; hybrid bought +0.07 context_recall; faithfulness was already saturated, so neither moved it much" — a *diagnosis*, not a vibe. Every RAG system you ship after this gets shipped on *some* evidence whether you produced it or not — the engineer who produced a calibrated, four-metric, multi-variant report is the one whose pipeline doesn't quietly regress in production, and whose Phase III agents inherit an eval suite that already works. The score survived calibration, and you can prove it.
