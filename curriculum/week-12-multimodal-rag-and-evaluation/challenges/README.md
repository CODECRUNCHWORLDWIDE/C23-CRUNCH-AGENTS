# Week 12 — Challenges

The exercises drill the mechanics — multimodal answering, the four metrics, judge calibration. **The challenge makes you the engineer who has to ship the evidence.** You run a Ragas evaluation suite across three real variants of your Phase II pipeline, behind a calibrated LLM-as-judge, and you write the report that says — with numbers, not vibes — which change improved which metric, and by how much.

## Index

1. **[Challenge 1 — The Ragas report across three variants](challenge-01-ragas-report-three-variants.md)** — the four metrics (faithfulness, context recall, context precision, answer relevancy) over three pipeline variants (baseline dense / +reranker / +hybrid from weeks 8–10), with a calibrated judge (10 human labels, Cohen's kappa, tuned threshold), a four-metrics × three-variants plot, and the finding: which metric moved most for which change. (~150 min)

This challenge **is the Phase II milestone in lab form** — the Ragas evaluation report the syllabus names as the deliverable for the end of Phase II. It is not optional in spirit: it's the artifact you defend at the architecture review and the eval suite Phase III (week 13 LangGraph) consumes to grade agent trajectories. The skill — changing one pipeline variable, measuring four answer-quality axes, trusting the judge only after calibrating it, and committing to a finding with a kappa behind it — is what separates a junior who "ran Ragas once" from an engineer who shipped a measured, defensible RAG system. Do it.
