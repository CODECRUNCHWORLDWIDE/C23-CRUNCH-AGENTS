# Week 8 — Challenges

The exercises drill the mechanics — extraction, the three chunkers, the size sweep. **The challenge makes you the engineer who has to pick.** You run all four chunking strategies against one fixed embedding and one fixed store, read the metrics, and commit to a winner you can defend — the way the decision actually gets made on a real RAG project.

## Index

1. **[Challenge 1 — The chunking A/B](challenge-01-chunking-ab.md)** — four strategies (token-window 512/1024, semantic-paragraph, recursive, late chunking), one embedding (BGE-large), one store (pgvector). Report MRR / Recall@5 / faithfulness, and pick a winner with reasons. (~150 min)

Challenges are optional for passing the week, but this one *is* the syllabus deliverable in lab form and the single best preparation for the Week 12 architecture review, where you defend your retrieval pipeline to a reviewer. Do it. The skill — changing one variable, reading the metric, and committing to a choice with a number behind it — is what separates a junior who "tried some chunkers" from an engineer who shipped the right one and can say why.
