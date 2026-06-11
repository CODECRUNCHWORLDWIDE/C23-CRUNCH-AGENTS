# Week 9 — Challenges

The exercises drill each layer in isolation. **The challenge makes you the architect.** You build the full retrieval stack and measure what every layer is actually worth — on one fixed gold set, with one fixed metric — then defend the pipeline with a table instead of a hunch.

## Index

1. **[Challenge 1 — The cumulative-lift chart](challenge-01-cumulative-lift.md)** — build the whole stack (BM25 → dense → hybrid+RRF → +reranker → +HyDE), run the *same* 40-query gold set under each layer with the *same* week-7 `evaluate()`, and produce a cumulative-lift table that shows exactly where the gains come from — and where a layer added nothing. (~90 min)

Challenges are optional for passing the week, but this one is the single best preparation for the Phase II architecture review in Week 12, where you defend your retrieval pipeline to a reviewer. Do it. The skill — knowing which layer earns its latency and being able to *prove* it on your own data — is what separates a junior who "added a reranker because the blog said to" from a senior who measured the lift and can tell you the number.
