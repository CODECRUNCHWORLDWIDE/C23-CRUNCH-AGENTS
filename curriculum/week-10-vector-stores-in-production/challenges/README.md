# Week 10 — Challenges

The exercises drill the mechanics — stand up three stores, see filtered ANN break, run the recovery drill. **The challenge makes you the engineer who has to pick the store and defend it at an architecture review.** You run the same pipeline against all three stores, measure the operational axes that actually decide it (ingest, filtered-recall, p95, time-to-recover), and commit to a store you can defend — the way the decision actually gets made on a real RAG project.

## Index

1. **[Challenge 1 — The store bakeoff](challenge-01-store-bakeoff.md)** — three stores (pgvector, Qdrant, Weaviate), one pipeline (your week-9 hybrid retrieval). Measure ingest throughput, query p50/p95, filtered-search recall at a selective filter, config complexity, and time-to-recover from a simulated index loss. Pick a store with an architecture memo. (~150 min)

Challenges are optional for passing the week, but this one *is* the syllabus hands-on lab in committed form and the single best preparation for the Week 12 architecture review, where you defend your whole retrieval pipeline — store included — to a reviewer. Do it. The skill — holding the pipeline fixed, varying only the store, measuring the *operational* axes (not just latency), and committing to a choice with a recovery number behind it — is what separates a junior who "used a vector database" from an engineer who chose the right one and can say why it survives a 2 AM index loss.
