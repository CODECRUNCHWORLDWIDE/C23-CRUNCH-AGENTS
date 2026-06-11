# Week 6 — Challenges

The exercises drill the mechanics — bring up three engines, chart the quant trade-offs, build the benchmark. **The challenge makes you the engineer who has to pick the engine and defend the choice.** You run the same model on all three engines through the same benchmark, read the throughput-vs-concurrency curves, and commit to a serving recommendation you can defend at a deployment review — the way the decision actually gets made.

## Index

1. **[Challenge 1 — The three-engine bakeoff](challenge-01-three-engine-bakeoff.md)** — one model (Qwen2.5-7B), three engines (Ollama, llama.cpp, vLLM), the 100-prompt benchmark at rising concurrency. Report prefill/decode tokens/sec, TTFT, p50/p95, VRAM, and aggregate throughput. Pick a serving engine with reasons. (~150 min)

Challenges are optional for passing the week, but this one *is* the syllabus hands-on lab in committed form and the single best preparation for the Phase I milestone, where you serve your week-5 agent on a local model and defend the serving choice. Do it. The skill — holding the model fixed, varying only the engine, reading the curve, and committing to a choice with a number behind it — is what separates a junior who "ran a model locally once" from an engineer who can say which engine to ship and why it survives concurrency.
