# Week 20 — Challenges

The exercises drill the mechanics — build an engine, block an injection, score a matrix. **The challenge makes you the engineer who has to ship.** You take the *same* Qwen2.5-14B on the *same* H100 as week 19, serve it the NVIDIA way, guard it with a rail, benchmark it honestly against the vLLM baseline, and write the memo that decides which one survives into the capstone.

## Index

1. **[Challenge 1 — Qwen2.5-14B: NeMo vs vLLM, with a rail](challenge-01-qwen-nemo-vs-vllm.md)** — deploy Qwen2.5-14B via NeMo Inference / Triton on the same H100 as week 19, add a NeMo Guardrails policy that blocks one specific prompt-injection class, benchmark against the week-19 vLLM deployment apples-to-apples, and decide which would survive in production for the capstone. (~150 min, GPU-gated for the benchmark; the policy + decision halves run CPU-only.)

This challenge is **optional for passing the week, but it is the syllabus production-decision lab** — the whole reason this week exists. It is the single best preparation for the capstone (weeks 22–24), where you defend your serving + safety architecture to a reviewer. The skill it builds — running a *fair* benchmark, accounting for the rail's cost, and committing to a stack with numbers and a memo behind the choice — is what separates an engineer who "tried NeMo" from one who *chose* a serving stack on purpose and can say why. Do it.
