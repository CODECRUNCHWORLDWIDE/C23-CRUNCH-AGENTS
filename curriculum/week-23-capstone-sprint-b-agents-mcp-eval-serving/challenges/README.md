# Week 23 — Challenges

One challenge this week, and it is the spine of Sprint B: get the capstone's **thin slice green end-to-end** on the 100-question gold set. Everything in the lectures and exercises feeds into it.

## Index

1. **[Challenge 1 — End-to-end green](challenge-01-end-to-end-green.md)** — wire the supervisor, the MCP corpus tool, the two-tier serving, the Ragas + calibrated-judge eval, and the OTel tracing into one runnable system, and make the eval gate print `PASS` on the 100-question gold set. (~3 hours)

## How to work the challenge

- The challenge is the integration. The exercises gave you the parts (the supervisor router, the corpus MCP server, the cost-tracked router); the challenge assembles them and proves the assembly works by a *measured* gate, not a demo.
- **Thin slice first.** Do not perfect any component before the whole slice runs once. One query → supervisor → retrieval-agent → corpus.search → writing-agent → critique-agent → answer, with a trace and a Ragas score. Get *that* green, then deepen by what the eval flags.
- **The gate is the deliverable.** A demo that answers your favorite query is not a pass. `python -m capstone.eval run --gold gold/eval_100.jsonl --gate` printing `PASS` is.
- Read the trace when the gate fails. The whole point of the OTel wiring is that a red gate plus a trace tells you *which route or tool* dragged the score down in thirty seconds.

This challenge is the rehearsal for the mini-project (ship the runnable artifact) and the proving ground for week 24 (break it on purpose). Build it traced, budgeted, and gated.
