# Week 18 — Challenges

The exercises drill the mechanics — read a trace, emit correct `gen_ai.*` spans, compute the rollups. **The challenge is the Phase III milestone in lab form.** You instrument your real week-13/15/17 multi-agent system end to end, dual-export every span to *both* self-hosted Langfuse and Phoenix, build the three dashboards, inject one failure, and find it from the dashboard in under five minutes.

## Index

1. **[Challenge 1 — Instrument and find the failure](challenge-01-instrument-and-find-the-failure.md)** — OTel-instrument the Phase III stack, dual-export to Langfuse + Phoenix, build the three dashboards (token usage per route, p95 latency per agent step, retrieval-precision over time), inject one synthetic failure, and locate it from a dashboard in under five minutes. (~150 min)

This challenge is optional for passing the week, but it **is** the Phase III observability milestone — the capstone (weeks 22–24) assumes its output (traces flowing to Langfuse + Phoenix, the three dashboards live). Do it. The skill it builds — turning "the agent is broken" into "the retriever's span returned empty at 14:05," from a dashboard, while the user is still waiting — is exactly what separates an engineer who *shipped* an agent from one who can *operate* one.
