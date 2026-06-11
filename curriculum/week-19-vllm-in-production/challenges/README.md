# Week 19 — Challenges

The exercises drill the mechanics — stand up the server, benchmark the curve, compute the cost. **The challenge makes you the engineer who has to decide.** You stand up vLLM serving Qwen2.5-14B on an H100, put LiteLLM in front with a vendor fallback, benchmark throughput at concurrency 1/8/32/128, compute cost-per-million-tokens, and write the break-even memo that decides self-host-vs-vendor for the capstone's local tier — the way the call actually gets made.

## Index

1. **[Challenge 1 — The concurrency sweep](challenge-01-concurrency-sweep.md)** — serve Qwen2.5-14B, route it through LiteLLM with a Claude fallback, run the full concurrency sweep (1/8/32/128), read the throughput curve, compute $/MTok, and write the break-even serving memo. (~150 min; `--simulate` path documented for no-GPU completion)

Challenges are optional for passing the week, but this one **is the syllabus self-hosted-economics lab** in hands-on form, and the single best preparation for the capstone's serving tier and week 24's chaos drill. Do it. The skill — loading the server, reading the curve, turning tokens/sec into a dollar number, and committing to a serving decision with the number behind it — is what separates "I ran vLLM once" from an engineer who shipped a self-hosted tier and can defend the economics.
