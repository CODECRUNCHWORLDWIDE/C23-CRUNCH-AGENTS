# Week 21 — Challenges

The exercises drill the mechanics — account for cost, cache by meaning, route by difficulty. **The challenge makes you the engineer who has to cut the bill without breaking the product.** You run a routing-plus-semantic-cache pipeline over a 500-query workload, measure the cost reduction *and* the quality delta against an all-frontier baseline, and write the cost-reduction memo that proves you saved the money *and* held the quality — the way the call actually gets made on a real LLM product.

## Index

1. **[Challenge 1 — The cost-reduction lab](challenge-01-cost-reduction.md)** — route easy/hard + a semantic cache over a fixed 500-query workload, measure the cost cut and the quality delta vs the all-frontier baseline, plot the cache-hit rate over time, and write the cost-reduction memo with the saving *and* the preserved-quality number. (~150 min; `--mock` path for no-key completion)

Challenges are optional for passing the week, but this one **is the syllabus cost-engineering lab** in hands-on form, and the single best preparation for the capstone's required cost report. Do it. The skill — cutting cost measurably while proving the answers held, on a fixed labeled workload — is what separates a junior who "made it cheaper" (and quietly degraded the product) from an engineer who shipped an 88% cost cut with a quality delta inside tolerance and can prove both.
