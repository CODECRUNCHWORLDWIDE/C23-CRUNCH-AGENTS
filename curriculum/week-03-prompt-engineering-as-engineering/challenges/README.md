# Week 3 — Challenges

The exercises drill the mechanics. **The challenge is the syllabus hands-on lab made real:** you take a prompt that is genuinely failing on a customer-support dataset and turn it into a regression-tested artifact with reproducible scores — the way you'd harden a prompt before it ships to real customers.

## Index

1. **[Challenge 1 — Regression-test a prompt](challenge-01-regression-test-a-prompt.md)** — start from a poorly-performing customer-support prompt, build a `promptfoo` harness with 30 golden examples, iterate through six prompt versions while committing each to git with its measured pass rate, and deliver a regression-tested prompt with reproducible scores. (~90 min)

Challenges are optional for passing the week, but this one is the single best preparation for the Phase I capstone milestone, where the syllabus requires "prompts versioned in git; promptfoo regression tests committed." It is also the closest thing in the course to the real workflow of a production AI engineer hardening a prompt under review. Do it. The skill — turning "the prompt feels better" into "v6 passes 28/30, here is the diff and the SHA" — is exactly what separates someone who "has prompted an LLM" from an engineer who can own a prompt in production.
