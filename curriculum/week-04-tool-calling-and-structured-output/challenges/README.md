# Week 4 — Challenges

The exercises drill the mechanics. **The challenge makes you the platform engineer.** You're asked to build a tool surface that doesn't care which model is on the other end — one registry, two vendors, zero per-vendor tool code — and then to prove with a number that it works on both.

## Index

1. **[Challenge 1 — The cross-vendor tool bridge](challenge-01-cross-vendor-tool-bridge.md)** — define each tool exactly once (name, schema, Python impl, validation) and write two thin adapters so the same registry drives `claude-opus-4-8` and a local `qwen2.5:7b-instruct`. Then run a 20-task accuracy benchmark against both and report the gap. (~90 min)

Challenges are optional for passing the week, but this one is the single best preparation for the **Week 5 agent loop**, which imports a registry of exactly this shape, and for the **capstone**, where a routing layer sends easy tasks to a local model and hard ones to a frontier model — over one tool surface. The skill — keeping tools vendor-neutral so you can swap the model in a weekend — is exactly the "the course is the engineering, not the import" contract from the README, made concrete.
