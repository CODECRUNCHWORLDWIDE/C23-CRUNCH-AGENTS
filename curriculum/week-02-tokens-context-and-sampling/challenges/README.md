# Week 2 — Challenges

The exercises drill the mechanics. **The challenge puts you in the seat where someone hands you a working-but-expensive pipeline and asks "can you cut this bill without making it worse?"** You audit a real document-processing pipeline's token budget, find where the tokens go, and propose cuts you can defend with numbers — the way it happens when finance asks why the LLM line item doubled.

## Index

1. **[Challenge 1 — The token-budget audit](challenge-01-token-budget-audit.md)** — take a document-processing pipeline that summarizes incident reports, instrument it for per-request token accounting, find the waste (a bloated system prompt, redundant context, the wrong tier, output verbosity), and deliver a before/after cost report that cuts spend without losing quality. (~90 min)

Challenges are optional for passing the week, but this one is the single best rehearsal for the Phase IV cost-engineering work and for the most common real interview prompt in applied AI — "this is costing us $X/month, make it cheaper." The skill it builds — measure where the tokens go, cut the waste, prove quality held — is exactly what separates an engineer who can own a budget from one who just calls the API. Do it.
