# Week 11 — Challenges

The exercises drill the mechanics — budget a window, build a rolling summary, run the regression test. **The challenge makes you the engineer who has to build the whole memory system and prove it works.** You wire all three tiers, choose an eviction policy, run the 40-turn benchmark against a no-memory baseline, and commit to a design you can defend — the way an agent's memory actually gets built and justified.

## Index

1. **[Challenge 1 — The three-tier memory agent](challenge-01-three-tier-memory.md)** — episodic (rolling summary), semantic (vector store of user facts), procedural (tool-history log), under a context budget with an eviction policy. Run the 40-turn memory benchmark, compare LRU vs salience eviction, and measure recall against a no-memory baseline. (~150 min)

Challenges are optional for passing the week, but this one *is* the syllabus hands-on lab in committed form and the single best preparation for the Phase II milestone, which requires three memory tiers wired and measured. Do it. The skill — building the tiers, budgeting the context like a cache, choosing an eviction policy with a measured recall comparison, and proving the agent remembers a turn-3 fact in turn 38 — is what separates a junior who "added some memory" from an engineer who built a memory system and can show, with a number, exactly what it bought.
