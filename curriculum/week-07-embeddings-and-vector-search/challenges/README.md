# Week 7 — Challenges

The exercises drill the mechanics. **The challenge makes you the engineer who has to pick.** You're handed one corpus, four embedding models, and a question every team eventually faces: *which embedding do we ship, and how do you know?* You answer it the only way that counts — with a measured bakeoff and a one-page memo a reviewer can act on.

## Index

1. **[Challenge 1 — The embedding bakeoff](challenge-01-embedding-bakeoff.md)** — embed one legal corpus with four models (three open, one vendor or a fourth open if you have no API key), run a 40-query gold set, report top-1 / top-5 / MRR per model, and defend a pick in a one-page memo. (~90 min)

Challenges are optional for passing the week, but this one is the single best preparation for the **Phase II architecture review in week 12**, where you defend your embedding choice to a reviewer. Do it. The skill — running a fair bakeoff and reading the numbers honestly instead of reaching for the leaderboard's #1 — is exactly what separates an engineer who "knows embeddings" from one who can make a defensible production decision.

This challenge is also the direct on-ramp to the mini-project: the bakeoff harness you build here *is* the `crunchrag_embed` module, and you'll carry it into weeks 8 and 9. Build it once, build it well.
