# Week 14 — Challenges

The exercises drill the mechanics — a Mastra agent, a supervisor that routes, a pipeline that resumes after a crash. **The challenge makes you the engineer who has to choose a stack and defend it.** You build the *same* supervisor in two languages, wire one of them to Inngest for event-driven durable invocation, crash it on purpose, and write down — with evidence — which stack you'd ship and why.

## Index

1. **[Challenge 1 — The polyglot supervisor](challenge-01-polyglot-supervisor.md)** — the same supervisor in Mastra (TypeScript) and LangGraph (Python, week 13); wire the Mastra one to Inngest so a new file in S3 (or a local equivalent) triggers a durable research run; compare developer ergonomics, type safety, and the resume-after-crash story across both stacks. (~150 min)

This is the **syllabus hands-on lab** in lab form, and it earns the headline skill of the week: **polyglot agent design**. Challenges are optional for passing the week, but this one *is* the deliverable the syllabus describes, and the single best preparation for the production weeks (18 observability; 19–24 serving and the capstone), where "which stack, and does it resume?" is the question a reviewer actually asks. Do it. The skill — expressing one architecture in two ecosystems, and scoring *ergonomics* and *durability* as separate axes — is what separates an engineer who "used a framework" from one who *chose* it, with a crash test behind the choice.
