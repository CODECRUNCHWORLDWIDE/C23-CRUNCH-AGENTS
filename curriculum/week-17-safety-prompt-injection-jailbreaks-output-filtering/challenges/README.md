# Week 17 — Challenges

The exercises drill the pieces — write attacks, build filters, measure ASR on a toy agent. **The challenge makes you the engineer who red-teams a real system and signs off on it.** You attack your own week-15 MCP-tool agent with 25 adversarial prompts, measure the attack-success-rate, harden it with three defense layers, re-measure, and write the threat model — the way a security review of an agentic product actually gets done, where the deliverable is a number and an honest list of what still gets through.

## Index

1. **[Challenge 1 — Red-team your agent](challenge-01-red-team-your-agent.md)** — 25 adversarial prompts (direct, indirect, tool-argument) against your week-15 agent, ASR before and after three defenses (input filtering, structured argument validation, output classifier), and a one-page threat model. (~150 min)

Challenges are optional for passing the week, but this one *is* the syllabus deliverable in lab form and a required piece of the **Phase III milestone** (a multi-agent system with a written threat model and live week-17 defenses). Do it. The skill — attacking your own system, measuring what lands, hardening, re-measuring, and writing an honest threat model that names the residual — is what separates an engineer who *says* their agent is safe from one who can show the ASR dropped from 0.64 to 0.08 and name the two attacks that still get through. And in week 24's chaos drill, the prompt-injection-on-a-tool scenario tests *these defenses* under fire — so the harder you red-team here, the calmer that 4 AM incident is.
