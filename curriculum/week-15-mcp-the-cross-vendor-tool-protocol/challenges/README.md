# Week 15 — Challenges

The exercises drill the mechanics — trace a session, build one server, drive one client. **The challenge makes you the engineer who ships a tool surface.** You stand up two MCP servers, run them over two different transports, consume them from a single LangGraph agent, and put the whole surface through a security review — the way a real agent integration actually gets built and signed off.

## Index

1. **[Challenge 1 — Two servers, two transports, one agent](challenge-01-two-servers-two-transports.md)** — a filesystem server (stdio) and a corpus-search server (streamable HTTP), consumed together from a LangGraph agent, with a path-traversal + argument-validation security review and a one-page memo. (~150 min)

Challenges are optional for passing the week, but this one *is* the syllabus deliverable in lab form and the direct input to week 17, where the red team attacks exactly this surface. Do it. The skill — exposing a tool over a stable protocol, consuming it transport-agnostically, and proving it's safe before you ship — is what separates an engineer who "got a tool working" from one who shipped a tool surface they can defend in a review. And in week 17, "defend in a review" stops being a metaphor: you'll write 25 adversarial prompts against this exact agent.
