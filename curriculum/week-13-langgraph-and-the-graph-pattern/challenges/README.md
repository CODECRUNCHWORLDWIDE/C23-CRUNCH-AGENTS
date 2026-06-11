# Week 13 — Challenges

The exercises drill the mechanics — your first state graph, the four-node ReAct graph, the checkpointer that survives a kill. **The challenge is the syllabus hands-on lab.** You take your *actual* week-5 ReAct loop, re-implement it as a LangGraph state graph with a SQLite checkpointer, run the *same* 25-task benchmark through both, and produce the comparison the syllabus asks for: lines of code, observability, and resumability — loop versus graph, with numbers.

## Index

1. **[Challenge 1 — From ReAct loop to state graph](challenge-01-react-loop-to-graph.md)** — re-implement the week-5 hand-rolled ReAct agent as a LangGraph state graph (plan, retrieve, execute, critique) with a SQLite checkpointer so it survives a process kill; run the same 25-task benchmark; compare LoC, observability, and resumability against the loop. (~150 min)

Challenges are optional for passing the week, but this one *is* the syllabus lab in lab form, and it's the single best preparation for the **Phase III multi-agent capstone** (weeks 22–23), where this exact graph becomes the supervisor at the center of a multi-agent system. Do it. The skill — taking a tangled loop and making it an explicit, checkpointed, budgeted graph, then *proving* the graph is better with a measured comparison — is what separates an engineer who "used LangGraph once" from one who can defend the architecture choice with numbers.
