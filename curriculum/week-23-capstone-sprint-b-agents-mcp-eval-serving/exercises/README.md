# Week 23 — Exercises

Three focused drills that take you from "supervisor routes a query" to "I built the corpus tool surface and the cost-tracked router." Each takes 45–60 minutes. Do them in order — exercise 3 routes the model the supervisor in exercise 1 chose, and exercise 2's corpus server is the tool the retrieval-agent calls.

## Index

1. **[Exercise 1 — Route the supervisor](exercise-01-route-the-supervisor.md)** — build the supervisor router node (structured decision, not parsed prose), wire its conditional edge, and read its routing decisions in a trace. (~50 min, guided)
2. **[Exercise 2 — The corpus MCP server](exercise-02-mcp-corpus-server.py)** — write the custom private-corpus search MCP server with the `mcp` SDK and harden the companion filesystem tool against path traversal. (~55 min, runnable)
3. **[Exercise 3 — The cost-tracked router](exercise-03-cost-tracked-router.py)** — build the easy-vs-hard classifier, route local-vs-vendor, and account for per-request cost so the capstone's cost report is real numbers. (~50 min, runnable)

## How to work the exercises

- Work in your **Sprint A capstone repo** (the week-22 workspace). These exercises produce the supervisor, the MCP surface, and the router that the mini-project assembles into the shipped system.
- Install the deps as each exercise needs them: `pip install langgraph langchain-core "mcp" anthropic litellm`. The vLLM tier is optional for the exercises (they mock the model call where a GPU isn't needed); the mini-project needs it live.
- Set `ANTHROPIC_API_KEY` in your environment. The supervisor router, the judge, and the vendor route all call `claude-opus-4-8`. The exercises keep the call counts tiny so the spend is a few cents.
- **Read the trace, not the print statements.** Exercise 1's whole point is that the routing decision is visible in the trace. When the supervisor routes wrong, open the trace and read the `reason` field — don't add `print()` and re-run.
- **Validate tool arguments before doing anything.** Exercise 2's path-traversal test must reject `../../etc/passwd`. A tool is an RCE primitive; the test proves your defense holds.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone. Exercise 2 runs the MCP server over stdio and exercises one tool call against it; it needs no GPU and no corpus (it ships a tiny in-memory corpus so the file always runs). Exercise 3 needs `ANTHROPIC_API_KEY` for the classifier; it mocks the served call so you can run it without a vLLM cluster.

```bash
# with ANTHROPIC_API_KEY set, and the venv active:
python3 exercise-02-mcp-corpus-server.py     # runs the server + a self-test client
python3 exercise-03-cost-tracked-router.py   # classifies + routes + prints the cost table
```

The first `claude-opus-4-8` call may take a few seconds (adaptive thinking). Don't pass `budget_tokens` or `temperature` — both 400 on this model; the exercises use `thinking={"type":"adaptive"}` and `output_config={"effort":...}`, which is the 2026-current surface.

## A note on determinism

The supervisor's routing decision is *mostly* deterministic given the same state, but the model can occasionally pick differently — that's why the decision is structured (an enum) rather than free text, and why you log the `reason`. The cost arithmetic in exercise 3 is fully deterministic given the token counts. If your routing decision flips between runs on a borderline query, that's expected; the gold-set eval (the mini-project) averages over 100 questions so a single flip doesn't move the score.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-23` to compare.
