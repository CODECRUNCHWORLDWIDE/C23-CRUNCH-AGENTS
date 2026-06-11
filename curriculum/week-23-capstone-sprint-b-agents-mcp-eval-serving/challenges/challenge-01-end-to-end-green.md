# Challenge 1 — End-to-End Green

**Time estimate:** ~3 hours.

## Problem statement

You have the parts: a supervisor that routes (Exercise 1), a corpus MCP server (Exercise 2), a cost-tracked local-vs-vendor router (Exercise 3), and from Sprint A the hybrid retrieval and the memory tiers. This challenge assembles them into one runnable system and proves the assembly works the only way it can honestly be proven: by a *measured eval gate* on the 100-question gold set, green.

This is the integration risk of the capstone, front-loaded. By the end you have the thin slice running end-to-end, instrumented, and gated — the foundation the mini-project ships and the system week 24 attacks.

## What "done" looks like

```
$ python -m capstone.eval run --gold gold/eval_100.jsonl --gate
... (100 questions run through the supervisor graph) ...
ragas: faithfulness=0.91 context_recall=0.88 context_precision=0.84 answer_relevancy=0.90
judge (calibrated): 0.87 on 50-q subset
GATE: PASS (faithfulness 0.91 >= 0.85, judge 0.87 >= 0.80)
```

Plus, for any single question, a trace you can open in Langfuse *and* Phoenix that shows the routing chain, the tool calls, the model that served each turn, and the token cost.

## The build, in order (thin slice first)

Do these in order. Do **not** skip ahead to deepen a component before the slice runs once.

1. **Wire the supervisor to real subordinates.** Replace Exercise 1's stub subordinates with real ones: the retrieval-agent calls the corpus MCP server (Exercise 2) over stdio; the writing-agent calls the routed model (Exercise 3) to synthesize a grounded answer from the retrieved context; the critique-agent checks the draft is grounded; the code-agent calls the calculator MCP tool when computation is needed.

2. **Serve the two tiers.** Bring up vLLM for the local 7B (or Ollama if no GPU), put LiteLLM in front with the vendor fallback to `claude-opus-4-8`, and point the writing-agent's routed call at LiteLLM. Confirm an easy query is served local and a hard one vendor (read the LiteLLM logs / the trace).

3. **Instrument with OTel.** Every supervisor decision, subordinate run, MCP tool call, and model request is a span following the Gen-AI conventions, exported to both Langfuse and Phoenix. Confirm one query produces a readable trace in both.

4. **Build the eval suite.** Run the 100-question gold set through the full system, collect `(question, answer, contexts, ground_truth)` per question, run Ragas (the four metrics), run the calibrated judge on a 50-question subset (10 human-labeled calibration examples in the judge prompt), and implement the gate (thresholds from Lecture 2).

5. **Make it green.** Run the gate. If it's red, read the trace for the failing questions and triage by metric: low context recall → retrieval missing material (chunking/`k`/retriever); low context precision → too much noise (tighten `k`, lean on reranker); low faithfulness → writing-agent confabulating (tighten the grounding prompt); judge disagrees with humans → calibration ambiguous (fix the rubric). Deepen the flagged layer, re-run, repeat until `PASS`.

## What is fixed (do not change to chase a number)

- **The gold set.** 100 questions, fixed. Do not edit the gold answers to match your output — that's grading your own homework.
- **The metric thresholds.** Faithfulness ≥ 0.85, answer relevancy ≥ 0.80, context precision ≥ 0.75, context recall ≥ 0.80, judge ≥ 0.80. If you can't hit them, fix the system, not the threshold.
- **The two-tier split.** The local tier must serve *some* of the easy routes — a capstone that routes everything to the vendor is not the cost-engineered system the syllabus specifies (and the cost report will show it).

## The two traps

> **Trap 1 — the "demo-green" trap.** Your favorite three queries answer beautifully, so you call it done. The gate runs on 100 questions; a system tuned to three will score badly on the other 97. Run the *full* gate, not a spot-check, and fix the tail.

> **Trap 2 — the "uncalibrated judge" trap.** You run an LLM-judge with no calibration examples and it gives everything a 4. That number is a vibe. Put your 10 human labels in the judge prompt as the rubric anchor and spot-check that the judge agrees with them before you trust the 50-question score. An uncalibrated 0.87 and a calibrated 0.87 are not the same number.

## Acceptance criteria

- [ ] `python -m capstone.eval run --gold gold/eval_100.jsonl --gate` runs all 100 questions through the supervisor graph and prints `GATE: PASS`.
- [ ] The supervisor routes (it does not do the retrieval/writing itself); the trace shows the routing chain per question.
- [ ] The retrieval-agent calls the corpus MCP server over a transport (stdio is fine for the slice); the writing-agent calls the routed model via LiteLLM.
- [ ] An easy query is served by the local tier and a hard query by `claude-opus-4-8`, visible in the trace / LiteLLM logs.
- [ ] One query's trace is readable in both Langfuse and Phoenix, showing routing, tool calls, the serving model, and token cost.
- [ ] The judge is calibrated (10 human labels in the prompt) and spot-checked to agree with them.
- [ ] Committed.

## Deliverable

A green gate on the 100-question set, plus a short note (`notes/week-23/gate.md`) recording: the four Ragas numbers, the judge mean, which metric you had to fix to get green and how, and a link to one trace. This note is the evidence for the mini-project's eval report and the rubric's measurement axis.
