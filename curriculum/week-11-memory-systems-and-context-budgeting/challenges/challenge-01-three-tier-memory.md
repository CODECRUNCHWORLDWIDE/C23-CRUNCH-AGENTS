# Challenge 1 — The Three-Tier Memory Agent

**Time estimate:** ~150 minutes.

## Problem statement

You have a chat agent and a memory problem: it forgets the user's project name nine turns after hearing it. You are going to fix it the only way that's defensible — build the three memory tiers, budget the context like the cache it is, choose an eviction policy, and *prove* the agent remembers a turn-3 fact in turn 38 with a measured recall rate against a no-memory baseline. Then you write down *which* tier and *which* eviction policy did the work — because a memory system you can't measure is a memory system you got lucky with.

This is the syllabus hands-on lab in committed form. The output is a working three-tier memory agent and a number: recall rate on a 40-turn benchmark, three-tier vs no-memory, with the eviction policy chosen by a measured comparison.

## The three tiers

Build exactly these three, each with its own storage and retrieval:

1. **Episodic** — a *rolling summary* of older turns (Lecture 1 §3) plus the recent N turns verbatim. "What was said." Recency-retrieved.
2. **Semantic** — a *vector store of user facts* (your week-10 `crunchstore` adapter). Extract durable facts from each turn, upsert them, retrieve by similarity to the query. "What's true." This is the tier that makes the turn-38 recall work.
3. **Procedural** — a *tool-history log* of the agent's tool calls and outcomes. "What I did." By-tool / recency retrieved; check it before re-running a tool.

## What is fixed (do not let these vary)

- **The benchmark:** a 40-turn conversation that plants durable facts early (turn 3: "project is Helios"; turn 7: "I prefer Python"; turn 12: "enterprise tier") and asks about them late (turns 34–40), with distractor turns filling the gap so a fact must *survive* to be recalled.
- **The metric:** recall rate (facts recalled / facts asked), measured for the three-tier agent *and* a no-memory baseline (recent window only). The delta is the headline.
- **The budget:** a fixed context budget (e.g. 4096 tokens) with named slices (system / semantic / episodic / recent / query), measured with the model's tokenizer, enforced on every turn.

## The memory loop

The whole agent reduces to a loop: observe → summarize → assemble-under-budget → respond → (measure).

```python
from crunchstore.adapters import load            # week 10, the semantic tier

class MemoryAgent:
    def __init__(self, budget_tokens=4096):
        self.episodic_summary = ""
        self.recent = []                          # verbatim recent turns
        self.semantic = load("pgvector")          # week-10 store of user facts
        self.semantic.create("user_facts", dim=1024)
        self.procedural = []                      # tool-history log
        self.budget = budget_tokens

    def observe(self, turn):
        self.recent.append(turn)
        if len(self.recent) > RECENT_WINDOW:
            aged_out = self.recent.pop(0)
            self.episodic_summary = roll_summary(self.episodic_summary, aged_out)
        fact = extract_durable_fact(turn)         # LLM or heuristic
        if fact:
            self.semantic.upsert("user_facts", [(fact_id(fact), embed(fact), {"text": fact})])

    def respond(self, query):
        facts = self.semantic.search("user_facts", embed(query), k=3)   # retrieve
        prompt = assemble_under_budget(           # budget + edge placement + eviction
            system, facts, self.episodic_summary, self.recent, query, self.budget)
        return llm(prompt)
```

The no-memory baseline is the *same loop with the semantic tier and the episodic summary disabled* — just the recent window. That control is what makes the recall delta meaningful.

## Acceptance criteria

- [ ] A `challenge-01/` directory with a runnable `memory_agent.py` implementing all three tiers and the memory loop.
- [ ] The **semantic tier uses your week-10 `crunchstore` adapter** (or the toy embedder from Exercise 3 if the store is unavailable, noted as such) — durable facts are upserted and retrieved by similarity.
- [ ] A **context budget** with named slices, measured with the model's tokenizer (not characters), enforced every turn; when over budget, content is evicted by an explicit policy.
- [ ] The **40-turn regression test** runs the three-tier agent and the no-memory baseline and prints a recall rate for each.
- [ ] The three-tier agent **recalls the turn-3 fact at turn 38**; the baseline does not — the recall delta is reported.
- [ ] An **eviction comparison**: run the benchmark under LRU vs salience-weighted eviction and report which recalls more old-but-important facts.
- [ ] A one-page `memory-memo.md` that states the recall delta, which tier carried it, the eviction policy chosen *with the number that justifies it*, and the budget allocation.
- [ ] At least one **promise-format line**, e.g. `turn 38: "what's my project called?" -> "Helios" ✓ (semantic) — 18/20 vs 2/20 baseline`.

## The trap (read after a first attempt)

The trap is **trusting the rolling summary to keep the important facts.** The rolling summary is *lossy* (Exercise 2) — each pass can drop a fact, and once dropped it's gone. If your only memory of "the project is Helios" is the episodic summary, then by turn 38 a summarization pass may have dropped it and recall fails. The fix is the **semantic tier**: extract durable facts and store them in the vector store, where they survive *regardless* of what the rolling summary forgets. If your three-tier recall is no better than the baseline, the most likely cause is that durable facts aren't reaching the semantic tier — check the extraction and the upsert first. The whole reason there are *three* tiers is that no single one is enough: episodic bounds tokens, semantic guarantees recall, procedural prevents tool-loops.

A second, subtler trap: **measuring recall only at short gaps.** If you ask about the turn-3 fact at turn 6, the baseline *still has it in the recent window* and "passes" — so the test shows no memory benefit. You must ask *late* (turn 34+), after the fact has scrolled out of the recent window, so the test actually measures *survival*. A gap shorter than the recent window measures nothing (exactly Exercise 3's design).

## Stretch goals

- **Salience-weighted eviction.** Score each memory by recency × importance and evict the lowest; compare its turn-38 recall to plain LRU on the benchmark (Lecture 2 §4). Predict which wins, then measure.
- **Knowledge-graph memory.** Store facts as `(subject, relation, object)` triples in Postgres and answer a *multi-hop* memory question ("what does the company my user works at use for CI?") that flat vector memory can't (Lecture 1 §2.2; week 10's GraphRAG idea for memory).
- **Lost-in-the-middle measurement.** Place the retrieved fact in the *middle* vs at the *edge* of the assembled prompt and measure the recall difference with a real model (Lecture 2 §3).
- **Procedural payoff.** Plant a scenario where the agent would re-run a failing tool; show that the procedural log lets it avoid the repeat (the "infinite tool-call loop" prevention from week 5).

## Why this matters

The Phase II milestone requires three memory tiers wired and measured, and the capstone's agent has to hold a multi-turn conversation without forgetting the user's context. The reviewer will not ask you to recite the three tiers — they'll start a conversation, tell the agent something on turn 3, and check whether it remembers on turn 38. This challenge *is* that conversation, rehearsed and measured: you built the tiers, you have the recall number, you can say which tier carried it and which eviction policy you chose and why. Every agent you ship after this either remembers its conversation or doesn't — the engineer who *measured* its memory, against a baseline, with an eviction policy chosen by number, is the one whose agent doesn't ask the user to repeat themselves on turn 38. The fact survived the budget, and you can prove it.
