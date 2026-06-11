# Mini-Project — `crunchmem`: The Three-Tier Memory System + Regression Harness

> Build a reusable memory system that any chat agent can import to gain three memory tiers — episodic (rolling summary), semantic (vector store of user facts), procedural (tool-history log) — budgeted under a fixed context window with a pluggable eviction policy, and a 40-turn regression harness that proves the agent remembers a turn-3 fact in turn 38 against a no-memory baseline. So "does this agent remember, and how do you know?" becomes a command, not a vibe.

This is the artifact that turns agent memory from folklore into a measurement. After this week, giving an agent memory is `from crunchmem import MemoryAgent` and reading a recall rate — not hoping the rolling summary kept the right facts. The system is tier-pluggable, budget-enforcing, and eviction-policy-swappable, and the semantic tier reuses your week-10 `crunchstore` adapter **unchanged**.

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** This memory system is the Phase II milestone's "three memory tiers wired and measured" deliverable, and it's imported directly by the **capstone's supervisor agent**, which holds multi-turn conversations over a 10 GB corpus and must not forget the user's context across a long session. The regression harness you build here generalizes into week 12's Ragas evaluation discipline. Build it well now; the capstone leans on it.

---

## What you will build

A small Python package `crunchmem` with four deliverables:

1. **`crunchmem/tiers.py`** — the three memory tiers behind a uniform interface: `EpisodicMemory` (rolling summary + recent window), `SemanticMemory` (the `crunchstore`-backed fact store), `ProceduralMemory` (the tool-history log). One `observe`/`retrieve` shape per tier, so the agent never has to remember that semantic embeds and procedural logs.
2. **`crunchmem/budget.py`** — the context budgeter: allocate token slices, measure with the model's tokenizer, place important content at the edges (lost-in-the-middle), and evict by a pluggable policy when over budget.
3. **`crunchmem/agent.py`** — the memory loop: observe (promote durable facts to semantic, log tools to procedural, summarize episodic) → assemble-under-budget → respond. Plus the no-memory baseline (same loop, tiers disabled).
4. **`crunchmem/bench.py`** — the 40-turn regression harness: plant facts early, ask late, score recall for the three-tier agent and the baseline, and compare eviction policies.

By the end you have a public repo of ~450–550 lines of Python that any future agent can `from crunchmem import MemoryAgent` and stop forgetting.

---

## Why a module and not a notebook

You could do all of this in a Jupyter notebook. Don't — not as the artifact. A module gives you:

- **Reuse.** The capstone imports your `MemoryAgent`. A notebook gets copy-pasted, drifts, and rots.
- **A fixed measurement.** The benchmark, the recall metric, and the "three-tier vs baseline" control live in code, version-controlled. "Did the new eviction policy help?" is answered by re-running the *same* `bench.py`, not by eyeballing a cell.
- **A CLI.** `bench --eviction salience --turns 40` is greppable, scriptable, and CI-able. A notebook cell is none of those.

Notebooks are great for *exploring* one conversation's memory by eye. The thing you ship and depend on is a module. That's the senior-shop convention in 2026.

---

## Package layout

```
crunchmem/
├── pyproject.toml
├── docker-compose.yml          # pgvector for the semantic tier (from week 10)
├── README.md                   # the recall results + the memory memo
├── benchmarks/
│   └── memory_bench.json       # the 40-turn benchmark: planted facts + ask turns
├── crunchmem/
│   ├── __init__.py
│   ├── tiers.py                # EpisodicMemory, SemanticMemory, ProceduralMemory
│   ├── budget.py               # the context budgeter + eviction policies
│   ├── agent.py                # the memory loop + the no-memory baseline
│   ├── bench.py                # the 40-turn regression harness
│   └── cli.py                  # the `bench` command
└── tests/
    ├── test_tiers.py           # each tier stores and retrieves correctly
    └── test_budget.py          # the budget enforces the cap and evicts by policy
```

Your week-10 `crunchstore` package is a dependency; `SemanticMemory` uses its adapter **unchanged**.

---

## Deliverable 1 — `tiers.py` (the three tiers)

This is the heart of the project. Each tier has a different storage and retrieval pattern; the uniform `observe`/`retrieve` shape hides that.

```python
"""crunchmem.tiers — the three memory tiers behind one observe/retrieve shape.

The per-tier mechanics (episodic summarizes, semantic embeds, procedural logs)
live HERE. The agent loop just observes turns and retrieves for a query.
"""
from __future__ import annotations

from crunchstore.adapters import load          # week 10, the semantic tier's store


class EpisodicMemory:
    """What was said: rolling summary of older turns + recent verbatim window."""
    def __init__(self, recent_window=4, summarize=None):
        self.summary = ""
        self.recent: list[dict] = []
        self.window = recent_window
        self._summarize = summarize            # the rolling-summary fn (Lecture 1 §3)

    def observe(self, turn: dict) -> None:
        self.recent.append(turn)
        if len(self.recent) > self.window:
            aged = self.recent.pop(0)
            self.summary = self._summarize(self.summary, [aged])

    def retrieve(self) -> tuple[str, list[dict]]:
        return self.summary, list(self.recent)


class SemanticMemory:
    """What's true: durable facts in the crunchstore vector store, by similarity."""
    def __init__(self, embed, collection="user_facts", dim=1024):
        self.store = load("pgvector")
        self.store.create(collection, dim=dim)
        self.collection, self.embed, self._n = collection, embed, 0

    def observe(self, fact: str | None) -> None:
        if not fact:
            return
        self.store.upsert(self.collection,
                          [(f"f{self._n}", self.embed(fact), {"text": fact})])
        self._n += 1

    def retrieve(self, query: str, k=3) -> list[str]:
        # TODO 1: embed the query, search the store, return the fact texts.
        ...


# TODO 2: ProceduralMemory — an append-only tool-history log. observe(tool_call,
#   result); retrieve(tool_name) returns prior calls/outcomes for that tool, so the
#   agent can avoid re-running a failing call (the week-5 loop-prevention).
```

> **The rule the project enforces:** durable facts go to **semantic** memory, not just the episodic summary. The rolling summary is lossy (Exercise 2); if the only copy of "project: Helios" is the summary, a summarization pass can drop it and turn-38 recall fails. `extract_durable_fact` + `SemanticMemory.observe` is the load-bearing path — if `grep -rn "summary" crunchmem` is the *only* place a planted fact lives, you've fallen in the trap.

---

## Deliverable 2 — `budget.py` (the context budgeter + eviction)

The budgeter sizes the assembled prompt. It must:

- **Allocate** token slices (system / semantic / episodic / recent / query).
- **Measure** with the *model's tokenizer* (`count_tokens` for Claude; the model's tokenizer for a local model) — never characters, never `tiktoken` for Claude (Lecture 2 §2).
- **Place** the most important content (semantic facts, the query) at the *edges* (lost-in-the-middle, Lecture 2 §3).
- **Evict** by a pluggable policy when over budget.

```python
def assemble_under_budget(system, semantic_facts, episodic_summary, recent_turns,
                          query, budget_tokens, count, eviction):
    """Build the prompt sized to the budget; evict by `eviction` if over."""
    # TODO 3: measure each slice with `count`; if the total exceeds budget_tokens,
    #   call `eviction(...)` to drop the lowest-value content until it fits; place
    #   semantic_facts near the top and query at the bottom (the edges).
    ...


def evict_lru(items, budget_tokens, count):
    """Keep the most recent items that fit (Lecture 2 §4.1)."""
    # TODO 4: keep newest-first until the budget is hit.
    ...


def evict_salience(items, budget_tokens, count, alpha=0.7, beta=0.3):
    """Keep highest (alpha*salience + beta*recency) items that fit (Lecture 2 §4.2)."""
    # TODO 5: score, sort, keep the highest-scoring that fit.
    ...
```

The non-negotiables `budget.py` enforces:

- **Tokens are counted with the model's tokenizer.** A character budget is forbidden — it's the budget-that-lies bug.
- **The budget is enforced every turn.** When over, content is *evicted by policy*, not truncated blindly.
- **Important content is edge-placed.** The retrieved fact and the query don't get buried in the middle.

---

## Deliverable 3 — `agent.py` (the memory loop + baseline)

```python
class MemoryAgent:
    def __init__(self, llm, embed, summarize, budget=4096, eviction=evict_salience):
        self.episodic = EpisodicMemory(summarize=summarize)
        self.semantic = SemanticMemory(embed=embed)
        self.procedural = ProceduralMemory()
        self.llm, self.budget, self.eviction = llm, budget, eviction

    def observe(self, turn):
        self.episodic.observe(turn)
        self.semantic.observe(extract_durable_fact(turn))     # the load-bearing path
        # tool calls -> self.procedural.observe(...)

    def respond(self, query):
        facts = self.semantic.retrieve(query, k=3)
        summary, recent = self.episodic.retrieve()
        prompt = assemble_under_budget(SYSTEM, facts, summary, recent, query,
                                       self.budget, self.count, self.eviction)
        return self.llm(prompt)


class NoMemoryBaseline(MemoryAgent):
    """The control: same loop, semantic + episodic-summary disabled — recent only."""
    def respond(self, query):
        _, recent = self.episodic.retrieve()
        prompt = assemble_under_budget(SYSTEM, [], "", recent, query,
                                       self.budget, self.count, self.eviction)
        return self.llm(prompt)
```

The baseline is the *same code with the tiers off* — that's what makes the recall delta a clean measurement of what memory bought.

---

## Deliverable 4 — `bench.py` + `cli.py` (the regression harness)

```bash
python -m crunchmem bench \
    --benchmark benchmarks/memory_bench.json \
    --eviction salience \
    --turns 40
```

It should run the three-tier agent and the no-memory baseline through the 40-turn benchmark and print:

```
MEMORY REGRESSION — three-tier vs no-memory baseline
  three-tier  : recall 18/20 = 0.90   (semantic tier carried turns 3,7,12)
  no-memory   : recall  2/20 = 0.10   (forgot everything past the recent window)
  delta       : +0.80

  turn 38: "what's my project called?" -> "Helios" ✓  (semantic)
  turn 41: "what's my deadline?"        -> "I don't have that" ✗  (lost in summary)

eviction comparison (three-tier):
  LRU       : recall 0.75
  salience  : recall 0.90   <- keeps the important old facts; ship this
```

The winner line makes the *judgment call* the table sets up: salience beats LRU on this benchmark because it keeps the old-but-important facts — a decision, printed and defended.

---

## Rules

- **You may** read the memory papers, the framework docs, the lecture notes, and your week-10 code.
- **You must not** rely on the rolling summary alone to keep durable facts — extract them to the semantic tier (the trap).
- **You must not** measure the context budget in characters (or `tiktoken` for a Claude model) — use the model's tokenizer.
- **You must not** ask about a planted fact *within* the recent window — ask late, after it's scrolled out, or the test measures nothing.
- **You must** include the no-memory baseline as the control; the recall delta is the headline.
- Python 3.12, `anthropic` (or a local model from week 6), `sentence-transformers`, your `crunchstore` package, `numpy`, plus `pytest`. Docker for pgvector.

---

## Acceptance criteria

- [ ] A public GitHub repo named `c23-week-11-crunchmem-<yourhandle>`.
- [ ] All three tiers implemented behind a uniform interface; the semantic tier uses `crunchstore` **unchanged**.
- [ ] A context budget measured with the model's tokenizer, enforced every turn, with a pluggable eviction policy (LRU and salience both implemented).
- [ ] `bench.py` runs the three-tier agent and the no-memory baseline on a 40-turn benchmark and reports a recall rate for each, with the delta.
- [ ] The three-tier agent recalls a turn-3 fact at turn 38 (asked *after* it scrolled out of the recent window); the baseline does not.
- [ ] An eviction comparison (LRU vs salience) with the recall numbers and a chosen policy.
- [ ] `pytest` passes, with at least:
  - `test_tiers.py`: each tier stores and retrieves correctly; a durable fact reaches the semantic tier.
  - `test_budget.py`: the budget enforces the cap and evicts by the selected policy.
- [ ] A `README.md` with the recall results, the run commands, and the **one-page memory memo** (the recall delta, which tier carried it, the eviction policy chosen with its number, and the budget allocation).
- [ ] Committed and pushed.

---

## Grading rubric (100 points)

| Area | Points | What we look for |
|---|---:|---|
| **Three tiers** | 25 | Episodic (rolling summary + recent), semantic (crunchstore facts), procedural (tool log) all implemented behind a uniform interface; durable facts reach semantic, not just the summary. |
| **Context budget** | 20 | Token slices allocated; measured with the model's tokenizer (never characters/tiktoken); enforced every turn; important content edge-placed. |
| **Eviction policy** | 15 | LRU and salience both implemented and pluggable; the salience-vs-LRU recall comparison is measured. |
| **Regression measurement** | 25 | The 40-turn benchmark with a no-memory baseline control; recall rate reported for both; turn-3-in-turn-38 demonstrated; delta is the headline. |
| **Tests** | 10 | `test_tiers` and `test_budget` green; the durable-fact-to-semantic path and the budget cap are proven. |
| **Docs & hygiene** | 5 | Clear README + memo, no secrets, sensible commits, no DB volumes checked in. |

**90+** is portfolio-grade and ready to back the capstone's supervisor agent. **70–89** works but leans on the summary (soft semantic tier) or measures at too-short a gap. **Below 70** means the memory isn't measured against a baseline — fix that first, because the Phase II milestone and the capstone both require *measured* memory.

---

## Stretch goals

- **Knowledge-graph memory tier.** Add `(subject, relation, object)` triples in Postgres and answer a multi-hop memory question flat vector memory can't (Lecture 1 §2.2).
- **Lost-in-the-middle measurement.** Place the retrieved fact in the middle vs at the edge and measure the recall difference with a real model (Lecture 2 §3).
- **Hierarchical summarization.** Replace the flat rolling summary with a multi-level one for very long conversations; measure whether it improves recall on a 100-turn benchmark (Lecture 1 §3.2).
- **CI.** A GitHub Actions workflow that spins up pgvector, runs `pytest`, and runs a headless three-tier-vs-baseline bench. Green check on every push.

---

## How this connects to the rest of C23

- **Week 10 (vector stores)** gave you `crunchstore`; this project uses it as the semantic memory tier — the same adapter, pointed at accumulated facts instead of a corpus.
- **Week 12 (evaluation)** generalizes this regression test into Ragas + LLM-as-judge; your recall-vs-baseline discipline is the seed of that eval mindset, and the Phase II milestone requires *this* — three tiers, measured.
- **The capstone** imports your `MemoryAgent` so the supervisor holds a long conversation without forgetting the user's context; your turn-38 recall number is what says it works.

When you've finished, push the repo and take the [quiz](../quiz.md).
