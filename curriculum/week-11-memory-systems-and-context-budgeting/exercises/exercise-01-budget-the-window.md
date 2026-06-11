# Exercise 1 — Budget the Context Window Like a Cache

**Goal:** Allocate a token budget across the parts of a memory-equipped agent's prompt — system, semantic facts, episodic summary, recent turns, query — *enforce* it, and watch a naive "append every turn forever" approach blow past the window while the budgeted approach stays inside it and still recalls the fact. You will train the single most important habit of memory engineering: **the context window is a finite cache, and you spend it on purpose, not by accretion.**

**Estimated time:** 45 minutes. Guided.

---

## Setup

```bash
pip install anthropic                 # for count_tokens; or use a local tokenizer
export ANTHROPIC_API_KEY=sk-ant-...   # or point at a local model from week 6
```

The one rule that runs through the whole exercise: **count tokens with the model's tokenizer, never characters** (week 2's lesson). For Claude:

```python
import anthropic
client = anthropic.Anthropic()

def n_tokens(text: str, model="claude-sonnet-4-6") -> int:
    return client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    ).input_tokens
```

> Do **not** use `tiktoken` to count Claude tokens — it's a different tokenizer and undercounts. A budget measured in the wrong tokenizer is a budget that lies. If you're on a local model, use *that* model's tokenizer.

---

## Step 1 — The naive append-everything agent (watch it grow)

The naive "memory" is to keep appending every turn to the prompt. Simulate a long conversation and watch the token count climb:

```python
SYSTEM = "You are a helpful assistant with a good memory."

turns = []
for i in range(1, 41):                              # a 40-turn conversation
    turns.append({"role": "user", "content":
                  f"Turn {i}: here is some conversational content that takes up space."})
    turns.append({"role": "assistant", "content":
                  f"Acknowledged turn {i}, here is a reply of moderate length."})

# Naive prompt = system + EVERY turn, every time.
def naive_prompt_tokens(turns):
    body = SYSTEM + "\n" + "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    return n_tokens(body)

print("naive prompt tokens after 40 turns:", naive_prompt_tokens(turns))
```

Watch the number. It grows *linearly* with the conversation — and you pay it *every single turn*, because you re-send the whole history (the API is stateless). On a long conversation this either hits the context-window limit (hard failure) or just costs a fortune and degrades accuracy (lost in the middle). **This is a cache with no eviction.**

---

## Step 2 — Define a budget

Now allocate the window deliberately. Pick a total budget (well under the window) and slice it:

```python
class ContextBudget:
    def __init__(self, total: int):
        self.total = total
        self.slices = {"system": 0, "semantic": 0, "episodic": 0,
                       "recent": 0, "query": 0}

    def used(self):       return sum(self.slices.values())
    def fits(self):       return self.used() <= self.total
    def over_by(self):    return max(0, self.used() - self.total)

budget = ContextBudget(total=2048)        # a deliberately small budget to feel the squeeze

# Suggested allocation (a tuned choice you sweep against the benchmark, not a constant):
ALLOC = {"system": 200, "semantic": 400, "episodic": 400, "recent": 800, "query": 248}
```

The allocation is a *hyperparameter* (like chunk size in week 8, the quant level in week 6) — you'd sweep it against the turn-38 benchmark and pick what maximizes recall within the budget. For now, the discipline is what matters: a budget exists, and it's measured.

---

## Step 3 — Assemble the prompt under the budget

Build the prompt from the tiers, sized to the slices. The recent turns get the `recent` slice (keep the newest that fit); older turns get summarized into the `episodic` slice; the user's durable facts go in `semantic`:

```python
def fit_recent(turns, budget_tokens):
    """Keep the most recent turns that fit the 'recent' slice (LRU on the transcript)."""
    kept, used = [], 0
    for t in reversed(turns):
        tok = n_tokens(t["content"])
        if used + tok > budget_tokens:
            break
        kept.append(t); used += tok
    return list(reversed(kept)), used

semantic_facts = "project: Helios; language: Python; tier: enterprise"   # from semantic tier
episodic_summary = "User discussed setup, asked about pricing, chose the enterprise tier."

recent, recent_used = fit_recent(turns, ALLOC["recent"])
budget.slices["system"]   = n_tokens(SYSTEM)
budget.slices["semantic"] = n_tokens(semantic_facts)
budget.slices["episodic"] = n_tokens(episodic_summary)
budget.slices["recent"]   = recent_used
budget.slices["query"]    = n_tokens("What's my project called?")

print("budgeted prompt tokens:", budget.used(), "/", budget.total,
      "fits:", budget.fits())
```

Compare `budget.used()` (constant, bounded) to `naive_prompt_tokens(turns)` (linear, growing). The budgeted prompt is a *fraction* of the naive one — and it *still contains the answer* ("project: Helios" in the semantic slice), because the durable fact was promoted to semantic memory instead of being left to scroll out of the transcript.

---

## Step 4 — Place the important content at the edges (lost in the middle)

A free quality lever (Lecture 2 §3): put the semantic facts and the query at the *start and end* of the assembled prompt, not buried in the middle of the episodic dump — because models recall edge content far better than middle content (the U-shaped curve). Assemble in this order:

```
[semantic facts]        <- near the top edge
[episodic summary]
[recent turns]
[the user's query]      <- at the bottom edge
```

Same tokens, better recall. The fact the model needs ("Helios") sits where the model reads best.

---

## Step 5 — Write down what you found

Build a small table in `notes/week-11/budget.md`:

| Approach | Tokens after 40 turns | Contains the turn-3 fact? | Cost per turn (grows or constant?) |
|---|---|---|---|
| Naive append-everything | | | |
| Budgeted (3 tiers) | | | |

The point is the contrast: the naive approach grows without bound and *still* loses the turn-3 fact in the middle; the budgeted approach is constant-size and keeps the fact retrievable.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] You measured the **naive prompt's token growth** over 40 turns (linear) and can state the failure mode (window overflow / cost / lost-in-the-middle).
- [ ] You built a **budget** with named slices, measured with the *model's tokenizer* (not characters, not `tiktoken`).
- [ ] The **budgeted prompt fits** the total and is a fraction of the naive prompt's size.
- [ ] The budgeted prompt **still contains the turn-3 fact** (because it was promoted to the semantic slice), placed at an *edge* for recall.
- [ ] `notes/week-11/budget.md` has the comparison table filled in.

---

## Stretch

- **Sweep the allocation.** Try `recent`-heavy vs `semantic`-heavy splits of the same total budget and predict which recalls the turn-3 fact better. (Semantic-heavy should win on old facts.) Measure it in Exercise 3.
- **Plant a fact in the middle.** Build a long context with the answer in the *middle* vs at the *edge*, ask a real model, and watch recall drop for the middle placement — the lost-in-the-middle effect, measured (Lecture 2 §3, and a README stretch goal).
- **Enforce eviction.** When `budget.over_by() > 0`, drop the lowest-value content (start with LRU on the recent turns) until it fits — the eviction policy from Lecture 2 §4, made concrete.

---

When this feels comfortable, move to [Exercise 2 — Rolling-summary episodic memory](exercise-02-rolling-summary.py).
