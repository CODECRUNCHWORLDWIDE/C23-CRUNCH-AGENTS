#!/usr/bin/env python3
# Exercise 3 — The memory regression test (the turn-38 test, vs a no-memory baseline)
#
# Goal: Make memory a MEASURED capability, not a vibe. You will run a 40-turn
#       conversation that plants durable facts EARLY (turn 3: "project is Helios")
#       and asks about them LATE (turn 38: "what's my project called?"), with
#       distractor turns in between so a fact must SURVIVE to be recalled. You score
#       RECALL RATE for a THREE-TIER agent (episodic summary + semantic vector
#       memory) vs a NO-MEMORY baseline (recent turns only). The delta — e.g. 4/4 vs
#       0/4 — is the number that justifies the whole memory system.
#
# Estimated time: 50 minutes. Runnable.
#
# WHAT THIS DEPENDS ON (read before running)
#
#   * REQUIRED: `pip install numpy`. By default the semantic tier uses a SELF-
#     CONTAINED bag-of-words embedding + cosine retrieval (deterministic, no model
#     download, no DB), so the harness runs anywhere today. The LESSON — the recall
#     delta — is identical to running it through a real embedder + crunchstore.
#   * PRODUCTION PATH (documented at the bottom): swap the toy embedder for a real
#     SentenceTransformer and the dict store for your week-10 crunchstore adapter;
#     the retrieve_fn shape is unchanged. Same measurement, at scale.
#   * The "agent answer" here is a simple retrieval-then-format (no LLM needed to
#     make the point): the three-tier agent ANSWERS from what it retrieved; the
#     no-memory agent can only see the recent window, so it can't answer about turn 3.
#
# ACCEPTANCE CRITERIA
#
#   [ ] Prints recall rate for the THREE-TIER agent and the NO-MEMORY baseline.
#   [ ] The three-tier agent recalls the turn-3 fact at turn 38; the baseline does not
#       (the fact scrolled out of its recent window).
#   [ ] You can point at the delta and say "this is what the semantic memory tier
#       bought me, and here is the number."
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import re

import numpy as np

RECENT_WINDOW = 4          # the no-memory baseline sees only this many recent turns

# --- The benchmark: plant facts EARLY, ask LATE, distractors in between ---------
# (plant_turn, fact_text, fact_key, ask_turn, question, expected_answer)
PLAN = [
    (3,  "My project is called Helios.",          "project", 38, "what is my project called?",        "helios"),
    (7,  "I prefer to write code in Python.",      "lang",    36, "what language do I prefer?",         "python"),
    (12, "My company is on the enterprise tier.",  "tier",    34, "what tier is my company on?",        "enterprise"),
    (16, "My deadline is the 14th.",               "deadline",40, "when is my deadline?",               "14th"),
]
N_TURNS = 40


def build_turns():
    plants = {p[0]: p for p in PLAN}
    turns = []
    for i in range(1, N_TURNS + 1):
        if i in plants:
            turns.append({"turn": i, "content": plants[i][1], "fact": True})
        else:
            turns.append({"turn": i,
                          "content": f"Turn {i}: routine chatter about logistics.",
                          "fact": False})
    return turns


# --- Toy semantic memory: bag-of-words embedding + cosine retrieval -------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(text: str, vocab: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=np.float32)
    for tok in tokenize(text):
        if tok in vocab:
            v[vocab[tok]] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class SemanticMemory:
    """The semantic tier: store durable facts, retrieve by similarity. This is the
    crunchstore role from week 10, pointed at accumulated facts instead of a corpus."""
    def __init__(self, vocab):
        self.vocab = vocab
        self.facts: list[str] = []
        self.vecs: list[np.ndarray] = []

    def store(self, fact: str):
        self.facts.append(fact)
        self.vecs.append(embed(fact, self.vocab))

    def retrieve(self, query: str, k=2) -> list[str]:
        if not self.facts:
            return []
        qv = embed(query, self.vocab)
        sims = np.array([float(qv @ v) for v in self.vecs])
        order = np.argsort(-sims)[:k]
        return [self.facts[i] for i in order if sims[i] > 0]


def extract_durable_fact(turn: dict) -> str | None:
    """A tiny fact-extractor: turns flagged as facts get promoted to semantic memory.
    (A real system uses an LLM to decide what's durable; the flag stands in here.)"""
    return turn["content"] if turn["fact"] else None


def answer_contains(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


def run_agent(turns, plan, use_semantic: bool, vocab) -> dict:
    """Replay the conversation. At each ASK turn, the agent answers from what it can
    see: the no-memory agent sees only the recent window; the three-tier agent ALSO
    retrieves from semantic memory. Score recall."""
    semantic = SemanticMemory(vocab)
    asks = {p[3]: p for p in plan}
    results = []

    for i in range(1, N_TURNS + 1):
        turn = turns[i - 1]
        # Observe: promote durable facts to semantic memory (three-tier only).
        if use_semantic:
            fact = extract_durable_fact(turn)
            if fact:
                semantic.store(fact)

        if i in asks:
            _, _, _, _, question, expected = asks[i]
            recent = turns[max(0, i - RECENT_WINDOW):i]          # the recent window
            visible = " ".join(t["content"] for t in recent)
            if use_semantic:
                visible += " " + " ".join(semantic.retrieve(question))  # + semantic tier
            ok = answer_contains(visible, expected)
            results.append((i, question, expected, ok))

    recalled = sum(1 for r in results if r[3])
    return {"recalled": recalled, "asked": len(results),
            "rate": recalled / max(len(results), 1), "detail": results}


def main() -> int:
    turns = build_turns()
    # Build a shared vocabulary over all content + questions (toy embedder).
    corpus = [t["content"] for t in turns] + [p[4] for p in PLAN]
    vocab = {tok: idx for idx, tok in
             enumerate(sorted({t for text in corpus for t in tokenize(text)}))}

    three_tier = run_agent(turns, PLAN, use_semantic=True, vocab=vocab)
    baseline = run_agent(turns, PLAN, use_semantic=False, vocab=vocab)

    print("MEMORY REGRESSION TEST — plant facts early, ask late (the turn-38 test)\n")
    for label, res in (("THREE-TIER (episodic + semantic)", three_tier),
                       ("NO-MEMORY baseline (recent window only)", baseline)):
        print(f"=== {label} ===")
        for ask_turn, q, expected, ok in res["detail"]:
            plant_turn = next(p[0] for p in PLAN if p[3] == ask_turn)
            mark = "OK " if ok else "!! "
            print(f"  {mark}turn {ask_turn:>2}: {q!r} (planted turn {plant_turn}) "
                  f"-> expected {expected!r} {'RECALLED' if ok else 'FORGOTTEN'}")
        print(f"  recall rate: {res['recalled']}/{res['asked']} "
              f"= {res['rate']:.2f}\n")

    delta = three_tier["rate"] - baseline["rate"]
    print(f"DELTA: three-tier {three_tier['rate']:.2f} vs baseline "
          f"{baseline['rate']:.2f}  (+{delta:.2f})")
    if three_tier["recalled"] > baseline["recalled"]:
        print("PASS: the three-tier agent recalls early facts at late turns; the "
              "no-memory baseline FORGETS them as soon as they scroll out of the "
              "recent window. The recall delta is THE number that justifies the "
              "memory system — 'it feels like it remembers' is a vibe; "
              f"{three_tier['recalled']}/{three_tier['asked']} vs "
              f"{baseline['recalled']}/{baseline['asked']} is a measurement.")
    else:
        print("NOTE: the delta collapsed — check that durable facts are reaching the "
              "semantic tier and that the recent window is small enough that the "
              "baseline genuinely can't see the planted turn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (deterministic with the toy embedder)
# -----------------------------------------------------------------------------
#
# MEMORY REGRESSION TEST — plant facts early, ask late (the turn-38 test)
#
# === THREE-TIER (episodic + semantic) ===
#   OK turn 34: 'what tier is my company on?' (planted turn 12) -> expected 'enterprise' RECALLED
#   OK turn 36: 'what language do I prefer?' (planted turn 7) -> expected 'python' RECALLED
#   OK turn 38: 'what is my project called?' (planted turn 3) -> expected 'helios' RECALLED
#   OK turn 40: 'when is my deadline?' (planted turn 16) -> expected '14th' RECALLED
#   recall rate: 4/4 = 1.00
#
# === NO-MEMORY baseline (recent window only) ===
#   !! turn 34: 'what tier is my company on?' (planted turn 12) -> expected 'enterprise' FORGOTTEN
#   !! turn 36: 'what language do I prefer?' (planted turn 7) -> expected 'python' FORGOTTEN
#   !! turn 38: 'what is my project called?' (planted turn 3) -> expected 'helios' FORGOTTEN
#   !! turn 40: 'when is my deadline?' (planted turn 16) -> expected '14th' FORGOTTEN
#   recall rate: 0/4 = 0.00
#
# DELTA: three-tier 1.00 vs baseline 0.00  (+1.00)
# PASS: ... 4/4 vs 0/4 is a measurement.
#
# READ IT: the baseline sees only the last 4 turns, so by turn 34+ every planted fact
# (turns 3-16) has scrolled out — it recalls NOTHING. The three-tier agent promoted
# each durable fact to SEMANTIC memory at plant time, so at ask time it RETRIEVES the
# fact by similarity to the question and answers. THAT delta (1.00 vs 0.00) is what
# the semantic tier bought. Swap the toy embedder for a real SentenceTransformer +
# your week-10 crunchstore and the measurement is identical at scale:
#
#   from crunchstore.adapters import load
#   store = load("pgvector"); store.create("user_facts", dim=1024)
#   # store.upsert(...) each durable fact; store.search(query, k) at ask time
#
# Same harness, same recall delta, production store. The mini-project wires all
# three tiers (episodic summary + this semantic tier + a procedural log) under a
# context BUDGET and runs exactly this regression test.
# -----------------------------------------------------------------------------
