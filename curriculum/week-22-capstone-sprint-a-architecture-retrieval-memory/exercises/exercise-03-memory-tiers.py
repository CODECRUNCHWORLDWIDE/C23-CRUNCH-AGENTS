#!/usr/bin/env python3
# Exercise 3 — The three memory tiers and the turn-38 regression test
#
# Goal: Implement the three memory tiers -- episodic (rolling summary), semantic
#       (facts as vectors), procedural (action log) -- behind clean interfaces,
#       and pass the regression test: a load-bearing fact ("Project Halibut")
#       survives to turn 38. The lesson: the fact survives NOT because it's still
#       in the context window (the raw transcript is long gone) but because the
#       SEMANTIC tier stored it as a durable fact and read_semantic surfaced it.
#       The test proves the memory ARCHITECTURE, not the model's context length.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   Pure Python (no DB needed). The semantic tier uses an in-memory vector list;
#   the episodic tier uses a simple summary string; procedural is an append log.
#   The "model" is a deterministic stub so the regression test is reproducible.
#
#       python3 exercise-03-memory-tiers.py
#
#   Embeddings: BGE if available; hashing fallback otherwise.
#
# ACCEPTANCE CRITERIA
#
#   [ ] Three tiers behind interfaces: read_/write_ episodic, semantic, procedural.
#   [ ] The turn-38 regression test passes: a fact planted at turn 1 is recalled
#       at turn 38, AFTER the raw early turns are gone from active context.
#   [ ] You can show WHICH tier preserved the fact (semantic), and explain why
#       the episodic summary alone would have lost it.
#   [ ] The diagnosis path (extract -> store -> retrieve -> include) is visible.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# --- Embedder: BGE or hashing fallback ----------------------------------------
try:
    from sentence_transformers import SentenceTransformer

    _M = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def embed(text: str) -> np.ndarray:
        return _M.encode(text, normalize_embeddings=True)
    EMBEDDER = "BAAI/bge-large-en-v1.5"
except Exception:
    _DIM = 256

    def embed(text: str) -> np.ndarray:
        v = np.zeros(_DIM, dtype=np.float32)
        for w in text.lower().split():
            v[hash(w) % _DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v
    EMBEDDER = "hashing-fallback"


@dataclass
class Fact:
    text: str
    vec: np.ndarray


# --- Tier 1: episodic (bounded rolling summary) -------------------------------
class Episodic:
    def __init__(self, max_chars: int = 400):
        self.summary = ""
        self.max = max_chars

    def write(self, turn: str) -> None:
        # naive rolling summary: append, then truncate (loses old detail -- which
        # is WHY a load-bearing fact must NOT rely on this tier alone).
        self.summary = (self.summary + " " + turn).strip()
        if len(self.summary) > self.max:
            self.summary = self.summary[-self.max:]   # drop the oldest

    def read(self) -> str:
        return self.summary


# --- Tier 2: semantic (durable facts as vectors) ------------------------------
class Semantic:
    def __init__(self):
        self.facts: list[Fact] = []

    def write(self, fact_text: str) -> None:
        self.facts.append(Fact(fact_text, embed(fact_text)))

    def read(self, query: str, k: int = 3) -> list[str]:
        if not self.facts:
            return []
        q = embed(query)
        scored = sorted(self.facts, key=lambda f: float(np.dot(q, f.vec)), reverse=True)
        return [f.text for f in scored[:k]]


# --- Tier 3: procedural (action log) ------------------------------------------
class Procedural:
    def __init__(self):
        self.log: list[str] = []

    def write(self, action: str) -> None:
        self.log.append(action)

    def read(self, k: int = 10) -> list[str]:
        return self.log[-k:]


# --- The memory layer + a tiny stub agent -------------------------------------
@dataclass
class Memory:
    episodic: Episodic = field(default_factory=Episodic)
    semantic: Semantic = field(default_factory=Semantic)
    procedural: Procedural = field(default_factory=Procedural)


FACT_PATTERNS = [  # the EXTRACTION pass: pull durable facts from a user turn
    (re.compile(r"my project is (?:called |named )?(\w+)", re.I),
     "the user's project is named {}"),
    (re.compile(r"my deadline is (.+)", re.I), "the user's deadline is {}"),
    (re.compile(r"i prefer (\w+)", re.I), "the user prefers {}"),
]


class Agent:
    def __init__(self):
        self.mem = Memory()
        self.turn = 0

    def _extract_facts(self, user_msg: str) -> None:
        # The quiet failure point: a fact never extracted can never be recalled.
        for pat, template in FACT_PATTERNS:
            m = pat.search(user_msg)
            if m:
                self.mem.semantic.write(template.format(m.group(1)))

    def chat(self, user_msg: str) -> str:
        self.turn += 1
        self._extract_facts(user_msg)            # 1. EXTRACT to semantic
        self.mem.episodic.write(f"user: {user_msg}")   # also episodic (compressed)
        # assemble context: episodic summary + relevant semantic facts
        facts = self.mem.semantic.read(user_msg, k=3)   # 3. RETRIEVE
        self.mem.procedural.write(f"turn {self.turn}: answered")
        # the "answer": echo back the most relevant fact if the query asks for one
        if "project" in user_msg.lower() and "call" in user_msg.lower():
            for f in facts:
                if "project is named" in f:
                    return f.split("named ")[-1]   # 4. INCLUDE in the answer
        return f"(answer using {len(facts)} facts + summary)"


def main() -> int:
    print(f"embedder: {EMBEDDER}\n")
    agent = Agent()

    # Turn 1: plant the load-bearing fact.
    agent.chat("My project is called Halibut.")
    print("turn 1: planted 'My project is called Halibut.'")

    # Turns 2-37: filler that pushes the raw early turns out of episodic.
    for i in range(36):
        agent.chat(f"Tell me about topic number {i} in some detail please.")
    print("turns 2-37: filler (raw early turns now gone from episodic summary)")

    # Show the episodic summary has LOST the fact (it was compressed away):
    print(f"\nepisodic summary now: ...{agent.mem.episodic.read()[-80:]!r}")
    print(f"  'Halibut' in episodic summary? "
          f"{'Halibut' in agent.mem.episodic.read()}")

    # Turn 38: the recall test.
    answer = agent.chat("What's my project called?")
    print(f"\nturn 38 -- recall test: 'What's my project called?' -> {answer!r}")

    assert "Halibut" in answer, "REGRESSION FAILED: the fact did not survive"
    print("\nREGRESSION TEST PASSED: the fact survived to turn 38.")
    print("LESSON: 'Halibut' is gone from the episodic summary (compressed away)")
    print("but read_semantic() recalled it -- because the SEMANTIC tier stored it")
    print("as a durable fact when it was extracted at turn 1. The test proves the")
    print("memory ARCHITECTURE (the semantic tier), not the context length. If it")
    print("had relied on episodic alone, it would have failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape)
# -----------------------------------------------------------------------------
#
# embedder: BAAI/bge-large-en-v1.5
#
# turn 1: planted 'My project is called Halibut.'
# turns 2-37: filler (raw early turns now gone from episodic summary)
#
# episodic summary now: ...'topic number 35 in some detail please. user: Tell me about topic number 36...'
#   'Halibut' in episodic summary? False
#
# turn 38 -- recall test: 'What's my project called?' -> 'Halibut'
#
# REGRESSION TEST PASSED: the fact survived to turn 38.
# LESSON: 'Halibut' is gone from the episodic summary (compressed away) but
# read_semantic() recalled it ...
#
# NOTE: the key line is "'Halibut' in episodic summary? False" followed by the
# recall succeeding -- that's the proof the SEMANTIC tier did the work, not the
# context window. To see the diagnosis path (Lecture 2 §3b): comment out the
# self._extract_facts call and the test FAILS at the extraction stage; comment
# out the semantic read in chat() and it fails at the retrieve stage. Each
# failure points at a different tier/stage -- which is why the test is diagnostic.
# -----------------------------------------------------------------------------
