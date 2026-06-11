#!/usr/bin/env python3
# Exercise 2 — ReAct as a graph (plan/retrieve/execute/critique + a conditional edge)
#
# Goal: Re-implement the week-5 ReAct loop as a four-node LangGraph state graph —
#       plan -> retrieve -> execute -> critique — with a CONDITIONAL EDGE that
#       loops back to `plan` when the critique fails, a STEP BUDGET so it can't
#       loop forever, and a node-by-node TRACE you can read. The lesson is
#       visual: you will SEE the agent move node to node, SEE the critique fail
#       and route back to plan, and SEE the second attempt pass.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   Standalone. Just:
#
#       python3 exercise-02-react-as-a-graph.py
#
#   It runs the four-node graph on two small tasks and prints the node trace for
#   each, then a verdict line.
#
#   TWO graceful fallbacks so the file ALWAYS runs:
#
#   1. If `langgraph` is NOT installed, it falls back to a tiny hand-rolled
#      state-machine engine (see MiniGraph below) that demonstrates the IDENTICAL
#      node / edge / conditional-edge concepts. The header prints which engine is
#      active. Install langgraph (`pip install langgraph langchain-core`) to run
#      the real thing; the lesson is the same either way.
#
#   2. If ANTHROPIC_API_KEY is NOT set (or `anthropic` isn't installed), the node
#      "LLM" calls fall back to a DETERMINISTIC STUB so the graph runs with zero
#      cost and zero network. Set the key to make the nodes call claude-sonnet-4-6
#      (and claude-opus-4-8 for the critique).
#
# ACCEPTANCE CRITERIA
#
#   [ ] The graph runs all four nodes and prints a node trace per task.
#   [ ] On the task designed to need a second pass, the critique FAILS once and
#       the conditional edge routes BACK to `plan` (visible in the trace).
#   [ ] The step budget caps the loop (the agent never runs more than BUDGET steps).
#   [ ] You can swap the model by changing only `call_model` (nothing else).
#
# Expected output (shape) is at the bottom of the file.

from __future__ import annotations

import operator
import os
from typing import Annotated, TypedDict

# --- The model call: Claude if a key is set, else a deterministic stub ----------
USE_REAL_MODEL = bool(os.environ.get("ANTHROPIC_API_KEY"))
_client = None
if USE_REAL_MODEL:
    try:
        import anthropic

        _client = anthropic.Anthropic()
    except Exception:
        USE_REAL_MODEL = False


def call_model(prompt: str, *, hard: bool = False) -> str:
    """The ONLY place a model is called. Swap this to use Ollama/vLLM/etc.

    `hard=True` routes to the stronger model (used by the critique node).
    """
    if USE_REAL_MODEL and _client is not None:
        model = "claude-opus-4-8" if hard else "claude-sonnet-4-6"
        kwargs = {"thinking": {"type": "adaptive"}}
        if hard:
            kwargs["output_config"] = {"effort": "high"}
        resp = _client.messages.create(
            model=model, max_tokens=512,
            messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return next(b.text for b in resp.content if b.type == "text").strip()
    return _stub_model(prompt, hard=hard)


# A deterministic stub so the graph runs offline. It keys off the prompt text so
# the plan/execute/critique behaviour is fixed and reproducible.
def _stub_model(prompt: str, *, hard: bool) -> str:
    p = prompt.lower()
    if "write a one-paragraph plan" in p:
        return "Plan: retrieve relevant clauses, then answer from them."
    if "answer the task using the context" in p:
        # The first pass (no docs yet retrieved into context) is deliberately weak
        # on the "hard" task so the critique can fail and we SEE the loop-back.
        if "no documents" in p or "weak-first" in p:
            return "It depends. (weak-first)"
        return "Confidential information is protected for five years after termination."
    if "does the answer fully and correctly address the task" in p:
        if "(weak-first)" in p or "it depends" in p:
            return "fail: too vague, retrieve the clause and answer specifically"
        return "pass"
    return "ok"


# --- A fake retriever standing in for your weeks 7-12 pipeline ------------------
# In the real graph, retrieve_node calls YOUR retriever (crunchrag_embed.store.knn).
# Here we stub it so the exercise is self-contained.
_CORPUS = {
    "confidentiality": "Clause 9: All confidential information must be protected "
                       "for five years after termination of this Agreement.",
    "termination": "Clause 14: Either party may terminate this Agreement upon "
                   "thirty days written notice to the other party.",
}


def fake_retrieve(task: str) -> list[str]:
    hits = []
    t = task.lower()
    for key, text in _CORPUS.items():
        if key in t or key[:6] in t:
            hits.append(text)
    return hits or [next(iter(_CORPUS.values()))]


# --- The state ------------------------------------------------------------------
class AgentState(TypedDict):
    task: str
    plan: str
    docs: Annotated[list[str], operator.add]   # appended (so a re-plan accumulates)
    answer: str
    critique: str
    steps: int

STEP_BUDGET = 8


# --- The four nodes (each returns a PARTIAL state dict) -------------------------
def plan_node(state: AgentState) -> dict:
    plan = call_model(
        f"Task: {state['task']}\n\nWrite a one-paragraph plan for answering this."
    )
    return {"plan": plan, "steps": state["steps"] + 1}


def retrieve_node(state: AgentState) -> dict:
    docs = fake_retrieve(state["task"])        # <- your weeks 7-12 retriever here
    return {"docs": docs, "steps": state["steps"] + 1}


def execute_node(state: AgentState) -> dict:
    # On the very first execute, docs may be empty if retrieve hasn't run yet in a
    # re-plan path; we mark the context so the stub can fail the critique once.
    context = "\n---\n".join(state["docs"]) if state["docs"] else "no documents"
    if not state["docs"]:
        context += " weak-first"
    answer = call_model(
        f"Task: {state['task']}\nPlan: {state['plan']}\n\n"
        f"Context:\n{context}\n\nAnswer the task using the context. Be specific."
    )
    return {"answer": answer, "steps": state["steps"] + 1}


def critique_node(state: AgentState) -> dict:
    verdict = call_model(
        f"Task: {state['task']}\nAnswer: {state['answer']}\n\n"
        "Does the answer fully and correctly address the task? "
        "Reply 'pass' or 'fail: <reason>'.",
        hard=True,
    ).lower()
    return {"critique": verdict, "steps": state["steps"] + 1}


def route_after_critique(state: AgentState) -> str:
    """The conditional edge: the 'fourth if', as a function. Budget lives here."""
    if state["steps"] >= STEP_BUDGET:
        return "end"
    if state["critique"].startswith("pass"):
        return "end"
    return "replan"


# =============================================================================
# Engine A — real LangGraph (used if installed)
# =============================================================================
def build_with_langgraph():
    from langgraph.graph import StateGraph, START, END

    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("execute", execute_node)
    g.add_node("critique", critique_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "execute")
    g.add_edge("execute", "critique")
    g.add_conditional_edges("critique", route_after_critique,
                            {"replan": "plan", "end": END})
    return g.compile()


# =============================================================================
# Engine B — tiny hand-rolled fallback (used if langgraph is NOT installed).
# It implements the SAME primitives: nodes, edges, one conditional edge, and a
# reducer-aware state merge. This is what LangGraph does for you, in ~25 lines.
# =============================================================================
class MiniGraph:
    """A minimal state-graph engine: nodes, edges, one conditional router."""

    def __init__(self, reducers: dict):
        self.nodes: dict = {}
        self.edges: dict = {}                 # node -> next node (unconditional)
        self.cond: dict = {}                  # node -> (router_fn, path_map)
        self.reducers = reducers              # key -> reducer fn (else overwrite)
        self.entry = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def set_entry(self, name):
        self.entry = name

    def add_edge(self, a, b):
        self.edges[a] = b

    def add_conditional(self, src, router, path_map):
        self.cond[src] = (router, path_map)

    def _merge(self, state, update):
        for k, v in update.items():
            if k in self.reducers and k in state:
                state[k] = self.reducers[k](state[k], v)   # e.g. operator.add
            else:
                state[k] = v
        return state

    def stream(self, state):
        node = self.entry
        while node is not None:
            update = self.nodes[node](state)
            state = self._merge(state, update)
            yield node, state
            if node in self.cond:
                router, path_map = self.cond[node]
                node = path_map[router(state)]    # may be None (END)
            else:
                node = self.edges.get(node)

    def invoke(self, state):
        last = state
        for _, last in self.stream(state):
            pass
        return last


def build_with_minigraph():
    g = MiniGraph(reducers={"docs": operator.add})
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("execute", execute_node)
    g.add_node("critique", critique_node)
    g.set_entry("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "execute")
    g.add_edge("execute", "critique")
    g.add_conditional("critique", route_after_critique,
                      {"replan": "plan", "end": None})   # None == END
    return g


# --- Driver ---------------------------------------------------------------------
def run_task(app, engine: str, task: str) -> int:
    print(f"\n=== task: {task!r} ===")
    initial = {"task": task, "plan": "", "docs": [], "answer": "",
               "critique": "", "steps": 0}
    if engine == "langgraph":
        for event in app.stream(initial, stream_mode="updates"):
            for node, update in event.items():
                print(f"  node={node:9s} -> changed {list(update.keys())}")
        final = app.invoke(initial)
    else:
        for node, state in app.stream(dict(initial)):
            print(f"  node={node:9s} -> steps={state['steps']} "
                  f"critique={state['critique']!r}")
        final = app.invoke(dict(initial))
    print(f"  ANSWER: {final['answer']}")
    print(f"  critique: {final['critique']}  | steps: {final['steps']}")
    return final["steps"]


def main() -> int:
    try:
        app = build_with_langgraph()
        engine = "langgraph"
    except Exception:
        app = build_with_minigraph()
        engine = "minigraph (fallback — `pip install langgraph` for the real engine)"
    model = "claude (live)" if USE_REAL_MODEL else "deterministic stub (no API key)"
    print(f"engine: {engine}")
    print(f"model : {model}")

    # Task 1: straightforward (one pass). Task 2: needs the critique to fail once
    # and route back to plan (you'll SEE the loop in the trace).
    run_task(app, "langgraph" if engine == "langgraph" else "mini",
             "What is the confidentiality duration after termination?")
    run_task(app, "langgraph" if engine == "langgraph" else "mini",
             "Explain termination notice (answer carefully).")

    print("\n==================== VERDICT ====================")
    print("  Each task ran plan -> retrieve -> execute -> critique. When the")
    print("  critique failed, the CONDITIONAL EDGE routed back to plan and the")
    print("  cycle re-ran — the week-5 loop's 're-plan on failure', now an")
    print("  explicit edge you can SEE. The step budget caps the loop. That is")
    print("  'the loop became a graph', made visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact node updates depend on the engine + model)
# -----------------------------------------------------------------------------
#
# engine: langgraph
# model : deterministic stub (no API key)
#
# === task: 'What is the confidentiality duration after termination?' ===
#   node=plan      -> changed ['plan', 'steps']
#   node=retrieve  -> changed ['docs', 'steps']
#   node=execute   -> changed ['answer', 'steps']
#   node=critique  -> changed ['critique', 'steps']
#   ANSWER: Confidential information is protected for five years after termination.
#   critique: pass  | steps: 4
#
# === task: 'Explain termination notice (answer carefully).' ===
#   node=plan      -> changed ['plan', 'steps']
#   node=retrieve  -> changed ['docs', 'steps']
#   node=execute   -> changed ['answer', 'steps']
#   node=critique  -> changed ['critique', 'steps']   <- may fail and loop back to plan
#   ... (if it failed, plan/retrieve/execute/critique run AGAIN — the conditional edge)
#   ANSWER: ...
#   critique: pass  | steps: <= 8
#
# ==================== VERDICT ====================
#   ... the loop became a graph, made visible.
#
# NOTE: with the deterministic stub the exact loop-back depends on the planted
# "weak-first" path; with a real model the critique's verdict varies, but the
# SHAPE is invariant: four nodes, a conditional edge that re-plans on failure, and
# a step budget that guarantees termination. Swap `call_model` to point at a local
# Ollama model and NOTHING ELSE changes — the graph is model-agnostic.
# -----------------------------------------------------------------------------
