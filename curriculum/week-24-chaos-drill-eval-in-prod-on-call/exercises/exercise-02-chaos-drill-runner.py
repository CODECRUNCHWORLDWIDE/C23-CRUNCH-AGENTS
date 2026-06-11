#!/usr/bin/env python3
# Exercise 2 — The chaos-drill runner
#
# Goal: Build the runner that executes a chaos drill safely: record the steady-
#       state baseline, inject ONE fault, probe the system every second, measure
#       the user-visible impact and the recovery time, REVERT cleanly (always,
#       even on error), and emit a postmortem skeleton you fill in.
#
# The lesson is the lecture's anatomy: every drill has five parts, and the revert
# runs in a `finally`. A drill that can't cleanly undo its fault is an outage you
# caused. This runner enforces the discipline.
#
# Estimated time: 55 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   This file ships a SIMULATED capstone (a fake supervisor with a fake vLLM
#   cluster + LiteLLM fallback) so you can run a full GPU-node-loss drill with NO
#   GPU, NO cluster, NO network. You see the SHAPE of a drill end to end. In the
#   mini-project you swap inject()/revert()/probe() for calls against your real
#   Sprint B `docker compose` stack.
#
#       python3 exercise-02-chaos-drill-runner.py
#
#   It runs a node-loss drill: kills simulated replicas one at a time, shows the
#   LiteLLM fallback firing (local -> vendor), measures recovery, reverts, and
#   prints a postmortem skeleton with the measured timeline filled in.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The runner records a steady-state baseline BEFORE injecting.
#   [ ] It injects ONE fault, probes every second, and records a timeline.
#   [ ] The revert runs in a `finally` (even if a probe raises) — the system is
#       ALWAYS returned to steady state at exit.
#   [ ] It computes recovery time and user-visible error count.
#   [ ] It emits a POSTMORTEM.md skeleton with the timeline filled in.
#   [ ] The node-loss drill shows graceful degradation: failover keeps
#       error_rate at 0% while latency degrades — not error_rate=100%.
#
# Expected output (shape) is at the bottom of the file.

from __future__ import annotations

import time
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# A SIMULATED capstone: a fake vLLM cluster behind a fake LiteLLM router with a
# vendor fallback. Killing all local replicas fails over to the vendor (slower
# but UP) — exactly the resilience the real drill tests. Replace with your stack.
# --------------------------------------------------------------------------- #
@dataclass
class FakeStack:
    local_replicas: set[str] = field(default_factory=lambda: {"vllm-0", "vllm-1", "vllm-2"})
    vendor_up: bool = True

    def kill(self, replica: str) -> None:
        self.local_replicas.discard(replica)

    def restore(self, replicas: set[str]) -> None:
        self.local_replicas |= replicas

    def query(self) -> dict:
        """Route a query the way LiteLLM would: local if any replica is up,
        else vendor fallback. Raise only if EVERYTHING is down."""
        if self.local_replicas:
            return {"served_by": "local-fast", "latency": 2.1, "error": False}
        if self.vendor_up:
            return {"served_by": "vendor-hard", "latency": 3.8, "error": False}
        raise RuntimeError("all tiers down — no fallback available")


STACK = FakeStack()


# --------------------------------------------------------------------------- #
# The drill runner. This is the part you keep and point at your real system.
# --------------------------------------------------------------------------- #
def _elapsed(t0: float) -> str:
    return f"t+{int(time.monotonic() - t0)}s"


def run_drill(name, inject, revert, probe, steps):
    """Run one chaos drill. Returns the timeline and a postmortem skeleton.

    inject(): cause the ONE fault (may be called multiple times for staged faults).
    revert(): undo it (runs in a finally — ALWAYS).
    probe():  return a dict measuring the system NOW.
    steps:    list of (delay_s, inject_callable_or_None) staging the fault.
    """
    timeline = []
    baseline = probe()                                  # 1. steady-state hypothesis
    timeline.append(("t+0", "steady state", baseline))
    errors = 0
    t0 = time.monotonic()
    try:
        for delay, stage in steps:
            time.sleep(delay)
            if stage is not None:
                stage()                                  # 4. inject (staged)
                timeline.append((_elapsed(t0), "FAULT INJECTED", None))
            m = probe()                                  # measure after each stage
            timeline.append((_elapsed(t0), "probe", m))
            if m.get("error"):
                errors += 1
    finally:
        revert()                                         # 5. revert — ALWAYS
        timeline.append((_elapsed(t0), "FAULT REVERTED", probe()))

    # recovery: the system returned to a non-error state at revert.
    recovered = not timeline[-1][2].get("error", False)
    return {
        "name": name,
        "baseline": baseline,
        "timeline": timeline,
        "user_visible_errors": errors,
        "recovered": recovered,
    }


def postmortem_skeleton(result) -> str:
    """Emit a POSTMORTEM.md skeleton with the measured timeline filled in."""
    lines = [f"# Postmortem — {result['name']}", ""]
    lines.append("## Summary")
    lines.append("_One paragraph: what happened, impact, duration._")
    lines.append("")
    lines.append("## Timeline (measured)")
    for ts, label, m in result["timeline"]:
        served = f" served_by={m['served_by']}" if m and "served_by" in m else ""
        err = f" error={m['error']}" if m and "error" in m else ""
        lines.append(f"- `{ts}` {label}{served}{err}")
    lines.append("")
    lines.append("## Impact")
    lines.append(f"- user-visible errors during fault: {result['user_visible_errors']}")
    lines.append(f"- recovered to steady state: {result['recovered']}")
    lines.append("")
    lines.append("## What worked\n_The fallback/defense/backup that held._\n")
    lines.append("## What didn't\n_The gap the fault exposed (and the patch)._\n")
    lines.append("## Root cause (blameless)\n_The SYSTEM cause, not a person._\n")
    lines.append("## Action items\n- [ ] _owned, dated follow-up_\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The GPU-node-loss drill, staged: kill one replica, then the rest, watch the
# fallback fire. Replace inject/revert/probe with calls to your real stack.
# --------------------------------------------------------------------------- #
def node_loss_drill():
    killed = set()

    def kill_one():
        # TODO 1: kill ONE replica from STACK.local_replicas, record it in
        #   `killed`, and (after the first stage) kill the rest so the fallback
        #   fires. Hint: pop a replica name and call STACK.kill(name).
        raise NotImplementedError("implement kill_one (TODO 1)")

    def revert():
        STACK.restore(killed)        # restore everything we killed

    def probe():
        return STACK.query()         # raises only if EVERYTHING is down

    # Stage 1: kill one replica (degrade). Stage 2: kill the rest (force fallback).
    steps = [
        (1, kill_one),               # kill one -> still local, degraded
        (1, kill_one),               # kill another
        (1, kill_one),               # kill the last -> fallback to vendor
        (1, None),                   # observe steady-degraded state
    ]
    return run_drill("GPU node loss", inject=kill_one, revert=revert,
                     probe=probe, steps=steps)


if __name__ == "__main__":
    result = node_loss_drill()
    print(f"=== Drill: {result['name']} ===")
    for ts, label, m in result["timeline"]:
        served = f"  served_by={m['served_by']}" if m and "served_by" in m else ""
        print(f"  {ts:<7} {label}{served}")
    print(f"\nuser-visible errors: {result['user_visible_errors']}")
    print(f"recovered: {result['recovered']}")
    print()
    print(postmortem_skeleton(result))

# --------------------------------------------------------------------------- #
# EXPECTED OUTPUT (shape)
# --------------------------------------------------------------------------- #
#
# === Drill: GPU node loss ===
#   t+0     steady state  served_by=local-fast
#   t+1s    FAULT INJECTED
#   t+1s    probe  served_by=local-fast       (degraded, still local)
#   t+2s    FAULT INJECTED
#   t+2s    probe  served_by=local-fast
#   t+3s    FAULT INJECTED
#   t+3s    probe  served_by=vendor-hard       (fallback fired — slower but UP)
#   t+4s    probe  served_by=vendor-hard
#   t+4s    FAULT REVERTED  served_by=local-fast
#
# user-visible errors: 0
# recovered: True
#
# # Postmortem — GPU node loss
# ## Summary
# ...
# ## Timeline (measured)
# - `t+0` steady state served_by=local-fast
# ... (full timeline) ...
# ## What worked
# ## What didn't
# ## Root cause (blameless)
# ## Action items
#
# The KEY result: user-visible errors == 0. The fallback kept the system UP
# through total local-tier loss, degrading latency (local 2.1s -> vendor 3.8s)
# but serving every query. That is graceful degradation. If your real drill
# shows errors > 0, the LiteLLM health-check/cooldown/fallback config is the bug.
# --------------------------------------------------------------------------- #
