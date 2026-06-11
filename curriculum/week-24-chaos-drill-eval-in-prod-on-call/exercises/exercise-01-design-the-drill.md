# Exercise 1 — Design the Drill

**Estimated time:** ~45 minutes. Guided.

## Goal

Before you inject a single fault, design the drill. A chaos drill without a steady-state hypothesis is an outage; a drill without a tested revert is an outage you can't undo. This exercise produces the *plan* for one of the three required capstone drills — the document you execute (with the runner in Exercise 2) in the controlled window. The lesson is the lecture's anatomy: every drill has five parts, and skipping any one turns rehearsal into roulette.

You will write a one-page drill plan. It is graded on whether it is *runnable and safe*, not on length.

## Setup

You need your Sprint B capstone runnable (or at least its architecture in front of you) so the plan references real components — your LiteLLM config, your MCP tool defenses, your vector store and its backup. No code runs in this exercise; it's the design that makes the next one safe.

## Pick one drill

Choose one of the three required drills to plan in detail (the challenge runs all three; here you plan one well):

- **GPU node loss** — kill a vLLM replica; verify LiteLLM failover.
- **Prompt-injection on a tool** — inject a hostile instruction via a retrieved document; verify defense-in-depth.
- **Index corruption** — corrupt 5% of the vector store; measure the faithfulness regression; restore from backup.

## Write the five parts

Your plan (`notes/week-24/drill-plan.md`) must have all five:

### 1. The steady-state hypothesis

State the measurable "normal" you expect to hold — as a *number*, measured before the fault.

**Acceptance:** the hypothesis is a metric with a threshold (e.g. "error_rate = 0% and p95 < 2.5s", or "Ragas faithfulness >= 0.85 on the 100-question gold set"), not a feeling. You name *how* you'll measure it.

### 2. The blast radius

State the bounded scope of the fault.

**Acceptance:** the scope is bounded and named (one replica by name; one document by id; 5% of the index, a specific count). You explain why this bound contains the damage if the hypothesis is wrong.

### 3. The controlled window

State when the drill runs, who's watching, and what the SLO is during the window.

**Acceptance:** a defined window with you (or the team) watching, and a statement that the system is back to steady state outside it.

### 4. The tested revert

State exactly how you undo the fault — and how you've *tested* that the revert works *before* injecting.

**Acceptance:** the revert is concrete (`docker start vllm-2`; `corpus.delete(poisoned_id)`; `restore_from_backup(...)`) AND you state how you confirmed it works first. For the index drill, this means: take the backup, restore it onto a copy, confirm health — *then* plan the corruption. A revert you haven't tested is not a revert.

### 5. The measurement and the success criterion

State what you probe (continuously), and what outcome counts as "the system is production."

**Acceptance:** you name the probe (a real query through the supervisor; a faithfulness run on the gold set) and the success criterion (e.g. "zero user-visible errors during the fault; degraded p95 < +2s"; or "faithfulness drops measurably AND recovers within 60s of restore"). You also state what a *failing* outcome would teach you (the patch you'd write).

## The reflection

End the plan with one paragraph: **if this drill fails — if the failover doesn't fire, the defense is bypassed, or the eval doesn't detect the corruption — what have you learned, and is that a good or bad outcome?**

**Acceptance:** you recognize that a *failing* drill is the most valuable kind — it found a real gap on your schedule, with a revert ready, instead of in production at 3 AM. The drill's purpose is to find the gap, not to pass.

## Why this comes first

The single most common way a chaos drill goes wrong is injecting a fault you can't cleanly undo, or one whose effect you can't measure because you never recorded a baseline. The plan prevents both. Seniors don't improvise chaos; they design it. This plan is the artifact that makes Exercise 2's runner safe to point at your real system — and it's the spine of the postmortem you'll write in the challenge.

## What you've built

A runnable, safe drill plan for one of the three required drills, with a measurable steady-state hypothesis, a bounded blast radius, a tested revert, and a clear success criterion. Commit it. Exercise 2 builds the runner that executes a plan like this; the challenge runs all three.
