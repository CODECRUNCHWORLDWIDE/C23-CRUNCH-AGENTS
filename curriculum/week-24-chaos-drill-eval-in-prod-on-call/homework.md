# Week 24 Homework

Six problems that produce the capstone's final deliverables — the chaos-drill postmortem, the on-call runbook, the eval-in-prod gate, and the deploy mechanics — and close out C23. The full set should take about **5 hours**. Work in your Sprint B capstone repo so every problem feeds the final submission.

The headline deliverables are **Problem 1 — the chaos-drill postmortem** and **Problem 4 — the on-call runbook**; both are required parts of the capstone and the career engineering pack.

Have your **Sprint B capstone** runnable (`docker compose up`), a **tested vector-store backup**, and `ANTHROPIC_API_KEY` set. If Sprint B is broken, fix it first — there's nothing to attack otherwise.

Each problem includes a **problem statement**, **acceptance criteria**, a **hint**, and an **estimated time**.

---

## Problem 1 — The chaos-drill postmortem (headline deliverable)

**Problem statement.** Run the three required drills (GPU node loss, prompt-injection on a tool, index corruption) against your capstone in one controlled window, and write `POSTMORTEM.md` in the standard incident format covering all three.

**Acceptance criteria.**

- All three drills run; each has a measured timeline (baseline → fault → impact → revert → recovery).
- Drill 1: 0 user-visible errors (or the failover config fixed and re-run to 0).
- Drill 2: the attack is stopped or patched; the before/after attack rate and the layer that held are reported.
- Drill 3: faithfulness drops measurably and recovers after a backup restore; the recovery time is recorded.
- `POSTMORTEM.md` is blameless and hits every section: summary, timeline per drill, impact, what worked, what didn't, root cause, action items.
- Committed.

**Hint.** Use the Exercise 2 runner pointed at your real stack to capture the timelines, and the Exercise 1 plan structure for each drill. Test every revert before its fault — especially the index restore (on a copy first). A drill that *fails* is a successful drill; document the gap and the patch.

**Estimated time.** 1 hour 15 minutes (plus the drill window).

---

## Problem 2 — The eval-in-prod gate

**Problem statement.** Wire a trace-replay eval-in-prod gate (Exercise 3's logic, against real traces from Langfuse/Phoenix) into your deploy pipeline so a candidate change that regresses on the real query distribution is blocked before it ships.

**Acceptance criteria.**

- A gate replays a dataset of real production traces through a candidate writing-agent and scores faithfulness with your calibrated judge.
- A candidate that regresses below the incumbent baseline (beyond tolerance) is blocked; the regressed traces are named.
- `notes/week-24/eval-in-prod.md` documents the gate, the baseline, and one run where a candidate was blocked.
- Committed.

**Hint.** Reuse Exercise 3's gate logic; swap the simulated traces for a real dataset captured by your OTel tracing and the mock faithfulness for your `claude-opus-4-8` calibrated judge. The point is that a fluent-but-confabulating candidate fails the gate even though it'd pass a casual demo.

**Estimated time.** 50 minutes.

---

## Problem 3 — A safe deploy: blue/green or canary

**Problem statement.** Implement one safe-deploy mechanism for a model/prompt change — blue/green (two stacks, atomic switch + rollback) or canary-by-cohort (5% of users first, ramp by metrics) — and demonstrate a deploy *and a rollback*.

**Acceptance criteria.**

- A new writing-agent prompt (or model) can be deployed and rolled back via the chosen mechanism.
- You demonstrate both the deploy *and* the rollback (a config switch for blue/green; a cohort ramp/roll-back for canary).
- `notes/week-24/deploy.md` records which mechanism you chose, why, and the deploy + rollback you ran.
- Committed.

**Hint.** Blue/green is the simpler first one: two stacks behind your LiteLLM router (or load balancer), `active_stack` flips between them. The rollback is the demonstration — show the flip-back is instant. For canary, deterministic per-user assignment (`hash(user_id) % 100 < 5`) keeps a user's experience consistent.

**Estimated time.** 50 minutes.

---

## Problem 4 — The on-call runbook (headline deliverable)

**Problem statement.** Write `production-runbook.md` — the career-pack on-call runbook for your agentic system — against the syllabus spec: the alerts you respond to, the dashboards you read, the common incident classes, the escalation path, and the postmortem template.

**Acceptance criteria.**

- An **alert set**: cost spike, latency spike, faithfulness/hallucination spike, attack-rate spike, error-rate spike — each with a fire-condition and a first action.
- The **three dashboards** (token usage by route, p95 latency by agent step, faithfulness over time), each mapped to the alert that opens it first.
- **Incident-class procedures** distilled from your chaos drills: GPU node loss, prompt-injection campaign, index corruption, cost spike — each with the *measured recovery time* from your drills.
- An **escalation path** and a **postmortem template**.
- The runbook is narrative and operable by someone who didn't build the system.
- Committed.

**Hint.** This is where the chaos drills pay off: the "index corruption → restore from backup, recovery ~45s" line in the runbook is the number you *measured* in Problem 1. The runbook is the bridge between the trace (you can see the problem) and the fix (you know what to do). Write it for a stranger at 3 AM.

**Estimated time.** 1 hour.

---

## Problem 5 — The 5-minute video walkthrough

**Problem statement.** Record the 5-minute video walkthrough the capstone requires: narrate one happy path, one tool call, one retrieval, and one failure mode (use one of your chaos drills as the failure mode).

**Acceptance criteria.**

- A ~5-minute video, narrated by you, demonstrating: one happy-path query end-to-end (with the trace), one MCP tool call, one retrieval, and one failure mode (a chaos drill — e.g. the node-loss failover or the prompt-injection defense holding).
- The video link is in the repo README.
- Committed.

**Hint.** The failure-mode segment is what separates a demo reel from a capstone walkthrough. Show the system *surviving* a fault you injected — the node-loss failover keeping it up, or the injection defense holding — and narrate *why* it held. That's the moment that proves it's production.

**Estimated time.** 45 minutes.

---

## Problem 6 — The portfolio entry

**Problem statement.** Write the capstone's `portfolio.md` entry per the syllabus career-pack spec: one image (the architecture diagram), two paragraphs, links to the repo and the video, and a "if I had two more weeks" section.

**Acceptance criteria.**

- `portfolio.md` has the architecture image, two paragraphs framing the system for a recruiter, repo + video links, and an honest "two more weeks" section.
- The "two more weeks" section names a *specific* next step (e.g. "an online judge alerting on live faithfulness", "a fourth drill for vendor outage"), not a vague aspiration.
- Committed.

**Hint.** The "two more weeks" section is, in spirit, your Sprint B cut list grown up — it shows you know exactly what the system *doesn't* do yet and what you'd build next. That self-awareness reads as senior.

**Estimated time.** 40 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Chaos-drill postmortem (headline) | 1 h 15 min |
| 2 — Eval-in-prod gate | 50 min |
| 3 — Safe deploy (blue/green or canary) | 50 min |
| 4 — On-call runbook (headline) | 1 h 0 min |
| 5 — 5-minute video walkthrough | 45 min |
| 6 — Portfolio entry | 40 min |
| **Total** | **~5 h 20 min** |

When you've finished all six, you have the complete final capstone: the system (Sprint B), the postmortem, the runbook, the video, and the portfolio entry. Assemble them in the [mini-project](./mini-project/README.md), submit for the sealed review, and take the [quiz](./quiz.md) — the last one. You're done. Well done.
