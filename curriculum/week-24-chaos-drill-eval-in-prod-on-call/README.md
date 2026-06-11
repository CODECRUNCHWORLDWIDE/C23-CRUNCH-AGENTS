# Week 24 — Chaos Drill, Eval-in-Prod, On-Call

This is the last week. You shipped the Production Agentic Research Assistant in Sprint B (week 23): a supervisor routing to four agents, an MCP tool surface, a vLLM cluster behind LiteLLM, a green eval gate, full tracing. It works. The demo query lands. The gate is green. And that is precisely the moment a senior AI engineer gets suspicious — because *working in the happy path is not the same as being production*. This week you find out which one you have, by breaking the system on purpose, in a controlled four-hour window, and writing down what happened.

By Sunday you will have run three chaos drills against your capstone, run eval-in-prod against shadow and live traffic, stood up the deploy mechanics that make a model change survivable (blue/green, canary by cohort), written an on-call runbook for an agentic system, and produced a postmortem in the standard incident format. The capstone is then complete: the system, the chaos-drill postmortem, the 5-minute video, and the career engineering pack.

The one sentence to internalize before you read another line — it's the lecture title, and it's the entire philosophy of the week:

> **You do not know if your system is production until you have lost a node, eaten an attack, and corrupted an index — on purpose, in a controlled window.**

Here's why that's not bravado. Every system fails. The only question is whether it fails *on your schedule, with you watching, with a rollback ready* — or at 3 AM, in front of a user, with you bisecting by print statement. Chaos engineering is the discipline of moving the failure from the second column to the first. You inject the GPU-node loss yourself, on a Tuesday afternoon, with the LiteLLM fallback config open and a stopwatch running, so that when it happens for real you already know the recovery time and you already know the fix held. The drill is not destruction for its own sake; it is *rehearsal*.

There's a corollary worth taping next to it — it's the on-call truth:

> **The runbook you write before the incident is the runbook you read during it. The one you write after is the postmortem.** A 3-AM page is not the time to figure out which dashboard shows retrieval precision.

## Learning objectives

By the end of this week, you will be able to:

- **Design and run** a chaos drill for an agentic system: define the steady-state hypothesis, inject one controlled fault, measure the user-visible impact and the recovery time, and revert the fault cleanly.
- **Execute** the three required capstone drills — **GPU node loss** (kill a vLLM replica; verify LiteLLM fails over to the remaining replicas and the vendor fallback), **prompt-injection on a tool** (inject a malicious instruction via a retrieved document; verify your week-17 / Sprint-B tool defenses hold or write the patch), and **retrieval index corruption** (corrupt 5% of the vector store; measure the Ragas-faithfulness regression; restore from backup; verify recovery time).
- **Run eval-in-prod**: replay production traces through a candidate prompt/model, run shadow traffic against a new version without serving it to users, and use Arize Phoenix to score live traffic — so you catch a regression before your users do.
- **Deploy safely**: blue/green model deploys (two stacks, instant switch, instant rollback) and canary-by-cohort (a new model serves 5% of users first), and explain when each is the right tool.
- **Write** an on-call runbook for an LLM-backed product: the alerts you respond to, the dashboards you read, the common incident classes (cost spike, latency spike, hallucination spike, attack), the escalation path, and the postmortem template.
- **Write** a postmortem in the standard incident format — timeline, what happened, what worked, what didn't, blameless analysis, action items — and explain why blameless is the only kind that improves the system.
- **Reason** about the difference between a system that *works* and a system that is *production*: graceful degradation, recovery time, defense-in-depth, and the humility that comes from having watched your own system fail on purpose.

## Prerequisites

This week assumes you have completed **C23 weeks 1–23**, or have equivalent fluency. Specifically:

- You shipped **Sprint B (week 23)**: the supervisor graph, the MCP tool surface (hardened — validation, path-traversal defense, rate limiting), the vLLM cluster behind LiteLLM (with the vendor fallback configured), the green eval gate, and OTel tracing to Langfuse and Phoenix. **This week attacks that exact system** — if Sprint B isn't runnable, you have nothing to break; fix it first.
- You have the **LiteLLM fallback** configured (week 19, week 23): a dead local tier degrades to the vendor, and a multi-replica vLLM survives one replica dying. The GPU-node-loss drill tests this.
- You have the **week-17 prompt-injection defenses** and the Sprint-B tool hardening live: input filtering, structured tool-argument validation, an output classifier. The prompt-injection drill tests these.
- You have the **Ragas eval suite** (week 12, week 23) and a **vector store backup** (week 10 taught the operational story; you need an actual backup to restore from). The index-corruption drill tests both.
- You can read an **OTel trace** in Langfuse and Phoenix (week 18, week 23). Every drill is diagnosed from the trace.

You need the capstone running (the Sprint B `docker compose up` stack) and a few hours of rented GPU for the vLLM tier so the node-loss drill is real. If you ran the local tier on Ollama, the node-loss drill still works — you kill the Ollama process and verify the fallback — it's just less dramatic than killing a GPU replica.

## Topics covered

- **Chaos engineering for agents:** the steady-state hypothesis, the blast radius, the controlled window, the revert; why you inject faults on purpose rather than wait for them; the difference between a fragile system and a graceful one.
- **The three drills:** GPU node loss and the LiteLLM failover; prompt-injection through a retrieved document and the defense-in-depth that should stop it; vector-index corruption, the Ragas-faithfulness regression it causes, and the backup-restore recovery.
- **Eval-in-prod:** replaying production traces through a candidate version; shadow traffic (run the new version, don't serve it, compare); online LLM-as-judge on live traffic via Phoenix; the gap between offline eval and production reality.
- **Safe deploys:** blue/green (two full stacks, atomic switch, atomic rollback); canary by cohort (5% of users on the new model first, watch the metrics, ramp or roll back); when each fits an agentic system.
- **On-call for LLM products:** the alert set (cost spike, latency spike, hallucination/faithfulness spike, attack-rate spike, error-rate spike); the dashboards (token usage by route, p95 latency by agent step, retrieval precision over time); the escalation path; the runbook as a living document.
- **The postmortem:** the standard incident format (timeline, impact, what worked, what didn't, action items); blameless analysis and why blame makes systems worse; the postmortem as the closing loop of the chaos drill.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                              | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-------------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Chaos engineering for agents; the steady-state hypothesis; drill 1 | 2h      |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Prompt-injection + index-corruption drills; defense-in-depth      |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Eval-in-prod, blue/green, canary; on-call runbooks; postmortems   |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | The chaos-drill window (run all three); capture the timeline       |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The postmortem; the runbook; the video walkthrough                 |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Capstone close-out — postmortem, video, career pack polish         |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, final submission                                     |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                                   | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | Chaos engineering, LiteLLM failover, OWASP LLM Top 10, Phoenix eval-in-prod, postmortem, and Anthropic SDK references |
| [lecture-notes/01-chaos-drills-for-agentic-systems.md](./lecture-notes/01-chaos-drills-for-agentic-systems.md) | Chaos engineering for agents, the steady-state hypothesis, and the three required drills in depth |
| [lecture-notes/02-eval-in-prod-deploys-and-on-call.md](./lecture-notes/02-eval-in-prod-deploys-and-on-call.md) | Eval-in-prod, blue/green and canary deploys, the on-call runbook, and the blameless postmortem |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-design-the-drill.md](./exercises/exercise-01-design-the-drill.md) | Write the steady-state hypothesis and the drill plan for one of the three required drills |
| [exercises/exercise-02-chaos-drill-runner.py](./exercises/exercise-02-chaos-drill-runner.py) | A drill runner that injects a fault, probes the system, measures impact + recovery, and emits a postmortem skeleton |
| [exercises/exercise-03-eval-on-traces.py](./exercises/exercise-03-eval-on-traces.py) | Replay production traces through a candidate version and gate the deploy on the replayed eval |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-run-the-chaos-drill.md](./challenges/challenge-01-run-the-chaos-drill.md) | Run all three drills in one 4-hour window and write the postmortem |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the chaos-drill postmortem and the on-call runbook |
| [mini-project/README.md](./mini-project/README.md) | The capstone close-out — postmortem, runbook, video, career pack; the final deliverable |

## The "it failed gracefully" promise

C23 uses a recurring marker, and this final week's version is the one that matters most: the drill where the system *survives* a fault you injected on purpose.

```
$ python -m drills.run gpu-node-loss --duration 300
[t+0s]   steady state: p95=2.1s  error_rate=0.0%  route=local-fast (3 vLLM replicas)
[t+12s]  FAULT INJECTED: killed vLLM replica vllm-2
[t+13s]  LiteLLM: replica vllm-2 unhealthy -> routing to vllm-0, vllm-1
[t+14s]  p95=2.4s  error_rate=0.0%  (degraded, not down)
[t+45s]  killed vllm-0, vllm-1 -> local tier fully down
[t+46s]  LiteLLM fallback: local-fast -> vendor-hard (claude-opus-4-8)
[t+48s]  p95=3.8s  error_rate=0.0%  (vendor fallback serving, slower but UP)
[t+300s] FAULT REVERTED: vLLM replicas restored, routing back to local
RECOVERY: 0 user-visible errors. Degraded p95 +1.7s during vendor fallback. ✓
```

If that drill instead prints `error_rate=100%` the moment you kill a replica, your system *works* but is not *production* — and you just learned that on a Tuesday with a revert ready, not at 3 AM in front of a user. That is the entire point of the week: to make the failure happen on your schedule, measure the recovery, and either confirm the defense held or write the patch.

## Stretch goals

If you finish the regular work early and want to push further:

- **Add a fourth drill: the vendor outage.** Block the Anthropic API at the network level and verify the system degrades to local-only (with a banner, or a "reduced quality" notice) rather than hard-failing the hard routes. Measure the faithfulness drop on the hard-route questions.
- **Wire an online LLM-as-judge on live traffic** in Phoenix: score a sampled 5% of production answers in real time and alert when the rolling faithfulness drops below threshold. This is eval-in-prod made continuous — the thing that catches a silent regression a static gate misses.
- **Run a canary by cohort for real**: deploy a new writing-agent prompt to 5% of (synthetic) users, watch the per-cohort faithfulness for an hour, and either ramp to 100% or roll back — and write the one-paragraph decision memo.
- **Game-day with a partner**: have a classmate inject a fault you don't know in advance (a corrupted config, a throttled vendor key, a poisoned document) and time how long it takes you to find it from the dashboards alone. That's the real on-call drill.

## Up next

There is no next week — this is the end of C23. What's next is the thing the whole course was for: you ship the capstone, you record the 5-minute video, you submit the postmortem, and you take the career engineering pack (interview-prep drills, the on-call runbook you wrote this week, the portfolio) into the sealed review. You finish the course able to do the thing the syllabus promised on day one: read an agent trace and tell a friend exactly why it failed on the third question — and now, also, why it *survived* the fourth. You have lost a node, eaten an attack, and corrupted an index, on purpose, in a controlled window. You know if your system is production. Most people never find out until 3 AM.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
