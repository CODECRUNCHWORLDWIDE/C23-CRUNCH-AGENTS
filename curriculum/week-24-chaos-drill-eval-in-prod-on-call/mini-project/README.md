# Mini-Project — Capstone Close-Out: Postmortem, Runbook, Video, Career Pack

> Finish C23. Take the Production Agentic Research Assistant you shipped in Sprint B, run the three chaos drills against it, and assemble the complete final capstone: the runnable system, the chaos-drill postmortem, the on-call runbook, the 5-minute video walkthrough, and the career engineering pack. This is the last thing you build in the course, and it's the thing the sealed-review panel grades.

This is the end. Twenty-three weeks built the system; this week proves it's production and packages it for the people who decide whether you graduate. The deliverable is not a new system — it's the *evidence* that the system you shipped survives the real world: a postmortem that documents three injected faults and how it held, a runbook that makes it operable by a human at 3 AM, a video that shows it working *and* surviving a failure, and a portfolio that frames it for a recruiter. The Sprint B mini-project asked "does it work?" This one asks "is it production, and can you prove it?"

**Estimated time:** ~12 hours, split across Thursday, Friday, and Saturday in the suggested schedule.

**Compounds forward:** there is nothing after this. This is the capstone the syllabus has been building toward since week 1. It gates the certificate. It's the centerpiece of your portfolio. The career engineering pack you assemble here — the runbook, the interview-prep drills, the portfolio — is what you carry into your first AI-engineering interview. Build it like it's the thing a hiring manager reads, because it is.

---

## What you will assemble

The final capstone has five parts. The *system* (part 1) you shipped in Sprint B; this week you produce the four artifacts that prove and package it.

1. **The runnable system** (from Sprint B) — the supervisor graph, the MCP tool surface, the two-tier serving, the green eval gate, the OTel tracing. `docker compose up` brings it up.
2. **The chaos-drill postmortem** (`POSTMORTEM.md`) — three drills run, measured timelines, what held, what you patched, blameless analysis, action items.
3. **The on-call runbook** (`production-runbook.md`) — alerts, dashboards, incident-class procedures with measured recovery times, escalation, postmortem template.
4. **The 5-minute video walkthrough** — one happy path, one tool call, one retrieval, one failure mode (a chaos drill).
5. **The career engineering pack** — the runbook (part 3), the interview-prep drills, and the portfolio entry.

By the end you have a repo a sealed-review panel can clone, `docker compose up`, read the postmortem and runbook, watch the video, and grade against the public rubric.

---

## Why the artifacts are the deliverable, not just the system

A working system with no postmortem, no runbook, and no video is a *prototype*. The artifacts are what make it *production* and what make it *gradeable*:

- **The postmortem** is the evidence you broke it on purpose and learned. Without it, "it's resilient" is a vibe. With it, "the failover recovers in seconds, the injection defense held at layer 2, the index restores in 45s" is a measured fact.
- **The runbook** is the evidence someone *other than you* can operate it. A system only you can run at 3 AM is not production; it's a bus-factor-of-one liability.
- **The video** is the evidence it actually works — narrated, showing one real failure surviving, not a cherry-picked demo.
- **The career pack** is the evidence you can *communicate* the work — to a recruiter, to an interviewer, to a teammate.

The rubric weights measurement and write-up heavily, and grades *vibes-only submissions as fails*. The artifacts are how you avoid that.

---

## Deliverable layout

```
capstone/
├── docker-compose.yml          # the Sprint B stack — one command, whole system
├── README.md                   # one command, env vars, architecture (Mermaid), video link
├── POSTMORTEM.md               # the chaos-drill postmortem (3 drills)
├── production-runbook.md       # the on-call runbook
├── portfolio.md                # the recruiter-facing portfolio entry
├── interview-prep/             # the system-design + technical + behavioral drills
│   ├── system-design.md        # design a RAG product / multi-agent system / LLM gateway / eval pipeline
│   ├── technical-drills.md     # write a chunker / ReAct loop / MCP server tool / routing layer
│   └── behavioral.md           # behavioral drills for AI-first companies
├── drills/
│   ├── run.py                  # the chaos-drill runner (Exercise 2, pointed at the real stack)
│   ├── gpu_node_loss.py        # drill 1
│   ├── prompt_injection.py     # drill 2
│   └── index_corruption.py     # drill 3
├── eval_in_prod/
│   ├── replay.py               # trace-replay gate (Exercise 3, real traces)
│   └── deploy.py               # blue/green or canary
└── capstone/                   # the Sprint B system (supervisor, MCP, serving, eval, telemetry)
```

---

## Deliverable 1 — run the drills, write the postmortem

Run all three drills (the challenge) in one window, capture the timelines with the runner, and write `POSTMORTEM.md`.

```python
# drills/run.py — drive all three drills, emit one postmortem
from drills import gpu_node_loss, prompt_injection, index_corruption
from drills.runner import run_drill, postmortem_section

def main():
    results = [
        run_drill("GPU node loss", **gpu_node_loss.plan()),
        run_drill("Prompt-injection on a tool", **prompt_injection.plan()),
        run_drill("Index corruption", **index_corruption.plan()),
    ]
    with open("POSTMORTEM.md", "w") as f:
        f.write("# Chaos-Drill Postmortem — Production Agentic Research Assistant\n\n")
        for r in results:
            f.write(postmortem_section(r))   # timeline, impact, what held/didn't
```

The non-negotiables:

- **Tested revert before every fault.** Especially the index restore — on a copy first.
- **One fault at a time, bounded blast radius.** One replica, one document, 5% of the index.
- **Blameless.** Every root cause is a *system* cause; the action items are owned and dated.
- **A failing drill is documented, not hidden.** If the injection got through, the postmortem records the bypass, the patch, and the before/after attack rate.

---

## Deliverable 2 — the on-call runbook

`production-runbook.md` turns the chaos-drill findings into 3-AM-ready procedures. The measured recovery times *are* the drill outputs.

```markdown
## Incident: GPU node loss
**Alert:** error-rate spike OR latency spike on the local tier.
**First dashboard:** p95 latency by agent step.
**Expected behavior:** LiteLLM fails over to surviving replicas, then to the
vendor (claude-opus-4-8). Verified in the chaos drill: 0 user-visible errors,
degraded p95 +1.7s during vendor fallback.
**If error-rate is non-zero:** the health-check/cooldown/fallback config is the
bug — restart the replica and confirm the fallback chain. Recovery: ~seconds.
**Escalate if:** the vendor is also down (check vendor status) — degrade to a
"reduced service" banner.
```

Write one such block per incident class (node loss, prompt-injection campaign, index corruption, cost spike, latency spike, faithfulness spike), each anchored to a *measured* recovery time from your drills.

The test of a good runbook is simple: hand it to someone who didn't build the system and ask them to walk through a node-loss incident. If they can follow it — open the right dashboard, recognize the expected behavior, take the first action, know when to escalate — the runbook works. If they get stuck on "which dashboard?" or "is this normal?", the runbook has a gap. Write it for that stranger, in plain narrative prose, not as a terse checklist only you can decode. The whole value of a runbook is that it works when *you* are not the one reading it.

---

## Deliverable 3 — the video and the career pack

The 5-minute video (homework Problem 5) and the career pack (the runbook + interview-prep + portfolio) round out the submission. The video's failure-mode segment shows the system *surviving* a chaos drill — that's the production proof. The portfolio's "two more weeks" section names a specific next step, the grown-up version of your Sprint B cut list.

The career pack is three files, each with a distinct audience:

- **`production-runbook.md`** (Deliverable 2) — for a teammate operating the system at 3 AM. The chaos-drill recovery times anchor it.
- **`interview-prep/`** — for *you*, preparing for AI-engineering interviews. Three files: `system-design.md` (the design drills — design a RAG product, a multi-agent system, an LLM gateway, an eval pipeline — using your capstone as the worked example); `technical-drills.md` (the from-scratch drills — write a chunker, a ReAct loop, an MCP server tool, a routing layer — all of which you've built this course); `behavioral.md` (the behavioral drills for AI-first companies). The point is that your capstone *is* the answer to most of these — you can walk an interviewer through a real multi-agent system you shipped, not a hypothetical.
- **`portfolio.md`** — for a recruiter scanning in thirty seconds. One image (the architecture diagram), two paragraphs, links to the repo and video, and the honest "if I had two more weeks" section. This is the front door to your work; make the two paragraphs count.

### Why the career pack is part of the capstone, not an afterthought

The syllabus gates the career pack on capstone delivery for a reason: the artifacts that prove the system is production (postmortem, runbook) are the *same* artifacts that make you hireable. An engineer who can say "I built a multi-agent RAG system, broke it on purpose, measured the failover recovery, and wrote the runbook" is demonstrating exactly the senior judgment a hiring manager screens for. The career pack isn't extra work bolted onto the capstone — it's the capstone *reframed for the people who decide what happens next*. Write it as the thing it is: the bridge from "I finished the course" to "I can do this job."

---

## Rules

- **You may** reuse everything from weeks 1–23 and the exercises: the Sprint B system, the drill runner, the eval-on-traces gate.
- **You must** run the three required drills against your *own* capstone, one fault at a time, with tested reverts.
- **You must not** hide a failing drill — a bypass found is a patch written and a before/after rate reported.
- **You must not** ship a postmortem that blames a person — every root cause is a system cause.
- **You must** anchor the runbook's recovery times to numbers you *measured* in the drills, not aspirations.
- The vendor fallback and any online judge use `claude-opus-4-8` with `thinking={"type":"adaptive"}` and `output_config={"effort":...}` — never `budget_tokens` or `temperature`.

---

## Acceptance criteria (the final capstone)

- [ ] `docker compose up` brings up the whole Sprint B system; a sample query runs end-to-end with a trace.
- [ ] All three chaos drills run against the live system, each with a measured timeline and recovery time.
- [ ] `POSTMORTEM.md` is blameless, in the standard format, covering all three drills (held / patched, recovery times, action items).
- [ ] Drill 1: 0 user-visible errors (failover proven or fixed). Drill 2: attack stopped or patched to a 0 success rate, with before/after rate. Drill 3: faithfulness drop detected and recovered from backup with a recorded time.
- [ ] `production-runbook.md` has the alert set, the three dashboards, per-incident procedures with measured recovery times, escalation, and a postmortem template.
- [ ] An eval-in-prod gate (trace replay) blocks a regressed candidate before it ships.
- [ ] A safe-deploy mechanism (blue/green or canary) demonstrates a deploy *and* a rollback.
- [ ] A ~5-minute narrated video shows one happy path, one tool call, one retrieval, and one failure mode (a chaos drill).
- [ ] The career pack is complete: `production-runbook.md`, `interview-prep/`, `portfolio.md`.
- [ ] Committed and pushed; the README links the video and documents the one command.

---

## Grading rubric (100 points — the final-capstone rubric)

| Axis | Points | What "meets" looks like |
|---|---:|---|
| **Correctness** | 20 | The system does what the spec says: supervisor routes, tools fire, retrieval grounds, the eval gate is green. Runs from `docker compose up`. |
| **Resilience (chaos drills)** | 25 | Three drills run against the live system; failover holds (or is fixed); injection defense holds (or is patched); corruption is detected and recovered. Recovery times measured. |
| **Measurement** | 20 | Ragas + calibrated judge on the 100-q set; eval-in-prod gate; the postmortem's numbers are real. Vibes do not count. |
| **Observability & operability** | 15 | Full OTel tracing; the runbook is operable by a stranger; alerts map to dashboards; recovery times anchored to the drills. |
| **Write-up & communication** | 15 | Blameless postmortem; clear runbook; recruiter-ready portfolio; the video shows a real failure surviving. |
| **Hygiene & polish** | 5 | No secrets committed; the architecture diagram is in sync; clean commits; the README's one command works on a clean checkout. |

**90+** is sealed-review-pass, portfolio-grade, interview-ready. **70–89** meets the bar with a soft spot (a drill not fully run, a runbook missing a measured number, a vibes-y postmortem). **Below 70** means the capstone is a prototype, not a production system — the artifacts that prove resilience and operability are missing.

---

## A note on submission order

Build the artifacts in dependency order, not in reading order:

1. **Confirm the Sprint B system runs** (`docker compose up` on a clean checkout) — everything else attacks or describes it.
2. **Run the three drills** and capture the timelines — the recovery numbers feed both the postmortem and the runbook.
3. **Write the postmortem** from the captured timelines.
4. **Write the runbook**, anchoring its recovery times to the postmortem's measured numbers.
5. **Record the video**, using a drill as the failure-mode segment.
6. **Write the portfolio and interview-prep**, framing the finished system for the reader.

Doing it out of order — writing the runbook before the drills, say — produces a runbook full of guessed recovery times you then have to go back and fix. The drills come first because they produce the *measurements* every other artifact cites.

## Stretch goals

- **A fourth drill — vendor outage.** Block the Anthropic API at the network level; verify the system degrades to local-only with a notice rather than hard-failing the hard routes. Measure the faithfulness drop and add it to the postmortem.
- **Online judge alerting.** Score a sampled 5% of live answers in real time in Phoenix and alert when rolling faithfulness drops — eval-in-prod made continuous. Add the alert to the runbook.
- **A game-day with a partner.** Have a classmate inject a fault you don't know in advance; time how long it takes you to find it from the dashboards alone. Write up the game-day as a postmortem.
- **A canary-by-cohort decision memo.** Deploy a new writing-agent prompt to 5% of synthetic users, watch per-cohort faithfulness for an hour, and write the one-paragraph ramp-or-rollback decision.

---

## How this closes C23

- **Sprint A (week 22)** built the retrieval and memory.
- **Sprint B (week 23)** wired the agents, the MCP surface, the serving, the eval, and the tracing into a runnable system.
- **This week** broke it on purpose, measured how it fails and recovers, built the eval-in-prod and deploy mechanics that keep it healthy, and packaged the whole thing — system, postmortem, runbook, video, career pack — for the sealed review.

You set out, in week 1, to be able to read an agent trace and tell a friend exactly why it failed on the third question. You can now do that — and you can tell them why it *survived* the fourth: because you lost a node, ate an attack, and corrupted an index, on purpose, in a controlled window, and watched the defenses hold. You know whether your system is production. Submit it, record the video, take the last quiz, and graduate.

That's C23. Well done.

---

*If you find errors in this material, please open an issue or send a PR.*
