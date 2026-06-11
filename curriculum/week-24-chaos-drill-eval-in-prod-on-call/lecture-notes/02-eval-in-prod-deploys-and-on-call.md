# Lecture 2 — Eval-in-Prod, Safe Deploys, and On-Call

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can run eval-in-prod (replay production traces through a candidate version, run shadow traffic, score live traffic with an online LLM-as-judge in Phoenix), deploy a model change safely (blue/green for instant rollback, canary-by-cohort for gradual exposure), write an on-call runbook for an LLM-backed product (alerts, dashboards, incident classes, escalation), and write a blameless postmortem in the standard incident format.

Lecture 1 broke the system on purpose, once, in a window. This lecture turns those one-off drills into ongoing practice — the things that keep a production agentic system healthy *between* chaos drills. A chaos drill is an *event*; the practices in this lecture are *continuous*. Together they answer the two halves of "is it production?": the drills prove it survives a fault *today*, and the eval-in-prod, deploys, and on-call practices keep it surviving as the system, the corpus, and the traffic change *over time*. A system that passed its drills last month but has no continuous eval and no runbook is production-*ish* — it was good once and nobody's watching now. The mantra for the week:

> **The runbook you write before the incident is the runbook you read during it. The one you write after is the postmortem.**

And the one this lecture exists to enforce:

> **Offline eval tells you the system was good on the gold set last week. Eval-in-prod tells you it's good on real traffic right now.** Those are different claims, and only the second one keeps you out of trouble.

---

## 1. Eval-in-prod — why the gold set isn't enough

Your Sprint B eval gate scored the system on a 100-question gold set. That gate is necessary — it stops you shipping an obviously broken build. But it is not sufficient, for two reasons:

1. **The gold set is static; production is not.** Real users ask questions your gold set never anticipated. The corpus drifts as documents are added. A model update changes behavior. A system that's green on the gold set can be quietly failing on the long tail of real queries — and the gold set will never tell you, because it doesn't contain them.
2. **The gold set is offline; regressions happen online.** A new writing-agent prompt that scores 0.91 on the gold set might score 0.78 on the actual distribution of production queries. You find out from production, or you find out from a user complaint. Eval-in-prod is how you find out from production *first*.

Eval-in-prod has three modes, in increasing order of "closeness to live."

### 1.1 Trace replay (offline-on-prod-data)

You have a dataset of *real production traces* (captured by your OTel tracing to Langfuse/Phoenix). To test a candidate change — a new prompt, a new model — you *replay* those traces through the candidate and score the results. This catches regressions on the real query distribution without touching live traffic.

```python
# Replay yesterday's production queries through a candidate writing-agent prompt
def replay_eval(traces, candidate_prompt):
    rows = []
    for tr in traces:                       # real production queries + their contexts
        answer = writing_agent(tr.query, tr.contexts, prompt=candidate_prompt)
        rows.append({"question": tr.query, "answer": answer,
                     "contexts": tr.contexts, "ground_truth": tr.shipped_answer})
    return ragas_faithfulness(rows)          # score the candidate on REAL queries


baseline = replay_eval(yesterdays_traces, current_prompt)     # 0.89
candidate = replay_eval(yesterdays_traces, candidate_prompt)  # 0.84 -> REGRESSION
# Gate the deploy: candidate is worse on real traffic. Don't ship.
```

This is the offline half of eval-in-prod and the safest first step: you catch the regression on a desk, on real data, before any user sees the candidate.

### 1.2 Shadow traffic

A step closer: run the candidate version against *live* requests, *but don't serve its output to users*. The user gets the current version's answer; the candidate's answer is computed in parallel, scored, and logged. After enough shadow traffic, you compare the candidate's online quality to the incumbent's — on truly live data — with zero user risk.

```
       user request
            │
     ┌──────┴───────┐
     ▼              ▼
  current      candidate (shadow)
  version       version
     │              │
  served to     scored + logged
   user          (NOT served)
```

Shadow traffic is how you validate a model upgrade or a prompt change against production reality before betting any user on it. The cost is doubled inference on the shadowed fraction; the payoff is catching a regression the gold set and even trace-replay missed.

The implementation is a fork: when a request comes in, you serve the user with the current version *and*, asynchronously, run the candidate, log its answer, and score it later. The user never waits on the shadow.

```python
async def handle_request(query, contexts):
    current = await current_version(query, contexts)   # served to the user NOW
    # Fire-and-forget the shadow; the user does not wait on it.
    asyncio.create_task(shadow(query, contexts, current))
    return current


async def shadow(query, contexts, current_answer):
    candidate = await candidate_version(query, contexts)
    score_current = await judge(query, current_answer, contexts)
    score_candidate = await judge(query, candidate, contexts)
    phoenix.log_shadow(query, score_current, score_candidate)   # compare offline
```

Three things to get right in a shadow setup:

- **The shadow must not affect the user.** It runs async, its output is never served, and a shadow failure (a crash in the candidate) must not break the real response. Wrap it so a candidate exception is logged, not propagated.
- **Sample, don't shadow everything.** Doubling inference on 100% of traffic is expensive; shadow a representative 10–20% for a bounded window (a day, a week), gather enough scored pairs for a confident comparison, then stop.
- **Compare paired, not aggregate.** The strongest signal is the candidate's score *minus* the incumbent's score *on the same query* — paired differences cancel out the per-query difficulty and surface the true quality delta. An aggregate "candidate averaged 0.86, incumbent 0.88" is weaker than "the candidate was worse on 340 of 2,000 paired queries, concentrated in multi-hop questions."

Shadow traffic is the gold standard of pre-deploy validation because it's the only one that runs the candidate on *real, live, concurrent* requests without any user risk. It's more expensive than trace replay and more involved to wire, which is why you reach for it on the high-stakes changes — a model swap, a major prompt rewrite — rather than every tweak.

### 1.3 Online LLM-as-judge

The continuous version: sample a fraction of *live, served* answers (say 5%) and score them in real time with a calibrated LLM-as-judge (`claude-opus-4-8`, the same judge from Sprint B), exported to Phoenix. A rolling faithfulness metric over live traffic. When it drops below threshold, you get paged — *before* the complaints roll in.

```python
def online_judge_sample(answer_event, sample_rate=0.05):
    if random.random() > sample_rate:
        return
    score = calibrated_judge(answer_event.question,
                             answer_event.answer,
                             answer_event.contexts)   # claude-opus-4-8 judge
    phoenix.log_eval("online_faithfulness", score, trace_id=answer_event.trace_id)
    # Phoenix's rolling alert fires if the windowed mean drops below 0.80.
```

This is eval-in-prod made continuous — the thing that catches a *silent* regression a static gate never will, because the gate ran last week and this regression started an hour ago. The week-18 dashboards (faithfulness over time) become *alerting* surfaces, not just observability.

A fair objection: eval-in-prod needs *production traffic*, and a brand-new capstone has none. How do you run trace replay or an online judge with zero real users? Three bootstrapping moves:

- **Synthetic traffic from the gold set.** Replay your 100 gold questions through the live system to generate the first traces. They're not real users, but they exercise the full pipeline and populate Phoenix with trace data you can practice the eval-in-prod machinery on.
- **Expanded synthetic distribution.** Use a model to generate *variations* of the gold questions — paraphrases, harder multi-hop versions, edge cases — to approximate the long tail the static gold set lacks. This is a cheap stand-in for the real query distribution until you have one.
- **Treat the chaos drill itself as traffic.** The probes you send during each drill produce traces and scores; the index-corruption drill's before/after faithfulness comparison *is* an eval-in-prod measurement.

The point of bootstrapping isn't to fake having users — it's to build and exercise the *machinery* (the replay harness, the online judge, the alerting) so it's ready the moment real traffic arrives. A capstone that has the eval-in-prod plumbing wired and tested against synthetic traffic is far more production-ready than one that plans to "add monitoring later." Later is when the regression has already shipped.

One cost note on the online judge: you sample, you don't score everything. Scoring 100% of live answers with `claude-opus-4-8` would roughly double your inference cost (every answer is generated *and* judged). A 5% sample gives you a statistically meaningful rolling metric at 5% of the judging cost. If you need a higher sample rate for a high-stakes route, drop the judge to `claude-haiku-4-5` — a cheaper judge that's calibrated against the same human labels is a reasonable trade for volume, as long as you've confirmed it agrees with your labels on the calibration set. The principle is the same as the rest of the course: spend the expensive token only where it earns its keep, and the online judge is no exception.

And a measurement-integrity note: the online judge must use the *same* calibrated rubric as the offline gate. If your offline gate's faithfulness threshold is 0.85 against one rubric and your online judge uses a different prompt with a different scale, the two numbers aren't comparable — a "0.80" online and a "0.80" offline mean different things, and you can't tell whether production has regressed relative to the gate. Share the rubric and the calibration set across both. Consistency of measurement is what lets you say "production is worse than the build we shipped" with confidence.

> **The progression:** offline gold-set gate (Sprint B) → trace replay on prod data → shadow traffic → online judge on live traffic. Each step is closer to production reality and catches regressions the previous step missed. A serious agentic product runs all of them.

### 1.4 A worked example of why each layer matters

Make the progression concrete with a single candidate change — a new writing-agent prompt that you *think* is better. Here's what each layer tells you, and what it can't:

- **The offline gold-set gate says:** "0.91 faithfulness on the 100 gold questions — ship it." But the gold set is questions you wrote months ago. It doesn't contain the gnarly multi-hop question a real user asked yesterday, or the question about a document added to the corpus last week. *The gate's blind spot: the real query distribution.*
- **Trace replay says:** "0.84 on yesterday's 2,000 production queries — *regression*, don't ship." The replay caught what the gold set missed: the new prompt is worse on the real distribution, specifically on the long-tail questions the gold set never had. You fix the prompt and replay again until it's not a regression. *Replay's blind spot: it scores against the answers that shipped, which may themselves be imperfect, and it's offline — it doesn't catch live-traffic effects like latency-induced timeouts.*
- **Shadow traffic says:** "on live requests, the candidate matches the incumbent's quality and adds 200ms — acceptable." Now you've validated on truly live data, with the real latency and the real concurrency, at zero user risk. *Shadow's blind spot: it costs double inference, so you run it on a sample for a bounded time, not forever.*
- **The online judge says (after you ship):** "live faithfulness has been 0.89 for three days, holding." The continuous monitor confirms the candidate is *still* good a week later, when the corpus has drifted and the query mix has shifted. *This is the only layer that catches a regression that starts after the deploy.*

Notice the pattern: each layer is *necessary but not sufficient*, and each catches a class of regression the cheaper layer behind it misses. You don't pick one — you stack them, cheapest-and-earliest first (the offline gate gates CI), progressively closer to production (replay, then shadow), then continuous (the online judge). The offline gate is the fast, cheap first filter; the online judge is the slow, expensive last line. Together they give you confidence at every stage from "don't ship a broken build" to "production is still healthy a week later." That full stack is what "eval-in-prod" actually means in a serious shop — not one magic online metric, but a layered defense against regression, mirroring the defense-in-depth you built against prompt injection.

---

## 2. Safe deploys — changing the model without breaking prod

You've validated a candidate with eval-in-prod. Now you ship it — without a flag day. Two patterns, each fitting a different risk profile.

### 2.1 Blue/green — instant switch, instant rollback

Run *two* full stacks: **blue** (current) and **green** (candidate). All traffic goes to blue. You deploy the candidate to green, smoke-test it, and then flip the router to send all traffic to green — atomically. If green misbehaves, you flip back to blue — atomically. The rollback is a config switch, not a redeploy.

```yaml
# The LiteLLM router (or your load balancer) points at one stack at a time.
active_stack: blue          # flip to green to deploy; flip back to roll back
stacks:
  blue:  { writing_agent: prompt_v3, model: claude-opus-4-8 }
  green: { writing_agent: prompt_v4, model: claude-opus-4-8 }   # candidate
```

Blue/green's superpower is the **instant, atomic rollback**. When a deploy goes wrong at 3 AM, you don't want to be redeploying — you want to flip one switch and be back on the known-good stack in seconds. The cost is running two full stacks during the transition.

### 2.2 Canary by cohort — gradual exposure

A more cautious pattern: the candidate (green) serves a *small cohort* first — 5% of users — while 95% stay on blue. You watch the per-cohort metrics (faithfulness, latency, error rate, cost) for the canary cohort for an hour. If they hold, you ramp: 5% → 25% → 100%. If they regress, you roll the canary back to 0% — and only 5% of users ever saw the bad version.

```python
def route_by_canary(user_id, canary_fraction=0.05):
    # Deterministic per-user assignment so a user has a consistent experience.
    if hash(user_id) % 100 < canary_fraction * 100:
        return "green"      # candidate
    return "blue"           # current
```

Canary fits when the change is risky and the blast radius of a bad version must be *minimized* (only 5% exposed) rather than *fast-reverted* (blue/green's bet). For an agentic system, canary-by-cohort lets you watch a new writing-agent prompt's real faithfulness on a slice of real users before everyone gets it.

> **When to use which:** blue/green when you can tolerate exposing 100% briefly and want the fastest possible rollback. Canary when you want to *bound* who's exposed and ramp by the metrics. Many teams use both — canary to ramp, blue/green stacks as the rollback target.

### 2.3 What's different about deploying a *model* change

Deploying a new model or a new prompt is not quite like deploying new code, and the differences matter for which safety net you reach for:

- **The regression is often silent.** New code that breaks usually throws an error or fails a test. A new prompt that subtly degrades faithfulness produces *fluent, plausible, wrong* answers — no error, no crash, just worse. This is why eval-in-prod (§1) gates the deploy *before* the safe-deploy mechanism ships it: the gate catches the silent regression that a green CI build and a smoke test would miss.
- **The blast radius is per-answer, not per-request.** A code bug might 500 every request. A model regression makes *some* answers worse — the hard ones, the edge cases — while the easy ones look fine. A demo of three easy queries won't show it. Canary-by-cohort is well suited here because it lets the metric (per-cohort faithfulness over many real queries) surface a degradation that no single query reveals.
- **The cache is model-scoped.** Switching models invalidates your prompt cache and your vLLM prefix cache — the new model pays cold cache writes on its first requests. Blue/green's brief 100% exposure means a cache-cold spike right at switch time; canary's gradual ramp warms the new model's cache more gently. Factor the cache-warm cost into the rollout if your serving economics depend on it.
- **Rollback must be *instant* and *complete*.** Half-rolling-back a model change (some users on old, some on new, no clean state) is worse than either. Both blue/green (atomic flip) and canary (set the cohort fraction to 0) give you a clean, complete revert. Don't improvise a model rollback by editing the prompt back by hand under pressure — that's how you ship a typo at 3 AM.

The senior framing: a model deploy is a *quality* change, not just a *functional* one, so the safety net has to be a *quality* net (eval-in-prod, per-cohort metrics) on top of the *functional* net (blue/green rollback). Code-deploy instincts alone will let a silent regression through.

To make the choice concrete, a quick decision guide for *your* capstone:

- **A small prompt tweak to the writing-agent, low risk** → trace replay to confirm no regression, then blue/green flip. Fast, simple, instantly reversible.
- **A model swap (local 7B → 13B, or a new vendor model), higher risk** → shadow traffic first to validate online quality, then canary-by-cohort to ramp, with blue/green stacks as the rollback target. The expensive change gets the full safety stack.
- **A change to the supervisor's routing logic, system-wide blast radius** → canary by cohort, watching per-cohort cost *and* faithfulness (a routing bug shows up as both a cost spike and a quality change). Ramp slowly; a routing regression affects every query.
- **An emergency fix to a live incident** → blue/green, because you need the *instant* rollback if the fix makes things worse. Canary's gradual ramp is too slow when you're already on fire.

The through-line: match the deploy mechanism to the *risk and reversibility* of the change, and always put a *quality* gate (eval-in-prod) in front of the *functional* deploy. A capstone that can articulate this — "I'd canary the model swap and blue/green the prompt tweak, here's why" — is demonstrating exactly the production judgment the sealed review and the interview are looking for.

---

## 3. The on-call runbook for an agentic system

> **An agentic system without traces is a closed-box.** (Week 18.) An agentic system *with* traces but *without a runbook* is a box you can see into but don't know how to fix at 3 AM. The runbook is the bridge.

A runbook is the pre-written response to each incident class. You write it *before* the incident — calmly, with the chaos-drill findings fresh — so that during the incident you *read* it instead of improvising. It has four parts.

### 3.1 The alert set

The alerts you actually respond to. Each must be *actionable* — if you can't do anything about it, it's noise, not an alert.

| Alert | Fires when | First action |
|-------|-----------|--------------|
| **Cost spike** | $/query rolling mean > 2× baseline | Check routing — is a misroute sending everything to the vendor? (Sprint B classifier) |
| **Latency spike** | p95 by agent step > SLO | Check the trace — which agent step is slow? vLLM saturated? Vendor slow? |
| **Faithfulness/hallucination spike** | online judge rolling mean < 0.80 | Check recent deploys — did a prompt/model change regress? Roll back the canary. |
| **Attack-rate spike** | output classifier flagging > N/min | Prompt-injection campaign? Check the tool-defense logs (drill 2's defenses). |
| **Error-rate spike** | 5xx / failed runs > threshold | Check failover — is a vLLM replica dead and the fallback not firing? (Drill 1.) |

### 3.2 The dashboards

The three dashboards from week 18, now read *during* an incident:

- **Token usage by route** — the cost-spike dashboard. Shows whether the local-vs-vendor split has shifted (a misroute).
- **p95 latency by agent step** — the latency-spike dashboard. Localizes the slowness to a specific subordinate agent or the serving tier.
- **Retrieval precision / faithfulness over time** — the quality dashboard. Shows whether a regression started, and when (correlate with deploys).

A senior move: each alert in §3.1 names *which dashboard to open first*. The runbook removes the "which graph do I even look at?" panic.

### 3.3 The incident classes

For each common incident, the runbook has a short procedure — the distilled finding from the chaos drills:

- **GPU node loss** → "LiteLLM should fail over automatically (drill 1). If error-rate is non-zero, the health-check/cooldown config is wrong; restart the replica or force the vendor fallback. Recovery: ~seconds if failover works."
- **Prompt-injection campaign** → "The tool defenses should hold (drill 2). Check the path-traversal and output-classifier logs. If something leaked, rotate any exposed credential and patch the bypassed layer. Recovery: patch + redeploy."
- **Index corruption** → "Restore from the latest backup (drill 3). Recovery: ~45s by the rehearsed restore. Confirm faithfulness returns to baseline."
- **Cost spike** → "Check the classifier; a misroute sends everything to the vendor. Force easy routes local until fixed."

These procedures are the *output* of the chaos drills. The drill measured the recovery time; the runbook records it so the next person knows what "normal recovery" looks like.

### 3.4 The escalation path and the postmortem template

Who to page when the first responder is stuck, the severity definitions, and — critically — a *template* for the postmortem so that every incident produces a consistent write-up. The runbook is a living document: every chaos drill and every real incident updates it.

A good escalation path answers three questions explicitly:

- **When do you escalate?** Not "when you're stuck" (too vague) but a *condition*: "if the failover hasn't recovered the system within 5 minutes," "if the attack-rate alert keeps firing after the first mitigation," "if you don't recognize the incident class." A clear escalation trigger prevents both the hero who flails for an hour and the panicker who escalates a known-trivial issue.
- **Who do you escalate to?** A named role (the on-call lead, the platform owner), not a person who might be on vacation. Roles survive turnover; names don't.
- **What do you hand off?** The incident state: what you've observed, what you've tried, the current dashboards. A handoff with no state means the next responder starts from zero — the worst possible position in an incident.

### 3.5 A worked incident — reading the runbook under fire

Here's how the runbook *gets used*. It's 2 AM. The cost-spike alert fires: $/query rolling mean is 3× baseline. You open the runbook to the cost-spike entry:

1. **First dashboard:** token usage by route. You see the local-vs-vendor split has flipped — 90% of queries are going to the vendor when it's normally 30%.
2. **First hypothesis (from the runbook):** the easy-vs-hard classifier is mislabeling. You check a few recent classifier decisions in the trace — they're all "hard," even for trivial lookups.
3. **First action (from the runbook):** force easy routes local while you investigate (`local_only=true` flag), which immediately drops the cost back toward baseline.
4. **Root-cause investigation:** the classifier's model or prompt changed in a recent deploy. You roll back the canary (or flip the blue/green stack) for the classifier change.
5. **Recovery confirmed:** cost back to baseline, split back to 30/70. You note the timeline and write the postmortem in the morning.

Notice what the runbook did: it told you *which dashboard*, *what to suspect*, *what to do first* (a fast mitigation), and *how to recover*. None of that required you to think clearly at 2 AM — which is good, because at 2 AM you don't think clearly. The runbook *is* the thinking, done in advance, when you were calm. That's the whole value proposition: **move the cognition from the incident to before the incident.** This is why you write the runbook this week, with the chaos-drill findings fresh, rather than discovering you need it during the first real outage.

---

## 4. The blameless postmortem

The chaos drill ends in a postmortem. So does every real incident. The format is the standard incident write-up (Google SRE):

1. **Summary** — one paragraph: what happened, impact, duration.
2. **Timeline** — the minute-by-minute record (this is the drill's measured timeline, or the incident's).
3. **Impact** — user-visible effect: errors served, degraded latency, dollars, faithfulness drop.
4. **What worked** — the defenses/fallbacks/backups that held. (In a drill, this is the resilience you proved.)
5. **What didn't** — the gaps the fault exposed. (In a drill, this is the patch you wrote.)
6. **Root cause** — the *system* cause, not the person.
7. **Action items** — concrete, owned, dated follow-ups.

Each section does a job: the summary orients a reader with thirty seconds; the timeline grounds the analysis in facts; the impact quantifies the cost; what-worked and what-didn't separate proven resilience from found gaps; the root cause points at the system; and the action items turn learning into change. A postmortem missing any one is incomplete — most often it's the action items (so nothing changes) or the timeline (so the analysis is guesswork).

A note on the timeline section: it is the most valuable and the most often skimped. The timeline is the *minute-by-minute record* of what happened and when — fault injected at t+12s, failover detected at t+13s, recovered at t+300s. For a real incident it's reconstructed from logs and traces; for a chaos drill it's the measured output of your runner. A precise timeline is what lets the postmortem compute recovery time, identify the detection lag, and tell a coherent story. A vague timeline ("things were broken for a while, then we fixed it") makes the rest of the postmortem guesswork. Capture timestamps as you go — your runner does this automatically, which is half the reason it exists.

The non-negotiable property is **blameless**.

> **Why blameless is the only kind that works:** a postmortem that names and blames a person teaches everyone to *hide* incidents, because admitting one gets you punished. A blameless postmortem assumes people act reasonably given the information they had, and asks: *what about the system let this happen, and how do we change the system so the next reasonable person doesn't hit it?* Blame fixes nobody and makes the next incident harder to learn from. Blamelessness fixes the system. This is not a soft nicety — it is the mechanism by which an organization gets *more* reliable over time instead of more secretive.

For your capstone, the postmortem covers all three drills: GPU node loss (what held, recovery time), prompt-injection (what held or what you patched, attack success before/after), and index corruption (faithfulness regression, recovery time). It's a required final deliverable, and it's the document that proves you didn't just *run* the drills — you *learned* from them.

### 4.1 Action items that actually close the loop

The weakest part of most postmortems is the action items: a list of vague aspirations ("improve monitoring", "be more careful with the index") that nobody owns and nobody does. A good action item has three properties:

- **Owned.** A named person (or you) is responsible. "Someone should add an alert" never happens; "I will add the faithfulness alert" does.
- **Dated.** A due date, even a soft one. An undated action item is a wish.
- **Concrete and verifiable.** "Add a LiteLLM health-check cooldown of 30s and re-run the node-loss drill to confirm zero errors" is verifiable — you can check whether it was done and whether it worked. "Improve resilience" is not.

A worked contrast, from a real index-corruption finding:

> ❌ **Weak:** "We should monitor the index better."
>
> ✅ **Strong:** "[Owner: you] [Due: before submission] Add an online faithfulness judge sampling 5% of live answers, with a Phoenix alert when the rolling mean drops below 0.80, so silent index degradation pages us instead of being noticed by a user. Verify by re-running the index-corruption drill and confirming the alert fires."

The strong version names the owner, the date, the concrete change, *and* the verification. The weak version is a feeling. Graders (and good engineers) can tell the difference instantly — and the strong version is the one that actually gets done, because it's specific enough to start on Monday morning.

The action items are where the drill's *learning* becomes the system's *improvement*. A postmortem with measured timelines and no action items is a nice story; a postmortem whose action items get done is how the system gets more robust over time. In your capstone, each drill that found a gap (a failover that errored, an injection that bypassed a layer, an eval blind to corruption) produces at least one owned, dated, verifiable action item — and ideally you *do* it and re-run the drill before you submit, so the postmortem can say "patched and re-verified."

### 4.2 Why the postmortem closes the chaos-drill loop

Trace the full loop: you formed a steady-state hypothesis, injected a fault, measured the impact and recovery, reverted, and now you write it up. The write-up is what makes the drill *transferable* — it turns a thing that happened in your head on a Tuesday afternoon into a document a teammate (or future-you) can read and act on. Without the postmortem, you learned something but the *organization* learned nothing; the next person hits the same gap. With it, the recovery time goes into the runbook, the patch goes into the code, and the next drill starts from a stronger baseline. The postmortem is the closing bracket on the chaos drill, and the opening bracket on the system's next iteration.

---

## 5. What you can do now

You can:

- Run eval-in-prod in all three modes — trace replay on prod data, shadow traffic, online LLM-as-judge on live traffic — and explain why each catches regressions the offline gold-set gate misses.
- Deploy a model change with blue/green (instant atomic rollback) or canary-by-cohort (bounded exposure, ramp by metrics), and pick the right one for the risk.
- Write an on-call runbook for an LLM product: the actionable alert set, the three dashboards mapped to incidents, the per-incident-class procedures distilled from the chaos drills, and the escalation path.
- Write a blameless postmortem in the standard incident format, and explain why blame makes systems less reliable, not more.

That is the close of C23. You built a Production Agentic Research Assistant, you broke it on purpose in a controlled window, you measured how it failed and recovered, you stood up the eval-in-prod and deploy mechanics that keep it healthy, and you wrote the runbook and the postmortem that make it operable by a human at 3 AM. You know — by measurement, not by hope — whether your system is production. The mini-project this week is to *finish it*: the postmortem, the runbook, the 5-minute video, and the career pack. Then you submit, and you graduate able to do the thing the syllabus promised on day one — and a great deal more.

---

## 6. The alerting discipline — actionable, not noisy

One more thing the runbook depends on, because it's the thing that makes the whole on-call system work or fail: **every alert must be actionable**. An alert that fires but has no clear response trains the on-call engineer to ignore alerts — and an ignored alert is worse than no alert, because it gives false confidence that *something* would catch a problem.

The test for whether an alert belongs in your set: *when it fires at 3 AM, is there a specific thing the responder does?* If yes, it's an alert. If the honest answer is "look at it and probably go back to sleep," it's noise — turn it into a dashboard you check during business hours, not a page.

This is why the alerts in §3.1 are tied to *SLO breaches* and *clear thresholds*, not raw metrics:

- **Don't alert on:** "p95 latency is 2.3s" (a number with no judgment attached).
- **Do alert on:** "p95 latency exceeded the 2.5s SLO for 5 consecutive minutes" (a breach, with a duration to filter transients, and a runbook entry to respond to).

The 5-minute duration is the *transient filter* — it stops a one-second blip from paging you. The SLO is the *judgment* — it encodes what "too slow" means for *your* product. Together they turn a raw metric into an actionable alert. Get this wrong (alert on raw metrics, no duration, no SLO) and you'll page yourself into alert fatigue within a week, then start ignoring the pages, then miss the real incident. Alert fatigue is how good monitoring becomes useless monitoring.

For an agentic system specifically, the quality alerts (faithfulness/hallucination spike, attack-rate spike) are the ones most teams forget — they alert on latency and errors (the request/response instincts) but not on *answer quality*, which is the thing that actually matters for an LLM product. A system can be fast, error-free, and confidently wrong. The online judge from §1 is what makes the faithfulness alert possible; wire it, and you can page on "the answers got worse" — the alert that catches the silent regression no latency or error metric ever will.

This closes the loop the whole week has been drawing: the chaos drills measured how the system fails; the eval-in-prod catches regressions; the deploy mechanics ship changes safely; and the runbook — with its actionable, SLO-tied alerts mapped to dashboards and procedures — is what turns all of that measurement into a system a human can *operate*. A production agentic system is not just one that works; it's one that *fails predictably, recovers measurably, regresses detectably, deploys safely, and is operable by someone who didn't build it.* You now have all five.

---

*If you find errors in this material, please open an issue or send a PR.*
