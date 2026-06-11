# Week 24 — Resources

Every resource here is **free** or has a free tier. The chaos-engineering principles, LiteLLM, Arize Phoenix, Langfuse, and the OWASP LLM Top 10 are all open. The Anthropic SDK is free to install; the vendor fallback (`claude-opus-4-8`) and any online LLM-as-judge consume paid tokens, but the drill spend is small.

This is the final week — it pulls forward almost everything: weeks 17 (safety/prompt-injection), 18 (observability), 19 (vLLM/LiteLLM), 10 (vector-store operations/backup), 12 (Ragas), and the Sprint B system from week 23. The resources below assume you have that system runnable.

## Required reading (work it into your week)

- **Principles of Chaos Engineering** — the canonical statement: steady-state hypothesis, vary real-world events, run in production, minimize blast radius, automate. Read it once, then re-read the steady-state-hypothesis section before you design a drill:
  <https://principlesofchaos.org/>
- **OWASP Top 10 for LLM Applications** — the threat taxonomy; LLM01 (prompt injection) is the one the prompt-injection drill exercises. Read LLM01 and the "insecure tool use" entries:
  <https://owasp.org/www-project-top-10-for-large-language-model-applications/>
- **LiteLLM — routing & fallbacks** — the failover config the GPU-node-loss drill tests: a dead replica or a dead local tier degrades to the vendor. Read the `fallbacks` and health-check sections:
  <https://docs.litellm.ai/docs/routing>
- **Arize Phoenix — evaluation** — running evals over traces, online evaluation, and the eval-in-prod patterns you use to catch a regression before users do:
  <https://docs.arize.com/phoenix/evaluation/llm-evals>

## Chaos & resilience references

- **Netflix — Chaos Monkey / the chaos-engineering origin** — the lineage; useful for the *why* (fail on purpose, on your schedule) even though you're not running their tooling:
  <https://netflix.github.io/chaosmonkey/>
- **Google SRE Book — Postmortem culture** — the blameless postmortem, the standard incident format, and why blame makes systems worse. The template your postmortem follows:
  <https://sre.google/sre-book/postmortem-culture/>
- **Google SRE Book — Managing incidents** — incident command, roles, and the escalation path your runbook documents:
  <https://sre.google/sre-book/managing-incidents/>
- **The on-call chapter (SRE Workbook)** — alert fatigue, the alert set, and what makes an alert actionable rather than noise:
  <https://sre.google/workbook/alerting-on-slos/>

## The three drills (have these open on Thursday)

- **LiteLLM — health checks & cooldowns** — how LiteLLM detects a dead replica and routes around it; the mechanism behind the GPU-node-loss failover:
  <https://docs.litellm.ai/docs/proxy/health>
- **OWASP — LLM01 Prompt Injection** — direct vs indirect injection; the indirect-via-retrieved-document attack is the one the prompt-injection drill runs:
  <https://genai.owasp.org/llmrisk/llm01-prompt-injection/>
- **pgvector / Qdrant backup & restore** — the operational story for restoring the vector store after the corruption drill; whichever store your capstone uses:
  <https://github.com/pgvector/pgvector> · <https://qdrant.tech/documentation/concepts/snapshots/>
- **Ragas — faithfulness** — the metric the index-corruption drill watches; corrupting 5% of the store should drop faithfulness measurably:
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/>

## Eval-in-prod & deploys

- **Arize Phoenix — online evaluation** — scoring live/sampled production traffic; the continuous version of the static gate:
  <https://docs.arize.com/phoenix/evaluation/concepts-evals/evaluation>
- **Langfuse — datasets & experiments** — replaying a dataset of production traces through a candidate prompt/model; the offline half of eval-in-prod:
  <https://langfuse.com/docs/datasets/overview>
- **Blue/green & canary (general)** — the deploy patterns adapted to model serving; the concepts are stable even as the tooling shifts:
  <https://martinfowler.com/bliki/BlueGreenDeployment.html>

## On-call & incident response

- **PagerDuty — incident response docs** — a practical, open guide to incident roles, severities, and comms; a good complement to the SRE book:
  <https://response.pagerduty.com/>
- **The C23 production runbook spec** — the career-pack `production-runbook.md` you write this week: alerts, dashboards, incident classes (cost/latency/hallucination/attack), escalation, postmortem template. The spec is in the syllabus's career-engineering-pack section.

## The vendor fallback & the judge (the Anthropic SDK)

- **Anthropic Python SDK** — `pip install anthropic`. The vendor fallback (`claude-opus-4-8`) that keeps the hard routes up when the local tier dies, and the online LLM-as-judge for eval-in-prod. Adaptive thinking via `thinking={"type":"adaptive"}`, depth via `output_config={"effort":...}`:
  <https://github.com/anthropics/anthropic-sdk-python>
- **Anthropic — error handling & stop reasons** — handling `refusal`, rate limits, and overload during a drill; the SDK retries 429/5xx automatically, which matters when you throttle the vendor key as a stretch drill:
  <https://platform.claude.com/docs/en/api/errors>

## Models you'll use this week

- **`claude-opus-4-8`** — the vendor fallback for the hard routes (the GPU-node-loss drill fails over to it) and the online LLM-as-judge for eval-in-prod. Adaptive thinking only; `thinking={"type":"adaptive"}` + `output_config={"effort":...}`. Never `budget_tokens` / `temperature` (both 400).
- **`claude-haiku-4-5`** — a cheap option for a high-volume online judge if you sample-score a large fraction of live traffic.
- **The local 7B/13B on vLLM (or Ollama)** — the tier you *kill* in the node-loss drill; the thing the fallback routes around.

## Tools you'll use this week

- **The Sprint B capstone** — your week-23 system. This week attacks it. Nothing new to build except the drill runner and the postmortem.
- **`litellm`** — the router whose failover the node-loss drill tests.
- **`ragas`** — the faithfulness metric the index-corruption drill watches.
- **`arize-phoenix`** — the eval-in-prod backend; online judge + trace replay.
- **`anthropic`** — the vendor fallback and the online judge.
- **A vector-store backup** — whatever your store offers (pgvector dump, Qdrant snapshot). You restore from it in the corruption drill; if you don't have one, making it is the first thing you do.

## A note on the controlled window

The chaos drill runs in a **single 4-hour window** against your *own* capstone — never against someone else's system, never against a shared resource, never without a revert ready. The discipline is: state the steady-state hypothesis, define the blast radius (one replica, one document, 5% of the index), inject *one* fault, measure, revert, write it down. The fault must be *reversible* and the revert must be *tested first* — you confirm you can restore the index from backup *before* you corrupt it, not after. Chaos engineering is rehearsal, not roulette.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Chaos engineering** | Injecting controlled faults to learn how a system fails, on your schedule. |
| **Steady-state hypothesis** | The measurable "normal" you expect to hold during a drill (e.g. error_rate=0%). |
| **Blast radius** | The bounded scope of a fault (one replica, one doc, 5% of the index). |
| **Graceful degradation** | The system gets slower / lower-quality but stays *up* under a fault. |
| **Failover** | Automatically routing around a dead component (LiteLLM → vendor fallback). |
| **Recovery time** | Wall-clock from fault-injected to steady-state-restored. |
| **Indirect prompt injection** | A malicious instruction smuggled in via a *retrieved document*, not the user prompt. |
| **Defense-in-depth** | Layered defenses (input filter + arg validation + output classifier) so one miss isn't fatal. |
| **Eval-in-prod** | Scoring real production traffic (or replayed traces), not just an offline gold set. |
| **Shadow traffic** | Running a new version against real requests *without serving its output to users*. |
| **Blue/green** | Two full stacks; switch traffic atomically; roll back atomically. |
| **Canary** | A new version serves a small cohort (e.g. 5%) first; ramp or roll back by the metrics. |
| **Runbook** | The pre-written guide to responding to an incident class. |
| **Postmortem** | The after-incident write-up: timeline, impact, what worked/didn't, action items. |
| **Blameless** | A postmortem that fixes the system, not the person — the only kind that improves things. |

---

*If a link 404s, please open an issue so we can replace it.*
