# Week 24 — Quiz

Thirteen questions. The last quiz of C23. Take it with your lecture notes closed. Aim for 11/13. Answer key is at the bottom — don't peek.

---

**Q1.** Why inject faults into your system on purpose?

- A) To prove the system can't break.
- B) To move the failure from "at 3 AM, in front of a user, no rollback" to "on your schedule, watching, with a tested revert" — so you measure the real recovery time and confirm the defense held before a user finds the gap.
- C) Because faults never happen otherwise.
- D) To increase the error rate for marketing.

---

**Q2.** What is the steady-state hypothesis in a chaos drill?

- A) A guess about what might break.
- B) The measurable "normal" — a *number* (error_rate=0%, p95<2.5s, faithfulness>=0.85) measured before the fault, so you can tell degradation from noise.
- C) The name of the fault.
- D) The revert procedure.

---

**Q3.** Why must the revert be *tested before* the fault is injected?

- A) It doesn't; you can test it after.
- B) Because corrupting your only index (or killing the only replica) with an untested or broken revert turns a controlled drill into a real outage you caused. For the index drill, restore the backup onto a copy and confirm health first.
- C) Testing the revert disables the fault.
- D) The revert is optional.

---

**Q4.** In the GPU-node-loss drill, what outcome shows the system is *production*?

- A) error_rate jumps to 100% the instant a replica dies.
- B) Zero user-visible errors and bounded degraded latency: LiteLLM routes around the dead replica and, when the local tier is fully down, fails over to the vendor (`claude-opus-4-8`) — graceful degradation, not an outage.
- C) The system stays at exactly the same latency.
- D) The supervisor stops routing.

---

**Q5.** The GPU-node-loss drill prints error_rate=100% the moment you kill a replica. What's the likely bug and the fix?

- A) The model is broken; retrain it.
- B) LiteLLM's health check / cooldown isn't configured, so requests route to the dead replica and fail; the fix is to enable health checks + cooldown and the `fallbacks` chain (and a valid vendor key).
- C) The gold set is wrong.
- D) Nothing; 100% errors is expected on a node loss.

---

**Q6.** The prompt-injection drill plants a hostile instruction in a *retrieved document*. This is:

- A) A direct prompt injection from the user.
- B) An *indirect* prompt injection (OWASP LLM01) — smuggled in via retrieved content, not the user prompt — which is why defense-in-depth (input filter + tool-arg validation + output classifier) matters: assume one layer is bypassed and the next must hold.
- C) A denial-of-service attack.
- D) A data-corruption attack.

---

**Q7.** In the prompt-injection drill, the answer leaks `/etc/passwd` contents. How should you treat this?

- A) As a failed drill to hide.
- B) As the most valuable possible outcome: you found a real vulnerability before an attacker did, on your schedule. Write the patch (fix the bypassed layer), re-run until the attack success rate is 0, and report the rate before and after.
- C) As proof the system is fine.
- D) As a reason to remove the filesystem tool entirely.

---

**Q8.** In the index-corruption drill, you corrupt 5% of the vector store and Ragas faithfulness does *not* move. What does that tell you?

- A) The corruption didn't happen.
- B) Your eval has a hole — it's scoring something other than true grounding, or the gold set doesn't exercise the corrupted region. A real grounding eval should drop when retrieval is corrupted. Fix the eval.
- C) The system is perfectly resilient.
- D) Faithfulness is the wrong metric.

---

**Q9.** Why is the offline gold-set gate (Sprint B) not enough for production?

- A) It's too expensive.
- B) It's static and offline — it scores last week's gold set, not the live, drifting distribution of real queries. A build green on the gold set can be quietly failing on the long tail; eval-in-prod (trace replay, shadow traffic, online judge) is how you find out from production first.
- C) Ragas can't run on production.
- D) The gold set is always wrong.

---

**Q10.** What is *shadow traffic* in eval-in-prod?

- A) Serving a new version to a small cohort.
- B) Running a candidate version against *live* requests but *not serving its output to users* — the user gets the current version; the candidate's answer is scored and logged. Zero user risk, real-data validation.
- C) Replaying old traces offline.
- D) Logging all traffic to a shadow database.

---

**Q11.** Blue/green vs canary-by-cohort — what's the key difference?

- A) They're the same thing.
- B) Blue/green runs two full stacks and switches/rolls back *atomically and instantly* (you may expose 100% briefly); canary exposes a *small cohort first* (e.g. 5%), ramps by the metrics, and *bounds* who sees a bad version. Choose by whether you want fastest rollback or minimized exposure.
- C) Blue/green is for code, canary is for models.
- D) Canary has no rollback.

---

**Q12.** Why must a postmortem be *blameless*?

- A) To be polite.
- B) A postmortem that blames a person teaches everyone to hide incidents (admitting one gets you punished); a blameless one asks what about the *system* let this happen and how to change it, which is the mechanism by which an organization gets more reliable instead of more secretive.
- C) Because no one is ever at fault.
- D) Blameless postmortems are faster to write.

---

**Q13.** "The runbook you write before the incident is the one you read during it." What does a good on-call runbook for an agentic system contain?

- A) The full source code.
- B) An actionable alert set (cost/latency/faithfulness/attack/error spikes), the dashboards mapped to each incident, per-incident-class procedures distilled from the chaos drills (with measured recovery times), the escalation path, and a postmortem template — a living document.
- C) Only the architecture diagram.
- D) A list of who to blame.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Move the failure onto your schedule; measure real recovery; confirm the defense held first. (Lecture 1 §1.)
2. **B** — A measured number, taken before the fault, so degradation is distinguishable from noise. (Lecture 1 §2.)
3. **B** — An untested revert turns a drill into an outage; test the restore on a copy first. (Lecture 1 §2, §5.)
4. **B** — Zero user-visible errors + bounded degradation = graceful degradation = production. (Lecture 1 §3.)
5. **B** — Health-check/cooldown + fallback chain misconfigured; that's the fix. (Lecture 1 §3.)
6. **B** — Indirect injection (LLM01); defense-in-depth assumes one layer is bypassed. (Lecture 1 §4.)
7. **B** — A leak is the most valuable outcome: you found it first; patch and re-run to a 0 rate. (Lecture 1 §4.)
8. **B** — A non-moving faithfulness on corrupted retrieval means the eval has a hole. (Lecture 1 §5.)
9. **B** — The gold set is static and offline; eval-in-prod scores live, drifting traffic. (Lecture 2 §1.)
10. **B** — Shadow traffic runs the candidate on live requests without serving its output. (Lecture 2 §1.2.)
11. **B** — Blue/green = atomic instant rollback; canary = bounded exposure, ramp by metrics. (Lecture 2 §2.)
12. **B** — Blame drives incidents underground; blamelessness fixes the system. (Lecture 2 §4.)
13. **B** — Alerts, dashboards, per-incident procedures from the drills, escalation, template — living. (Lecture 2 §3.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md) — and the final capstone close-out. This is the last quiz of C23. Well done getting here.
