# Week 21 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 22. Answer key is at the bottom — don't peek.

---

**Q1.** What is the correct ordering of "cheapest token" in this week's mantra, and why does it drive the whole week?

- A) The cheapest token is the most expensive model's; quality is all that matters.
- B) The cheapest token is the one you don't generate (caching avoids it entirely); the second cheapest is one a small model handles instead of a frontier model (routing). Every lever this week is one of those two moves.
- C) The cheapest token is the shortest one; minimize word length.
- D) All tokens cost the same; the mantra is about latency.

---

**Q2.** Why attribute cost *per route and per feature* rather than tracking a single monthly total?

- A) Vendors require per-feature billing.
- B) A single total ("$4,000 last month") isn't actionable; per-route attribution from `usage` reveals *where* the money goes — e.g. a trivial classify feature running 50K calls/day on a frontier model — which tells you exactly which lever to pull where.
- C) Per-feature tracking is cheaper to compute.
- D) It's only needed for the carbon report.

---

**Q3.** For most models, which costs more per token — input or output — and what does that imply for cost engineering?

- A) Input; so compress the answer.
- B) Output (typically 4–5× the input price); so a long *answer* is more expensive than a long *prompt*, and capping verbose output (a `max_tokens` ceiling, a "be concise" instruction, structured output) attacks the expensive side directly.
- C) They're equal; the distinction doesn't matter.
- D) Input; so always cache the output.

---

**Q4.** Exact-match caching is "necessary but insufficient." Why insufficient for natural-language workloads?

- A) It's too slow to compute a hash.
- B) Users rarely ask the *exact same bytes* twice — "what's your refund window?" and "how long do I have to get a refund?" are the same question and different strings, so exact-match misses the paraphrase. Semantic caching closes that gap.
- C) Exact-match returns wrong answers.
- D) It requires a GPU.

---

**Q5.** In a semantic cache, the cosine threshold is the cost-vs-correctness knob. What does a *too-loose* threshold cause?

- A) It speeds up the embedder.
- B) More queries count as "similar enough," so the hit rate (and saving) rises — but you start returning a cached answer to *different* questions (refund served for return), so the wrong-answer rate climbs. The "saving" is a quality regression in disguise.
- C) It disables the cache.
- D) It only affects latency, not correctness.

---

**Q6.** When you sweep the semantic-cache threshold, why is "maximize the hit rate" the wrong objective?

- A) Hit rate can't be measured.
- B) The loosest thresholds have the highest hit rate AND the highest wrong-answer rate; maximizing hits alone picks a config that serves many wrong answers. The right objective is "maximize hit rate *subject to* the wrong-answer rate staying under tolerance" — quote the threshold with *both* numbers.
- C) The hit rate is always 100% at every threshold.
- D) Higher hit rate always means higher cost.

---

**Q7.** A semantic cache serves "30 days" to every paraphrase of "what's our refund window?" — but the policy changed to 60 days last week. What's the lesson?

- A) Semantic caches don't work for policies.
- B) A cache without an invalidation story (TTL, version key, or event-driven purge) is a time bomb on any answer that can go stale — it made the system cheaper *and wrong*, the worst quadrant. Cache freely what can't go stale; cache carefully (with invalidation) what can.
- C) The threshold was too tight.
- D) You should never cache.

---

**Q8.** Prompt caching gives a large discount on a repeated prefix. What single mistake defeats it entirely?

- A) Using too long a system prompt.
- B) Breaking byte-stability of the prefix — interpolating a timestamp, a per-request ID, or an unsorted JSON dump at the front — so the cache never matches; `cache_read_input_tokens` stays zero and you've paid the write premium for nothing. Freeze the prefix, put volatile content last.
- C) Using prompt caching with a small model.
- D) Caching the output instead of the input.

---

**Q9.** Prompt compression (summarization or LLMLingua token pruning) reduces input tokens. What's the discipline?

- A) Compress as aggressively as possible to maximize savings.
- B) Measure the quality cost of the compression against a labeled set and operate *before* the quality cliff — compress the fat (boilerplate, redundancy) freely, the substance carefully. There's a ratio where quality falls off; your safe point is before it.
- C) Compression never affects quality.
- D) Only compress the output.

---

**Q10.** A model router's classifier can make two kinds of error. Why are they *not* symmetric in cost?

- A) They're symmetric; both cost the same.
- B) A false-easy error (hard query → cheap model) costs *quality* (a worse answer, invisible in the cost dashboard); a false-hard error (easy query → frontier model) only costs *money* (the frontier model handles easy queries fine). So you bias the classifier conservative — toward escalating when unsure — because missing a hard query is worse than overpaying.
- C) False-hard errors crash the system.
- D) Only false-easy errors exist.

---

**Q11.** A cascade tries the cheap model first and escalates on a verifier failure. With cheap = $0.10 and frontier = $1.00, at what escalation rate does the cascade stop beating the all-frontier baseline?

- A) At any escalation rate above 0%.
- B) At ~90%: expected cost is `0.10 + P(escalate) × 1.00`, which equals the $1.00 baseline when `P(escalate) = 0.90`. Below 90% escalation the cascade is cheaper — a very forgiving bar, which is why cascades usually pay even with an imperfect cheap model.
- C) At 50%.
- D) Cascades never beat the baseline.

---

**Q12.** The cascade's verifier must satisfy two constraints. What are they?

- A) It must be a frontier model and run on a GPU.
- B) It must be *much cheaper* than the escalation it gates (a format check, self-consistency vote, or cheap small-model judge — not a frontier call, which would defeat the purpose), and *calibrated* so its pass/fail tracks actual answer quality (else it leaks quality if too lenient or cost if too strict).
- C) It must always pass.
- D) It must use the same model as the frontier tier.

---

**Q13.** The Anthropic/OpenAI Batch APIs give 50% off. What's the trade, and which traffic should use them?

- A) The trade is quality; only use them for unimportant queries.
- B) The trade is pure *latency* (results within up to 24h) with **zero quality cost** — the answers are identical. So segment traffic by latency tolerance and batch everything patient (evals, bulk processing, offline generation); keep latency-sensitive traffic (interactive chat) on the online path.
- C) The trade is accuracy; batched answers are worse.
- D) They cost more but are faster.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Don't generate the token (cache), or let a small model handle it (route). Every lever is one of those two moves. (Lecture 1 mantra.)
2. **B** — Per-route attribution from `usage` reveals where the money goes and which lever to pull; a single total is not actionable. (Lecture 1 §1, §1b worked table.)
3. **B** — Output is typically 4–5× input price, so verbose answers are the expensive side; cap output to attack it. (Lecture 1 §1.)
4. **B** — Users rarely repeat exact bytes; exact-match misses paraphrases, which semantic caching catches. (Lecture 1 §2.)
5. **B** — Too-loose → more hits but more wrong answers (refund served for return); the saving is a quality regression. (Lecture 1 §3.)
6. **B** — Loosest thresholds maximize both hit rate and wrong-answer rate; the objective is "max hit rate subject to wrong-answer tolerance." (Lecture 1 §3.)
7. **B** — A cache without invalidation goes stale and serves wrong-but-previously-correct answers; cache carefully what can change. (Lecture 1 §3b.)
8. **B** — Breaking prefix byte-stability defeats prompt caching; freeze the prefix, volatile content last, verify with `cache_read_input_tokens`. (Lecture 1 §4.)
9. **B** — Measure the quality cost and operate before the cliff; compress fat freely, substance carefully. (Lecture 1 §5.)
10. **B** — False-easy costs quality (worse, invisible); false-hard costs money (harmless); bias conservative toward 'hard'. (Lecture 2 §1b.)
11. **B** — `0.10 + 0.90 × 1.00 = 1.00 = baseline`; below 90% escalation the cascade wins. The forgiving bar is why cascades usually pay. (Lecture 2 §2.)
12. **B** — The verifier must be much cheaper than the escalation and calibrated to real quality. (Lecture 2 §3.)
13. **B** — Pure latency trade, zero quality cost; batch everything patient, keep interactive traffic online. (Lecture 2 §4.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
