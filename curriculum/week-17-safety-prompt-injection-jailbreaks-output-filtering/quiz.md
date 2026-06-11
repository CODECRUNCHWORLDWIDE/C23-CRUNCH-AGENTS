# Week 17 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 18. Answer key is at the bottom — don't peek.

---

**Q1.** Why is prompt injection the *defining* LLM security issue? Pick the *most complete* answer.

- A) Because LLMs are slow.
- B) Because an LLM cannot reliably distinguish instructions from the developer from instructions in the data — it's all tokens in one context window — and there is no LLM equivalent of a parameterized query to guarantee the separation.
- C) Because models hallucinate.
- D) Because tools are expensive.

---

**Q2.** What is the difference between direct and indirect prompt injection?

- A) Direct is faster than indirect.
- B) **Direct** injection is when the attacker controls the user input and types the malicious instruction; **indirect** injection is when the attacker plants the instruction in content the agent later reads (a retrieved document, a webpage, a tool result) — so it rides in through a trusted channel and the user's own request looks innocent.
- C) Direct attacks are illegal and indirect ones aren't.
- D) They're the same thing.

---

**Q3.** Why is indirect injection scarier than direct injection in an agent that uses RAG?

- A) It isn't — direct is always worse.
- B) The injection rides in through the retrieval pipeline you *built to be helpful*, affects whoever's agent retrieves the poisoned content (not just the attacker), and is invisible at the entry point because the user's request looks completely benign — so an input filter on the user message never sees it.
- C) Indirect attacks use more tokens.
- D) Indirect attacks only work on open-weight models.

---

**Q4.** Why does an agent with tools have a worse injection problem than a plain chatbot?

- A) It doesn't — they're equivalent.
- B) A chatbot that's injected *says* something wrong (bounded); an agent that's injected *does* something wrong — it calls a tool (read a file, run a query, exfiltrate data). The tool surface turns "said something wrong" into "did something harmful."
- C) Agents are slower.
- D) Chatbots can't be injected.

---

**Q5.** Which of these is the OWASP LLM Top 10 entry about giving an agent more tool power than the task needs?

- A) Prompt Injection (LLM01).
- B) **Excessive Agency** — it enlarges the *blast radius*: when an attack lands, how much damage it can do. Least-privilege tools shrink the blast radius.
- C) Insecure Output Handling.
- D) Sensitive Information Disclosure.

---

**Q6.** Why is "defense in depth" (layered defenses) the right posture for LLM safety?

- A) Because one good filter is all you need.
- B) Because there's no complete fix for injection (no parameterized-query equivalent), so every individual defense can be bypassed by *some* attack — you stack independent layers so an attack must defeat all of them, and measure each layer's contribution.
- C) Because layers are cheaper than a single strong filter.
- D) Because the OWASP standard mandates exactly five layers.

---

**Q7.** What is the honest assessment of a regex/keyword input filter?

- A) It fully solves prompt injection.
- B) It's a **speed bump, not a wall** — it catches the obvious un-obfuscated attacks for near-zero cost (so it earns a place as the first layer) but is trivially bypassed by obfuscation (base64, leetspeak, foreign language). Never rely on it alone; stack a classifier behind it.
- C) It's useless and should never be used.
- D) It catches obfuscated attacks better than a classifier.

---

**Q8.** Why is structured tool-argument validation the *load-bearing* defense layer?

- A) Because it's the cheapest.
- B) Because it's **deterministic and model-independent** — it holds *even when the input filter fails and the model is successfully steered* into the malicious call. A `resolve()`-then-`is_relative_to()` check can't be talked out of its logic, unlike a model or a probabilistic filter.
- C) Because it runs first.
- D) Because it replaces the need for output filtering.

---

**Q9.** Why must you filter the model's *output*, not just its input?

- A) You don't — input filtering is sufficient.
- B) The model's output is *also* untrusted (insecure output handling): a steered/jailbroken model produces malicious output, so you check what comes out (for exfil signatures, harmful content, policy violations) before returning it or acting on it downstream.
- C) Output filtering is faster than input filtering.
- D) Because the OWASP standard requires it.

---

**Q10.** You measure a defense and it drops attack-success-rate from 0.6 to 0.1 but also drops benign-pass-rate from 1.0 to 0.7. What have you built?

- A) An excellent defense — ASR is way down.
- B) A **denial-of-service against your own users** — the filter is so aggressive it blocks 30% of legitimate traffic. You measure *both* axes (ASR down AND benign-pass-rate up) and tune to the knee of the trade-off; a security win that wrecks usability isn't a win.
- C) A perfect defense.
- D) A bug in the harness.

---

**Q11.** What makes attack-success-rate a *number* rather than a judgment call?

- A) Running the attacks many times.
- B) A **mechanically-checkable success criterion** for each attack — the canary string appears in the output, a planted file was created, the system prompt leaked. "The agent acted weird" isn't measurable; "the canary appeared" is. (For genuinely fuzzy criteria, use a *calibrated* LLM-as-judge.)
- C) Using a bigger model.
- D) Asking a human to score each one.

---

**Q12.** Your red-team produces an ASR table: input filter bought −0.24, arg validation bought −0.24, output classifier bought −0.08, and a "tone softener" layer bought −0.00. What do you do with the tone-softener layer?

- A) Keep it — every layer helps.
- B) **Strip it** — a defense that buys 0.00 ASR reduction is theater, adding cost and latency for no measured benefit. The ASR-per-layer table exists precisely to find and remove the layers that don't earn their place.
- C) Make it more aggressive.
- D) Move it to the front.

---

**Q13.** Your threat model concludes "our agent is secure against prompt injection." What's wrong with it?

- A) Nothing — that's the goal.
- B) **No agent is secure against injection** — it has no complete fix. The honest threat model names the *residual*: "ASR is 0.08; these two attacks still land; we accept/mitigate them because X." A zero-risk claim is security theater that makes a reviewer stop trusting the document.
- C) It should claim 100% security.
- D) It used the wrong filter.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — The model can't separate developer instructions from data instructions, and there's no parameterized-query equivalent. The structural root of injection. (Lecture 1 §1.)
2. **B** — Direct = attacker types it; indirect = attacker plants it in data the agent reads. (Lecture 1 §2.)
3. **B** — Indirect rides in through the trusted retrieval channel, affects whoever retrieves the poison, and is invisible at entry (the user's request is benign). (Lecture 1 §2.2.)
4. **B** — A tool turns "said something wrong" into "did something harmful." That's why the mantra is about tools. (Lecture 1 §1.)
5. **B** — Excessive Agency = blast radius. Least privilege shrinks it. (Lecture 1 §4.)
6. **B** — No complete fix → stack independent layers and measure each. (Lecture 2 §1.)
7. **B** — A speed bump: catches the obvious cheaply, misses obfuscation; never alone. (Lecture 2 §2.1.)
8. **B** — Deterministic and model-independent: it holds when the filter fails and the model is steered. (Lecture 2 §3.)
9. **B** — Output is untrusted too (insecure output handling): filter it for exfil/harmful content. (Lecture 2 §4.)
10. **B** — A DoS against your own users; measure both axes and tune to the knee. (Lecture 2 §2.3, §6.2; the challenge's second trap.)
11. **B** — A mechanically-checkable success criterion (canary, planted artifact); calibrated LLM-judge for fuzzy cases. (Lecture 2 §6.1; Exercise 1.)
12. **B** — Strip the zero-delta layer; the per-layer table exists to find theater. (Lecture 2 §6.3.)
13. **B** — No agent is injection-proof; name the residual. A zero-risk claim is theater. (Lecture 2 §6.4; the challenge trap.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
