# Week 12 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 13 (and Phase III). Answer key is at the bottom — don't peek.

---

**Q1.** What does Ragas's **faithfulness** metric measure, and what failure does it uniquely catch?

- A) Whether the right chunk was retrieved; it catches retrieval misses.
- B) Whether every claim in the *answer* is supported by the *retrieved context*; it catches hallucination (the model asserting things the context never said).
- C) Whether the answer is grammatically correct; it catches typos.
- D) Whether the embedding model is normalized; it catches index bugs.

---

**Q2.** **Context recall** and **context precision** are a complementary pair. State the difference correctly.

- A) They are the same metric computed two ways.
- B) Context recall asks "did retrieval bring back *all* the context the reference answer needs?" (a miss problem); context precision asks "are the retrieved chunks *relevant and ranked high*, or padded with junk?" (a noise problem).
- C) Context recall measures answer length; context precision measures answer speed.
- D) Context recall is for text; context precision is for images.

---

**Q3.** **Answer relevancy** catches a failure the other three metrics miss. Which one?

- A) A hallucinated claim.
- B) A retriever that missed a chunk.
- C) An answer that is faithful and well-grounded but *off-topic* — it answers a different question than the one asked.
- D) A slow generator.

---

**Q4.** Why does Ragas report **four** metrics instead of one collapsed "quality score"?

- A) Because four numbers look more rigorous.
- B) Because the four metrics are *orthogonal* — each localizes a different failure stage (hallucination vs. retrieval miss vs. retrieval noise vs. off-topic) — and a single score hides *which stage* to fix.
- C) Because Ragas can't compute an average.
- D) Because each metric needs a different embedding model.

---

**Q5.** You are wiring Claude as the LLM-as-judge. Which call is correct for `claude-opus-4-8`?

- A) `messages.create(model="claude-opus-4-8-20260101", temperature=0.0, top_p=0.9, ...)`
- B) `messages.parse(model="claude-opus-4-8", thinking={"type":"adaptive"}, output_config={"effort":"high"}, output_format=PydanticModel, ...)` — no temperature/top_p, no date suffix, structured output for a parseable verdict.
- C) `messages.create(model="opus", budget_tokens=4000, temperature=0.7, ...)`
- D) `completions.create(model="claude-opus-4-8", max_tokens=100)`

---

**Q6.** Why do you **calibrate** an LLM-as-judge against human labels before trusting its scores?

- A) Calibration makes the judge faster.
- B) An uncalibrated judge may emit confident decimals while agreeing with humans only at chance (kappa ≈ 0); calibration against ~10 human labels measures the agreement so you know whether the metric means anything.
- C) Calibration is required by the Ragas API.
- D) It reduces the token cost of the judge.

---

**Q7.** Your judge agrees with the human labels on 9 of 10 examples (90% raw agreement), but 9 of those 10 humans labeled "faithful." Why might **Cohen's kappa** still be low?

- A) Kappa is always low; ignore it.
- B) Because raw agreement is inflated by *chance*: a judge that always says "faithful" agrees 90% of the time on a 90%-faithful set while discriminating nothing — kappa corrects for that chance agreement, so it can be near zero.
- C) Because kappa only works on more than 100 examples.
- D) Because the judge used the wrong model.

---

**Q8.** In the judge-calibration sweep, what is the **threshold τ**, and how do you pick it?

- A) τ is the temperature; you set it to 0.
- B) τ is the score cutoff above which the judge's continuous score counts as "pass"; you sweep it and pick the τ that *maximizes Cohen's kappa* against the human labels.
- C) τ is the number of retrieved chunks; you set it to 5.
- D) τ is fixed at 0.5 by the Ragas spec.

---

**Q9.** What is **self-preference bias** in an LLM-as-judge, and what's the fix?

- A) The judge prefers shorter answers; lower max_tokens.
- B) The judge rates answers from its *own model family* higher; the fix is structural — never use the same model as both generator and judge.
- C) The judge prefers the first option; reverse the order.
- D) The judge prefers cheaper models; use a more expensive one.

---

**Q10.** A **VLM** (Qwen2.5-VL, Claude vision) and a **CLIP/SigLIP** embedding are two different multimodal tools. State the difference.

- A) They're the same thing with different names.
- B) A VLM *reads* an image to generate text (e.g. "the insurance amount is $1,000,000"); CLIP/SigLIP *embeds* an image into a shared space so a text query can *retrieve* it — but CLIP does not read fine detail off a chart.
- C) CLIP generates text; VLMs only embed.
- D) VLMs work on text only; CLIP works on images only.

---

**Q11.** Your corpus is a financial filing where "2025 revenue" is answered only by a bar chart. You have two architectures: (A) extract+describe the figure into a text chunk, or (B) embed the page image and answer with a VLM. What determines which wins?

- A) Option A always wins because text retrieval is better.
- B) It depends on the corpus and you *measure* both: A's success hinges on whether the index-time description captured the number; B's hinges on whether CLIP/ColPali retrieved the right page for the VLM to read. Different failure surfaces — let the eval decide.
- C) Option B always wins because images are richer.
- D) Neither works; charts can't be retrieved.

---

**Q12.** Where do **ASR (Whisper)** and **TTS (Piper/XTTS)** sit relative to a RAG pipeline?

- A) They *are* the RAG pipeline.
- B) They're pipeline-*adjacent adapters*: ASR turns audio into the text your RAG indexes (input edge); TTS turns the RAG's answer into speech (output edge). You evaluate the *text* in the middle, not the audio.
- C) They replace the embedding model.
- D) They are image-generation tools.

---

**Q13.** In the three-variant Ragas report (baseline / +reranker / +hybrid), you hold the generator constant and change only the retriever between variants. Why?

- A) To save money on the generator.
- B) So a metric delta between variants is attributable to the *retrieval change alone* — if you also changed the generator or prompt, "which metric moved for which change" becomes unanswerable (two changes at once).
- C) Because the generator doesn't affect the metrics.
- D) Because Ragas requires identical generators.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Faithfulness = fraction of the answer's claims supported by the context; it catches hallucination, which a retrieval metric can't see. (Lecture 2 Part 1.1.)
2. **B** — Recall = "did we get *all* the right context?" (miss); precision = "is the context clean and ranked?" (noise). Complementary. (Lecture 2 Part 1.2–1.3.)
3. **C** — Answer relevancy catches the faithful-but-off-topic answer (e.g. answering with the payment schedule when asked about confidentiality). (Lecture 2 Part 1.4.)
4. **B** — The four metrics are orthogonal failure detectors; a collapsed score hides which stage failed and is therefore not actionable. (Lecture 2 Part 1, the four-metric mental model.)
5. **B** — `claude-opus-4-8`, adaptive thinking + `output_config={"effort":"high"}`, structured output via `messages.parse`, no temperature/top_p/top_k, no date suffix. (Lecture 2 Part 3.2.)
6. **B** — An uncalibrated judge can emit confident decimals at chance-level agreement; you calibrate against ~10 human labels to know the metric means something. (Lecture 2 Part 4.1.)
7. **B** — Raw agreement is inflated by chance under class imbalance; Cohen's kappa subtracts the chance agreement, exposing that an always-"faithful" judge discriminates nothing. (Lecture 2 Part 4.2; Exercise 3 "the trap".)
8. **B** — τ is the continuous-score cutoff for "pass"; you sweep it and pick the kappa-maximizing value, then report the metric *at* that τ. (Lecture 2 Part 4.3; Exercise 3.)
9. **B** — Self-preference (self-enhancement) bias inflates own-family scores; the structural fix is judge ≠ generator. (Lecture 2 Part 5; challenge "the trap".)
10. **B** — A VLM reads an image to text; CLIP/SigLIP embed for retrieval and do not read fine chart detail. Two tools: one reads, one finds. (Lecture 1 §1–2.)
11. **B** — It's corpus-dependent with different failure surfaces; you measure both architectures with the eval rather than assuming. (Lecture 1 §3.)
12. **B** — ASR and TTS are adapters on the input/output edges; you evaluate the text in the middle, not the audio. (Lecture 1 §5.)
13. **B** — One variable at a time: holding the generator fixed makes a metric delta attributable to the retrieval change, which is the entire "which metric moved for which change" finding. (Challenge "the trap"; Lecture 2 Part 2.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md) and the Phase II milestone report.
