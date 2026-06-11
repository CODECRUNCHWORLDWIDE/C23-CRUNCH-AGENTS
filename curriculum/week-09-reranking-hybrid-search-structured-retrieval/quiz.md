# Week 9 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 10. Answer key is at the bottom — don't peek.

---

**Q1.** Why does dense (embedding) retrieval struggle with a query like "clause 14" or "$1,000,000"?

- A) The embedding model crashes on numbers and identifiers.
- B) Those are exact surface strings; embeddings generalize away from surface form, so an identifier or exact amount gets smeared into the topical neighborhood rather than matched literally.
- C) pgvector cannot store vectors for numeric text.
- D) The query is too short to embed.

---

**Q2.** In BM25, what does the `k1` parameter control?

- A) The number of documents returned.
- B) Term-frequency saturation — how fast the contribution of additional occurrences of a term flattens out.
- C) The length-normalization strength.
- D) The IDF weighting.

---

**Q3.** In BM25, what does the `b` parameter control?

- A) Length normalization — how much a long document is penalized relative to the average length.
- B) The number of query terms considered.
- C) Term-frequency saturation.
- D) The cosine threshold.

---

**Q4.** What is the Reciprocal Rank Fusion score of a document, and what is the standard `k`?

- A) `sum over rankers of rank_r(d)`, with k=10.
- B) `sum over rankers of 1 / (k + rank_r(d))`, with rank 1-based and k≈60.
- C) `max over rankers of cosine_r(d)`, with k=1.
- D) `average of the normalized scores`, with k tuned per query.

---

**Q5.** Why is RRF preferred over weighted score normalization for combining a dense list and a BM25 list?

- A) RRF is faster to compute on a GPU.
- B) RRF fuses on rank, so it needs no score calibration — dense cosine (≈0–1) and unbounded BM25 scores live on incomparable scales, which makes weighted score fusion fragile and tuning-heavy.
- C) Score normalization always loses the right document.
- D) RRF uses the LLM to combine the lists.

---

**Q6.** What is the key architectural difference between a bi-encoder and a cross-encoder?

- A) A bi-encoder is bigger; a cross-encoder is smaller.
- B) A bi-encoder encodes query and document separately (so document vectors can be precomputed and indexed); a cross-encoder encodes the (query, document) pair together and outputs a relevance score, so nothing can be precomputed.
- C) A bi-encoder uses cosine; a cross-encoder uses Euclidean distance.
- D) They are the same model used at different temperatures.

---

**Q7.** Why do you apply a cross-encoder reranker only to the first-stage top-k (e.g. top 50 → top 5), not the whole corpus?

- A) Because the reranker can only read 50 documents total.
- B) Because the cross-encoder's cost is one forward pass per candidate (linear in candidates); running it over millions of documents per query is prohibitively slow, so you rerank only the small candidate set the first stage already surfaced.
- C) Because pgvector limits results to 50.
- D) Because the reranker requires exactly 50 inputs.

---

**Q8.** A reranker takes a candidate set where the right document is at first-stage rank 4 and moves it to rank 1. Which metric reflects this lift most?

- A) Recall@5 — it jumps from 0 to 1.
- B) MRR — the reciprocal rank of the right document goes from 1/4 to 1; Recall@5 was already 1.0 (it was in the top 5) and cannot improve.
- C) Embedding dimension.
- D) Index build time.

---

**Q9.** What does ColBERT's late-interaction MaxSim compute?

- A) The cosine between two single pooled vectors.
- B) For each query token, the maximum similarity to any document token, summed across query tokens — using per-token embeddings, so it sits between a bi-encoder and a cross-encoder on cost and quality.
- C) The BM25 score of the document.
- D) The mean of all token embeddings.

---

**Q10.** What is the core idea of HyDE (Hypothetical Document Embeddings)?

- A) Embed the query twice and average the vectors.
- B) Generate a hypothetical answer with an LLM, embed *that* (it's document-shaped, so it lands near the real answer), and retrieve with the hypothetical's vector — the hypothetical may be wrong, but you only use its embedding.
- C) Hash the query into a fixed bucket.
- D) Use the LLM to rerank the final results.

---

**Q11.** Your HyDE layer raises Recall@5 but lowers MRR on the gold set. What does that mean?

- A) HyDE is broken and should be deleted.
- B) The hypothetical answer pulled the right document into the top-5 (helping recall) but its hallucination also dragged a near-miss above the true answer (hurting the rank) — a precision cost you must report, not a free win.
- C) The gold set is wrong.
- D) The reranker undid HyDE's work.

---

**Q12.** A question is "which agreements expire before 2027?" Why can't dense + BM25 + reranking answer it, and what should you use?

- A) They can answer it; just add more candidates.
- B) The answer is a computation (filter/aggregate over structured date fields) that isn't written down in any clause to retrieve — it must be computed with SQL (text-to-SQL), not retrieved from a vector store.
- C) The reranker needs a bigger model.
- D) You should embed the year 2027 as a query.

---

**Q13.** Which is the *load-bearing* safety control when executing LLM-generated SQL?

- A) A polite system-prompt instruction telling the model to only write SELECT.
- B) A read-only database role granted SELECT only — so even if the model emits `DROP TABLE`, the engine rejects it regardless of application code; paired with parse-and-validate (single read-only SELECT) and a schema allowlist.
- C) Running the query twice and comparing.
- D) Asking the user to confirm the SQL.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — Identifiers and exact amounts are surface strings; embeddings are trained to generalize away from surface form, so the exact token gets smeared into a topical neighborhood instead of matched. This is the exact-match miss that motivates BM25. (Lecture 1 §1.1.)
2. **B** — `k1` controls term-frequency saturation: the curve `f·(k1+1)/(f+k1)` rises then flattens, and `k1` sets how fast. At `k1=0`, TF is ignored entirely (binary present/absent). (Lecture 1 §2.3.)
3. **A** — `b` controls length normalization via the `|d|/avgdl` ratio: `b=1.0` fully penalizes long documents, `b=0.0` ignores length. The Okapi default is `0.75`. (Lecture 1 §2.3.)
4. **B** — RRF score is `sum over rankers of 1/(k + rank_r(d))`, rank 1-based, k≈60 (Cormack et al. 2009). A missing doc contributes 0. (Lecture 1 §3.2–3.4.)
5. **B** — RRF fuses on rank, so it needs no score calibration. Dense cosine (≈0–1) and unbounded BM25 scores are on incomparable scales, making weighted score fusion fragile and tuning-heavy. Rank-based robustness is RRF's whole pitch. (Lecture 1 §3.1–3.2.)
6. **B** — A bi-encoder encodes query and document separately (document vectors precomputable, so it scales); a cross-encoder encodes the pair together and outputs a relevance score, so nothing precomputes — which is why it's expensive and used only for reranking. (Lecture 2 §1.)
7. **B** — The cross-encoder's cost is linear in candidates (one forward pass each), so you rerank only the small first-stage candidate set; running it over the corpus defeats the reason you have a first stage. (Lecture 2 §1.)
8. **B** — MRR weights the rank of the first relevant doc (1/rank), so moving the right doc from rank 4 to 1 takes RR from 0.25 to 1.0. Recall@5 was already 1.0 (it was in the top 5) and can't improve — the reranker's lift lives in MRR. (Lecture 2 §2; Exercise 3.)
9. **B** — ColBERT keeps per-token embeddings; MaxSim takes, for each query token, the max similarity to any document token, then sums. Token-level matching with precomputable document tokens puts it between a bi- and cross-encoder. (Lecture 2 §3.)
10. **B** — HyDE generates a hypothetical answer, embeds *that* (it's document-shaped, so it lands nearer the real answer than the query would), and retrieves with the hypothetical's vector. The hypothetical can be wrong — only its embedding is used. (Lecture 2 §5; Gao et al. 2022.)
11. **B** — Recall up + MRR down means HyDE pulled the right doc into the top-5 but its hallucination also raised a near-miss above the true answer. That precision cost is the trade-off you report, not hide — a layer helping one metric and hurting another is a decision. (Lecture 2 §5; Challenge 1 trap.)
12. **B** — The answer is a filter/aggregate over structured date fields — not written in any clause, so there's nothing to retrieve. It must be computed with SQL via text-to-SQL, the structured-retrieval path. (Lecture 2 §6.)
13. **B** — A read-only role is the load-bearing control: the database rejects writes regardless of application code or model output. The prompt instruction is the *weakest* control; the role, the parse-and-validate (single read-only SELECT), and the schema allowlist are what actually keep you safe. (Lecture 2 §6.1.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
