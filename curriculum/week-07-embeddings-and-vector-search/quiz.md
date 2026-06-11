# Week 7 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 8. Answer key is at the bottom — don't peek.

---

**Q1.** What is an embedding, at a systems level?

- A) The raw token IDs produced by the tokenizer.
- B) A fixed-length dense vector that represents a piece of text's meaning, produced by an encoder transformer with a pooling step.
- C) The probability distribution over the next token.
- D) A hash of the document used for deduplication.

---

**Q2.** Your embeddings are L2-normalized to unit length. How do cosine similarity, dot product, and Euclidean distance compare for *ranking* results?

- A) They produce three different rankings, so the metric choice is critical.
- B) They rank identically; on unit vectors cosine = dot, and L2 distance is a monotonic function of cosine.
- C) Only cosine works; dot and L2 are undefined for normalized vectors.
- D) Dot product ranks the reverse of cosine.

---

**Q3.** You embed your documents with BGE but forget the query instruction prefix on your *queries*. What happens?

- A) The program crashes with a dimension-mismatch error.
- B) Recall silently drops — no error, just worse results — because you violated BGE's training convention for queries.
- C) The vectors come out the wrong dimension and the index rejects them.
- D) Nothing; the prefix is cosmetic.

---

**Q4.** When reading the MTEB leaderboard to pick a model for RAG, which column should you sort by?

- A) The overall MTEB average across all task types.
- B) The Clustering task, since RAG groups similar documents.
- C) The Retrieval task (e.g. nDCG@10), since that's what RAG retrieval is.
- D) The Classification task, since you classify queries.

---

**Q5.** Why is `ef_search` the single most important *runtime* knob for an HNSW index?

- A) It changes the embedding dimension at query time.
- B) It controls how many candidates the search explores per query, directly trading recall for latency, and can be set per query without rebuilding.
- C) It is the only knob that affects build time.
- D) It selects which similarity metric the index uses.

---

**Q6.** You build a pgvector index with `vector_l2_ops` but write your queries with the `<=>` (cosine) operator. What happens?

- A) Postgres returns wrong results with no warning.
- B) The query errors out immediately.
- C) Postgres can't use the index for that operator and falls back to a full sequential scan — correct results, mysteriously slow.
- D) The cosine operator is automatically converted to L2.

---

**Q7.** What does Recall@5 of 0.85 on a single-relevant-doc gold set mean?

- A) 85% of the retrieved documents are relevant.
- B) For 85% of queries, the relevant document appeared somewhere in the top 5 results.
- C) The fifth result is relevant 85% of the time.
- D) The model scored 0.85 on MTEB.

---

**Q8.** MRR (Mean Reciprocal Rank) rewards what, that Recall@5 does not?

- A) Finding more total documents.
- B) Putting the first relevant document *higher* in the ranking — rank 1 scores 1.0, rank 5 scores 0.2.
- C) Using a larger embedding dimension.
- D) Lower query latency.

---

**Q9.** You want to switch your production system from a 1024-dim embedding to a 768-dim one. What does that actually require?

- A) A one-line config change; pgvector auto-migrates.
- B) Nothing; vectors of different dimensions can share a column.
- C) A new column/table sized to 768, a new index, and re-embedding the entire corpus — a redeploy-everything event.
- D) Only re-embedding the queries, not the documents.

---

**Q10.** Why is `BEST_EFFORT`... wait — why is *brute-force* (exact) search the right tool for computing the *ground truth* when measuring ANN recall?

- A) Because it's the fastest.
- B) Because it returns the true nearest neighbours every time, so you have something exact to compare your approximate index against.
- C) Because it uses less memory than HNSW.
- D) Because it scales to billions of vectors.

---

**Q11.** The "negation failure" of dense embeddings means:

- A) Negative numbers can't be stored in a vector.
- B) "The tenant must pay rent" and "The tenant must not pay rent" often embed *close together* despite being opposites, because the negation is a tiny part of the shared meaning.
- C) Cosine similarity returns negative values for negated sentences.
- D) The model refuses to embed sentences containing "not".

---

**Q12.** Reading an HNSW recall-vs-`ef_search` curve, where is the right operating point?

- A) The maximum `ef_search`, for maximum recall regardless of latency.
- B) The minimum `ef_search`, for minimum latency regardless of recall.
- C) The "elbow" — the `ef_search` where recall is already high (~0.97+) but latency is still small; spending compute past it buys latency for no recall.
- D) Exactly `ef_search = 100`, always.

---

**Q13.** Your retrieval Recall@5 is much lower than the embedding's MTEB card promised. Per the Lecture 2 debugging tree, what should you check *first*?

- A) Switch to a bigger embedding model.
- B) Whether queries were embedded with the right prefix/`input_type` and whether all vectors are normalized the same way — the query/document asymmetry and normalization are the most common silent causes.
- C) Rebuild the entire corpus from scratch.
- D) Increase `max_tokens` on the LLM.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — An embedding is a fixed-length dense vector from an encoder transformer plus pooling (and usually normalization). It's not token IDs (the tokenizer's output) and not a next-token distribution (the decoder's output). (Lecture 1 §1.)
2. **B** — On unit vectors, cosine = dot product, and `||a−b||² = 2 − 2(a·b)`, so L2 distance is a monotonic function of cosine. All three rank identically — which is why the metric choice barely matters on normalized embeddings. (Lecture 1 §4.)
3. **B** — The prefix is a training convention; dropping it on queries is a silent recall loss, not an error. The single most common "my retrieval is mysteriously bad" bug. (Lecture 1 §2, §7.)
4. **C** — RAG is a retrieval task; sort by the Retrieval column (nDCG@10), not the blended average, which can be dominated by clustering/classification. (Lecture 1 §5.)
5. **B** — `ef_search` controls per-query candidate exploration, trades recall for latency directly, and is settable per query with no rebuild. The runtime dial of the iron triangle. (Lecture 2 §1.2.)
6. **C** — A mismatched index op means Postgres can't use the index for that operator, so it does a full scan: correct results, slow. The pgvector gotcha. (Lecture 2 §2.2.)
7. **B** — Recall@5 with one relevant doc = fraction of queries whose relevant doc is in the top 5. It's a property of queries, not of the retrieved set's purity. (Lecture 2 §3.2.)
8. **B** — MRR weights *rank* of the first relevant doc (1/rank), rewarding putting the answer first — which matters because LLMs weight the top of context more. Recall@5 only cares whether it's in the top 5 at all. (Lecture 2 §3.2.)
9. **C** — Dimension is locked into the column and index; changing it means new column/index plus re-embedding everything. There is no migration shortcut. (Lecture 1 §7.)
10. **B** — Brute force is exact, so it's the ground truth you measure your approximate index's recall against. It's slow and doesn't scale, which is *why* you use ANN in production and brute force only for the measurement. (Lecture 2 §1.1, §3.1, Exercise 3.)
11. **B** — Dense embeddings capture topical similarity well and logical structure (negation) poorly, so opposites embed close. This is the blind spot week 9's hybrid/lexical search fixes. (Lecture 1 §1.)
12. **C** — The elbow: recall has plateaued but latency is still low. Past it you pay latency for no recall gain; before it you sacrifice recall. (Lecture 2 §1.1; Exercise 3.)
13. **B** — Per the debugging tree, prefix/`input_type` asymmetry and normalization are the first, most common, silent causes — check them before reaching for a different model. The embedding usually isn't the problem. (Lecture 2 §4; Lecture 1 §7.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
