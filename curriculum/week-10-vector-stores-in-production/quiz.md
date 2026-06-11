# Week 10 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 11. Answer key is at the bottom — don't peek.

---

**Q1.** What's the single best reason pgvector is "the default" vector store for most teams?

- A) It has the fastest query latency of any store.
- B) It's a Postgres extension, so your team already operates it — backups, replication, monitoring, recovery, and transactional joins against your relational data come for free; lowest operational surprise.
- C) It's the only store that supports HNSW.
- D) It scales to billions of vectors better than anything else.

---

**Q2.** The week's mantra is "pick the store with the operational story you can live with at 2 AM, not the one with the best benchmark." What does this mean in practice?

- A) Always pick the slowest store.
- B) Weight operational criteria — recovery time, filtered-search recall, ingest, familiarity — above the leaderboard QPS, because the benchmark is the vendor's good day and the 2 AM story is your bad day.
- C) Never benchmark anything.
- D) Only pick stores that have a managed cloud tier.

---

**Q3.** Why is *filtered* ANN ("find the closest vectors WHERE tenant='acme'") harder than plain ANN?

- A) Filters aren't supported by any vector store.
- B) The ANN index is built over all vectors for global nearest-neighbor search; restricting to a metadata-matching subset must be reconciled with the index, and the naive reconciliations (post-filter, pre-filter) have failure modes.
- C) Metadata can't be stored alongside vectors.
- D) Filters require a separate database.

---

**Q4.** A naive post-filter (ANN over everything, then drop non-matches) returns *nothing* for a rare tenant even though that tenant has relevant vectors. Why?

- A) The tenant's data was deleted.
- B) The rare tenant's vectors aren't in the global ANN top-K candidates, so after filtering there's nothing left — the recall collapses silently on selective filters.
- C) Post-filtering is not allowed in production.
- D) The embedding model can't represent rare tenants.

---

**Q5.** What does Qdrant's "filterable HNSW" (native/in-filter) do that post-filter and pre-filter don't?

- A) It pre-computes all possible filters.
- B) It applies the metadata filter *during* the HNSW graph traversal, so it gets the index's speed *and* the filter's exactness — no recall collapse on selective filters, no brute-force scan on broad ones.
- C) It stores a separate index per tenant.
- D) It disables the vector index when filtering.

---

**Q6.** Why is a metadata (payload) index as important as the vector index for multi-tenant retrieval?

- A) It isn't; only the vector index matters.
- B) Because nearly every real query filters by something (tenant, document set, time range); without an index on the filtered field, the filter scans every row's metadata and degrades as the data grows.
- C) Because it replaces the need for backups.
- D) Because it makes embeddings smaller.

---

**Q7.** In the index-loss recovery drill, what is the headline number and why does it matter?

- A) Query latency; it's all that matters in production.
- B) Time-to-recover (from "index gone" to "Recall@5 back to baseline"); it can reorder the stores versus how query latency ranked them, because a store that queries faster but restores via a 4-hour re-embed is the worse production choice.
- C) Ingest throughput; it's the only operational metric.
- D) The number of replicas.

---

**Q8.** Why is "we can just re-embed the corpus" not a backup strategy?

- A) Re-embedding is illegal.
- B) Re-embedding a large corpus through an embedding model takes hours, during which retrieval is down — that's an outage, not a fast recovery; a real backup (snapshot, pg_dump) restores in seconds-to-minutes.
- C) Embeddings can't be regenerated.
- D) Re-embedding changes the gold set.

---

**Q9.** Which schema change to a vector store is the *expensive* one, and why?

- A) Adding a filterable metadata field — it requires re-embedding.
- B) Changing the embedding model or dimension — the vectors are different in the new model's space, so the whole index must be re-embedded and rebuilt from source; there's no shortcut.
- C) Renaming a collection — it rebuilds the HNSW graph.
- D) Adding a replica — it re-embeds the corpus.

---

**Q10.** What class of question does GraphRAG answer that flat vector retrieval structurally cannot?

- A) Simple factual lookups like "what's the confidentiality duration?"
- B) Multi-hop ("which clauses does the termination clause depend on?") and global/thematic ("what are the main themes across the corpus?") questions — answered by traversing an entity graph or combining community summaries, not by nearest-neighbor over chunks.
- C) Questions about a single sentence.
- D) Questions with typos.

---

**Q11.** How does GraphRAG (Microsoft, 2024) answer a *global* question that no single chunk contains?

- A) By embedding the whole corpus as one vector.
- B) By extracting an entity/relationship graph, clustering it into communities, summarizing each community with an LLM, and combining the community summaries (map-reduce) to answer corpus-wide questions.
- C) By increasing the chunk size to the whole document.
- D) By using a bigger embedding model.

---

**Q12.** What is "agentic RAG," and what's the honest trade-off?

- A) RAG that only runs on agents; it's always better.
- B) The agent chooses the retriever/store/strategy per query (or skips retrieval); it lifts results on *heterogeneous* query distributions but adds a routing call (latency, tokens, a new failure mode), so on homogeneous queries the fixed pipeline wins — measure the lift before shipping the router.
- C) RAG without a vector store.
- D) A way to avoid measuring retrieval.

---

**Q13.** In the store bakeoff, unfiltered Recall@5 comes out nearly identical across pgvector, Qdrant, and Weaviate. What does this tell you?

- A) The bakeoff is broken.
- B) The stores have comparable basic ANN quality on this corpus, so the decision is made on the *operational* axes — recovery time, filtered-recall at a selective filter, ingest, familiarity — not on basic recall.
- C) You should pick the cheapest store at random.
- D) One store must be cheating.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — pgvector is Postgres; your team already operates it. Familiarity + transactional joins = lowest operational surprise. (Lecture 1 §1, §2.1.)
2. **B** — Weight recovery/filtering/ingest/familiarity above leaderboard QPS; the benchmark is the good day, the 2 AM story is the bad one. (Lecture 1 §5; the mantra.)
3. **B** — The global ANN index must be reconciled with a subset restriction; the naive reconciliations have failure modes. (Lecture 1 §3.)
4. **B** — The rare tenant's vectors aren't in the global top-K, so post-filter drops everything — silent recall collapse on selective filters. (Lecture 1 §3.1; Exercise 2.)
5. **B** — Filtering *during* traversal gets index speed + filter exactness; no collapse, no brute-force scan. (Lecture 1 §3.3.)
6. **B** — Most real queries filter; without a metadata index the filter scans all rows and degrades with scale. (Lecture 1 §4.)
7. **B** — Time-to-recover; it reorders the stores because a fast-query/slow-restore store is the worse production choice. (Lecture 2 §2.)
8. **B** — Re-embedding is hours of downtime — an outage, not a backup; a snapshot/pg_dump restores fast. (Lecture 2 §1.1, §2.)
9. **B** — Changing the embedding/dimension means the vectors are different; full re-embed + rebuild from source. (Lecture 2 §1.3.)
10. **B** — Multi-hop and global/thematic questions, via graph traversal and community summaries. (Lecture 2 §3.)
11. **B** — Entity graph → communities → LLM summaries → combine summaries (map-reduce) for corpus-wide answers. (Lecture 2 §3; arXiv 2404.16130.)
12. **B** — The agent routes per query; lifts heterogeneous distributions, costs a routing call; measure the lift first. (Lecture 2 §4.)
13. **B** — Comparable ANN quality → the decision is operational (recovery, filtered-recall, familiarity), not basic recall. (Lecture 1 §6; Lecture 2 §5.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
