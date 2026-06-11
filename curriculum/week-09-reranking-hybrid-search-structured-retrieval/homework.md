# Week 9 Homework

Six problems that revisit the week's topics and force the layered-retrieval discipline into your fingers. The full set should take about **5 hours**. Work in your Week 9 Git repository (the same workspace as the exercises and the `crunchrag_hybrid` mini-project) so every problem produces at least one commit you can point to at the Phase II architecture review in Week 12.

The headline deliverable is **Problem 4 — the cumulative-lift chart and memo**, called out explicitly in the syllabus ("Chart the cumulative lift on the same 40-query set"). Treat it as the artifact a reviewer reads, not a journal entry.

Each problem includes a short **problem statement**, **acceptance criteria** so you know when you're done, a **hint** if you get stuck, and an **estimated time**.

Have pgvector running (`docker run -d --name crunch-pg -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17`) and your venv active with `rank-bm25`, `sentence-transformers`, `psycopg[binary]`, and `numpy` installed. Problems run against the legal corpus and 40-query gold set from your week-7/8 work; the reranker download (`BAAI/bge-reranker-v2-m3`, ~600 MB) should be done before you start Problem 3.

---

## Problem 1 — BM25 wins where dense lost

**Problem statement.** Build a `rank-bm25` index over the legal corpus (with the doc-id trick from Exercise 1). Identify the five queries in the 40-query gold set where **dense retrieval misses but BM25 hits** (an identifier, "Delaware", "$1,000,000", and two others you find), and the five where **dense hits but BM25 misses** (paraphrase queries). Tabulate both sets at `notes/week-09/bm25-vs-dense.md`.

**Acceptance criteria.**

- `notes/week-09/bm25-vs-dense.md` lists the five BM25-wins-dense-loses queries and the five dense-wins-BM25-loses queries, each with the gold `doc_id` and each retriever's top-1.
- Every result comes from a real run (BM25 from `rank-bm25`, dense from your week-7 pgvector pipeline), not from memory.
- You state, in one sentence, the pattern: which *kind* of query goes to which retriever.
- Committed.

**Hint.** The BM25 winners are the queries that hinge on an exact token (id, jurisdiction, money). The dense winners are paraphrases sharing no words with the answer. If you can't find five of each, your gold set is too easy — add a couple of harder paraphrase queries.

**Estimated time.** 40 minutes.

---

## Problem 2 — RRF, and the k-sensitivity check

**Problem statement.** Implement `rrf_fuse` (or import it from your mini-project), fuse your dense and BM25 ranked lists over the 40-query gold set, and measure hybrid Recall@5 and MRR. Then **sweep `k` ∈ {10, 30, 60, 100}** and record how little Recall@5 moves. Tabulate at `notes/week-09/rrf-sweep.md`.

**Acceptance criteria.**

- `notes/week-09/rrf-sweep.md` shows hybrid Recall@5 and MRR at the four `k` values.
- The numbers show Recall@5 barely moving across `k` (a spread of a point or two).
- You state, in one sentence, why the *insensitivity* to `k` is a feature, not a bug (rank-based fusion is robust; if it were wildly k-sensitive you'd suspect a 0-based-rank bug).
- Committed.

**Hint.** If Recall@5 swings wildly with `k`, you have a bug — most likely a 0-based rank (use `enumerate(..., start=1)`) or you fused score-normalized lists instead of rank lists. The whole point of RRF is that `k` barely matters in the 30–100 range.

**Estimated time.** 45 minutes.

---

## Problem 3 — The reranker's MRR lift

**Problem statement.** Take your hybrid top-50 candidate set and rerank it with `BAAI/bge-reranker-v2-m3` to a top-5, over the 40-query gold set. Measure Recall@5 and MRR **before** (hybrid) and **after** (hybrid + reranker). Then find and report the 3–4 specific queries the reranker *flipped* from a poor rank to rank 1. Write it up at `notes/week-09/reranker-lift.md`.

**Acceptance criteria.**

- `notes/week-09/reranker-lift.md` shows hybrid vs hybrid+reranker Recall@5 and MRR, with the actual numbers.
- You observe that MRR moves more than Recall@5 (the reranker reorders; it rarely adds new docs to the top-5).
- You list the 3–4 queries the reranker pulled to rank 1, with the clause it rescued.
- You confirm the reranker only saw the first-stage top-50, never the whole corpus.
- Committed.

**Hint.** Rerank a *generous* candidate set (50), not the hybrid top-5 — the reranker can only reorder docs it sees, so a narrow candidate set caps its lift. The flipped queries are usually the ones where a topical distractor (sharing a word like "termination") had outranked the true answer.

**Estimated time.** 50 minutes.

---

## Problem 4 — The cumulative-lift chart and memo (headline deliverable)

**Problem statement.** This is the syllabus deliverable. Run the full stack — `bm25` → `dense` → `hybrid+RRF` → `+reranker` → `+HyDE` — over the legal corpus on the 40-query gold set, measuring each layer with the **same** week-7 `evaluate()`. Produce the cumulative-lift table and a one-page memo at `notes/week-09/cumulative-lift-memo.md` against this template:

1. **The table** — five rows (one per layer), columns top-1 / Recall@5 / MRR / Δ Recall@5, every number from a real `evaluate()` run.
2. **The biggest lift** — one sentence: which layer added the most Recall@5, and why.
3. **The cheapest meaningful lift** — one sentence: which layer gave the most lift per unit of added cost/latency (name it — usually the reranker).
4. **The honest row** — one sentence on the layer that added little or nothing (a +0.00 HyDE row is a *result*; say so).
5. **Ship / cut** — one sentence: which layers you'd deploy and which you'd cut, with the deciding reason.

**Acceptance criteria.**

- `notes/week-09/cumulative-lift-memo.md` exists, fits on roughly one page (350–550 words), and hits all five sections.
- Every metric in the table is **computed** from a real `evaluate()` run on the fixed 40-query gold set, with `evaluate()` imported from week 7 unchanged.
- The reranker is applied only to the first-stage top-k.
- The memo names a *specific* cheapest-lift layer and a *specific* ship/cut decision — not "everything helped."
- You are honest about HyDE: report its actual delta, even if it's +0.00 or negative on this corpus.
- Committed.

**Hint.** The strongest memos *cut* a layer. If your HyDE row is +0.00, the senior move is to say "we cut HyDE — it added latency and no recall on this corpus" rather than keeping it because it's a "best practice." Naming the layer you removed is more convincing than listing the ones you kept.

**Estimated time.** 1 hour.

---

## Problem 5 — HyDE: where it helps and where it hurts

**Problem statement.** Implement HyDE (Lecture 2 §5): generate a hypothetical answer with an LLM, embed it as a document, retrieve with that vector. Run it on the 40-query gold set and find **one query where HyDE helps** (the hypothetical lands near the right clause) and **one where it hurts** (the hypothetical hallucinates off-topic and drags retrieval the wrong way). Document both at `notes/week-09/hyde-cases.md`, including the actual hypothetical text the LLM generated.

**Acceptance criteria.**

- `notes/week-09/hyde-cases.md` shows one help case and one hurt case, each with: the query, the LLM's hypothetical answer, the retrieved top-3 with and without HyDE, and the gold doc.
- You explain, in two sentences, *why* HyDE helped one and hurt the other (document-shaped hypothetical lands near the answer vs. a hallucinated topic drags the vector off).
- You note that the hypothetical is embedded as a *document* (no BGE query prefix) and *why*.
- Committed.

**Hint.** HyDE helps most on short, keyword-poor paraphrase queries the LLM can flesh out into a plausible clause. It hurts when the query is ambiguous and the LLM confidently writes a clause about the *wrong* subject. Use `claude-opus-4-8` with `thinking={"type": "adaptive"}` for the generation, and a system prompt that asks for a single declarative clause.

**Estimated time.** 50 minutes.

---

## Problem 6 — A safe text-to-SQL leg

**Problem statement.** Stand up a tiny SQLite database of the same contracts (an `agreements` table with `party`, `start_date`, `end_date`, `annual_fee_cents`, `governing_law`). Implement text-to-SQL for one structured question the vector store can't answer (e.g. "which agreements are governed by Delaware law?" or "what is the total annual fee?"). Lock it down per Lecture 2 §6: generate the SQL, **validate it is a single read-only SELECT**, and execute it. Document the safety surface at `notes/week-09/text2sql.md`.

**Acceptance criteria.**

- `notes/week-09/text2sql.md` shows the schema, the question, the generated SQL, and the result rows.
- Your code **parses and validates** the generated SQL — rejecting anything that isn't a single SELECT, and rejecting DML/DDL keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.) — *before* executing.
- You explain, in two sentences, the full safety surface: read-only role (or, in SQLite, a read-only connection), parse-and-validate, schema allowlist — and why the prompt instruction alone is the weakest control.
- You demonstrate the validator *rejecting* a malicious generation (feed it a hand-written `DROP TABLE agreements;` and show it's blocked).
- Committed.

**Hint.** Use `sqlglot` to parse: confirm exactly one statement and that its top-level kind is `SELECT`, then reject banned keywords. SQLite has no `GRANT`, so the role equivalent is opening the connection read-only (e.g. a URI with `mode=ro`). The point is layered defense — the validator and the read-only connection together, not the prompt.

**Estimated time.** 45 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — BM25 wins where dense lost | 40 min |
| 2 — RRF + k-sensitivity | 45 min |
| 3 — Reranker MRR lift | 50 min |
| 4 — Cumulative-lift chart + memo (headline) | 1 h 0 min |
| 5 — HyDE help/hurt cases | 50 min |
| 6 — Safe text-to-SQL leg | 45 min |
| **Total** | **~4 h 50 min** |

When you've finished all six, push your repo and make sure the `crunchrag_hybrid` [mini-project](./mini-project/README.md) is in the same workspace — Week 10 imports the layered retriever. Then take the [quiz](./quiz.md) with your notes closed.

---

## Grading rubric (homework)

Homework is graded on the same four axes as the weekly mini-project, scaled to the problem set.

| Axis | Weight | What "meets" looks like |
|---|---:|---|
| **Correctness** | 30% | RRF is correct (1-based, k=60); the reranker sees only the first-stage top-k; the text-to-SQL validator actually blocks `DROP`; the metrics are computed correctly. |
| **Engineering quality** | 25% | The week-7 `evaluate()` is reused unchanged across all layers; the gold set is fixed; no convention leaks; sensible commits. |
| **Measurement** | 25% | Every claim is backed by a number from a real run — the lift table, the k-sweep, the reranker delta, the HyDE cases. Vibes do not count. |
| **Write-up** | 20% | The memo (Problem 4) hits all five sections, names a *specific* cheapest-lift layer and ship/cut decision, and is honest about HyDE. The other notes are clear and reproducible. |

Graders are instructed to **fail vibes-only submissions** — a lift chart with no measured metrics, a memo that says "the reranker felt better," or a text-to-SQL leg with no validator, is not a "meets." The whole week is about replacing intuition with numbers and replacing "trust the model's SQL" with layered defense; the homework is graded the same way.
