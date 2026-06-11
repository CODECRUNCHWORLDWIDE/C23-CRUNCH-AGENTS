# Week 8 Homework

Six problems that revisit the week's topics and force chunking literacy into your fingers. The full set should take about **5 hours**. Work in your Week 8 Git repository (the same workspace as the exercises and the `crunchrag_chunk` mini-project) so every problem produces at least one commit you can point to at the Week 12 architecture review.

The headline deliverable is **Problem 4 — the one-page chunking A/B memo**, called out explicitly in the syllabus. Treat it as the artifact a reviewer reads, not a journal entry.

Have your **week-7 `crunchrag_embed` package** importable (`evaluate()` and `store.py` are reused unchanged) and pgvector running (`docker run -d -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17`). If week 7 is broken, fix it first — this week depends on it.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

---

## Problem 1 — Extract one real PDF and prove the answer survived

**Problem statement.** Take a real PDF (a contract, a paper, or the generated `services_agreement.pdf` from Exercise 1). Extract it with **PyMuPDF**, then with **one** other tool (Unstructured *or* an OCR path), and produce `notes/week-08/extraction.md` containing: the raw extracted text from each tool, and a one-paragraph comparison of what each got right and wrong. The required check: confirm a specific known answer (e.g. "five years after termination", or any fact you can verify) is present and *ungarbled* in the extraction you'll feed downstream.

**Acceptance criteria.**

- `notes/week-08/extraction.md` exists with raw output from two extraction paths.
- A named, verifiable fact from the document is shown present and ungarbled in the chosen extraction.
- One paragraph comparing the two tools, ending in which one you'd feed to the chunker and why.
- Committed.

**Hint.** If `page.get_text("text")` is empty, your PDF is scanned — switch to the OCR path (render at 300 DPI, then `pytesseract.image_to_string`). Don't OCR a native-text PDF; you'd degrade good text. Look at the text with your eyes before trusting it.

**Estimated time.** 40 minutes.

---

## Problem 2 — Implement and unit-test the recursive chunker

**Problem statement.** In your `crunchrag_chunk` package, implement `RecursiveChunker` (TODO 2 in `chunkers.py`): split on `["\n\n", "\n", ". ", " ", ""]` with length measured by the BGE tokenizer, recursing into pieces still over `size`. Write `tests/test_chunkers.py` proving that (a) no chunk exceeds the token budget, and (b) a paragraph-sized clause stays *whole* in one chunk at a size larger than the clause.

**Acceptance criteria.**

- `RecursiveChunker` is implemented and importable via `chunkers.load("recursive")`.
- `pytest tests/test_chunkers.py` passes with at least two assertions: max-chunk-size respected, and a known clause survives whole.
- Chunk size is measured in **BGE tokens**, not characters (`n_tokens()`), demonstrated in the test.
- Committed.

**Hint.** Port your Exercise 2 `recursive_chunks` and wrap each piece in a `Chunk`. For the "clause stays whole" test, use a clause that's a single paragraph and a `size` comfortably above its token count — then assert the clause text appears in exactly one chunk (normalize whitespace before matching).

**Estimated time.** 45 minutes.

---

## Problem 3 — The chunk-size sweep on the real gold set

**Problem statement.** Run the chunk-size sweep (Exercise 3's logic, or your `crunchrag_chunk` harness) against the **full 40-query legal gold set** from week 7, sweeping at least five chunk sizes (e.g. 64/128/256/512/1024). Produce `notes/week-08/size-sweep.md` with the Recall@5-and-MRR-vs-size table and a sentence naming the peak and the size you'd ship.

**Acceptance criteria.**

- A table of Recall@5 and MRR for at least five chunk sizes against the 40-query gold set.
- The curve's peak is identified, and you state the chunk size you'd ship *and the number that justifies it*.
- You note whether the peak is interior (a real sweet spot) or at an edge (sweep wider), per Lecture 1 §7.
- Committed.

**Hint.** Reuse `evaluate()` from `crunchrag_embed` unchanged — pass it a `retrieve_fn` that chunks at the swept size. If the peak is at an edge of your sweep, widen the range; the *shape* (a peak) is what you're proving. Keep the embedding fixed (BGE) so size is the only variable.

**Estimated time.** 50 minutes.

---

## Problem 4 — The one-page chunking A/B memo (headline deliverable)

**Problem statement.** This is the syllabus deliverable. Run the four-strategy A/B from Challenge 1 (token-window 512, token-window 1024, recursive, late chunking) against the fixed embedding (BGE; jina-v3 for late) and fixed store (pgvector) on the 40-query gold set. Write a **one-page** memo at `notes/week-08/chunking-ab-memo.md` against this template:

1. **Decision** — one sentence: which chunking strategy you ship, and its headline number.
2. **The table** — the four strategies with Recall@5, MRR, faithfulness, chunk count, and which model each used.
3. **Why this winner, on this corpus** — the mechanism (e.g. "recursive keeps each clause whole because clauses are paragraphs; fixed-512 splits ~X%"), not a general claim.
4. **The trade-off accepted** — what you gave up (cost, complexity, a long-context model) for the win.
5. **The late-chunking caveat** — late chunking's number includes a model swap (jina-v3), so its result is "late chunking *with jina-v3*," not late chunking in isolation; state how you'd isolate it (an early-vs-late control at the same model).
6. **One per-query trace** — in the promise format: `q12 ("five-year confidentiality") -> clause_09 (rank 1) ✓` for the winner, plus one query where a losing strategy split or buried the answer.

**Acceptance criteria.**

- `notes/week-08/chunking-ab-memo.md` exists, fits roughly one page (350–550 words), and hits all six headings.
- The store, gold set, and (for non-late strategies) embedding are demonstrably identical across runs — only the chunker varies.
- The winner is justified by a **specific** mechanism on this corpus, not "it felt better."
- The late-chunking model-swap caveat is stated honestly.
- At least one per-query trace in the promise format.
- Committed.

**Hint.** The chunk→clause mapping is load-bearing: gold is in clause ids, you retrieve chunks, so a strategy "finds clause 9" iff one of its chunks ranks in top-k. Skip the mapping and your table is meaningless (the challenge's trap). Generate faithfulness only for the top two strategies to keep it cheap; the retrieval metrics are the spine.

**Estimated time.** 1 hour.

---

## Problem 5 — Handle a table without smearing it

**Problem statement.** Find or create a PDF page with a table (a fee schedule, a parameter list). Extract the table *as a table* (`page.find_tables()` via PyMuPDF, or `pdfplumber.extract_tables()`), serialize it to Markdown, and make it **one chunk**. Then deliberately do it wrong — flatten the table to plain text and let your chunker smear it — and embed both. Show, with a query against the table's content (e.g. "what is the insurance amount"), that the structure-preserving chunk retrieves better.

**Acceptance criteria.**

- A table extracted as structured cells and serialized to Markdown as one chunk, in `notes/week-08/tables.md`.
- The flattened-and-smeared version for contrast.
- A query against the table's content showing the structure-preserving chunk ranks higher (with the similarity scores).
- One sentence on why header→value structure matters to the embedding.
- Committed.

**Hint.** Serialize as `| header | header |` / `| value | value |` so the embedding sees "Insurance | $1,000,000" as a unit, not "Insurance ... [200 other words] ... $1,000,000". Keep the table's caption/heading as metadata. This is Lecture 2 §3.1, measured.

**Estimated time.** 45 minutes.

---

## Problem 6 — Metadata injection: does the heading move recall?

**Problem statement.** Take your winning chunker from Problem 4. For each chunk, prepend its section heading (e.g. "Termination", "Confidentiality") to the chunk text *before* embedding. Re-run `evaluate()` and compare Recall@5/MRR with and without the heading injection. Record whether the heading helped, hurt, or did nothing — with the numbers.

**Acceptance criteria.**

- A before/after Recall@5 and MRR comparison (heading-injected vs bare chunks), in `notes/week-08/metadata-injection.md`.
- A one-sentence conclusion: did injecting the heading move retrieval on this corpus, and by how much?
- The embedding and store are unchanged; only the chunk *text* (heading prepended) varies — one variable.
- Committed.

**Hint.** The heading carries a word the clause body may lack — a clause about ending the contract that never says "termination" still retrieves better if "Termination" is prepended. But measure it; on a corpus where clause bodies already contain their topic words, the lift may be zero. Either result is a valid finding *if you have the number*.

**Estimated time.** 40 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Extract one PDF, prove the answer survived | 40 min |
| 2 — Implement + test the recursive chunker | 45 min |
| 3 — Chunk-size sweep on the 40-query gold set | 50 min |
| 4 — Chunking A/B memo (headline) | 1 h 0 min |
| 5 — Handle a table without smearing it | 45 min |
| 6 — Metadata injection: does the heading help? | 40 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunchrag_chunk` [mini-project](./mini-project/README.md) is in the same workspace — Week 9 imports it and starts from your winning chunker. Then take the [quiz](./quiz.md) with your notes closed.
