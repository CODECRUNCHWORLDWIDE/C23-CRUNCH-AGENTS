# Week 8 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 9. Answer key is at the bottom — don't peek.

---

**Q1.** Why must you chunk a 30-page document instead of embedding it as a single vector? Pick the *most complete* reason.

- A) Postgres can't store a vector longer than 1024 floats.
- B) One chunk becomes one vector that blurs the whole span; the encoder has a token ceiling (BGE truncates at 512) and silently drops the rest; and the LLM context budget makes retrieving a whole document wasteful and dilutive.
- C) Embedding models refuse documents over one page.
- D) Chunking is required by the pgvector API.

---

**Q2.** You configure a fixed-window chunker with `chunk_size=512` and feed it raw character counts. What's the bug?

- A) Nothing — 512 characters and 512 tokens are the same thing.
- B) Chunk size must be measured in the *embedding model's tokens*, not characters; counting characters means your chunks are a different (and wrong) size than the encoder's budget.
- C) 512 is too small for any corpus.
- D) Fixed-window chunkers can't take a size parameter.

---

**Q3.** What does sliding-window overlap actually buy you, and what does it cost?

- A) It makes embeddings faster; it costs nothing.
- B) An answer that straddles a chunk boundary survives whole in at least one chunk; it costs storage/embedding redundancy and produces duplicate hits in the top-k.
- C) It increases the encoder's token ceiling; it costs accuracy.
- D) It removes the need for an embedding model; it costs latency.

---

**Q4.** In recursive (LangChain-style) chunking with separators `["\n\n", "\n", ". ", " ", ""]`, why does a legal clause that is its own paragraph usually survive whole?

- A) Because recursive chunking embeds each clause twice.
- B) Because the first separator `"\n\n"` puts boundaries around paragraphs, so the splitter cuts *between* clauses, not through them — only falling to finer separators for pieces still over the size budget.
- C) Because recursive chunking ignores the chunk size.
- D) Because periods are never used as separators.

---

**Q5.** Semantic-paragraph chunking decides boundaries by:

- A) Splitting every N characters.
- B) Embedding each sentence and splitting where the cosine *distance* between adjacent sentences spikes (a meaning shift), thresholded by a percentile.
- C) Asking an LLM to summarize each page.
- D) Splitting on every newline.

---

**Q6.** State the late-chunking mechanism (Jina, 2024) correctly.

- A) Embed each chunk independently, then average all the chunk vectors into one.
- B) Embed the *whole document* through a long-context encoder in one forward pass to get per-token contextual vectors, then mean-pool the token embeddings within each chunk's span — so each chunk's vector carries the surrounding context.
- C) Use an LLM to rewrite each chunk before embedding.
- D) Chunk the document, embed the chunks, then re-chunk the embeddings.

---

**Q7.** Why can't you late-chunk a 30-page document through `BAAI/bge-large-en-v1.5`?

- A) BGE isn't an embedding model.
- B) BGE truncates at 512 tokens, so it can't do the one-forward-pass-over-the-whole-document that late chunking requires; late chunking needs a long-context encoder like jina-embeddings-v3.
- C) BGE can't mean-pool.
- D) Late chunking only works with OpenAI models.

---

**Q8.** The chunk-size-vs-Recall@5 curve typically peaks in the middle. Why does Recall@5 *drop* at very large chunk sizes?

- A) Large chunks exceed Postgres's row limit.
- B) The chunk vector becomes a blurry average of many ideas (the one relevant clause is one of nine in the chunk), so a specific query matches it only weakly — even though the answer is self-contained inside it.
- C) Large chunks are always truncated to 512 tokens.
- D) MRR and Recall@5 measure opposite things at large sizes.

---

**Q9.** You have a native-text PDF (text is selectable). Which extraction tool do you reach for *first*, and why?

- A) LlamaParse, because it's the most powerful.
- B) PyMuPDF, because it extracts the embedded text directly with no model inference — fast and free — and you escalate to heavier tools only if it demonstrably fails.
- C) Tesseract, because OCR is always the safest path.
- D) MinerU, because every PDF is scientific.

---

**Q10.** `page.get_text("text")` returns an empty string on a PDF. What does this most likely mean?

- A) The PDF is corrupt and unrecoverable.
- B) The PDF is *scanned* — an image of text with no embedded text — so PyMuPDF is the wrong tool and you need OCR (Tesseract or Surya).
- C) PyMuPDF is not installed.
- D) The document has too many pages.

---

**Q11.** When would you choose Surya over Tesseract for OCR?

- A) Never — Tesseract is strictly better.
- B) On a multi-column or table-heavy scan where reading order matters: Surya does layout and reading-order detection first, while Tesseract reads top-to-bottom/left-to-right and interleaves columns into nonsense.
- C) Only when you have no GPU.
- D) When the document is already native text.

---

**Q12.** In the chunking A/B harness, your gold set is in *clause ids* but you retrieve *chunks*. What must you do before scoring, and why?

- A) Nothing; chunk ids and clause ids are interchangeable.
- B) Map each chunk hit back to its source clause id (de-duplicated, rank-preserving) and score on clauses — because different strategies produce different chunks, so scoring raw chunk ids makes the strategies incomparable.
- C) Re-chunk the gold set to match each strategy.
- D) Use a different gold set for each strategy.

---

**Q13.** Late-chunking-with-jina beats recursive-with-BGE by 0.01 Recall@5 in your A/B. What's the honest conclusion?

- A) Late chunking is definitively the best chunking strategy.
- B) You changed *two* things (the chunker *and* the model), so you've shown "jina-v3 + late chunking beats BGE + recursive," not that late chunking beats recursive — to isolate the chunking effect you'd run jina-v3 with early chunking as a control.
- C) Recursive chunking is broken.
- D) The 0.01 delta proves BGE is the wrong embedding.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — All three constraints: the one-vector blur, the encoder's 512-token ceiling (with silent truncation), and the LLM context budget. (Lecture 1 §1.)
2. **B** — Chunk size is in the embedding model's tokens, not characters; LangChain's default `length_function` counts characters, which is the classic misconfiguration. (Lecture 1 §2, §4.)
3. **B** — Overlap rescues boundary-straddling answers; it costs storage redundancy and duplicate top-k hits. It's insurance, not a cure. (Lecture 1 §3.)
4. **B** — The `"\n\n"` separator cuts around paragraphs first; finer separators only kick in for oversized pieces. A clause that is a paragraph survives whole. (Lecture 1 §4.)
5. **B** — Embed sentences, split at adjacent-sentence cosine-distance spikes, thresholded by percentile. The embedding model is the boundary detector. (Lecture 1 §5.)
6. **B** — One forward pass over the whole document → per-token contextual vectors → mean-pool per chunk span. That's the mechanism, and why context is preserved. (Lecture 1 §6; arXiv 2409.04701.)
7. **B** — BGE's 512-token ceiling makes the whole-document forward pass impossible; late chunking needs a long-context encoder (jina-v3, 8192 tokens). (Lecture 1 §6.)
8. **B** — Too-large chunks dilute the vector (the answer is one idea among many), so specific queries match weakly. Too-small splits answers. The peak is in between. (Lecture 1 §7.)
9. **B** — PyMuPDF first on native-text PDFs: direct text extraction, no inference, fast and free. Escalate only on demonstrated failure. (Lecture 2 §1.1, §1.4.)
10. **B** — Empty `get_text()` means a scanned (image) PDF; PyMuPDF is the wrong tool and you need OCR. (Lecture 2 §1.1, §2.)
11. **B** — Surya does layout + reading-order detection, so it handles multi-column/table scans where Tesseract interleaves columns. (Lecture 2 §2.1–2.2.)
12. **B** — Map chunk hits back to source clause ids before scoring; different strategies make different chunks, so raw chunk-id scoring is incomparable. This is *the* A/B trap. (Lecture 2 §5.2; challenge trap.)
13. **B** — Two changes at once (chunker + model). The honest isolation is jina-v3 + early chunking as a control; the late-vs-early gap *at the same model* is the real lift. (Lecture 2 §5.1; challenge second trap.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
