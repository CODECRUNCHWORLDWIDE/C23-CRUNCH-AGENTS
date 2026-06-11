# Week 8 — Exercises

Three focused drills that take you from "raw PDF" to "I measured which chunk size wins." Each takes 30–60 minutes. Do them in order — exercise 3 reuses the chunkers you build in exercise 2, which assumes the extraction intuition from exercise 1.

## Index

1. **[Exercise 1 — Extract a PDF three ways](exercise-01-extract-a-pdf.md)** — extract a real PDF with PyMuPDF, Unstructured, and an OCR path (Tesseract), and compare what each gets right and wrong. (~45 min, guided)
2. **[Exercise 2 — The chunkers](exercise-02-chunkers.py)** — implement fixed token-window, sliding-window-with-overlap, and recursive chunkers from scratch and inspect where each one puts its boundaries. (~50 min, runnable)
3. **[Exercise 3 — The chunk-size sweep](exercise-03-chunk-size-sweep.py)** — sweep chunk size against the legal gold set and chart Recall@5 vs chunk size, so "chunk size is a hyperparameter" stops being a slogan. (~50 min, runnable)

## How to work the exercises

- Create a fresh virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install the deps as each exercise needs them: `pip install pymupdf "unstructured[pdf]" pytesseract pillow transformers sentence-transformers "psycopg[binary]" numpy`. (Tesseract also needs the *engine*: `apt install tesseract-ocr` or `brew install tesseract`.)
- **Look at the text before you trust it.** Exercise 1's whole point is that extraction failures are visible to the eye and invisible to the metrics. Open the raw output and read it.
- **Count tokens in the model's units, not characters.** Every chunker measures size with the BGE tokenizer (`AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")`), because that's the budget the encoder sees. Counting characters is the classic off-by-a-lot bug.
- When retrieval is bad, walk the §6 decision tree from Lecture 2 *before* you touch the embedding. Extraction first, cleaning second, chunking third, embedding last.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape*, you're not done.

## Running the Python exercises

The two `.py` files are standalone. Exercise 3 needs Postgres + pgvector running (same container as week 7) and the `crunchrag_embed` package importable; its header documents the fallback if you don't have them.

```bash
docker run -d --name crunch-pg \
  -e POSTGRES_PASSWORD=crunch \
  -p 5432:5432 \
  pgvector/pgvector:pg17

# then, with the venv active:
python3 exercise-02-chunkers.py
python3 exercise-03-chunk-size-sweep.py
```

The first `SentenceTransformer("BAAI/bge-large-en-v1.5")` call downloads ~1.3 GB. Do it on good wifi, not five minutes before a deadline.

## A note on determinism

Chunkers are deterministic — the same text and parameters give the same boundaries every run. The embedding is deterministic too. The only wobble is ANN search (week 7's lesson), which is tiny on a corpus this size. So the chunk-size sweep's Recall@5 curve is reproducible: if you can't reproduce the *peak*, something changed (the tokenizer, the corpus, the gold set), and that's worth finding.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-08` to compare.
