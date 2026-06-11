# Exercise 1 — Extract a PDF Three Ways

**Goal:** Take one real PDF and extract its text three ways — PyMuPDF (the fast native-text path), Unstructured (the typed-element path), and an OCR path (Tesseract on a rendered page) — then *compare what each one gets right and wrong*. You will train the single most important habit of extraction: **look at the text before you trust it.** Extraction failures are invisible to your metrics and obvious to your eyes.

**Estimated time:** 45 minutes. Guided.

---

## Setup

Install the three toolchains and an engine for Tesseract:

```bash
pip install pymupdf "unstructured[pdf]" pytesseract pillow
# Tesseract needs the engine too, not just the Python wrapper:
#   Ubuntu/Debian:  sudo apt install tesseract-ocr
#   macOS:          brew install tesseract
```

Pick a PDF. Any of these works — choose the one that teaches you the most:

- **A clean native-text PDF** (a contract, a Word/LaTeX export). PyMuPDF will read it perfectly; this is your baseline.
- **A messy mixed PDF** (a report with headings, lists, tables, headers/footers). Unstructured will shine here.
- **A scanned PDF** (a photo/scan of a page — print a page and photograph it, or find a scan). PyMuPDF returns *nothing*; the OCR path is the only one that works.

If you have no PDF handy, generate a native-text one from the legal corpus so the rest of the week's clauses are in it:

```python
# make_legal_pdf.py — a tiny native-text PDF of the legal clauses.
import pymupdf

clauses = [
    "1. This Agreement is entered into between the Company and the Contractor.",
    "7. The annual fee shall be paid in twelve equal monthly installments.",
    "9. All confidential information must be protected for five years after termination.",
    "12. The Contractor shall maintain professional liability insurance of $1,000,000.",
    "14. Either party may terminate this Agreement upon thirty days written notice.",
    "18. This Agreement is governed by the laws of the State of Delaware.",
    "27. Any dispute shall be resolved by binding arbitration in San Francisco.",
]
doc = pymupdf.open()
page = doc.new_page()
page.insert_text((72, 72), "SERVICES AGREEMENT\n\n" + "\n\n".join(clauses), fontsize=11)
doc.save("services_agreement.pdf")
print("wrote services_agreement.pdf")
```

```bash
python3 make_legal_pdf.py
```

---

## Step 1 — PyMuPDF, the fast native-text path

```python
import pymupdf

doc = pymupdf.open("services_agreement.pdf")
print(f"pages: {doc.page_count}")
text = "\n".join(page.get_text("text") for page in doc)
print("----- PyMuPDF text -----")
print(text)
```

Read the output. On a native-text PDF you should see every clause, clean, in order. **Confirm clause 9 reads "five years after termination" with no garbling** — that's the answer that has to survive the whole pipeline.

Now try the `"blocks"` mode and notice you get position information you could use to drop a header/footer by its y-coordinate:

```python
for block in doc[0].get_text("blocks"):
    x0, y0, x1, y1, btext, *_ = block
    print(f"y={y0:6.1f}  {btext[:50]!r}")
```

> If `get_text("text")` returns an **empty string**, your PDF is *scanned* (an image of text). That's not a bug — it's the signal that PyMuPDF is the wrong tool and you need OCR (Step 3).

---

## Step 2 — Unstructured, the typed-element path

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf("services_agreement.pdf")
print("----- Unstructured typed elements -----")
for el in elements:
    print(f"{type(el).__name__:14s} | {str(el)[:60]}")
```

The difference from PyMuPDF: you don't get a wall of text, you get *typed* elements — `Title`, `NarrativeText`, `ListItem`, possibly `Header`/`Footer`. This is the value. Two things to do:

1. **Drop the boilerplate by type** — this is cleaning for free:

```python
KEEP = {"Title", "NarrativeText", "ListItem", "Table"}
body = [el for el in elements if type(el).__name__ in KEEP]
print(f"kept {len(body)} of {len(elements)} elements after dropping headers/footers")
```

2. **Chunk by title** — group elements under their heading, with the heading attached:

```python
from unstructured.chunking.title import chunk_by_title

chunks = chunk_by_title(elements, max_characters=2000, combine_text_under_n_chars=200)
for i, ch in enumerate(chunks):
    print(f"--- chunk {i} ---\n{str(ch)[:200]}\n")
```

Notice that the chunk carries its section context. On a messy multi-section report this is dramatically better than a wall of text; on a flat one-page contract the advantage is smaller. **That corpus-dependence is the lesson** — Unstructured earns its cost where structure is the value.

---

## Step 3 — The OCR path (Tesseract on a rendered page)

This path is the *only* one that works on a scanned PDF, and it's instructive even on a native one (you'll see OCR is lossier than reading embedded text — so you only use it when you have to).

```python
import io
import pymupdf
import pytesseract
from PIL import Image

doc = pymupdf.open("services_agreement.pdf")
page = doc[0]
# Render the page to an image at 300 DPI — OCR accuracy lives or dies on resolution.
pix = page.get_pixmap(dpi=300)
img = Image.open(io.BytesIO(pix.tobytes("png")))
ocr_text = pytesseract.image_to_string(img, lang="eng")
print("----- Tesseract OCR text -----")
print(ocr_text)
```

Compare the OCR output to the PyMuPDF output of the *same* page:

- On a **native-text** PDF, OCR should be *close* but may introduce small errors (a stray character, a misread digit). That's why you don't OCR a PDF that has real text — you'd be degrading perfectly good text. PyMuPDF wins here.
- On a **scanned** PDF, OCR is the only option, and now resolution and preprocessing matter. Re-render at 150 DPI and watch accuracy collapse; that's the lesson about DPI.

> **Stretch — Surya.** If you have a GPU (or patience for the CPU fallback), run the same scanned page through Surya (`pip install surya-ocr`) and diff against Tesseract. On a single-column page they're close; on a multi-column or table-heavy page, Surya's reading-order detection pulls ahead and Tesseract interleaves the columns. That diff *is* the reason layout-aware OCR exists.

---

## Step 4 — Compare, and write down what you found

Build a small comparison table in `notes/week-08/extraction-compare.md`:

| Tool | Got clause 9 clean? | Reading order correct? | Headers/footers handled? | Tables preserved? | Speed (eyeball) | When I'd use it |
|---|---|---|---|---|---|---|

Fill one row per tool against *your* PDF. The "When I'd use it" column is the point: you're building the escalation ladder from Lecture 2 §1 from your own observation, not from the table in the notes.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] PyMuPDF extracts your native-text PDF and **clause 9 reads "five years after termination"** with no garbling (the answer survived extraction).
- [ ] Unstructured returns **typed elements**, and you dropped at least the `Header`/`Footer` types (or argued your PDF has none) and ran `chunk_by_title`.
- [ ] The Tesseract OCR path runs on a rendered page and you compared its output to PyMuPDF's on the same page, noting at least one difference.
- [ ] `notes/week-08/extraction-compare.md` has one row per tool with the "when I'd use it" column filled from your own observation.
- [ ] You can state, in one sentence, *why* you would not OCR a native-text PDF (you'd degrade perfectly good embedded text).

---

## Stretch

- Take a genuinely **scanned** page (photograph a printed page) and run all three paths. Watch PyMuPDF return empty, and watch OCR quality depend on how straight and high-contrast your photo is. Deskew it (rotate to straight) and re-OCR — accuracy jumps.
- Run a **two-column** PDF (a paper) through PyMuPDF `"text"` and watch the columns interleave into nonsense. Then run it through MinerU (`pip install -U "mineru[core]"; mineru -p paper.pdf -o ./out`) and confirm the reading order is reconstructed. That's the §1.3 lesson, measured.
- Extract a page with a **table** via `page.find_tables()` and serialize it to Markdown (Lecture 2 §3.1). Confirm the header→value structure survives — "Insurance | $1,000,000" — instead of smearing into prose.

---

When this feels comfortable, move to [Exercise 2 — The chunkers](exercise-02-chunkers.py).
