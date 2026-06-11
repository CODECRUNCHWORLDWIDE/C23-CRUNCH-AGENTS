# Exercise 1 — VLM vs. Claude Over a Figure

**Goal:** Take one PDF page that has a *figure or table*, render it to an image, and ask the **same question** over the picture two ways — an **open VLM** (Qwen2.5-VL, local) and **Claude vision** (`claude-opus-4-8`, over the API) — then compare what each gets right. Then, separately, embed a few images with **CLIP** and run a text→image retrieval, to feel the difference between a VLM that *reads* an image and a CLIP embedding that *finds* one. You will internalize the central multimodal lesson: the figure is often the answer, and there are two distinct tools for it — one reads, one retrieves.

**Estimated time:** 45 minutes. Guided.

---

## Setup

```bash
pip install pymupdf pillow sentence-transformers anthropic
# For the OPEN VLM leg (optional; needs a GPU for reasonable speed):
pip install transformers torch qwen-vl-utils
#   ...or run a VLM via Ollama with no Python deps:  ollama run qwen2.5vl
# For the CLAUDE leg: export ANTHROPIC_API_KEY=sk-ant-...
```

You need a PDF page with something visual on it — a chart, a table rendered as an image, a diagram. Any real PDF works. If you have none handy, generate one with a tiny table figure so the rest of the exercise has a target:

```python
# make_figure_pdf.py — a one-page PDF with a small "fee schedule" table drawn on it.
import pymupdf

doc = pymupdf.open()
page = doc.new_page()
page.insert_text((72, 72), "SCHEDULE A — FEES", fontsize=14)
# Draw a simple table the text extractor will struggle with but a VLM can read.
rows = [("Item", "Amount"),
        ("Annual fee", "$120,000"),
        ("Liability insurance", "$1,000,000"),
        ("Late payment rate", "1.5% / month")]
y = 110
for r0, r1 in rows:
    page.insert_text((72, y), f"{r0:<24}{r1}", fontsize=11)
    page.draw_line((72, y + 4), (320, y + 4))
    y += 28
doc.save("schedule_a.pdf")
print("wrote schedule_a.pdf")
```

```bash
python3 make_figure_pdf.py
```

---

## Step 1 — Render the PDF page to an image

A VLM reads *pixels*, so the first move is always to rasterize the page (week-8 `get_pixmap`, now for a vision model instead of OCR). Render at a high DPI — a VLM, like OCR, reads a sharp image better than a blurry one:

```python
import pymupdf

doc = pymupdf.open("schedule_a.pdf")
page = doc[0]
pix = page.get_pixmap(dpi=200)         # rasterize the page
pix.save("schedule_a_page.png")
print("wrote schedule_a_page.png")
```

Open the PNG and confirm the table is legible. That image is what both the open VLM and Claude will read.

---

## Step 2 — Ask the open VLM (Qwen2.5-VL)

The open-weights leg. Ask a specific question whose answer lives in the *figure*, not in extractable text:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

image = Image.open("schedule_a_page.png")
QUESTION = "What is the liability insurance amount in this table?"
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": QUESTION},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
print("OPEN VLM:", processor.batch_decode(out, skip_special_tokens=True)[0])
```

> **No GPU?** Run the same model via Ollama (`ollama run qwen2.5vl`) through its OpenAI-compatible API with an image, or skip to Step 3 and note in your write-up that you compared Claude against the *expected* VLM behavior. The lesson — VLMs read figures — lands either way.

---

## Step 3 — Ask Claude vision the *same* question

The frontier leg. Note the authoritative image-block shape (base64 source, media type, then the text block) and the model facts: `claude-opus-4-8`, no `temperature`:

```python
import base64
import anthropic

client = anthropic.Anthropic()
with open("schedule_a_page.png", "rb") as f:
    b64 = base64.standard_b64encode(f.read()).decode("utf-8")

QUESTION = "What is the liability insurance amount in this table?"
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=512,
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64",
                                      "media_type": "image/png", "data": b64}},
        {"type": "text", "text": QUESTION},
    ]}],
)
print("CLAUDE:", "".join(b.text for b in response.content if b.type == "text"))
```

---

## Step 4 — Compare, and prove the text pipeline would have missed it

Now the punchline. Run your week-8 PyMuPDF *text* extraction on the same page and see what a text-only RAG would have indexed:

```python
import pymupdf
doc = pymupdf.open("schedule_a.pdf")
print("----- text extraction (what a text-only RAG sees) -----")
print(doc[0].get_text("text"))
```

Depending on how the table was drawn, the text extraction may smear the cells or miss the structure — which is the whole point. **The VLM and Claude both read "$1,000,000" off the picture; the text pipeline may not have a clean "Liability insurance | $1,000,000" chunk to retrieve.** Write down, in `notes/week-12/multimodal-compare.md`:

| Path | Got the insurance amount? | Read the structure? | Cost / hardware | When I'd use it |
|---|---|---|---|---|
| Text extraction (PyMuPDF) | | | free, CPU | |
| Open VLM (Qwen2.5-VL) | | | free, GPU | |
| Claude vision | | | API, no hardware | |

---

## Step 5 — CLIP: a text query *retrieves* an image

The other half of multimodal: not reading an image, but *finding* the right one. Embed a few page images with CLIP and retrieve with a text query:

```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image

clip = SentenceTransformer("clip-ViT-B-32")     # one model, both modalities
images = [Image.open("schedule_a_page.png")]    # add more rendered pages to make it interesting
img_vecs = clip.encode(images, normalize_embeddings=True)

q = clip.encode("a fee schedule table", normalize_embeddings=True)
scores = util.cos_sim(q, img_vecs)[0]
for i, s in enumerate(scores):
    print(f"image {i}: score={float(s):.3f}")
```

Notice what just happened: a *text* query ranked *images*. That's multimodal retrieval. Notice also what CLIP did **not** do — it didn't tell you the insurance amount. CLIP *finds* the relevant picture (the gist: "a fee schedule table"); a VLM *reads* it. Those are two different tools, and a real multimodal RAG uses both: CLIP/ColPali to retrieve the page, a VLM to answer from it.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] You rendered a PDF page with a figure/table to a PNG at a sensible DPI.
- [ ] You asked the **same question** of an open VLM (or its documented Ollama/fallback equivalent) **and** Claude vision, and recorded both answers.
- [ ] You ran the week-8 **text extraction** on the same page and can state whether a text-only RAG would have had a clean chunk containing the answer.
- [ ] `notes/week-12/multimodal-compare.md` has the comparison table filled from your own observation, including the "when I'd use it" column.
- [ ] You ran the **CLIP text→image retrieval** and can state, in one sentence, the difference between CLIP *finding* an image and a VLM *reading* one.

---

## Stretch

- **Describe-at-index-time (Option A).** Use Claude vision to *caption* the figure ("a fee schedule listing an annual fee of $120,000 and liability insurance of $1,000,000"), store that caption as a text chunk, and confirm a BGE text retriever now finds the figure for the query "insurance amount." That's the text-anchored multimodal architecture from Lecture 1 §3, built.
- **ColQwen visual retrieval.** Install `colpali-engine`, index 3–5 rendered pages with `vidore/colqwen2.5-v0.2`, and retrieve with a text query using late-interaction scoring (Lecture 1 §4). Compare its ranking to CLIP's on the same pages — late interaction should localize better on dense pages.
- **A multimodal judge.** Take the figure question and have `claude-opus-4-8` *judge* whether the open VLM's answer is faithful to the figure — by passing the image block *and* the VLM's answer in one message. That's a multimodal LLM-as-judge, the bridge to Lecture 2.

---

When this feels comfortable, move to [Exercise 2 — The Ragas metrics from scratch](exercise-02-ragas-metrics-from-scratch.py).
