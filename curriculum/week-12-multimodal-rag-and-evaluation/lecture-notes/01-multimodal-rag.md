# Lecture 1 — Multimodal RAG: When the Answer Is in the Picture

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can name the 2026 multimodal stack (VLMs, CLIP/SigLIP image embeddings, ColPali visual retrieval), choose between *embedding the page image* and *embedding the extracted text* for a figure-heavy corpus, build a multimodal answer step over a PDF page with both an open VLM and Claude vision, and place ASR (Whisper) / TTS (Piper, XTTS) / image generation (SDXL, Flux) correctly as pipeline-adjacent capabilities rather than RAG itself.

For eleven weeks your RAG pipeline has had one blind spot it never admitted to: it can only retrieve and reason over **text**. Every PDF you chunked, you chunked the *prose* of. Every figure, every chart, every scanned page, every table-as-an-image — extracted to nothing, or to garbled OCR, and dropped. On a contract that's mostly clauses, fine. On a financial filing where the number you need lives in a chart, or a research paper where the result is a figure, or a scanned form where the layout *is* the meaning — your text-only pipeline retrieves the caption and misses the answer.

If you remember one sentence from this lecture, remember this one:

> **The figure is often the answer.** A pipeline that extracts only prose has thrown away exactly the content a human reader would point at. Multimodal RAG is how you stop dropping it.

There's a corollary that frames the whole architecture decision:

> **You can retrieve text *about* an image, or you can retrieve the image itself.** Those are two different pipelines with two different failure modes, and choosing between them — embed the extracted text vs. embed the page picture — is the central engineering decision of multimodal RAG.

This lecture is the multimodal half of the week; Lecture 2 is the evaluation half. They meet in the lab: a Ragas suite that can score answers from a *multimodal* pipeline, with a calibrated judge, is the Phase II milestone. But first you have to build the multimodal pipeline, and that starts with understanding the three pieces of the 2026 stack.

---

## 1. Vision-language models — an image encoder bolted onto an LLM

A **vision-language model (VLM)** is, mechanically, exactly what the name says: an image encoder whose output is projected into the token space of an LLM, so the LLM can "read" an image the way it reads text. You hand it an image and a text prompt; it attends over both and generates text. That's it. The image becomes a sequence of visual tokens prepended to your text tokens, and the model answers.

The open VLMs you should know in 2026:

- **LLaVA** — the original open recipe (a CLIP-style vision encoder + a projection layer + an LLM). It's the reference architecture every later VLM iterates on. Good to know exists; usually superseded by the newer families below for real work.
- **Qwen2.5-VL** (Alibaba) — the strong, general-purpose open VLM. Reads documents, charts, and figures well; handles high-resolution inputs and bounding-box grounding. The **default open VLM** for this week.
- **Phi-3.5-Vision** (Microsoft) — small and efficient, runs on modest hardware. The lightweight option when you can't fit a 7B+ VLM on your GPU.
- **InternVL** (OpenGVLab) — a high-accuracy family, especially strong on document and chart understanding. The heavier open option when accuracy matters more than footprint.

And the frontier path: **Claude** with vision (`claude-opus-4-8`, `claude-sonnet-4-6`) reads images natively. In a multimodal RAG pipeline a VLM plays one of two roles:

1. **Answerer** — given the retrieved page image(s) and the question, generate the answer directly from the picture. This is "multimodal RAG" in the fullest sense: the generator sees the figure.
2. **Describer / captioner** — at *index* time, generate a text description of each figure ("a bar chart showing revenue rising from $2M in 2023 to $5M in 2025"), and store *that text* as a searchable chunk pointing at the image. Now a text retriever can find the figure, and at answer time you can hand the actual image to the generator.

Here is the open-VLM answerer path, the one you'll build in Exercise 1. Run a VLM locally over a rendered PDF page:

```python
# An open VLM (Qwen2.5-VL) answering a question over a PDF page image.
# pip install transformers torch pillow qwen-vl-utils
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

image = Image.open("page_with_chart.png")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What was the 2025 revenue in the chart?"},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
generated = model.generate(**inputs, max_new_tokens=128)
answer = processor.batch_decode(generated, skip_special_tokens=True)[0]
print(answer)
```

And the **same task** via Claude vision — the frontier path, no GPU, the authoritative image-block shape you'll reuse in the judge later. Note the message-content shape: a list with an `image` block (base64 or URL) followed by a `text` block.

```python
# The same question over the same page, via Claude vision.
# pip install anthropic
import base64
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

with open("page_with_chart.png", "rb") as f:
    b64 = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=512,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
                {"type": "text", "text": "What was the 2025 revenue in the chart?"},
            ],
        }
    ],
)
# Iterate content blocks; collect the text.
answer = "".join(b.text for b in response.content if b.type == "text")
print(answer)
```

Two things to internalize about that Claude call, because they recur in the judge in Lecture 2. First, the **image block shape** — `{"type":"image","source":{"type":"base64","media_type":"...","data":<b64>}}` (or `{"type":"url","url":"..."}`) — is the canonical way to send Claude an image, and it's *exactly* how a multimodal judge or answerer reads a PDF page. Second, on `claude-opus-4-8` you do **not** pass `temperature`/`top_p`/`top_k` — they 400. The model uses adaptive thinking; if you want more reasoning effort you pass `output_config={"effort":"high"}`, which matters for the judge but is overkill for a simple read-the-chart query.

The honest trade-off between the two paths: the open VLM is free, private, and self-hosted (the open-weights path the course insists on), but needs a GPU for reasonable latency and is weaker on hard charts and dense documents. Claude vision is stronger and needs no hardware, but it's an API call with a cost and a dependency. **Build both. Compare them on your figures.** That comparison *is* Exercise 1, and it's how you decide which answerer your pipeline ships.

---

## 2. Image embeddings — CLIP and SigLIP put images and text in one space

A VLM *reads* an image to generate text. An **image-embedding model** does something different: it turns an image into a vector *in the same space as text vectors*, so a text query can retrieve an image by cosine similarity — exactly the way your BGE text embedding lets a text query retrieve a text chunk, but now one of the two sides is a picture.

**CLIP** (OpenAI, 2021) is the model that made this work. It was trained contrastively on 400M image–caption pairs to pull matching image and text vectors together and push mismatched ones apart. The result: one shared embedding space where "a photo of a dog" (the text) lands near a picture of a dog (the image). **SigLIP** (Google, 2023) is the successor — same idea, a sigmoid loss instead of CLIP's softmax contrastive loss, and stronger zero-shot matching. For our purposes they're drop-in interchangeable; SigLIP is the better default in 2026 when you can get it.

The mechanics, with `sentence-transformers` (which you already have from week 7):

```python
# CLIP image+text retrieval: a text query retrieves the nearest image.
# pip install sentence-transformers pillow
from sentence_transformers import SentenceTransformer, util
from PIL import Image

model = SentenceTransformer("clip-ViT-B-32")  # one model, encodes BOTH modalities

# Embed a set of images (e.g. rendered PDF pages or figures).
images = [Image.open(p) for p in ["fig1.png", "fig2.png", "fig3.png"]]
img_vecs = model.encode(images, normalize_embeddings=True)

# Embed a TEXT query into the SAME space.
query_vec = model.encode("revenue growth bar chart", normalize_embeddings=True)

# Cosine similarity ranks the images by relevance to the text query.
scores = util.cos_sim(query_vec, img_vecs)[0]
ranking = scores.argsort(descending=True)
for idx in ranking:
    print(f"image {int(idx)}: score={float(scores[idx]):.3f}")
```

The headline: **you embedded a text query and got a ranked list of images.** That's multimodal retrieval — the same brute-force-or-ANN cosine search you've done all of Phase II, but the corpus is pictures. You could store these CLIP image vectors in pgvector right next to your BGE text vectors (different table, different dim) and have a retriever that returns both text chunks and figure images for a query.

Where CLIP/SigLIP shine and where they don't:

- **Strong** on natural images and the *gist* of a figure — "a bar chart," "a photo of a circuit board," "a diagram with arrows." A text query about the *kind* of image retrieves it well.
- **Weak** on dense text-in-images and fine detail — CLIP's vision encoder downsamples; it does *not* read the numbers off a chart or the words in a scanned paragraph. It knows "this is a table," not "the insurance value is $1,000,000." For reading the *content* of a document image you want a VLM (§1) or visual document retrieval (§4), not a CLIP embedding.

So CLIP/SigLIP are a **retrieval** tool — "find me the relevant picture" — not a **reading** tool. That distinction sets up the central decision of the next section.

---

## 3. The central decision — embed the page image, or embed the extracted text?

Here is the engineering choice that defines multimodal RAG over PDFs with figures. You have a document with prose *and* figures. When you build the retrieval index, you have two fundamentally different options for the visual content:

**Option A — extract + describe (text-anchored).** Run your week-8 extraction pipeline. For each figure, either keep its caption *or* run a VLM to generate a text description of it, and store *that text* as a chunk (with a pointer to the image file in metadata). Your index is entirely **text** — BGE vectors as before — and figures are represented by their captions/descriptions. At answer time, you retrieve text chunks; if a chunk points at an image, you can hand the image to a VLM generator.

**Option B — embed the page image (vision-anchored).** Render each page (or each figure) to an image and embed it with CLIP/SigLIP (or, better, ColPali — §4). Your index is **image vectors**. A text query retrieves page images directly, and a VLM answers from the retrieved page picture. You may skip text extraction entirely.

The trade-offs, honestly:

| Axis | Option A (extract + describe) | Option B (embed page image) |
|---|---|---|
| **Reads fine detail?** | Yes — the VLM description can capture "revenue rose to $5M" at index time | Only at answer time, when the VLM sees the retrieved page |
| **Retrieval quality on text** | Excellent — it's your proven BGE text pipeline | Weaker for prose; CLIP doesn't read text well |
| **Survives bad OCR / complex layout** | No — garbage extraction is garbage indexed | **Yes** — there's no extraction step to fail (esp. ColPali) |
| **Index cost** | One VLM call per figure at index time (or free if captions suffice) | Embed every page image; storage is image vectors |
| **Best for** | Text-heavy docs with occasional figures | Figure-heavy, scan-heavy, layout-dependent docs |

The decision rule, stated plainly: **if your corpus is mostly text with a few figures, stay text-anchored (Option A) and describe the figures into it. If your corpus is figure-heavy, scanned, or layout-dependent — where extraction keeps failing — go vision-anchored (Option B).** And like everything in Phase II, you don't *guess* which — you measure both with the eval harness from Lecture 2. A multimodal pipeline's Ragas faithfulness on figure questions tells you which architecture actually answered them.

A note on the *hybrid* in practice, because the cleanest production systems are rarely pure-A or pure-B. The common pattern in 2026 is: index the prose text-anchored (Option A — your proven BGE pipeline) *and* index the page images vision-anchored (Option B — CLIP/ColPali) into a second store, then at query time retrieve from both and let the generator (a VLM) see whichever the retriever ranked highest. The prose questions hit the text index; the figure questions hit the image index; and you don't have to pick one architecture for a corpus that has both kinds of content. The cost is two indexes and a fusion step (the week-9 hybrid-fusion machinery, now across *modalities* instead of across dense/lexical). The benefit is that no question class is structurally unanswerable. As always, you justify the added complexity with the eval — if the figure-question faithfulness doesn't move when you add the image index, you didn't need it for this corpus.

A concrete worked example. Take a financial filing where "what was 2025 revenue?" is answered only by a bar chart. Option A's success hinges on whether your VLM *described the chart with the number* at index time — if the description says "a revenue chart" but not "$5M," the text chunk retrieves but the answer isn't in it, and faithfulness tanks. Option B's success hinges on whether CLIP/ColPali *retrieved the right page* — if it did, the VLM reads $5M off the chart at answer time. Same question, two completely different failure surfaces. The eval is what tells you which one your corpus suffers from.

---

## 4. Visual document retrieval — ColPali skips OCR entirely

There's a third path worth knowing exists, because it's the cleanest answer to "my extraction keeps failing on complex layouts": **ColPali** (and its stronger sibling **ColQwen**). The insight: instead of OCR→chunk→embed (three stages that each fail), embed the page *image* directly with a VLM backbone and retrieve with **late interaction** (ColBERT-style) scoring.

The mechanism, briefly. Where CLIP gives you *one* vector per image, ColPali gives you *many* — a grid of patch embeddings, one per image region, plus per-token embeddings on the query side. Scoring is **MaxSim**: for each query token, find its best-matching image patch, and sum those maxes. This late-interaction scoring is far more precise than single-vector cosine because a query token like "insurance" can match the specific *region* of the page where the insurance value sits, rather than the whole-page average. The payoff: you retrieve over rendered pages with no OCR, no chunking, no extraction — the three most fragile stages of week 8's pipeline — gone.

```python
# ColPali / ColQwen: retrieve over PAGE IMAGES with late-interaction scoring.
# pip install colpali-engine
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from PIL import Image
import torch

model = ColQwen2_5.from_pretrained("vidore/colqwen2.5-v0.2", torch_dtype=torch.bfloat16)
processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")

# Index: embed page images (multi-vector, one set of patch embeddings per page).
pages = [Image.open(p) for p in ["page1.png", "page2.png", "page3.png"]]
page_batch = processor.process_images(pages).to(model.device)
with torch.no_grad():
    page_embeddings = model(**page_batch)  # multi-vector per page

# Query: embed the text query (multi-vector, one per query token).
q_batch = processor.process_queries(["what is the insurance amount?"]).to(model.device)
with torch.no_grad():
    query_embeddings = model(**q_batch)

# Late-interaction (MaxSim) scoring ranks pages by query-token-to-patch match.
scores = processor.score_multi_vector(query_embeddings, page_embeddings)
print(scores)  # one relevance score per page; argmax is the page to feed the VLM
```

The honest caveat: ColPali's multi-vector index is *bigger* than a single-vector index (many vectors per page) and the late-interaction scoring is heavier than cosine. It's the right tool for **figure-heavy, scan-heavy, layout-dependent** corpora where Option A's extraction keeps failing — not the default for a clean text contract where BGE already wins. You don't need to build ColPali this week; you need to *know it exists* so that when a reviewer asks "why is your extraction so fragile?" you can answer "for this corpus I'd retrieve over page images with ColQwen and skip extraction entirely." That's the §6 stretch goal in the README.

---

## 5. ASR and TTS — pipeline-adjacent, not RAG itself

Two capabilities frequently sit *next to* a RAG system and get confused for part of it. They're not. They're **adapters** on the input and output edges.

**ASR (Automatic Speech Recognition)** turns audio into text — and that text is then chunked, embedded, and indexed by your *existing* text RAG. ASR doesn't change your retrieval; it *feeds* it. The moment a meeting recording or a podcast becomes a transcript, it's a text document like any other. The 2026 stack:

- **Whisper** (OpenAI) — the open reference ASR. Accurate, multilingual, the model everyone compares against. `pip install openai-whisper`.
- **whisper.cpp** — the C/C++ port that runs Whisper *fast on CPU* (and Apple Silicon) with no Python or GPU. The self-hosted, low-resource path.
- **faster-whisper** — the CTranslate2 reimplementation: same accuracy, much faster on GPU. The production path.

```python
# ASR: audio -> transcript, which then enters your TEXT RAG unchanged.
# pip install faster-whisper
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")  # CPU-friendly
segments, info = model.transcribe("meeting_recording.mp3")
transcript = " ".join(seg.text for seg in segments)
# `transcript` is now a normal text document: chunk it (week 8), embed it (week 7),
# index it (week 10), retrieve it (week 9). ASR is the ADAPTER, not the RAG.
print(transcript[:200])
```

**TTS (Text-to-Speech)** is the mirror image: it turns your RAG's *answer* back into audio, on the output edge. It changes nothing about retrieval or generation; it just renders the final string as speech. The stack:

- **Piper** (Rhasspy/OHF) — fast, local, CPU-only neural TTS. The lightweight self-hosted option.
- **XTTS / Coqui TTS** — higher quality, voice-cloning-capable, heavier. The premium self-hosted option.

```python
# TTS: the RAG's final answer string -> speech (output edge only).
# Piper runs as a CLI/binary; XTTS via the coqui TTS Python API:
# pip install TTS
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="Confidential information must be protected for five years after termination.",
    file_path="answer.wav",
    speaker_wav="reference_voice.wav",
    language="en",
)
```

Why this placement matters for *evaluation* (the week's real subject): **you evaluate the text, not the audio.** A voice-RAG's quality is its transcript's faithfulness and its answer's relevancy — the same Ragas metrics — not the naturalness of the TTS voice. ASR introduces a *new* error source (a mis-transcription that corrupts a chunk before it's ever indexed), which is exactly the kind of upstream failure week 8's "walk up the pipeline" discipline catches. Place ASR/TTS as adapters, evaluate the text in the middle, and you don't get confused about what you're measuring.

---

## 6. Image generation — adjacent, and not on the eval path

For completeness, because they come up: **SDXL** (Stable Diffusion XL) and **FLUX.1** (Black Forest Labs) are the open **text-to-image generation** models in 2026, run via the `diffusers` library. They take a text prompt and produce a *new* image.

```python
# SDXL / Flux: text -> a NEW image. Adjacent capability, not part of RAG eval.
# pip install diffusers torch
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
image = pipe("a clean architecture diagram of a RAG pipeline").images[0]
image.save("generated.png")
```

Why this is *adjacent* and not core: a RAG system *retrieves* and *reasons over* existing content; image generation *fabricates new* content. They share a "multimodal" label and nothing else. You will not put SDXL or Flux on the eval path this week — there's no "faithfulness" of a generated picture in the RAG sense. Know they exist, know they're a generation capability (sometimes paired with RAG for, e.g., generating an illustrative figure from retrieved facts), and keep them off your evaluation harness. The Ragas metrics in Lecture 2 score *answers grounded in retrieved context* — a generated image isn't that.

---

## 7. The multimodal RAG pipeline, end to end

Putting it together, here's where each piece sits in a pipeline that handles a figure-heavy PDF — and where it connects to the weeks you've already done:

```
                         ┌─────────── INDEX TIME ───────────┐
audio ──[Whisper ASR]──► text ─┐
PDF prose ─────────────────────┼─► chunk (wk8) ─► BGE embed (wk7) ─► pgvector (wk10)
PDF figures ─┬─[VLM describe]──┘                     (Option A: text-anchored)
             └─[render page]──► CLIP/ColPali embed ─► image index   (Option B: vision-anchored)

                         ┌─────────── QUERY TIME ───────────┐
query ─► retrieve (wk9: rerank/hybrid) ─► top-k {text chunks, page images}
      ─► generate: text-only LLM  OR  VLM (Claude vision / Qwen2.5-VL) over page images
      ─► answer ──[TTS Piper/XTTS]──► speech (optional output edge)
      ─► EVALUATE (Lecture 2): Ragas faithfulness / context_recall / context_precision /
                               answer_relevancy, with a CALIBRATED judge
```

Read the seams. Weeks 7–11 built the text spine (chunk → embed → store → retrieve → memory). This lecture adds the **modalities** at the edges: ASR feeds text in, VLMs and CLIP/ColPali handle figures, TTS renders speech out. And the whole thing is *still measured the same way* — which is the entire point of the handoff to Lecture 2. A multimodal pipeline that you can't evaluate is no better than a text pipeline you can't evaluate; the modality is new, the discipline isn't. When a figure-question's faithfulness is low, you walk up the multimodal pipeline exactly as you walked up the text one: did retrieval bring back the right page image? did the VLM read the chart correctly? is the answer grounded in what the VLM saw? Same procedure, more modalities.

---

## 8. Recap

You should now be able to:

- Name the **VLM landscape** — LLaVA (the reference recipe), Qwen2.5-VL (the strong open default), Phi-3.5-Vision (the lightweight option), InternVL (the heavy-accuracy option), and Claude vision (the frontier) — and describe a VLM as an image encoder projected into an LLM's token space, used either as an **answerer** (read the figure) or a **describer** (caption it at index time).
- Use **CLIP/SigLIP** to embed images and text into one space so a text query retrieves an image — and know their limit: they retrieve the *gist* of a figure, they do **not** read fine detail off a chart (that's a VLM's job).
- Make the **central architecture decision** — embed the extracted text (Option A, text-anchored, best for text-heavy corpora) vs. embed the page image (Option B, vision-anchored, best for figure/scan/layout-heavy corpora) — and know to *measure* both rather than guess.
- Explain **ColPali/ColQwen** visual document retrieval — multi-vector page-image embeddings with late-interaction (MaxSim) scoring that skips OCR/chunking/extraction entirely — and when it's the right tool (fragile-extraction, figure-heavy corpora).
- Place **ASR** (Whisper / whisper.cpp / faster-whisper) and **TTS** (Piper / XTTS) as pipeline-adjacent **adapters** — content in, speech out — that feed and render the text RAG without changing it, and **image generation** (SDXL / Flux) as adjacent and off the eval path.
- See the **end-to-end multimodal pipeline** and know that it's evaluated by the *same* discipline as the text pipeline — which is exactly what Lecture 2 builds.

Next: how to *measure* any of this — the four Ragas metrics, the LLM-as-judge, and the calibration step that makes the judge's number trustworthy. Continue to [Lecture 2 — Evaluation and the Calibrated LLM-as-Judge](./02-evaluation-and-llm-as-judge.md).

---

## References

- *CLIP: Learning Transferable Visual Models From Natural Language Supervision* — Radford et al., OpenAI, 2021: <https://arxiv.org/abs/2103.00020>
- *SigLIP: Sigmoid Loss for Language Image Pre-Training* — Zhai et al., 2023: <https://arxiv.org/abs/2303.15343>
- *ColPali: Efficient Document Retrieval with Vision Language Models* — Faysse et al., 2024: <https://arxiv.org/abs/2407.01449>
- *Qwen2.5-VL model card*: <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>
- *LLaVA model card*: <https://huggingface.co/llava-hf/llava-1.5-7b-hf>
- *Phi-3.5-Vision model card*: <https://huggingface.co/microsoft/Phi-3.5-vision-instruct>
- *InternVL model card*: <https://huggingface.co/OpenGVLab/InternVL2_5-8B>
- *Anthropic — vision (image-block message shape)*: <https://docs.anthropic.com/en/docs/build-with-claude/vision>
- *Whisper*: <https://github.com/openai/whisper> · *whisper.cpp*: <https://github.com/ggml-org/whisper.cpp> · *faster-whisper*: <https://github.com/SYSTRAN/faster-whisper>
- *Piper TTS*: <https://github.com/OHF-Voice/piper1-gpl> · *Coqui XTTS*: <https://github.com/coqui-ai/TTS>
- *SDXL*: <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0> · *FLUX.1*: <https://huggingface.co/black-forest-labs/FLUX.1-dev>
- *`sentence-transformers` image search (CLIP)*: <https://www.sbert.net/examples/applications/image-search/README.html>
