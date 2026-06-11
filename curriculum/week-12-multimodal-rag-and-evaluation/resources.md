# Week 12 — Resources

Every resource here is **free** or has a free tier. Ragas, DeepEval, promptfoo, and TruLens are open source. The multimodal stack (CLIP, SigLIP, the VLMs, ColPali, Whisper, Piper, XTTS, SDXL, Flux) is open-weights and self-hostable. The Anthropic SDK is the *frontier* path for the LLM-as-judge — every lab has an open-model judge path documented next to it, never as the only option.

Library names and APIs move every cohort — the *concepts* (the four Ragas metrics, judge calibration, Cohen's kappa, the page-image-vs-text decision) are stable. When a specific page 404s, search the project's docs for the function name.

This week sits on top of weeks 7–11. The retrieval metrics (`Recall@5`, `MRR`) and the `crunchrag_embed` package come from there; the resources below assume you have that harness, and Ragas's *answer* metrics layer on top of your *retrieval* metrics, not instead of them.

## Required reading (work it into your week)

- **Ragas — available metrics** — the canonical reference for faithfulness, context recall, context precision, and answer relevancy. Read each metric's "how it's computed" section; the LLM sub-calls are the whole point:
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>
- **Ragas — evaluating a RAG application** — the `EvaluationDataset` + `evaluate()` workflow, the `SingleTurnSample` shape, and how to plug a custom LLM/embeddings backend:
  <https://docs.ragas.io/en/stable/getstarted/rag_evaluation/>
- **Anthropic — vision** — the image-block message shape (`{"type":"image","source":{...}}`) you use for a VLM judge/answerer over PDF page images, and the supported media types:
  <https://docs.anthropic.com/en/docs/build-with-claude/vision>
- **CLIP (OpenAI) — Learning Transferable Visual Models From Natural Language Supervision** — the paper that put images and text in one embedding space, so a text query can retrieve an image:
  <https://arxiv.org/abs/2103.00020>

## The evaluation toolchain (have these open Wednesday–Friday)

- **Ragas docs (home)** — the standard open-source RAG eval; metrics, datasets, and LLM/embedding wrappers. The spine of this week:
  <https://docs.ragas.io/>
- **Ragas — bring your own LLM** — how to wrap *any* model (a local vLLM/Ollama OpenAI-compatible endpoint, or Claude) as the metric judge backend; this is what makes the open path and the frontier path use identical metrics:
  <https://docs.ragas.io/en/stable/howtos/customizations/customize_models/>
- **DeepEval** — `pip install deepeval`. The "Pytest for LLMs": metric *assertions* you run in CI (`assert_test`, `GEval`, faithfulness/relevancy metrics). Reach for it when you want eval in your test suite, not just a report:
  <https://github.com/confident-ai/deepeval>
- **promptfoo** — `npx promptfoo@latest init`. Prompt/model *matrix* testing: run the same eval across many prompts × many models and diff the table. The right tool for "which prompt/model wins?", config-driven:
  <https://www.promptfoo.dev/>
- **TruLens** — `pip install trulens`. Trace-level *feedback functions* (groundedness, context relevance, answer relevance) instrumented into a running app, with a dashboard. Reach for it when you want per-call instrumentation, not a batch report:
  <https://www.trulens.org/>

## Vision-language models (the VLM model cards)

- **Qwen2.5-VL** (Alibaba) — the strong open VLM in 2026; reads documents, charts, and figures well. The default open VLM for this week's multimodal exercise:
  <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>
- **LLaVA** — the original open "image encoder + LLM" recipe; the reference architecture every later VLM iterates on:
  <https://huggingface.co/llava-hf/llava-1.5-7b-hf>
- **Phi-3.5-Vision** (Microsoft) — a small, efficient VLM that runs on modest hardware; the lightweight option when you can't fit a 7B+ VLM:
  <https://huggingface.co/microsoft/Phi-3.5-vision-instruct>
- **InternVL** (OpenGVLab) — a high-accuracy open VLM family, strong on document and chart understanding; the heavier open option:
  <https://huggingface.co/OpenGVLab/InternVL2_5-8B>
- **Ollama (vision models)** — the easiest local path to a VLM: `ollama run qwen2.5vl` or `ollama run llava`, then an OpenAI-compatible API with image input. The zero-friction open VLM path:
  <https://ollama.com/library>

## Image embeddings and visual retrieval

- **`sentence-transformers` — CLIP models** — `SentenceTransformer("clip-ViT-B-32")` embeds both images and text into one space; the simplest image-embedding path:
  <https://www.sbert.net/examples/applications/image-search/README.html>
- **SigLIP** (Google) — a CLIP successor with a sigmoid loss; stronger zero-shot image-text matching, drop-in for CLIP in the same one-space recipe:
  <https://huggingface.co/google/siglip-so400m-patch14-384>
- **`open_clip`** — the open re-implementation of CLIP with many checkpoints (LAION-trained); reach for it when you want a specific CLIP variant `sentence-transformers` doesn't ship:
  <https://github.com/mlfoundations/open_clip>
- **ColPali / ColQwen (the `colpali-engine`)** — visual document retrieval: embed page *images* with a VLM backbone and score with ColBERT-style late interaction, skipping OCR entirely. The "retrieve over the page picture" approach:
  <https://github.com/illuin-tech/colpali>
- **ColPali paper** — Faysse et al., 2024, the method and the ViDoRe benchmark that motivates retrieving over rendered pages instead of extracted text:
  <https://arxiv.org/abs/2407.01449>

## ASR and TTS (pipeline-adjacent: content in, speech out)

- **Whisper** (OpenAI) — the open ASR model that turns audio into the text your RAG indexes; the reference ASR. `pip install openai-whisper`:
  <https://github.com/openai/whisper>
- **whisper.cpp** — the C/C++ port that runs Whisper fast on CPU (and Apple Silicon), no Python/GPU required; the self-hosted, low-resource ASR path:
  <https://github.com/ggml-org/whisper.cpp>
- **faster-whisper** — the CTranslate2 reimplementation, much faster than reference Whisper on GPU with the same accuracy; the production ASR path. `pip install faster-whisper`:
  <https://github.com/SYSTRAN/faster-whisper>
- **Piper** (Rhasspy/OHF) — a fast, local neural TTS that turns an answer back into speech on CPU; the lightweight self-hosted TTS:
  <https://github.com/OHF-Voice/piper1-gpl>
- **XTTS / Coqui TTS** — higher-quality, voice-cloning-capable TTS; the heavier self-hosted TTS option:
  <https://github.com/coqui-ai/TTS>

## Image generation (adjacent — you'll meet it, not build it into the eval)

- **SDXL** (Stability AI) — Stable Diffusion XL via the `diffusers` library; the open text-to-image workhorse. Adjacent to RAG, not part of the eval:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>
- **FLUX.1** (Black Forest Labs) — the strong open image-generation family in 2026; the modern alternative to SDXL:
  <https://huggingface.co/black-forest-labs/FLUX.1-dev>
- **`diffusers`** — the Hugging Face library that runs both SDXL and Flux with a uniform pipeline API:
  <https://github.com/huggingface/diffusers>

## Papers and references worth your time (free)

- **G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment** (Liu et al., 2023) — the LLM-as-judge with chain-of-thought-and-form-filling recipe; the empirical case that a well-prompted judge can align with humans:
  <https://arxiv.org/abs/2303.16634>
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** (Zheng et al., 2023) — the taxonomy of judge biases (position, verbosity, self-enhancement) you must defend against, with measured magnitudes:
  <https://arxiv.org/abs/2306.05685>
- **RAGAS: Automated Evaluation of Retrieval Augmented Generation** (Es et al., 2023) — the paper behind the library; how faithfulness/relevance/context metrics are defined and validated:
  <https://arxiv.org/abs/2309.15217>
- **SigLIP: Sigmoid Loss for Language Image Pre-Training** (Zhai et al., 2023) — why the sigmoid loss beats CLIP's softmax contrastive loss for image-text matching:
  <https://arxiv.org/abs/2303.15343>
- **Cohen's kappa** — the inter-rater agreement statistic you compute to calibrate the judge against humans; the Wikipedia entry is a clean reference for the formula and interpretation bands:
  <https://en.wikipedia.org/wiki/Cohen%27s_kappa>

## Models you'll use this week

- **`claude-opus-4-8`** — the LLM-as-judge (most capable; adaptive thinking, `output_config={"effort":"high"}`, structured outputs via `messages.parse`). The frontier judge path. **No** `temperature`/`top_p`/`top_k` and **no** date suffix on the model id.
- **`claude-sonnet-4-6`** / **`claude-haiku-4-5`** — the cheaper Claude judges for cost-sensitive sweeps; same SDK, same adaptive-thinking shape.
- **A local OpenAI-compatible judge** — a vLLM or Ollama endpoint serving Qwen/Llama, wrapped as the Ragas judge backend. The open path, so no lab requires an API key.
- **`Qwen/Qwen2.5-VL-7B-Instruct`** (or `ollama run qwen2.5vl`) — the open VLM for the multimodal exercise's open-weights leg.
- **`clip-ViT-B-32`** (via `sentence-transformers`) — the image+text embedding model for the CLIP retrieval exercise.
- **`BAAI/bge-large-en-v1.5`** — still your text embedding from weeks 7–11; the answers you evaluate come from a pipeline built on it.

## Tools you'll use this week

- **`ragas`** — `pip install ragas`. The four metrics + `EvaluationDataset` + `evaluate()`. The spine of the week.
- **`anthropic`** — `pip install anthropic`. `client = anthropic.Anthropic()`; `client.messages.create(...)` and `client.messages.parse(...)` for the structured-output judge.
- **`deepeval`** / **`promptfoo`** / **`trulens`** — the survey tools; install only the one you survey hands-on.
- **`sentence-transformers`** — CLIP/SigLIP image+text embeddings; you already have it from week 7.
- **`transformers`** + **`torch`** (or **Ollama**) — to run an open VLM locally for the multimodal exercise.
- **`pymupdf`** — `pip install pymupdf`. Render a PDF page to an image (`page.get_pixmap`) for the multimodal leg; from week 8.
- **`scikit-learn`** — `pip install scikit-learn`. `cohen_kappa_score` for judge calibration (the exercises also ship a from-scratch kappa so the dependency is optional).
- **`matplotlib`** — `pip install matplotlib`. The four-metrics × three-variants plot for the milestone report.
- **`crunchrag_embed`** — your week-7 package. This week imports `evaluate()` and `store.py` **unchanged** and wraps an answer-quality layer (Ragas) around them.

## A note on the corpus

The exercises and mini-project run against the same small **legal corpus** as weeks 7–11 — a synthetic services agreement of ~50 clauses plus a 40-question gold set — so the retrieval metrics and `evaluate()` carry over unchanged. This week adds, per gold question, a **reference answer** (the ground-truth response) and the **retrieved contexts**, because Ragas's answer metrics need the generated answer, the reference, and the contexts — not just "which clause is relevant." The multimodal exercise uses a **PDF page with a figure or table** (render any real PDF, or the generated `services_agreement.pdf` from week 8, to an image): the point is to ask a VLM a question over the *picture* of the page, which the text pipeline can't do when the answer lives in a chart. The calibration step adds **10 human-labeled examples** (you label them — agreeing or disagreeing with a reference answer) so the judge can be checked against a human ground truth.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? Catches hallucination. (Ragas) |
| **Context recall** | Did retrieval bring back *all* the context the reference answer needs? Catches a retriever that misses. (Ragas) |
| **Context precision** | Are the retrieved chunks *relevant* (and ranked high), or is the context padded with junk? Catches noisy retrieval. (Ragas) |
| **Answer relevancy** | Does the answer actually address the question asked, or wander? Catches off-topic/evasive answers. (Ragas) |
| **LLM-as-judge** | Using an LLM to score an answer (faithful? relevant?) instead of a human or exact-match. |
| **Calibration** | Checking the judge's scores against human labels before you trust them. |
| **Cohen's kappa** | Inter-rater agreement corrected for chance; how much judge and human agree *beyond luck*. |
| **Self-preference bias** | A judge rates answers from its own model family higher. Don't judge with the generator. |
| **Position bias** | A judge favors the first (or last) option in a pairwise comparison regardless of content. |
| **Verbosity bias** | A judge rates longer answers higher even when the extra length is filler. |
| **VLM** | Vision-language model — an image encoder bolted onto an LLM (LLaVA, Qwen2.5-VL, Phi-Vision, InternVL). |
| **CLIP / SigLIP** | Image+text embedding models: one shared space, so a text query retrieves an image. |
| **ColPali / ColQwen** | Visual document retrieval — embed page *images* and score with ColBERT-style late interaction; skips OCR. |
| **ASR** | Automatic Speech Recognition — audio → text (Whisper, whisper.cpp, faster-whisper). Gets content *into* RAG. |
| **TTS** | Text-to-Speech — text → audio (Piper, XTTS). Gets the answer *out of* RAG. |
| **EvaluationDataset** | Ragas's container of samples (question, answer, contexts, reference) that `evaluate()` scores. |
| **Recall@k / MRR** | The *retrieval* metrics from week 7; Ragas's *answer* metrics sit on top of them. |

---

*If a link 404s, please open an issue so we can replace it.*
