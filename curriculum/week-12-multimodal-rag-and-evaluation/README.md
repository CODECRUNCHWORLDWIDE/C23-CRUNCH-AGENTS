# Week 12 — Multimodal RAG and Evaluation

Welcome to the week that decides whether the last five weeks were real. You can chunk perfectly (week 8), rerank and fuse (week 9), run a production vector store (week 10), and budget context like a miser (week 11) — and still have no idea whether your pipeline actually *works*, because you never measured it. By Friday you will be able to look at any RAG system and state, with numbers, how faithful its answers are, whether its retriever recalls the right context, and how much you trust the judge that told you so. You will treat **evaluation as the deliverable**, not the afterthought — because if you cannot measure it, you cannot ship it.

This is the **last week of Phase II — RAG & Memory Systems**, and it sits on top of weeks 7–11. Everything here assumes you can embed a corpus, chunk it, rerank it, store it, and retrieve with a `crunchrag_embed.evaluate()` harness that returns Recall@5 and MRR. This week points that discipline at a new and harder target: not *did the right chunk come back* (a retrieval metric), but *is the generated answer grounded, complete, and relevant* (an answer metric) — and *how do you trust the LLM that judged it*. The headline deliverable is the **Phase II milestone: a Ragas evaluation report** across three pipeline variants, with a calibrated LLM-as-judge behind every number.

The one sentence to internalize before you read another line:

> **RAG without Ragas is a vibe.** A demo that "looks good" is an anecdote; a faithfulness score of 0.81 on a 40-question gold set across three variants is evidence. Phase II ends when you can show the evidence.

Here's why that's not hyperbole. Retrieval metrics (Recall@5, MRR) tell you the right context *came back*. They say nothing about whether the generator then *used* it, *hallucinated past* it, or answered a *different question*. Two pipelines with identical Recall@5 can produce wildly different answers — one grounded, one confabulated — and only an **answer-level** metric catches the difference. Ragas is the open-source standard for those metrics: faithfulness, context recall, context precision, answer relevancy. Each one catches a specific failure the others miss. Learning what each catches — and how to trust the judge that computes it — is the engineering of this week.

There's a corollary worth taping next to last week's mantra:

> **An LLM-as-judge you didn't calibrate is a random number generator with good vocabulary.** Before you believe a judge's 0.81, you check it against human labels — ten of them, minimum — and you compute the agreement. A judge that disagrees with humans is not measuring what you think it's measuring.

And because this is also a **multimodal** week, one more:

> **The figure is often the answer.** A pipeline that extracts only prose and silently drops every chart, table image, and diagram has thrown away exactly the content a reader would point at. Multimodal RAG — embedding page images, asking a vision-language model over a figure — is how you stop dropping it.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** the multimodal RAG landscape in 2026 — vision-language models (LLaVA, Qwen2.5-VL, Phi-3.5-Vision, InternVL), image embeddings (CLIP, SigLIP), and visual document retrieval (ColPali/ColQwen) — and state when you embed the *page image* versus the *extracted text*.
- **Build** a multimodal answer step: render a PDF page (or figure) to an image and ask both an **open VLM** and **Claude vision** the same question over it, then compare — the open-weights path and the frontier path, side by side.
- **Place** ASR (Whisper, whisper.cpp, faster-whisper) and TTS (Piper, XTTS) correctly as *pipeline-adjacent* capabilities — they get content *into* and *out of* a RAG system — and know image generation (SDXL, Flux) is adjacent, not core.
- **Define** the four Ragas metrics precisely — **faithfulness**, **context recall**, **context precision**, **answer relevancy** — and state exactly what failure each one catches and the others miss.
- **Run** a Ragas `EvaluationDataset` + `evaluate()` with *any* judge backend: a local open model (vLLM/Ollama OpenAI-compatible endpoint) **or** Claude (`claude-opus-4-8`) — the open path and the frontier path, same metrics.
- **Calibrate** an LLM-as-judge against human labels: collect 10 labeled examples, compute agreement and **Cohen's kappa**, sweep the judge threshold, and pick the calibrated operating point — so the judge's number means something.
- **Survey** the eval-tooling landscape — DeepEval, promptfoo, TruLens — and know what each adds over raw Ragas (assertions/CI, prompt-matrix testing, trace-level instrumentation).
- **Name and avoid** the judge pitfalls: self-preference (judging with the model that generated the answer), position bias, verbosity bias, and the cost of a max-effort judge on a 40-question gold set.

## Prerequisites

This week assumes you have completed **C23 weeks 1–11**, or have equivalent fluency. Specifically:

- You finished **weeks 7–11** and have a working Phase II pipeline: embed → chunk → (rerank/hybrid) → store → retrieve, with a `crunchrag_embed.evaluate()` that returns Recall@5 and MRR on a 40-query gold set. **This week imports that harness and wraps an answer-quality layer around it** — if it's broken, fix it first.
- Python 3.12 on Linux, macOS, or WSL2; a virtualenv you can `pip install` into; Docker for pgvector (from week 10).
- You can call the Anthropic SDK (`pip install anthropic`) from earlier weeks. We use `claude-opus-4-8` as the LLM-as-judge with adaptive thinking; you will also wire an open-model judge so every lab has an open path.
- You're comfortable with Recall@5 and MRR as *retrieval* metrics. This week adds *answer* metrics on top of them — Ragas's faithfulness/context-recall/context-precision/answer-relevancy sit above your retrieval numbers, not instead of them.

You do **not** need a GPU for the Ragas/judge work (it's API or small-model-driven). A VLM and CLIP/SigLIP run faster on a GPU but have CPU fallbacks, and the multimodal exercise's open-VLM leg degrades gracefully if you have no GPU — the Claude-vision leg always works over the API.

## Topics covered

- **Vision-language models (VLMs):** LLaVA, Qwen2.5-VL, Phi-3.5-Vision, InternVL via transformers or Ollama; what a VLM is (an image encoder bolted onto an LLM) and what it's for (reading a figure, a chart, a scanned page) in a RAG pipeline.
- **Image embeddings:** CLIP and SigLIP via `sentence-transformers`/`open_clip` — one shared space for images *and* text, so a text query can retrieve an image. The "embed the page image vs embed the extracted text" decision.
- **Multimodal RAG over PDFs with figures:** the two architectures — (a) extract+describe (VLM captions the figure → text chunk) and (b) embed-the-page-image (CLIP/ColPali retrieves the rendered page) — and when each wins.
- **Visual document retrieval:** ColPali/ColQwen — retrieve over page *images* directly with late-interaction (ColBERT-style) scoring, skipping fragile OCR/extraction entirely. Worth knowing exists.
- **ASR and TTS, pipeline-adjacent:** Whisper / whisper.cpp / faster-whisper turn audio into the *text* your RAG indexes; Piper / XTTS turn an *answer* back into speech. Adjacent capabilities, placed correctly.
- **Image generation, adjacent:** SDXL and Flux — you'll meet them, you won't build them into the eval; they're a generation capability, not a retrieval or eval one.
- **Ragas — the standard retrieval eval:** the four metrics (faithfulness, context_recall, context_precision, answer_relevancy), each defined precisely with what it catches; the `EvaluationDataset` + `evaluate()` harness; running it with Claude **or** a local model as the judge backend.
- **LLM-as-judge with calibration:** what a judge is, why you never trust it raw, the 10-human-label calibration step, agreement and **Cohen's kappa**, threshold tuning, and the bias taxonomy (self-preference, position, verbosity).
- **The eval-tooling survey:** DeepEval (assertions + pytest/CI), promptfoo (prompt/model matrix testing), TruLens (trace-level feedback functions) — what each adds and when you reach for it.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                          | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|---------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Multimodal RAG; VLMs; CLIP/SigLIP; page-image vs text         |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Visual retrieval (ColPali); ASR/TTS adjacent; exercise 1      |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Ragas: the four metrics, what each catches; from-scratch lab  |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | LLM-as-judge; calibration; Cohen's kappa; the judge harness   |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The Ragas report run + variant comparison; tooling survey     |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work (`crunchrag_eval`)                     |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, milestone-report polish                         |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                               | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | Ragas/DeepEval/promptfoo/TruLens docs, CLIP/SigLIP/ColPali, the VLM model cards, Whisper/Piper/XTTS, SDXL/Flux, models/tools, the glossary cheat-sheet |
| [lecture-notes/01-multimodal-rag.md](./lecture-notes/01-multimodal-rag.md) | VLMs, image embeddings, multimodal RAG over PDF figures, the page-image-vs-text question, ColPali, ASR/TTS/image-gen as adjacent |
| [lecture-notes/02-evaluation-and-llm-as-judge.md](./lecture-notes/02-evaluation-and-llm-as-judge.md) | The four Ragas metrics, the eval harness, LLM-as-judge with calibration (Cohen's kappa, threshold tuning), the DeepEval/promptfoo/TruLens survey, judge pitfalls |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-vlm-vs-claude-over-a-figure.md](./exercises/exercise-01-vlm-vs-claude-over-a-figure.md) | Render a PDF page to an image, ask an open VLM and Claude vision the same question, compare; plus a CLIP image+text retrieval |
| [exercises/exercise-02-ragas-metrics-from-scratch.py](./exercises/exercise-02-ragas-metrics-from-scratch.py) | Implement the four Ragas-style metrics from scratch on a fixed dataset so they stop being magic |
| [exercises/exercise-03-judge-calibration-sweep.py](./exercises/exercise-03-judge-calibration-sweep.py) | Calibrate an LLM-as-judge against 10 human labels: agreement, Cohen's kappa, threshold sweep, pick the operating point |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge (the Phase II milestone in lab form) |
| [challenges/challenge-01-ragas-report-three-variants.md](./challenges/challenge-01-ragas-report-three-variants.md) | The headline lab: Ragas suite over three pipeline variants, calibrated judge, four-metrics × three-variants plot, identify which metric moved most for which change |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the Ragas evaluation report memo (the Phase II milestone) |
| [mini-project/README.md](./mini-project/README.md) | The `crunchrag_eval` package — Ragas + calibrated judge + multimodal-aware answerer, importable, with a CLI |

## The "the score survived calibration" promise

C23 uses a recurring marker for every exercise that ends in a number you can actually trust — a metric whose judge was *checked against humans* before you believed it:

```
$ python ragas_report.py --variant reranker --gold gold_40.json --judge claude
variant=reranker  judge=claude-opus-4-8 (calibrated, threshold=0.62, kappa=0.71)
  faithfulness:       0.88   context_recall:    0.91
  context_precision:  0.79   answer_relevancy:  0.84
  q12 ("five-year confidentiality duration") faithful=1.0 grounded in ctx_09 ✓
     answer: "Confidential information must be protected for five years after termination."
```

If that faithfulness 0.88 comes from a judge that agrees with humans only 50% of the time (kappa near zero), it's a coin flip wearing a decimal point. The point of week 12 is to make the score *mean something* — a calibrated judge, a real gold set, three variants compared on the same axis — and to prove it with a Cohen's kappa, not a vibe about which pipeline "feels" better.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **Ragas metrics documentation** end to end until you can explain, for each metric, the exact LLM sub-calls it makes (faithfulness decomposes the answer into claims and verifies each; context precision asks "is this retrieved chunk relevant to the question?" per chunk): <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>. Then re-derive your from-scratch metrics (Exercise 2) to match Ragas's decomposition exactly.
- Read the **ColPali paper** (Faysse et al., 2024) until you can explain why retrieving over page *images* with late-interaction scoring beats an OCR→chunk→embed pipeline on figure-heavy documents: <https://arxiv.org/abs/2407.01449>. Then index 20 PDF pages with ColQwen and compare its retrieval to your text pipeline by eye.
- Build a **second judge** with a *different* model (a local Qwen or Llama, vs. Claude) and measure inter-judge agreement (kappa between the two judges). Where they disagree is where your metric is least trustworthy.
- Add a **verbosity-bias probe**: take 10 answers, pad half with correct-but-redundant filler, and check whether the judge scores the padded ones higher. If it does, your judge has a length bias you must correct for.

## Up next

This is the **end of Phase II.** Your Ragas evaluation report — four metrics, three variants, a calibrated judge — is the Phase II milestone and the centerpiece of your architecture review. Push your `crunchrag_eval` mini-project before you move on.

**Week 13 opens Phase III — Agent Architectures**, starting with **LangGraph**: you stop measuring a single retrieval-then-generate pipeline and start building *agents* — graphs of tool-using, branching, looping LLM calls. The eval discipline you built this week comes with you: an agent without an eval harness is the same vibe-shipped liability as a RAG pipeline without Ragas, just with more moving parts. Phase III consumes your `crunchrag_eval` suite to grade agent trajectories, not just answers. The measurement habit you earned here is the thing that keeps the agents honest.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
