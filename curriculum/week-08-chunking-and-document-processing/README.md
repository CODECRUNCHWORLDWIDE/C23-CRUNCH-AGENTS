# Week 8 — Chunking and Document Processing

Welcome to the week that decides whether last week mattered. You can pick the best embedding on MTEB, tune `ef_search` to the elbow, and index a million vectors flawlessly — and still retrieve garbage, because you chunked the corpus wrong. By Friday you will be able to look at any RAG pipeline and state, with evidence, what its chunking strategy is, why, and what it costs in retrieval quality to get it wrong. You will treat **chunk size as a hyperparameter** — something you A/B test and tune, not something you guess once and forget.

This is week 2 of **Phase II — RAG & Memory Systems**, and it sits on top of week 7. Everything here assumes you can embed a corpus, index it in pgvector, and *measure* retrieval with the `crunchrag_embed` harness you built. This week you point that harness at a new variable: not *which model* embeds the text, but *how the text is cut up before it's embedded*. The headline lab is a chunking A/B harness, and it imports last week's `evaluate()` unchanged.

The one sentence to internalize before you read another line:

> **Chunking is the part of RAG that determines whether the rest of RAG is doing anything.**

Here's why that's not hyperbole. An embedding represents one chunk as one vector. If your chunk is a whole 30-page document, that single vector is a blurry average of thirty pages — too coarse to match a specific query. If your chunk is one sentence, you've shredded the context — the answer spans three sentences you split apart, and no single chunk contains it. The right chunk is the *unit of meaning that answers a question*: big enough to be self-contained, small enough to be specific. Finding that unit, for *your* corpus, is the engineering of this week. And — like everything in retrieval — you find it by measuring, not by guessing.

There's a corollary worth taping next to last week's mantra:

> **A retrieval failure is more often a chunking failure than an embedding failure.** When Recall@5 is bad, suspect the chunks before you suspect the model.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** why chunking exists at all — the embedding-is-one-vector constraint, the LLM context budget, and the tension between specificity and self-containedness.
- **Enumerate** the chunking strategies in practical use in 2026 — fixed token-window, sliding-window with overlap, recursive (LangChain-style), semantic-paragraph, and **late chunking** — and state the trade-off each makes.
- **Choose** a chunk size and overlap for a corpus class (legal clauses, prose, code, tables) and justify it as a tuned hyperparameter, not a default.
- **Extract** text from real documents with the 2026 toolchain — **PyMuPDF** for fast native PDFs, **Unstructured** for messy mixed documents, **MinerU** and **LlamaParse** for hard layouts — and know which tool fits which document.
- **Run** OCR with **Tesseract** (the workhorse) and **Surya** (the modern layout-aware option) on scanned pages, and handle table extraction and multimodal (text + figure + caption) pages.
- **Build** a chunking-strategy A/B harness: fix the embedding and the vector store, vary only the chunker, and report retrieval **MRR**, **Recall@5**, and answer **faithfulness** delta — then pick a winner with reasons.
- **Reason** about the document-processing pipeline end to end: extraction → cleaning → chunking → metadata → embedding, and where each stage silently corrupts retrieval if you get it wrong.
- **Understand** late chunking specifically — why embedding the whole document *first* and pooling chunk vectors *after* preserves cross-chunk context that naive chunking destroys.

## Prerequisites

This week assumes you have completed **C23 weeks 1–7**, or have equivalent fluency. Specifically:

- You finished **week 7** and have the `crunchrag_embed` mini-project: you can embed a corpus, index it in pgvector, and run `evaluate()` to get top-1 / Recall@5 / MRR. **This week imports that module directly** — if it's broken, fix it first.
- Python 3.12 on Linux, macOS, or WSL2; a virtualenv you can `pip install` into; Docker for pgvector.
- You can read a PDF's structure conceptually — pages, text runs, images — even if you've never parsed one programmatically. We start at "how do I get text out of a PDF" and build up.
- You're comfortable with the week-7 retrieval metrics. We reuse Recall@5 and MRR all week and add answer **faithfulness** as a new signal.

You do **not** need a GPU for the chunking work (it's CPU-bound text processing). **Surya** OCR and some layout models prefer a GPU but have CPU fallbacks; **Tesseract** is pure CPU. You do **not** need prior document-AI experience — we cover the extraction toolchain from scratch.

## Topics covered

- **Why chunk at all:** the one-vector-per-chunk constraint, the LLM context budget, and the specificity-vs-self-containedness tension that every strategy negotiates.
- **Fixed token-window chunking:** split every N tokens (512, 1024); the simplest baseline, its blindness to structure, and why it's still the right *first* thing to try.
- **Sliding-window with overlap:** windows that overlap by M tokens so an answer that straddles a boundary survives in at least one chunk; choosing the overlap; the storage cost of redundancy.
- **Recursive chunking (LangChain-style):** split on a hierarchy of separators (paragraphs → sentences → words) so chunks respect structure where it exists; the default that ships in most frameworks.
- **Semantic-paragraph chunking:** split where the *meaning* shifts (embedding-distance between adjacent sentences); more faithful to the document, more expensive to compute.
- **Late chunking (Jina, 2024):** embed the *whole* document through the model first, then pool per-chunk vectors from the token embeddings — so each chunk's vector "knows" the surrounding context that naive chunking throws away.
- **Document extraction:** **PyMuPDF** (fast, native-text PDFs), **Unstructured** (messy mixed documents, returns typed elements), **MinerU** (scientific/complex layouts), **LlamaParse** (LLM-powered parsing of hard documents); choosing by document type.
- **OCR and tables:** **Tesseract** (the CPU workhorse), **Surya** (layout-aware, modern), table extraction, and the multimodal page (text + figure + caption) problem.
- **Chunking as evaluation:** the A/B harness — fix embedding + store, vary the chunker, report MRR / Recall@5 / faithfulness delta — and the discipline of changing one variable at a time.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                       | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Why chunk; fixed/sliding/recursive strategies; chunk size   |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Semantic + late chunking; the chunker exercises             |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Document extraction (PyMuPDF/Unstructured/MinerU); OCR       |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | The A/B methodology; building the chunking harness          |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The A/B run + winner memo; extraction clinic                |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                       |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                   |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                             | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The chunking strategy references, the extraction toolchain docs, OCR docs, and the late-chunking paper |
| [lecture-notes/01-chunking-strategies.md](./lecture-notes/01-chunking-strategies.md) | Why chunk, the five strategies (fixed/sliding/recursive/semantic/late), and chunk size as a hyperparameter |
| [lecture-notes/02-document-extraction-and-evaluation.md](./lecture-notes/02-document-extraction-and-evaluation.md) | The extraction toolchain, OCR/tables/multimodal, and the chunking A/B methodology |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-extract-a-pdf.md](./exercises/exercise-01-extract-a-pdf.md) | Extract text from a real PDF three ways and compare what each tool gets right |
| [exercises/exercise-02-chunkers.py](./exercises/exercise-02-chunkers.py) | Implement fixed, sliding-window, and recursive chunkers and inspect their boundaries |
| [exercises/exercise-03-chunk-size-sweep.py](./exercises/exercise-03-chunk-size-sweep.py) | Sweep chunk size against the gold set and chart Recall@5 vs chunk size |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-chunking-ab.md](./challenges/challenge-01-chunking-ab.md) | The full chunking A/B: four strategies, one embedding, report MRR/Recall@5/faithfulness, pick a winner |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the one-page chunking A/B memo |
| [mini-project/README.md](./mini-project/README.md) | The `crunchrag_chunk` A/B harness — pluggable chunkers + the eval loop |

## The "the answer survived the chunking" promise

C23 uses a recurring marker for every exercise that ends in retrieval actually working *because* the chunking was right:

```
$ python chunk_ab.py --strategy recursive --size 512 --overlap 64
strategy=recursive size=512 overlap=64
  Recall@5: 0.88  MRR: 0.74  faithfulness: 0.81
  q12 ("five-year confidentiality duration") -> chunk_09 (rank 1) ✓
     "...confidential information must be protected for five years after termination."
```

If that confidentiality answer lands in *two* chunks (split mid-clause) or in *none* (swallowed by a giant chunk), the retrieval fails no matter how good your embedding is. The point of week 8 is to make the chunk *contain the answer* — and to prove it with a measured Recall@5 delta between strategies, not a vibe about which "feels" better.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **Late Chunking paper** (Günther et al., Jina, 2024) until you can explain why pooling chunk vectors from a full-document forward pass beats embedding chunks independently: <https://arxiv.org/abs/2409.04701>. Then implement it against `jina-embeddings-v3` and measure the lift on your corpus.
- Build a **semantic chunker from scratch**: embed each sentence, compute the cosine distance between adjacent sentences, and split where the distance spikes (a meaning boundary). Compare its boundaries to recursive chunking's by eye.
- Take a **scanned PDF** (print a page and photograph it, or find a scan) and run it through Tesseract *and* Surya. Diff the extracted text. Where does each fail — multi-column? tables? rotated text?
- Add a **metadata-aware** chunker that attaches the section heading to each chunk before embedding (so a clause about "Termination" carries that word even if the clause body doesn't). Measure whether the heading injection moves Recall@5.

## Up next

Week 9 takes the chunking literacy you built here and adds the cheapest meaningful win in RAG: **reranking, hybrid search, and structured retrieval.** You'll take your best chunking strategy from this week, add BM25 lexical search, fuse it with your dense retrieval, and put a reranker on top — measuring the lift at each layer. Push your mini-project before you start it; week 9 builds directly on the `crunchrag_chunk` harness.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
