# Week 8 — Resources

Every resource here is **free** or has a free tier. The chunking libraries are open source. The extraction toolchain (PyMuPDF, Unstructured, MinerU, Surya, Tesseract) is open; LlamaParse has a free monthly page quota and an open-only fallback is documented for every lab. The late-chunking paper is on arXiv. Model cards live on Hugging Face.

Library names and APIs move every cohort — the *concepts* (chunk boundaries, overlap, the extraction→chunking→embedding pipeline, the A/B harness) are stable. When a specific page 404s, search the project's docs for the function name.

This week sits on top of week 7. The retrieval metrics (`Recall@5`, `MRR`) and the `crunchrag_embed` package come from there; the resources below assume you have that harness.

## Required reading (work it into your week)

- **LangChain text splitters** — the canonical reference for recursive character/token splitting, the separator hierarchy, and `chunk_size`/`chunk_overlap`. Read the recursive splitter section twice:
  <https://python.langchain.com/docs/concepts/text_splitters/>
- **Late Chunking paper** — Günther et al., *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models* (Jina AI, 2024). The mechanism that makes a chunk's vector "know" its surrounding context:
  <https://arxiv.org/abs/2409.04701>
- **PyMuPDF documentation** — the fast native-text PDF extractor; read `page.get_text()` modes (`"text"`, `"blocks"`, `"dict"`) and the table-finder:
  <https://pymupdf.readthedocs.io/>
- **Unstructured documentation** — the "messy mixed document" partitioner that returns *typed elements* (`Title`, `NarrativeText`, `Table`, `ListItem`); read the partitioning and chunking-by-title pages:
  <https://docs.unstructured.io/>

## The chunking strategy references

- **Pinecone — chunking strategies** — a practical survey of fixed/recursive/semantic chunking with the size/overlap trade-offs spelled out:
  <https://www.pinecone.io/learn/chunking-strategies/>
- **LlamaIndex — node parsers / chunking** — `SentenceSplitter`, `SemanticSplitterNodeParser`, and the metadata-attached-node model:
  <https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/>
- **`tiktoken`** — OpenAI's BPE tokenizer; you use it (or the HF tokenizer) to count tokens for token-window chunking so "512 tokens" means tokens, not characters:
  <https://github.com/openai/tiktoken>
- **Hugging Face tokenizers** — the `AutoTokenizer` you use when the embedding model isn't OpenAI's; BGE counts tokens with its own WordPiece tokenizer, and chunk size must be measured in *that* model's tokens:
  <https://huggingface.co/docs/transformers/main_classes/tokenizer>

## The extraction toolchain (have these open on Wednesday)

- **PyMuPDF (`pymupdf`)** — `pip install pymupdf`. Fast, C-backed, great on native-text PDFs; the first tool you reach for:
  <https://pymupdf.readthedocs.io/en/latest/the-basics.html>
- **Unstructured (`unstructured`)** — `pip install "unstructured[pdf]"`. Returns typed elements; the right tool for mixed docs (PDF + HTML + email + slides):
  <https://github.com/Unstructured-IO/unstructured>
- **MinerU** — the open scientific/complex-layout extractor (formulas, multi-column, reading order) from OpenDataLab; the tool for papers and dense technical PDFs:
  <https://github.com/opendatalab/MinerU>
- **LlamaParse** — LLM-powered parsing for *hard* documents (financial filings, scanned forms, nested tables). Free monthly page quota; an API key is required:
  <https://docs.cloud.llamaindex.ai/llamaparse/getting_started>

## OCR and tables

- **Tesseract** — the CPU OCR workhorse, via `pytesseract`. Install the engine (`apt install tesseract-ocr`) plus the Python wrapper:
  <https://github.com/tesseract-ocr/tesseract>
- **`pytesseract`** — the thin Python binding you'll actually call:
  <https://github.com/madmaze/pytesseract>
- **Surya** — modern, layout-aware OCR + line/layout/reading-order detection from Datalab; GPU-preferred with a CPU fallback:
  <https://github.com/datalab-to/surya>
- **`pdfplumber`** — focused table extraction from native PDFs (`page.extract_tables()`); a clean companion to PyMuPDF when the tables matter:
  <https://github.com/jsvine/pdfplumber>
- **`camelot`** — the other table-extraction option for lattice/stream-ruled tables:
  <https://github.com/camelot-dev/camelot>

## Evaluation (the A/B spine — reused from week 7)

- **pgvector README** — still your store; `vector_cosine_ops` + the `<=>` operator. The chunking A/B fixes this and varies only the chunker:
  <https://github.com/pgvector/pgvector>
- **MTEB leaderboard** — the *Retrieval* tab; you keep the embedding fixed (BGE-large) this week, but the leaderboard is where you'd pick it:
  <https://huggingface.co/spaces/mteb/leaderboard>
- **Ragas — faithfulness** — the LLM-as-judge "is the answer grounded in the retrieved context?" metric you'll use as the *secondary* signal behind Recall@5/MRR:
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/>

## Papers worth your time (free on arXiv)

- **Late Chunking** (Günther et al., Jina AI, 2024) — embed the whole document through a long-context model, then mean-pool per-chunk token spans. The headline late-chunking reference:
  <https://arxiv.org/abs/2409.04701>
- **Dense X Retrieval: What Retrieval Granularity Should We Use?** (Chen et al., 2023) — the "propositions vs sentences vs passages" study; the empirical case that chunk granularity is a tuned variable, not a default:
  <https://arxiv.org/abs/2312.06648>
- **LongEmbed / long-context embedding evaluation** — why the 512-token ceiling of older encoders forces chunking, and what changes when the encoder context grows:
  <https://arxiv.org/abs/2404.12096>
- **jina-embeddings-v3** (Günther et al., 2024) — the long-context model the late-chunking lab uses; task LoRA adapters + Matryoshka dims:
  <https://arxiv.org/abs/2409.10173>

## Models you'll use this week

- **`BAAI/bge-large-en-v1.5`** — the fixed embedding for the A/B (1024-dim, normalized). Same model as week 7, so chunking is the only variable:
  <https://huggingface.co/BAAI/bge-large-en-v1.5>
- **`jinaai/jina-embeddings-v3`** — the long-context (8192-token) model for the late-chunking leg; you embed the *whole document* through it, then pool:
  <https://huggingface.co/jinaai/jina-embeddings-v3>
- **`BAAI/bge-large-en-v1.5` tokenizer** — load it via `AutoTokenizer.from_pretrained(...)` to count tokens in the model's own units for token-window chunking.

## Tools you'll use this week

- **`pymupdf`** — `pip install pymupdf`. `import fitz` (or `import pymupdf`). Native-text PDF extraction.
- **`unstructured`** — `pip install "unstructured[pdf]"`. Typed-element partitioning for mixed docs.
- **`pytesseract` + `tesseract-ocr`** — OCR for scanned pages; pure CPU.
- **`surya-ocr`** — `pip install surya-ocr`. Layout-aware OCR; GPU-preferred.
- **`tiktoken` / `transformers`** — token counting for the token-window chunker.
- **`crunchrag_embed`** — your week-7 package. This week imports `evaluate()` and `store.py` **unchanged**.

## A note on the corpus

The exercises and mini-project run against the same small **legal corpus** as week 7 — a synthetic services agreement of ~50 clauses plus a 40-question gold set — but this week the *unchunked* document matters: you extract a multi-clause document, then chunk it, and the question is whether a given chunk strategy keeps each answer (e.g. clause_14 termination, clause_09 five-year confidentiality) inside a single retrievable chunk. The PDF exercise uses any real PDF you have; the chunking A/B uses the legal corpus so the gold set and `evaluate()` carry over from week 7 unchanged.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Chunk** | The unit of text that becomes one embedding/one vector — the retrieval granularity. |
| **Fixed token-window** | Split every N tokens (512/1024); the simplest baseline. |
| **Sliding window / overlap** | Windows that overlap by M tokens so an answer straddling a boundary survives. |
| **Recursive splitting** | Split on a separator hierarchy (paragraphs → sentences → words); LangChain's default. |
| **Semantic chunking** | Split where adjacent-sentence embedding distance spikes — a meaning boundary. |
| **Late chunking** | Embed the whole document first, then mean-pool token embeddings per chunk span (Jina 2024). |
| **Token** | The model's unit of text; chunk size is measured in *the embedding model's* tokens, not characters. |
| **Extraction** | Getting text (and structure) out of a source document — PDF, scan, HTML. |
| **OCR** | Optical Character Recognition — turning pixels of text into characters (Tesseract, Surya). |
| **Typed element** | Unstructured's output unit: `Title`, `NarrativeText`, `Table`, `ListItem`. |
| **Reading order** | The correct sequence of text in a multi-column or complex-layout page. |
| **Faithfulness** | Whether a generated answer is grounded in the retrieved context (LLM-as-judge / Ragas). |
| **Recall@k** | Fraction of relevant chunks found in the top-k results (from week 7). |
| **MRR** | Mean Reciprocal Rank — averages 1/(rank of first relevant chunk) (from week 7). |
| **A/B harness** | Fix embedding + store, vary only the chunker, compare metrics, pick a winner. |

---

*If a link 404s, please open an issue so we can replace it.*
