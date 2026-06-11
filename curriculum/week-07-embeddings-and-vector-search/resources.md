# Week 7 — Resources

Every resource here is **free** or has a free tier. The embedding model cards live on Hugging Face. The MTEB leaderboard is open. The pgvector and HNSW docs are open. Papers are on arXiv. Where a vendor API is referenced (OpenAI, Cohere, Voyage), the docs are public and the comparison lab uses a few cents of credit; an open-only path is documented for every lab.

Model names and leaderboard positions move every cohort — the *concepts* (encoder embeddings, ANN indexes, retrieval metrics) are stable. When a specific model card 404s, search Hugging Face for the family name (`BAAI/bge`, `Alibaba-NLP/gte`, `jinaai/jina-embeddings`).

## Required reading (work it into your week)

- **Sentence-Transformers documentation** — the library you will use to run open embeddings, with the pooling and normalization details that bite:
  <https://www.sbert.net/>
- **MTEB leaderboard** — the Massive Text Embedding Benchmark, on Hugging Face Spaces. Read the *Retrieval* tab, not the overall average:
  <https://huggingface.co/spaces/mteb/leaderboard>
- **pgvector README** — the canonical reference for the `vector` type, the distance operators, and the HNSW/IVFFlat index syntax. Read the indexing section twice:
  <https://github.com/pgvector/pgvector>
- **OpenAI embeddings guide** — the `text-embedding-3` family, the `dimensions` parameter, and the cost table:
  <https://platform.openai.com/docs/guides/embeddings>

## Model cards (have these open on Monday)

- **BGE — `BAAI/bge-large-en-v1.5`** — the open default for English; note the query instruction prefix:
  <https://huggingface.co/BAAI/bge-large-en-v1.5>
- **BGE-M3 — `BAAI/bge-m3`** — multilingual, multi-granularity (dense + sparse + ColBERT in one model); you'll meet its sparse output again in week 9:
  <https://huggingface.co/BAAI/bge-m3>
- **GTE — `Alibaba-NLP/gte-large-en-v1.5`** — strong English retrieval, 8192-token context:
  <https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5>
- **jina-embeddings-v3 — `jinaai/jina-embeddings-v3`** — task-specific LoRA adapters + Matryoshka dimensions:
  <https://huggingface.co/jinaai/jina-embeddings-v3>
- **nomic-embed-text-v1.5 — `nomic-ai/nomic-embed-text-v1.5`** — long context, Matryoshka, fully open training data:
  <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>
- **E5-Mistral — `intfloat/e5-mistral-7b-instruct`** — a 7B LLM repurposed as an embedder; powerful, heavy:
  <https://huggingface.co/intfloat/e5-mistral-7b-instruct>

## Vendor embedding docs

- **Cohere `embed-v3`** — note the `input_type` parameter (`search_document` vs `search_query`) — getting it wrong silently tanks recall:
  <https://docs.cohere.com/docs/embeddings>
- **Voyage AI embeddings** — `voyage-3` and the domain-specific lines (`voyage-code`, `voyage-law`):
  <https://docs.voyageai.com/docs/embeddings>

## ANN index references (the ones you'll re-read on Wednesday)

- **HNSW paper** — Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*:
  <https://arxiv.org/abs/1603.09320>
- **FAISS wiki — index selection guide** — the practical "which index for how many vectors" decision table, even if you use pgvector:
  <https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index>
- **ScaNN** — Google's anisotropic-quantization library and the paper behind it:
  <https://github.com/google-research/google-research/tree/master/scann>
- **pgvector HNSW tuning** — the `m`, `ef_construction`, and `ef_search` parameters and how they trade recall for speed:
  <https://github.com/pgvector/pgvector#hnsw>

## Papers worth your time (free on arXiv)

- **Text Embeddings by Weakly-Supervised Contrastive Pre-training** (Wang et al., 2022) — the E5 paper; the clearest exposition of how modern retrieval embeddings are trained:
  <https://arxiv.org/abs/2212.03533>
- **MTEB: Massive Text Embedding Benchmark** (Muennighoff et al., 2022) — read the task taxonomy section so you know what the leaderboard is actually measuring:
  <https://arxiv.org/abs/2210.07316>
- **Matryoshka Representation Learning** (Kusupati et al., 2022) — why nomic and jina let you truncate the vector and keep most of the quality:
  <https://arxiv.org/abs/2205.13147>
- **Billion-scale similarity search with GPUs** (Johnson et al., 2017) — the FAISS paper; the IVF and product-quantization ideas you'll meet in pgvector and Qdrant:
  <https://arxiv.org/abs/1702.08734>

## Tools you'll use this week

- **`sentence-transformers`** — `pip install sentence-transformers`. Loads any of the open models above and gives you `.encode()`.
- **`pgvector` + Postgres** — run via `docker run -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17`. The mini-project ships a `docker-compose.yml`.
- **`psycopg`** (v3) — `pip install "psycopg[binary]"`. The Postgres driver the exercises use.
- **`numpy`** — for the by-hand cosine and the brute-force ground truth.
- **`openai` / `cohere`** (optional) — only for the vendor leg of the bakeoff; the open path needs neither.

## A note on the corpus

The exercises and mini-project run against a small **legal corpus** — a synthetic 50-clause services agreement plus a 40-question gold set — that ships in the mini-project repo skeleton. It mirrors the "50-page legal corpus" from the syllabus lab at a size that embeds in under a minute on CPU. You may swap in your own corpus; the harness is corpus-agnostic. The point is the *methodology*, not the documents.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Embedding** | A fixed-length dense vector that represents a piece of text's meaning. |
| **Encoder** | The transformer that turns tokens into the embedding (vs a decoder, which generates tokens). |
| **Pooling** | How per-token vectors become one vector: CLS-token or mean-pooling. |
| **Normalization** | Scaling a vector to unit length (L2 norm = 1); makes cosine and dot product rank identically. |
| **Cosine similarity** | The cosine of the angle between two vectors; 1 = identical direction, 0 = orthogonal. |
| **Dot product** | Sum of element-wise products; equals cosine for unit vectors. |
| **ANN** | Approximate Nearest Neighbour — fast, slightly-inexact vector search. |
| **HNSW** | Hierarchical Navigable Small World — the graph-based ANN index pgvector defaults to. |
| **IVF** | Inverted File index — clusters vectors, searches only the nearest clusters. |
| **`ef_search`** | HNSW's runtime knob: higher = more accurate, slower. |
| **`nprobe`** | IVF's runtime knob: how many clusters to scan. |
| **MTEB** | Massive Text Embedding Benchmark — the standard (and over-chased) leaderboard. |
| **Recall@k** | Fraction of relevant docs found in the top-k results. |
| **MRR** | Mean Reciprocal Rank — averages 1/(rank of first relevant doc). |
| **Matryoshka** | An embedding trained so you can truncate it to fewer dims and keep most quality. |
| **Re-embed** | Re-running the whole corpus through a new model — the cost of switching embeddings. |

---

*If a link 404s, please open an issue so we can replace it.*
