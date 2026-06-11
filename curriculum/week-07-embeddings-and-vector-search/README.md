# Week 7 — Embeddings and Vector Search

Welcome to the week where retrieval stops being a black box. By Friday you will be able to look at any RAG system and state, without hand-waving, what embedding model it uses, why, what its vector index actually does when you call `.search()`, and how to measure whether the whole thing is finding the right documents. You will read an **MTEB** leaderboard the way a backend engineer reads a benchmark suite — skeptically, with the deployment in mind, not the score.

This is the first week of **Phase II — RAG & Memory Systems**. Everything you built in Phase I (the agent loop, tool calling, local inference) assumed the model already knew the answer. From here on, it doesn't. The corpus does. Your job is to get the right slice of the corpus in front of the model at the right time, and that begins with turning text into vectors and searching those vectors fast.

The one thing to internalize before you read another line: **an embedding is a lossy compression of meaning into a fixed-length vector, and "similar vectors" only approximates "relevant text."** A `bge-large` embedding of a 400-token paragraph is 1024 floats. That's it. All the nuance of that paragraph — every clause, every qualifier, every negation — is squeezed into 1024 numbers. Cosine similarity between two such vectors is a *useful* signal, but it is not truth. The whole discipline of retrieval is about knowing when that signal is good enough, when it lies to you, and what to layer on top when it does. This week we build the layer that everything else stands on.

There's a sentence the rest of this phase keeps coming back to, and it's worth taping to your monitor now:

> **Embedding choice rarely changes the system; retrieval-strategy choice almost always does.**

You will spend this week learning embeddings in depth — and one of the lessons is that swapping `bge-large` for `gte-large` usually moves your top-5 recall by a point or two, while adding a reranker (week 9) or fixing your chunking (week 8) moves it by ten. Learn embeddings well, pick one with reasons, and then stop fiddling with them.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** what an embedding is at a systems level — a fixed-length dense vector produced by a transformer encoder, trained so that semantically related text lands close together under a distance metric — and why that is both powerful and lossy.
- **Enumerate** the 2026 open and vendor embedding families — **BGE** (BAAI), **GTE** (Alibaba), **jina-embeddings-v3**, **nomic-embed-text**, **E5-Mistral**, and the vendor lines (OpenAI `text-embedding-3`, Cohere `embed-v3`, Voyage) — and state the dimension, max sequence length, and license of each.
- **Read** the **MTEB** leaderboard skeptically: distinguish the task types, spot the overfit-to-leaderboard models, and reason about the gap between a benchmark score and your corpus.
- **Choose** a similarity metric — cosine, dot product, Euclidean — and explain why, for normalized embeddings, cosine and dot product rank identically and the choice mostly stops mattering.
- **Describe** how an approximate-nearest-neighbour (**ANN**) index works — **HNSW**, **IVF**, and **ScaNN** — the recall/latency/memory trade-offs of each, and the knobs (`ef_search`, `nprobe`, `m`) that move them.
- **Build** an end-to-end retrieval pipeline: embed a corpus with a real model, index it in **pgvector**, and run a query benchmark that reports **top-1**, **top-5**, and **MRR**.
- **Measure** retrieval quality as an engineer — build a small gold set, compute the metrics yourself, and defend an embedding choice in a one-page memo instead of a vibe.
- **Reason** about the operational facts that bite in production — embedding-dimension lock-in, the cost of re-embedding a corpus when you switch models, normalization, and batching throughput.

## Prerequisites

This week assumes you have completed **C23 weeks 1–6**, or have equivalent fluency. Specifically:

- Python 3.12 on Linux, macOS (Apple Silicon), or WSL2. You can create a virtualenv, `pip install`, and run a script without fighting your environment.
- You can call a hosted LLM (you did this in week 1) and run a local model through **Ollama** (week 6). We use both this week — a local embedding model for most work, a vendor embedding for the comparison.
- You are comfortable with **Docker** and `docker compose`. We run **Postgres + pgvector** in a container.
- You understand, from week 1, that an LLM is a function from tokens to a distribution over tokens. An embedding model is a *different* head on a similar transformer: it's a function from tokens to a single vector. We make that distinction precise in Lecture 1.
- You can read a `numpy` array, a `torch` tensor's `.shape`, and a SQL `SELECT` without panicking.

You do **not** need a GPU. Every embedding model this week runs on CPU or Apple Silicon (slower, but unblocked). A 24 GB GPU makes the larger models pleasant; the labs document the CPU path. You do **not** need prior vector-database experience — we start at "what is a vector index" and build up.

## Topics covered

- **Embeddings as compressed semantic representations:** the encoder transformer, the pooling step (CLS-token vs mean-pooling), why the output is a single fixed-length vector, and what "the embedding space has geometry" actually means.
- **The 2026 open embedding families:** **BGE** (`bge-large-en-v1.5`, `bge-m3`), **GTE** (`gte-large-en-v1.5`, `gte-Qwen2`), **jina-embeddings-v3** (task-LoRA, Matryoshka), **nomic-embed-text-v1.5** (long context, Matryoshka), **E5-Mistral-7B-instruct** (LLM-as-embedder) — dimensions, sequence lengths, instruction-prefix conventions, and licenses.
- **Vendor embeddings:** OpenAI `text-embedding-3-small`/`-large` (with `dimensions` truncation), Cohere `embed-v3` (with `input_type`), Voyage (`voyage-3`); when a hosted embedding earns its cost and when an open model is the right call.
- **The MTEB leaderboard, read skeptically:** the task taxonomy (retrieval, reranking, clustering, classification, STS), why a model that tops the clustering task may be mediocre at *your* retrieval task, and how leaderboard-chasing produces overfit models.
- **Similarity metrics:** cosine, dot product, Euclidean (L2); why normalization makes cosine and dot equivalent for ranking; when L2 differs; and the practical advice that the metric matters far less than the embedding and the chunking.
- **ANN indexes:** brute-force (exact) vs approximate; **HNSW** (the graph, `m`, `ef_construction`, `ef_search`, and the recall/latency curve); **IVF** (`nprobe`, the coarse quantizer); **ScaNN** (anisotropic quantization at Google scale); the recall/latency/memory triangle you cannot escape.
- **pgvector in practice:** the `vector` column type, `<=>` (cosine), `<#>` (negative dot), `<->` (L2) operators, building an `hnsw` index, `SET hnsw.ef_search`, and why pgvector is the sane default in 2026.
- **Retrieval evaluation as engineering:** building a gold set, computing **Recall@k**, **MRR**, and **top-1/top-5**, and using those numbers — not intuition — to pick a model.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                   | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|---------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | What an embedding is; the model families; MTEB literacy |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Similarity metrics; embedding a corpus; pgvector setup  |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | ANN indexes (HNSW/IVF/ScaNN); the recall/latency curve  |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Retrieval evaluation; the embedding-bakeoff harness     |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The bakeoff memo; index-tuning clinic                   |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                  |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                               |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                         | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The embedding model cards, MTEB, pgvector and HNSW docs, and the papers worth your time |
| [lecture-notes/01-embeddings-as-compressed-meaning.md](./lecture-notes/01-embeddings-as-compressed-meaning.md) | What an embedding is, the 2026 model families, MTEB read skeptically, and similarity metrics |
| [lecture-notes/02-ann-indexes-and-vector-search.md](./lecture-notes/02-ann-indexes-and-vector-search.md) | HNSW, IVF, ScaNN, the recall/latency/memory triangle, pgvector in practice, and evaluation |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-embed-and-inspect.md](./exercises/exercise-01-embed-and-inspect.md) | Embed text with three open models, inspect the vectors, and prove cosine ranking by hand |
| [exercises/exercise-02-pgvector-knn.py](./exercises/exercise-02-pgvector-knn.py) | Stand up pgvector, embed a small corpus, build an HNSW index, and run k-NN queries |
| [exercises/exercise-03-recall-vs-efsearch.py](./exercises/exercise-03-recall-vs-efsearch.py) | Sweep `ef_search` against a brute-force ground truth and chart the recall/latency curve |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-embedding-bakeoff.md](./challenges/challenge-01-embedding-bakeoff.md) | Run four embeddings over one corpus, report top-1/top-5/MRR, and defend a pick |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the one-page embedding-choice memo |
| [mini-project/README.md](./mini-project/README.md) | The `crunchrag_embed` bakeoff harness — a reusable embedding-and-eval module |

## The "retrieval that actually retrieves" promise

C23 uses a recurring marker for every exercise that ends in a retrieval system actually finding the right document:

```
$ python eval.py --embedding bge-large --k 5
queries: 40 | Recall@1: 0.62 | Recall@5: 0.85 | MRR: 0.71
top result for q07 ("notice period for termination"):
  doc_id=clause_14  score=0.81  "Either party may terminate this Agreement upon..."
```

If `Recall@5` is far below what the model's MTEB card promised, you are not done. The most common cause is not the embedding — it's a query/document mismatch (you embedded documents but forgot the model wants an instruction prefix on queries), a normalization bug, or a chunking problem you'll fix next week. The point of week 7 is to make that `Recall@5` line *measured*, *reproducible*, and *honest* — and to make a bad number loud instead of a vibe.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **HNSW paper** (Malkov & Yashunin, 2018) until you can draw the search descent through layers from memory: <https://arxiv.org/abs/1603.09320>. Then explain why `ef_search` is the single most important runtime knob.
- Implement **Matryoshka truncation** by hand: embed with `nomic-embed-text-v1.5`, take the first 256 dimensions of the 768-dim vector, re-normalize, and measure how much Recall@5 you lose for the 3x storage win. Chart it.
- Build a tiny **HNSW from scratch** in ~120 lines of Python — insert with a greedy graph, search with the layered descent — and race it against pgvector on a 10k-vector set. You will respect the production implementations afterward.
- Quantize your vectors to **`int8`** (scalar quantization) and to **binary** (1 bit/dim, Hamming distance), and measure the recall cost of each against `float32`. This is the storage lever the big vector stores pull at scale.

## Up next

Week 8 takes the embedding-and-search literacy you built here and points it at the part of RAG that decides whether any of it works: **chunking and document processing**. You will A/B chunking strategies over a real corpus using the exact `crunchrag_embed` harness you build this week. Push your mini-project before you start it — week 8 imports it.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
