# Week 9 — Resources

Every resource here is **free** or has a free tier. The BM25 and RRF references are open papers and library docs. The reranker model cards live on Hugging Face. The HyDE paper is on arXiv. Where a vendor API is referenced (Cohere rerank, the LLM you call for HyDE and text-to-SQL), the docs are public and the labs spend a few cents of credit; an open-only path is documented for every lab.

Model names and library versions move every cohort — the *concepts* (lexical scoring, rank fusion, cross-encoder reranking, hypothetical-document retrieval, the text-to-SQL safety surface) are stable. When a specific model card 404s, search Hugging Face for the family name (`BAAI/bge-reranker`, `colbert-ir`, `answerdotai/answerai-colbert`).

## Required reading (work it into your week)

- **`rank-bm25`** — the pure-Python BM25 library the exercises use (`BM25Okapi`, `BM25L`, `BM25Plus`). Read the README to see exactly which scoring variant `BM25Okapi` implements:
  <https://github.com/dorianbrown/rank_bm25>
- **The Reciprocal Rank Fusion paper** — Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009). Short, decisive, and the source of the k=60 default:
  <https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf>
- **Sentence-Transformers `CrossEncoder` docs** — the reranker API you'll call: `CrossEncoder(...).predict([(query, passage), ...])` and `.rank(query, passages)`:
  <https://www.sbert.net/docs/cross_encoder/usage/usage.html>
- **HyDE — *Precise Zero-Shot Dense Retrieval without Relevance Labels*** (Gao, Ma, Lin & Callan, 2022; arXiv:2212.10496). The paper that says: don't embed the query, embed a *hypothetical answer*:
  <https://arxiv.org/abs/2212.10496>

## Reranker model cards (have these open Wednesday)

- **`BAAI/bge-reranker-v2-m3`** — the open default cross-encoder reranker; multilingual, lightweight, takes `(query, passage)` pairs and returns a relevance score. This is the model the exercises and mini-project use:
  <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- **`BAAI/bge-reranker-v2-gemma`** — the heavier, higher-quality LLM-based reranker in the same family, for when m3 isn't enough:
  <https://huggingface.co/BAAI/bge-reranker-v2-gemma>
- **`FlagEmbedding`** — BAAI's library; the `FlagReranker` loader is the alternative to `sentence-transformers`'s `CrossEncoder` for the bge rerankers, and the home of the bge-m3 unified dense+sparse+ColBERT model:
  <https://github.com/FlagOpen/FlagEmbedding>
- **Cohere Rerank** — the vendor reranker API (`rerank-3.5` / `rerank-v3.5`); pass a query and a list of documents, get back ranked indices with relevance scores. The cleanest "no model download" path:
  <https://docs.cohere.com/docs/rerank-overview>

## ColBERT and late interaction

- **RAGatouille** — the friendly wrapper that makes ColBERT indexing and search a few lines of Python; the easiest way to add a late-interaction leg:
  <https://github.com/AnswerDotAI/RAGatouille>
- **`answerdotai/answerai-colbert-small-v1`** — a small, strong, modern ColBERT model; a sane default for a late-interaction retriever in 2026:
  <https://huggingface.co/answerdotai/answerai-colbert-small-v1>
- **ColBERTv2 paper** — Santhanam et al., *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction* (2021). The token-level MaxSim mechanism, explained by its authors:
  <https://arxiv.org/abs/2112.01488>

## Hybrid search and lexical/sparse references

- **BM25 — the Robertson & Zaragoza monograph**, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009). The definitive treatment of TF saturation, IDF, and the `k1`/`b` parameters:
  <https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf>
- **pgvector — hybrid search / full-text** — how to combine pgvector's `<=>` with Postgres `tsvector`/`tsquery` full-text search in one query, the no-extra-infra hybrid path:
  <https://github.com/pgvector/pgvector#hybrid-search>
- **Tantivy** — the Rust full-text search library (a Lucene in Rust) with `tantivy-py` bindings; the production BM25 engine to graduate to when `rank-bm25` runs out of room:
  <https://github.com/quickwit-oss/tantivy>
- **`bm25s`** — a fast BM25 implementation built on sparse `scipy` matrices; orders of magnitude faster than `rank-bm25` while staying pure-Python-friendly, when your corpus outgrows the toy size:
  <https://github.com/xhluca/bm25s>
- **`Qdrant` hybrid + sparse vectors** — the production pattern of storing dense and sparse (BM25/SPLADE) vectors together and fusing server-side; you'll meet this for real in week 10:
  <https://qdrant.tech/articles/hybrid-search/>

## Structured retrieval / text-to-SQL

- **Postgres `GRANT` and roles** — the read-only role is the load-bearing safety control for text-to-SQL; this is the canonical reference for `GRANT SELECT` and revoking write:
  <https://www.postgresql.org/docs/current/sql-grant.html>
- **OWASP SQL Injection Prevention Cheat Sheet** — parameterization, allowlisting, and least privilege; the generated-SQL threat is a SQL-injection threat with the model as the attacker:
  <https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html>
- **The Spider text-to-SQL benchmark** — the standard cross-domain text-to-SQL dataset; useful for understanding where generated SQL goes wrong (joins, aggregation, nesting):
  <https://yale-lily.github.io/spider>

## Papers worth your time (free on arXiv)

- **Reciprocal Rank Fusion** (Cormack et al., 2009) — linked above; read it first, it's four pages.
- **HyDE** (Gao et al., 2022; arXiv:2212.10496) — linked above; the cleanest argument for query transformation in retrieval.
- **ColBERT** (Khattab & Zaharia, 2020; arXiv:2004.12832) — the original late-interaction paper, for the MaxSim intuition:
  <https://arxiv.org/abs/2004.12832>
- **SPLADE** (Formal et al., 2021; arXiv:2107.05720) — learned sparse retrieval; the modern bridge between "lexical" and "neural," and where the bge-m3 sparse output comes from:
  <https://arxiv.org/abs/2107.05720>
- **Query2doc** (Wang et al., 2023; arXiv:2303.07678) — the LLM-query-expansion cousin of HyDE; useful contrast for the query-rewriting section:
  <https://arxiv.org/abs/2303.07678>

## Tools you'll use this week

- **`rank-bm25`** — `pip install rank-bm25`. The BM25 leg of the exercises and the mini-project. No infra, pure Python.
- **`sentence-transformers`** — `pip install sentence-transformers`. Loads `BAAI/bge-reranker-v2-m3` via `CrossEncoder`; you already have it from week 7.
- **`psycopg`** (v3) + **`pgvector`** — the dense leg, unchanged from weeks 7–8 (`docker run pgvector/pgvector:pg17`).
- **`cohere`** (optional) — `pip install cohere`. Only for the Cohere `rerank-3.5` leg; the open path needs `bge-reranker-v2-m3` instead.
- **`ragatouille`** (optional) — `pip install ragatouille`. Only for the ColBERT stretch leg.
- **An LLM client** — `anthropic` (or your week-1 provider). HyDE and text-to-SQL both need one generation call per query.

## A note on the corpus

The exercises and mini-project run against the same small **legal corpus** you've used since week 7 — the synthetic 50-clause services agreement plus the 40-question gold set — so the lift you measure this week stacks directly on the Recall@5 you measured in weeks 7 and 8. The clause ids are stable: `clause_14` is termination ("thirty days written notice"), `clause_09` is confidentiality ("five years after termination"), `clause_07` is the annual fee in twelve installments, `clause_12` is the $1,000,000 insurance, `clause_18` is Delaware law, `clause_27` is arbitration in San Francisco. The point is the *methodology* — measuring lift on a fixed gold set — not the documents.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Dense retrieval** | Search by embedding similarity — matches meaning, misses exact words. |
| **Sparse / lexical retrieval** | Search by word overlap (BM25) — matches exact words, misses paraphrase. |
| **BM25** | The standard lexical scoring function: TF × IDF with saturation (`k1`) and length normalization (`b`). |
| **TF (term frequency)** | How often a query term appears in a document. |
| **IDF (inverse document frequency)** | How rare a term is across the corpus; rare terms weigh more. |
| **`k1`** | BM25's term-frequency saturation knob; higher = TF keeps mattering, lower = it plateaus fast (~1.2–2.0). |
| **`b`** | BM25's length-normalization knob; 1.0 = fully penalize long docs, 0.0 = ignore length (~0.75). |
| **Hybrid search** | Running dense and sparse retrieval and combining their results. |
| **RRF** | Reciprocal Rank Fusion: `Σ_r 1/(k + rank_r(d))`, k≈60; rank-based, no score calibration needed. |
| **Bi-encoder** | Encodes query and document *separately* into vectors; fast, precomputable, used for first-stage retrieval. |
| **Cross-encoder** | Encodes query and document *together*; accurate, expensive, used only to rerank the top-k. |
| **Reranker** | A cross-encoder that re-scores first-stage candidates against the query and reorders them. |
| **ColBERT / late interaction** | Per-token embeddings compared with MaxSim; between a bi- and cross-encoder on cost and quality. |
| **Query rewriting** | Transforming a vague query into a more retrievable one before search. |
| **HyDE** | Generate a hypothetical answer with an LLM, embed *that*, and retrieve with it. |
| **Text-to-SQL** | Generating a SQL query from a natural-language question to retrieve from a database. |
| **Cumulative lift** | The Recall@5/MRR gain each pipeline layer adds, measured on one fixed gold set. |

---

*If a link 404s, please open an issue so we can replace it.*
