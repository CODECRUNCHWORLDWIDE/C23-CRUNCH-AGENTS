# Lecture 1 — Embeddings as Compressed Meaning: Models, MTEB, and Metrics

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can explain what an embedding is at a systems level, name the 2026 open and vendor embedding families with their dimensions and quirks, read the MTEB leaderboard without being fooled by it, and choose a similarity metric with a reason rather than a habit.

If you remember one sentence from this entire week, remember this one:

> **An embedding is a lossy compression of a piece of text's meaning into a fixed-length vector, and "nearby vectors" is a useful but imperfect proxy for "relevant text." The art of retrieval is knowing when the proxy holds and what to do when it breaks.**

Phase I treated the LLM as the system. You prompted it, called tools with it, ran it locally. But an LLM only knows what was in its training data, frozen at a cutoff. The moment your product needs to answer questions about *your* documents — a contract signed last Tuesday, a wiki page, a support ticket — the model is blind. **Retrieval-Augmented Generation (RAG)** fixes that by fetching relevant text from your corpus and putting it in the prompt. And retrieval starts here: turning text into vectors you can search.

---

## 1. What an embedding actually is

You already know, from week 1, that a decoder-only LLM is a function from a sequence of tokens to a distribution over the next token. An **embedding model** is a close cousin with a different job. It is an *encoder* — usually a transformer — that takes a sequence of tokens and produces a **single fixed-length vector** that represents the meaning of the whole sequence.

```
"Either party may terminate this agreement with 30 days notice."
        │
   tokenizer  →  [101, 4321, 2283, ...]   (a sequence of token IDs)
        │
   encoder transformer  →  per-token hidden states  (seq_len × hidden_dim)
        │
   pooling  →  one vector  (hidden_dim,)
        │
   normalize  →  one unit vector  e.g. (1024,) with ||v|| = 1
```

Three things make this different from generation:

1. **The output is one vector, not a sequence.** A 400-token paragraph and a 4-token query both come out as, say, a 1024-dimensional vector. That fixed size is what lets you store millions of them in one index and compare any two with a single dot product.
2. **There is a pooling step.** The transformer produces one hidden state per token. To get a single vector you must *pool* them. Two conventions dominate: take the special `[CLS]` token's state (BERT-style; BGE does this), or **mean-pool** across all token states (E5, GTE, nomic do this). You rarely choose this yourself — the model card tells you, and `sentence-transformers` does it for you — but when you implement pooling by hand (Exercise 1) you must match the model's convention or your vectors are garbage.
3. **The output is usually normalized.** Most retrieval embeddings are L2-normalized to unit length. This is why, for these models, cosine similarity and dot product give the *same ranking* (§4). Some models normalize for you; some don't. Getting this wrong is a top-three silent bug in RAG.

The model was trained — via contrastive learning on huge collections of (query, relevant-passage) and (query, irrelevant-passage) pairs — so that **related text lands close together** under a distance metric, and unrelated text lands far apart. That training objective is the whole game. "Close together in this 1024-dim space" *means* "semantically related" *only because* the model was optimized to make it so. There is nothing magic about the geometry; it's learned.

### Why "lossy" matters

Compressing a paragraph to 1024 floats throws away enormous detail. Consider:

- *"The tenant must pay rent on the first of the month."*
- *"The tenant must not pay rent on the first of the month."*

To a human these are opposites. To many embedding models they're *very close* — the negation is one token in a sea of shared meaning, and the vectors land near each other. This is the classic **negation failure**, and it's why dense retrieval alone is not enough (the entire premise of week 9). Embeddings capture topical similarity beautifully and logical structure poorly. Know that going in.

---

## 2. The 2026 open embedding families

There are dozens of embedding models. In practice, on an English RAG system in 2026, you reach for one of five families. Here is the working knowledge.

### BGE (BAAI) — the open default

The **BGE** (BAAI General Embedding) family is the most-deployed open embedding line. `bge-large-en-v1.5` produces **1024-dim** vectors, handles **512 tokens** of input, is CLS-pooled, and is normalized. It is Apache-2.0 licensed (commercial use is fine). The one catch you *must* know: BGE was trained with a **query instruction prefix**. When you embed a *query* you prepend `"Represent this sentence for searching relevant passages: "`; when you embed a *document* you prepend nothing. Forget the prefix on queries and your recall quietly drops. `sentence-transformers` handles this if you pass the right prompt, but you have to know it's there.

`bge-m3` is the multilingual, multi-granularity sibling: it emits a dense vector, a sparse (lexical) vector, *and* ColBERT-style multi-vector output from one model. You'll meet its sparse head again in week 9 when we build hybrid search.

### GTE (Alibaba) — strong and long-context

The **GTE** (General Text Embeddings) family from Alibaba is BGE's main rival on English retrieval. `gte-large-en-v1.5` is **1024-dim**, mean-pooled, and — crucially — handles **8192 tokens** of context, far more than BGE's 512. If your chunks are long (week 8), GTE may save you from truncation. `gte-Qwen2-7B-instruct` is the heavyweight LLM-based variant that tops leaderboards at the cost of being a 7B model.

### jina-embeddings-v3 — task adapters and Matryoshka

`jina-embeddings-v3` is **1024-dim** by default, handles **8192 tokens**, and adds two clever features. First, **task-specific LoRA adapters**: you tell it whether you're doing `retrieval.query`, `retrieval.passage`, `separation`, `classification`, or `text-matching`, and it swaps a small adapter to specialize. Second, **Matryoshka representation**: the vector is trained so you can *truncate* it — keep the first 256 of 1024 dims, re-normalize, and retain most of the quality at a quarter of the storage. This is a real lever at scale.

### nomic-embed-text-v1.5 — open data, long context, Matryoshka

`nomic-embed-text-v1.5` is **768-dim**, handles **8192 tokens**, is Matryoshka-trained (truncatable to 512/256/128/64), and is notable for being **fully open** — training data and code published, not just weights. It also uses task prefixes (`search_query:`, `search_document:`, `classification:`, `clustering:`). If reproducibility or auditability matters to your shop, nomic is the honest pick.

### E5-Mistral — the LLM-as-embedder

`e5-mistral-7b-instruct` takes a different path: it's a **7B-parameter LLM** fine-tuned to emit embeddings (**4096-dim**). It is excellent and expensive — running it needs real GPU memory and the vectors are 4x the size of a 1024-dim model, which quadruples your index storage and slows search. It's the model you reach for when quality is everything and you have the hardware. For most systems it is overkill; `bge-large` or `gte-large` is the right default.

### The summary table

| Model | Dim | Max tokens | Pooling | Prefix? | License | Note |
|---|---:|---:|---|---|---|---|
| `bge-large-en-v1.5` | 1024 | 512 | CLS | query only | MIT | The open default |
| `bge-m3` | 1024 | 8192 | CLS | no | MIT | Dense + sparse + ColBERT |
| `gte-large-en-v1.5` | 1024 | 8192 | mean | no | Apache-2.0 | Long context |
| `jina-embeddings-v3` | 1024 | 8192 | mean | task | CC-BY-NC* | LoRA tasks + Matryoshka |
| `nomic-embed-text-v1.5` | 768 | 8192 | mean | task | Apache-2.0 | Open data, Matryoshka |
| `e5-mistral-7b-instruct` | 4096 | 32768 | last-token | instruct | MIT | LLM-as-embedder, heavy |

\* Check jina's current license before commercial deployment — the non-commercial clause is the kind of thing that bites at the worst time.

---

## 3. Vendor embeddings

Sometimes a hosted embedding is the right call: you don't want to run a GPU, you want someone else's SLA, or the vendor model is genuinely better on your domain. The three you'll meet:

### OpenAI `text-embedding-3`

`text-embedding-3-small` (1536-dim) and `text-embedding-3-large` (3072-dim) are the workhorses. The clever feature is the **`dimensions` parameter**: these models are Matryoshka-trained, so you can ask for a *shorter* vector (e.g. `dimensions=256` from the 3072-dim large model) and trade a little quality for a lot of storage. Cost is per-token and cheap, but it's per-token *forever* — every query and every re-embed costs money.

```python
from openai import OpenAI
client = OpenAI()
resp = client.embeddings.create(
    model="text-embedding-3-large",
    input=["Either party may terminate with 30 days notice."],
    dimensions=1024,   # Matryoshka truncation: 3072 -> 1024
)
vec = resp.data[0].embedding   # a 1024-float list, already normalized
```

### Cohere `embed-v3`

Cohere's `embed-english-v3.0` / `embed-multilingual-v3.0` are strong retrieval models with one parameter you *must* get right: **`input_type`**. You pass `search_document` when embedding corpus text and `search_query` when embedding a query. This is the same query/document asymmetry as BGE's prefix, made explicit. Pass the wrong `input_type` and recall tanks silently — no error, just bad results.

### Voyage

Voyage AI's `voyage-3` line is a retrieval-focused vendor with domain-specific variants (`voyage-law-2`, `voyage-code-3`). On specialized corpora — legal, code — a domain model can beat a general one by a meaningful margin. Worth a leg in your bakeoff if your corpus is specialized.

### The honest guidance

For ~80% of systems in 2026, an open `bge-large` or `gte-large` running on a modest GPU is the right default: free at inference, no per-query cost, no data leaving your network, and within a point or two of the vendor models on most corpora. Reach for a vendor embedding when (a) you genuinely can't run a GPU, (b) your corpus is a domain where a specialized vendor model wins measurably, or (c) the operational simplicity of "someone else runs it" is worth the per-query cost. **Decide with a measurement (§6 of Lecture 2), not a brand preference.**

---

## 4. Similarity metrics — and why they matter less than you think

Once everything is a vector, "find similar text" becomes "find nearby vectors." Three distance/similarity functions are used:

### Cosine similarity

The cosine of the angle between two vectors:

```
cos(a, b) = (a · b) / (||a|| · ||b||)
```

It's 1 when the vectors point the same direction, 0 when orthogonal, −1 when opposite. It ignores magnitude — only direction matters. This is the default for text embeddings, because the *direction* of the vector encodes meaning and the *magnitude* is mostly noise.

### Dot product

Just the numerator: `a · b = Σ aᵢbᵢ`. It's cheaper (no normalization division) and, here's the key fact:

> **For L2-normalized vectors (||a|| = ||b|| = 1), cosine similarity and dot product are identical.** `cos(a,b) = a·b / (1·1) = a·b`.

Since most retrieval embeddings are normalized, the cosine-vs-dot choice usually **doesn't change the ranking at all**. Vector databases default to dot product on normalized vectors precisely because it's the same answer for less compute.

### Euclidean (L2) distance

`||a − b|| = √Σ(aᵢ − bᵢ)²`. The straight-line distance. For normalized vectors, L2 distance and cosine similarity produce the *same ranking* too (smaller L2 distance ⇔ larger cosine), because `||a−b||² = 2 − 2(a·b)` when both are unit vectors. So even L2 collapses to the same order on normalized embeddings.

### The practical takeaway

If your embeddings are normalized — and most are — **the three metrics rank your results identically**. The metric choice is real only for un-normalized embeddings, where magnitude carries information. So:

- Default to **cosine** (or, equivalently, dot product on normalized vectors). It's what every model card assumes.
- Make sure your vectors are **normalized**. If the model doesn't do it, do it yourself: `v / np.linalg.norm(v)`.
- Match the metric to your **index**. pgvector's `<=>` is cosine, `<#>` is negative dot product, `<->` is L2. Pick the operator that matches how you built the index, or you'll get nonsense.
- Then **stop thinking about the metric.** It is the least important decision this week. Embedding choice matters more; chunking (week 8) and reranking (week 9) matter far more than that.

---

## 5. Reading the MTEB leaderboard without being fooled

The **Massive Text Embedding Benchmark (MTEB)** is the standard leaderboard for embedding models. It's genuinely useful and genuinely misleading, and you need both halves of that.

### What MTEB actually measures

MTEB is not one task — it's a suite across categories:

- **Retrieval** — given a query, rank passages (this is the one you care about for RAG).
- **Reranking** — re-order a candidate list (week 9 territory).
- **Clustering** — group similar texts.
- **Classification** — predict a label from the embedding.
- **STS** (Semantic Textual Similarity) — score how similar two sentences are.
- **Pair classification, summarization, bitext mining** — others.

The headline "MTEB average" blends all of these. A model can top the average by being excellent at clustering and classification while being merely-OK at retrieval. **For RAG you want the Retrieval tab, ranked by nDCG@10, not the overall average.** Reading the wrong column is the most common MTEB mistake.

### The overfit problem

MTEB is public, so models get trained — sometimes deliberately, sometimes by leakage — to score well *on MTEB's datasets specifically*. A model at the very top of the leaderboard may have seen MTEB-adjacent data in training and may generalize *worse* to your weird corpus than a model three spots down. The leaderboard measures performance on MTEB; it does not measure performance on your legal contracts or your support tickets. Treat a high MTEB score as a *hint*, not a *verdict*.

### How to read it like an engineer

1. Filter to the **Retrieval** task.
2. Filter to a **model size you can actually serve** — a 7B embedder that needs an A100 is irrelevant if you're on a laptop.
3. Note the **dimension** — a 4096-dim model costs 4x the storage and search time of a 1024-dim model for maybe a point of recall.
4. Pick **two or three** plausible candidates.
5. **Benchmark them on your own corpus** with your own gold set (§6 of Lecture 2). The number that decides is *your* Recall@5, not MTEB's nDCG@10.

This is why the mini-project this week is a *bakeoff harness*, not a model recommendation. The leaderboard narrows the field; your data picks the winner.

---

## 6. A worked example: embedding and comparing by hand

Let's make this concrete. We embed three short texts with `bge-large` and check, by hand, that the geometry does what we claimed.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# BGE wants the query-instruction prefix on QUERIES, not documents.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

docs = [
    "Either party may terminate this agreement upon thirty days written notice.",
    "The annual rent shall be paid in twelve equal monthly installments.",
    "This contract is governed by the laws of the State of Delaware.",
]
query = "How much notice is required to end the contract?"

# normalize_embeddings=True gives unit vectors -> cosine == dot product.
doc_vecs = model.encode(docs, normalize_embeddings=True)
q_vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)

# Cosine == dot product for unit vectors. Just take the dot.
scores = doc_vecs @ q_vec        # shape (3,)
ranking = np.argsort(-scores)    # highest score first

for rank, idx in enumerate(ranking):
    print(f"#{rank+1}  score={scores[idx]:.3f}  {docs[idx]}")
```

Expected shape of output:

```
#1  score=0.78  Either party may terminate this agreement upon thirty days written notice.
#2  score=0.41  This contract is governed by the laws of the State of Delaware.
#3  score=0.39  The annual rent shall be paid in twelve equal monthly installments.
```

The termination clause wins because "notice required to end the contract" is semantically nearest to "terminate ... upon thirty days written notice" — even though it shares almost no exact words ("notice" is the only overlap). *That* is the power of dense retrieval: it matches meaning, not keywords. (And the fact that pure keyword search would also have found "notice" is exactly why week 9 combines both.)

Try the experiment that teaches the lesson: drop the `QUERY_PREFIX` and re-run. The scores shift and the ranking can break, because you violated BGE's training convention. That five-line mistake is a real production bug.

---

## 7. The operational facts that bite

A few things that aren't in the model card but will cost you a weekend:

**Dimension lock-in.** Your vector column, your index, and every stored vector are all sized to one dimension. Switching from a 1024-dim model to a 768-dim model means a *new column, a new index, and re-embedding the entire corpus*. There is no migration shortcut. Choose with this in mind — switching is a redeploy-everything event.

**Re-embedding cost.** Changing embeddings means re-running every document through the new model. For a 10 GB corpus that's hours of GPU time (open model) or real dollars (vendor model). This is the deeper reason behind the week's mantra: don't fiddle with the embedding once you've picked it, because every change pays this tax.

**Normalization, again.** If half your vectors are normalized and half aren't (e.g. you re-embedded some documents with a different flag), your similarity scores are meaningless and your ranking is corrupt. Normalize *everything*, in one place, and assert it.

**Batching throughput.** Embedding one document at a time wastes the GPU. `model.encode(list_of_docs, batch_size=64)` is many times faster than a Python loop. When you embed a corpus, batch.

**Query/document asymmetry.** BGE's prefix, Cohere's `input_type`, nomic's `search_query:`/`search_document:`, jina's task adapter — these all encode the same idea: *queries and documents are embedded differently*. Get the asymmetry right or lose recall silently. This is the single most common "my retrieval is mysteriously bad" bug, and it produces no error.

---

## 8. Matryoshka, quantization, and the storage bill

Two 2026 techniques change the economics of an embedding without changing which model you pick — and both are levers you reach for *after* the bakeoff, when the model is chosen and the corpus is large.

**Matryoshka embeddings (MRL).** Models trained with Matryoshka Representation Learning (`nomic-embed-text-v1.5`, `jina-embeddings-v3`, OpenAI's `text-embedding-3-*`) pack the most important information into the *first* dimensions of the vector. That means you can **truncate** a 1024-dim vector to 256 dims and keep most of the retrieval quality — the leading dimensions carry the signal. The trade is explicit and measurable: a 768→256 truncation cuts storage and index size by 3× and speeds up every distance computation, at the cost of a few points of Recall@5. You truncate *then re-normalize* (the truncated vector is no longer unit length):

```python
full = model.encode(text, normalize_embeddings=True)   # e.g. (768,)
truncated = full[:256]
truncated = truncated / np.linalg.norm(truncated)      # re-normalize after slicing
```

The discipline is the same as everything this week: don't guess that 256 dims is "good enough" — run the bakeoff at 768 and at 256 and read the Recall@5 delta. On many corpora the drop is under two points, which is a fine trade for a 3× smaller index. On a hard corpus it's not. You measure.

**Scalar and binary quantization.** A `float32` vector at 1024 dims is 4 KB. Across ten million chunks that's 40 GB of vectors before the index overhead. **Scalar quantization** (store each dimension as an `int8` instead of a `float32`) cuts that 4×; **binary quantization** (one bit per dimension, compared with Hamming distance) cuts it 32×, at a larger recall cost that you claw back with a re-scoring pass over the top candidates in full precision. pgvector supports `halfvec` (16-bit) natively, and the binary-then-rerank pattern is the standard way to serve a billion-scale index on commodity RAM. For this course's corpus you do not need quantization — but you must know the lever exists, because "the index doesn't fit in RAM" is the first wall you hit at scale, and quantization (not a bigger box) is usually the answer.

> **The order of operations at scale:** pick the model by quality (the bakeoff), then truncate (Matryoshka) to the smallest dimension that holds your Recall@5, then quantize if the index still doesn't fit. Each step is a measured trade, not a default.

---

## 9. Domain and language: where the leaderboard lies hardest

The MTEB Retrieval average is computed over a fixed set of mostly-English, mostly-general benchmarks. Two situations break the leaderboard's predictive power entirely, and both are common in real products.

**Domain shift.** A model that tops MTEB on Wikipedia-style passages can underperform a lower-ranked model on legal clauses, clinical notes, or code, because the *register* of the text is different from anything the model saw enough of in training. The legal corpus you embed this week is a mild example — dense, clause-structured, citation-heavy text that looks nothing like a news article. The only reliable signal is your own 40-query gold set on your own corpus. When a teammate says "but model X is #1 on MTEB," the answer is "on *their* data; show me the number on *ours*."

**Language.** If any of your corpus or queries are non-English, the English-only leaderboard is irrelevant. Reach for an explicitly multilingual model — `BAAI/bge-m3` (100+ languages, and it emits dense *and* sparse vectors, which you'll meet again in week 9) or `intfloat/multilingual-e5-large`. A monolingual English model on Spanish queries doesn't error; it silently returns worse results, which is the failure mode this whole week is training you to distrust.

The takeaway compounds the week's mantra: the leaderboard narrows your candidate list; your gold set picks the winner. Never let a leaderboard rank be the last word on a model for *your* job.

### Instruction-tuned and LLM-based embedders

Two newer families are worth knowing because they change the cost/quality frontier. **Instruction-tuned embedders** (`intfloat/e5-mistral-7b-instruct`, the `gte-Qwen2` line, the `instructor` family) let you prepend a task *instruction* to the query — e.g. "Given a legal question, retrieve the clause that answers it" — and the model adapts its embedding to the task. On a specialized retrieval task this can buy real recall, because you're telling the model what "relevant" means for *your* job instead of relying on its generic notion. The cost is that these are large (a 7B embedder needs a GPU and is slow to run over a big corpus), so they sit at the quality-first end of the trade.

**LLM-as-embedder** is the same idea taken further: a decoder-only LLM, with last-token or mean pooling, repurposed as an encoder. `e5-mistral` is the canonical 2026 example. These top several MTEB leaderboards but are heavy enough that you rarely embed a 10 GB corpus with one — the usual pattern is a small fast open model (BGE/GTE) for the bulk corpus and an instruction-tuned heavyweight only when a domain genuinely needs it and you've *measured* that the lift justifies the GPU bill. The decision is the same shape as everything else this week: candidates from the leaderboard, winner from your gold set, and the cost of running the model is part of the score.

---

## 10. Recap

You should now be able to:

- Explain an embedding as a fixed-length, lossy, learned compression of text meaning, produced by an encoder transformer with a pooling and (usually) a normalization step.
- Name the 2026 open families — BGE, GTE, jina-v3, nomic, E5-Mistral — with their dimensions, context lengths, and prefix conventions, and the vendor lines (OpenAI, Cohere, Voyage) with their key parameters.
- Read the MTEB leaderboard skeptically: the Retrieval tab, not the average; overfit-to-leaderboard risk; benchmark on your own data.
- Choose a similarity metric and explain why, on normalized embeddings, cosine, dot, and L2 rank identically — so the metric is the least important choice this week.
- Recognize the operational gotchas: dimension lock-in, re-embedding cost, normalization, batching, and query/document asymmetry.

Next up: how the vectors are *searched* fast (ANN indexes), how pgvector does it, and how you *measure* whether any of it works. Continue to [Lecture 2 — ANN Indexes and Vector Search](./02-ann-indexes-and-vector-search.md).

---

## References

- *Sentence-Transformers documentation*: <https://www.sbert.net/>
- *MTEB leaderboard*: <https://huggingface.co/spaces/mteb/leaderboard>
- *MTEB paper* (Muennighoff et al., 2022): <https://arxiv.org/abs/2210.07316>
- *BGE model card*: <https://huggingface.co/BAAI/bge-large-en-v1.5>
- *GTE model card*: <https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5>
- *Matryoshka Representation Learning* (Kusupati et al., 2022): <https://arxiv.org/abs/2205.13147>
- *OpenAI embeddings guide*: <https://platform.openai.com/docs/guides/embeddings>
