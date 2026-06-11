# Exercise 1 — Embed and Inspect

**Goal:** Build hands-on intuition for what an embedding *is*. You will embed text with three open models, look at the actual numbers, prove by hand that cosine similarity ranks results the way Lecture 1 claimed, and then reproduce the query/document-prefix bug on purpose so you recognize it forever.

**Estimated time:** 45 minutes. Guided.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install sentence-transformers numpy
```

The first run downloads each model. `bge-large` is ~1.3 GB; `gte-large` is ~1.3 GB; `nomic-embed-text` is ~550 MB. Start the download early.

---

## Step 1 — Embed one sentence and look at it

Save as `inspect.py` and run it.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

text = "Either party may terminate this agreement upon thirty days written notice."
vec = model.encode(text, normalize_embeddings=True)

print(f"type:        {type(vec).__name__}")
print(f"shape:       {vec.shape}")        # (1024,) for bge-large
print(f"dtype:       {vec.dtype}")        # float32
print(f"first 8:     {np.round(vec[:8], 4)}")
print(f"L2 norm:     {np.linalg.norm(vec):.6f}")   # ~1.000000 because normalized
print(f"min / max:   {vec.min():.4f} / {vec.max():.4f}")
```

**What to notice.** The output is a 1024-element float32 array. The norm is 1.0 (that's the `normalize_embeddings=True`). The individual numbers are meaningless on their own — no single dimension means "is about termination." Meaning lives in the *whole vector's direction*, which is exactly why we compare with cosine, not by reading dimensions.

Now flip `normalize_embeddings=False` and re-run. The norm is no longer 1.0. Note the value — you'll need to normalize by hand in Step 3 to make cosine work, and this is your reminder that *you* are responsible for normalization when the model doesn't do it.

---

## Step 2 — Embed three texts and rank a query by hand

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

docs = [
    "Either party may terminate this agreement upon thirty days written notice.",
    "The annual rent shall be paid in twelve equal monthly installments.",
    "This contract is governed by the laws of the State of Delaware.",
]
query = "How much notice is needed to end the contract?"

doc_vecs = model.encode(docs, normalize_embeddings=True)            # (3, 1024)
q_vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)  # (1024,)

# Cosine == dot product for unit vectors. One matrix-vector product gives all 3 scores.
scores = doc_vecs @ q_vec
order = np.argsort(-scores)

print("ranking (best first):")
for rank, idx in enumerate(order, start=1):
    print(f"  #{rank}  score={scores[idx]:.3f}  {docs[idx]}")
```

**Expected shape of output:**

```
ranking (best first):
  #1  score=0.78  Either party may terminate this agreement upon thirty days written notice.
  #2  score=0.41  This contract is governed by the laws of the State of Delaware.
  #3  score=0.39  The annual rent shall be paid in twelve equal monthly installments.
```

The termination clause wins, and it shares almost no words with the query ("notice" is the only overlap). That's dense retrieval matching *meaning*. Exact scores vary by a few hundredths across machines; the *ranking* should be stable.

---

## Step 3 — Prove cosine == dot product == (for unit vectors) by hand

Don't take Lecture 1 §4 on faith. Compute all three the long way and confirm they agree on order.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Embed WITHOUT normalization so we can normalize by hand and see it matter.
raw = model.encode(
    ["terminate the contract", "pay the monthly rent"],
    normalize_embeddings=False,
)
q_raw = model.encode("how do I end the agreement", normalize_embeddings=False)

def normalize(v):
    return v / np.linalg.norm(v)

for i, label in enumerate(["terminate", "rent"]):
    a, b = normalize(raw[i]), normalize(q_raw)
    cosine = float(a @ b)                          # unit vectors -> cosine
    dot = float(raw[i] @ q_raw)                    # un-normalized dot (different scale!)
    l2 = float(np.linalg.norm(a - b))              # L2 on unit vectors
    # For unit vectors: ||a-b||^2 == 2 - 2*(a.b). Confirm:
    identity = 2 - 2 * cosine
    print(f"{label:10s}  cosine={cosine:.4f}  rawdot={dot:.4f}  "
          f"L2={l2:.4f}  (2-2cos)={identity:.4f}  L2^2={l2**2:.4f}")
```

**What to confirm:**

- `cosine` ranks "terminate" above "rent" for the query "how do I end the agreement."
- `L2^2` equals `2 - 2*cosine` (the identity from the lecture) — so smaller L2 ⇔ larger cosine, i.e. **L2 and cosine rank identically on unit vectors.**
- The *un-normalized* `rawdot` is on a totally different scale — that's why you must normalize before comparing, and why mixing normalized and un-normalized vectors corrupts your ranking.

---

## Step 4 — Reproduce the prefix bug on purpose

This is the most valuable five minutes of the exercise. Run the Step 2 ranking again, but **drop the `QUERY_PREFIX`** from the query embedding:

```python
q_vec_noprefix = model.encode(query, normalize_embeddings=True)   # NO prefix
scores_bad = doc_vecs @ q_vec_noprefix
order_bad = np.argsort(-scores_bad)
print("WITHOUT prefix:")
for rank, idx in enumerate(order_bad, start=1):
    print(f"  #{rank}  score={scores_bad[idx]:.3f}  {docs[idx]}")
```

Compare the scores to Step 2. They shift — usually the gap between the right answer and the distractors *shrinks*, and on a harder query the ranking can flip outright. You just reproduced the single most common silent RAG bug: **forgetting BGE's query instruction prefix.** No error fired. The retrieval just got worse. Now you'll catch it in a code review.

---

## Step 5 — Compare three models on the same query

Swap the model and re-run Step 2 for `Alibaba-NLP/gte-large-en-v1.5` and `nomic-ai/nomic-embed-text-v1.5`. Note: each has its *own* prefix convention.

- `bge-large`: query prefix `"Represent this sentence for searching relevant passages: "`, none on docs.
- `gte-large`: no prefix needed (it's instruction-free for retrieval).
- `nomic`: prefix `"search_query: "` on queries, `"search_document: "` on documents. **Pass `trust_remote_code=True`** when loading nomic.

```python
nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
doc_vecs = nomic.encode(["search_document: " + d for d in docs], normalize_embeddings=True)
q_vec = nomic.encode("search_query: " + query, normalize_embeddings=True)
```

Record the top-1 result and its score for each model. They should *all* rank the termination clause first on this easy query — which is exactly Lecture 1's point: **embedding choice rarely changes the system on easy queries.** Where models differ is the hard, ambiguous queries, and you can't see that on three sentences — you need the bakeoff (the challenge and mini-project) with a real gold set.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] You can state the dimension of `bge-large` (1024), `gte-large` (1024), and `nomic` (768) from your own output, not from memory.
- [ ] Your by-hand cosine ranks the termination clause first for the notice query.
- [ ] You confirmed `L2^2 == 2 - 2*cosine` numerically on unit vectors.
- [ ] You reproduced the prefix bug: dropping `QUERY_PREFIX` measurably changes the BGE scores.
- [ ] All three models rank the termination clause #1 on the easy query, and you can explain why that *doesn't* prove they're equivalent.

---

## Stretch

- **Matryoshka by hand.** Embed with nomic (768-dim), then truncate to the first 256 dims, re-normalize (`v[:256] / norm(v[:256])`), and re-rank. How much does the ranking change? (On easy queries: barely.) This is the storage-vs-quality lever from Lecture 1.
- **Negation failure.** Embed "the tenant must pay rent" and "the tenant must not pay rent" and compute their cosine similarity. It'll be high (often > 0.9) despite being opposites. That's the dense-retrieval blind spot week 9 fixes with lexical search.
- **Cross-model cosine is meaningless.** Try comparing a `bge` document vector to a `gte` query vector. The number you get is garbage — vectors from different models live in different spaces. Confirm you understand *why* you can never mix embeddings from two models in one index.

---

When this feels comfortable, move to [Exercise 2 — pgvector k-NN](exercise-02-pgvector-knn.py).
