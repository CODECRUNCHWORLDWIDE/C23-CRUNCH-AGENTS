# Lecture 2 — Rerankers and Structured Retrieval

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can explain the bi-encoder/cross-encoder distinction, rerank a candidate set with `bge-reranker-v2-m3`, reason about ColBERT late interaction, implement query rewriting and HyDE, and generate SQL from natural language safely — read-only role, parameterized execution, schema allowlist.

If you remember one sentence from this lecture, remember the syllabus mantra:

> **A reranker is the cheapest meaningful win in RAG. Use one.**

Last lecture you built a first stage that recalls well: dense retrieval for meaning, BM25 for words, fused with RRF. But "recalls well" means *the right document is somewhere in the top 50* — not *the right document is at rank 1*. The LLM reads the top of its context most carefully, so rank matters. A reranker is the layer that turns "in the top 50" into "at rank 1," and it costs one model and ~30 ms per query. This lecture is that layer, plus the two techniques for the queries first-stage retrieval can't reach: HyDE (when the query embeds badly) and text-to-SQL (when the answer isn't in the vector store at all).

---

## 1. Bi-encoders vs cross-encoders: the core distinction

Everything about reranking follows from one architectural difference.

A **bi-encoder** — your week-7 embedding model — encodes the query and the document *separately*, into two independent vectors, and compares them with cosine similarity. The crucial property: the document vectors can be computed **ahead of time** and stored in the index. At query time you embed only the query (one forward pass) and do a fast vector search over millions of precomputed document vectors. This is why dense retrieval scales — the expensive part (embedding the corpus) happens once, offline.

A **cross-encoder** encodes the query and the document *together*, as a single concatenated input, through the transformer, and outputs a single relevance score. The model's attention layers let every query token attend to every document token and vice versa — so it can capture fine-grained relevance ("does *this* passage actually answer *this* query?") that two independent vectors cannot. The crucial cost: there is **nothing to precompute**. The score depends on the query, so you must run a full forward pass for *every* (query, document) pair at query time.

```
BI-ENCODER (retrieval — fast, precomputable)
  query  ──► [encoder] ──► q_vec ─┐
                                   ├─► cosine(q_vec, d_vec)   d_vec precomputed offline
  doc    ──► [encoder] ──► d_vec ─┘

CROSS-ENCODER (reranking — accurate, query-time only)
  [query [SEP] doc] ──► [encoder] ──► relevance score        nothing precomputable
```

Now the economics are obvious. You **cannot** run a cross-encoder over a million documents per query — that's a million forward passes, seconds to minutes of latency. But you *can* run it over the **top 50** that first-stage retrieval already surfaced. So the pattern, universally, is two-stage:

> **First stage (bi-encoder + BM25, fused):** cheap, high-recall, retrieve the top 50 candidates from millions. **Second stage (cross-encoder reranker):** expensive, high-precision, re-score those 50 and keep the top 5.

You rerank only the top-k because the cross-encoder's cost is *linear in the number of candidates*. Rerank 50 candidates and you pay 50 forward passes (~30–100 ms on CPU for a small reranker, much less on GPU). Rerank 50,000 and you've thrown away the entire reason you have a first stage. The first stage's job is to make the candidate set small enough that the cross-encoder is affordable; the cross-encoder's job is to order that small set correctly.

---

## 2. The reranker in code: `BAAI/bge-reranker-v2-m3`

The open default reranker in 2026 is `BAAI/bge-reranker-v2-m3` — a multilingual cross-encoder, lightweight enough to run on CPU for small candidate sets, and the model the exercises and mini-project use. `sentence-transformers` loads it as a `CrossEncoder`:

```python
from sentence_transformers import CrossEncoder

# Loads the cross-encoder; first run downloads ~600 MB.
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

query = "how long must confidential information be protected after the contract ends?"

# These are the first-stage candidates — say, the fused top-5 from Lecture 1.
candidates = [
    ("clause_14", "Either party may terminate this Agreement upon thirty days written notice."),
    ("clause_09", "All confidential information must be protected for five years after termination."),
    ("clause_18", "This Agreement is governed by the laws of the State of Delaware."),
    ("clause_07", "The annual fee shall be paid in twelve equal monthly installments."),
    ("clause_12", "The Contractor shall maintain professional liability insurance of $1,000,000."),
]

# The cross-encoder scores (query, passage) PAIRS jointly. Build the pairs:
pairs = [(query, text) for _, text in candidates]
scores = reranker.predict(pairs)   # one float per pair; higher = more relevant

# Re-sort the candidates by the reranker's score.
reranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
for (doc_id, text), score in reranked:
    print(f"{doc_id}  rerank={score:.3f}  {text[:60]}")
```

What you'll see: even if the first-stage list had `clause_09` (the confidentiality clause that actually answers the query) at rank 2 or rank 4 — because dense retrieval found `clause_14` "more topically similar" thanks to the shared word "termination" — the cross-encoder reads the query and `clause_09` *together*, sees that `clause_09` literally states the five-year protection period, and scores it highest. The right answer moves to rank 1. That movement is the lift, and it shows up in **MRR** (which cares about the rank of the right answer) far more than in Recall@5 (which only cares whether it's in the top 5 at all).

`sentence-transformers` also gives you a convenience method that does the pairing and sorting for you:

```python
# .rank() takes the query and a list of passages, returns ranked results with indices.
ranked = reranker.rank(query, [text for _, text in candidates], top_k=3)
for r in ranked:
    doc_id = candidates[r["corpus_id"]][0]
    print(f"{doc_id}  score={r['score']:.3f}")
```

### 2.1 The alternative loader: FlagEmbedding

The same model can be loaded through BAAI's own `FlagEmbedding` library, which some shops prefer because it's the canonical home of the bge family (and of bge-m3's unified dense+sparse+ColBERT output):

```python
from FlagEmbedding import FlagReranker

# normalize=True maps scores to [0, 1] via sigmoid; otherwise raw logits.
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
score = reranker.compute_score([query, candidates[1][1]], normalize=True)
print(score)   # a single relevance score for that one (query, passage) pair
```

Functionally equivalent to the `CrossEncoder` path for our purposes. Use whichever your stack already pulls in; the mini-project uses `sentence-transformers` because you already have it from week 7.

### 2.2 The vendor path: Cohere `rerank-3.5`

If you don't want to download or host a model, Cohere's rerank API (`rerank-3.5`, also written `rerank-v3.5`) is the cleanest managed reranker. You send a query and a list of documents; it returns ranked indices with relevance scores:

```python
import cohere

co = cohere.ClientV2()   # reads CO_API_KEY from the environment

response = co.rerank(
    model="rerank-3.5",
    query=query,
    documents=[text for _, text in candidates],
    top_n=3,
)
for result in response.results:
    doc_id = candidates[result.index][0]
    print(f"{doc_id}  relevance={result.relevance_score:.3f}")
```

It's a cross-encoder under the hood — same two-stage discipline applies (rerank only the first-stage top-k), you're just paying per call instead of per GPU-second. The trade-off is the usual open-vs-vendor one from week 7: no model to host and strong quality, against per-query cost and a network hop. For the labs we use the open `bge-reranker-v2-m3`; the Cohere path is the production option when you'd rather not run the model.

---

## 3. ColBERT and late interaction: the middle ground

Bi-encoders are cheap and coarse; cross-encoders are expensive and precise. **ColBERT** sits between them with an idea called **late interaction**.

Instead of pooling a document into *one* vector (bi-encoder) or scoring the full (query, doc) pair through the transformer (cross-encoder), ColBERT keeps a *per-token* embedding for every token in the document and every token in the query. At scoring time it computes, for each query token, the maximum similarity to any document token (**MaxSim**), and sums those maxima:

```
ColBERT_score(q, d) = Σ_{i in query tokens}  max_{j in doc tokens}  cos(q_i, d_j)
```

The "late" in late interaction means the query and document are encoded *independently* (like a bi-encoder, so document token embeddings can be precomputed and indexed) but *interact* at the token level at scoring time (capturing some of the fine-grained matching a cross-encoder gets). It's more expensive than a bi-encoder (you store and compare many vectors per document, not one) and cheaper than a cross-encoder (no full forward pass per pair), with quality in between.

In 2026 the easiest way to add a ColBERT leg is **RAGatouille**, a wrapper that makes indexing and search a few lines:

```python
from ragatouille import RAGPretrainedModel

# A small, strong, modern ColBERT model.
colbert = RAGPretrainedModel.from_pretrained("answerdotai/answerai-colbert-small-v1")

documents = [text for _, text in candidates]
colbert.index(collection=documents, index_name="legal_clauses")
results = colbert.search(query="five year confidentiality duration", k=3)
for r in results:
    print(r["score"], r["content"][:60])
```

Where does it fit in your lift table? As its own layer, between your bi-encoder and your cross-encoder, on both quality and latency. It's the stretch leg in this week's challenge precisely because it's the interesting middle of the quality/cost curve. For most pipelines, dense + BM25 + a cross-encoder reranker is enough; ColBERT earns its place when you need better first-stage recall than a single-vector bi-encoder gives and can't afford to rerank a huge candidate set.

---

## 4. Query rewriting: fix the query before you search

Sometimes the retrieval fails not because your index is bad but because the *query* is bad — too vague, too conversational, missing the keywords the corpus uses. The cheapest fix is to rewrite the query with an LLM before you embed it.

The canonical case is a multi-turn chat. The user asks "what about the confidentiality one?" — a query that embeds to nothing useful, because it has no content words. With the conversation as context, an LLM rewrites it to "what is the duration of the confidentiality obligation in the agreement?" — a query that retrieves `clause_09`. The rewrite is a single cheap generation call:

```python
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment

def rewrite_query(history: list[str], followup: str) -> str:
    """Rewrite a context-dependent follow-up into a standalone, retrievable query."""
    convo = "\n".join(history)
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        thinking={"type": "adaptive"},
        system=(
            "Rewrite the user's follow-up question into a single standalone search "
            "query that includes the entities and intent from the conversation. "
            "Output ONLY the rewritten query, nothing else."
        ),
        messages=[{"role": "user", "content": f"Conversation:\n{convo}\n\nFollow-up: {followup}"}],
    )
    return next(b.text for b in msg.content if b.type == "text").strip()


rewritten = rewrite_query(
    history=["User: How long is the term of the agreement?",
             "Assistant: The initial term is two years (clause_03)."],
    followup="what about the confidentiality one?",
)
# -> "What is the duration of the confidentiality obligation in the agreement?"
```

Query rewriting is a precision/latency trade: you pay one LLM call per query for (sometimes) much better retrieval. Use it where queries are genuinely under-specified — chat, voice — and skip it where they're already clean (a search box where users type full questions). Like every layer this week, you decide whether to keep it by *measuring* the lift, not by assuming it.

---

## 5. HyDE: embed a hypothetical answer, not the query

HyDE — **Hypothetical Document Embeddings** (Gao, Ma, Lin & Callan, 2022; arXiv:2212.10496) — is a more radical query transformation, and it's beautiful. The problem it solves: a *query* and a *document* are different kinds of text. A query is a short question ("how long is confidential info protected?"); the answer is a declarative clause ("All confidential information must be protected for five years after termination."). They don't embed into quite the same neighborhood, because they're written differently — questions and answers have different surface form.

HyDE's move: don't embed the query. Instead, ask an LLM to *write a hypothetical answer* to the query — a fake clause that **looks like** the document you're hoping to find — and embed *that*. The hypothetical answer is written in the same declarative, document-shaped register as the real clauses, so its embedding lands much closer to the real answer's embedding. You retrieve with the hypothetical's vector, not the query's.

It does not matter that the hypothetical answer is *wrong* (the LLM is hallucinating a plausible clause, possibly with the wrong number). You never show it to the user. You only use its *embedding* to find the *real* document, which has the correct information. The hallucination is a feature: you're using the LLM's sense of "what would an answer to this look like" purely as a retrieval bridge.

```python
import anthropic
from sentence_transformers import SentenceTransformer

client = anthropic.Anthropic()
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")   # week-7 dense model


def hyde_embed(query: str):
    """Generate a hypothetical answer, embed THAT, and return its vector."""
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=200,
        thinking={"type": "adaptive"},
        system=(
            "Write a single short, plausible contract clause that would directly "
            "answer the user's question. Write it as a clause, in the declarative "
            "style of a legal agreement. Do not hedge or explain — just the clause."
        ),
        messages=[{"role": "user", "content": query}],
    )
    hypothetical = next(b.text for b in msg.content if b.type == "text").strip()
    # Embed the HYPOTHETICAL ANSWER, not the query. No BGE query prefix here —
    # the hypothetical is document-shaped, so we embed it like a document.
    return embedder.encode(hypothetical, normalize_embeddings=True), hypothetical


vec, hypo = hyde_embed("how long must confidential information be kept private after the deal ends?")
# `hypo` might be: "Confidential Information shall remain protected for a period of
#  five (5) years following the termination of this Agreement." -- close to clause_09!
# Use `vec` for the k-NN search instead of the query's own embedding.
```

**Where HyDE helps and where it hurts — and why you must measure it.** HyDE shines on hard paraphrase queries where the query and answer are written very differently, and on short/keyword-poor queries the LLM can flesh out. It *hurts* when the hypothetical hallucinates off-topic — if the LLM invents a clause about the wrong subject, you retrieve with a vector pointing at the wrong neighborhood, and recall can *drop*. It also adds an LLM call of latency per query. This is the layer in your lift table most likely to come back "+0.00" or even negative — and a measured negative is a *result*: it tells you, on this corpus, HyDE isn't worth it. That honesty (recall the week-9 promise table) is the whole point of measuring every layer.

---

## 6. Structured retrieval: when the answer is in a database

Everything so far assumes the answer lives in *text* — a clause you can embed and retrieve. But some questions have no textual answer to retrieve, because the answer is a *computation over structured data*:

- "Which agreements expire before 2027?"
- "What's the total annual fee across all active contracts?"
- "How many vendors have professional-liability coverage below $1,000,000?"

No amount of dense retrieval, BM25, or reranking answers these, because the answer isn't *written down anywhere* — it has to be *computed* by filtering, joining, and aggregating rows. The right tool is **SQL**, and the technique is **text-to-SQL**: an LLM translates the natural-language question into a SQL query, you run it against the database, and you return (or summarize) the rows.

```python
import anthropic

client = anthropic.Anthropic()

SCHEMA = """
CREATE TABLE agreements (id INTEGER PRIMARY KEY, party TEXT, start_date DATE,
                         end_date DATE, annual_fee_cents INTEGER, governing_law TEXT);
CREATE TABLE clauses (id INTEGER PRIMARY KEY, agreement_id INTEGER, kind TEXT, body TEXT);
"""

def question_to_sql(question: str) -> str:
    """Ask the LLM for a single read-only SQL query against the known schema."""
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=400,
        thinking={"type": "adaptive"},
        system=(
            "You translate questions into a single SQLite SELECT query against the "
            "schema below. Output ONLY the SQL — no prose, no code fences. The query "
            "MUST be a single read-only SELECT. Never write INSERT, UPDATE, DELETE, "
            "DROP, ALTER, or any statement that modifies data.\n\n" + SCHEMA
        ),
        messages=[{"role": "user", "content": question}],
    )
    return next(b.text for b in msg.content if b.type == "text").strip()
```

That prompt is necessary but **nowhere near sufficient** for safety. The model's SQL is *untrusted input* — treat a generated query exactly as you'd treat a query string an anonymous user typed into a box. The threat model is SQL injection with the LLM as the attacker (or as a confused deputy for a prompt-injecting user). Three controls, all required.

### 6.1 The safety surface

**(1) A read-only database role.** This is the load-bearing control. Connect with a Postgres role that has been granted `SELECT` and nothing else — no `INSERT`, `UPDATE`, `DELETE`, `DROP`. If the model emits `DROP TABLE agreements;`, the *database* rejects it because the role lacks the privilege, regardless of what your application code does. Defense in the database, not just in the prompt.

```sql
-- Set up once, as an admin. The text-to-SQL app connects as this role ONLY.
CREATE ROLE rag_readonly LOGIN PASSWORD '...';
GRANT CONNECT ON DATABASE contracts TO rag_readonly;
GRANT USAGE ON SCHEMA public TO rag_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_readonly;
-- Explicitly NO INSERT/UPDATE/DELETE/DDL. A write attempt fails at the engine.
```

**(2) Validate and constrain before executing.** Even read-only, you don't blindly `execute()` model output. Parse it, confirm it's a single statement, confirm it's a `SELECT`, reject anything with multiple statements (the `;` injection), and enforce a `LIMIT`:

```python
import sqlglot   # a real SQL parser/transpiler; pip install sqlglot

def validate_select(sql: str) -> str:
    """Reject anything that isn't a single, read-only SELECT. Raise on violation."""
    statements = sqlglot.parse(sql, read="sqlite")
    if len(statements) != 1:
        raise ValueError("exactly one statement allowed")
    stmt = statements[0]
    if stmt.key.upper() != "SELECT":
        raise ValueError(f"only SELECT permitted, got {stmt.key.upper()}")
    # Block DML/DDL keywords that a SELECT should never contain.
    banned = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
              "CREATE", "GRANT", "REVOKE", "ATTACH", "PRAGMA"}
    tokens = {t.text.upper() for t in sqlglot.tokenize(sql)}
    if tokens & banned:
        raise ValueError(f"banned keyword(s): {tokens & banned}")
    return sql
```

**(3) A schema allowlist.** Tell the model only about the tables and columns it's allowed to touch (note the `SCHEMA` constant above lists exactly two tables), and validate that the generated query references only allowlisted tables. The model can't query a `users` or `secrets` table it was never told exists *and* whose name your validator rejects. You can walk the parsed AST (`sqlglot` gives you the table list) and confirm every referenced table is in your allowlist.

Put together, the safe execution path is: **generate → validate (single read-only SELECT, allowlisted tables) → execute via a read-only role with a parameterized driver → cap the result size.** Never hand a raw model string to a writable connection. The prompt instruction "only write SELECT" is the *least* important of the four controls — it's a hint to the model, not a guarantee. The role, the validator, and the allowlist are what actually keep you safe.

### 6.2 When to route to SQL vs the vector store

In a real RAG agent you don't choose by hand — you let a router (or the agent itself) decide: a question about *what a clause says* goes to the vector store (retrieve text); a question that *counts, filters, or aggregates over structured fields* goes to text-to-SQL. "What does the termination clause say?" → retrieve `clause_14`. "How many agreements terminate on 30 days' notice?" → `SELECT COUNT(*) ... WHERE ...`. Knowing which questions belong to which engine is the architecture judgment you'll defend in week 12.

---

## 7. Recap

You should now be able to:

- State the bi-encoder/cross-encoder distinction and explain why you rerank only the first-stage top-k (the cross-encoder's cost is linear in candidates; nothing is precomputable).
- Rerank a candidate set with `BAAI/bge-reranker-v2-m3` via `sentence-transformers` `CrossEncoder` (and name `FlagReranker` and Cohere `rerank-3.5` as alternatives), and explain why the lift shows up in MRR.
- Describe ColBERT late interaction (token-level MaxSim) and place it between a bi- and cross-encoder on the quality/cost curve.
- Implement query rewriting for under-specified queries, and HyDE — generate a hypothetical answer, embed *that*, retrieve with it — and explain why a wrong hypothetical can still help (and when it hurts).
- Generate SQL from natural language for structured questions, and lock it down: read-only role, parse-and-validate to a single read-only SELECT, schema allowlist, never execute raw model SQL on a writable connection.

That's the full toolbox. Next you measure all of it: BM25 → dense → hybrid+RRF → +reranker → +HyDE, on the same 40-query gold set, one row per layer. The mantra one more time, because it's the week's thesis: **a reranker is the cheapest meaningful win in RAG — use one** — and now you can prove it with a number.

---

## References

- *HyDE — Precise Zero-Shot Dense Retrieval without Relevance Labels* — Gao, Ma, Lin & Callan, 2022 (arXiv:2212.10496): <https://arxiv.org/abs/2212.10496>
- *`BAAI/bge-reranker-v2-m3` model card*: <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- *Sentence-Transformers `CrossEncoder` usage*: <https://www.sbert.net/docs/cross_encoder/usage/usage.html>
- *`FlagEmbedding` (FlagReranker, bge-m3)*: <https://github.com/FlagOpen/FlagEmbedding>
- *Cohere Rerank (`rerank-3.5`)*: <https://docs.cohere.com/docs/rerank-overview>
- *ColBERTv2* — Santhanam et al., 2021: <https://arxiv.org/abs/2112.01488>
- *RAGatouille* (ColBERT made easy): <https://github.com/AnswerDotAI/RAGatouille>
- *`answerdotai/answerai-colbert-small-v1` model card*: <https://huggingface.co/answerdotai/answerai-colbert-small-v1>
- *Query2doc* — Wang et al., 2023 (the query-expansion cousin of HyDE): <https://arxiv.org/abs/2303.07678>
- *OWASP SQL Injection Prevention Cheat Sheet* (the text-to-SQL threat model): <https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html>
- *Postgres `GRANT`* (the read-only role): <https://www.postgresql.org/docs/current/sql-grant.html>
