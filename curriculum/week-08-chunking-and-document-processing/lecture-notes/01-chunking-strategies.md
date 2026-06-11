# Lecture 1 — Chunking Strategies: The Five Cuts and Why Size Is a Hyperparameter

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can name the five chunking strategies in 2026 practice (fixed token-window, sliding-window, recursive, semantic-paragraph, late chunking), state the trade-off each makes, choose a chunk size for a corpus class and *defend it as a tuned hyperparameter*, and predict which strategy will keep a given answer inside a single retrievable chunk.

If you remember one sentence from this entire week, remember this one:

> **Chunking is the part of RAG that determines whether the rest of RAG is doing anything.** You can pick the best embedding on MTEB and tune `ef_search` to the elbow, and still retrieve garbage — because the answer was split across two chunks, or drowned in a thirty-page one.

There's a corollary you should tape next to last week's mantra:

> **A retrieval failure is more often a chunking failure than an embedding failure.** When Recall@5 is bad, suspect the chunks before you suspect the model.

Last week you treated the embedding as the variable: same corpus, same store, swap the model, measure. This week you freeze the model (BGE-large) and the store (pgvector) and make *the chunker* the variable. Everything that follows is in service of one measurable question: **which way of cutting the document up keeps each answer findable?**

---

## 1. Why chunk at all

You don't *have* to chunk. You could embed each document as one vector and call it a day. Three hard constraints say no.

**Constraint 1 — one chunk is one vector, and one vector is a blurry average.** An encoder turns a span of text into a single fixed-length vector (1024 floats for BGE-large). That vector is a *summary* of the whole span. Embed a 30-page contract as one vector and you get the contract's "average meaning" — too coarse to match a specific query like "what's the confidentiality duration?" The signal you want (one clause) is averaged away by twenty-nine pages you don't. Smaller chunk → sharper, more specific vector.

**Constraint 2 — the encoder has a token ceiling.** BGE-large truncates at 512 tokens. Feed it 30 pages and it silently keeps the first ~512 tokens and *throws the rest away*. Your "document vector" is actually a "first-half-page vector." This is one of the most common silent bugs in beginner RAG: the embedding ran, returned a vector, no error — and 95% of the document never reached the model. Even long-context encoders (jina-v3 at 8192 tokens, used in §6) have a ceiling; chunking is how you stay under it deliberately instead of getting truncated by accident.

**Constraint 3 — the LLM's context budget.** Even if you could embed a whole document as one vector, you'd retrieve the whole document and stuff it into the generator's prompt. That's expensive (tokens cost money and latency) and it *dilutes* the prompt — the model has to find the one relevant clause inside thirty pages of context, and long-context models measurably degrade when the needle is buried. Retrieving a tight, relevant chunk is cheaper *and* produces better answers.

So you chunk. And now the tension that defines the whole week:

> **Specificity vs self-containedness.** A chunk should be **small enough to be specific** (so its one vector sharply represents one idea) and **big enough to be self-contained** (so the answer to a question actually lives inside it, with the context needed to interpret it). Every strategy below is a different negotiation of that tension.

The legal corpus makes this concrete. Clause 9 reads: *"All confidential information must be protected for five years after termination."* If your chunk boundary lands mid-clause — "...must be protected for five" | "years after termination" — then *no single chunk* contains the answer to "how long is the confidentiality obligation?" The embedding can't help you; the answer was destroyed before it was ever embedded. That's a chunking failure, and no embedding upgrade fixes it.

---

## 2. Fixed token-window chunking — the right first thing to try

The simplest chunker: walk the token stream, emit a chunk every N tokens. N is typically 512 or 1024. No overlap, no structure awareness.

The one thing people get wrong: **size is in tokens, not characters.** "512 tokens" is not "512 characters" and not "512 words." A token is the *model's* unit, and you must count with the *embedding model's* tokenizer, because that's the budget the encoder actually sees. For BGE you load its tokenizer; for an OpenAI embedder you'd use `tiktoken`. Counting characters and hoping is the classic off-by-a-lot bug.

```python
from transformers import AutoTokenizer

# BGE counts tokens with its own WordPiece tokenizer. Count in ITS units.
tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")


def fixed_token_chunks(text: str, size: int = 512) -> list[str]:
    """Split `text` into chunks of at most `size` tokens, no overlap."""
    ids = tok.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(ids), size):
        window = ids[start:start + size]
        chunks.append(tok.decode(window))
    return chunks
```

What it's good for: it's a **baseline**. It's fast, deterministic, dependency-light, and it gives you a number to beat. Always run it first. A more sophisticated chunker that can't beat fixed-512 on your corpus isn't earning its complexity.

What it's blind to: **structure**. It will happily slice through the middle of clause 9, through the middle of a sentence, through the middle of a word's worth of tokens. On prose with no structure it's fine; on a contract full of self-contained clauses it's the strategy most likely to split an answer. That blindness is exactly what the next four strategies fix, each in a different way.

> **Default to start from:** fixed token-window, 512 tokens. It's the baseline every other strategy must beat on *your* measured Recall@5.

---

## 3. Sliding-window with overlap — let the boundary-straddling answer survive

Fixed windows have a brutal failure mode: an answer that straddles a boundary lands in *neither* chunk fully. Sliding-window fixes it by making consecutive windows **overlap** by M tokens. The same boundary-straddling text now appears in the tail of one chunk *and* the head of the next — so at least one chunk contains the whole answer.

```python
def sliding_window_chunks(
    text: str, size: int = 512, overlap: int = 64
) -> list[str]:
    """Token windows of `size` that overlap their neighbour by `overlap` tokens."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    ids = tok.encode(text, add_special_tokens=False)
    step = size - overlap          # advance by less than a full window
    chunks = []
    for start in range(0, len(ids), step):
        window = ids[start:start + size]
        if not window:
            break
        chunks.append(tok.decode(window))
        if start + size >= len(ids):
            break
    return chunks
```

The lever is the **overlap ratio**, `overlap / size`. Common values are 10–20% (64 tokens on a 512 window is 12.5%). The trade-off is pure storage and redundancy:

- **More overlap** → an answer is more likely to survive whole in some chunk → higher recall, *but* you store more vectors (each token appears in multiple chunks), pay more to embed, and you get duplicate hits in your top-k (the same passage retrieved twice in slightly different windows).
- **Less overlap** → cheaper, fewer duplicates, but the boundary-straddling answer is at risk again.

Overlap is the cheapest insurance against the boundary problem, and it's why almost every production chunker has *some* overlap. But it's insurance, not a cure: a 512-token window with 64 overlap still splits a clause that happens to land exactly on the step boundary. The cure is to put boundaries where the *structure* already is — which is recursive chunking.

> **Rule of thumb:** start at 10–15% overlap. If your gold-set answers are short and self-contained (legal clauses), modest overlap is plenty. If answers span several sentences (prose, narrative), more overlap helps — measure it.

---

## 4. Recursive chunking — respect structure where it exists

This is the default that ships in LangChain and LlamaIndex, and it's the one you'll reach for most. The idea: a document already has structure — paragraphs, sentences, words — and a good chunk boundary is one that *coincides* with a structural boundary. So instead of cutting blindly at token N, you split on a **hierarchy of separators**, falling back to finer ones only when a piece is still too big.

The separator hierarchy, coarsest to finest:

```
["\n\n",   # paragraph breaks (best place to cut)
 "\n",      # line breaks
 ". ",      # sentence ends
 " ",       # word boundaries (last resort)
 ""]        # raw character split (only if a single "word" exceeds the budget)
```

The algorithm: try to split the text on the coarsest separator. If the resulting pieces all fit under `chunk_size`, you're done — those are great boundaries (full paragraphs). If a piece is still too big, recurse *into that piece* with the next finer separator. You only ever fall to character-level splitting for pathological input (a single 600-token "word," e.g. a base64 blob). The result: chunks that end at paragraph/sentence boundaries wherever possible, and only get violent when they have to.

LangChain's `RecursiveCharacterTextSplitter` does exactly this:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=lambda s: len(tok.encode(s, add_special_tokens=False)),
)
chunks = splitter.split_text(document_text)
```

Two things to notice. First, `length_function` — by default LangChain measures `chunk_size` in **characters**, which is wrong for a token budget. Pass a function that counts the *embedding model's tokens* so "512" means what the encoder sees. This is the single most common misconfiguration of this splitter. Second, recursive splitting *still* takes a `chunk_overlap`, so it combines the boundary-respecting of recursion with the straddle-insurance of overlap.

Why it's the workhorse: on real documents with real structure — and a legal contract is *all* structure, one clause per paragraph — recursive chunking keeps each clause intact far more often than fixed windows do. Clause 9 is a paragraph; the paragraph separator `"\n\n"` puts a boundary on either side of it; the whole five-year confidentiality answer survives in one chunk. That's the promise made good.

Where it's weak: it respects *typographic* structure (newlines, periods), not *semantic* structure. If two unrelated ideas share a paragraph, recursive chunking keeps them together; if one idea spans three short paragraphs, recursive chunking splits them. For that you need a chunker that reads *meaning*, not whitespace.

---

## 5. Semantic-paragraph chunking — split where the meaning shifts

Semantic chunking puts boundaries where the *topic* changes, not where the punctuation is. The mechanism is elegant and worth understanding even if you rarely ship it:

1. Split the document into sentences.
2. Embed every sentence (with the same encoder you'll use for retrieval).
3. Walk adjacent sentences and compute the cosine *distance* between each pair.
4. A **spike** in distance means the topic just shifted — that's a boundary. Split there.
5. Group the runs of similar sentences between boundaries into chunks.

The intuition: sentences about the same thing have nearby embeddings; the sentence where the subject changes is far from its predecessor. You're using the embedding model itself as a topic-boundary detector.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")


def semantic_chunks(sentences: list[str], percentile: float = 90.0) -> list[str]:
    """Split where the cosine distance between adjacent sentences spikes."""
    vecs = model.encode(sentences, normalize_embeddings=True)
    # Distance between each sentence and the next (1 - cosine for unit vectors).
    dists = [1.0 - float(np.dot(vecs[i], vecs[i + 1]))
             for i in range(len(sentences) - 1)]
    if not dists:
        return [" ".join(sentences)]
    threshold = float(np.percentile(dists, percentile))  # the "spike" cutoff
    chunks, current = [], [sentences[0]]
    for i, d in enumerate(dists):
        if d >= threshold:                # topic shift -> boundary
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    chunks.append(" ".join(current))
    return chunks
```

The lever is the **percentile threshold** — how big a distance spike counts as a boundary. A 90th-percentile threshold splits only at the sharpest 10% of transitions (few, large chunks); a 70th-percentile threshold splits more aggressively (many, small chunks). It's the semantic analogue of chunk size, and it's just as much a tuned hyperparameter.

The cost is real: you embed *every sentence* just to decide the boundaries, then (usually) re-embed the resulting chunks. On a big corpus that's a meaningful compute bill before you've stored a single retrieval vector. The payoff is boundaries that track meaning — a chunk is a *topic*, not a fixed token count. On documents where topics drift inside paragraphs (prose, transcripts, mixed reports), semantic chunking can beat recursive. On already-structured documents (the legal corpus, where each clause is its own paragraph), recursive usually matches it for far less compute. **Measure before you pay for semantics.**

---

## 6. Late chunking — embed the whole document first, pool the chunks after

The previous four strategies all share a hidden flaw: **they embed each chunk independently.** When you embed clause 9 alone, the encoder never sees that this clause is *about termination* (which clause 14 established two paragraphs earlier), or that "the Agreement" refers to the services agreement defined on page 1. Each chunk's vector is computed in isolation, blind to the document around it. Pronouns, defined terms, and cross-references lose their referents. This is the **context-loss problem** of naive chunking.

**Late chunking** (Günther et al., Jina AI, 2024 — arXiv 2409.04701) inverts the order of operations to fix exactly this. The name is the mechanism: you chunk *late*, after embedding, not before.

The standard ("early") pipeline is: **chunk → embed each chunk → store.**
The late-chunking pipeline is: **embed the whole document → chunk the token embeddings → pool each chunk's span → store.**

Concretely:

1. Run the **entire document** through a long-context encoder (jina-embeddings-v3, 8192-token context) in **one forward pass**, with `output_value="token_embeddings"` so you get the per-token contextual embeddings — *not* one pooled vector, but one vector per token. Because it's one forward pass over the whole document, every token's embedding has attended to every other token: clause 9's tokens "know" about clause 14, the defined terms, the whole context.
2. Decide chunk boundaries **as token spans** (using any boundary strategy — fixed, recursive, whatever).
3. For each chunk, **mean-pool the token embeddings within its span** to get that chunk's vector.

Because the per-token embeddings were computed with full-document attention, each chunk's pooled vector carries the surrounding context that independent embedding throws away. The chunk for clause 9 now encodes "five-year confidentiality *of this services agreement, after termination as defined in clause 14*" — even though the clause text alone never says any of that.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# A long-context model so the WHOLE document fits in one forward pass.
jina = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)


def late_chunk_embed(
    document: str, span_token_ranges: list[tuple[int, int]]
) -> np.ndarray:
    """Embed the whole doc once, then mean-pool each chunk's token span.

    span_token_ranges: [(start_tok, end_tok), ...] chunk boundaries as token indices
    into the SAME tokenization the model used. Returns one vector per chunk.
    """
    # One forward pass over the entire document -> per-token contextual vectors.
    token_vecs = jina.encode(
        document,
        output_value="token_embeddings",   # NOT the pooled sentence vector
        convert_to_numpy=True,
    )
    chunk_vecs = []
    for start, end in span_token_ranges:
        span = token_vecs[start:end]                     # this chunk's tokens
        pooled = span.mean(axis=0)                        # mean-pool the span
        pooled = pooled / np.linalg.norm(pooled)          # normalize for cosine
        chunk_vecs.append(pooled)
    return np.vstack(chunk_vecs)
```

The trade-offs, honestly:

- **It needs a long-context encoder.** You can't late-chunk a 30-page document through BGE-large's 512-token window — the whole point is one forward pass over the whole doc. Late chunking is a *jina-v3 / long-context* technique. (For documents longer than even 8192 tokens, you late-chunk within overlapping macro-windows; the paper covers this.)
- **It changes only the embedding, not the boundaries.** You still choose chunk spans somehow. Late chunking is orthogonal to *where* you cut — it changes *how* each cut is embedded. You can late-chunk with fixed, recursive, or semantic boundaries.
- **The lift is biggest where context matters most** — documents full of pronouns, defined terms, and cross-references (contracts, papers, codebases). On a corpus of independent, self-contained one-liners the lift is smaller, because there's less surrounding context to lose.

Late chunking is the most important *new* idea on this list. It's why "embed then chunk" is now a real option and not a typo. In the week's A/B you'll run it as one of four strategies and measure whether its context-preservation actually moves Recall@5 on the legal corpus — where the defined-term and cross-reference structure is exactly the kind of context it's built to keep.

---

## 7. Chunk size is a hyperparameter, not a guess

Here is the thesis the whole week is built on. **Chunk size is a hyperparameter.** You do not pick 512 because a blog post said so. You *sweep* it against your gold set and read the Recall@5 curve, the same way you swept `ef_search` last week.

The curve has a predictable shape, and the shape *is* the specificity-vs-self-containedness tension made visible:

- **Too small** (e.g. 128 tokens) — chunks are hyper-specific but answers get *split*. The five-year confidentiality clause lands across two chunks; neither contains the whole answer; Recall@5 drops. The vector is sharp but it's a sharp picture of half an answer.
- **Too large** (e.g. 2048 tokens) — answers are self-contained (the whole clause is in there) but the chunk vector is *diluted*. Clause 9 is one paragraph buried in eight others; the chunk's vector is the average of nine clauses, and a query about confidentiality matches it only weakly. Recall@5 drops again.
- **Just right** — somewhere in the middle, the chunk is big enough to hold a whole answer and small enough that its vector is dominated by that answer. Recall@5 peaks.

The peak is **corpus-dependent**. There is no universal best chunk size:

| Corpus class | Typical sweet spot | Why |
|---|---|---|
| **Legal clauses** | 256–512 tokens | Answers are short, self-contained clauses; small chunks stay specific without splitting. |
| **Prose / narrative** | 512–1024 tokens | Ideas span several sentences; you need room for a self-contained thought. |
| **Code** | structure-defined (function/class) | Token count is the wrong unit; split on syntactic boundaries, not a fixed N. |
| **Tables / records** | one row or one record per chunk | The natural unit is the row; splitting a row destroys it. |

This is why the syllabus deliverable is an **A/B memo** and not a recommendation to "use 512." You *measure* your corpus. The sweep in Exercise 3 and the A/B in the challenge are that measurement: same embedding (BGE-large), same store (pgvector), vary the chunk size and the strategy, read the Recall@5 and MRR, and pick a winner *with the numbers that justify it*. A chunk size you can't defend with a curve is a chunk size you guessed.

> **The discipline:** never report a chunk size without the sweep that produced it. "512 with 64 overlap, recursive" is an answer. "512 because it's standard" is a confession.

---

## 8. The pipeline this strategy lives in

Chunking is one stage of a five-stage pipeline. Get any earlier stage wrong and the best chunker can't save you; get chunking right and you still need the later stages. The whole chain:

```
extraction  ->  cleaning  ->  chunking  ->  metadata  ->  embedding  ->  store
   (L2)         (L2)          (L1)          (here)         (week 7)      (week 7)
```

- **Extraction** (Lecture 2) — get text out of the source (PDF, scan, HTML). If extraction garbles clause 9 into "fi ve years," no chunker recovers it.
- **Cleaning** (Lecture 2) — strip headers/footers, fix hyphenation, drop page numbers. Boilerplate that repeats on every page pollutes every chunk and tanks retrieval.
- **Chunking** (this lecture) — cut the cleaned text into retrieval units.
- **Metadata** — attach the section heading, document id, page number, and clause id to each chunk *before* embedding. A clause about termination that carries the heading "Termination" in its metadata retrieves better than the bare clause body — and the metadata is what lets you filter ("only clauses from contract X") and cite ("clause 9, page 4") at retrieval time. The stretch goal in the README — inject the heading into the chunk text before embedding — is a measurable lever here.
- **Embedding → store** — week 7, unchanged.

The reason to see the whole pipeline now: when Recall@5 is bad next week, you'll walk *up* this chain — store? no, that's week-7-clean; embedding? no, BGE is fixed; chunking? probably; cleaning? maybe; extraction? check the raw text. The corollary mantra — *a retrieval failure is more often a chunking failure than an embedding failure* — is really "walk up the pipeline, and the chunking and extraction stages are where the bodies are buried."

---

## 9. Recap

You should now be able to:

- Explain **why chunking exists** — the one-vector-per-chunk blur, the encoder's token ceiling (and its silent truncation), and the LLM context budget — and state the specificity-vs-self-containedness tension every strategy negotiates.
- Implement and contrast the **five strategies**: fixed token-window (baseline), sliding-window (overlap insurance), recursive (structure-respecting, the workhorse), semantic-paragraph (meaning-boundary, expensive), and late chunking (embed-whole-doc-first, context-preserving).
- State the **late-chunking mechanism** correctly — one forward pass over the whole document through a long-context encoder, then mean-pool token embeddings per chunk span — and *why* it preserves the cross-chunk context naive chunking destroys (arXiv 2409.04701).
- Treat **chunk size as a tuned hyperparameter**: predict the too-small/too-large Recall@5 curve, name the corpus-dependent sweet spot, and refuse to report a size without the sweep behind it.
- Place chunking in the **extraction → cleaning → chunking → metadata → embedding** pipeline and know which stage to suspect when retrieval fails.

Next: how to get the text out of real documents in the first place — the extraction toolchain, OCR, tables, and the A/B methodology that turns "which chunker?" into a measured decision. Continue to [Lecture 2 — Document Extraction and the A/B Methodology](./02-document-extraction-and-evaluation.md).

---

## References

- *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models* — Günther et al., Jina AI, 2024: <https://arxiv.org/abs/2409.04701>
- *LangChain text splitters (recursive splitting, separators, chunk_size/overlap)*: <https://python.langchain.com/docs/concepts/text_splitters/>
- *LlamaIndex node parsers (SentenceSplitter, SemanticSplitterNodeParser)*: <https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/>
- *Dense X Retrieval: What Retrieval Granularity Should We Use?* — Chen et al., 2023: <https://arxiv.org/abs/2312.06648>
- *Pinecone — chunking strategies*: <https://www.pinecone.io/learn/chunking-strategies/>
- *jina-embeddings-v3* (the long-context model the late-chunking lab uses): <https://huggingface.co/jinaai/jina-embeddings-v3>
- *`tiktoken`* (token counting): <https://github.com/openai/tiktoken>
- *Hugging Face tokenizers* (`AutoTokenizer`, counting in the model's tokens): <https://huggingface.co/docs/transformers/main_classes/tokenizer>
