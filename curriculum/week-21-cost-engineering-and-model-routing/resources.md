# Week 21 — Resources

Every resource here is **free** or has a free tier. The caching, compression, and routing libraries are open source. The pricing docs are public. The only money you spend is a few dollars of API usage running the 500-query workload (and the routing lab keeps most of it on a cheap local or small model precisely to keep that bill low). The semantic cache reuses your week-7 pgvector setup at zero new cost.

Library names and the exact vendor SKUs/prices move every cohort — the *concepts* (token accounting, the cache-hit-rate lever, the cost-vs-quality trade, the route/cascade decision, the break-even between batching and latency) are stable. When a specific price or page is stale, check the vendor's current pricing page; the *method* doesn't change.

This week sits on top of week 19 (the local cheap tier you route *to*) and week 7 (the pgvector store the semantic cache uses). It feeds the capstone's cost-tracked routing and its required cost report.

## Required reading (work it into your week)

- **Anthropic pricing & token-counting** — the per-model input/output prices and the `count_tokens` endpoint; the source of truth for the cost math's vendor side. Read the prompt-caching pricing (cache-write 1.25×, cache-read ~0.1×) carefully:
  <https://platform.claude.com/docs/en/pricing>
- **Anthropic prompt caching** — the prefix-reuse discount, `cache_control`, and the frozen-prefix discipline; the single biggest free cost lever for repeated context:
  <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
- **GPTCache** — the canonical semantic-cache library (embed the query, vector-lookup, return on similarity); read the architecture and the eviction/similarity-threshold pages even if you build your own over pgvector:
  <https://github.com/zilliztech/GPTCache>
- **LLMLingua** — Microsoft's prompt-compression toolkit (drop low-information tokens before sending); read the README and the compression-ratio / quality trade discussion:
  <https://github.com/microsoft/LLMLingua>

## The caching references

- **Anthropic prompt caching (deep)** — placement, TTL (5m vs 1h), the silent-invalidator failure modes (a timestamp in the prefix kills the cache); same lesson as week 8's chunking-prefix discipline:
  <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
- **pgvector README** — your semantic cache store: `vector_cosine_ops` + the `<=>` operator. The cache is a pgvector lookup with a cosine threshold (week 7's machinery, new use):
  <https://github.com/pgvector/pgvector>
- **OpenAI prompt caching** — the equivalent vendor discount on the OpenAI side, for cross-vendor literacy:
  <https://platform.openai.com/docs/guides/prompt-caching>

## Routing and cascade references

- **LiteLLM routing** — the router that *executes* your routing decision (model aliases, per-request model selection); the same proxy from week 19, now used for *choice* not just *fallback*:
  <https://docs.litellm.ai/docs/routing>
- **RouteLLM** — an open framework for training/serving a query router (easy→cheap, hard→strong); read it for the router-as-classifier framing and the cost-quality evaluation methodology:
  <https://github.com/lm-sys/RouteLLM>
- **FrugalGPT** — Chen et al., 2023, the paper that named the LLM cascade (try cheap, escalate on a scoring check) and the cost-reduction-with-quality-preservation result; the conceptual spine of cascades:
  <https://arxiv.org/abs/2305.05176>

## Compression references

- **LLMLingua / LongLLMLingua** — token-level prompt compression; the compression-ratio vs quality trade is the thing to measure:
  <https://github.com/microsoft/LLMLingua>
- **Selective Context** — an alternative compression approach (prune low-self-information content); useful for comparison:
  <https://github.com/liyucheng09/Selective_Context>

## Batched inference

- **Anthropic Message Batches** — async batch processing at 50% of standard price; up to 24h turnaround; the lever for non-latency-sensitive traffic:
  <https://platform.claude.com/docs/en/build-with-claude/batch-processing>
- **OpenAI Batch API** — the equivalent 50%-off async batch on the OpenAI side:
  <https://platform.openai.com/docs/guides/batch>
- **vLLM continuous batching** (recap from week 19) — the *online* batching that makes your self-hosted tier cheap; distinct from the *offline* batch APIs above:
  <https://docs.vllm.ai/en/latest/>

## Models you'll use this week

- **The cheap tier:** your **week-19 vLLM** Qwen2.5-7B/14B, or **`claude-haiku-4-5`** as a stand-in cheap model if you have no GPU. This is where the router sends easy queries.
- **The frontier tier:** **`claude-sonnet-4-6`** or **`claude-opus-4-8`** for hard queries and cascade escalations — the expensive, high-quality path you route *to* sparingly.
- **The embedding model:** **`BAAI/bge-large-en-v1.5`** (week 7) for the semantic cache's query embeddings — same model, so the cache vectors live in the same space you already know.
- **The judge (for the quality delta):** a frontier model as LLM-as-judge to score whether the cheap/cached answers held vs the baseline (the calibrated-judge pattern from week 12).

## Tools you'll use this week

- **`anthropic`** — `pip install anthropic`. The vendor SDK; `client.messages.create(...)` with `thinking={"type":"adaptive"}` and `output_config={"effort":...}`; `response.usage` is your cost source.
- **`openai`** — `pip install openai`. Used to talk to the local vLLM tier (OpenAI-compatible) and for cross-vendor routing.
- **`sentence-transformers` / `transformers`** — `pip install sentence-transformers`. The BGE embedder for the semantic cache.
- **`psycopg[binary]`** — `pip install "psycopg[binary]"`. pgvector access for the semantic cache store.
- **`llmlingua`** — `pip install llmlingua`. Prompt compression for the stretch goal.
- **`numpy`** — cosine similarity and the cost/quality aggregation.

## A note on the workload

The cost lab runs against a **fixed 500-query workload** with a known *difficulty mix* — a labeled blend of easy questions (FAQ-style, repetitive, a small model handles them) and hard questions (multi-step reasoning, the frontier model is worth it) — plus a small set of *near-duplicate* queries so the semantic cache has something to hit. Fixing the workload is what makes the cost reduction and the quality delta reproducible run-to-run: you change the routing/cache config and re-measure against the *same* 500 queries. The mini-project takes the workload as config so you can point it at the capstone's real query distribution in week 22.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Token accounting** | Attributing cost to features/routes by counting input + output tokens at the per-model price. |
| **Exact-match cache** | Hash the request; return the stored response if seen before. Trivial, only catches identical queries. |
| **Semantic cache** | Embed the query, look up a vector store, return a cached answer above a cosine threshold. |
| **Cosine threshold** | The similarity cutoff for a cache hit; the cost-vs-correctness knob (loose = more hits, more wrong answers). |
| **Prompt caching** | The vendor/self-hosted discount for reusing a stable prompt *prefix*'s KV (cache-read ~0.1× price). |
| **Prompt compression** | Reducing input tokens (summarize, or prune low-information tokens via LLMLingua) at some quality cost. |
| **Model routing** | A classifier sends each query to the cheapest model that can handle it (easy→small, hard→frontier). |
| **Cascade** | Try the cheap model; escalate to the expensive one only if a verification/confidence check fails. |
| **Expected cost** | For a cascade: `cheap_cost + P(escalate) × expensive_cost` — the math that says whether it pays. |
| **Batch API** | Async processing at 50% off with up to 24h turnaround; for non-latency-sensitive traffic. |
| **Cache-hit rate** | Fraction of queries served from cache (no generation); the headline caching metric, tracked over time. |
| **Quality delta** | The change in answer quality (LLM-judge / accuracy) vs the baseline; must stay inside tolerance. |

---

*If a link 404s, please open an issue so we can replace it.*
