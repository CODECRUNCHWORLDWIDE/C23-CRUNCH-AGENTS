# Lecture 1 — Token Accounting, Caching, and Compression: The Cheapest Token Is the One You Don't Generate

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can attribute the cost of an LLM feature precisely from `usage` logs (input vs output, the prompt-caching discount, per-route attribution); build and tune an exact-match and a semantic cache, treating the cosine threshold as the cost-vs-correctness knob; apply prompt caching correctly (frozen prefix, volatile suffix) and measure the cache-read savings; and compress a prompt with summarization or LLMLingua-style pruning while measuring the quality cost, so you compress only where it's free.

If you remember one sentence from this entire week, remember this one:

> **The cheapest token is the one you do not generate. The second cheapest is the one a 7B handles instead of a 70B.**

There's a corollary you should tape next to it:

> **Cost engineering that silently degrades quality is not a saving — it's a bug you'll pay for later.** Every lever this week trades cost against quality. The discipline is to measure *both*: the cost reduction *and* the quality delta, and to accept the trade only when the delta is acceptable.

Lecture 1 covers the first half of the toolkit — the levers that *don't generate the token at all* or *send fewer tokens*: accounting (so you know where the money is), caching (so you don't generate a repeated answer twice), prompt caching (so you don't re-process a repeated prefix), and compression (so you send fewer input tokens). Lecture 2 covers the levers that *send the tokens to a cheaper model*: routing, cascades, and batching. The unifying frame is the cost ordering above — every technique is a move in that ordering, and the whole week is doing the moves *measurably*.

---

## 1. Token accounting: you can't optimize what you don't measure

Before you cut a cost, you measure it, per feature and per route. The atom of LLM cost is the **token**, priced differently for input and output, and reported on every response:

```python
resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=512, messages=[...])
print(resp.usage)
# Usage(input_tokens=1840, output_tokens=210,
#       cache_creation_input_tokens=0, cache_read_input_tokens=1536)
```

That `usage` block is the ground truth — never estimate tokens with `len(text.split())`, which is off by 20–40% and varies by content; use `usage` or the `count_tokens` endpoint. The cost of one call is:

```
cost = input_tokens         × input_price_per_token
     + output_tokens        × output_price_per_token
     + cache_creation_tokens × (input_price × 1.25)   # cache-write premium
     + cache_read_tokens    × (input_price × 0.1)      # cache-read discount
```

Two facts that shape every decision:

- **Output tokens are the expensive ones.** For most models, output is 4–5× the input price (e.g. Claude Sonnet at $3 in / $15 out per million). So a long *prompt* is cheaper than a long *answer*. Cost engineering that caps verbose output (a `max_tokens` ceiling, a "be concise" instruction, a structured-output schema) attacks the expensive side directly.
- **Cache reads are nearly free.** A cached prefix bills at ~0.1× the input price (§4). When a stable prefix repeats, the read discount is enormous — which is why prompt caching is the first lever to reach for on any workload with repeated context.

Accounting is not a one-time exercise — it's a *dashboard*. The capstone requires "cost per query at the median, p95, and p99, broken down by route." You build that from `usage` logs: tag each call with its route (`local-7b`, `frontier`, `cache-hit`), accumulate cost per tag, and plot it. The dashboard is how you find the leak — the one feature that's quietly sending 4000-token prompts to Opus on every keystroke. **You cannot route or cache intelligently until you know which queries cost what**, and that knowledge comes from per-route attribution, not a single monthly total.

> **The discipline:** instrument cost at the call site, tagged by route and feature, from day one. A single "we spent $4000 last month" number tells you nothing actionable; "$3200 of it was the summarize feature sending uncached 6K-token prompts to Opus" tells you exactly what to fix.

Let's do the attribution concretely, because the per-route breakdown is where the leaks become visible. Suppose your product has three features, and you've tagged every `usage` log with its feature and the model it used:

```
feature        model           calls/day   avg in   avg out   $/day
-------------  --------------  ----------  -------  --------  ------
chat           claude-sonnet      40,000     800      300     ~$300
summarize      claude-opus         2,000    6,000      400     ~$80
classify       claude-opus        50,000      200       10     ~$55
```

Read this table like a cost detective. `chat` is the bulk of the spend, but it's high-volume user-facing work — hard to cut without hurting the product. `classify` is 50,000 calls/day of a *trivial* task (200 in, 10 out) running on **Opus** — a frontier model doing a job a 7B would ace. That's $55/day, ~$1,650/month, of pure waste: routing `classify` to a cheap model (Lecture 2) cuts it ~10× for no quality loss, because the task is easy. `summarize` sends 6,000-token prompts uncached; if those prompts share a stable instruction prefix, prompt caching (§4) halves the input cost. **The table doesn't just tell you the total — it tells you which lever to pull where:** route `classify`, cache `summarize`, leave `chat` alone until you have a cheaper idea. Without per-feature, per-model attribution you'd see "$435/day" and have no idea that a third of it is a frontier model doing a 7B's job. The dashboard is the map; the levers are the route.

---

### 1b. The cost ladder, named

It helps to have the levers of this whole week ranked by their typical payoff and their cost, so you reach for them in order. From cheapest-to-apply and biggest-bang to most-effortful:

| Lever | Effort | Quality cost | Typical payoff | When it wins |
|------|--------|--------------|----------------|--------------|
| **Prompt caching** | trivial | none | large on repeated prefixes | any workload with a stable system prompt / tool schema |
| **Exact-match cache** | trivial | none | small–medium | deterministic repeats, retries |
| **Batch API** (Lecture 2) | low | none | 50% on eligible traffic | any latency-tolerant traffic |
| **Semantic cache** | medium | some (threshold-gated) | large on repetitive NL traffic | FAQ-like, paraphrase-heavy workloads |
| **Model routing** (Lecture 2) | medium | some (classifier-gated) | large on mixed-difficulty traffic | easy-majority workloads |
| **Cascade** (Lecture 2) | medium–high | some (verifier-gated) | large when cheap model succeeds often | hard-to-classify-but-easy-to-verify |
| **Prompt compression** | medium | some (ratio-gated) | medium on long-context | big uncacheable contexts |

The first three rows have *no quality cost* — they're free wins you apply unconditionally (the answer is identical, you only changed how it was processed or when). The rest trade some quality, gated by a threshold you tune. So the strategy writes itself: **apply the free levers first (prompt caching, exact-match, batching everything patient), then the quality-trading levers in order of payoff, measuring the quality delta at each step.** The free levers often get you a third of the way for nothing; the rest is the careful, measured part. This ordering is the practical algorithm for cost-engineering any workload, and the mini-project's harness is its implementation.

## 2. Exact-match caching: the trivial, necessary, insufficient win

The simplest cache: hash the request (model + messages + params), and if you've answered that exact request before, return the stored response. Zero generation, zero cost, zero latency.

```python
import hashlib, json

def cache_key(model: str, messages: list, **params) -> str:
    payload = json.dumps({"model": model, "messages": messages, **params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def cached_complete(client, model, messages, store, **params):
    key = cache_key(model, messages, **params)
    if key in store:
        return store[key]                      # HIT: $0, instant
    resp = client.messages.create(model=model, messages=messages, **params)
    store[key] = resp.content[0].text
    return store[key]
```

Note `sort_keys=True` — the same lesson as week 8's prompt-caching prefix discipline: non-deterministic serialization means the same logical request hashes differently and never hits. Sort the keys.

Exact-match caching is **necessary** (it's free and catches genuine repeats — the same FAQ asked verbatim, a retried identical request, a deterministic pipeline step) but **insufficient** for natural-language workloads, because users rarely ask the *exact same bytes* twice. "What's your refund window?" and "How long do I have to get a refund?" are the same question and different strings — exact-match misses the second. That gap is what semantic caching closes.

> **Rule of thumb:** always put exact-match caching in front (it's nearly free to add and catches deterministic repeats and retries), then add semantic caching for the natural-language near-duplicates. The two stack: exact-match first (cheapest check), semantic second (catches paraphrases).

---

## 3. Semantic caching: cache by meaning, not by bytes

Semantic caching embeds the query and looks for a *semantically similar* past query whose answer you can reuse. The mechanism is exactly week 7's retrieval, repurposed:

1. Embed the incoming query (BGE-large, the model you know).
2. Look up the nearest past query in a pgvector cache (`<=>` cosine distance).
3. If the nearest neighbor's similarity is **above a threshold** (e.g. 0.92 cosine), return its cached answer — a hit. Otherwise, generate, and store the new (query-embedding, answer) pair.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

def semantic_complete(client, query, cache_store, threshold=0.92, **params):
    q_vec = embedder.encode(query, normalize_embeddings=True)
    hit = cache_store.nearest(q_vec)           # pgvector: ORDER BY embedding <=> q_vec LIMIT 1
    if hit is not None and hit.similarity >= threshold:
        return hit.answer, "cache-hit", 0.0    # reuse the stored answer
    resp = client.messages.create(messages=[{"role": "user", "content": query}], **params)
    answer = resp.content[0].text
    cache_store.insert(q_vec, query, answer)
    return answer, "generated", cost_of(resp.usage)
```

The **cosine threshold is the cost-vs-correctness knob**, and it has a predictable trade-off curve, exactly like week 8's chunk-size sweep:

- **Loose threshold** (e.g. 0.80) → more queries count as "similar enough" → higher hit rate → bigger cost saving, *but* you start returning the cached answer to questions that are only *roughly* similar — "what's your refund window?" served from the cache for "what's your *return* window?", which is a different policy. **Loose caching returns wrong answers.**
- **Tight threshold** (e.g. 0.98) → only near-identical queries hit → fewer wrong answers, *but* a low hit rate that barely saves money — you've made it almost exact-match.
- **Just right** → somewhere in the middle, the hit rate is meaningful *and* the hits are genuinely the same question. You find it by sweeping the threshold against your labeled workload and reading the cost-saving and the wrong-answer-rate curves — both, together.

```
hit rate / cost saving         wrong-answer rate
   │           ____             │          ____
   │         /                  │        /        (loose: cheap but WRONG)
   │       /                    │      /
   │     /                      │ ___/
   └──────────────── threshold  └──────────────── threshold
     loose ........ tight         loose ........ tight
```

The sweet spot is where the hit rate is still high but the wrong-answer rate hasn't started climbing — and that point is *workload-dependent*, so you measure it (Exercise 2 is exactly this sweep). The two failure modes to name: **a too-loose threshold serves wrong answers cheaply** (the saving is fake — you saved money and gave the wrong answer), and **staleness** — a cached answer to "what's our refund window?" goes wrong the day the policy changes. Semantic caches need an invalidation story (TTL, version key, manual purge on policy change) precisely because the world moves and the cache doesn't.

> **The discipline:** never ship a semantic-cache threshold you didn't sweep against a labeled set. "0.92 because a blog said so" is a guess; "0.92 because at 0.92 we hit 41% with a 0.5% wrong-answer rate, and 0.88 doubled the wrong answers" is an answer.

Here's what a threshold sweep actually produces, so you know what you're reading (this is Exercise 2's output). Take a labeled set of query pairs — some genuine paraphrases (should hit), some superficially-similar-but-different questions (should *not* hit) — and sweep:

```
threshold   hit-rate   wrong-answer-rate   net-saving
  0.80        0.62          0.084            (FAKE: 8% of answers wrong)
  0.85        0.51          0.031
  0.90        0.44          0.009
  0.92        0.41          0.005            <- sweet spot
  0.95        0.29          0.001
  0.98        0.12          0.000            (barely better than exact-match)
```

Read the trade right to left: at 0.98 the cache is so strict it's almost exact-match (12% hits, no wrong answers). As you loosen, the hit rate climbs — but watch the wrong-answer column. Between 0.92 and 0.90 the hit rate barely moves (0.41 → 0.44) while the wrong-answer rate nearly doubles (0.5% → 0.9%); below 0.85 it explodes (3% → 8%). The sweet spot is the loosest threshold *before* the wrong-answer rate starts climbing steeply — here, ~0.92. Notice this is **not** "maximize the hit rate": 0.80 has the highest hit rate and the worst product, because the 8% wrong answers are a quality disaster masquerading as a cost win. The right objective is "maximize hit rate *subject to* wrong-answer rate staying under tolerance," and the sweep is how you find that constrained optimum. Quote the threshold *with* both numbers, always — a hit rate without a wrong-answer rate is half the story and the dangerous half.

---

### 3b. Cache invalidation: the hard problem nobody mentions

"There are only two hard things in computer science: cache invalidation and naming things." Semantic caching inherits the first in a sharp form, and skipping it is how a cost win becomes a correctness incident. The cache stores an answer; the world that produced that answer keeps changing. Three invalidation strategies, in increasing order of effort and safety:

- **TTL (time-to-live).** Every cache entry expires after a fixed interval (an hour, a day). Simple, and bounds staleness to the TTL — but it's a blunt instrument: a never-changing fact (the speed of light) gets needlessly re-generated every TTL, and a fact that changed *within* the TTL is served stale until it expires. TTL is the default; it's right when staleness for up-to-the-TTL is tolerable.
- **Version keying.** Tag each cache entry with a version of the underlying source (the policy document's revision, the knowledge base's commit hash). When the source changes, bump the version, and entries keyed to the old version stop matching. Surgical — only the affected answers invalidate — but requires you to *know* what source each cached answer depends on, which isn't always tractable for free-form generation.
- **Event-driven purge.** On a known change (the refund policy is updated, a product is discontinued), explicitly purge the cache entries that touch it. The most precise, the most work, and it requires a mapping from "thing that changed" to "cached answers affected" that you have to maintain.

The failure mode to fear: a customer asks "what's our refund window?", the policy changes from 30 to 60 days, and the semantic cache keeps confidently serving "30 days" to every paraphrase of the question for the life of the entry. That's not a cost saving — it's the cache actively giving wrong, *and previously-correct*, answers. The cache made the system cheaper and wrong, and "cheaper and wrong" is the worst quadrant. **A semantic cache without an invalidation story is a time bomb on any answer that can go stale**, which is most factual answers in a real product. Decide the invalidation strategy when you build the cache, not after the first stale-answer incident.

> **Rule of thumb:** cache freely the answers that *can't* go stale (definitions, math, stable explanations); cache carefully, with a TTL or version key, the answers that *can* (policies, prices, inventory, anything about the current state of the world). The dangerous cache is the one that treats a volatile fact as if it were a permanent one.

## 4. Prompt caching: don't re-process the prefix you already sent

Caching (§2–3) avoids *generating* a repeated answer. **Prompt caching** avoids *re-processing* a repeated *input prefix*. You met it in earlier weeks and in week 19's `--enable-prefix-caching`; here's the cost framing.

When every request shares a large stable prefix — a long system prompt, a tool schema, a few-shot block, a retrieved document you ask several questions about — the model normally re-reads (prefills) that prefix on every call, and you pay full input price for it every time. Prompt caching stores the prefix's processed state (the KV cache) and reuses it: the first call pays a small write premium (~1.25× input price for the cached portion), and every subsequent call pays the **read** price (~0.1×) for that prefix instead of full price.

```python
resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    system=[{
        "type": "text",
        "text": LONG_STABLE_SYSTEM_PROMPT,        # the frozen prefix
        "cache_control": {"type": "ephemeral"},   # cache it
    }],
    messages=[{"role": "user", "content": user_question}],   # the volatile suffix
)
# resp.usage.cache_read_input_tokens > 0 on calls after the first => the cache hit
```

The math: a 2000-token system prompt that precedes a 50-token question, asked 100 times. Uncached, you pay `100 × 2050 × input_price`. Cached, you pay `1 × 2000 × 1.25 × input_price` (the write) + `99 × 2000 × 0.1 × input_price` (the reads) + `100 × 50 × input_price` (the always-uncached questions). The 2000-token prefix's cost drops from ~`200,000` token-equivalents to ~`22,300` — roughly a 9× cut on the prefix. On context-heavy workloads (RAG, agents with big system prompts), prompt caching alone can halve the bill.

The catch is the same byte-stability discipline from week 8 and week 19: **the cached prefix must be byte-identical across calls.** Interpolate a timestamp, a per-request ID, or an unsorted JSON dump into the front of the prompt and the cache never hits — `cache_read_input_tokens` stays zero and you've paid the write premium for nothing. **Put the stable content first and the volatile content last.** Verify with `cache_read_input_tokens`: if it's zero across repeated calls, a silent invalidator is in your prefix. This is identical in spirit to week 19's prefix caching — vendor or self-hosted, the rule is the same: freeze the prefix.

> **Rule of thumb:** prompt caching is the first cost lever to reach for on any workload with repeated context, because it's free quality-wise (the answer is identical — you only changed how the input was processed) and the discount is large. The only way to get it wrong is to break prefix stability.

---

## 5. Prompt compression: send fewer input tokens

When the input itself is large and not perfectly cacheable — a long document you summarize, a big retrieved context, a verbose conversation history — you can **compress** it: send fewer input tokens to get (approximately) the same answer. Two families:

**Summarization (controllable, lossy).** Replace a long context with a shorter summary before sending it to the expensive model. You control the compression ratio (summarize 4000 tokens to 800), and the loss is whatever the summary dropped. This is the same rolling/hierarchical summarization from week 11's memory tiers, used here as a cost lever: a cheap model produces the summary, the expensive model answers from it. The trade is explicit — you saved input tokens, you may have lost a detail the summary omitted.

**Token pruning (LLMLingua-style).** LLMLingua and similar tools drop *low-information tokens* from a prompt — words a small language model judges the big model can reconstruct or doesn't need — keeping the high-information ones. It compresses a prompt by 2–5× while aiming to preserve the answer. It's more surgical than summarization (it prunes within the prompt rather than rewriting it) but less interpretable (the compressed prompt is partly unreadable to humans).

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor()
result = compressor.compress_prompt(long_context, rate=0.5)   # keep ~50% of tokens
compressed = result["compressed_prompt"]
# send `compressed` to the expensive model instead of `long_context`
```

The discipline is the one this whole week turns on: **measure the quality cost of the compression, and only compress where it's free (or cheap enough).** Compress a context 2× and check the answer-quality delta on your labeled set. If quality held, you saved input tokens for free. If quality dropped, you've traded correctness for cost — sometimes worth it, sometimes a regression in disguise. There's a compression ratio where quality starts to fall off a cliff (the prompt lost something load-bearing); your safe operating point is *before* that cliff, found by sweeping the ratio and watching the quality delta. The README stretch goal is exactly this sweep.

> **The discipline:** compression is a cost-vs-quality trade like every other lever. Sweep the ratio, watch the quality delta, and operate before the cliff. Compress the fat (boilerplate, redundancy) freely; be careful compressing the substance.

---

## 6. The levers stack — and the order matters

These four levers (plus Lecture 2's routing/cascades/batching) compose, and a cost-engineered pipeline applies them in a sensible order, cheapest check first:

```
incoming query
   │
   ├─ exact-match cache?  ── HIT ─→ return ($0)          # §2: cheapest possible
   │
   ├─ semantic cache?     ── HIT ─→ return ($0)          # §3: catch paraphrases
   │
   ├─ compress the prompt if it's long                   # §5: fewer input tokens
   │
   ├─ prompt caching on the stable prefix                # §4: cheap repeated prefix
   │
   └─ route to the cheapest model that can answer        # Lecture 2: cheap > frontier
         │
         └─ cascade: escalate only if the cheap answer fails a check   # Lecture 2
```

Each layer that hits *short-circuits* the more expensive layers below it. A query answered from the exact-match cache never touches an embedder, a compressor, or a model. A query answered from the semantic cache never touches a model. A query that does reach a model goes to the *cheapest* one that can handle it, with its prefix cached and its prompt compressed. The savings multiply: a 40% cache-hit rate means 40% of queries cost $0; of the remaining 60%, routing sends most to the cheap tier; of those, prompt caching and compression cut the per-call cost further. **The naive pipeline sends every query, full-length, uncached, to the frontier model. The engineered pipeline does that to a small fraction of queries.** That multiplication is how an 88% cost reduction happens — not one magic lever, but five stacked levers each taking a bite, measured at each step.

The ordering also reflects cost-of-the-check: an exact-match hash is nearly free to compute, so it goes first; a semantic lookup costs an embedding, so it goes second; compression and a model call are progressively more expensive. You check the cheap things first so you can skip the expensive things when they hit.

---

## 6b. A worked stacking example with numbers

The §6 diagram shows the *order*; here's the *arithmetic*, so the multiplication of savings is concrete. Start with a baseline: 1,000 queries/day, all sent full-length and uncached to a frontier model, average $0.04/query.

```
baseline:  1000 × $0.04                                      = $40.00/day
```

Now apply the levers one at a time and watch the bill fall:

```
+ exact-match cache (5% of queries are exact repeats/retries):
    950 × $0.04                                              = $38.00/day   (-5%)

+ semantic cache @ 0.92 (another 35% are paraphrase hits):
    (950 − 333 cache hits) = 617 × $0.04                     = $24.68/day   (-38% cumulative)

+ prompt caching on the shared system prefix (cuts input cost ~40% on the 617):
    617 × $0.028                                             = $17.28/day   (-57% cumulative)

+ routing (70% of the 617 are easy -> 10× cheaper local tier):
    (185 hard × $0.028) + (432 easy × $0.0028)               = $6.39/day    (-84% cumulative)
```

From $40 to $6.39 — an 84% cut — and *no single lever did it*. Exact-match took 5%, semantic caching took the biggest single bite (catching the paraphrase repeats), prompt caching shaved the input cost on everything that still hit a model, and routing moved the easy majority to the cheap tier. Each lever operates on what the previous ones left, so the percentages compound multiplicatively, not additively. This is why the headline numbers in this field sound implausible until you do the arithmetic: an "88% cost reduction" isn't one heroic optimization, it's four or five ordinary ones stacked, each taking a bite of a progressively smaller bill.

The corollary for *where to spend your engineering time*: the order of the bites tells you the order of the payoff. On a repetitive workload, caching is the biggest lever (it zeroes out whole queries); on a workload of mostly-unique-but-easy queries, routing is the biggest (caching has nothing to hit). You don't know which until you measure your workload's repetition rate and difficulty mix — which is, once more, why the labeled workload and the per-route dashboard come first. **Measure the workload, then apply the levers in the order their payoff dictates, and re-measure after each.**

## 6c. The one number that ties it together: cost per query

For the capstone's cost report and for any real product, the headline metric is **cost per query**, reported at the median, p95, and p99. Why those three? Because the *distribution* of per-query cost is as important as the average:

- **The median** tells you the typical query's cost — usually low, because most queries are cheap (cached or routed to the local tier).
- **The p95 and p99** tell you the *expensive tail* — the queries that escalated to the frontier model, ran a big context through Opus, or triggered a cascade. These are where the budget risk lives: a system with a great median but a brutal p99 will blow its budget on a traffic mix that's slightly more "hard" than expected.

A cost-per-query distribution like "median $0.0003, p95 $0.02, p99 $0.06" tells a story: the typical query is nearly free (cache or local tier), but the hardest 1% cost 200× the median (frontier escalations). That's a healthy shape — cheap by default, expensive only where it has to be. An unhealthy shape is "median $0.04, p95 $0.05" — everything is expensive, nothing is cached or routed, the levers aren't working. **The cost-per-query distribution is the diagnostic; the average alone hides whether your cheap path is actually carrying the load.** Report all three, and the shape will tell you whether your cost engineering is doing its job or just averaging out a uniformly expensive system.

## 7. The carbon footnote (it aligns with cost)

The syllabus mentions the carbon story, and like in week 19 it aligns with the cost argument rather than competing with it. Every token you don't generate is energy you don't spend; every query a 7B answers instead of a 70B is ~10× less compute and ~10× less energy; every cache hit is a forward pass that never ran. **The same levers that cut your bill cut your carbon footprint, in roughly the same proportion**, because both scale with tokens-times-model-size. There's no tension here: cost-engineering *is* carbon-engineering for LLM workloads. The one nuance is that a cache hit on a *self-hosted, already-running* GPU saves less marginal energy than a cache hit that avoids a *vendor* call (the vendor's GPU was going to be busy anyway, pooled across customers) — but the direction is the same, and "generate fewer/cheaper tokens" is the right move for both the bill and the planet.

---

## 7b. Where the levers fit in the capstone

This isn't an academic exercise — every lecture in this week maps to a specific capstone deliverable, and it's worth naming the connection so the work feels load-bearing. The capstone (the production agentic research assistant) requires a **cost report** — "total cost per query at the median, p95, and p99, broken down by route, with cache-hit accounting." That report is built from exactly the machinery here: the per-route token accounting (§1) produces the breakdown, the semantic cache (§3) produces the cache-hit accounting, and the cost-per-query distribution (§6c) produces the median/p95/p99 view. The capstone also runs a *hybrid* serving model — local 7B/13B for cheap traffic, a vendor frontier model for hard traffic — which is precisely the routing of Lecture 2, executed through the LiteLLM router from week 19.

So this week is not "learn some cost tricks"; it's "build the cost layer the capstone is graded on." The `crunchroute` harness (the mini-project) is the seed of that layer — you point it at the capstone's real query distribution in week 22 and it produces the cost report. Treat the labeled-workload discipline, the threshold sweeps, and the quality-delta measurement as practice for the artifact a sealed-review panel will read, because that's what they are. A capstone that ships a routing layer with no cost report, or a cost report with no quality delta, fails the measurement axis of the rubric — and this week is where you learn to not let that happen.

## 8. Recap

You should now be able to:

- **Account** for LLM cost from the `usage` block — input vs output pricing (output is the expensive side), the cache-write premium and cache-read discount — and build a *per-route, per-feature* cost dashboard rather than a single monthly total.
- Build an **exact-match cache** (hash the request, sorted keys) — necessary and free but insufficient for natural language — and a **semantic cache** (embed, pgvector lookup, cosine threshold) that catches paraphrases.
- Tune the **semantic-cache cosine threshold** as the cost-vs-correctness knob, sweeping it against a labeled set and watching *both* the hit rate and the wrong-answer rate, and name the staleness/invalidation problem.
- Apply **prompt caching** correctly — frozen prefix, volatile suffix — compute the prefix-reuse discount, and verify hits via `cache_read_input_tokens`.
- **Compress** prompts (summarization or LLMLingua token pruning), measure the quality delta against a labeled set, and operate before the quality cliff — compressing the fat freely, the substance carefully.
- See the levers as a **stacked, short-circuiting pipeline** (exact → semantic → compress → prompt-cache → route → cascade) where the multiplication of savings produces the headline cost reduction, and where the same levers cut carbon in proportion.

Carry these one-line takeaways into the exercises:

- Output tokens are the expensive ones; cache reads are nearly free; attribute cost per route, not per month.
- Exact-match first (free, catches repeats), semantic second (catches paraphrases, threshold-gated).
- The semantic-cache threshold is swept against a labeled set; quote it with *both* the hit rate and the wrong-answer rate.
- Every cacheable answer needs an invalidation story; the dangerous cache treats a volatile fact as permanent.
- Prompt caching wants a frozen prefix and a volatile suffix; verify with `cache_read_input_tokens`.
- Compression is a measured trade; operate before the quality cliff.
- The levers stack and short-circuit; the savings multiply; apply the free ones first.

Next: the levers that send tokens to a *cheaper model* — routing with a classifier, cascades with escalation, the expected-cost math, and batched inference — plus the measurement that proves the cost fell and the quality held. Continue to [Lecture 2 — Model Routing, Cascades, and Batching](./02-model-routing-cascades-and-batching.md).

---

## References

- *Anthropic pricing & prompt caching*: <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
- *GPTCache (semantic caching)*: <https://github.com/zilliztech/GPTCache>
- *LLMLingua (prompt compression)*: <https://github.com/microsoft/LLMLingua>
- *FrugalGPT* — Chen et al., 2023 (cascades, cost reduction with quality preservation): <https://arxiv.org/abs/2305.05176>
- *pgvector* (the semantic cache store): <https://github.com/pgvector/pgvector>
