# Week 21 — Cost Engineering and Model Routing

Welcome to the week your system stops being expensive by accident. For twenty weeks you have called models without much thought to the bill — a few cents here, a Claude call there. This week the bill is the subject. You learn to account for every token, cache the answers you can, compress the prompts you must send, route easy questions to a cheap small model and hard ones to a frontier model, and prove with a measured workload that your cost-engineered pipeline costs a fraction of the naive one — *without* losing answer quality. By Friday you can look at any LLM product and say, with a number, what each feature costs per query and where the money is leaking.

This is week 3 of **Phase IV — Production AI & Capstone**, and it builds directly on week 19. Week 19 taught you *how cheap a self-hosted token can be* (continuous batching, the break-even). This week teaches you *how to send fewer tokens to the expensive model in the first place* — which is the larger lever. The two compose: route the easy traffic to your cheap local vLLM tier (week 19), the hard traffic to a vendor, cache what repeats, compress what's long, and the capstone's cost report writes itself. Everything here feeds the capstone's "cost-tracked routing between local 7B/13B and a vendor frontier model" and its required per-query cost report.

The one sentence to internalize before you read another line:

> **The cheapest token is the one you do not generate. The second cheapest is the one a 7B handles instead of a 70B.**

Here is why that ordering is the whole week. Every cost-engineering technique is one of those two moves. **Caching** (exact-match or semantic) is "don't generate the token at all — you already have the answer." **Prompt compression** is "send fewer input tokens to get the same answer." **Prompt caching** is "don't re-process the tokens you already sent." **Model routing** and **cascades** are "let the small/cheap model handle it, and only escalate to the big/expensive model when the small one can't." Stack them and a workload that costs $1000/month naively costs $150 with the same answer quality. The skill is doing it *measurably* — proving the savings and proving the quality held — not guessing.

There is a corollary worth taping next to it:

> **Cost engineering that silently degrades quality is not a saving — it's a bug you'll pay for later.** Routing every query to the 7B saves money and tanks your hard-question accuracy. A semantic cache with too loose a threshold returns a "close enough" answer that's wrong. Every lever this week trades cost against quality, and the discipline is to measure *both* — cost reduction *and* the quality delta — and to accept the trade only when the quality delta is acceptable.

## Learning objectives

By the end of this week, you will be able to:

- **Account** for the cost of an LLM feature precisely — per-token input/output pricing, the prompt-caching discount, per-route and per-feature attribution — and read a cost dashboard that says where the money goes.
- **Build** an exact-match response cache and a **semantic cache** (embed the query, look up a pgvector store, return the cached answer above a cosine threshold), and tune the threshold against the cost-vs-correctness trade-off.
- **Apply** prompt caching (the vendor/self-hosted prefix-reuse discount) correctly — frozen prefix, volatile suffix — and measure the cache-read savings.
- **Compress** prompts with summarization and with **LLMLingua**-style token pruning, and measure the quality cost of the compression so you only compress where it's free.
- **Route** queries with a small classifier: easy → local cheap model, hard → frontier vendor model; and build a **cascade** (try the cheap model, escalate to the expensive one only if a confidence/verification check fails) — and measure the cost reduction and the quality delta against a single-model baseline.
- **Batch** non-latency-sensitive work through the OpenAI/Anthropic Batch APIs (50% discount) and vLLM continuous batching, and decide when latency tolerance makes batching the right lever.
- **Measure** a cost-engineered pipeline end to end on a fixed workload: total cost, per-route cost, cache-hit rate over time, and the quality delta vs the naive baseline — and write the cost-reduction memo the syllabus asks for.

## Prerequisites

This week assumes you have completed **C23 weeks 1–20**, or have equivalent fluency. Specifically:

- You finished **week 19** and have the `crunchserve` harness or equivalent: you can stand up a local vLLM model behind LiteLLM and you understand the self-hosted-vs-vendor cost story. This week routes *to* that local tier as the cheap path.
- You finished **week 7** and can embed text and query a pgvector store — the semantic cache is a pgvector lookup with a cosine threshold, reusing exactly that machinery.
- You are comfortable with the `openai` and `anthropic` SDKs and have metered token usage from `response.usage` before. Cost accounting is built on that `usage` block.
- You can write a small classifier (a few-shot prompt, a tiny fine-tune, or a heuristic) — the router's "easy vs hard" decision is a classification problem.

You do **not** need a GPU for most of this week (the cost math, caching, compression, and routing logic are CPU/API-bound). The *local cheap model* in the routing lab can be your week-19 vLLM server, an Ollama model, or — if you have no GPU — a small vendor model (`claude-haiku-4-5`) standing in for "the cheap tier" so the routing *logic* and the cost *measurement* are reachable without local inference.

## Topics covered

- **Token accounting:** input vs output pricing, the prompt-caching discount, per-route and per-feature cost attribution, and building a cost dashboard from `usage` logs.
- **Exact-match caching:** hash the request, return the stored response; the trivial win for repeated identical queries, and why it's necessary but rarely sufficient.
- **Semantic caching:** embed the query, look up a pgvector cache, return the cached answer above a cosine threshold; the threshold as the cost-vs-correctness knob; staleness and invalidation.
- **Prompt caching:** the vendor and self-hosted prefix-reuse discount; frozen-prefix / volatile-suffix discipline; measuring the cache-read fraction.
- **Prompt compression:** summarization (lossy, controllable) and **LLMLingua**-style token pruning (drop low-information tokens); measuring the quality cost so you compress only where it's free.
- **Model routing:** a classifier sends easy queries to a cheap small model and hard ones to a frontier model; building the classifier; the cost reduction and the quality delta.
- **Cascades:** try the cheap model first, escalate to the expensive one only if a verification/confidence check fails; the expected-cost math; when a cascade beats a router.
- **Batched inference:** OpenAI Batch / Anthropic Batch (50% discount, async), vLLM continuous batching; the latency-tolerance question that decides when to batch.
- **Speculative decoding as a cost lever** (recap from week 19) and the **carbon** story: fewer/cheaper tokens is also fewer grams of CO₂.
- **The measurement:** total cost, per-route cost, cache-hit rate over time, quality delta vs baseline — the cost-reduction memo.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                           | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-----------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Token accounting; exact + semantic caching; prompt caching      |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | Semantic cache threshold tuning; prompt compression             |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Model routing; the classifier; cascades; the expected-cost math |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Batching; the measurement methodology; building the harness     |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The 500-query workload run + cost memo; routing clinic          |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work                                          |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, memo polish                                       |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                                 | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The pricing docs, GPTCache/semantic-cache references, LLMLingua, batch-API docs, routing references |
| [lecture-notes/01-token-accounting-caching-and-compression.md](./lecture-notes/01-token-accounting-caching-and-compression.md) | Token accounting, exact + semantic caching, prompt caching, prompt compression |
| [lecture-notes/02-model-routing-cascades-and-batching.md](./lecture-notes/02-model-routing-cascades-and-batching.md) | The router + classifier, cascades, the expected-cost math, batched inference, the measurement |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-token-accounting.md](./exercises/exercise-01-token-accounting.md) | Meter a real conversation, attribute cost per route, build a cost table from `usage` |
| [exercises/exercise-02-semantic-cache.py](./exercises/exercise-02-semantic-cache.py) | Build a semantic cache over pgvector and sweep the cosine threshold against cost vs correctness |
| [exercises/exercise-03-router.py](./exercises/exercise-03-router.py) | Build an easy/hard classifier router and a cascade; measure cost reduction and quality delta |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-cost-reduction.md](./challenges/challenge-01-cost-reduction.md) | The full cost-reduction lab: route + semantic cache over 500 queries, measure cost cut and quality delta, write the memo |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the one-page cost-reduction memo |
| [mini-project/README.md](./mini-project/README.md) | The `crunchroute` cost-engineering harness — router + cascade + semantic cache + cost reporting |

## The "the cheap path answered correctly" promise

C23 uses a recurring marker for every exercise that ends in a query being served *cheaply and correctly* because the routing and caching were right:

```
$ python route.py --workload workload.jsonl --threshold 0.92
queries=500  routing=classifier  cache=semantic(0.92)
  cost: $0.84   (baseline all-frontier: $7.20  ->  88% reduction)
  cache-hit rate: 0.41   routed-to-local: 0.52   escalated: 0.07
  quality delta vs baseline: -0.01 (within tolerance)  ✓
  q137 ("what's our refund window?") -> cache HIT (0.97)  $0.00  ✓
```

If that quality delta reads -0.15 instead of -0.01, your cost saving is a quality regression in disguise — you routed too aggressively or cached too loosely. The point of week 21 is to cut the cost *and prove the answers held* — to show an 88% reduction with a quality delta inside tolerance, not a vibe about how "routing saves money."

## Stretch goals

If you finish the regular work early and want to push further:

- Build a **calibrated cascade**: instead of a hard escalation rule, score the cheap model's answer with a lightweight verifier (an LLM-judge or a confidence heuristic) and escalate only below a tuned threshold. Plot the cost-vs-quality frontier as you move the threshold.
- Add **LLMLingua** prompt compression to your longest prompts and measure the token reduction *and* the answer-quality delta. Find the compression ratio where quality starts to drop — that's your safe operating point.
- Run the same workload through the **Anthropic Batch API** for the non-urgent queries and measure the 50% discount against the latency cost (results in up to 24h). Decide which slice of your traffic is batch-eligible.
- Build the **cost dashboard** the capstone requires: per-route cost, cache-hit rate over time, and cost-per-query at p50/p95/p99 — three views from your `usage` logs. This is a direct down-payment on a capstone deliverable.

## Up next

Week 22 is the **Capstone Sprint A** — you start building the production agentic research assistant for real, landing its architecture, hybrid retrieval, and memory tiers. The cost-engineered routing layer you build this week becomes the capstone's serving-and-cost layer; the semantic cache becomes a real component; the cost report becomes a required deliverable. Push your `crunchroute` harness before you start — the capstone's cost-tracked routing starts from it.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
