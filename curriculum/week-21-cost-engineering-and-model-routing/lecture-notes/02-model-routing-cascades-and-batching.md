# Lecture 2 — Model Routing, Cascades, and Batching: Let the 7B Handle It

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can build a model router that sends easy queries to a cheap small model and hard ones to a frontier model, and a cascade that escalates only when a cheap answer fails a check; do the expected-cost math that says whether a cascade pays; decide when latency tolerance makes batched inference the right lever; and measure a cost-engineered pipeline end to end — total cost, per-route cost, cache-hit rate, and the quality delta vs the naive baseline — to write a defensible cost-reduction memo.

Lecture 1 covered the levers that avoid generating the token or send fewer input tokens. This lecture covers the second clause of the week's mantra:

> **The second cheapest token is the one a 7B handles instead of a 70B.**

Routing and cascades are how you act on that. The throughline:

> **Most queries are easy, and a cheap model answers them correctly. The expensive model is for the minority of hard queries where its quality is worth the price. Routing is the discipline of spending the frontier model only where it earns its cost.**

---

## 1. Model routing: a classifier in front of the model fleet

The core idea: not every query needs your best model. A typical workload is a *mix* — many easy queries (an FAQ lookup, a simple classification, a short rewrite) that a 7B answers as well as a 70B, and a minority of hard queries (multi-step reasoning, nuanced analysis, code generation) where the frontier model's quality genuinely matters. **Routing** puts a cheap **classifier** in front of the fleet that decides, per query, which model should handle it:

```python
def route(query: str) -> str:
    """Return the model alias to use for this query."""
    if is_hard(query):          # the classifier
        return "frontier"       # claude-opus-4-8 / sonnet — expensive, high quality
    return "local-7b"           # the cheap tier (your week-19 vLLM) — cheap, good enough

def routed_complete(query, clients):
    model = route(query)
    return clients[model].complete(query), model
```

The classifier `is_hard(query)` can be:

- **A heuristic** — query length, presence of reasoning keywords ("explain why", "step by step", "compare"), the number of retrieved documents. Cheap, transparent, a fine baseline.
- **A few-shot prompt to a small model** — ask a cheap model "is this query easy or hard?" with a handful of labeled examples. More flexible, costs a small classification call (which the routing then saves many times over).
- **A trained classifier** — a small fine-tune or a logistic regression over query embeddings, trained on a labeled easy/hard set (the RouteLLM approach). Most accurate, most work.

The economics: if 70% of your traffic is easy and goes to a model that's 10× cheaper, and 30% is hard and goes to the frontier model, your blended cost is `0.7 × cheap + 0.3 × expensive` instead of `1.0 × expensive` — roughly a 65% cut if the cheap model is 10× cheaper. **The win is the fraction of traffic you can safely keep off the expensive model**, which is exactly what the classifier decides.

And the trap, which is the week's recurring discipline: **measure the quality delta, not just the cost.** Routing a hard query to the 7B saves money *and gives a worse answer*. If your classifier mislabels hard queries as easy, you've degraded quality to save cost — a bad trade. The router is only as good as the classifier's accuracy on the *hard* class, and you measure that against the labeled workload: cost reduction *and* the quality delta on the routed-to-cheap queries. A router that cuts cost 65% with a -0.01 quality delta is a win; one that cuts cost 70% with a -0.15 delta is a regression you'll regret in production.

> **The discipline:** a router's value is `cost_saved × (quality_held)`. Report both. The classifier's job is to keep hard queries *off* the cheap tier; its failure mode is false-easy classifications, which you catch by measuring quality on the routed-to-cheap set.

### 1b. Why the *direction* of the classifier's errors matters

A subtle but important point: a router's two error types are not symmetric in cost. The classifier can be wrong two ways:

- **False-easy** (routes a *hard* query to the cheap model). The user gets a worse answer. This is a **quality** failure, and it's the one that hurts in production, because it's invisible in the cost dashboard — the query was answered cheaply, the bill went down, and only the user noticed the answer was bad.
- **False-hard** (routes an *easy* query to the frontier model). The user gets a fine answer (the frontier model handles easy queries well); you just overpaid. This is a **cost** failure — wasteful but harmless to quality.

Because false-easy errors cost quality and false-hard errors only cost money, **you tune the classifier to be conservative — biased toward escalating to the frontier model when unsure.** A router that occasionally over-routes to the expensive tier (false-hard) is leaking a little money; a router that occasionally under-routes hard queries to the cheap tier (false-easy) is leaking quality, which is worse. So the classifier's threshold isn't set to maximize accuracy symmetrically — it's set to minimize false-easy errors even at the cost of some false-hard ones. This is the same asymmetry as a medical test where a false negative is worse than a false positive: you bias toward caution on the expensive-to-miss class. In routing, the expensive-to-miss class is "hard," and missing it means a bad answer, so you bias toward calling things hard.

This is why the quality delta is measured specifically on the *routed-to-cheap* set — that's where false-easy errors live, and that's the set that tells you whether your conservative bias is conservative enough.

---

## 2. Cascades: try cheap, escalate only on failure

A **cascade** is a smarter version of routing that doesn't require the classifier to be right *up front*. Instead of deciding the model before generating, you:

1. **Try the cheap model first.**
2. **Check its answer** — with a verifier (an LLM-judge, a confidence score, a schema/format check, a self-consistency check).
3. **Escalate to the expensive model only if the check fails.**

```python
def cascade_complete(query, cheap_client, frontier_client, verify):
    cheap_answer = cheap_client.complete(query)
    if verify(query, cheap_answer):          # good enough?
        return cheap_answer, "cheap", cost_cheap
    frontier_answer = frontier_client.complete(query)   # escalate
    return frontier_answer, "escalated", cost_cheap + cost_frontier
```

The elegance: the cheap model handles everything it *can* handle (no classifier needed to predict difficulty), and the expensive model is invoked only for the queries the cheap model demonstrably failed. This is the FrugalGPT pattern (Chen et al., 2023), and it can match the frontier model's quality at a fraction of the cost — because most queries pass the cheap model's check and never escalate.

The **expected-cost math** tells you whether a cascade pays:

```
expected_cost = cost_cheap + P(escalate) × cost_frontier
```

Every query pays `cost_cheap` (you always try the cheap model first). A fraction `P(escalate)` *also* pays `cost_frontier`. Compare to the all-frontier baseline `cost_frontier`:

- If `P(escalate)` is low (the cheap model handles most queries) and `cost_cheap << cost_frontier`, the cascade is much cheaper. E.g. cheap = $0.10, frontier = $1.00, escalate 20% of the time → `0.10 + 0.20 × 1.00 = $0.30` per query vs $1.00 baseline — a 70% cut.
- If `P(escalate)` is *high* (the cheap model fails often), the cascade can cost *more* than the baseline — you pay for the cheap attempt *plus* the frontier call on most queries. `0.10 + 0.80 × 1.00 = $0.90`... and you also paid the $0.10 cheap attempt on the 80% that escalated, so the cascade overhead bit. **A cascade only pays when the cheap model succeeds often enough that the wasted cheap attempts cost less than the routing classifier would have.**

So the lever is **the cheap model's success rate on your workload** and the **quality of the verifier**. A verifier that's too lenient passes bad cheap answers (quality drops); too strict escalates everything (cost rises to the baseline plus overhead). The verifier threshold is a cost-vs-quality knob, swept like the semantic-cache threshold (Lecture 1 §3). The README stretch goal is exactly this: a calibrated cascade where you tune the verifier threshold and plot the cost-vs-quality frontier.

Let's put real numbers on the break-even, because "a cascade only pays when the cheap model succeeds often enough" deserves a threshold. With `cost_cheap = $0.10` and `cost_frontier = $1.00`, the cascade's expected cost is `0.10 + P(escalate) × 1.00`, and the all-frontier baseline is `1.00`. Set them equal to find the escalation rate where the cascade stops paying:

```
0.10 + P(escalate) × 1.00  =  1.00
P(escalate)                =  0.90
```

So with a cheap model that's 10× cheaper, the cascade beats the baseline as long as it escalates **less than 90%** of the time — a very forgiving bar. Even a mediocre cheap model that fails half its queries gives `0.10 + 0.50 × 1.00 = $0.60` per query, a 40% saving. The cascade only loses when the cheap model is nearly useless (escalates >90%), at which point you're paying the cheap-attempt tax on almost everything. **This is why cascades are attractive: the break-even escalation rate is high, so even an imperfect cheap model usually pays.** The catch is the verifier cost (§3) — add `cost_verify` to the cheap side and the break-even tightens; a verifier that costs as much as the frontier call erases the advantage. The practical lesson: keep the cheap model good enough to clear, say, a 50% success rate, and keep the verifier near-free, and the cascade is a reliable 40–70% cut.

**Router vs cascade — when to use which:**

- **Router** when you can classify difficulty cheaply and accurately *before* generating (you have a good classifier, or a clear heuristic). One model call per query; no wasted cheap attempts.
- **Cascade** when difficulty is hard to predict up front but easy to *verify* after the fact (you can cheaply check whether an answer is good). No classifier needed; pays the cheap-attempt overhead on escalations.
- **Both** in practice: a router for the obviously-easy and obviously-hard queries, a cascade for the ambiguous middle. The capstone's routing layer is some blend of these, tuned to its workload.

---

## 3. The verifier: how do you know the cheap answer is good?

The cascade lives or dies on the verifier, so it's worth naming the options, cheapest first:

- **Format / schema check.** If the answer must be valid JSON, a date, a number in a range — check it programmatically. Free, deterministic, catches the cheap model producing malformed output. Use it always when the output has structure.
- **Confidence heuristic.** Self-consistency (sample the cheap model 3×; if the answers agree, trust it; if they diverge, escalate), or a logprob-based confidence if the model exposes it. Cheap, no extra model.
- **LLM-as-judge.** A model scores whether the cheap answer is correct/complete/grounded (the calibrated-judge pattern from week 12). More expensive (it's another model call) and you must calibrate it, but it's the most general verifier. Use a *cheap* judge so the verification doesn't eat the savings — a Haiku-class judge verifying a 7B's answer, escalating to Opus only on a fail.

The verifier cost is part of the cascade's expected cost: `cost_cheap + cost_verify + P(escalate) × cost_frontier`. A verifier that costs as much as the frontier model defeats the purpose — so verifiers are deliberately cheap (a format check, a self-consistency vote, a small-model judge). The art is a verifier that's cheap enough to be nearly free but accurate enough to keep `P(escalate)` honest (it escalates the genuinely-failed answers, not the merely-different-from-Opus ones).

> **The discipline:** the verifier must be much cheaper than the escalation it gates, and calibrated against a labeled set so its pass/fail tracks actual answer quality. An uncalibrated verifier turns the cascade into either a quality leak (too lenient) or a cost leak (too strict).

---

## 4. Batched inference: trade latency for a discount

Some traffic isn't latency-sensitive — overnight document processing, bulk classification, dataset generation, eval runs. For that traffic, **batching** is a large, free-on-quality cost lever, and there are two distinct kinds (don't conflate them):

**Offline batch APIs (Anthropic Batch, OpenAI Batch).** Submit a batch of requests; get results within (up to) 24 hours; pay **50% of the standard price**. The vendor runs your requests when its capacity is idle, and passes the efficiency back as a discount. The trade is pure latency: you wait up to a day, you pay half. For any traffic that can tolerate that latency, it's a 2× cost cut with zero quality cost.

```python
batch = client.messages.batches.create(requests=[
    Request(custom_id=f"q-{i}", params=MessageCreateParamsNonStreaming(
        model="claude-sonnet-4-6", max_tokens=512,
        messages=[{"role": "user", "content": q}]))
    for i, q in enumerate(queries)
])
# poll batch.processing_status until "ended"; results at 50% price
```

**Online continuous batching (vLLM, week 19).** This is the *self-hosted* batching — the server fills a batch from concurrent live requests every decode step. It's not a discount you opt into; it's how your self-hosted tier achieves its low cost-per-token in the first place. It serves *latency-sensitive* traffic (the requests are live), and its "discount" is the throughput multiplier from week 19.

The decision: **how much of your traffic can tolerate batch latency?** The latency-sensitive slice (a user waiting for a chat response) must go through the online path (vLLM continuous batching, or a real-time vendor call). The latency-tolerant slice (overnight bulk work) should go through the offline batch API for the 50% cut. A cost-engineered system *segments* its traffic by latency tolerance and batches everything it can. The capstone's eval runs and any bulk re-processing are prime batch candidates; its interactive query path is not.

> **Rule of thumb:** if a chunk of work doesn't need an answer in the next few seconds, run it through the Batch API for 50% off. The only cost is patience, and a lot of LLM work (evals, bulk processing, offline generation) is patient.

### 4b. Segmenting traffic by latency tolerance

The practical move is to *segment* your traffic into latency tiers and route each tier to the right execution path. A worked segmentation for a typical product:

```
traffic segment            latency need      execution path           cost effect
------------------------   ---------------   ----------------------   -----------
interactive chat           seconds           online (vLLM / vendor)   full price, low latency
autocomplete / suggestions sub-second        online, cheap model      cheap, must be fast
nightly eval suite         hours             offline Batch API        50% off
bulk re-embedding/re-tag   hours             offline Batch API        50% off
report generation (queued) minutes           online, but can queue    full price or batch
```

The interactive segments *must* be online — a user staring at a spinner won't wait 24 hours, so the Batch API is off the table for them and their cost lever is routing/caching, not batching. The patient segments (evals, bulk jobs) have no human waiting, so they go to the Batch API for the flat 50% cut with zero quality cost — the cheapest lever available, requiring nothing but the willingness to wait. The middle (queued reports, "we'll email you when it's ready") is a judgment call: if "minutes" is acceptable, keep it online; if "by tomorrow" is acceptable, batch it.

The mistake to avoid is treating all traffic as one latency tier. A team that runs its nightly 10,000-question eval suite through the real-time API is paying double for work that has no deadline — the eval results aren't needed until morning. Segment first, then batch everything that can wait. **The Batch API discount is the rare cost lever with no quality trade at all** — the answers are identical, you just get them later — which makes it the first thing to apply to any traffic you can prove is patient.

## 4c. The worked end-to-end pipeline

Let's trace the full stacked pipeline (Lecture 1 §6 plus this lecture's routing/cascade) on a few example queries, so the short-circuiting and the per-query cost are concrete:

```
query: "what's your refund window?"   (asked earlier in a paraphrase)
  → exact-match cache: miss
  → semantic cache: HIT (0.96 similarity to "how long for a refund?")
  → return cached answer.   cost: $0.00   path: cache-hit

query: "classify this ticket: 'my card was declined'"
  → exact-match: miss
  → semantic cache: miss (novel ticket text)
  → router: classifier says EASY (short, templated task)
  → local-7b answers.   cost: $0.0003   path: local

query: "compare our Q3 strategy doc against the competitor analysis and flag risks"
  → exact-match: miss
  → semantic cache: miss
  → router: classifier says HARD (multi-doc reasoning)
  → prompt caching on the shared instruction prefix (cache-read)
  → frontier (Opus) answers.   cost: $0.04   path: frontier

query: "summarize this 8-page contract"
  → caches: miss
  → router: HARD
  → cascade: local-7b drafts a summary
  → verifier (cheap judge): the draft missed the liability clause -> FAIL
  → escalate to frontier.   cost: $0.0003 + $0.02 = $0.0203   path: escalated
```

Four queries, four different paths, four wildly different costs — from $0.00 (cache) to $0.04 (frontier). The naive pipeline would have sent all four to the frontier model at ~$0.04 each ($0.16 total); the engineered pipeline cost $0.0206 total — an 87% cut on this tiny sample, with each query handled at the *cheapest path that gets it right*. The cache-hit query cost nothing; the easy classify cost a fraction of a cent on the local tier; only the genuinely-hard multi-doc query paid full frontier price; and the cascade caught a cheap-model failure and escalated *just that one*. Scale this over 500 queries with realistic difficulty and duplicate mixes and you get the headline 88% reduction — not from one lever, but from each query finding its cheapest correct path through the stack.

---

## 4d. Building the classifier in practice

The router's classifier is a real engineering decision, so let's be concrete about the build path, cheapest-effort first. Each option trades build effort against routing accuracy:

**1. The heuristic (an afternoon).** A handful of rules over cheap signals: query length, keyword presence ("explain", "compare", "why", "step by step", "analyze" → hard; "what is", "list", "yes/no" → easy), number of retrieved documents, presence of code. Transparent, free to run, no training. It's the right *first* classifier — ship it, measure its quality delta, and only build something heavier if the heuristic's false-easy rate is too high.

```python
HARD_KEYWORDS = {"explain", "compare", "analyze", "why", "step by step", "evaluate"}

def is_hard_heuristic(query: str, n_docs: int = 0) -> bool:
    q = query.lower()
    if len(q.split()) > 40:                      return True   # long = probably complex
    if any(k in q for k in HARD_KEYWORDS):       return True   # reasoning keywords
    if n_docs > 3:                               return True   # multi-doc synthesis
    return False
```

**2. The few-shot classifier (a day).** Ask a *cheap* model to label the query, with a few examples in the prompt. More flexible than keywords (it generalizes), costs one small classification call per query (which the routing saves many times over). Use a Haiku-class model so the classification is nearly free relative to the frontier call it might avoid.

**3. The trained classifier (a week).** Embed the query (BGE, the model you know) and train a logistic regression or a small MLP on a labeled easy/hard set — the RouteLLM approach. Most accurate, requires a labeled training set and an eval, and gives you a tunable decision threshold (which is where you apply the §1b conservative bias — set the threshold to minimize false-easy).

The build-effort ladder mirrors the rest of the course: start with the cheapest thing that might work (the heuristic), *measure it* against the labeled set, and climb only when the measurement says the simpler classifier is leaking too much quality or money. A team that builds a trained router before measuring whether a keyword heuristic suffices has over-engineered the classifier — the same mistake as reaching for a framework before the loop needs one.

## 4e. Routing anti-patterns

Four ways routing goes wrong in practice, so you can spot them:

- **Routing on cost alone, ignoring quality.** The classic — push everything to the cheap tier because the dashboard looks great, and discover (from users, not the dashboard) that the hard-question quality cratered. The fix is the quality delta on the routed-to-cheap set, measured every time.
- **A classifier more expensive than the savings.** If your "is this hard?" classifier is itself a frontier-model call, you've spent the frontier cost to *decide* whether to spend the frontier cost — net loss. The classifier must be much cheaper than the call it's gating (a heuristic, or a small-model few-shot).
- **Static routing on drifting traffic.** A heuristic tuned on last quarter's queries decays as the traffic mix shifts. Routing decisions need periodic re-measurement against fresh labeled data — the difficulty distribution of "easy" and "hard" is not static.
- **No fallback path.** The cheap tier (your vLLM cluster) goes down, and the router has nowhere to send the easy traffic. Routing (by choice) and fallback (by failure) are different mechanisms (week 19 §3b) and you need *both* — the router decides the intended tier, the fallback catches the case where the intended tier is unavailable. A router without a fallback is a single point of failure.

Each of these is the same lesson wearing a different hat: routing is a measured, monitored, fallback-protected decision, not a fire-and-forget config. The capstone's routing layer has all four guards because all four failures show up in production.

## 5. Putting it together: the measurement

Lecture 1 §6 showed the stacked pipeline. The thing that makes it *engineering* and not *hope* is the measurement, and it has four components the capstone requires:

1. **Total cost** on the fixed workload, vs the naive all-frontier baseline. The headline number ("88% reduction").
2. **Per-route cost** — how much went to cache hits ($0), the cheap tier, the frontier tier, escalations. This is where you see *which* lever did the work, and where the remaining cost lives.
3. **Cache-hit rate over time** — does it climb as the cache warms? A cold cache hits 0%; a warm one on a repetitive workload hits 40%+. Plotting it over the workload shows the cache earning its keep.
4. **The quality delta vs baseline** — the non-negotiable. Score the engineered pipeline's answers against the baseline's (LLM-judge or accuracy on the labeled set). The cost saving is only real if this delta is inside tolerance. **An 88% cost cut with a -0.15 quality delta is not an 88% saving — it's a quality regression with a cost number attached.**

```
COST REDUCTION REPORT (500-query workload)
  baseline (all-frontier)   : $7.20
  engineered pipeline       : $0.84     -> 88% reduction
  breakdown:
    cache hits   (41%)      : $0.00
    local-7b     (44%)      : $0.31
    frontier     (8%)       : $0.42
    escalations  (7%)       : $0.11
  cache-hit rate over time  : 0% -> 41% (warmed over the run)
  quality delta vs baseline : -0.01  (tolerance: -0.03)   ✓ WITHIN TOLERANCE
```

That report is the syllabus deliverable, and it's the capstone's cost report in miniature. The discipline is the same as every measurement week in this course: a cost claim is only a claim if it comes with the method and the quality delta. "We cut cost 88%" is a headline; "we cut cost 88% with a -0.01 quality delta on a 500-query labeled workload, with 41% from caching and the rest from routing" is an engineering result you can defend at a review.

---

## 6. The hard part: keeping quality honest under cost pressure

A closing warning, because it's the way cost engineering goes wrong in production. Every lever this week has a setting that saves *more* money at *some* quality cost: a looser cache threshold, an aggressive compression ratio, a router that sends more to the cheap tier, a lenient verifier. There is always pressure — from a budget, from a dashboard, from a well-meaning optimization — to push those settings toward "cheaper," and each individual push looks small. Stacked, they silently erode quality until the product is measurably worse and nobody noticed because each step was a "small saving."

The defense is the labeled set and the quality delta. **Every cost change is re-measured against the same labeled workload, and a change that pushes the quality delta past tolerance is rejected no matter how much it saves.** This is why the workload is fixed and labeled (the resources note), why the memo reports the quality delta next to the cost, and why the rubric fails a cost reduction with an unacceptable quality regression. Cost engineering is not "make it cheaper"; it's "make it as cheap as possible *subject to* quality holding" — a constrained optimization, and the constraint is the part that's easy to drop under pressure and expensive to drop in production.

> **The discipline, one more time:** cost down, quality held, both measured, on the same labeled set, every time. A saving you can't show preserved quality is a regression you haven't noticed yet.

### 6b. A note on production monitoring vs offline measurement

The labeled-set measurement above is the *offline* gate — you run it before shipping a cost change. But production traffic drifts, and a routing/caching config that held quality on last month's labeled set can decay as the query mix shifts. So the offline gate has a production counterpart: **monitor the cost-engineering metrics live, and alarm when they move.** Three signals to watch:

- **Cache-hit rate trending down.** If your semantic cache's hit rate drops, either traffic got more diverse (less cacheable) or your invalidation is purging too aggressively. Either way, your cost is rising and you want to know before the bill does.
- **Escalation rate trending up.** If the cascade's `P(escalate)` climbs, the cheap model is failing more — maybe the traffic got harder, maybe the cheap model regressed. Rising escalation means rising cost (you're hitting the frontier model more) *and* possibly a quality problem (the cheap model is struggling).
- **Frontier-route fraction trending up.** If the router is sending more traffic to the expensive tier, your classifier's "hard" rate is climbing — again, a traffic-mix shift you want to catch and re-tune for.

These connect directly to week 18's observability work and week 24's on-call runbook: a *cost spike* is one of the named incident classes, and these three metrics are how you detect and diagnose it. The offline labeled-set measurement proves a config is *good before shipping*; the production monitoring proves it *stays good after shipping*. Both are required — a cost layer with offline gates but no production monitoring will silently decay, and the first you'll hear of it is an unexpected invoice. The capstone's observability dashboard includes these cost-engineering metrics for exactly this reason; this week builds the offline gate, week 24 wires the live alarm.

---

## 6c. The cost-quality frontier

A useful way to think about the whole week's work is as finding the right point on a **cost-quality frontier**. Plot every configuration of your pipeline as a point: x-axis is cost (lower is better), y-axis is quality (higher is better). The naive all-frontier pipeline sits at the top-right: maximum quality, maximum cost. The all-cheap pipeline sits at the bottom-left: minimum cost, degraded quality.

```
quality
   │  ● all-frontier (max quality, max cost)
   │   ╲
   │    ●  router @ conservative threshold
   │     ╲
   │      ● cascade @ calibrated verifier   ← the frontier: best quality per dollar
   │       ╲
   │        ● router @ aggressive (quality starting to drop)
   │         ╲
   │          ● all-cheap (min cost, degraded)
   └──────────────────────────────── cost
        cheap ................... expensive
```

The configurations that matter are the ones on the **frontier** — the curve of "best quality achievable at each cost." A configuration *inside* the frontier (worse quality *and* higher cost than some other config) is strictly dominated; you'd never choose it. Your job is (a) to find the frontier by sweeping your thresholds (cache threshold, classifier threshold, verifier threshold, compression ratio), and (b) to choose the point on it that meets your quality tolerance at the lowest cost. "Choose the cheapest config whose quality delta is within tolerance" is exactly "find the leftmost frontier point above the quality floor."

This reframes every threshold sweep in this week as *mapping the frontier*. The semantic-cache sweep (Lecture 1 §3) traces part of it; the classifier-bias tuning (§1b) traces part; the verifier calibration (§3) traces part. Stack them and you've mapped the achievable cost-quality trade-offs for your workload, and the memo's job is to point at the chosen frontier point and justify it: "we operate at the cascade-with-calibrated-verifier point — 88% cheaper than all-frontier, with a -0.01 quality delta inside our -0.03 tolerance." That's a defensible engineering choice; "we made it cheaper" is not.

## 7. Recap

You should now be able to:

- Build a **model router** with a classifier (heuristic, few-shot, or trained) that sends easy queries to a cheap model and hard ones to the frontier model, and measure its value as `cost_saved × quality_held`.
- Build a **cascade** that tries the cheap model and escalates on a verifier failure, do the **expected-cost math** (`cost_cheap + P(escalate) × cost_frontier`), and say when a cascade beats a router and when it costs more than the baseline.
- Choose a **verifier** (format check, self-consistency, cheap LLM-judge) that's much cheaper than the escalation it gates and calibrated to track real quality.
- Decide when **batched inference** applies — segment traffic by latency tolerance, route the patient slice through the Batch API for 50% off, keep the latency-sensitive slice on the online (continuous-batching / real-time) path.
- **Measure** the engineered pipeline end to end — total cost vs baseline, per-route breakdown, cache-hit rate over time, and the quality delta — and write the cost-reduction memo that reports the saving *and* the preserved quality.
- Keep **quality honest under cost pressure** — every cost change re-measured against the same labeled set, rejected if it pushes the quality delta past tolerance.

Carry these one-line takeaways into the exercises:

- A router's value is `cost_saved × quality_held`; report both, always.
- Classifier errors are asymmetric — false-easy costs quality (worse), false-hard costs money (better); bias conservative.
- A cascade pays when the cheap model clears often enough; the break-even escalation rate is forgiving (~90% with a 10× gap).
- The verifier must be much cheaper than the escalation it gates, and calibrated to real quality.
- Segment traffic by latency tolerance; batch everything patient for a free 50% cut.
- Measure total cost, per-route cost, cache-hit rate over time, and the quality delta — the last is non-negotiable.
- Find the cost-quality frontier by sweeping thresholds; operate at the cheapest point above your quality floor.
- Gate offline on a labeled set; monitor live for drift (hit rate down, escalation up, frontier fraction up).

Next week is **Capstone Sprint A** — you start building the production agentic research assistant for real, and this cost-engineered routing layer becomes its serving-and-cost layer. Push your `crunchroute` harness; the capstone's cost-tracked routing starts from it.

---

## References

- *FrugalGPT* — Chen et al., 2023 (cascades, cost reduction with quality preservation): <https://arxiv.org/abs/2305.05176>
- *RouteLLM* (router-as-classifier framework): <https://github.com/lm-sys/RouteLLM>
- *Anthropic Message Batches* (50% async discount): <https://platform.claude.com/docs/en/build-with-claude/batch-processing>
- *LiteLLM routing* (executing the routing decision): <https://docs.litellm.ai/docs/routing>
- *Anthropic pricing* (the per-model prices the cost math uses): <https://platform.claude.com/docs/en/pricing>
