# Lecture 2 — The Landscape, the Cards, and the Licenses

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can place any 2026 model on the open/closed and capability/cost map, read a model card and extract the six load-bearing facts, separate a leaderboard rank from product fit, and read a model's license well enough to tell a founder "yes we can ship this" or "no, here's the clause that stops us."

Lecture 1 was the function. This lecture is the catalog the function ships in — the messy, fast-moving, commercially-loaded reality of *which* models exist, what their documentation does and doesn't tell you, and what their licenses actually permit. This is the part of the field that rotates every cohort. The *skill* — reading a card, reading a license, picking with reasons — does not rotate. That is what we teach.

The sentence to carry out of this lecture:

> **Embedding the right model in your system is a procurement decision, not a vibe. A model has a capability tier, a cost curve, a context budget, a latency profile, and a license — and you do not get to skip any of them because a leaderboard said it was "the best."**

---

## 1. The two axes that organize everything

There are dozens of models. You will never memorize them, and you should not try; they rotate. Instead, place any model on two axes and most decisions fall out.

### Axis 1 — Open-weights vs closed-weights

- **Closed-weights (frontier vendor APIs).** The weights are not downloadable; you call an API. In 2026 the frontier tiers are **Claude 4 class** (Anthropic — `claude-opus-4-8` most capable, `claude-sonnet-4-6` balanced, `claude-haiku-4-5` fast/cheap), **GPT-5 class** (OpenAI), and **Gemini 2.5 class** (Google). You rent capability and operational simplicity; you accept vendor dependency, per-token pricing, and data-handling terms you must read.
- **Open-weights (downloadable).** The weights ship; you can run them yourself on Ollama, vLLM, llama.cpp, etc. In 2026 the major families are **Llama 4** (Meta), **Qwen 3** (Alibaba), **Mistral** (Mistral AI), **Gemma 3** (Google), and **DeepSeek**. You gain control, on-prem capability, no per-token vendor fee (you pay for compute), and freedom from a vendor's roadmap — *if* the license lets you (see §5).

This axis is **not** "good vs cheap" or "smart vs dumb." It is **control vs convenience**. C23's stance, stated in the charter: ~80% of labs run on open-weights/local so you never become a "vendor-locked graduate" who can't operate without someone's credentials; the frontier APIs are taught as the scale path and the hard-capability path, never as the only path.

### Axis 2 — Capability vs cost

Within either column, models span a capability/cost range. Roughly:

- **Frontier tier** — highest capability, highest price, biggest context. `claude-opus-4-8` (and peers). Use when correctness on a hard task is worth the cost.
- **Balanced tier** — strong capability, mid price, the production workhorse. `claude-sonnet-4-6`, mid-size open models (Qwen 3 mid, Llama 4 mid). Most production traffic lives here.
- **Fast/cheap tier** — good-enough capability, lowest price/latency. `claude-haiku-4-5`, small open models (`qwen2.5:7b`, `llama3.2:3b`, `gemma3:4b`). Use for classification, extraction, routing, and high-volume simple tasks.

The week-21 routing lab is built entirely on this axis: send easy queries to a 7B, hard queries to a frontier model, and pay the frontier price only when you must. You can't route intelligently until you can place a model on this axis, which is why we start here.

```
                 high capability
                       │
   claude-opus-4-8 ●   │   ● GPT-5 / Gemini 2.5
       Llama-4-big ●   │   ● claude-sonnet-4-6
                       │   ● Qwen-3-mid
   ────── open ────────┼──────── closed ───────  (control vs convenience)
                       │
       qwen2.5:7b  ●   │   ● claude-haiku-4-5
       gemma3:4b   ●   │
                       │
                 low cost / low latency
```

The exact dots move every quarter. The *axes* don't. When you onboard a new model, your first two questions are always: which column (can I self-host it / do I want to?), and which row (capability vs cost for *this* job).

---

## 2. The 2026 open-weights families, briefly

You need a working map, not encyclopedic recall. Here is the one-line characterization of each major open family as of 2026 — enough to know what to reach for and what to read more about.

- **Llama 4 (Meta).** The most widely-deployed open family; broad ecosystem support, every tool speaks it. Source-available **community license** with an acceptable-use policy and a 700M-monthly-active-user clause (§5). Strong general capability across sizes.
- **Qwen 3 (Alibaba).** Excellent capability-per-parameter, strong multilingual and coding, wide size range. License varies *by variant* — many are Apache-2.0, some are not — so you check the specific checkpoint, not "Qwen."
- **Mistral (Mistral AI).** Lean, efficient, mostly **Apache-2.0** on the open releases — the cleanest license story of the bunch for commercial use. Some specialized models are under a separate research/commercial license; check.
- **Gemma 3 (Google).** Capable small-to-mid models, permissive *commercial* terms but under Google's **Gemma Terms of Use** with an enforceable prohibited-use policy (not pure OSI-open). Good Apple-Silicon / on-device story.
- **DeepSeek.** Strong reasoning-oriented open models, competitive at the high end of open capability, generally permissive licensing — verify per release.

Two meta-points that outlast the specifics:

1. **"Open-weights" is a spectrum of licenses, not a binary.** Mistral-Apache and Llama-community are both "open-weights" and worlds apart legally (§5). Never say "it's open" and stop there.
2. **Per-parameter capability keeps climbing.** A 2026 7B model is dramatically more capable than a 2023 7B. "Small model" is not "weak model." This is why the routing strategy (week 21) works: a 7B handles a surprising fraction of real traffic.

---

## 3. Reading a model card without falling for the benchmark

A **model card** is the document that ships with a model. It is the equivalent of a datasheet for a chip. Beginners skim it for the benchmark table and stop. Engineers read it for the six load-bearing facts that decide whether the model fits the job. Here they are, in priority order.

### The six facts to extract from every card

1. **License.** First, always. Decides whether you can legally ship at all (§5). If the license forbids your use case, nothing else on the card matters.
2. **Context window.** The token ceiling (Stage 2 from Lecture 1). Decides whether your prompts + retrieved context + output fit. A model that's brilliant but caps at 8k tokens is useless for a 50-page-document task.
3. **Training cutoff date.** The model knows nothing past this date (frozen MLP weights, Lecture 1 §3). Decides how much you must supply via retrieval. A model with a 2024 cutoff "doesn't know" 2026 events — by design, not by failure.
4. **Intended use / out-of-scope use.** Cards state what the model is built for and explicitly *not* built for. The out-of-scope section is where vendors disclaim use cases they know it's bad at or that are legally fraught. Read it; it's a warning label.
5. **Modalities and sizes.** Text-only or vision-capable? Which parameter sizes ship? Which tokenizer? This decides whether it can do your task at all and how much it costs to run.
6. **Eval methodology and caveats.** *Not* the headline number — the footnotes. How was it evaluated? Few-shot or zero-shot? On which benchmarks? With what known contamination risks? The caveats tell you how much to trust the headline.

### Why the leaderboard rank is not product fit

Here is the lesson that takes new engineers a painful project to learn, delivered free:

> **A leaderboard measures average performance on a fixed benchmark distribution. Your product is one specific point, not the average. The model that tops the leaderboard may be worse on *your* task, and you will not know until you measure on *your* data.**

Concrete failure modes of benchmark-driven selection:

- **Benchmark ≠ your distribution.** A model that's #1 on a broad reasoning benchmark may underperform on your narrow customer-support classification, because your task isn't in the benchmark mix. Average rank is a weak predictor of point performance.
- **Contamination.** If a benchmark's test set leaked into training data, the score is inflated. Cards increasingly disclose decontamination efforts; absence of that disclosure is itself a signal.
- **The score ignores cost, latency, and license.** A leaderboard rank says nothing about whether the model fits your $0.002/call budget, your 2-second latency target, or your ability to ship under its license. Those are *your* constraints, and the leaderboard doesn't know them.
- **Human-preference Elo (LMArena) measures "which answer do people like more," not "which is correct for my extraction schema."** Useful for chat-style orientation, misleading for structured tasks.

The discipline this produces, and the one the whole course enforces via the rubric's *measurement* axis: **pick a small set of candidate models from the landscape, then measure them on your own held-out task with your own metric.** The leaderboard narrows the field from "everything" to "a few"; your measurement picks the one. That is exactly what `llmpick` (this week's mini-project) automates: query candidates, measure cost/latency/quality on the actual prompt, recommend with numbers.

---

## 4. The capability/cost/latency cross-section in practice

When you actually choose, you weigh four numbers against your task. Here's how each is sourced and what it costs you to get wrong.

| Dimension | Where you get it | Cost of getting it wrong |
|---|---|---|
| **Capability** | Measure on *your* held-out task; leaderboards only to shortlist. | Ship a model that fails your task quietly; users hit edge cases you never tested. |
| **Cost** | Token pricing × measured tokens-in/out per call (use the model's own tokenizer). | Budget blows up at scale; a 10× price difference is invisible at demo volume and fatal at production volume. |
| **Latency** | Measure TTFT and TPOT (Lecture 1 §4) under realistic prompt sizes. | Miss your latency SLO; "fast on a one-liner" ≠ "fast on a 10k-token prompt." |
| **License** | Read the license; §5. | Legal/commercial blocker discovered post-build; the worst time to find it. |

The mini-project forces all four into one decision. The reason the assignment gives `--budget` and `--latency-target` as inputs is that **a model recommendation is meaningless without a constraint to recommend against.** "Which model is best?" has no answer. "Which model meets a 2-second latency target and a $0.002 budget for *this* prompt class, at acceptable quality?" has a defensible, measurable answer. Frame every selection that way.

---

## 5. Licensing as an engineering constraint

This is the section that separates a graduate who ships from one who builds something they can't legally deploy. You are not a lawyer, and this is not legal advice — but a senior AI engineer reads a model license the way they read an API rate limit: as a hard constraint that shapes the design. Here is the working knowledge.

### The three buckets

1. **True open-source (OSI-approved): Apache-2.0, MIT.** You can use commercially, modify, redistribute, and build proprietary products on top, with minimal obligations (keep the license notice; Apache adds a patent grant and a "state your changes" nicety). Mistral's open releases and many Qwen variants live here. **This is the bucket you want** when you need maximum freedom. If a model is Apache-2.0, you can almost always ship.

2. **Source-available / community licenses: Llama Community License, Gemma Terms.** Downloadable, usable commercially *with conditions*. These are **not** OSI-open. The conditions that bite:
   - **Acceptable / prohibited use policies.** Enforceable lists of disallowed uses (e.g., certain harmful applications). You inherit these and must comply.
   - **Scale clauses.** Llama's community license requires a separate license from Meta if your product (or your affiliates') exceeds **700 million monthly active users** at the time of release. Irrelevant for a startup on day one; a real consideration for a hyperscaler or a wildly successful product. A founding engineer is expected to know this clause exists.
   - **Naming / derivative obligations.** Some licenses require derivative models to carry the family name (e.g., "Llama" in the name) and to pass along the license terms.
   - **Distribution obligations.** If you redistribute the weights or a fine-tune, you typically must include the license and may have attribution requirements.

3. **Research-only / non-commercial.** Some checkpoints (often the most cutting-edge research releases) are licensed for research only. **You cannot ship a product on these.** Discovering this after you've built a demo on one is a classic, avoidable, painful mistake. Check first.

### How to actually read a license (10 minutes, every time)

Open the LICENSE file and Ctrl-F for these terms. Each one maps to a question:

| Search for | The question it answers |
|---|---|
| `commercial` | Can I use this in a product I charge for? |
| `monthly active` / `MAU` / `revenue` | Is there a scale threshold that changes the terms? |
| `derivative` | If I fine-tune it, what governs the result? |
| `distribute` / `redistribution` | If I ship the weights (or a fine-tune), what must I include? |
| `prohibited` / `acceptable use` | What uses are explicitly forbidden? |
| `trademark` / `name` | Must my derivative carry the family name? |
| `as is` / `warranty` / `liability` | (Always disclaimed — but confirm there's nothing unusual.) |

Ten minutes of Ctrl-F up front saves a re-architecture later. The exercise this week (`exercise-01`) makes you do exactly this for three real licenses and write a one-line "can we ship?" verdict for each.

### The vendor-API analog

Closed frontier models have no weight license — you can't download them — but they have **terms of service** and **data-handling policies** that are the equivalent constraint. The questions shift but the discipline is identical:

- Can I use outputs commercially? (Usually yes, with conditions.)
- Is my input data used for training? (Vendor-dependent; enterprise tiers typically say no.)
- What are the usage/rate limits and the regional availability?
- What's the data residency and retention story?

Reading a vendor's ToS is the closed-weights version of reading an open license. Same skill, different document.

---

## 6. A worked selection, end to end

Let's pick a model for a concrete job, showing the reasoning a reviewer would expect.

**The job:** classify incoming support tickets into one of eight categories. High volume (100k/day). Latency target 1 second. Budget: pennies per thousand. Data is customer PII, must not be used for training.

**The reasoning:**

1. **Capability tier.** Eight-way classification of short text is *easy* — a fast/cheap tier model handles it. We do **not** need frontier capability. Shortlist: `claude-haiku-4-5` (hosted fast tier), `qwen2.5:7b` (local).
2. **Cost.** At 100k/day, even fractions of a cent compound. The local `qwen2.5:7b` has zero marginal token cost (you pay for the box it runs on). The hosted Haiku tier is cheap but non-zero. We measure both.
3. **Latency.** Both can hit sub-second on short input. We measure TTFT under realistic ticket length.
4. **License / data.** Local Qwen: check the specific variant's license for commercial use (`qwen2.5:7b` is permissive — confirm). Data never leaves our box — strong PII story. Hosted Haiku: confirm the enterprise data-handling terms say inputs aren't trained on.
5. **Decision.** For high-volume, PII-sensitive, easy classification, the local 7B is the lead candidate: zero marginal cost, data stays on-prem, sub-second latency, permissive license. The hosted Haiku tier is the fallback if local serving ops are too heavy for the team. **We measure both on a held-out set of 200 labeled tickets and pick the one with the better accuracy-per-cost.**

Notice what this reasoning is *not*: it is not "use the highest-ranked model." A frontier model on this job would be slower, ~10× more expensive, send PII to a vendor, and add no accuracy on an easy task. The leaderboard would have pointed you at exactly the wrong choice. The *constraints* — volume, latency, PII, budget — picked the model. That is the whole discipline.

This is the reasoning your challenge memo (`challenge-01`) and your `llmpick` tool must produce: a recommendation with a tier rationale, measured numbers, and the trade-off named.

---

## 7. Recap

You should now be able to:

- Place any model on the open/closed (control vs convenience) and capability/cost axes, and explain why those axes — not a single "best model" — drive selection.
- Give the one-line characterization of the major 2026 open families (Llama 4, Qwen 3, Mistral, Gemma 3, DeepSeek) and the frontier closed tiers (Claude 4 / GPT-5 / Gemini 2.5 class).
- Extract the six load-bearing facts from a model card (license, context window, cutoff, intended/out-of-scope use, modalities/sizes, eval caveats) and explain why a leaderboard rank is not product fit.
- Read a license in ten minutes with the Ctrl-F checklist, sort it into true-open / source-available / research-only, and find the clauses (commercial, MAU, derivative, distribute, prohibited use) that decide whether you can ship.
- Produce a defensible model selection that is driven by the task's constraints and backed by measured numbers — not by a leaderboard.

Next: the exercises put this into your hands — read three real cards, build the uniform client, and measure prefill vs decode yourself. Continue to [the exercises](../exercises/README.md).

---

## References

- *Llama 4 model card and license* (Meta) — community license, 700M-MAU clause: <https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md>
- *Qwen 3* (Alibaba) — per-variant licensing: <https://qwenlm.github.io/blog/qwen3/>
- *Gemma Terms of Use + Prohibited Use Policy* (Google): <https://ai.google.dev/gemma/terms>
- *Apache License 2.0* — the true-open baseline: <https://www.apache.org/licenses/LICENSE-2.0>
- *Anthropic Messages API — Models overview* (current frontier model IDs, context, pricing): <https://docs.claude.com/en/docs/about-claude/models/overview>
- *Artificial Analysis — models comparison* (cost/latency/quality cross-section): <https://artificialanalysis.ai/models>
- *LMArena leaderboard* (human-preference Elo, read skeptically): <https://lmarena.ai/leaderboard>
