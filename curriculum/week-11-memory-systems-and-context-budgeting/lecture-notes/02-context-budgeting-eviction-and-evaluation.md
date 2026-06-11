# Lecture 2 — Context Budgeting, Eviction, and Measuring Memory

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can budget a context window like a cache — allocate token slices to system / semantic / episodic / query, measure the budget, and enforce it — design an eviction policy (LRU, salience-weighted, TTL) for when the budget is full, explain the "lost in the middle" effect and why it makes budgeting a *quality* lever, and measure memory with a regression test that proves the agent remembers a turn-3 fact in turn 38.

Lecture 1 built the three tiers and summarized episodic memory. This lecture is about the *budget* that decides how much of each tier reaches the model, the *eviction* that runs when the budget is tight, and the *measurement* that proves the whole thing works.

> **Context is the most expensive cache on the planet. Spend it like one** — which means: have a budget, enforce it, and evict the lowest-value content when you're over. A memory system without a budget is a leak waiting to OOM the context window.

---

## Part 1 — The context window as a cache

The central metaphor of the week, made literal. A CPU cache is small, fast, and expensive; you can't fit everything, so you keep what you'll need soonest and evict the rest, and a *miss* (the thing you need isn't there) is costly. The LLM context window is exactly this:

- **Small and finite.** Even a 200k-token window is finite, and you pay (money + latency + the lost-in-the-middle tax) for every token you fill it with.
- **A "hit" is the fact being in context** when the model needs it. A "miss" is the fact having been evicted (or never retrieved), so the model can't use it and either hallucinates or says "I don't know."
- **You manage it with a budget and an eviction policy** — exactly like a cache.

The naive approach — append every turn to the prompt forever — is a cache with *no eviction*: it grows until it overflows (context-window error), and long before that it's slow, expensive, and degraded (lost in the middle, Part 3). The discipline of this lecture is to treat the window as the bounded cache it is.

> **The framing:** every token in the window is a cache line you chose to keep. Memory engineering is deciding *what to keep* and *what to evict* under a fixed budget — not "how much can I cram in."

---

## Part 2 — Budgeting: allocating the token slices

A context budget is a *deliberate allocation* of the window's tokens across the parts of the prompt. You decide, up front, roughly how many tokens each part gets, measure the actual usage, and enforce the cap. A sensible default allocation for a memory-equipped agent:

| Slice | Typical share | What it holds | Source tier |
|---|---|---|---|
| **System prompt** | fixed (e.g. 500 tok) | instructions, persona, tool schemas | — |
| **Semantic memory** | ~15–25% | retrieved durable facts about the user/world | semantic |
| **Episodic summary** | ~15–25% | the rolling summary of older turns | episodic |
| **Recent turns** | ~30–40% | the last N turns verbatim | episodic |
| **User query + headroom** | remainder | the current query + room for the response | — |

The exact percentages are a tuned choice (like chunk size in week 8 and the quant level in week 6), not a constant — you sweep them against the memory benchmark and pick what maximizes turn-38 recall within the budget. The *discipline* is what matters: you have a budget, you measure against it, and you don't silently blow past it.

```python
# Count tokens with the model's own tokenizer — never guess (the week-2 lesson).
import anthropic
client = anthropic.Anthropic()

def n_tokens(text: str, model="claude-sonnet-4-6") -> int:
    return client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    ).input_tokens


class ContextBudget:
    def __init__(self, total: int):
        self.total = total
        self.slices = {"system": 0, "semantic": 0, "episodic": 0,
                       "recent": 0, "query": 0}

    def fits(self) -> bool:
        return sum(self.slices.values()) <= self.total

    def used(self) -> int:
        return sum(self.slices.values())

    def over_by(self) -> int:
        return max(0, self.used() - self.total)
```

The load-bearing rule: **count tokens with the model's own tokenizer, not characters** — the same lesson as week 2 (tokens, not characters) and week 8 (chunk size in tokens). For Claude, that's `client.messages.count_tokens(...)`; for a local model, the model's tokenizer. A budget measured in characters is a budget that lies. (Never reach for `tiktoken` to count Claude tokens — it's a different tokenizer and undercounts.)

> **The discipline:** allocate a token budget per slice, *measure* with the right tokenizer, and *enforce* the cap. When the assembled prompt exceeds the budget, you don't truncate blindly — you *evict* by policy (Part 4). A budget you don't enforce is a wish.

---

## Part 3 — Lost in the middle: why budgeting is a quality lever, not just cost

Here's the finding that turns context budgeting from a *cost* concern into a *quality* concern. *Lost in the Middle* (Liu et al., 2023, arXiv 2307.03172) showed that long-context models recall information **far better when it's at the start or end of the context than in the middle** — accuracy traces a **U-shaped curve**: high at the edges, sagging in the middle. Bury the relevant fact in the middle of a long context and the model often *misses it even though it's right there*.

The implications for memory engineering are direct and important:

- **A full context is not a remembering context.** Stuffing 100k tokens of raw history into the window doesn't mean the model can *use* the fact in the middle of it — the lost-in-the-middle effect means the buried fact is functionally invisible. This is the mechanism behind "more context is not more memory" (Lecture 1's corollary).
- **A tight, well-budgeted context beats a full one — on accuracy, not just cost.** Retrieving the *one relevant fact* into a small context (where it's near an edge and uncrowded) produces a *better answer* than dumping everything and hoping the model finds it in the middle. Budgeting isn't penny-pinching; it's how you keep the answer findable.
- **Placement matters.** Put the most important content (the retrieved semantic fact, the user's actual query) at the *start or end* of the assembled prompt, not buried in the middle of a long episodic dump. This is a free quality lever — same tokens, better recall, just reordered.

So the budget does double duty: it controls cost *and* it keeps the model accurate by refusing to bury the answer. The stretch goal has you *measure* this yourself — plant a fact at the start, middle, and end of a long context and watch recall trace the U.

> **The lesson:** budgeting is a *quality* lever because of lost-in-the-middle. A smaller, well-placed context where the answer is near an edge beats a huge context where the answer is buried in the sagging middle. Spend the cache on the right tokens, in the right place.

---

## Part 4 — Eviction policies: what to drop when the budget is full

When the assembled prompt exceeds the budget, something has to go. *Which* something is an **eviction policy** — a design decision, not a default. Three policies, from simplest to smartest.

### 4.1 LRU — least recently used

Drop the *oldest* content first. For episodic memory this means: keep the recent turns, evict (or summarize) the oldest. It's the classic cache policy, it's simple, and it's a reasonable default — recency *is* a decent proxy for relevance in a conversation, because the current topic is usually recent.

```python
def evict_lru(turns: list[dict], budget_tokens: int, count) -> list[dict]:
    """Keep the most recent turns that fit; the rest get summarized/dropped."""
    kept, used = [], 0
    for turn in reversed(turns):                 # newest first
        t = count(turn["content"])
        if used + t > budget_tokens:
            break
        kept.append(turn)
        used += t
    return list(reversed(kept))                  # restore chronological order
```

**Where LRU fails:** the *important* fact is old. The user told you their project name in turn 3; it's the least-recently-used thing by turn 38, so pure LRU evicts it — exactly the fact the regression test checks for. LRU on the *episodic* tier is fine (old turns get summarized, not lost), but you must *also* have promoted that fact to *semantic* memory (Lecture 1) so it survives LRU eviction of the transcript. This is why the three tiers matter: LRU can evict the turn, but the fact lives on in semantic memory.

### 4.2 Salience-weighted — drop the least important

Smarter: score each memory by *importance* (salience), and evict the *lowest-salience* content regardless of age. Salience can come from a heuristic (facts about the user score high; pleasantries score low) or an LLM judgment ("rate how important this is to remember, 0–1"). Often combined with recency: `score = α·salience + β·recency`, evict the lowest.

```python
def evict_salience(memories: list[dict], budget_tokens: int, count,
                   alpha=0.7, beta=0.3) -> list[dict]:
    """Keep highest-scoring memories that fit. score = salience + recency."""
    n = len(memories)
    scored = []
    for i, m in enumerate(memories):
        recency = i / max(n - 1, 1)               # 0=oldest, 1=newest
        scored.append((alpha * m["salience"] + beta * recency, m))
    scored.sort(key=lambda s: -s[0])              # highest score first
    kept, used = [], 0
    for _, m in scored:
        t = count(m["content"])
        if used + t > budget_tokens:
            continue                              # skip this one, try the next
        kept.append(m); used += t
    return kept
```

**Why it beats LRU on memory:** salience keeps the turn-3 project name (high salience) even though it's old, and evicts a recent "thanks!" (low salience). On a memory benchmark, salience-weighted eviction recalls more turn-3-in-turn-38 facts than plain LRU — which you *measure* in the challenge. The cost is the salience score (a heuristic or an LLM call per memory).

### 4.3 TTL — time to live

Some memories *expire*. "The user is currently debugging the login flow" is true now and false next week. A **TTL** (time to live) attaches an expiry to a memory and evicts it when stale — independent of budget pressure. TTL is right for *transient* facts (current task, current mood) and wrong for *durable* ones (the project name, which has no expiry). A good system uses TTL on transient memories *and* salience on the rest.

> **Choosing eviction:**
> - **LRU** → simple, recency-as-relevance; fine for the episodic transcript (old turns summarize). Default.
> - **Salience-weighted** → keep the important old fact, drop the recent trivial one; the win on memory benchmarks.
> - **TTL** → expire transient facts that stop being true.
>
> And the meta-lesson: **eviction from the episodic transcript is survivable only because durable facts live in semantic memory.** The tiers and the eviction policy work together — that's the whole architecture.

---

## Part 5 — Measuring memory: the turn-38 regression test

Everything above is only as good as your ability to prove it works — and "the agent feels like it remembers" is a vibe, not a measurement (the C23 stance, the syllabus's "fail vibes-only submissions" rule). Memory has a *specific, measurable* property: does a fact stated early survive to be used late? The **memory regression test** measures exactly that.

The test, in one sentence: **run a multi-turn conversation that plants facts early and asks about them late, and score whether the agent recalls them — against a no-memory baseline.**

```python
def memory_regression_test(agent, plant_recall_pairs, distractor_turns):
    """plant_recall_pairs: [(turn_to_plant_fact, fact, later_turn_to_ask, question,
                             expected_answer)]. distractor_turns fill the gap so the
    fact must SURVIVE many turns to be recalled (the turn-38 test)."""
    transcript, results = [], []
    plants = {p[0]: p for p in plant_recall_pairs}
    asks = {p[2]: p for p in plant_recall_pairs}
    max_turn = max(p[2] for p in plant_recall_pairs)

    for turn in range(1, max_turn + 1):
        if turn in plants:
            _, fact, _, _, _ = plants[turn]
            transcript.append({"role": "user", "content": fact})   # plant it
            agent.observe(transcript[-1])                           # agent stores to tiers
        elif turn in asks:
            _, _, _, question, expected = asks[turn]
            answer = agent.respond(question)                        # recall under budget
            ok = expected.lower() in answer.lower()
            results.append((turn, question, expected, answer, ok))
        else:
            filler = distractor_turns[turn % len(distractor_turns)]  # noise
            transcript.append({"role": "user", "content": filler})
            agent.observe(transcript[-1])

    recalled = sum(1 for r in results if r[4])
    return {"recalled": recalled, "asked": len(results),
            "recall_rate": recalled / max(len(results), 1), "detail": results}
```

What the test exposes:

- **Memory vs no-memory baseline.** The no-memory agent (just the recent N turns, no semantic tier) forgets the turn-3 fact as soon as it scrolls out of the recent window — it recalls maybe 2/20. The three-tier agent recalls 18/20 because the fact was promoted to semantic memory and retrieved at turn 38. *That delta is the number that justifies the whole memory system.*
- **Which tier is doing the work.** If recall drops when you disable the semantic tier, semantic memory is carrying it. If it drops when you shrink the recent window, episodic is. The test localizes the contribution.
- **Eviction policy comparison.** Run the same test under LRU vs salience-weighted eviction (Part 4); salience recalls more of the old-but-important facts. The test is how you *choose* the eviction policy with a number.

> **The "it remembered in turn 38" promise, made measurable:**
> ```
> turn 03: user says "my project is called Helios"  -> stored (semantic)
> ... 34 distractor turns ...
> turn 38: "what's my project called?"  -> "Helios" ✓  (recalled from semantic)
> three-tier: 18/20 facts recalled   vs   no-memory baseline: 2/20
> ```
> The recall delta — 18 vs 2 — is the measurement. Not "it feels like it remembers"; a number, against a baseline, that says exactly how much memory bought you.

---

## Part 6 — Putting it together: the memory loop

The week converges on a loop the mini-project implements. On each turn:

1. **Observe.** Append the turn to episodic; extract durable facts → upsert to semantic (`crunchstore`); log any tool call → procedural.
2. **Summarize.** Fold older episodic turns into the rolling summary (Lecture 1 §3) when the recent window overflows its slice.
3. **Assemble under budget.** Retrieve relevant semantic facts (by similarity to the query), take the episodic summary + recent turns, and assemble the prompt — *sized to the budget*, with the most important content placed at the *edges* (Part 3), and *evicted by policy* (Part 4) if over.
4. **Respond.** Send the budgeted prompt; the answer reflects what survived into context.
5. **Measure.** The regression test (Part 5) verifies the turn-3 fact reaches the model in turn 38, against the no-memory baseline.

Do that and you've done the engineering the week teaches: not "I gave the agent memory," but "I built three tiers, budgeted the context like the cache it is, chose an eviction policy with a measured recall comparison, and proved the agent remembers the user's project name 35 turns later — 18/20 versus a 2/20 baseline." The fact survived the budget, and you can prove it.

---

## Part 7 — Prompt caching and the budget: a real tension

The budget so far has been about *what fits*. There's a second axis that changes how you *order* the budget: **Anthropic prompt caching**. A stable prefix of the prompt — `tools` → `system` → early `messages`, in render order — is cached, and on a repeat request the cached span bills at ~0.1× input cost instead of full price. For a memory agent that fires a request every turn over the same system prompt and the same durable facts, that's most of your input cost served from cache. But the cache is a **prefix match**: any byte change anywhere in the prefix invalidates everything after it. That single rule drives the budget's *layout*.

The implication for budget *design* is concrete: **order the slices by volatility, stablest first.** The system prompt and the durable semantic facts (the user's project name, their tier — things that don't change turn to turn) go at the *front*, behind a `cache_control` breakpoint, so they cache and stay cached. The volatile content — the new user query, the freshly-rolled episodic summary, a just-retrieved transient fact — goes *after* the breakpoint, where it's expected to change and invalidating it costs nothing downstream (there's nothing cached after it to lose).

```python
# Stable prefix first (caches), volatile suffix last (cheap to invalidate).
resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": system_prompt + "\n\n[Durable facts]\n" + durable_semantic_facts,
        "cache_control": {"type": "ephemeral"},   # everything up to here caches
    }],
    messages=[{"role": "user", "content":
        f"[Conversation so far]\n{episodic_summary}\n\n"   # volatile — after the breakpoint
        + "\n".join(f"{t['role']}: {t['content']}" for t in recent_turns)
        + f"\n\n{query}"}],                                 # most volatile — last
)
# Verify it actually cached — zero across repeats means a silent invalidator upstream.
assert resp.usage.cache_read_input_tokens > 0 or resp.usage.cache_creation_input_tokens > 0
```

Now the honest tension, because it's real and you should not pretend it away. **This ordering fights the lost-in-the-middle advice from Part 3.** Caching wants the *stable* content first; lost-in-the-middle wants the *most important* content at the edges (start *or* end). Those agree on one thing — durable facts near the front is good for both — but they disagree on the *query*. Caching is happy with the query last (it's volatile, it belongs after the breakpoint, and "last" is fine for the cache). Lost-in-the-middle is *also* happy with the query last (the end is an edge, recall is high there). So the query placement is not actually in conflict — both want it at the end. The conflict bites the *episodic summary*: it's volatile (so caching wants it late, after the breakpoint) but it can be long, and a long volatile block sitting between the cached prefix and the final query pushes the *durable facts toward the middle*, into the sag. The resolution is to keep the volatile middle *small* — that's exactly what summarization (Lecture 1 §3) and the budget (Part 2) already do. A tight episodic summary is not just cheaper; it keeps the cached, durable facts close to the front edge instead of being shoved into the middle by a bloated transcript. Budgeting, caching, and lost-in-the-middle all point the same way once the volatile slice is kept lean.

> **The caching rule for memory budgets:** stable content first (system + durable semantic facts, behind a `cache_control` breakpoint), volatile content last (the rolling summary, recent turns, the query). It caches the expensive durable prefix and keeps the volatile slice — which you keep *small* — from burying the durable facts in the lost-in-the-middle sag. Don't interpolate a timestamp or a per-request ID into the prefix, or you invalidate the cache every turn and pay full price for the whole memory system.

---

## Part 8 — Salience scoring, in depth

Part 4.2 used a `salience` field as if it fell from the sky. It doesn't — *you* compute it, and how you compute it is a real engineering choice with a real cost. There are two families.

**Heuristics — cheap, deterministic, no LLM call.** Score from cheap signals you already have:

- **Fact type.** A `(user.project, name, ...)` fact scores high; a pleasantry scores ~0. The `(entity, attribute)` key from Lecture 1 §5 already tells you the type — `name`, `preference`, `decision` are durable and high-salience; `mood`, `current_task` are transient and low.
- **Explicit user emphasis.** "*Important*: always deploy on Fridays," "remember that..." — the user is telling you it matters. Pattern-match the emphasis markers and bump the score.
- **Frequency of reference.** A fact the conversation keeps returning to is load-bearing; count how often it's been retrieved or restated and let that raise salience.

**An LLM judge — accurate, but it costs a call.** Ask `claude-sonnet-4-6` to rate "how important is this to remember long-term, 0–1." It catches importance the heuristics miss (a one-line fact that's pivotal but matches no pattern). The catch is cost: an LLM call *per memory*. If you re-score every memory on every eviction, you pay an LLM call per memory per eviction — quadratic-ish and absurd.

**The amortization that makes LLM-scoring affordable: score once, at write time.** Salience is a property of the *fact*, not of *this eviction*, so compute it when the fact is first extracted and **store it on the row** (alongside `value` and `embedding` from Lecture 1 §5). Eviction (Part 4) then *reads* the stored score — zero LLM calls at eviction time, which is the hot path. Re-score only on the rare event that the fact's *value* changes (an upsert), not on every budget check.

```python
def score_salience(fact: dict) -> float:
    """Compute salience ONCE, at write time. Cheap heuristics gate the LLM call."""
    s = {"name": 0.9, "preference": 0.7, "decision": 0.8,
         "mood": 0.1, "current_task": 0.2}.get(fact["attribute"], 0.5)
    if fact.get("user_emphasized"):          # "important:", "remember that..."
        s = max(s, 0.9)
    # Only spend an LLM call when the heuristic is genuinely uncertain (mid-band).
    if 0.4 <= s <= 0.6:
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=8,
            system="Rate how important this fact is to remember long-term. "
                   "Reply with one number 0.0-1.0, nothing else.",
            messages=[{"role": "user", "content":
                       f"{fact['entity']}.{fact['attribute']} = {fact['value']}"}],
        )
        s = float(next(b.text for b in msg.content if b.type == "text").strip())
    return s   # caller stores this on the row; eviction reads it, never recomputes it
```

The mid-band gate is the second amortization: heuristics resolve the obvious cases for free, and the LLM judge is spent only on the genuinely ambiguous middle. Most facts never hit the model.

> **The salience rule:** compute salience **once, at write time**, from cheap heuristics first (fact type, user emphasis, reference frequency) and an LLM judge only for the uncertain middle — then *store it on the row*. Eviction reads the stored score; it never re-scores. Salience is a property of the fact, so paying for it once per fact (not once per eviction) is the difference between an affordable memory system and one that makes an LLM call every time the budget gets tight.

---

## Part 9 — Measuring context efficiency, not just recall

The turn-38 regression test (Part 5) measures *recall*: did the fact reach the model and get used? That's necessary but not sufficient, because **an agent can have high recall and still waste the budget.** Retrieve 10 facts to answer a question that needed 1, and recall is perfect — the one fact you needed was in there — but nine slices of the budget were spent on context the answer never touched. On a 200k window that's tolerable; on a tight budget where lost-in-the-middle is biting, those nine wasted facts are actively *crowding out* the one that mattered and pushing it toward the sag. Recall alone won't catch this; you need a second metric.

Call it **context utilization**: of the tokens you spent on retrieved memory, how many did the answer actually *use*? You can't read the model's mind, so use a proxy — **did the answer cite or depend on the retrieved fact?** An LLM judge ("which of these retrieved facts did the answer actually rely on?") gives you the dependent set; the ratio of used-fact tokens to retrieved-fact tokens is your utilization.

```python
def context_utilization(query, retrieved_facts, answer) -> dict:
    """Of the retrieved facts we paid for, how many did the answer actually use?
    Proxy: an LLM judge marks which facts the answer depended on."""
    listing = "\n".join(f"{i}: {f['value']}" for i, f in enumerate(retrieved_facts))
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=128,
        system=("Given a question, the facts retrieved to answer it, and the answer, "
                "return a JSON array of the indices of the facts the answer actually "
                "DEPENDED ON. Exclude facts that were retrieved but unused."),
        messages=[{"role": "user", "content":
                   f"Q: {query}\nFacts:\n{listing}\nAnswer: {answer}"}],
    )
    used = set(json.loads(next(b.text for b in msg.content if b.type == "text")))
    used_tok = sum(n_tokens(retrieved_facts[i]["value"]) for i in used)
    spent_tok = sum(n_tokens(f["value"]) for f in retrieved_facts)
    return {"used": len(used), "retrieved": len(retrieved_facts),
            "utilization": used_tok / max(spent_tok, 1)}
```

What you do with the number: it's how you **tune `k`** (how many facts to retrieve) and the semantic slice's budget. Utilization near 1.0 with high recall means `k` is well-matched — you're retrieving about what you use. Utilization at 0.1 with high recall means `k` is too high — you're paying for ten facts to use one; *lower* `k` until utilization rises without recall dropping. The sweet spot is the largest budget reduction that holds recall flat — exactly the kind of measured trade-off the week keeps insisting on (the C23 stance: a number, not a vibe). Recall says "the fact got there"; utilization says "and you didn't drown it in nine you never used."

> **The efficiency rule:** recall is necessary but not sufficient — measure **context utilization** (used-fact tokens ÷ retrieved-fact tokens) alongside it. High recall with low utilization means you're retrieving far more than you use, wasting budget and crowding the answer toward the lost-in-the-middle sag. Tune `k` and the semantic slice *down* until utilization climbs without recall falling. The two metrics together — did the fact arrive, and was the budget spent on it — are how you size retrieval honestly.

---

## Part 10 — Failure modes of memory systems

A memory system that passes the turn-38 test on day one can still rot in production. Three failure modes you will hit, each with its mitigation — and each ties back to a discipline from earlier in the week.

**Stale memory.** The user renamed their project from Helios to Daedalus (Lecture 1 §5.1), but the old fact is still in the store and a similarity search retrieves it, so the agent confidently says "Helios." This is the fact-update problem from Lecture 1, viewed from the eviction side: a fact that *should* have been overwritten wasn't, and now it's actively wrong. *Mitigation:* the `(entity, attribute)` **upsert** (Lecture 1 §5.1) is the primary fix — one row per key, overwritten on change, so there's no stale duplicate to retrieve. Belt-and-suspenders: prefer the freshest fact on ties via the `updated_at` column, and put a **TTL** (Part 4.3) on facts that go stale on a clock rather than on an explicit rename ("currently debugging X").

**Memory poisoning.** A *wrong* fact gets extracted and stored — the user was quoting someone else, or the extractor hallucinated, or a typo became "the user's name is Jhon" — and because retrieval is by similarity, the wrong fact keeps coming back, turn after turn, stated with full confidence. Worse than no memory, because the agent *defends* it. *Mitigation:* don't trust extraction blindly. Gate writes on a **confidence threshold** (the LLM extractor from Lecture 1 §5 can emit a confidence; drop low-confidence facts rather than storing them), keep the **`updated_at` provenance** so a correction overwrites the poison via the same upsert key, and expose a **delete path** (Lecture 1 §7's per-user `DELETE`) so a detected bad fact can actually be removed, not just buried. The deeper point: extraction is a *source of errors*, and a memory store with no write-side validation faithfully remembers every one of them.

**Unbounded semantic growth.** Episodic memory is bounded by summarization and the recent-window cap, but *semantic* memory has no natural ceiling — every conversation extracts a few more durable facts and upserts them forever. Over months, a heavy user accumulates thousands of facts; retrieval slows, the top-`k` gets noisier (more near-duplicates and stale-but-not-wrong facts competing for the slot), and the durable tier — the one that was supposed to be the *reliable* one — degrades. *Mitigation:* semantic memory needs its **own consolidate-and-forget pass**, the durable-tier analog of episodic summarization. Periodically: merge near-duplicate facts (two rows that say the same thing under slightly different keys), drop facts whose salience (Part 8) is low *and* that haven't been retrieved in a long time (low salience × low frequency = forget), and TTL-expire anything transient that escaped its expiry. "Durable" was never meant to be "immortal" — it means *outlives the conversation*, not *accumulates without limit*.

> **The three failure modes, locked in:**
> - **Stale memory** → an un-updated fact retrieved as current. *Fix:* `(entity, attribute)` upsert + `updated_at` tie-break + TTL on clock-stale facts.
> - **Memory poisoning** → a wrong fact stored and re-retrieved with confidence. *Fix:* confidence-gate writes, keep provenance, expose a delete path. Extraction is a source of errors; validate on the way *in*.
> - **Unbounded growth** → semantic memory accumulates forever, degrading retrieval. *Fix:* a periodic consolidate-and-forget pass — merge duplicates, drop low-salience-low-frequency facts, expire transient ones.
>
> All three share one lesson: a memory system is not write-and-forget. It needs the same maintenance discipline as the cache it is — and "context is the most expensive cache on the planet" cuts both ways: the expensive thing also has to be *kept correct*, not just kept.

---

## Part 11 — Recap

You should now be able to:

- Treat the **context window as a cache** — small, finite, expensive, with hits (fact in context) and misses (fact evicted) — and manage it with a budget and an eviction policy instead of appending forever.
- **Budget the window**: allocate token slices (system / semantic / episodic-summary / recent / query), *measure* with the model's own tokenizer (never characters, never `tiktoken` for Claude), and *enforce* the cap.
- Explain **lost in the middle** (U-shaped recall — edges beat the middle) and why it makes budgeting a *quality* lever: a tight, edge-placed context beats a full one where the answer is buried.
- Design an **eviction policy**: LRU (simple, recency-as-relevance), salience-weighted (keep the important old fact — the memory-benchmark win), TTL (expire transient facts) — and understand that episodic eviction is survivable only because durable facts live in semantic memory.
- **Order the budget for prompt caching**: stable content first (system + durable facts behind a `cache_control` breakpoint), volatile content last — and reconcile the *real tension* with lost-in-the-middle by keeping the volatile middle small so the cached durable facts stay near the front edge.
- **Compute salience honestly**: score once at write time from cheap heuristics plus an LLM judge for the uncertain middle, store it on the row, and let eviction *read* it — never re-score on the hot path.
- **Measure efficiency, not just recall**: track **context utilization** (used-fact tokens ÷ retrieved-fact tokens) and tune `k` and the semantic slice down until utilization climbs without recall falling.
- **Name and mitigate the failure modes**: stale memory (upsert + `updated_at` + TTL), memory poisoning (confidence-gate writes, provenance, delete path), unbounded growth (a consolidate-and-forget pass) — because a memory system needs maintenance, not just construction.
- **Measure memory** with the turn-38 regression test: plant facts early, ask late, score recall against a no-memory baseline, and use the delta to justify the system and the test to choose the eviction policy.

Next: the exercises put this on a real agent — budget a window, build a rolling-summary episodic memory, and run the turn-38 regression test against a no-memory baseline. Continue to [the exercises](../exercises/README.md).

---

## References

- *Lost in the Middle: How Language Models Use Long Contexts* — Liu et al., 2023: <https://arxiv.org/abs/2307.03172>
- *MemGPT: Towards LLMs as Operating Systems* — Packer et al., 2023: <https://arxiv.org/abs/2310.08560>
- *Anthropic — token counting* (count with the model's tokenizer, not `tiktoken`): <https://docs.claude.com/en/docs/build-with-claude/token-counting>
- *Anthropic — context windows* (the finite budget you're spending): <https://docs.claude.com/en/docs/build-with-claude/context-windows>
- *Letta (formerly MemGPT)* — tiered memory with paging/eviction: <https://github.com/letta-ai/letta>
- *Ragas* (the eval discipline week 12 generalizes from this regression test): <https://docs.ragas.io/>
