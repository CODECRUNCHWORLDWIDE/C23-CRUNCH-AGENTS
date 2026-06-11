# Lecture 1 — The Three Memory Tiers and Summarization Strategies

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can name the three memory tiers (episodic, semantic, procedural), say exactly what belongs in each and why the split matters, and choose a summarization strategy (rolling, hierarchical, map-reduce) for a conversation's shape — so an agent stops forgetting the user's project name nine turns after hearing it.

If you remember one sentence from this entire week, remember this one:

> **Context is the most expensive cache on the planet. Spend it like one.** Every token costs money and latency, the window is finite, and the model gets measurably *worse* at the middle of a long context. You are not "adding memory"; you are *managing a cache* with summarization, eviction, and a budget.

There's a corollary you should tape next to it:

> **More context is not more memory.** A 200k-token window stuffed with raw history is *worse* than a 4k summary holding the three facts that matter — slower, costlier, and less accurate on the buried facts. Memory is *what you keep*, not *how much you can hold*.

For ten weeks your models were stateless. The API is stateless *by design* — you send the full conversation history every call, and the model remembers nothing on its own. Memory is the machinery *you* build around that stateless core to give an agent a past it can use. Everything that follows is in service of one shift: from "the model forgets everything between calls" to "the agent remembers the right things, at the right cost."

---

## 1. Why memory is not retrieval

You spent weeks 7–10 on retrieval: pull from a *fixed corpus* the chunks relevant to a query. Memory is a different thing, and conflating them is the first mistake:

- **Retrieval** reads a corpus that *exists before the conversation* and doesn't change because of it. The legal corpus is the same whether you ask it one question or a thousand.
- **Memory** *accumulates from the interaction itself*. What the user told you in turn 3, what tools the agent ran in turn 12, the running gist of a 40-turn conversation — none of that existed before the conversation; the conversation *created* it.

They share machinery (the semantic memory tier *is* a vector store, week 10's `crunchstore`), but they answer different questions. Retrieval answers "what does the corpus say about X?" Memory answers "what do *I* know about *this user* and *this conversation* that I learned along the way?" An agent needs both: retrieval for world knowledge, memory for interaction state. This week is the second one.

---

## 2. The three tiers — what goes where

Human memory has a useful structure that maps cleanly onto agents, and the syllabus names the three tiers directly. The split matters because *each tier has a different storage mechanism, a different lifetime, and a different retrieval pattern.* Put a fact in the wrong tier and you either lose it or pay too much to keep it.

### 2.1 Episodic memory — the turn history

**What it is:** the record of *what was said*, turn by turn — the conversation transcript. "The user asked about pricing, I answered, they followed up about the enterprise tier."

**Storage:** the message list itself, but *summarized* as it grows (§3) — because the raw transcript blows the context budget on a long conversation. Episodic memory is usually a **rolling summary of older turns** plus the **verbatim recent turns**.

**Lifetime:** the conversation. (It can be persisted across sessions, but its natural scope is "this conversation's history.")

**Retrieval pattern:** mostly *recency* — the last N turns verbatim, older turns compressed. You rarely vector-search the transcript; you keep the recent part and summarize the rest.

**The failure it prevents:** "what were we just talking about?" Without episodic memory, every turn starts cold and the agent can't follow a multi-turn thread.

### 2.2 Semantic memory — facts about the user and the world

**What it is:** *durable facts* extracted from the conversation — "the user's project is called Helios," "they're on the enterprise tier," "they prefer Python." Not the *transcript* of how you learned it, the *fact* itself.

**Storage:** a **vector store** (your week-10 `crunchstore` adapter) — embed each fact, retrieve by similarity to the current query. Optionally a **knowledge graph** when the *relationships* between facts matter (the user works at company X, X uses tool Y — a multi-hop question needs the graph, not just similarity; this is week 10's GraphRAG idea pointed at memory).

**Lifetime:** *durable* — semantic facts outlive the conversation. The user's project name is still true next week.

**Retrieval pattern:** *similarity* — when the user asks "what's my project called?", you embed the query and retrieve the matching fact ("project: Helios"), exactly like RAG, but over accumulated facts instead of a fixed corpus.

**The failure it prevents:** "you told me your project name 35 turns ago and I have no idea what it is." Semantic memory is how the agent answers a turn-38 question about a turn-3 fact — the "it remembered in turn 38" promise.

### 2.3 Procedural memory — tool histories and learned behaviors

**What it is:** the record of *what the agent did* and *how* — which tools it called, with what arguments, what worked, what failed. "I called `search_db(tenant='acme')` and it returned 3 rows; the `calculator` tool failed on a malformed input last turn."

**Storage:** a **tool-history log** — structured records of tool calls and outcomes, often append-only, queryable by tool or by recency.

**Lifetime:** varies — recent tool history for the current task; learned patterns ("this tool needs arguments in this format") can persist.

**Retrieval pattern:** by *tool* and by *recency* — "what did `search_db` return last time?", "have I already tried this approach?"

**The failure it prevents:** the agent re-running the same failing tool call in a loop (the "infinite tool-call loop" from week 5), or forgetting the result of a tool it called two turns ago and calling it again. Procedural memory is how an agent *learns from its own actions* within a task.

> **The three tiers, locked in:**
> - **Episodic** → *what was said* (turn history, rolling summary + recent verbatim, recency).
> - **Semantic** → *what's true* (facts about user/world, vector + KG, similarity, durable).
> - **Procedural** → *what I did* (tool histories + outcomes, log, by-tool/recency).
>
> A fact about the user goes in semantic; the transcript of the turn goes in episodic; the tool call that revealed it goes in procedural. Three different homes, because three different lifetimes and retrieval patterns.

---

## 3. Summarization strategies — compressing episodic memory

Episodic memory is where the context budget gets eaten alive, because the transcript grows without bound. You can't keep every turn verbatim — a 40-turn conversation is thousands of tokens, and most of it is no longer load-bearing. So you *summarize* the older turns. Three strategies, by the conversation's shape.

### 3.1 Rolling summary — the workhorse

Keep a *running summary* of the conversation, updated each turn (or every few turns). When a new turn happens, you fold it into the summary and drop the raw turn (beyond a recent window kept verbatim).

```python
import anthropic

client = anthropic.Anthropic()

def update_rolling_summary(prev_summary: str, new_turns: list[dict]) -> str:
    """Fold recent turns into the running summary. The summary is episodic memory."""
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in new_turns)
    msg = client.messages.create(
        model="claude-sonnet-4-6",          # a cheap, capable summarizer
        max_tokens=512,
        system=("You maintain a running summary of a conversation. Update the "
                "summary to incorporate the new turns. Keep durable facts "
                "(names, preferences, decisions); drop pleasantries. Be concise."),
        messages=[{"role": "user", "content":
                   f"Current summary:\n{prev_summary}\n\nNew turns:\n{transcript}\n\n"
                   "Return the updated summary."}],
    )
    return next(b.text for b in msg.content if b.type == "text")
```

**What it's good for:** the default. Cheap (one small LLM call per update), bounded (the summary stays roughly constant size), and it keeps a usable gist of the whole conversation. It's the right first thing to build.

**Where it's weak:** *lossy and order-dependent*. Each summarization pass can drop a detail that turns out to matter later, and once dropped it's gone. The fix is to *also* extract durable facts into **semantic memory** (§2.2) so they survive even if the rolling summary forgets them — which is why a good memory system uses *all three tiers*, not just a rolling summary.

### 3.2 Hierarchical summary — summaries of summaries

For *long* conversations, a single rolling summary gets too coarse — it's the average of everything, and the specific thing you need is averaged away (the same dilution problem as a too-large chunk in week 8). **Hierarchical summarization** keeps summaries at *multiple levels*: a fine summary of recent segments, a coarser summary of older segments, and a top-level summary of the whole. You retrieve at the granularity you need — recent detail when the question is about the last few turns, the coarse overview when it's about the whole conversation.

The structure mirrors a tree: leaf summaries (per segment) → mid summaries (per group of segments) → root summary (the whole conversation). When the conversation grows, you summarize a new segment and roll it up. This is more machinery than a flat rolling summary, and you reach for it when conversations are long enough that one flat summary loses the resolution you need.

### 3.3 Map-reduce summary — summarize chunks, then combine

When you need to summarize a *large batch at once* (e.g. condensing a 100-turn history in one shot, or summarizing a long document for memory), **map-reduce** is the pattern: *map* — summarize each chunk independently (parallelizable); *reduce* — combine the chunk summaries into one. It's the same map-reduce GraphRAG uses for community summaries (week 10), pointed at conversation history. It parallelizes well (each chunk summary is independent) and handles inputs too big for one context window, at the cost of more LLM calls and a combine step that can blur cross-chunk connections.

> **Choosing a strategy:**
> - **Rolling** → the default; ongoing conversation, update-as-you-go, cheap. Start here.
> - **Hierarchical** → long conversations where one flat summary loses resolution; multi-level detail.
> - **Map-reduce** → summarize a large batch at once; parallel, handles oversized inputs.
>
> And whichever you pick, *also* extract durable facts into semantic memory — summarization is lossy, and the facts that matter shouldn't depend on a summary remembering them.

---

## 4. The interaction: tiers + summarization together

Here's how the three tiers and the summarization strategies compose into a working memory system — the thing you build in the mini-project. On each turn:

1. **Episodic:** keep the last N turns verbatim; fold older turns into a **rolling summary** (§3.1). This is "what was said," compressed.
2. **Semantic:** extract any *durable facts* from the new turns ("project: Helios") and **upsert them into the vector store** (`crunchstore`). When the user asks something, **retrieve** the relevant facts by similarity and inject them. This is "what's true," and it's how a turn-38 question finds a turn-3 fact.
3. **Procedural:** log every tool call and its outcome to the **tool-history log**. Before re-running a tool, check the log. This is "what I did."

The prompt you actually send the model is then *assembled* from these tiers under a **budget** (Lecture 2): the system prompt + the retrieved semantic facts + the rolling episodic summary + the recent verbatim turns + the user's query — sized to fit the window, with the lowest-value content evicted when it's tight.

```python
def build_context(system, semantic_facts, episodic_summary, recent_turns, query):
    """Assemble the prompt from the three tiers. Lecture 2 adds the BUDGET that
    sizes each slice and evicts when the total exceeds the window."""
    return {
        "system": system,
        "messages": [
            {"role": "user", "content":
                f"[What I remember about you]\n{semantic_facts}\n\n"      # semantic
                f"[Conversation so far]\n{episodic_summary}\n\n"          # episodic (summary)
                + "\n".join(f"{t['role']}: {t['content']}" for t in recent_turns)  # episodic (verbatim)
                + f"\n\n{query}"},                                        # the query
        ],
    }
```

That assembly is the heart of the memory system. The three tiers feed it; the budget (next lecture) sizes it; and the regression test (Lecture 2) proves the turn-3 fact still reaches the model in turn 38.

---

## 5. Fact extraction — how the semantic tier decides what's durable

§3 told you to "extract durable facts into semantic memory." This section is the *how*. The rolling summary compresses everything; the semantic tier is pickier — it stores only **durable facts**, the things still true next week. The question is: out of a turn, what's a durable fact and what's conversational noise?

There are two extractors, and you'll likely run both.

**The heuristic extractor.** Cheap, deterministic, no LLM call. You pattern-match the obvious carriers of durable facts — "my name is X," "I'm working on Y," "I prefer Z," "we use W for CI." It's fast and free, but brittle: it misses anything phrased outside your patterns, and it can't tell a durable fact ("my project is Helios") from a transient one ("I'm debugging the login flow today"). Use it as a pre-filter, not the whole answer.

**The LLM extractor.** The real workhorse: ask `claude-sonnet-4-6` to read the new turns and emit structured **(entity, attribute, value)** facts — the same shape as a knowledge-graph triple, which is no accident (§6). The prompt should pull only *durable* facts and skip pleasantries, questions, and transient state.

```python
import anthropic, json

client = anthropic.Anthropic()

def extract_facts(new_turns: list[dict]) -> list[dict]:
    """Pull durable (entity, attribute, value) facts from recent turns.
    Returns [] when the turns carry nothing worth remembering."""
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in new_turns)
    msg = client.messages.create(
        model="claude-sonnet-4-6",          # cheap, capable extractor
        max_tokens=512,
        system=("Extract DURABLE facts about the user or their world as JSON "
                "objects {entity, attribute, value}. Durable = still true next "
                "week (names, roles, preferences, decisions, tools they use). "
                "SKIP transient state (current task, mood), questions, and "
                "pleasantries. Return a JSON array; [] if nothing durable."),
        messages=[{"role": "user", "content": transcript}],
    )
    text = next(b.text for b in msg.content if b.type == "text")
    return json.loads(text)
```

So `"my project is now called Helios"` yields `{"entity": "user.project", "attribute": "name", "value": "Helios"}`. That `entity.attribute` key is what makes the next problem solvable.

### 5.1 The de-duplication and fact-update problem

Naive memory *appends*: every extraction `INSERT`s a new row. That breaks the moment a fact *changes*. The user renames their project from "Helios" to "Daedalus" in turn 20 — append-only memory now holds **both** facts, and a turn-38 similarity search retrieves both, so the agent confidently tells the user their project is called Helios. *That is the central correctness bug of a semantic tier:* a stale fact that was never updated is worse than no fact, because the agent states it with confidence.

The fix is to treat the semantic store as a **key-value upsert**, not an append. The `(entity, attribute)` pair is the key; the `value` is what you overwrite. A new extraction for an existing key **updates** the row instead of inserting a duplicate.

```python
def upsert_fact(conn, user_id: str, fact: dict, embedding: list[float]):
    """UPDATE-not-duplicate: (user_id, entity, attribute) is the key."""
    conn.execute("""
        INSERT INTO semantic_memory (user_id, entity, attribute, value, embedding, updated_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (user_id, entity, attribute)
        DO UPDATE SET value = EXCLUDED.value,
                      embedding = EXCLUDED.embedding,
                      updated_at = now();
    """, (user_id, fact["entity"], fact["attribute"], fact["value"], embedding))
```

> **The fact-update rule:** semantic memory is an **upsert keyed on `(entity, attribute)`**, never a blind append. When the user changes a fact, you *overwrite* the value — one row per `(entity, attribute)`, always current. Append-only memory accumulates contradictions and the agent recalls the *wrong* one. This is the difference between a memory system and a memory *leak* of stale facts.

Two refinements you'll want once the basics work: keep an `updated_at` so you can break ties toward the freshest fact, and — for facts that genuinely have a history ("previously named Helios") — store the prior value rather than discarding it, so the agent can say "you renamed it from Helios." But the default is upsert: one current value per key.

---

## 6. Hybrid vector + knowledge-graph memory

§2.2 mentioned a knowledge graph "when the relationships between facts matter." Here's when that's not optional. Flat vector memory stores each fact as an independent embedded blob and retrieves by *similarity to the query*. That works when the answer is *one fact*. It **fails when the answer requires connecting facts** — because similarity to the query doesn't surface a fact the query never mentions.

Consider the multi-hop question: **"what does the company my user works at use for CI?"** The facts in memory are:

- `(user, works_at, Acme)`
- `(Acme, uses_for_ci, GitHub Actions)`

The query embeds near `works_at` (it mentions "the company my user works at"), so flat vector search retrieves `(user, works_at, Acme)` — and stops. The CI fact embeds near "CI / GitHub Actions," which the query *doesn't say*, so similarity never retrieves it. The answer needs a **two-hop traversal**: user → Acme → GitHub Actions. Similarity can't hop; a graph can. This is week 10's GraphRAG insight pointed straight at memory — when the *relationship* is the retrieval signal, you need edges, not just embeddings.

The store is a **(subject, relation, object) triple table** in Postgres, sitting alongside the pgvector facts:

```sql
CREATE TABLE memory_triples (
    user_id   text  NOT NULL,
    subject   text  NOT NULL,
    relation  text  NOT NULL,
    object    text  NOT NULL,
    embedding vector(1024),                 -- for the entry-point similarity hop
    PRIMARY KEY (user_id, subject, relation, object)
);
```

A multi-hop query runs in two stages: **(1) entry point** — vector-search to find the subject the query is about (`user → works_at → Acme`); **(2) traversal** — follow edges from that node to answer the rest (`Acme → uses_for_ci → ?`).

```python
def multi_hop(conn, user_id: str, start_subject: str, relations: list[str]) -> str:
    """Follow a chain of relations from a starting node. Each hop is a triple lookup.
    multi_hop(conn, uid, 'user', ['works_at', 'uses_for_ci']) -> 'GitHub Actions'."""
    node = start_subject
    for rel in relations:
        row = conn.execute(
            "SELECT object FROM memory_triples "
            "WHERE user_id = %s AND subject = %s AND relation = %s "
            "ORDER BY 1 LIMIT 1;",
            (user_id, node, rel),
        ).fetchone()
        if row is None:
            return None                     # chain broke — fact not in memory
        node = row[0]
    return node
```

You keep *both* stores because they answer different shapes of question. Flat pgvector answers "what's my project called?" (one-hop, similarity). The triple store answers "what CI does my employer use?" (multi-hop, relational). The `(entity, attribute, value)` facts from §5 map cleanly onto `(subject, relation, object)` triples — the extractor already emits the right shape, which is why we extracted facts that way. This is the hybrid memory the syllabus prefers: pgvector for similarity recall, a Postgres KG for relational recall, both keyed per user.

> **When you need the graph:** flat vector memory answers single-fact questions; it *cannot* answer a question whose answer is a fact the query doesn't mention but is *connected to* one that it does. The moment "the company my user works at" needs to resolve to a fact about *that company*, you need edges. Relationship-as-retrieval-signal = knowledge graph. (Week 10's GraphRAG, now pointed at memory.)

---

## 7. Cross-session vs within-session memory

The three tiers don't all live on the same timescale, and the timescale drives the storage design. **Episodic memory is naturally within-session** — its scope is "this conversation's turns," and a new session legitimately starts with an empty transcript. **Semantic memory naturally persists across sessions** — the user's project name is still true when they come back tomorrow; throwing it away at session end would re-break the turn-38 promise across the session boundary. Procedural memory splits: the current task's tool history is within-session, while learned patterns ("this tool wants ISO dates") can persist.

The design implications are concrete:

- **Per-user namespacing is mandatory.** Every persisted row carries a `user_id` (you saw it in §5's upsert and §6's triple table). Cross-session memory *means* you'll have many users' facts in one store, and retrieval must be scoped to the asking user — a missing `WHERE user_id = %s` is a privacy incident, one user recalling another's facts. Namespace at the store, not in application code you might forget.
- **Persist the rolling summary at session boundaries, deliberately.** When a session ends, you choose: persist the episodic rolling summary (so the next session opens with "last time we set up your CI") or let it reset and rely on the durable semantic facts to carry continuity. Persisting the summary gives warmer continuity; resetting it is cheaper and avoids carrying stale narrative. A common middle path: persist a short *cross-session* summary ("returning user; previously configured Helios's CI") and let the *within-session* detail reset.
- **PII and the right to be forgotten.** Stored user facts *are* personal data, and durable-by-design means they outlive the conversation that created them. That carries obligations the within-session transcript doesn't: a per-user delete that actually removes the rows (the `(entity, attribute)` key makes this a clean `DELETE WHERE user_id = %s`), a TTL on facts that shouldn't be kept indefinitely (Lecture 2 §4.3), and *not* extracting sensitive facts you have no reason to store. "Durable" is a commitment, not a free win — measure what you keep against what you're allowed to keep.

> **The timescale rule:** episodic is *within-session* (the transcript resets); semantic is *cross-session* (facts persist, so they need `user_id` namespacing and a delete path). The turn-38 promise lives in the durable tier precisely *because* it's the tier that survives a session boundary — which is also the tier that carries privacy weight.

---

## 8. A worked memory-loop trace

Abstractions land when you watch one fact travel the whole loop. Here is a single conversation, narrated turn by turn, showing a fact enter at turn 3, get extracted to semantic memory, the episodic summary roll as turns scroll out, and the fact get recalled at turn 38 — the "it remembered in turn 38" promise, traced.

```
turn 03  user: "my project is called Helios and we deploy on Fridays"
         observe -> episodic: appended verbatim (still in the recent window)
         extract -> claude-sonnet-4-6 returns two facts:
                    {user.project, name, "Helios"}
                    {user.project, deploy_day, "Friday"}
         upsert  -> two rows keyed (user, user.project, name|deploy_day)

turn 04-12  distractor turns (debugging, questions, pleasantries)
         extract -> mostly [] (transient state, no durable facts)
         episodic recent window now full (last N turns only)

turn 13  ROLL: turns 3-8 exceed the episodic slice -> folded into the
         rolling summary by claude-sonnet-4-6. The verbatim turn-3 text is
         now GONE from episodic. <- this is where naive memory forgets Helios.
         But the FACT is safe in semantic memory, untouched by the roll.

turn 20  user: "actually we renamed it to Daedalus"
         extract -> {user.project, name, "Daedalus"}
         upsert  -> ON CONFLICT(user, user.project, name) UPDATES the value.
                    One row, now "Daedalus". No duplicate. (§5.1)

turn 21-37  more distractors; rolling summary keeps rolling; recent window
         keeps only the last N turns verbatim.

turn 38  user: "what's my project called and when do we deploy?"
         retrieve -> embed the query, similarity-search semantic memory,
                     get {name: "Daedalus", deploy_day: "Friday"}
         assemble -> system + retrieved facts + episodic summary + recent
                     turns + query, sized to the budget (Lecture 2)
         respond  -> "Your project is Daedalus, and you deploy on Fridays."
```

Trace what each tier did. **Episodic** carried the recent turns and a rolling summary, but it had *dropped* the verbatim turn-3 text by turn 13 — episodic alone would have failed. **Semantic** held the durable facts across 35 turns and one rename, and the §5.1 upsert is why turn 38 says "Daedalus," not "Helios" — the bug a naive append-only store would have shipped. **The retrieval at turn 38 is a similarity hit**, not a scan of history. That is the whole architecture working: the fact survived the roll because it was promoted to the durable tier, and it stayed *correct* because the tier upserts instead of appends.

> **The trace, in one line:** a fact enters episodic *and* gets promoted to semantic at turn 3; the episodic verbatim is gone by turn 13 but the semantic fact survives; a rename at turn 20 *updates* it; turn 38 retrieves the *current* fact by similarity. Promotion + upsert + retrieval — that's how "it remembered in turn 38" actually happens.

---

## 9. Where the frameworks fit (placed, not adopted)

The syllabus names three memory frameworks. You build the tiers *from scratch* this week (the preferred path is self-built pgvector + a Postgres KG schema), and you *place* the frameworks so you know what they'd give you:

- **Letta (formerly MemGPT)** — tiered memory with *paging and eviction* between a small "main context" and a large external store, inspired by an operating system's virtual memory (the MemGPT paper, arXiv 2310.08560). It's the most direct realization of "context is a cache you page": when main context fills, it evicts to the external store and pages back in on demand. If you want the tiered-memory-with-eviction pattern as a library, Letta is it.
- **Zep** — a *conversation memory store*: it ingests the message history, builds a summary and a fact/entity graph, and serves memory back to your agent. It's the "managed episodic + semantic" option.
- **Mem0** — a *long-term memory layer* that extracts and stores salient facts across sessions and retrieves them — the "managed semantic memory across conversations" option.

The reason to build it yourself first (the C23 stance): you can't reason about eviction, budget, or the lost-in-the-middle effect if a framework hides them. Build the three tiers by hand, *feel* the context budget bite, and *then* you can pick a framework knowing exactly what it's doing for you — or keep the self-built pgvector + KG path, which the syllabus prefers because it's transparent and you already operate the store (week 10).

---

## 10. Recap

You should now be able to:

- Distinguish **memory from retrieval**: retrieval reads a fixed corpus; memory accumulates from the interaction — different questions, shared machinery.
- Name and place the **three tiers**: episodic (what was said — rolling summary + recent verbatim, recency), semantic (what's true — vector + KG, similarity, durable), procedural (what I did — tool-history log, by-tool/recency) — and put a fact in the right home.
- Choose a **summarization strategy**: rolling (the default, cheap, update-as-you-go), hierarchical (long conversations, multi-level resolution), map-reduce (large batch, parallel) — and always *also* extract durable facts to semantic memory because summarization is lossy.
- **Extract durable facts** for the semantic tier with a `claude-sonnet-4-6` `(entity, attribute, value)` extractor (over a brittle heuristic), and store them as an **upsert keyed on `(entity, attribute)`** — so a renamed fact *updates* one row instead of duplicating into a contradiction the agent recalls wrong.
- Reach for **hybrid vector + knowledge-graph memory** when the *relationship* is the retrieval signal: a `(subject, relation, object)` triple store answers multi-hop questions ("what CI does my employer use?") that flat similarity *cannot*, because the answer is a fact the query never mentions (week 10's GraphRAG, pointed at memory).
- Separate **within-session from cross-session** memory: episodic resets per session; semantic persists, which mandates `user_id` namespacing, a deliberate choice about persisting the rolling summary, and a PII/delete path for durable user facts.
- **Trace the memory loop** end to end: a fact promoted to semantic at turn 3 survives the episodic roll at turn 13, gets *updated* by a rename at turn 20, and is retrieved by similarity at turn 38 — promotion + upsert + retrieval.
- **Assemble the prompt from the tiers** — system + retrieved semantic facts + episodic summary + recent verbatim turns + query — the structure the budget (next lecture) sizes and the regression test verifies.
- **Place the frameworks** (Letta = tiered memory with paging/eviction, Zep = conversation memory store, Mem0 = long-term layer) against the self-built pgvector + KG path the syllabus prefers.

Next: how to *budget* that assembled context like the cache it is — token allocation, eviction policies, the lost-in-the-middle effect — and how to *measure* memory with a regression test that proves the agent remembers turn 3 in turn 38. Continue to [Lecture 2 — Context Budgeting, Eviction, and Evaluation](./02-context-budgeting-eviction-and-evaluation.md).

---

## References

- *MemGPT: Towards LLMs as Operating Systems* — Packer et al., 2023 (the tiered-memory-with-paging pattern Letta implements): <https://arxiv.org/abs/2310.08560>
- *Lost in the Middle: How Language Models Use Long Contexts* — Liu et al., 2023 (why a tight context beats a full one): <https://arxiv.org/abs/2307.03172>
- *Letta (formerly MemGPT)* — tiered memory with eviction: <https://github.com/letta-ai/letta>
- *Zep* — conversation memory store: <https://github.com/getzep/zep>
- *Mem0* — long-term memory layer (fact extraction + cross-session storage, §5/§7): <https://github.com/mem0ai/mem0>
- *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — Edge et al., 2024 (the GraphRAG idea §6 points at memory): <https://arxiv.org/abs/2404.16130>
- *pgvector* — vector similarity in Postgres (the semantic tier *and* the entry-point hop for the KG, §5/§6): <https://github.com/pgvector/pgvector>
- *Anthropic — tool use / structured output* (how the `claude-sonnet-4-6` fact extractor returns `(entity, attribute, value)` JSON, §5): <https://docs.claude.com/en/docs/build-with-claude/tool-use>
- *Anthropic — building with the Messages API* (the API is stateless; you send full history each call): <https://docs.claude.com/en/api/messages>
