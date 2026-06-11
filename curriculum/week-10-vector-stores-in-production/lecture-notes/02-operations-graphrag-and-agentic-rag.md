# Lecture 2 — Operations, the Recovery Drill, GraphRAG, and Agentic RAG

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can operate a vector store like the database it is — back it up, restore it, reason about replication and the rebuild-after-schema-change problem — and run an index-loss recovery drill that produces a time-to-recover number. You can explain GraphRAG (Microsoft's community-summary pattern) and when a knowledge graph answers questions a flat index can't, and describe agentic RAG (the agent chooses the retriever) and where it earns its complexity.

Lecture 1 placed the stores and named the criteria. This lecture is about the criterion that decides the most production choices and that nobody benchmarks until it bites them — **recovery** — plus two patterns that change *what retrieval is*: GraphRAG and agentic RAG.

> **A vector store is a database first.** The query is the easy part; the operations are the job. This lecture is the operations.

---

## Part 1 — Operating the store like a database

### 1.1 Backup and restore

Every vector store holds data you cannot afford to lose and cannot always cheaply regenerate. "We can just re-embed" is a tempting lie: re-embedding a 10 GB corpus through an embedding model is *hours* of compute, and during those hours your retrieval is down. So you back up, and — more importantly — you *practice the restore*, because a backup you've never restored is a hope, not a backup.

The three stores' backup stories, and why they differ operationally:

- **pgvector → it's Postgres's backup story.** `pg_dump`/`pg_restore`, base backups, point-in-time recovery, streaming replicas. The huge advantage: **your team already knows this.** Your existing Postgres backup automation covers your vectors for free. Recovery is a `pg_restore` your DBA has run a hundred times. This familiarity *is* the operational win Lecture 1 §1 named.
- **Qdrant → snapshots.** `create_snapshot` produces a point-in-time file per collection; `recover` restores it. Snapshots are *fast* to create and restore (it's a purpose-built mechanism), which is why Qdrant shines in the recovery drill — the index comes back in seconds, not a re-embed's hours.
- **Weaviate → the backup module.** Configure a backend (filesystem/S3/GCS) and call backup/restore; clean, a touch more setup than the other two.

The common thread: **the backup mechanism determines the recovery time, and the recovery time is a first-class selection criterion** (Lecture 1 §5). A store whose only "backup" is "re-embed the corpus" has a recovery time measured in hours and a real outage attached.

### 1.2 Replication and high availability

Backup protects against *data loss*; replication protects against *downtime*. A replica is a live copy on another node; if the primary dies, a replica serves traffic with little or no interruption. For a store that's on the critical path of your product (retrieval down = product down), you want HA:

- **pgvector** inherits Postgres streaming replication and the mature HA tooling around it (Patroni, managed Postgres HA).
- **Qdrant** offers a distributed mode with sharding *and* replication — shards spread the data, replicas protect each shard.
- **Weaviate** and **Milvus** have their own replication/HA stories (Milvus's separated-node architecture is built for it).

The reasoning is the same as any stateful service: how many replicas, in how many availability zones, with what failover time, for what cost. This is the SRE persona's home turf (the syllabus names the "SRE bridging into AI infrastructure"), and the vector store is just another stateful service in the topology — which is the whole "database first" point.

The topology choices, made concrete, because "set up HA" is not a plan:

- **Primary + N replicas (read scaling + failover).** One primary takes writes (ingest, upserts), replicas serve reads (queries). This buys two things: read throughput (queries fan out across replicas) and failover (a replica gets promoted when the primary dies). For a *read-heavy* retrieval workload — which RAG is, you ingest once and query forever — replicas are mostly free QPS. The cost is replication lag: a replica can be seconds behind on freshly-upserted vectors, so a query right after an ingest might miss the new chunk. Usually fine for RAG (the corpus isn't changing under the query); it bites only when ingest and query are tightly interleaved.
- **Sharding (horizontal capacity).** Split the vectors across shards by some key (hash of `chunk_id`, or by `tenant`). Each shard holds — and indexes — a *fraction* of the corpus, so the HNSW graph per node fits in RAM even when the whole corpus wouldn't. Qdrant's distributed mode and Milvus's data nodes both do this. Sharding is a *capacity* answer (the corpus is too big for one node's memory), not an availability answer — a single-replica shard is still a single point of failure. You shard for size and *also* replicate each shard for HA; the two are orthogonal levers.
- **Multi-AZ (zone failure).** Spread replicas across availability zones so a whole-AZ outage doesn't take retrieval down. The cost is cross-AZ network: replication traffic and sometimes query traffic crosses the AZ boundary, which adds latency and (on cloud) egress cost. The reasoning is the SLA's: if your retrieval SLA must survive an AZ loss, you pay the cross-AZ tax; if a few minutes of single-AZ downtime is acceptable, you don't.

The decision variable that ties these together is **failover time** — from "primary dead" to "a replica serving traffic." A managed Postgres HA setup (Patroni, or a cloud provider's managed failover) measures this in *tens of seconds*; a manual "notice it's down, promote a replica by hand" measures it in *minutes-to-whoever's-awake*. That failover number sits *alongside* the recovery-time number from the drill (§1.5) — they answer different questions. Failover is "the node died, how fast does another take over" (HA, replicas); recovery is "the *data/index* is gone or corrupt, how fast do I rebuild it" (backup, restore). You need both numbers, because a replica is a live copy of a *corrupt* index too — replication faithfully copies corruption, so HA does **not** protect you against the index-corruption scenario, which is exactly why the recovery drill exists.

> **HA and backup solve different failures.** Replication protects against a *node* dying (failover, tens of seconds). Backup/restore protects against the *data* being lost or corrupted (recovery, the drill's number). A replica of a corrupt index is a corrupt replica — so you need both, and the week-24 chaos drill targets the one HA can't save you from.

### 1.3 The rebuild-after-schema-change problem

Here's an operational reality that surprises people: **changing the schema can mean rebuilding the index.** Two flavors:

- **Adding a filterable field.** If you decide late that you need to filter on `clause_type` and you didn't index it at ingest, you add the metadata field and build its index — usually cheap, no re-embed, because the *vectors* don't change. The lesson from Lecture 1 §4: decide your filters *before* ingest to avoid even this.
- **Changing the embedding model or dimension.** This is the expensive one. If you swap embedding models (a new BGE version, a different model entirely), the vectors are *different* and the whole index must be **re-embedded and rebuilt from the source text**. There is no shortcut — the old vectors are meaningless in the new model's space. This is why the embedding choice (week 7) is a *long-lived* decision: changing it is a full re-ingest. Your migration plan for a model upgrade is "re-embed the corpus, build a parallel index, cut over" — and the ingest throughput (§ recovery) bounds how long that takes.

The takeaway: a "schema change" in a vector store ranges from cheap (add a metadata index) to a full re-ingest (change the embedding). Knowing which is which — and that the embedding model is the expensive-to-change one — is part of operating the store.

### 1.4 Monitoring and observability — seeing the degradation before the outage

A database you don't monitor is a database you discover is broken when a customer tells you. The vector store is no exception, and it has a failure mode most databases don't: it can **degrade silently in *quality*** — recall drops — while every infrastructure metric stays green. So you monitor two layers: the ordinary stateful-service metrics (the SRE knows these) *and* the retrieval-quality metrics (the AI engineer adds these). This is where week 10 hands off to week 18's observability and week 24's chaos drills — the metrics you define here are what the dashboards there watch.

The metrics that actually matter, and what each one tells you *before* a full failure:

- **Query latency p95 / p99.** The headline operational metric. A creeping p95 means the index is growing past RAM (HNSW is paging from disk), `ef_search` is too high for the load, or a replica is lagging and queries are queueing. p50 stays fine while p95 walks — which is *why* you watch the tail, not the mean. A step-change in p95 with no traffic change is usually the index spilling out of memory.
- **Recall drift.** The one metric ordinary database monitoring doesn't have. Run a small fixed **canary gold set** (a few dozen queries with known-good answers) on a schedule — every deploy, and hourly in production — and track Recall@5 over time. A *drop* means something changed the retrieval quality: a partial index corruption, a botched re-ingest, an embedding-version skew (some chunks embedded with the old model), or quantization pushed too far. Recall drift is how a corrupted index announces itself *before* it errors — the queries still return results, the results are just *wrong*, and only the canary catches it. **This is the single most important AI-specific metric, and almost nobody has it.**
- **Ingest lag.** The gap between "document written to the source of truth" and "its chunk is searchable in the index." A growing lag means ingest can't keep up with the write rate, or the embedding step is the bottleneck. For a changing corpus (the GraphRAG rebuild question, §3) this lag *is* your freshness SLA.
- **Index memory (RAM) and disk.** HNSW lives in memory (Lecture 1 §6); watch resident memory against the box's RAM, because the moment the graph doesn't fit, p95 explodes as it pages. Disk matters for the on-disk vectors, the WAL/segments, and — critically — **headroom for a restore** (you cannot restore a 10 GB snapshot onto a disk with 4 GB free). Track both, alert before either fills.
- **Replication lag and node health.** For the HA topology (§1.2): how far behind are the replicas, are all shards up, is any node flapping. A lagging replica serving stale reads is a subtle recall/freshness bug.

How a corrupted or degraded index *shows up* — the signature you learn to read, because it's the week-24 drill's whole point:

```text
healthy:    p95=18ms   canary Recall@5=0.88   ingest_lag=2s    mem=61%
degrading:  p95=18ms   canary Recall@5=0.71   ingest_lag=2s    mem=61%   <- recall drift, infra GREEN
failing:    p95=140ms  canary Recall@5=0.34   ingest_lag=90s   mem=61%   <- now latency too
gone:       queries ERROR / Recall@5=0.0                                  <- the 2 AM page
```

The lesson in that table: **the corruption is visible in recall *one or two stages before* it's visible in latency or errors.** A team watching only p95 and error rate sees green until the "failing" row; a team watching the canary recall sees the problem at the "degrading" row, with time to fail over to a replica or restore from backup *before* customers notice. That early-warning gap — recall drift caught while infra is still green — is the entire argument for adding the retrieval-quality layer to your monitoring, and it's the bridge to week-18 observability (instrument the retrieval, not just the box) and the week-24 chaos drill (where you'll *inject* index corruption and confirm your monitoring catches it at the "degrading" row, not the "gone" row).

> **Watch recall, not just latency.** A vector store degrades in *quality* before it degrades in *performance*. A scheduled canary-gold-set Recall@5 is the smoke detector that goes off while the infra dashboards are still green — without it, the first alert is a customer complaint. Define this metric in week 10; wire it to the dashboard in week 18; prove it fires in the week-24 drill.

### 1.5 A worked recovery-time estimate — snapshot-restore vs re-embed

Lecture 1 §5 ranked recovery time as a top selection criterion and §1.1 above asserted "re-embed is hours, snapshot is seconds." Here's the arithmetic that makes those numbers real, because "measure it" deserves a back-of-envelope you can do *before* the drill confirms it. Take a realistic capstone-scale corpus: **5 million vectors, ~10 GB** of source text and embeddings.

**Path A — snapshot-restore (the backup you took).** Restore is fundamentally a *copy* of bytes plus, at worst, an index load:

```text
restore = read_snapshot + load_index
10 GB snapshot off local/attached disk @ ~500 MB/s  ≈ 20 s
load the prebuilt HNSW graph into memory             ≈ seconds–low minutes
-----------------------------------------------------------------
time-to-recover  ≈ tens of seconds to a couple of minutes
```

A Qdrant snapshot or a pgvector base-backup that *includes the built index* lands here — you copy bytes and the index is ready. If the snapshot holds vectors but *not* the built index, add the HNSW rebuild (below) to this path — which is exactly why "does the snapshot include the index?" is a question you ask of every store (Lecture 1 §6).

**Path B — re-embed from source (the "backup" that isn't).** No usable snapshot, so you regenerate every vector by running the source text back through the embedding model, then rebuild the index:

```text
re-embed = corpus_size / embedding_throughput,  then build the index

Say a batched embedding pipeline does ~1,000 chunks/s (an optimistic
self-hosted BGE-large on a decent GPU, batched; a rate-limited hosted
embedding API can be far slower):

    5,000,000 chunks / 1,000 chunks/s        ≈ 5,000 s   ≈ 1.4 hours
    + HNSW build over 5M vectors             ≈ tens of minutes
-----------------------------------------------------------------
time-to-recover  ≈ ~2 hours  (and longer if the embedding API throttles you)
```

The contrast is the whole point: **seconds-to-minutes vs hours**, a 50–100× difference, for the *same* corpus, decided entirely by whether you have a real snapshot. And the re-embed number is *optimistic* — drop the throughput to 200 chunks/s (a throttled hosted embedding API, the common reality) and Path B is ~7 hours, an entire incident. During those hours, retrieval is **down**: every RAG query in your product returns nothing.

> **Recovery time is `corpus_size ÷ embedding_throughput` — unless you have a snapshot, in which case it's `snapshot_size ÷ disk_bandwidth`.** The first is hours; the second is seconds. The recovery drill (Part 2) *measures* which one your store gives you; this arithmetic tells you the answer *before* the drill — and tells you that "we'll re-embed" is a multi-hour outage wearing a backup's clothing.

Two operational corollaries fall out of the arithmetic. First, **embedding throughput is a recovery-time input** — the same number that bounds your re-index-after-model-swap (§1.3) bounds your worst-case recovery, so it's worth knowing your actual batched chunks/s. Second, **snapshot size sets the restore-disk headroom** the monitoring (§1.4) must guard: you cannot restore a 10 GB snapshot onto a box with 6 GB free, and discovering that *during* a restore turns a 2-minute recovery into a provision-a-bigger-disk incident.

---

## Part 2 — The index-loss recovery drill

This is the week's signature exercise and the source of the "it survived the index loss" promise. The chaos drill in week 24 includes "retrieval index corruption" as one of three scenarios; this is the rehearsal. The drill is simple to state and brutally clarifying to run:

1. **Establish the baseline.** Ingest the corpus, build the index, run `evaluate()` → record Recall@5 and the ingest throughput. This is "healthy."
2. **Take a backup.** Snapshot (Qdrant) / `pg_dump` (pgvector) / backup module (Weaviate). Record how long the backup took.
3. **Destroy the index.** Drop the collection / table. Confirm retrieval is now broken (Recall@5 → 0, or queries error). This is the "2 AM, the index is gone" moment.
4. **Restore.** From the backup. **Start a stopwatch.** Restore, rebuild whatever needs rebuilding, re-run `evaluate()`.
5. **Record time-to-recover.** From "destroyed" to "Recall@5 back to baseline." *That number* is the headline.

```python
def recovery_drill(store, name, gold, retrieve_fn) -> dict:
    # 1. baseline
    baseline = evaluate(gold, retrieve_fn, k=5)["Recall@k"]
    # 2. backup
    handle = store.snapshot(name)
    # 3. destroy
    store.drop(name)
    assert evaluate(gold, retrieve_fn, k=5)["Recall@k"] == 0.0  # confirmed down
    # 4. restore (timed)
    import time
    t0 = time.perf_counter()
    store.restore(name, handle)
    recovered = evaluate(gold, retrieve_fn, k=5)["Recall@k"]
    ttr = time.perf_counter() - t0
    # 5. record
    return {"baseline_recall": baseline, "recovered_recall": recovered,
            "time_to_recover_s": ttr}
```

What the drill teaches that no benchmark does:

- **The store that queries 2 ms faster but restores in 4 hours is the worse production choice.** The drill makes that visible: snapshot-restore (Qdrant) recovers in seconds; a store with no real backup recovers in a re-embed's hours. The recovery number *reorders* the stores from how the latency number ranked them.
- **"We can re-embed" is an outage, not a backup.** Run the drill *without* a backup (re-embed from source) and time it — that's your worst-case recovery, and it's almost always unacceptable.
- **Recovery must restore the *index*, not just the data.** Restoring the vectors but having to rebuild the HNSW index from scratch adds the index-build time to recovery; a snapshot that includes the built index recovers faster. Measure the whole path: data back *and* Recall@5 back.

> **The "it survived the index loss" promise, made measurable:**
> ```
> store=qdrant  baseline Recall@5=0.88
>   *** index dropped — retrieval is DOWN ***
>   restore from snapshot... Recall@5 back to 0.88 in 47s
>   RECOVERED in 47s
> ```
> 47 seconds is a blip; 4 hours is an incident. The drill is how you know which one your store gives you *before* the bad night, not during it.

---

## Part 3 — GraphRAG: when a flat index can't answer the question

Everything so far assumes flat retrieval: chunk → embed → nearest-neighbor → top-K. That pattern answers **local** questions superbly — "what's the confidentiality duration?" matches the clause that says it. But it *fails* on a class of questions that flat retrieval structurally cannot answer:

- **Multi-hop:** "Which clauses does the termination clause depend on?" — the answer requires *following relationships* between clauses, not finding one similar chunk.
- **Global / thematic:** "What are the main themes across this entire contract?" — no single chunk contains the answer; it's a property of the *whole corpus*, and nearest-neighbor over chunks can't synthesize it.

**GraphRAG** (Edge et al., Microsoft, 2024 — arXiv 2404.16130) is the pattern built for exactly these. The mechanism, in three moves:

1. **Extract a knowledge graph.** Run an LLM over the corpus to extract **entities** (parties, obligations, clauses, dates) and **relationships** (clause 9 *references* clause 14; the Contractor *owes* a duty to the Company). The corpus becomes a graph, not a bag of chunks.
2. **Build community summaries.** Cluster the graph into **communities** (densely-connected groups of entities) and have the LLM write a *summary* of each community. Now you have summaries at multiple levels — fine communities and coarse ones.
3. **Retrieve over the graph.** For a *local* question, traverse the graph from the relevant entities (multi-hop following). For a *global* question, retrieve and combine the *community summaries* (map-reduce over summaries) — which is how GraphRAG answers "main themes" that no chunk holds.

The honest trade-off:

- **GraphRAG is expensive to build.** Extracting the graph means an LLM pass over the whole corpus (entity + relationship extraction); summarizing communities is more LLM calls. For a static corpus you pay this once; for a fast-changing one it's a recurring cost.
- **It wins specifically on multi-hop and global questions** — the questions flat retrieval *can't* do. On the local questions flat hybrid retrieval already nails, GraphRAG adds cost for little gain. So it's a *complement*, not a replacement: flat hybrid for local, GraphRAG for multi-hop/global.
- **The vector store still matters** — GraphRAG stores entity embeddings and summary embeddings *somewhere*, often the same vector store. The graph is an *additional* structure, not a different database. (Some stores, like Weaviate with its cross-references, lean toward representing the graph natively — Lecture 1 §2.3.)

The stretch goal has you build a tiny GraphRAG over the legal corpus and find one question it answers that hybrid retrieval misses — that one question is the entire justification for the pattern, and finding it teaches you when to reach for it.

### 3.1 The build cost, counted — and when to rebuild

"GraphRAG is expensive to build" is true but lazy; a senior engineer puts numbers on it, because the cost *is* the decision. The build is **LLM passes over the corpus**, and you can estimate the token bill the way you'd estimate any batch LLM job. Two passes dominate:

1. **Entity + relationship extraction.** Every chunk goes through an LLM (in Microsoft GraphRAG, a structured-extraction prompt) that emits the entities and the relationships it found. So the input is *the whole corpus, once* — for a 10 GB / 5M-chunk corpus that's billions of input tokens — plus a smaller output per chunk (the extracted tuples). This is the floor: you cannot build the graph without reading every chunk through an LLM at least once.
2. **Community summarization.** After clustering the graph into communities, an LLM (use **claude-sonnet-4-6** here — the summaries want a capable model, and the summarization volume is far smaller than the extraction pass) writes a summary per community, at *each* level of the hierarchy. The token cost scales with the number of communities and the size of each, not the raw corpus — typically a fraction of the extraction pass, but non-trivial because coarse communities summarize a lot of text.

The honest framing: extraction is the *expensive, corpus-sized* pass; summarization is the *smaller, graph-sized* pass. Both are batch jobs you can cost in advance (chunks × avg tokens × price), and both are why GraphRAG is a *deliberate* investment, not a default. For the 50-clause teaching corpus the whole build is cents and minutes; the point of doing the arithmetic is so you can scale it in your head to the corpus that *isn't* 50 clauses.

**When to rebuild on a changing corpus** is the operational question that decides whether GraphRAG is viable for *your* corpus:

- **Full rebuild.** Re-extract and re-summarize the entire corpus. Correct, simple, and you pay the full token bill every time. Fine for a corpus that changes monthly; absurd for one that changes hourly.
- **Incremental update.** When a *few* documents change, re-extract only those, merge their entities/relationships into the existing graph, and re-summarize only the **affected communities** (the ones whose member entities changed). Much cheaper — you pay for the delta, not the corpus — but more complex, because merging entities (is this "the Company" the same entity already in the graph?) and deciding which communities are "affected" is real engineering. Microsoft GraphRAG has moved toward incremental indexing for exactly this reason.

The decision rule is freshness-vs-cost: **a slow-changing corpus rebuilds fully on a schedule; a fast-changing corpus needs incremental indexing or GraphRAG isn't worth it.** If your corpus churns faster than you can afford to re-extract it, the graph is perpetually stale, and a stale graph answers multi-hop questions *wrong* — worse than not having it. This is the same freshness SLA the ingest-lag metric (§1.4) tracks, applied to the graph instead of the flat index.

### 3.2 A worked example over the legal corpus

Make it concrete with one question the flat index *structurally* cannot answer and the graph *can*. Take the legal corpus and ask:

> *"If the Contractor breaches the confidentiality clause, what is the chain of obligations and remedies that the termination clause then triggers?"*

Watch what each retriever does:

- **Flat hybrid retrieval** embeds the question and returns the chunks most *similar* to it — probably the confidentiality clause and the termination clause themselves, because their text is similar to the query. But the *answer* — the **chain** confidentiality breach → triggers termination right → which invokes the remedies clause → which references the liability cap — lives in the *relationships between* those clauses, not in any single chunk's text. Flat retrieval hands the LLM the endpoints and hopes it infers the chain; if an intermediate clause (the remedies one) wasn't textually similar to the query, it's simply *absent* from the context, and the answer is silently incomplete.
- **GraphRAG** starts at the `confidentiality clause` entity and **traverses** the extracted relationships: `confidentiality —breach-triggers→ termination —invokes→ remedies —limited-by→ liability cap`. It follows the edges the extraction pass found, so the *intermediate* clause that flat retrieval missed is pulled in *because it's on the path*, not because it's textually similar. The multi-hop chain is the graph's home turf, and this question is the one sentence that justifies the build cost.

That single question — found, not assumed — is the deliverable of the GraphRAG stretch goal: build the tiny graph, find the multi-hop or global question hybrid retrieval misses, and you've *earned* the pattern by measuring where it wins, exactly as the week measures everything else.

---

## Part 4 — Agentic RAG: the agent chooses the retriever

The final pattern, and the bridge to Phase III. So far the retrieval *pipeline is fixed*: every query goes dense + BM25 + rerank, every time. **Agentic RAG** makes the *retrieval strategy a decision the agent makes per query*:

- **Which retriever?** A factual lookup goes to the vector store; a "list all clauses about X" goes to a metadata filter or BM25; a multi-hop question goes to GraphRAG; a "what's 2+2" *skips retrieval entirely.*
- **Which store/collection?** A multi-tenant agent picks the tenant's collection; a multi-corpus agent picks the right corpus.
- **Retrieve at all?** The biggest win is often the agent deciding *not* to retrieve — answering from its own knowledge or a tool call when retrieval would add latency and noise for no benefit.
- **Iterate?** The agent retrieves, judges whether it has enough, and retrieves *again* with a refined query if not (this connects to query rewriting / HyDE from week 9).

```python
# Agentic RAG, sketched: the agent ROUTES the query to a retriever (or none).
def agentic_retrieve(query: str, router_llm, retrievers: dict) -> list[str]:
    choice = router_llm.classify(query, options=[
        "vector",      # semantic lookup -> the vector store
        "filter",      # "list all X" -> metadata filter / BM25
        "graph",       # multi-hop / global -> GraphRAG
        "none",        # answerable without retrieval -> skip it
    ])
    if choice == "none":
        return []                       # the cheapest, often-best move
    return retrievers[choice](query)
```

The honest trade-off, in the C23 spirit:

- **The lift is real on heterogeneous query distributions** — when your queries are a *mix* of factual, list, multi-hop, and trivial, routing each to the right retriever beats forcing them all through one fixed pipeline.
- **The cost is a routing call (latency + tokens + a new failure mode)** — the router can mis-route, and now you have an extra LLM call per query and a classifier to evaluate. On a *homogeneous* query distribution (all factual lookups), the fixed pipeline wins because the router adds cost for a decision that's always the same.
- **Measure the lift, like everything.** The discipline is identical to the chunking A/B and the store bakeoff: run the agentic router *against* the fixed pipeline on your gold set, and ship the router only if it earns its complexity with a measured Recall/faithfulness gain that justifies the added latency and cost. (The stretch goal builds exactly this.)

Agentic RAG is where Phase II (retrieval) hands off to Phase III (agents): the retriever stops being a function you call and becomes a *tool the agent reasons about*. You'll build the full version when you have the agent graph (week 13) and the MCP tool surface (week 15); this week you meet the pattern and measure a simple router.

### 4.1 Evaluating the router — measuring lift, and catching mis-routing

"Measure the lift" was the mantra above; here is *how*, because a router you can't evaluate is a router you can't justify. The evaluation has the same shape as every A/B in the course — fixed everything except the one variable (here, router vs fixed pipeline) — with two twists specific to routing.

**Twist one: the query set must be heterogeneous, or the test is rigged.** The router's *entire* value proposition is handling a *mix* of query types (factual, list, multi-hop, trivial). Evaluate it on a homogeneous gold set (all factual lookups, like the week-7 set) and it can only *lose* — it adds a routing call to a decision that's always "vector," so you'd measure pure overhead and wrongly conclude routing is bad. So you build a **held-out heterogeneous query set** that deliberately spans the categories, with known-good answers, and you measure on *that*. The set design *is* the experiment; a sloppy set gives a meaningless number.

**Twist two: you measure a delta on two axes at once — quality up, cost up — and decide if the trade is worth it.**

```text
                       fixed pipeline      agentic router      delta
Recall@5 (overall)         0.71                0.84           +0.13   <- the lift
faithfulness               0.79                0.88           +0.09   <- fewer wrong-tool answers
p95 latency               210 ms              320 ms          +110 ms <- the routing-call cost
cost / query              $0.004              $0.006          +$0.002 <- extra LLM classify call
```

Read it like the store scorecard (Lecture 1 §5): the router earns its place only if the **quality lift justifies the added latency and cost** *for your workload*. A +0.13 Recall on a heterogeneous distribution where mis-retrieval is expensive (legal, medical) easily pays a +110 ms tail and a fraction of a cent; the same lift on a latency-critical, homogeneous workload does not. The number doesn't decide for you — it makes the trade *honest*, which is the whole discipline.

**The router's own failure mode — mis-routing — and how to catch it.** The router is a classifier, and classifiers are wrong sometimes. A mis-route sends a multi-hop question to the flat vector retriever (which returns the endpoints but misses the chain, §3.2) or sends a trivial "what's 2+2" through full GraphRAG (wasted latency and cost). Mis-routing is insidious because, like the silent filtered-recall collapse (Lecture 1 §3), **it produces a plausible-looking answer, not an error** — the wrong retriever still returns *something*. You catch it two ways:

- **Route-level accuracy.** Label the held-out set with the *correct* retriever per query and measure the router as a plain classifier: a confusion matrix over `{vector, filter, graph, none}`. Now you see *which* mis-routes happen — and the off-diagonal cells tell you where the router is weak (e.g., it keeps sending multi-hop questions to `vector`), which is a fixable prompt/few-shot problem.
- **End-to-end faithfulness as the backstop.** Route accuracy alone isn't enough (a "wrong" route that still gets the answer is fine; a "right" route that fails isn't), so you *also* watch end-to-end faithfulness on the same set. The confusion matrix tells you *why* the router is wrong; the faithfulness delta tells you whether it *matters*. Watch both, ship on both.

```python
# Router evaluation: lift vs fixed pipeline, AND route-level accuracy.
def eval_router(gold, router_retrieve, fixed_retrieve, route_labels) -> dict:
    router = evaluate(gold, router_retrieve, k=5)      # quality WITH routing
    fixed  = evaluate(gold, fixed_retrieve,  k=5)      # quality WITHOUT routing
    # route_labels: the correct retriever per query -> router-as-classifier
    correct = sum(router_choice(q) == lbl for q, lbl in route_labels.items())
    return {
        "recall_lift":  router["Recall@k"] - fixed["Recall@k"],
        "route_accuracy": correct / len(route_labels),     # catches mis-routing
        # latency/cost deltas measured alongside, per the scorecard
    }
```

> **A router is a classifier you must evaluate as one.** Measure the *lift* (Recall/faithfulness delta) on a *heterogeneous* held-out set to justify the routing call's latency and cost, *and* measure route-level accuracy (a confusion matrix) to catch mis-routing — which, like the filtered-recall collapse, fails silently by returning a plausible answer from the wrong retriever. Ship the router only when the lift is real and the mis-route rate is low.

---

## Part 5 — Putting it together: the architecture memo

The week converges on the memo you'll defend at the week-12 architecture review. Given your workload, you now reason:

1. **What's the workload?** Multi-tenant with heavy filtering (→ Qdrant or a well-indexed pgvector)? Already-on-Postgres with light filtering (→ pgvector)? Generative/graph-relational (→ Weaviate)? Billion-scale (→ Milvus)? Prototype (→ Chroma)?
2. **What's the recovery story you can live with?** Measure time-to-recover in the drill; weight it heavily.
3. **Does filtered search hold recall at your selectivity?** Measure recall *at a selective filter*, not just unfiltered latency.
4. **Do you have multi-hop/global questions?** If yes, plan GraphRAG *alongside* flat retrieval, and budget the build cost.
5. **Is your query distribution heterogeneous enough to justify agentic routing?** Measure the lift before you ship the router.

Answer those with *measured numbers* — ingest, p95, filtered-recall, time-to-recover, and the GraphRAG/agentic lift where relevant — and you've done the engineering the week teaches: not "I used a vector store," but "I chose *this* store for *this* workload because it recovers in 47 seconds, holds recall at my tenant filter, and my team already operates Postgres-shaped systems — and here are the numbers." The index survived the loss, and you can prove it came back.

---

## Part 6 — Recap

You should now be able to:

- **Operate a vector store like a database**: back it up (pgvector = Postgres's story your team knows; Qdrant = fast snapshots; Weaviate = backup module), and distinguish a cheap schema change (add a metadata index) from an expensive one (change the embedding → full re-ingest).
- **Choose an HA topology with the reasoning**: primary + replicas (read scaling + failover), sharding (capacity), multi-AZ (zone-failure survival) — and know that **HA and backup solve different failures**: replication gives failover in tens of seconds but faithfully copies a *corrupt* index, so it does not save you from the corruption scenario the recovery drill targets.
- **Monitor the store on the right axes**: query latency p95/p99, **recall drift** (a scheduled canary gold set — the AI-specific metric almost nobody has), ingest lag, index memory and disk headroom — and read the degradation signature where **recall drops one or two stages before latency or errors do**, which is the early warning that bridges to week-18 observability and the week-24 chaos drill.
- **Estimate recovery time before the drill confirms it**: snapshot-restore is `snapshot_size ÷ disk_bandwidth` (seconds-to-minutes); re-embed-from-source is `corpus_size ÷ embedding_throughput` plus the index build (hours) — a 50–100× gap for a 5M-vector / 10 GB corpus, decided entirely by whether you have a real snapshot.
- **Run the index-loss recovery drill**: baseline → backup → destroy → restore (timed) → record time-to-recover, and understand why that number *reorders* the stores from how query latency ranked them, and why "we'll re-embed" is an outage.
- **Explain GraphRAG and cost it**: extract an entity/relationship graph, build community summaries, retrieve over the graph for multi-hop and global questions flat retrieval can't answer — counting the corpus-sized extraction pass vs the smaller summarization pass (claude-sonnet-4-6), deciding **full vs incremental rebuild** by the corpus's churn rate, and naming the one legal-corpus multi-hop question that justifies the whole build.
- **Describe and evaluate agentic RAG**: the agent chooses the retriever/store/strategy (or skips retrieval); measure the **lift** (Recall/faithfulness delta) on a *heterogeneous* held-out set against added latency and cost, and catch the router's own failure mode — **mis-routing**, which fails silently with a plausible answer — via a route-level confusion matrix plus end-to-end faithfulness.
- **Write the architecture memo**: pick a store for a stated workload on operational criteria — recovery time, filtered-recall, ingest, familiarity — defended with measured numbers, not leaderboard QPS.

Next: the exercises put this on real stores — bring up three, measure filtered ANN where it breaks, and run the recovery drill that produces the time-to-recover number. Continue to [the exercises](../exercises/README.md).

---

## References

- *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — Edge et al., Microsoft, 2024: <https://arxiv.org/abs/2404.16130>
- *Microsoft GraphRAG (implementation)*: <https://github.com/microsoft/graphrag>
- *Qdrant — snapshots* (the fast recovery story): <https://qdrant.tech/documentation/concepts/snapshots/>
- *PostgreSQL backup & restore* (pgvector's recovery story): <https://www.postgresql.org/docs/current/backup.html>
- *Weaviate — backups*: <https://weaviate.io/developers/weaviate/configuration/backups>
- *Qdrant — distributed deployment* (sharding + replication; the HA topology): <https://qdrant.tech/documentation/guides/distributed_deployment/>
- *Patroni* (Postgres HA / automatic failover — pgvector's HA story): <https://github.com/patroni/patroni>
- *Microsoft GraphRAG — incremental indexing* (the full-vs-incremental rebuild question): <https://microsoft.github.io/graphrag/index/overview/>
- *LlamaIndex — router / agentic retrieval*: <https://docs.llamaindex.ai/en/stable/module_guides/querying/router/>
- *RAGAS — faithfulness & retrieval metrics* (the quality axes the router is measured on): <https://docs.ragas.io/>
- *Anthropic — Claude models* (claude-sonnet-4-6 for community summarization): <https://docs.anthropic.com/en/docs/about-claude/models>
- *Anthropic — Contextual Retrieval*: <https://www.anthropic.com/news/contextual-retrieval>
