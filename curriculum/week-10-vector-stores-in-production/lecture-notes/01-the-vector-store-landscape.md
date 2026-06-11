# Lecture 1 — The Vector-Store Landscape, Filtered ANN, and the Criteria That Actually Matter

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can place the five 2026 vector stores (pgvector, Qdrant, Weaviate, Milvus, Chroma) by the workload and operational story each fits, explain why **filtered ANN** is the problem where stores actually differ (pre-filter vs post-filter vs native in-filter), reason about metadata/payload indexes, and name the selection criteria that decide a production choice — which are mostly *not* the leaderboard QPS.

If you remember one sentence from this entire week, remember this one:

> **Pick the vector store with the operational story you can live with at 2 AM, not the one with the best benchmark.** The benchmark is the vendor's good day; the 2 AM story — backup, restore, replication, the rebuild after a schema change — is your bad day.

There's a corollary you should tape next to last week's mantras:

> **A vector store is a database first and a vector index second.** It has backups, replicas, migrations, and failure modes like any database. "Vector" is one index type; "store" is everything that keeps your data alive.

For three weeks the store was a given — pgvector in a container, `<=>`, an HNSW index. This week it becomes a decision. Everything that follows is in service of one shift: from "the store is where I put the vectors" to "the store is a database I operate, chosen for how it behaves when things go wrong."

A quick orientation to the lecture's arc, so you know where each idea lands: §1–§2 frame the choice and place the five stores; §3–§4 cover the technical heart (filtered ANN and metadata indexes — where stores actually differ); §5 names the criteria that *decide* a choice; §6 gives the index internals you need to reason about operations; §7 walks a worked decision; §8–§11 cover the adapter, the failure modes, when *not* to use a vector store, and the store-level compression levers. The operations half — backup, restore, recovery, GraphRAG, agentic RAG — is Lecture 2. Read this one for *which store and why*; read the next for *how to keep it alive*.

---

## 1. Why the store is a real decision (and why it's usually pgvector anyway)

Start with the honest baseline: **for most teams, most of the time, pgvector is the right answer, and the interesting question is when it isn't.**

pgvector is a Postgres extension. That one fact carries enormous operational weight: your team already runs Postgres, already backs it up, already has monitoring, replication, point-in-time recovery, access control, and a decade of muscle memory for it. Adding vectors is `CREATE EXTENSION vector;` and a new column. You don't add a *new system to operate* — you add a feature to one you already operate. The syllabus calls pgvector "the default" for exactly this reason: the lowest-operational-surprise choice is the one that's already in your stack.

So when *isn't* it pgvector? When one of these pressures shows up and pgvector strains:

- **Filtered search at scale** — heavy multi-tenant or faceted filtering where the metadata filter has to interact tightly with the ANN traversal (§3). This is Qdrant's home turf.
- **Billion-scale vectors** — when you outgrow what a single Postgres can index comfortably and need horizontal scale with separate query/index/data tiers. This is Milvus's home turf.
- **Generative/graph-leaning retrieval** — when you want the store to do cross-references, generative search, or graph-style relationships natively. This is Weaviate's lean.
- **Zero-config prototyping** — when you want an embedded store with no server to run at all. This is Chroma's ergonomics.

The decision, then, isn't "which store is best" (a meaningless question) but "does my workload push hard enough on filtering, scale, graph, or ergonomics to justify operating a *second* specialized system instead of the Postgres I already run?" Most of the time the answer is no, and that's fine — the senior move is knowing *which pressure* would flip it, and being able to measure whether you're actually under that pressure.

> **The framing:** start at pgvector (lowest operational surprise). Move off it only when a *specific, measured* pressure — filtering, scale, graph, or ergonomics — makes operating a second system worth it.

---

## 2. The five stores, placed by workload

Here is the map. For each, the one-line identity, the workload it wins, and the operational story (which is what you actually live with).

### 2.1 pgvector — the Postgres-native default

- **Identity:** a Postgres extension adding a `vector` type and HNSW/IVFFlat indexes.
- **Wins on:** "I already run Postgres." Transactional consistency with your *other* data (you can `JOIN` vectors against your relational tables — the user, the document, the permissions — in one query). The operational story is *Postgres's* story: `pg_dump`/`pg_restore`, streaming replication, PITR, your existing monitoring. That's the killer feature — not speed, *familiarity*.
- **Strains on:** very high-selectivity filtered search (filtering an HNSW index in Postgres has historically meant trade-offs — iterative scans, or filter-then-search), and billion-scale (one Postgres is one Postgres).
- **Operational story:** the best of the five for most teams, *because it's the one you already operate.* Recovery is `pg_restore`. Your DBA already knows it.

### 2.2 Qdrant — the filtered-ANN specialist

- **Identity:** a Rust vector database purpose-built for fast *filtered* vector search.
- **Wins on:** filtered ANN (§3). Qdrant's filterable HNSW combines the metadata filter *with* the graph traversal, so `vector search WHERE tenant='acme' AND clause_type='termination'` stays fast and keeps recall even when the filter is selective. Multi-tenant SaaS retrieval is its sweet spot. It's also fast on raw ANN and pleasant to operate (snapshots for backup, clean distributed mode).
- **Strains on:** "I don't want a second database." It's another system to run, back up, and monitor — justified when filtering is your bottleneck, overhead when it isn't.
- **Operational story:** strong. **Snapshots** make backup/restore clean and *fast* (which is why the recovery drill loves it); distributed deployment gives sharding + replication for HA.

### 2.3 Weaviate — generative and graph-leaning

- **Identity:** a schema-first vector database with generative search and cross-references (graph-style relationships between objects).
- **Wins on:** workloads where the store does more than store — built-in generative search (retrieve *and* generate in one call via modules), cross-references between objects (a `Clause` that references its `Contract`), and a graph lean that suits relationship-rich corpora.
- **Strains on:** simplicity — the schema-first model and module system are more to learn than "put vectors in a table."
- **Operational story:** good, with a backup module (filesystem/S3/GCS) and replication; heavier conceptual surface than pgvector or Qdrant.

### 2.4 Milvus — billion-scale

- **Identity:** a distributed vector database with separated **query / data / index** nodes, built for massive scale.
- **Wins on:** billions of vectors, high ingest, horizontal scale. The separated-node architecture lets you scale the part that's the bottleneck (more query nodes for QPS, more data nodes for capacity) independently.
- **Strains on:** small corpora — standing up Milvus (etcd + object storage + the Milvus nodes) for a 50-clause corpus is using a freight train to deliver a letter. The operational surface is large.
- **Operational story:** powerful but heavy; you take on a distributed system. Right when you're actually at scale, overkill when you're not.

### 2.5 Chroma — developer ergonomics

- **Identity:** an embedded/lightweight vector store optimized for developer experience.
- **Wins on:** prototyping, notebooks, small apps, "I want a vector store with `pip install` and zero server." `chromadb.PersistentClient` and you're storing vectors in three lines.
- **Strains on:** scale and production operations — it's the prototyping tool, not the billion-scale serving tier. (It has grown server and cloud options; the *ergonomics* identity is the durable one.)
- **Operational story:** minimal because there's minimal to operate — which is the point at prototyping scale and the limit at production scale.

> **The placement table (commit this):**
> - **pgvector** → you already run Postgres; want transactional joins; lowest operational surprise. *The default.*
> - **Qdrant** → filtered/multi-tenant search is your bottleneck; you'll pay to run a second system for it.
> - **Weaviate** → you want generative search and graph-style cross-references in the store.
> - **Milvus** → you are actually at billion-scale and need independent horizontal scaling.
> - **Chroma** → prototyping, notebooks, zero-config; not your production serving tier.

You're not asked to run all five this week — you'll run *three* (pgvector, Qdrant, Weaviate) deeply and place Milvus and Chroma correctly. An engineer who can say "this is a multi-tenant SaaS with heavy per-customer filtering, so Qdrant; or stay on pgvector if the filtering is light and I value the Postgres joins" is reasoning about the landscape, not reciting it.

---

## 3. Filtered ANN — the problem where stores actually differ

Here is the most important technical idea in the week, because it's where the stores genuinely diverge and where the wrong choice silently breaks production. Pure ANN — "find the 10 vectors closest to this query" — every store does well. The hard problem is **filtered ANN**: "find the 10 closest vectors *where* `tenant='acme'` *and* `clause_type='termination'`." Combine a vector search with a metadata filter and you've entered the territory that separates a toy from a production store.

Why is it hard? The ANN index (HNSW) is a graph built over *all* the vectors, optimized to find nearest neighbors *globally*. A filter says "but only consider this subset." There are three ways to reconcile them, each with a failure mode:

### 3.1 Post-filter — ANN first, then drop

Run the ANN search over everything to get the top-K, *then* discard the results that fail the filter. Simple, and fast when the filter is *broad* (most results pass). But when the filter is **selective** — say `tenant='acme'` matches 0.1% of vectors — the top-K from the global ANN search might contain *zero* acme vectors, because acme's vectors aren't in the global top-K. You filter them all out and return **nothing**, even though relevant acme vectors exist further down the ranking. This is the **recall collapse on selective filters**, and it's a brutal, silent production bug: the search "works" (no error) and returns too few or no results for exactly the tenants whose data is rare.

### 3.2 Pre-filter — filter first, then ANN

Apply the metadata filter first to get the matching subset, *then* do nearest-neighbor search within that subset. This is *exact* with respect to the filter (you only ever see acme's vectors). But if you can't search the ANN index restricted to a subset, you fall back to a brute-force scan of the subset — fine when the subset is small, slow when the filter is broad (a broad filter leaves millions of vectors to scan without the index's help). Pre-filter trades the recall-collapse risk for a potential latency cost on broad filters.

### 3.3 Native / in-filter — filter *during* traversal

The production answer: the store filters *while traversing* the ANN graph, only visiting nodes that pass the filter, so it gets the index's speed *and* the filter's exactness. This is **Qdrant's filterable HNSW** — it integrates the payload filter into the HNSW search itself, so a selective filter doesn't collapse recall *and* a broad filter doesn't lose the index. pgvector and Weaviate have their own approaches to filtered HNSW (pgvector's iterative index scans, Weaviate's filtered search), with their own trade-offs. **This is the single dimension on which the stores most differ, and it's why "Qdrant for filtered search" is the placement.**

> **The lesson, and the trap:** filtered ANN's behavior depends on **filter selectivity**, and the failure mode (post-filter recall collapse) is *silent*. You only catch it by *measuring recall at a selective filter*, not just latency. Exercise 2 does exactly this: it shows post-filter's recall collapsing on a high-selectivity filter while native filtering holds — the number that picks Qdrant over a naive post-filter, or tells you pgvector's filtered scan is good enough for your selectivity.

One more nuance the bakeoff exposes: selectivity isn't fixed — it *varies by tenant*. Your biggest customer has 30% of the data (a broad filter, post-filter is fine); your smallest has 0.01% (a selective filter, post-filter collapses). So a single "filtered-recall" number is misleading; you measure recall *across the selectivity range*, and the store you ship is the one that holds recall at your *worst* (most selective) tenant — because that's the tenant whose retrieval silently breaks first. "Works for the demo tenant" is precisely the bug; "holds recall at the 0.01% tenant" is the bar.

---

## 4. Metadata indexes — as important as the vector index

A filter is only fast if the field it filters on is *indexed*. This is obvious in relational databases (you index the column in the `WHERE`) and just as true in vector stores, where it's often overlooked because all the attention goes to the vector index.

Every production vector store lets you index the **metadata** (Qdrant calls it the *payload index*, Weaviate indexes properties, pgvector uses ordinary Postgres indexes on the metadata columns). Without it, a filter on `tenant` scans every row's metadata; with it, the store jumps to the matching rows. For multi-tenant retrieval — where *every* query filters by tenant — the metadata index isn't optional, it's the difference between a query that scales and one that degrades linearly with the number of tenants.

The design implication for your schema: decide *what you'll filter on* before you ingest, and index those fields. In this week's corpus we add `tenant`, `clause_type`, and `version` as filterable metadata precisely so the exercises have realistic filters — and so you practice indexing them. A vector store with a great HNSW index and *no* metadata index is half a production store; the filtered queries (which is most real queries — they're scoped to a tenant, a document set, a time range) will be the slow path.

---

## 5. The selection criteria that actually matter

Now the payoff of §1's framing. When you pick a store, the criteria that *decide* it are mostly not the leaderboard QPS. Rank them like a production engineer:

1. **Operational familiarity.** Can your team operate it at 2 AM? pgvector wins here for most teams because it's Postgres. A 2-ms-faster store you can't restore confidently is the worse choice. (This is the week's mantra.)
2. **Recovery story.** How fast from "index gone" to "Recall@5 back to baseline"? Snapshot-restore (Qdrant) vs `pg_restore` (pgvector) vs re-embed-from-scratch (the disaster). You'll *measure* this in the recovery drill (Lecture 2, Exercise 3). It's a first-class criterion, not an afterthought.
3. **Filtered-search behavior at your selectivity.** Does it hold recall on your most selective filter? (§3.) Measure recall *at a filter*, not just unfiltered.
4. **Ingest throughput.** How fast can you load (and reload) the corpus? This bounds your recovery time *and* your ability to re-index after a model change.
5. **Query latency (p50/p95).** Real, but usually *not* decisive between well-tuned stores — and the one everyone over-weights. Report it, but don't let it dominate the others.
6. **Config complexity.** Lines of config, time-to-first-query, conceptual surface. A store you can stand up in ten lines and reason about beats one that needs a distributed-systems PhD — unless you're at the scale that justifies the PhD.

The output of the week is a memo that scores the stores on *these* criteria, weighted for *your* workload — not a single "winner" on a single number. That's the architecture-review skill: defending a store choice on the axes that matter operationally.

> **The discipline:** never pick a store on query latency alone. Score it on recovery time, filtered-search recall, ingest, config complexity, and — above all — whether your team can operate it. The bakeoff measures all of these; the memo weights them for your workload.

A concrete scorecard makes the weighting honest. Here's the shape the bakeoff produces and the memo defends — weights chosen for the multi-tenant SaaS workload of §7 (filtering and recovery matter most; query latency least):

| Criterion | Weight | pgvector | Qdrant | Weaviate |
|---|---:|---|---|---|
| Operational familiarity | 30% | high (Postgres) | medium | medium |
| Recovery time | 25% | `pg_restore` (measure) | snapshot (fast) | backup module |
| Filtered-recall @ selectivity | 20% | measure | strong | measure |
| Ingest throughput | 10% | measure | measure | measure |
| Query p95 | 10% | measure | measure | measure |
| Config complexity | 5% | low | low | medium |

The cells that say "measure" are the ones the bakeoff fills with *your* numbers — and the weights are the *judgment* you defend at the review. A reviewer won't argue with a measured filtered-recall of 0.86; they'll argue with a 5% weight on recovery when your SLA says "recover in minutes." The scorecard forces both the measurement and the weighting into the open, which is exactly what an architecture review is for. Change the workload — a research prototype, a billion-vector index — and the weights shift (familiarity drops, scale rises), and a different store wins. That's not the scorecard being arbitrary; that's it being *workload-specific*, which is the whole point.

---

## 6. The ANN index under the hood — HNSW, IVF, and the knobs that matter

Before you can reason about *operations*, you need a working model of the index every store is built on, because the index is what you back up, rebuild, and tune. You met HNSW in week 7; here's the operational view.

**HNSW (Hierarchical Navigable Small World)** is the default ANN index in pgvector, Qdrant, and Weaviate. It's a *graph*: each vector is a node, connected to a handful of nearby neighbors, with a hierarchy of layers (sparse at the top, dense at the bottom) that lets a search "zoom in" — start at the top layer, greedily walk toward the query, drop a layer, repeat. The result is logarithmic-ish search instead of scanning every vector. Three knobs control the speed/recall/memory trade, and you should know them because they're what you tune when a store is too slow or too inaccurate:

- **`m`** — the number of neighbors per node (the graph's connectivity). Higher `m` → better recall, more memory, slower build. Typical: 16. This is fixed at build time; changing it rebuilds the index.
- **`ef_construction`** — how hard the build works to find good neighbors. Higher → better-quality graph, slower build. Typical: 64–200. Build-time only.
- **`ef_search`** (a.k.a. `ef`) — how many candidates the *search* explores. Higher → better recall, slower query. This is the **query-time** knob you tune live (you swept it in week 7): raise it until recall plateaus, then stop. It's the elbow you find on the recall-vs-`ef_search` curve.

**IVF (Inverted File)** is the other common index: partition the vectors into clusters (Voronoi cells), and at query time search only the few clusters nearest the query (the `nprobe` parameter). It's faster to build than HNSW and uses less memory, but its recall is more sensitive to the data distribution and the `nprobe` setting. pgvector offers both (`ivfflat` and `hnsw`); HNSW usually wins on recall, IVF on build speed and memory. The operational implication: **IVF needs the data present before you build it** (it clusters the actual vectors), so an IVF index built on an empty table is useless — a subtle ingest-order gotcha.

Why this matters for the week's operations:

- **The recovery time includes the index build.** Restoring the *vectors* is fast; rebuilding the *HNSW graph* over them can be slow (it's `O(n · ef_construction · log n)`-ish). A snapshot that includes the *built index* recovers faster than one that restores vectors and rebuilds (Lecture 2 §2). When you measure time-to-recover, you're measuring the build, not just the copy.
- **A schema change that touches the index is expensive.** Changing `m` or the embedding dimension means a full rebuild (Lecture 2 §1.3). Changing `ef_search` is free (query-time). Knowing which is which tells you whether a "tuning change" is a deploy or a re-ingest.
- **The index's memory footprint is real.** HNSW keeps the graph in memory; at scale, the graph itself (not just the vectors) sizes your RAM. This is part of why Milvus separates index nodes — so you can scale index memory independently.

> **The operational model:** the store is a database wrapped around an ANN index (usually HNSW). `ef_search` is your live recall knob; `m`/`ef_construction`/dimension are build-time and changing them rebuilds; the rebuild time *is* a chunk of your recovery time. You don't need to implement HNSW — you need to know which knob is free and which one is a re-ingest.

---

## 7. A worked store decision — reasoning, not reciting

To make §1–§6 concrete, walk a realistic decision the way you'd defend it at an architecture review. The scenario: a multi-tenant SaaS product, ~5 million chunks across ~2,000 tenants, every query scoped to one tenant, a team that already runs Postgres, and a hard requirement that retrieval recover within minutes of an index loss.

Reason through the criteria (§5), in order:

1. **Operational familiarity.** The team runs Postgres. That's a strong pull toward **pgvector** — they already back it up, monitor it, and can restore it at 2 AM. Starting anywhere else means operating a second system; you need a *measured* reason to pay that.
2. **Filtered-search behavior.** Every query filters by `tenant`, and 2,000 tenants means *selective* filters (each tenant is ~0.05% of the data). This is exactly where post-filter recall collapses (§3) and where Qdrant's filterable HNSW shines. The question becomes: does pgvector's filtered HNSW *hold recall* at this selectivity on *your* data? You **measure it** (Exercise 2's filtered-recall test at a selective filter). If pgvector holds recall (its iterative-scan filtering has improved a lot), stay — the Postgres familiarity wins. If it sags, **Qdrant** earns the second-system cost because filtering is your bottleneck.
3. **Recovery story.** "Recover within minutes" — measure time-to-recover for the candidate(s) (the recovery drill). pgvector's `pg_restore` of 5M vectors + HNSW rebuild might be minutes; Qdrant's snapshot restore might be seconds. If pgvector's recovery is within budget, the familiarity still wins; if the rebuild blows the budget, that's a point for Qdrant's snapshots.
4. **Ingest & scale.** 5M chunks is comfortably within a single Postgres's reach (it's not billion-scale), so **Milvus is overkill** — you'd be operating a distributed system for a single-node workload. Cross it off.
5. **The decision.** The honest answer is *"measure the filtered-recall and recovery-time on pgvector first; if both are within budget, ship pgvector for the familiarity and the transactional tenant joins; if filtered-recall sags or recovery is too slow, move the retrieval tier to Qdrant for the filterable HNSW and fast snapshots, accepting the second system."*

Notice what that decision *is*: not "Qdrant is best" or "pgvector is best," but a *measured*, workload-specific choice on the operational axes, with the leaderboard QPS nowhere in it. That's the architecture-review skill — and it's exactly what the bakeoff and the memo train. An engineer who can walk this reasoning, with the filtered-recall and recovery numbers to back each step, is the one whose store choice survives the review *and* the 2 AM page.

---

## 8. The one-pipeline-three-stores discipline

The exercises and the bakeoff hold the **pipeline fixed** (your week-9 hybrid retrieval: dense + BM25 + reranker, the same embedding, the same chunking-A/B winner) and vary only the *store*. This is the same one-variable-at-a-time discipline that runs through all of C23 — the chunking A/B in week 8, the engine bakeoff in week 6. The reason is identical: if you change the store *and* the embedding *and* the chunking, the Recall@5 delta could be anything, and you've learned nothing you can defend at the architecture review.

So: keep the corpus, chunking, embedding, and `evaluate()` from weeks 7–9 *unchanged*. Implement a thin **store adapter** with one interface (`create`, `upsert`, `search`, `search_filtered`, `snapshot`, `restore`) and one implementation per store. Now the only thing that differs across runs is the store behind the adapter, and the numbers — Recall@5, ingest throughput, p95, recovery time — mean something.

```python
# The adapter interface the whole week is built on. One interface, three stores.
class VectorStore:
    def create(self, name: str, dim: int) -> None: ...
    def upsert(self, name: str, rows: list[tuple[str, list[float], dict]]) -> None: ...
    def search(self, name: str, qv: list[float], k: int) -> list[str]: ...
    def search_filtered(self, name, qv, k, where: dict) -> list[str]: ...  # filtered ANN
    def snapshot(self, name: str) -> str: ...      # backup -> handle
    def restore(self, name: str, handle: str) -> None: ...   # the recovery story
```

That interface is the whole bakeoff, abstracted. `evaluate(gold, retrieve_fn, k)` from week 7 doesn't change; you just give it a `retrieve_fn` backed by a different adapter. The "it survived the index loss" promise lives in the `snapshot`/`restore` pair: the store that restores fastest is the store that came back before the customer hung up.

A subtlety the adapter must get right: the **distance metric must match the embedding's training**. BGE-large is trained for cosine similarity on normalized vectors, so every adapter must use the store's *cosine* operator (`vector_cosine_ops` in pgvector, `Distance.COSINE` in Qdrant, `cosine` in Weaviate) and the vectors must be normalized at upsert. If one adapter uses Euclidean distance by accident, its unfiltered Recall@5 will diverge from the others — and that divergence is *not* a store-quality difference, it's an adapter bug. The bakeoff's first sanity check is that all three stores agree on *unfiltered* recall (Exercise 1); they should, because the embedding and the metric are the same. When they don't, you've found a metric mismatch, not a better store.

---

## 8. Common failure modes — what actually breaks in production

The syllabus is built to prevent specific failure modes. Here are the vector-store ones, named so you recognize them before they page you:

- **The silent filtered-recall collapse (§3).** Post-filter on a selective tenant returns too few results, *with no error*. The product "works" in the demo (the demo tenant has lots of data) and quietly returns nothing for your smallest customers in production. Caught only by measuring recall *at a selective filter*. This is the most dangerous one because it's invisible until a customer complains.
- **The forgotten metadata index (§4).** Filtered queries are correct but *slow*, and they get slower as tenants accumulate, because the filter scans every row's metadata. Caught by watching filtered-query latency as the data grows, and fixed by indexing the filterable fields *before* ingest.
- **The "we'll re-embed" non-backup (Lecture 2).** A team treats "we can regenerate the vectors" as a backup, never runs the restore, and discovers on a bad night that recovery is a four-hour re-embed during which retrieval is down. Caught by *running the recovery drill* before the bad night.
- **The embedding-swap re-ingest surprise (Lecture 2).** Someone upgrades the embedding model expecting a config change and discovers it's a full re-embed-and-rebuild of the entire index, because the old vectors are meaningless in the new model's space. Caught by knowing the embedding is the *long-lived* decision (week 7) and planning a parallel-index cutover.
- **The single-Postgres scale wall.** A team rides pgvector past the point where a single Postgres can index the vectors comfortably and hits a memory or latency wall, then scrambles to migrate to Qdrant or Milvus under pressure. Caught by knowing the *pressure* (§1) that flips the decision and watching for it — the migration is much calmer when planned than when forced.

Every one of these is prevented by the week's discipline: measure the operational axes (filtered-recall at selectivity, recovery time, ingest, scale headroom), not just the demo-day query latency. The store that looks great in the demo and falls over at 2 AM is the store you didn't measure on the axes that matter.

---

## 9. The same query, three stores — what the adapter hides

To ground the abstraction, here's the *same* filtered query — "the 5 nearest chunks where `tenant='acme'`" — expressed in each store's native client. The point is to see how different the surfaces are, which is exactly why the adapter (§8) earns its keep: `ab.py` and `evaluate()` never see any of this.

**pgvector** — it's SQL. The vector search is `ORDER BY embedding <=> %s`, the filter is a `WHERE`, and both ride the same Postgres query planner:

```python
rows = conn.execute(
    "SELECT chunk_id FROM clauses "
    "WHERE meta->>'tenant' = %(tenant)s "         # the filter (GIN-indexed)
    "ORDER BY embedding <=> %(qv)s "              # cosine distance
    "LIMIT %(k)s",
    {"tenant": "acme", "qv": str(query_vec), "k": 5},
).fetchall()
```

**Qdrant** — it's a structured filter object passed *into* the search, so the filterable HNSW applies it during traversal (§3.3):

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue
hits = client.query_points(
    "clauses", query=query_vec, limit=5,
    query_filter=Filter(must=[                     # filtered DURING the ANN walk
        FieldCondition(key="tenant", match=MatchValue(value="acme"))]),
).points
```

**Weaviate** — it's a fluent query builder with a `where` clause, schema-first:

```python
clauses = client.collections.get("Clauses")
res = clauses.query.near_vector(
    near_vector=query_vec, limit=5,
    filters=Filter.by_property("tenant").equal("acme"),   # the filter
)
```

Three completely different surfaces — SQL, a filter object, a query builder — for the *same* retrieval. Without an adapter, every piece of code that retrieves would be store-specific, and swapping stores would mean rewriting the pipeline (and the migration that §8's failure mode warns about would be a rewrite, not a config change). With the adapter, `search_filtered(name, qv, k, where={"tenant": "acme"})` is the only thing the rest of the code knows, and the three implementations above live behind it. That's why the *first* deliverable of the mini-project is the adapter: it's the seam that makes the store a swappable decision instead of a load-bearing assumption baked into a thousand call sites.

And note the asymmetry the adapter must *expose*, not hide: the recovery story (`snapshot`/`restore`) is genuinely different per store (Qdrant's snapshot API vs pgvector's `pg_dump` vs Weaviate's backup module), and the *whole point* of the week is that this difference is a first-class selection criterion. The adapter unifies the *query* surface so the bakeoff is fair, and surfaces the *recovery* difference so the bakeoff is *meaningful*. Unify what should be equal (the retrieval), measure what genuinely differs (the operations) — that's the adapter's job, and the week's.

---

## 10. When *not* to reach for a vector store at all

A senior reflex worth installing before you spend a week tuning one: not every retrieval problem is a vector-search problem. Reaching for a vector store reflexively is its own failure mode. Three cases where something else is the right tool:

- **Exact lookup by key.** "Get the document with id `clause_14`" is a key-value lookup, not a nearest-neighbor search. A `WHERE id = ...` (or a dict) is exact, instant, and needs no embedding. Embedding a key to "search" for it is slower, fuzzier, and wrong.
- **Pure structured filtering.** "All clauses of type `termination` created after January" is a *metadata* query with no semantic component — it's a `WHERE clause_type = 'termination' AND created > ...`. The vector index adds nothing; a plain relational index serves it better. (This is *why* the metadata index matters even within a vector store — much of real retrieval traffic is structured.)
- **Lexical / exact-phrase search.** "Find the clause containing the exact string 'force majeure'" is a *lexical* match (BM25, or a full-text index), not a semantic one. Dense vector search can *miss* an exact rare term because the embedding smooths it away — which is exactly why week 9 added BM25 to the hybrid pipeline. For exact-phrase needs, lexical search is the primary tool, not the fallback.

The senior framing: a vector store is the right tool for **semantic similarity over unstructured text** — "find passages that *mean* something like this query." When the question is exact (by key, by metadata, by phrase), a relational index or a lexical index is faster, cheaper, and more correct. The best retrieval systems (your week-9 hybrid pipeline, the capstone) use *all three* — vector for semantic, BM25 for lexical, metadata indexes for structured — and route each query to the right one (the agentic-RAG pattern, Lecture 2 §4). Choosing the vector store this week doesn't mean every query goes through it; it means you've picked the *semantic* tier of a system that also has lexical and structured tiers. Knowing which tier answers a given query is the difference between a RAG system that's fast and correct and one that embeds a primary key.

---

## 11. Quantization and dimensionality at the store level

One more operational lever you'll meet at scale: the *store* can compress vectors too, not just the model (week 6). Two techniques:

- **Vector quantization (scalar / product / binary).** Stores can keep vectors in fewer bits — scalar quantization (float32 → int8), product quantization (split the vector into sub-vectors and code each), or binary quantization (1 bit per dimension). The payoff is the same as model quantization: less memory, faster distance computation, at a small recall cost. Qdrant and Milvus expose these directly; for a billion-vector index, the memory savings are the difference between fitting in RAM and not. The trade is identical to week 6's quant curve — measure the recall cost, ship the level at the knee.
- **Matryoshka / dimension truncation.** Some embeddings (jina-v3, OpenAI's `text-embedding-3`) are trained so you can *truncate* the vector to fewer dimensions with graceful recall degradation — a 1024-dim vector cut to 256 dims is 4× smaller and still useful. The store holds the truncated vectors; you trade a little recall for a lot of memory and speed.

The reason this lands in the *store* lecture and not the embedding lecture: it's an *operational* knob you turn when memory or latency is the constraint at scale, and it interacts with the index (a quantized HNSW is smaller and faster but slightly less accurate). It's the same shape as every trade in this course — a measured recall cost for a real resource saving — and you tune it the same way: sweep the level, read the recall curve, ship the knee. For the 50-clause corpus this week it's irrelevant (everything fits trivially); for the capstone's 10 GB corpus it's the difference between a comfortable index and an OOM. File it under "levers you'll reach for when the corpus gets big," alongside the index parameters of §6.

---

## 12. Recap

You should now be able to:

- Frame the store choice correctly: **start at pgvector** (lowest operational surprise, Postgres familiarity, transactional joins) and move off it only under a *specific measured pressure* — filtering, scale, graph, or ergonomics.
- **Place the five stores**: pgvector (default), Qdrant (filtered ANN), Weaviate (generative/graph), Milvus (billion-scale), Chroma (prototyping ergonomics) — by workload and operational story, not leaderboard.
- Explain **filtered ANN** as the problem where stores differ: post-filter (fast, but recall collapses on selective filters — a *silent* bug), pre-filter (exact, but slow on broad filters), and native in-filter (Qdrant's filterable HNSW — the production answer).
- Treat the **metadata/payload index** as a first-class index, decided before ingest, essential for multi-tenant/faceted filtering.
- Rank **selection criteria** like a production engineer: operational familiarity and recovery time first, filtered-search recall and ingest next, query latency real-but-not-decisive, config complexity last — and weight them for the workload.
- Hold the **pipeline fixed and vary only the store** behind a thin adapter, so the bakeoff measures the store and nothing else.
- Reason about the **ANN index** (HNSW's `m`/`ef_construction`/`ef_search`, IVF's `nprobe`) well enough to know which knob is a free query-time change and which is a rebuild — because the rebuild time is part of your recovery time.
- Walk a **store decision** by reasoning, not reciting: start at pgvector, measure the pressure (filtered-recall at selectivity, recovery time) that would flip it, and defend the choice on operational axes.
- Recognize the **failure modes** (silent filtered-recall collapse, forgotten metadata index, "we'll re-embed" non-backup, embedding-swap re-ingest, single-Postgres scale wall) before they page you.
- Know **when *not* to use a vector store** at all — exact key lookup, pure structured filtering, exact-phrase lexical search — and that the best systems route each query to the right tier (vector / lexical / structured).

A closing synthesis to carry into the operations lecture: the through-line of everything above is that a vector store is a *database*, and you choose and operate it like one — on familiarity, recovery, and correctness under your real query mix — not like a leaderboard entry. The five stores are five points on a trade-off surface (familiarity ↔ filtering ↔ scale ↔ ergonomics), and your workload picks the point. The index is the engine, the metadata index is what makes filtered traffic fast, and the recovery story is what you'll be living with at 2 AM. Hold the pipeline fixed, vary the store, measure the operational axes, and the choice defends itself. That discipline — measure the thing that decides the choice, not the thing that's easy to benchmark — is the same one you used for chunking (week 8), engines (week 6), and embeddings (week 7); the variable changes, the discipline doesn't.

Next: the operational half — backup, restore, replication, the index-loss recovery drill — plus GraphRAG and agentic RAG, the two patterns that change *what* retrieval is, not just where it lives. Continue to [Lecture 2 — Operations, GraphRAG, and Agentic RAG](./02-operations-graphrag-and-agentic-rag.md).

---

## References

- *pgvector* (the Postgres-native default; HNSW, `<=>`, filtering): <https://github.com/pgvector/pgvector>
- *Qdrant — filtering & filterable HNSW* (the filtered-ANN specialist): <https://qdrant.tech/articles/filtrable-hnsw/>
- *Weaviate documentation* (generative/graph-leaning, schema-first): <https://weaviate.io/developers/weaviate>
- *Milvus* (billion-scale, separated nodes): <https://github.com/milvus-io/milvus>
- *Chroma* (developer ergonomics): <https://github.com/chroma-core/chroma>
- *From Local to Global: A Graph RAG Approach* — Edge et al., Microsoft, 2024: <https://arxiv.org/abs/2404.16130>
- *pgvector — filtering with HNSW* (iterative index scans; the filtered-ANN trade in Postgres): <https://github.com/pgvector/pgvector#filtering>
- *Qdrant — snapshots* (the fast recovery story behind the time-to-recover metric): <https://qdrant.tech/documentation/concepts/snapshots/>
- *Qdrant — vector quantization* (scalar/product/binary; the store-level compression lever): <https://qdrant.tech/documentation/guides/quantization/>
- *HNSW — Efficient and robust approximate nearest neighbor search* (Malkov & Yashunin, 2016; the index's origin): <https://arxiv.org/abs/1603.09320>

---

*A note on the moving perimeter:* the five stores named here are the current cohort's instances of a stable spine — the *concepts* (filtered ANN, metadata indexes, recovery as a first-class criterion, the index/store distinction) outlast any specific store. Expect one in three to rotate per cohort as the field shifts; the discipline of choosing on operational axes is what you keep. If your favorite store is deprecated a year from now, you should be able to swap it behind the adapter in an afternoon — that swap-ability *is* the lesson.
