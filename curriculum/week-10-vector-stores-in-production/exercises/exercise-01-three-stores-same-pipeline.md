# Exercise 1 — Three Stores, One Pipeline, One Adapter

**Goal:** Bring up **pgvector**, **Qdrant**, and **Weaviate** in Docker and run the *same* retrieval against all three behind a *single* `VectorStore` adapter interface — proving the "one interface, three stores, swap the implementation" pattern that is the spine of the whole week. You will train the most important reflex of vector-store selection: **the store is a variable you can swap, and the pipeline (embedding, chunking, gold set) is a constant you hold fixed.**

**Estimated time:** 50 minutes. Guided.

---

## Setup

You'll bring up three database servers in containers and talk to each through a thin adapter. Hold everything *else* fixed: the BGE-large embedding from week 7, the chunking-A/B winner from week 8, the 40-query gold set, and `evaluate()`.

```bash
pip install qdrant-client weaviate-client "psycopg[binary]" sentence-transformers numpy

docker run -d --name crunch-pg -e POSTGRES_PASSWORD=crunch -p 5432:5432 pgvector/pgvector:pg17
docker run -d --name crunch-qdrant -p 6333:6333 qdrant/qdrant
docker run -d --name crunch-weaviate -p 8080:8080 -p 50051:50051 \
  -e DEFAULT_VECTORIZER_MODULE=none cr.weaviate.io/semitechnologies/weaviate:1.27.0
```

> You can do this exercise with **one** store if Docker resources are tight — the adapter pattern is the lesson, and one implementation proves it. Add the others for the challenge.

---

## Step 1 — The adapter interface

Define one interface every store implements. This is the contract the rest of the week is built on:

```python
from typing import Protocol

class VectorStore(Protocol):
    def create(self, name: str, dim: int) -> None: ...
    def upsert(self, name: str, rows: list[tuple[str, list[float], dict]]) -> None: ...
    def search(self, name: str, qv: list[float], k: int) -> list[str]: ...
    def search_filtered(self, name: str, qv: list[float], k: int,
                        where: dict) -> list[str]: ...
    def drop(self, name: str) -> None: ...
```

`rows` is `(chunk_id, vector, metadata)`. `metadata` carries the filterable fields — `tenant`, `clause_type`, `version` — that the filtered-ANN exercise needs. The whole point: `evaluate()` from week 7 doesn't know or care which store is behind the adapter.

---

## Step 2 — The Qdrant adapter

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

class QdrantStore:
    def __init__(self):
        self.c = QdrantClient(host="localhost", port=6333)

    def create(self, name, dim):
        self.c.recreate_collection(
            name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def upsert(self, name, rows):
        points = [PointStruct(id=i, vector=v, payload={"chunk_id": cid, **meta})
                  for i, (cid, v, meta) in enumerate(rows)]
        self.c.upsert(name, points)
        # Index the filterable fields (Lecture 1 §4) so filters are fast:
        for field in ("tenant", "clause_type", "version"):
            self.c.create_payload_index(name, field, field_schema="keyword")

    def search(self, name, qv, k):
        hits = self.c.query_points(name, query=qv, limit=k).points
        return [h.payload["chunk_id"] for h in hits]

    def search_filtered(self, name, qv, k, where):
        flt = Filter(must=[FieldCondition(key=key, match=MatchValue(value=val))
                           for key, val in where.items()])
        hits = self.c.query_points(name, query=qv, limit=k, query_filter=flt).points
        return [h.payload["chunk_id"] for h in hits]

    def drop(self, name):
        self.c.delete_collection(name)
```

Notice the `create_payload_index` calls — that's the metadata index from Lecture 1 §4, and forgetting it is why naive filtered queries get slow.

---

## Step 3 — The pgvector adapter

```python
import psycopg, json

class PgvectorStore:
    def __init__(self):
        self.conn = psycopg.connect("host=localhost dbname=postgres "
                                    "user=postgres password=crunch", autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def create(self, name, dim):
        self.conn.execute(f"DROP TABLE IF EXISTS {name}")
        self.conn.execute(
            f"CREATE TABLE {name} (chunk_id text, embedding vector({dim}), meta jsonb)")

    def upsert(self, name, rows):
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {name} (chunk_id, embedding, meta) VALUES (%s, %s, %s)",
                [(cid, str(v), json.dumps(meta)) for cid, v, meta in rows])
        self.conn.execute(
            f"CREATE INDEX ON {name} USING hnsw (embedding vector_cosine_ops)")
        # Metadata index on the JSONB fields you filter on (Lecture 1 §4):
        self.conn.execute(f"CREATE INDEX ON {name} USING gin (meta jsonb_path_ops)")

    def search(self, name, qv, k):
        rows = self.conn.execute(
            f"SELECT chunk_id FROM {name} ORDER BY embedding <=> %s LIMIT %s",
            (str(qv), k)).fetchall()
        return [r[0] for r in rows]

    def search_filtered(self, name, qv, k, where):
        clauses = " AND ".join(f"meta->>'{key}' = %({key})s" for key in where)
        rows = self.conn.execute(
            f"SELECT chunk_id FROM {name} WHERE {clauses} "
            f"ORDER BY embedding <=> %(qv)s LIMIT %(k)s",
            {"qv": str(qv), "k": k, **where}).fetchall()
        return [r[0] for r in rows]

    def drop(self, name):
        self.conn.execute(f"DROP TABLE IF EXISTS {name}")
```

This is the *same* pgvector you've used since week 7, now behind the adapter so `evaluate()` can't tell it apart from Qdrant.

---

## Step 4 — Run the same `evaluate()` against each

```python
from sentence_transformers import SentenceTransformer
# from crunchrag_embed.eval import evaluate  # week 7, unchanged

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def build_retrieve_fn(store, name):
    def retrieve_fn(query):
        qv = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        chunk_ids = store.search(name, qv, k=20)
        # map chunk ids -> clause ids, de-dup, preserve rank (weeks 8-9, unchanged)
        return dedupe_to_clauses(chunk_ids)
    return retrieve_fn

for store_name, store in {"pgvector": PgvectorStore(), "qdrant": QdrantStore()}.items():
    store.create("clauses", dim=1024)
    store.upsert("clauses", rows)             # rows = (chunk_id, vector, metadata)
    metrics = evaluate(gold, build_retrieve_fn(store, "clauses"), k=5)
    print(f"{store_name:10s}  Recall@5={metrics['Recall@k']:.3f}  MRR={metrics['MRR']:.3f}")
```

You should see **Recall@5 and MRR that are close across stores** — because the embedding, chunking, and gold set are identical; only the store differs. If they diverge a lot, you have a bug in an adapter (a different distance metric, a missing normalization), and finding it is the lesson: the stores should agree on *unfiltered* recall, and they diverge on the *operational* axes (filtering, recovery), not on basic ANN quality.

---

## Step 5 — Write down what you found

Build a small table in `notes/week-10/three-stores.md`:

| Store | Lines of adapter code | Recall@5 | MRR | Ingest felt (fast/slow) | When I'd use it |
|---|---|---|---|---|---|

Fill one row per store. The "When I'd use it" column is the point: you're building the Lecture 1 §2 placement (pgvector=default/familiar, Qdrant=filtered, Weaviate=generative/graph) from your own observation.

---

## Acceptance criteria

You can mark this exercise done when:

- [ ] At least **two** stores implement the same `VectorStore` interface and run the *same* `evaluate()` with the *same* embedding and gold set.
- [ ] Unfiltered **Recall@5 is close** across the stores (they agree on basic ANN quality); any large divergence is debugged to an adapter bug.
- [ ] Each store's adapter **indexes the filterable metadata fields** (`create_payload_index` / a GIN index) — not just the vector index.
- [ ] `notes/week-10/three-stores.md` has one row per store with the "when I'd use it" column filled from your own observation.
- [ ] You can state, in one sentence, *why* the stores agree on unfiltered recall but you'd still choose between them (operational axes, not ANN quality).

---

## Stretch

- Add the **Weaviate** adapter (schema-first; `collections.create` with `vectorizer_config=Configure.Vectorizer.none()` since you bring your own vectors). Confirm it too lands close on unfiltered Recall@5.
- Add **Chroma** (`chromadb.PersistentClient`) as a fourth, zero-server adapter and note how much *less* setup it took — the ergonomics lesson (Lecture 1 §2.5).
- Time the **ingest** (vectors/second) for each store on the same rows. This is the first operational number that actually differs — and it bounds your recovery time (Lecture 2 §2).

---

When this feels comfortable, move to [Exercise 2 — Filtered ANN](exercise-02-filtered-ann.py).
