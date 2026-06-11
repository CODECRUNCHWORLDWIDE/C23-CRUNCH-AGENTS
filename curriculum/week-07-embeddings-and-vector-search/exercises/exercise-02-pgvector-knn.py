#!/usr/bin/env python3
# Exercise 2 — pgvector k-NN (stand up a real vector index and query it)
#
# Goal: Embed a small legal corpus with a real open model, store the vectors in
#       Postgres + pgvector, build an HNSW index, and run k-NN queries that
#       return the RIGHT clause. This is the spine of every RAG retrieval call
#       you will write in Phase II.
#
# Estimated time: 45 minutes. Runnable.
#
# PREREQUISITES
#
#   1. Postgres + pgvector running. The fastest way:
#
#        docker run -d --name crunch-pg \
#          -e POSTGRES_PASSWORD=crunch \
#          -p 5432:5432 \
#          pgvector/pgvector:pg17
#
#   2. Python deps (in your venv):
#
#        pip install sentence-transformers "psycopg[binary]" numpy
#
#   3. Run it:
#
#        python3 exercise-02-pgvector-knn.py
#
# WHAT THIS PROVES
#
#   * The query/document asymmetry is real: BGE wants a prefix on queries.
#   * The pgvector operator (<=>) and the index op (vector_cosine_ops) must match.
#   * A k-NN query returns the right clause for a meaning-based query that shares
#     almost no keywords with the answer.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The script creates the table, inserts 8 clauses, builds the HNSW index,
#       and prints a PASS line.
#   [ ] For the query "how do I end the contract early", the #1 result is the
#       termination clause (clause_14).
#   [ ] You can explain why dropping the query prefix (try it!) hurts the result.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import sys

import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:crunch@localhost:5432/postgres"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
DIM = 1024  # bge-large-en-v1.5 output dimension
# BGE was trained with this instruction prefix on QUERIES (not documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# A tiny synthetic services-agreement corpus: (doc_id, clause text).
CORPUS: list[tuple[str, str]] = [
    ("clause_01", "This Agreement is entered into between the Company and the Contractor."),
    ("clause_07", "The annual fee shall be paid in twelve equal monthly installments."),
    ("clause_09", "All confidential information must be protected for five years after termination."),
    ("clause_12", "The Contractor shall maintain professional liability insurance of $1,000,000."),
    ("clause_14", "Either party may terminate this Agreement upon thirty days written notice."),
    ("clause_18", "This Agreement is governed by the laws of the State of Delaware."),
    ("clause_22", "Neither party shall be liable for delays caused by events beyond its control."),
    ("clause_27", "Any dispute shall be resolved by binding arbitration in San Francisco."),
]


def to_pgvector(v: np.ndarray) -> str:
    """pgvector accepts the textual literal form '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def setup_schema(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        f"""
        CREATE TABLE chunks (
            id        BIGSERIAL PRIMARY KEY,
            doc_id    TEXT NOT NULL,
            content   TEXT NOT NULL,
            embedding vector({DIM})
        )
        """
    )
    conn.commit()


def embed_and_insert(conn: psycopg.Connection, model: SentenceTransformer) -> None:
    texts = [text for _, text in CORPUS]
    # Documents get NO prefix for BGE. Batch the encode for throughput.
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=8)
    with conn.cursor() as cur:
        for (doc_id, text), vec in zip(CORPUS, vecs):
            cur.execute(
                "INSERT INTO chunks (doc_id, content, embedding) VALUES (%s, %s, %s)",
                (doc_id, text, to_pgvector(vec)),
            )
    conn.commit()


def build_index(conn: psycopg.Connection) -> None:
    # The index op (vector_cosine_ops) MUST match the query operator (<=>).
    # Build a mismatched op and Postgres silently does a full scan instead.
    conn.execute(
        "CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    conn.commit()


def knn(
    conn: psycopg.Connection,
    model: SentenceTransformer,
    query: str,
    k: int = 3,
    use_prefix: bool = True,
) -> list[tuple[str, str, float]]:
    text = (QUERY_PREFIX + query) if use_prefix else query
    qvec = model.encode(text, normalize_embeddings=True)
    conn.execute("SET hnsw.ef_search = 100")
    rows = conn.execute(
        "SELECT doc_id, content, 1 - (embedding <=> %s) AS sim "
        "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (to_pgvector(qvec), to_pgvector(qvec), k),
    ).fetchall()
    return [(doc_id, content, float(sim)) for doc_id, content, sim in rows]


def main() -> int:
    print(f"loading {MODEL_NAME} (first run downloads ~1.3 GB)...")
    model = SentenceTransformer(MODEL_NAME)

    try:
        conn = psycopg.connect(DSN)
    except psycopg.OperationalError as exc:
        print(f"FAIL: cannot connect to Postgres at {DSN}\n  {exc}")
        print("  Is the pgvector container running? See the header of this file.")
        return 1

    with conn:
        setup_schema(conn)
        embed_and_insert(conn, model)
        build_index(conn)
        print(f"inserted {len(CORPUS)} clauses and built an HNSW index.\n")

        query = "how do I end the contract early"
        results = knn(conn, model, query, k=3, use_prefix=True)
        print(f'query: "{query}"  (with BGE query prefix)')
        for rank, (doc_id, content, sim) in enumerate(results, start=1):
            print(f"  #{rank}  {doc_id}  sim={sim:.3f}  {content}")

        top_doc = results[0][0]
        if top_doc == "clause_14":
            print("\nPASS: the termination clause (clause_14) ranked #1 for a query "
                  "that shares no keywords with it. That is dense retrieval working.")
            exit_code = 0
        else:
            print(f"\nFAIL: expected clause_14 at #1, got {top_doc}. "
                  "Check normalization and the query prefix.")
            exit_code = 1

        # Teaching moment: show the SAME query WITHOUT the prefix.
        bad = knn(conn, model, query, k=3, use_prefix=False)
        print(f'\n(for contrast) same query WITHOUT the prefix:')
        for rank, (doc_id, content, sim) in enumerate(bad, start=1):
            print(f"  #{rank}  {doc_id}  sim={sim:.3f}")
        print("Compare the top scores. Dropping the prefix is the classic silent bug.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact sims vary by model version and machine)
# -----------------------------------------------------------------------------
#
# loading BAAI/bge-large-en-v1.5 (first run downloads ~1.3 GB)...
# inserted 8 clauses and built an HNSW index.
#
# query: "how do I end the contract early"  (with BGE query prefix)
#   #1  clause_14  sim=0.71  Either party may terminate this Agreement upon thirty days written notice.
#   #2  clause_09  sim=0.55  All confidential information must be protected for five years after termination.
#   #3  clause_18  sim=0.42  This Agreement is governed by the laws of the State of Delaware.
#
# PASS: the termination clause (clause_14) ranked #1 for a query that shares no
#       keywords with it. That is dense retrieval working.
#
# (for contrast) same query WITHOUT the prefix:
#   #1  clause_14  sim=0.64  ...
#   ...
# Compare the top scores. Dropping the prefix is the classic silent bug.
#
# NOTE: clause_09 ("...after termination") often ranks #2 because it shares the
# word "termination" topically — a preview of why lexical signals (week 9) and
# good chunking (week 8) matter even when dense retrieval already "works."
# -----------------------------------------------------------------------------
