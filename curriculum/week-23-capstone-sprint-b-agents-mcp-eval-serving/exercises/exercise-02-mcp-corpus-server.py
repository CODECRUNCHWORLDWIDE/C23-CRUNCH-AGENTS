#!/usr/bin/env python3
# Exercise 2 — The corpus MCP server (and a hardened filesystem tool)
#
# Goal: Write the custom private-corpus search MCP server the retrieval-agent
#       calls, plus a companion filesystem-read tool, and HARDEN the filesystem
#       tool against path traversal. The lesson is the week-15/week-17 rule made
#       load-bearing: A TOOL IS AN RCE PRIMITIVE. Validate every argument before
#       you do anything, and resolve every path against a sandbox root.
#
# Estimated time: 55 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   This file runs WITHOUT a real corpus and WITHOUT a network: it ships a tiny
#   in-memory corpus and a tiny sandbox directory so the server + a self-test
#   client run end-to-end. In the capstone you swap the in-memory corpus for your
#   Sprint A hybrid_search() and the sandbox for the real corpus root.
#
#       pip install mcp
#       python3 exercise-02-mcp-corpus-server.py
#
#   The self-test at the bottom (1) lists the tools, (2) calls corpus_search and
#   prints ranked hits, (3) calls fs_read on a legal path and prints the file,
#   (4) calls fs_read on a TRAVERSAL path and asserts it is REJECTED.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The server advertises corpus_search and fs_read via list_tools().
#   [ ] corpus_search validates: query is a non-empty string, k is in [1, 20].
#       Out-of-range args raise BEFORE any search runs.
#   [ ] corpus_search returns ranked chunks tagged with their source doc id.
#   [ ] fs_read resolves the requested path against SANDBOX and REJECTS any path
#       that escapes it (../../etc/passwd must raise, not read the file).
#   [ ] The tool descriptions are PRESCRIPTIVE about WHEN to call (not just what).
#   [ ] The self-test prints the four sections and the traversal assertion holds.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import asyncio
from pathlib import Path

import mcp.types as types
from mcp.server import Server

# --------------------------------------------------------------------------- #
# A tiny in-memory corpus so this file runs with no Sprint A dependency.
# In the capstone, replace search_corpus() with your hybrid_search() pipeline.
# --------------------------------------------------------------------------- #
_CORPUS = [
    ("doc_indemnity", "The indemnity cap for data-breach claims is two million "
     "dollars per incident, aggregate across the term."),
    ("doc_term", "This agreement has an initial term of three years, renewing "
     "annually unless terminated with ninety days notice."),
    ("doc_confidential", "Confidential information must be protected for five "
     "years after termination of the agreement."),
    ("doc_payment", "Invoices are net-30; late payments accrue interest at "
     "one and a half percent per month."),
]


def search_corpus(query: str, k: int) -> list[tuple[str, str]]:
    """Toy lexical scorer standing in for the Sprint A hybrid pipeline.

    Ranks by count of shared lowercased word stems. Deterministic, no network.
    """
    q = {w.lower().strip(".,?") for w in query.split() if len(w) > 3}
    scored = []
    for doc_id, text in _CORPUS:
        words = {w.lower().strip(".,?") for w in text.split()}
        overlap = len(q & words)
        scored.append((overlap, doc_id, text))
    scored.sort(key=lambda r: r[0], reverse=True)
    return [(doc_id, text) for _, doc_id, text in scored[:k]]


# --------------------------------------------------------------------------- #
# A tiny sandbox directory for the filesystem tool. We create it at startup so
# the legal-path read succeeds and the traversal read can be tested.
# --------------------------------------------------------------------------- #
SANDBOX = (Path(__file__).parent / "_sandbox_corpus").resolve()


def _ensure_sandbox() -> None:
    SANDBOX.mkdir(exist_ok=True)
    (SANDBOX / "readme.txt").write_text(
        "This file lives inside the sandbox and is safe to read.\n"
    )


def safe_path(requested: str) -> Path:
    """Resolve `requested` against SANDBOX and reject anything that escapes it.

    This is the path-traversal defense. The classic bug is prefix-matching the
    root string; the correct check resolves real paths and uses is_relative_to.
    """
    # TODO 1: build the candidate path as (SANDBOX / requested).resolve(), then
    #   raise ValueError("path escapes the sandbox") if it is NOT relative to
    #   SANDBOX. Use Path.is_relative_to (Python 3.9+). Return the path otherwise.
    raise NotImplementedError("implement safe_path (TODO 1)")


# --------------------------------------------------------------------------- #
# The MCP server.
# --------------------------------------------------------------------------- #
server = Server("capstone-corpus")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="corpus_search",
            # Prescriptive description: state WHEN to call, not just what it does.
            description=(
                "Search the private research corpus. Call this whenever you need "
                "source material to ground an answer in the documents. Returns the "
                "top-k ranked chunks, each tagged with its source document id."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "The natural-language search query."},
                    "k": {"type": "integer", "minimum": 1, "maximum": 20,
                          "default": 5,
                          "description": "Number of chunks to return (1-20)."},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="fs_read",
            description=(
                "Read a UTF-8 text file from the sandboxed corpus directory. Call "
                "this when you need the full text of a specific source file by path."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Path relative to the corpus root."},
                },
                "required": ["path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "corpus_search":
        # --- validate BEFORE searching (a tool is an RCE primitive) ---
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = arguments.get("k", 5)
        # TODO 2: reject k that is not an int in [1, 20] with a clear ValueError.

        hits = search_corpus(query, k=k)
        body = "\n\n".join(f"[{doc_id}] {text}" for doc_id, text in hits)
        return [types.TextContent(type="text", text=body)]

    if name == "fs_read":
        requested = arguments.get("path")
        if not isinstance(requested, str) or not requested:
            raise ValueError("path must be a non-empty string")
        # TODO 3: resolve via safe_path() (which rejects traversal), read the file
        #   as UTF-8 text, and return it as one TextContent. Let safe_path's
        #   ValueError propagate on a traversal attempt — do NOT swallow it.
        raise NotImplementedError("implement fs_read dispatch (TODO 3)")

    raise ValueError(f"unknown tool: {name}")


# --------------------------------------------------------------------------- #
# Self-test client: exercises the server in-process over an in-memory transport.
# This stands in for a LangGraph agent calling the tools. It proves the schema,
# the search, the legal read, and the traversal rejection.
# --------------------------------------------------------------------------- #
async def self_test() -> None:
    _ensure_sandbox()

    # We call the registered handlers directly (the in-process equivalent of an
    # MCP client round-trip) so the file is self-contained and needs no subprocess.
    print("=== 1. list_tools ===")
    tools = await list_tools()
    for t in tools:
        print(f"  {t.name}: {t.description.split('.')[0]}.")

    print("\n=== 2. corpus_search ===")
    out = await call_tool("corpus_search",
                          {"query": "indemnity cap for data-breach claims", "k": 2})
    print(out[0].text)

    print("\n=== 3. fs_read (legal path) ===")
    out = await call_tool("fs_read", {"path": "readme.txt"})
    print(out[0].text.strip())

    print("\n=== 4. fs_read (traversal path) — must be REJECTED ===")
    try:
        await call_tool("fs_read", {"path": "../../etc/passwd"})
        raise AssertionError("traversal was NOT rejected — defense failed!")
    except ValueError as e:
        print(f"  rejected as expected: {e}")

    print("\n=== 5. corpus_search bad arg — must be REJECTED ===")
    try:
        await call_tool("corpus_search", {"query": "x", "k": 999})
        raise AssertionError("out-of-range k was NOT rejected!")
    except ValueError as e:
        print(f"  rejected as expected: {e}")

    print("\nAll tool checks passed — the surface is hardened.")


if __name__ == "__main__":
    asyncio.run(self_test())

# --------------------------------------------------------------------------- #
# EXPECTED OUTPUT (shape — exact wording of hits depends on the toy scorer)
# --------------------------------------------------------------------------- #
#
# === 1. list_tools ===
#   corpus_search: Search the private research corpus.
#   fs_read: Read a UTF-8 text file from the sandboxed corpus directory.
#
# === 2. corpus_search ===
# [doc_indemnity] The indemnity cap for data-breach claims is two million ...
# [doc_term] This agreement has an initial term of three years ...
#
# === 3. fs_read (legal path) ===
# This file lives inside the sandbox and is safe to read.
#
# === 4. fs_read (traversal path) — must be REJECTED ===
#   rejected as expected: path escapes the sandbox
#
# === 5. corpus_search bad arg — must be REJECTED ===
#   rejected as expected: k must be an integer in [1, 20]
#
# All tool checks passed — the surface is hardened.
#
# --------------------------------------------------------------------------- #
# To run the SAME server over a real stdio transport (what a LangGraph agent or
# Claude Desktop connects to), replace the self_test() call with:
#
#     from mcp.server.stdio import stdio_server
#     async def main():
#         async with stdio_server() as (read, write):
#             await server.run(read, write, server.create_initialization_options())
#     asyncio.run(main())
#
# For the deployed, cross-client surface, use the streamable-HTTP transport
# instead (see the MCP Python SDK docs) — the tool logic above is unchanged.
# --------------------------------------------------------------------------- #
