#!/usr/bin/env python3
# Exercise 3 — Consume an MCP server from a programmatic ClientSession
#
# Goal: Drive an MCP server from YOUR code, end to end, over stdio. You will run
#       the full client lifecycle — open the transport, initialize (the
#       handshake), list_tools (discover), call_tool (invoke), read_resource
#       (context) — and ASSERT the results. The lesson: when your application is
#       the host, a ClientSession is how you reach a server, and the lifecycle
#       mirrors the protocol exactly (initialize FIRST, then operate).
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   Standalone. This file is BOTH the server and the client:
#     * Run with `--serve` and it behaves as a corpus-search MCP server (over
#       stdio). You do not run this yourself — the client spawns it.
#     * Run with no args and it acts as the CLIENT: it spawns a copy of itself
#       in --serve mode as a subprocess and drives it through the full lifecycle.
#
#         python3 exercise-03-consume-from-a-client.py
#
#   This self-contained shape means no second file and no network — the whole
#   client/server round trip runs in one command. The corpus is the same legal
#   corpus you've used since week 7, so a search for "five-year confidentiality"
#   returns clause_09, which the client asserts.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The client calls session.initialize() BEFORE any list/call (the handshake
#       must complete first — a call before initialize is a protocol error).
#   [ ] list_tools() returns search_corpus and get_clause.
#   [ ] call_tool("search_corpus", {"query": "five-year confidentiality"}) returns
#       a result whose top hit is clause_09 (the answer survived, end to end).
#   [ ] read_resource("corpus://clause_09") returns the clause text.
#   [ ] All assertions pass and the script prints PASS.
#
# Expected output (shape) is at the bottom of the file.

from __future__ import annotations

import asyncio
import sys

# --- The corpus: same legal clauses you've used since week 7 -------------------
CORPUS: dict[str, str] = {
    "clause_07": "The annual fee shall be paid in twelve equal monthly installments.",
    "clause_09": "All confidential information must be protected for five years after termination.",
    "clause_12": "The Contractor shall maintain professional liability insurance of one million dollars.",
    "clause_14": "Either party may terminate this Agreement upon thirty days written notice.",
    "clause_18": "This Agreement is governed by the laws of the State of Delaware.",
    "clause_27": "Any dispute shall be resolved by binding arbitration in San Francisco.",
}


# =============================================================================
# SERVER SIDE  (runs when invoked with --serve; spawned by the client)
# =============================================================================
def run_server() -> None:
    import re

    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("crunch-corpus")

    @mcp.tool()
    def search_corpus(query: str) -> list[dict]:
        """Search the legal corpus. Returns the best-matching clauses with a
        crude lexical score. Use when the user asks about contract terms."""
        terms = set(re.findall(r"\w+", query.lower()))
        hits = []
        for cid, text in CORPUS.items():
            words = set(re.findall(r"\w+", text.lower()))
            overlap = len(terms & words)
            if overlap:
                hits.append({"clause_id": cid, "score": overlap, "text": text})
        hits.sort(key=lambda h: -h["score"])
        return hits[:3]

    @mcp.tool()
    def get_clause(clause_id: str) -> str:
        """Fetch one clause by id (validated to clause_NN)."""
        # Argument validation: the model supplies this, so check its SHAPE.
        if not re.fullmatch(r"clause_\d{2}", clause_id):
            raise ValueError(f"invalid clause_id {clause_id!r}; expected clause_NN")
        if clause_id not in CORPUS:
            raise KeyError(clause_id)
        return CORPUS[clause_id]

    @mcp.resource("corpus://{clause_id}")
    def clause_resource(clause_id: str) -> str:
        """Read a clause as a RESOURCE (app-controlled context), addressed by URI."""
        return CORPUS.get(clause_id, f"<no such clause: {clause_id}>")

    mcp.run()  # stdio


# =============================================================================
# CLIENT SIDE  (the default mode: spawn the server, drive the full lifecycle)
# =============================================================================
async def run_client() -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Spawn THIS file in --serve mode as the server subprocess.
    params = StdioServerParameters(command=sys.executable, args=[__file__, "--serve"])

    failures = 0
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Handshake FIRST — nothing else is legal before this completes.
            await session.initialize()
            print("initialize -> handshake complete; session live")

            # 2. Discover the tools.
            listed = await session.list_tools()
            names = sorted(t.name for t in listed.tools)
            print("tools/list ->", names)
            if names != ["get_clause", "search_corpus"]:
                print("  FAIL: expected [get_clause, search_corpus]")
                failures += 1

            # 3. Call a tool: search for the five-year confidentiality clause.
            result = await session.call_tool(
                "search_corpus", {"query": "five-year confidentiality duration"}
            )
            # Tool results come back as content blocks; FastMCP returns the
            # structured value too. Pull the text and confirm clause_09 is top.
            text = "".join(
                getattr(b, "text", "") for b in result.content
            )
            print("tools/call search_corpus ->", text[:80].replace("\n", " "), "...")
            if "clause_09" not in text:
                print("  FAIL: top hit should be clause_09 (the answer survived)")
                failures += 1
            else:
                print("  OK: clause_09 retrieved end to end ✓")

            # 4. Call get_clause with a VALID id.
            clause = await session.call_tool("get_clause", {"clause_id": "clause_14"})
            ctext = "".join(getattr(b, "text", "") for b in clause.content)
            print("tools/call get_clause(clause_14) ->", ctext[:60], "...")
            if "terminate" not in ctext.lower():
                print("  FAIL: clause_14 should mention termination")
                failures += 1

            # 5. Call get_clause with an INVALID id — confirm it errors cleanly
            #    (the argument-validation defense surfaces as a tool error).
            bad = await session.call_tool("get_clause", {"clause_id": "DROP TABLE"})
            if not bad.isError:
                print("  FAIL: invalid clause_id should produce isError=True")
                failures += 1
            else:
                print("tools/call get_clause('DROP TABLE') -> isError ✓ (validated)")

            # 6. Read a RESOURCE by URI (app-controlled context, not a tool call).
            res = await session.read_resource("corpus://clause_09")
            rtext = "".join(getattr(c, "text", "") for c in res.contents)
            print("resources/read corpus://clause_09 ->", rtext[:60], "...")
            if "five years" not in rtext:
                print("  FAIL: resource should contain the five-year text")
                failures += 1

    print()
    if failures == 0:
        print("PASS: full client lifecycle works — initialize, discover, call, "
              "validate, read resource. The SAME server answers any MCP client; "
              "you just drove it from your own code.")
        return 0
    print(f"{failures} assertion(s) failed — re-read the lifecycle in Lecture 2 §2.2.")
    return 1


def main() -> int:
    if "--serve" in sys.argv:
        run_server()
        return 0
    return asyncio.run(run_client())


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape) for: python3 exercise-03-consume-from-a-client.py
# -----------------------------------------------------------------------------
#
# initialize -> handshake complete; session live
# tools/list -> ['get_clause', 'search_corpus']
# tools/call search_corpus -> All confidential information must be protected for five years ... ...
#   OK: clause_09 retrieved end to end ✓
# tools/call get_clause(clause_14) -> Either party may terminate this Agreement upon thirty days ...
# tools/call get_clause('DROP TABLE') -> isError ✓ (validated)
# resources/read corpus://clause_09 -> All confidential information must be protected for five ...
#
# PASS: full client lifecycle works — initialize, discover, call, validate, read
# resource. The SAME server answers any MCP client; you just drove it from your
# own code.
#
# NOTE: swap the transport to streamable HTTP and ONLY the opener changes
# (streamablehttp_client(url) instead of stdio_client(params)) — everything from
# ClientSession(...) down is identical. That symmetry is the transport
# abstraction paying off on the client side, the mirror of mcp.run() vs
# mcp.run(transport="streamable-http") on the server side.
# -----------------------------------------------------------------------------
