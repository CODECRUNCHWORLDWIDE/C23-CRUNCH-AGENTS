#!/usr/bin/env python3
# Exercise 2 — Build a sandboxed filesystem MCP server (and defend it from path traversal)
#
# Goal: Implement a FastMCP server that exposes a SANDBOXED filesystem tool
#       surface (list_files, read_file, write_file) and DEFEND it against the
#       defining vulnerability of any path-taking tool: path traversal (CWE-22).
#       The lesson is the security corollary of the whole week: the model (or a
#       prompt injection steering it) supplies the `path` argument, so you treat
#       it as hostile and prove it cannot escape the sandbox.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   Two modes:
#
#     1. As a real MCP server (inspect it, or wire it into a client):
#            mcp dev exercise-02-build-a-filesystem-server.py
#
#     2. As a standalone SELF-TEST of the path-traversal defense (no MCP client
#        needed, no network) — this is the graded part:
#            python3 exercise-02-build-a-filesystem-server.py --self-test
#
#   The self-test creates a temp sandbox, plants a "secret" file OUTSIDE it, and
#   fires a battery of traversal probes at read_file(). Every probe that targets
#   outside the sandbox MUST be blocked; every legitimate in-sandbox read MUST
#   succeed. The script prints PASS/FAIL per probe and an overall verdict.
#
# ACCEPTANCE CRITERIA
#
#   [ ] list_files / read_file / write_file are implemented as @mcp.tool().
#   [ ] _safe_path() resolves the path FIRST, then checks containment with
#       is_relative_to(SANDBOX). (Checking before resolving is the classic bug.)
#   [ ] The self-test passes: every traversal probe is BLOCKED, every legit read
#       SUCCEEDS, and the planted out-of-sandbox secret is never returned.
#   [ ] You can explain why string-matching on ".." is NOT sufficient (symlinks,
#       absolute paths, and encoded separators bypass it; resolved-containment
#       does not).
#
# Expected output (shape) is at the bottom of the file.

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# --- The sandbox root. Everything the server may touch lives under here. -------
# In --self-test mode we override this with a temp dir (see main()).
SANDBOX = Path(tempfile.gettempdir(), "crunch_mcp_sandbox").resolve()
SANDBOX.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("crunch-filesystem")


# --- The load-bearing defense: resolve FIRST, then check containment ----------
def _safe_path(relative_path: str) -> Path:
    """Resolve `relative_path` against the sandbox and PROVE it stays inside.

    The order is the whole defense:
      1. Join to SANDBOX and resolve() — this collapses '..' and follows the
         path to its true absolute location, so 'a/../../etc/passwd' becomes
         '/etc/passwd'.
      2. is_relative_to(SANDBOX) — reject anything whose resolved location is
         NOT under the sandbox root.

    Checking BEFORE resolving (e.g. 'reject any path containing ..') is brittle:
    absolute paths, symlinks, and encoded separators all slip past a string
    check but cannot slip past resolved-containment.
    """
    target = (SANDBOX / relative_path).resolve()
    if not target.is_relative_to(SANDBOX):
        raise ValueError(f"path escapes sandbox: {relative_path!r}")
    return target


# --- The tool surface ---------------------------------------------------------
@mcp.tool()
def list_files(subdir: str = "") -> list[str]:
    """List files under a subdirectory of the sandbox (relative paths)."""
    base = _safe_path(subdir) if subdir else SANDBOX
    if not base.is_dir():
        raise NotADirectoryError(subdir or ".")
    return sorted(str(p.relative_to(SANDBOX)) for p in base.iterdir())


@mcp.tool()
def read_file(relative_path: str) -> str:
    """Read a text file from inside the sandbox. Path is relative to the root."""
    target = _safe_path(relative_path)
    if not target.is_file():
        raise FileNotFoundError(relative_path)
    return target.read_text()


@mcp.tool()
def write_file(relative_path: str, content: str) -> str:
    """Write a text file inside the sandbox. Path is relative to the root."""
    target = _safe_path(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content)} chars to {relative_path}"


# --- The self-test: prove the defense holds against a battery of probes --------
def _self_test() -> int:
    global SANDBOX
    # Build an isolated sandbox in a fresh temp dir, with a SECRET planted OUTSIDE.
    root = Path(tempfile.mkdtemp(prefix="crunch_mcp_"))
    SANDBOX = (root / "sandbox").resolve()
    SANDBOX.mkdir()
    (SANDBOX / "allowed.txt").write_text("public contract text")
    (SANDBOX / "sub").mkdir()
    (SANDBOX / "sub" / "nested.txt").write_text("nested but allowed")
    secret = root / "SECRET.txt"          # sibling of the sandbox, NOT inside it
    secret.write_text("TOP SECRET — must never be read via the tool")

    print(f"sandbox: {SANDBOX}")
    print(f"secret (outside sandbox): {secret}\n")

    # (path, should_be_allowed)
    probes: list[tuple[str, bool]] = [
        ("allowed.txt", True),                 # legit
        ("sub/nested.txt", True),              # legit nested
        ("../SECRET.txt", False),              # classic traversal
        ("../../etc/passwd", False),           # deeper traversal
        ("sub/../../SECRET.txt", False),       # traversal via a legit subdir
        ("/etc/passwd", False),                # absolute path escape
        (str(secret), False),                  # absolute path to the secret
        ("./allowed.txt", True),               # legit with a leading ./
    ]

    passed = 0
    leaked = False
    for path, should_allow in probes:
        try:
            text = read_file(path)
            allowed = True
        except (ValueError, FileNotFoundError) as exc:
            allowed = False
            text = f"<blocked: {type(exc).__name__}>"

        ok = (allowed == should_allow)
        if allowed and "TOP SECRET" in text:
            leaked = True
            ok = False
        mark = "PASS" if ok else "FAIL"
        verdict = "allowed" if allowed else "blocked"
        want = "allow" if should_allow else "block"
        print(f"  [{mark}] read_file({path!r:38}) -> {verdict:8} (wanted {want})")
        passed += ok

    print(f"\n{passed}/{len(probes)} probes behaved correctly.")
    if leaked:
        print("CRITICAL: the out-of-sandbox SECRET was READ — your containment "
              "check is broken. Resolve the path BEFORE checking containment.")
        return 1
    if passed == len(probes):
        print("PASS: every traversal probe was blocked and every legit read "
              "succeeded. The model cannot escape the sandbox via `path`. This "
              "is the defense week 17's red team will try (and should fail) to "
              "break.")
        return 0
    print("Some probes misbehaved — re-read _safe_path(): resolve() FIRST, then "
          "is_relative_to(SANDBOX).")
    return 1


def main() -> int:
    if "--self-test" in sys.argv:
        return _self_test()
    # Otherwise, run as a real MCP server over stdio (inspect with `mcp dev`).
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape) for: python3 exercise-02-build-a-filesystem-server.py --self-test
# -----------------------------------------------------------------------------
#
# sandbox: /tmp/crunch_mcp_xxxx/sandbox
# secret (outside sandbox): /tmp/crunch_mcp_xxxx/SECRET.txt
#
#   [PASS] read_file('allowed.txt')                        -> allowed  (wanted allow)
#   [PASS] read_file('sub/nested.txt')                     -> allowed  (wanted allow)
#   [PASS] read_file('../SECRET.txt')                      -> blocked  (wanted block)
#   [PASS] read_file('../../etc/passwd')                   -> blocked  (wanted block)
#   [PASS] read_file('sub/../../SECRET.txt')               -> blocked  (wanted block)
#   [PASS] read_file('/etc/passwd')                        -> blocked  (wanted block)
#   [PASS] read_file('/tmp/crunch_mcp_xxxx/SECRET.txt')    -> blocked  (wanted block)
#   [PASS] read_file('./allowed.txt')                      -> allowed  (wanted allow)
#
# 8/8 probes behaved correctly.
# PASS: every traversal probe was blocked and every legit read succeeded. ...
#
# NOTE: If you "fix" _safe_path to check for '..' in the STRING before resolving,
# the absolute-path probes (/etc/passwd, /tmp/.../SECRET.txt) slip right through
# — they contain no '..' at all. That is exactly why the resolved-containment
# check is the correct defense and the string check is not. Try it and watch the
# self-test catch your regression.
# -----------------------------------------------------------------------------
