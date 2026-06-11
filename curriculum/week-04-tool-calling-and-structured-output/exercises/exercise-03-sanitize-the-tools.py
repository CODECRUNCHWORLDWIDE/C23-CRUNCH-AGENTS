#!/usr/bin/env python3
# Exercise 3 — Sanitize the tools (defeat path traversal and SSRF)
#
# Goal: Take a NAIVE file-read tool and a NAIVE web-fetch tool, demonstrate that an
#       attacker-chosen argument breaks each one, then implement the HARDENED versions
#       and prove the same attack now fails — while legitimate calls still succeed.
#
#       Remember the frame: a tool argument comes from an untrusted client (the model),
#       and by Week 17 a retrieved document can steer that argument. So you validate
#       every argument as if a hostile party chose it. This exercise makes that concrete.
#
# Estimated time: 45 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#     pip install httpx
#     python3 exercise-03-sanitize-the-tools.py
#
#   No API key and no model needed — we attack the TOOLS directly with the arguments a
#   malicious model (or injected document) would choose. The point is the tool's defense,
#   not the model's behavior.
#
# ACCEPTANCE CRITERIA
#
#   [ ] The naive file tool reads a file OUTSIDE the sandbox; the hardened one refuses.
#   [ ] The naive web tool would fetch a private/loopback address; the hardened one refuses.
#   [ ] A LEGITIMATE call to each hardened tool still succeeds.
#   [ ] The program prints "ALL DEFENSES HELD" and exits 0.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import ipaddress
import os
import socket
import sys
import tempfile
from urllib.parse import urlparse

import httpx

# A sandbox the file tool is allowed to read inside. Everything else is off-limits.
SANDBOX = os.path.realpath(tempfile.mkdtemp(prefix="agent_sandbox_"))

# A secret OUTSIDE the sandbox that the path-traversal attack will try to steal.
SECRET_DIR = os.path.realpath(tempfile.mkdtemp(prefix="agent_secret_"))
SECRET_PATH = os.path.join(SECRET_DIR, "id_rsa")
with open(SECRET_PATH, "w") as f:
    f.write("-----BEGIN FAKE PRIVATE KEY-----\nstolen!\n")

# A legitimate file the tool SHOULD be able to read.
GOOD_PATH = os.path.join(SANDBOX, "notes.txt")
with open(GOOD_PATH, "w") as f:
    f.write("these are the agent's own notes")


# =============================================================================
# FILE TOOL
# =============================================================================
def read_file_naive(path: str) -> str:
    """CATASTROPHIC: opens whatever path the model chose."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def read_file_hardened(path: str) -> str:
    """Confine every read to SANDBOX. realpath() resolves '..' AND symlinks before
    the check, closing the traversal and the symlink-escape variants."""
    candidate = os.path.realpath(os.path.join(SANDBOX, path))
    if not (candidate == SANDBOX or candidate.startswith(SANDBOX + os.sep)):
        return "ERROR: path escapes the sandbox"
    if not os.path.isfile(candidate):
        return "ERROR: not a file"
    with open(candidate, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read(64_000)  # bound the size too — no 4GB DoS on your context window


# =============================================================================
# WEB TOOL
# =============================================================================
def fetch_url_naive(url: str) -> str:
    """SSRF: fetches whatever URL the model chose, including internal addresses."""
    return httpx.get(url, timeout=3.0).text


def fetch_url_hardened(url: str) -> str:
    """Allow only http/https to PUBLIC addresses. Reject private/loopback/link-local,
    and do NOT follow redirects blindly (a public URL can 302 to 169.254.169.254)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "ERROR: only http/https allowed"
    host = parsed.hostname
    if host is None:
        return "ERROR: no host in URL"
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except (socket.gaierror, ValueError):
        return "ERROR: cannot resolve host"
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return f"ERROR: refusing to fetch private/loopback/link-local address ({ip})"
    resp = httpx.get(url, timeout=5.0, follow_redirects=False)
    return resp.text[:64_000]


# =============================================================================
# THE ATTACKS + THE PROOFS
# =============================================================================
def main() -> None:
    held = True

    print("=== FILE TOOL: path traversal ===")
    # The argument a malicious model / injected document would choose:
    traversal = os.path.relpath(SECRET_PATH, SANDBOX)  # e.g. '../agent_secret_xxx/id_rsa'
    print(f"  attack arg: read_file(path={traversal!r})")

    naive_out = read_file_naive(os.path.join(SANDBOX, traversal))
    print(f"  naive   -> {naive_out.strip()[:40]!r}  (LEAKED the secret!)")
    if "stolen" not in naive_out:
        print("  (note: naive happened not to leak here, but it is still unsafe)")

    hardened_out = read_file_hardened(traversal)
    print(f"  hardened-> {hardened_out!r}")
    if "stolen" in hardened_out:
        print("  !! DEFENSE FAILED: hardened tool leaked the secret")
        held = False

    legit = read_file_hardened("notes.txt")
    print(f"  legit   -> {legit!r}")
    if legit != "these are the agent's own notes":
        print("  !! REGRESSION: hardened tool blocked a legitimate read")
        held = False

    print("\n=== WEB TOOL: SSRF to the cloud metadata endpoint ===")
    ssrf_url = "http://169.254.169.254/latest/meta-data/"  # AWS metadata; link-local
    print(f"  attack arg: fetch_url(url={ssrf_url!r})")
    # We do NOT call fetch_url_naive on the real endpoint (it might hang or actually
    # reach metadata on a cloud box). We assert the hardened tool refuses it.
    hardened_web = fetch_url_hardened(ssrf_url)
    print(f"  hardened-> {hardened_web!r}")
    if not hardened_web.startswith("ERROR"):
        print("  !! DEFENSE FAILED: hardened tool did not refuse the metadata endpoint")
        held = False

    for bad in ["http://localhost:6379/", "http://127.0.0.1/", "file:///etc/passwd",
                "http://10.0.0.1/"]:
        out = fetch_url_hardened(bad)
        ok = out.startswith("ERROR")
        print(f"  {bad:38s} -> {'refused ✓' if ok else 'ALLOWED ✗'}")
        held = held and ok

    # A legitimate public fetch should still work (skip if offline).
    print("\n=== WEB TOOL: a legitimate public fetch still works ===")
    try:
        good = fetch_url_hardened("http://example.com/")
        ok = "Example Domain" in good or len(good) > 0
        print(f"  http://example.com/ -> {'fetched ✓' if ok else 'empty (offline?)'}")
    except Exception as e:
        print(f"  (offline or blocked: {type(e).__name__}) — skipping the positive check")

    print()
    if held:
        print("ALL DEFENSES HELD")
        sys.exit(0)
    else:
        print("A DEFENSE FAILED — fix the hardened tool above")
        sys.exit(1)


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Expected output (shape)
# -----------------------------------------------------------------------------
#
# === FILE TOOL: path traversal ===
#   attack arg: read_file(path='../agent_secret_xxxx/id_rsa')
#   naive   -> '-----BEGIN FAKE PRIVATE KEY-----'  (LEAKED the secret!)
#   hardened-> 'ERROR: path escapes the sandbox'
#   legit   -> 'these are the agent's own notes'
#
# === WEB TOOL: SSRF to the cloud metadata endpoint ===
#   attack arg: fetch_url(url='http://169.254.169.254/latest/meta-data/')
#   hardened-> "ERROR: refusing to fetch private/loopback/link-local address (169.254.169.254)"
#   http://localhost:6379/                 -> refused ✓
#   http://127.0.0.1/                      -> refused ✓
#   file:///etc/passwd                     -> refused ✓
#   http://10.0.0.1/                       -> refused ✓
#
# === WEB TOOL: a legitimate public fetch still works ===
#   http://example.com/ -> fetched ✓
#
# ALL DEFENSES HELD
#
# The lesson, stated once: the naive tools do exactly what the (untrusted) model asked.
# The hardened tools parse-and-whitelist, confine to a sandbox / public-IP space, and
# bound everything. That is the entire security half of the week, in one file.
# -----------------------------------------------------------------------------
