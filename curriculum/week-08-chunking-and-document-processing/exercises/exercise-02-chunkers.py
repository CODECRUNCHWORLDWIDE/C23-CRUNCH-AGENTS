#!/usr/bin/env python3
# Exercise 2 — The chunkers (fixed, sliding-window, recursive) and their boundaries
#
# Goal: Implement the three boundary-based chunkers from Lecture 1 FROM SCRATCH —
#       fixed token-window, sliding-window-with-overlap, and recursive (LangChain-
#       style separator hierarchy) — and INSPECT where each one puts its
#       boundaries on a real document. The lesson is visual: you will SEE fixed
#       windows slice through the middle of clause 9 ("five years after
#       termination") and SEE recursive chunking keep it whole.
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   Standalone. No database, no network required. Just:
#
#       python3 exercise-02-chunkers.py
#
#   It chunks a multi-clause services-agreement document three ways and prints,
#   for each strategy, the chunk count and the boundary report: whether each
#   "must-survive" answer (the termination clause, the five-year confidentiality
#   clause) landed WHOLE inside a single chunk, or got SPLIT across two.
#
#   Token counting: by default this uses the BGE tokenizer if `transformers` is
#   installed (so "512 tokens" means what the encoder sees). If it is not
#   installed, it falls back to a whitespace-word tokenizer so the file STILL
#   RUNS and the boundary lessons are identical — only the absolute token counts
#   differ. The header prints which tokenizer is active.
#
# ACCEPTANCE CRITERIA
#
#   [ ] All three chunkers run and print their chunk count and boundary report.
#   [ ] With a SMALL size (e.g. 24 tokens), the fixed chunker SPLITS at least one
#       must-survive clause (the report shows it spanning 2 chunks).
#   [ ] The recursive chunker keeps every must-survive clause WHOLE at the same
#       size, because each clause is its own paragraph and "\n\n" is the first
#       separator (the answer survived the chunking).
#   [ ] You can explain why sliding-window's overlap reduces splits vs fixed.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import re

# --- Tokenizer: BGE if available, whitespace fallback so the file always runs ---
try:
    from transformers import AutoTokenizer

    _HF = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

    def encode(text: str) -> list[int]:
        return _HF.encode(text, add_special_tokens=False)

    def decode(ids: list[int]) -> str:
        return _HF.decode(ids)

    TOKENIZER_NAME = "BAAI/bge-large-en-v1.5 (WordPiece)"
except Exception:  # transformers not installed, or offline — fall back.
    # A whitespace word tokenizer. ids ARE the word strings' indices into a list
    # we carry alongside; to keep decode() simple we encode to the words list and
    # decode by joining. We model "tokens" as words here.
    _VOCAB: list[str] = []

    def encode(text: str) -> list[int]:
        ids = []
        for word in text.split(" "):
            _VOCAB.append(word)
            ids.append(len(_VOCAB) - 1)
        return ids

    def decode(ids: list[int]) -> str:
        return " ".join(_VOCAB[i] for i in ids)

    TOKENIZER_NAME = "whitespace-word fallback (install `transformers` for BGE tokens)"


# --- The document: clauses as paragraphs, exactly like a real contract --------
DOCUMENT = """SERVICES AGREEMENT

1. This Agreement is entered into between the Company and the Contractor as of the effective date written above.

7. The annual fee shall be paid in twelve equal monthly installments due on the first business day of each month.

9. All confidential information must be protected for five years after termination of this Agreement.

12. The Contractor shall maintain professional liability insurance of one million dollars throughout the term.

14. Either party may terminate this Agreement upon thirty days written notice to the other party.

18. This Agreement is governed by the laws of the State of Delaware without regard to conflict of laws.

27. Any dispute arising under this Agreement shall be resolved by binding arbitration in San Francisco."""

# The phrases whose survival we check. If a phrase appears whole inside one chunk,
# the answer "survived the chunking"; if it's split across two, retrieval is dead.
MUST_SURVIVE = {
    "termination clause": "terminate this Agreement upon thirty days written notice",
    "five-year confidentiality": "protected for five years after termination",
    "insurance amount": "professional liability insurance of one million dollars",
}


# --- Chunker 1: fixed token-window --------------------------------------------
def fixed_token_chunks(text: str, size: int = 512) -> list[str]:
    """Split into chunks of at most `size` tokens, no overlap. The baseline."""
    ids = encode(text)
    return [decode(ids[i:i + size]) for i in range(0, len(ids), size)]


# --- Chunker 2: sliding-window with overlap -----------------------------------
def sliding_window_chunks(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Token windows of `size` that overlap their neighbour by `overlap` tokens."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    ids = encode(text)
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(ids), step):
        window = ids[start:start + size]
        if not window:
            break
        chunks.append(decode(window))
        if start + size >= len(ids):
            break
    return chunks


# --- Chunker 3: recursive (LangChain-style separator hierarchy) ----------------
def _token_len(text: str) -> int:
    return len(encode(text))


def recursive_chunks(
    text: str,
    size: int = 512,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[str]:
    """Split on a hierarchy of separators; recurse into pieces still over `size`.

    Boundaries land on paragraph breaks first, then lines, then sentences, then
    words, falling to raw characters only for pathological input. This is why a
    clause that IS a paragraph survives whole: "\\n\\n" cuts around it.
    """
    if _token_len(text) <= size:
        return [text] if text.strip() else []

    sep = separators[0]
    rest = separators[1:]
    if sep == "":
        # Last resort: hard split by tokens.
        ids = encode(text)
        return [decode(ids[i:i + size]) for i in range(0, len(ids), size)]

    pieces = text.split(sep)
    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = piece if not buffer else buffer + sep + piece
        if _token_len(candidate) <= size:
            buffer = candidate
        else:
            if buffer.strip():
                chunks.append(buffer)
            # The lone piece is still too big -> recurse with finer separators.
            if _token_len(piece) > size:
                chunks.extend(recursive_chunks(piece, size, rest))
                buffer = ""
            else:
                buffer = piece
    if buffer.strip():
        chunks.append(buffer)
    return chunks


# --- Boundary inspection ------------------------------------------------------
def survival_report(chunks: list[str]) -> dict[str, str]:
    """For each must-survive phrase: WHOLE (in one chunk), SPLIT (across two), or
    MISSING (in none). A normalized whitespace match keeps it robust to chunk
    boundaries inserting/removing spaces."""
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    norm_chunks = [norm(c) for c in chunks]
    report = {}
    for label, phrase in MUST_SURVIVE.items():
        p = norm(phrase)
        if any(p in c for c in norm_chunks):
            report[label] = "WHOLE"
        else:
            # Split detection: does a prefix end one chunk and a suffix start the next?
            words = p.split(" ")
            found_split = False
            for i in range(1, len(words)):
                head = " ".join(words[:i])
                tail = " ".join(words[i:])
                for a, b in zip(norm_chunks, norm_chunks[1:]):
                    if a.endswith(head) and b.startswith(tail):
                        found_split = True
                        break
                if found_split:
                    break
            report[label] = "SPLIT" if found_split else "MISSING"
    return report


def run_strategy(name: str, chunks: list[str]) -> bool:
    print(f"\n=== {name} ===")
    print(f"  chunks produced: {len(chunks)}")
    for i, ch in enumerate(chunks):
        preview = re.sub(r"\s+", " ", ch).strip()
        print(f"   [{i}] ({_token_len(ch):>3} tok) {preview[:64]}")
    report = survival_report(chunks)
    all_whole = True
    print("  boundary report:")
    for label, verdict in report.items():
        mark = "OK " if verdict == "WHOLE" else "!! "
        if verdict != "WHOLE":
            all_whole = False
        print(f"    {mark}{label:24s} -> {verdict}")
    return all_whole


def main() -> int:
    print(f"tokenizer: {TOKENIZER_NAME}")
    print(f"document tokens: {_token_len(DOCUMENT)}")

    # A deliberately SMALL size so fixed windows split clauses and recursive does not.
    SIZE = 24
    OVERLAP = 8

    fixed = fixed_token_chunks(DOCUMENT, size=SIZE)
    sliding = sliding_window_chunks(DOCUMENT, size=SIZE, overlap=OVERLAP)
    recursive = recursive_chunks(DOCUMENT, size=SIZE)

    fixed_ok = run_strategy(f"fixed token-window (size={SIZE})", fixed)
    sliding_ok = run_strategy(
        f"sliding-window (size={SIZE}, overlap={OVERLAP})", sliding
    )
    recursive_ok = run_strategy(f"recursive (size={SIZE})", recursive)

    print("\n==================== VERDICT ====================")
    print(f"  fixed     : all answers whole? {fixed_ok}")
    print(f"  sliding   : all answers whole? {sliding_ok}")
    print(f"  recursive : all answers whole? {recursive_ok}")
    if recursive_ok and not fixed_ok:
        print("  LESSON CONFIRMED: at this size, fixed windows SPLIT at least one")
        print("  answer while recursive keeps every clause WHOLE — because each")
        print("  clause is a paragraph and '\\n\\n' is the first separator. That is")
        print("  'the answer survived the chunking', made visible.")
        return 0
    print("  (Adjust SIZE: too large and fixed also keeps clauses whole; too small")
    print("   and even recursive must split. The interesting regime is in between.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact token counts depend on the active tokenizer)
# -----------------------------------------------------------------------------
#
# tokenizer: BAAI/bge-large-en-v1.5 (WordPiece)
# document tokens: 171
#
# === fixed token-window (size=24) ===
#   chunks produced: 8
#    [0] ( 24 tok) SERVICES AGREEMENT 1. This Agreement is entered into between ...
#    ...
#   boundary report:
#     OK termination clause        -> WHOLE
#     !! five-year confidentiality  -> SPLIT      <-- fixed sliced through clause 9
#     OK insurance amount          -> WHOLE
#
# === sliding-window (size=24, overlap=8) ===
#   chunks produced: 10
#    ...
#   boundary report:
#     OK termination clause        -> WHOLE
#     OK five-year confidentiality  -> WHOLE      <-- overlap rescued the straddle
#     OK insurance amount          -> WHOLE
#
# === recursive (size=24) ===
#   chunks produced: 8
#    [2] ( 19 tok) 9. All confidential information must be protected for five ...
#   boundary report:
#     OK termination clause        -> WHOLE
#     OK five-year confidentiality  -> WHOLE      <-- "\n\n" cut AROUND the clause
#     OK insurance amount          -> WHOLE
#
# ==================== VERDICT ====================
#   fixed     : all answers whole? False
#   sliding   : all answers whole? True
#   recursive : all answers whole? True
#   LESSON CONFIRMED: ... the answer survived the chunking, made visible.
#
# NOTE: which clause fixed-windows split depends on the exact token boundaries, so
# the SPLIT row may move — but the SHAPE is invariant: fixed splits something at a
# small size, overlap rescues most straddles, and recursive respects the paragraph
# structure and keeps each clause whole for free. Crank SIZE up to 512 and ALL
# three keep clauses whole (the clauses are tiny) — which is why size is a
# hyperparameter you sweep (Exercise 3), not a constant you assume.
# -----------------------------------------------------------------------------
