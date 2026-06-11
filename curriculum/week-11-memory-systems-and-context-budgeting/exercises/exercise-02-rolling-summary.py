#!/usr/bin/env python3
# Exercise 2 — Rolling-summary episodic memory (tokens saved vs facts retained)
#
# Goal: Build the rolling-summary episodic memory from Lecture 1 §3 and MEASURE the
#       trade it makes: a rolling summary keeps the conversation's token footprint
#       BOUNDED (instead of growing linearly) but is LOSSY — each summarization pass
#       can drop a detail. You'll see the tokens-saved and watch which planted facts
#       survive the summary, motivating why durable facts ALSO go to semantic memory.
#
# Estimated time: 50 minutes. Runnable.
#
# WHAT THIS DEPENDS ON (read before running)
#
#   * REQUIRED: nothing but the Python stdlib. By default this uses a DETERMINISTIC
#     stub summarizer (extractive: it keeps sentences mentioning planted keywords),
#     so the mechanics run anywhere, today, with no API key.
#   * REAL SUMMARIZER (optional): set ANTHROPIC_API_KEY and pass --real to summarize
#     with claude-sonnet-4-6 (a cheap, capable summarizer). The SHAPE of the result
#     is the same; the stub is for debugging the harness deterministically first.
#   * Token counting: the stub counts whitespace words as a tokenizer stand-in so the
#     file needs no model. With --real it uses the model's count_tokens (the right
#     way — Lecture 2 §2). The LESSON (bounded vs linear) is identical either way.
#
# ACCEPTANCE CRITERIA
#
#   [ ] Prints, per turn, the NAIVE footprint (all turns) vs the ROLLING footprint
#       (summary + recent window); the rolling footprint stays bounded while naive
#       grows linearly.
#   [ ] Reports which planted facts SURVIVED the rolling summary and which were lost
#       (the lossiness that motivates the semantic tier).
#   [ ] You can state the trade in one sentence: rolling summary bounds tokens but is
#       lossy, so durable facts must ALSO be promoted to semantic memory.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import argparse
import os
import re

RECENT_WINDOW = 4          # turns kept verbatim (the rest are summarized)

# A 40-turn conversation: a few PLANTED durable facts, the rest is filler.
PLANTED = {
    3: "My project is called Helios.",
    7: "I prefer to write code in Python.",
    12: "My company is on the enterprise tier.",
    16: "My deadline is the 14th.",          # NO durable keyword -> stub drops it
}
PLANTED_FACTS = {
    "project name": "Helios",
    "language": "Python",
    "tier": "enterprise",
    "deadline": "14th",                       # the one the lossy summary loses
}


def build_conversation(n_turns=40) -> list[dict]:
    turns = []
    for i in range(1, n_turns + 1):
        if i in PLANTED:
            content = PLANTED[i]
        else:
            content = (f"Turn {i}: some ordinary conversational content "
                       f"about scheduling and logistics, nothing durable here.")
        turns.append({"turn": i, "role": "user", "content": content})
    return turns


# --- Token counting: stub (words) or real (model tokenizer) --------------------
def make_counter(real: bool):
    if real:
        import anthropic
        client = anthropic.Anthropic()

        def count(text: str) -> int:
            return client.messages.count_tokens(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": text or " "}],
            ).input_tokens
        return count
    # Stub: whitespace words as a tokenizer stand-in (deterministic, no API).
    return lambda text: len(text.split())


# --- Summarizer: stub (extractive) or real (LLM) -------------------------------
def make_summarizer(real: bool):
    if real:
        import anthropic
        client = anthropic.Anthropic()

        def summarize(prev_summary: str, new_turns: list[dict]) -> str:
            transcript = "\n".join(f"{t['role']}: {t['content']}" for t in new_turns)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=("You maintain a concise running summary of a conversation. "
                        "Update it with the new turns. Keep durable facts (names, "
                        "preferences, decisions); drop pleasantries and logistics."),
                messages=[{"role": "user", "content":
                           f"Current summary:\n{prev_summary}\n\nNew turns:\n"
                           f"{transcript}\n\nReturn the updated summary."}],
            )
            return next(b.text for b in msg.content if b.type == "text").strip()
        return summarize

    # Stub summarizer: DELIBERATELY LOSSY — it keeps only sentences that look
    # "durable" (contain a capitalized proper noun or a preference verb), which
    # models the real summarizer's tendency to drop and sometimes lose facts.
    DURABLE = re.compile(r"\b(called|prefer|tier|named|project|company)\b", re.I)

    def summarize(prev_summary: str, new_turns: list[dict]) -> str:
        kept = [t["content"] for t in new_turns if DURABLE.search(t["content"])]
        combined = ([prev_summary] if prev_summary else []) + kept
        # Cap the summary length (bounded!) by keeping the most recent durable lines.
        return " ".join(combined)[-600:]
    return summarize


def fact_survives(fact_value: str, text: str) -> bool:
    return fact_value.lower() in text.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="use claude-sonnet-4-6 (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("--real needs ANTHROPIC_API_KEY; falling back to the deterministic stub.")
        args.real = False

    count = make_counter(args.real)
    summarize = make_summarizer(args.real)
    turns = build_conversation()

    summary = ""
    print(f"{'turn':>4} | {'naive tok':>9} | {'rolling tok':>11} | note")
    print("-" * 52)

    for i, _ in enumerate(turns, start=1):
        seen = turns[:i]
        recent = seen[-RECENT_WINDOW:]
        older = seen[:-RECENT_WINDOW]
        if older:
            summary = summarize(summary, older[-1:])     # fold the just-aged-out turn

        naive_tok = count("\n".join(t["content"] for t in seen))
        rolling_tok = count(summary) + count(
            "\n".join(t["content"] for t in recent))

        note = ""
        if i in PLANTED:
            note = f"<- planted: {PLANTED[i]!r}"
        if i in (10, 20, 40):
            note = f"(checkpoint) naive grows, rolling bounded"
        print(f"{i:>4} | {naive_tok:>9} | {rolling_tok:>11} | {note}")

    # --- Which planted facts SURVIVED the rolling summary at the end? -----------
    final_recent = turns[-RECENT_WINDOW:]
    rolling_state = summary + " " + " ".join(t["content"] for t in final_recent)
    print("\nfact survival in rolling memory after 40 turns:")
    survived = 0
    for label, value in PLANTED_FACTS.items():
        ok = fact_survives(value, rolling_state)
        survived += ok
        print(f"  {'OK ' if ok else '!! '}{label:14s} ({value!r}) "
              f"-> {'SURVIVED' if ok else 'LOST'}")

    naive_final = count("\n".join(t["content"] for t in turns))
    rolling_final = count(rolling_state)
    print(f"\ntoken footprint after 40 turns: naive={naive_final}  "
          f"rolling={rolling_final}  ({naive_final / max(rolling_final,1):.1f}x smaller)")
    print(f"facts surviving the rolling summary: {survived}/{len(PLANTED_FACTS)}")
    print("\nLESSON: the rolling summary keeps the footprint BOUNDED (constant-ish) "
          "while naive append grows LINEARLY — that's the win. But the summary is "
          "LOSSY: facts it drops are gone. So durable facts must ALSO be extracted "
          "to SEMANTIC memory (Exercise 3 / the mini-project), which is what makes a "
          "turn-3 fact reliably survive to turn 38. Rolling summary bounds tokens; "
          "the semantic tier guarantees recall. You need both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape; exact tokens depend on stub-vs-real and the summarizer)
# -----------------------------------------------------------------------------
#
# turn | naive tok | rolling tok | note
# ----------------------------------------------------
#    1 |        13 |          13 |
#    3 |        ~30 |         ~25 | <- planted: 'My project is called Helios.'
#   10 |       ~120 |         ~45 | (checkpoint) naive grows, rolling bounded
#   20 |       ~240 |         ~55 | (checkpoint) naive grows, rolling bounded
#   40 |       ~480 |         ~70 | (checkpoint) naive grows, rolling bounded
#
# fact survival in rolling memory after 40 turns:
#   OK project name   ('Helios')     -> SURVIVED   (matched a durable keyword)
#   OK language       ('Python')     -> SURVIVED   (matched a durable keyword)
#   OK tier           ('enterprise') -> SURVIVED   (matched a durable keyword)
#   !! deadline       ('14th')       -> LOST       <- no durable keyword; dropped
#
# token footprint after 40 turns: naive=500  rolling=71  (7.0x smaller)
# facts surviving the rolling summary: 3/4
#
# LESSON: ... the rolling summary bounds tokens but is LOSSY (here it lost the tier
# fact), so durable facts must ALSO go to semantic memory. Rolling summary bounds
# tokens; the semantic tier guarantees recall. You need both.
#
# READ IT: naive tokens climb every turn (you re-send the whole history each call —
# the API is stateless); rolling tokens flatten because old turns collapse into a
# bounded summary. That flattening is the whole point of episodic compression. The
# LOST fact is the whole point of the SEMANTIC tier: don't trust a summary to keep
# what matters — extract it. Run with --real to watch a real summarizer make the
# same trade with its own (different) drops.
# -----------------------------------------------------------------------------
