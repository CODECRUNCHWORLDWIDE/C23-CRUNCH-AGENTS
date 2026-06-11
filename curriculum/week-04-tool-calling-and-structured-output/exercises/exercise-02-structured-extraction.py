#!/usr/bin/env python3
# Exercise 2 — Structured extraction three ways
#
# Goal: Extract the SAME typed record (a Contact) three different ways and diff them:
#   (A) Anthropic messages.parse() with a Pydantic model      — the vendor JSON-mode path
#   (B) Anthropic raw output_config.format with a JSON Schema  — the same thing, by hand
#   (C) outlines grammar-constrained decoding on a local Qwen  — the local path
#
#       The lesson: all three give you a schema-conforming object, but the GUARANTEE
#       differs. (A) and (B) are vendor-enforced; (C) is decoder-enforced — the local
#       model is STRUCTURALLY INCAPABLE of emitting an invalid token. You will see (C)
#       never produce malformed JSON, where an unconstrained local model sometimes does.
#
# Estimated time: 45 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#     pip install anthropic pydantic outlines
#     export ANTHROPIC_API_KEY=sk-ant-...
#     python3 exercise-02-structured-extraction.py
#
#   Path (C) downloads Qwen2.5-7B the first time (several GB, needs a GPU or patience).
#   If you can't run it, set SKIP_LOCAL = True below — paths (A) and (B) still run and
#   still teach the vendor JSON-mode half.
#
# ACCEPTANCE CRITERIA
#
#   [ ] All enabled paths produce a Contact with the same name/email/plan/demo_requested.
#   [ ] You can state why (C)'s output CANNOT be malformed JSON (decoder masking).
#   [ ] You ran the "unconstrained baseline" at the bottom and saw it occasionally
#       wrap JSON in prose or miss a field — the failure (C) makes impossible.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import json

from pydantic import BaseModel

# Set True if you cannot run a local 7B (no GPU / no disk). (A) and (B) still run.
SKIP_LOCAL = False

PROMPT = (
    "Jane Doe reached out from jane@acme.com. She wants the Enterprise plan and "
    "explicitly asked for a live demo next week."
)


class Contact(BaseModel):
    name: str
    email: str
    plan: str
    demo_requested: bool


# -----------------------------------------------------------------------------
# Path A — Anthropic messages.parse() with a Pydantic model (the clean vendor path)
# -----------------------------------------------------------------------------
def extract_with_parse() -> Contact:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Extract the contact: {PROMPT}"}],
        output_format=Contact,
    )
    # response.parsed_output is a validated Contact instance — no json.loads, no try/except.
    return response.parsed_output


# -----------------------------------------------------------------------------
# Path B — Anthropic raw output_config.format with a JSON Schema (the same, by hand)
# -----------------------------------------------------------------------------
def extract_with_output_config() -> Contact:
    import anthropic

    client = anthropic.Anthropic()
    # Pydantic generates the JSON Schema for us; we hand it to the API directly.
    schema = Contact.model_json_schema()
    # The structured-output engine requires a closed object.
    schema["additionalProperties"] = False
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Extract the contact: {PROMPT}"}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    # The first text block is guaranteed to be valid JSON for `schema`.
    text = next(b.text for b in response.content if b.type == "text")
    return Contact.model_validate_json(text)


# -----------------------------------------------------------------------------
# Path C — outlines grammar-constrained decoding on a local Qwen (decoder-enforced)
# -----------------------------------------------------------------------------
def extract_with_outlines() -> Contact:
    import outlines

    # The model is loaded once; the generator masks the sampler at each step so only
    # tokens that keep the JSON valid for Contact can be emitted. It CANNOT produce
    # invalid JSON — there is no retry/repair step because there is nothing to repair.
    model = outlines.models.transformers("Qwen/Qwen2.5-7B-Instruct")
    generator = outlines.generate.json(model, Contact)
    return generator(f"Extract the contact as JSON: {PROMPT}")


# -----------------------------------------------------------------------------
# The unconstrained baseline — what (C) protects you FROM.
# -----------------------------------------------------------------------------
def unconstrained_local_baseline() -> str:
    """Ask the local model for JSON with NO grammar constraint. Sometimes it wraps the
    JSON in prose ('Here is the contact: {...}') or drops a field. Run it a few times."""
    import ollama

    resp = ollama.chat(
        model="qwen2.5:7b-instruct",
        messages=[{
            "role": "user",
            "content": (
                "Extract the contact as a JSON object with keys "
                "name, email, plan, demo_requested. Reply with ONLY the JSON.\n\n"
                + PROMPT
            ),
        }],
    )
    return resp["message"]["content"]


def main() -> None:
    print("=== Path A: messages.parse (vendor, Pydantic) ===")
    a = extract_with_parse()
    print(a.model_dump())

    print("\n=== Path B: output_config.format (vendor, raw schema) ===")
    b = extract_with_output_config()
    print(b.model_dump())

    assert a == b, "A and B should extract the same record"
    print("\nA == B  ✓  (same record, two vendor mechanisms)")

    if not SKIP_LOCAL:
        print("\n=== Path C: outlines grammar-constrained (local, decoder-enforced) ===")
        c = extract_with_outlines()
        print(c.model_dump())
        # c may differ slightly on a 7B model, but it is ALWAYS a valid Contact —
        # the decoder could not emit anything else.
        print("C is a valid Contact by construction  ✓")

        print("\n=== Unconstrained local baseline (run a few times) ===")
        for i in range(3):
            raw = unconstrained_local_baseline()
            try:
                Contact.model_validate_json(raw.strip())
                verdict = "parsed OK"
            except Exception as e:
                verdict = f"FAILED to parse: {type(e).__name__}"
            print(f"  run {i + 1}: {verdict}  | raw[:60]={raw.strip()[:60]!r}")
        print("\nThe baseline failures are exactly what grammar-constrained decoding "
              "makes IMPOSSIBLE. That is the whole value of path C.")
    else:
        print("\n(SKIP_LOCAL=True — skipped paths C and the baseline.)")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Expected output (shape; exact bytes vary by model and timing)
# -----------------------------------------------------------------------------
#
# === Path A: messages.parse (vendor, Pydantic) ===
# {'name': 'Jane Doe', 'email': 'jane@acme.com', 'plan': 'Enterprise', 'demo_requested': True}
#
# === Path B: output_config.format (vendor, raw schema) ===
# {'name': 'Jane Doe', 'email': 'jane@acme.com', 'plan': 'Enterprise', 'demo_requested': True}
#
# A == B  ✓  (same record, two vendor mechanisms)
#
# === Path C: outlines grammar-constrained (local, decoder-enforced) ===
# {'name': 'Jane Doe', 'email': 'jane@acme.com', 'plan': 'Enterprise', 'demo_requested': True}
# C is a valid Contact by construction  ✓
#
# === Unconstrained local baseline (run a few times) ===
#   run 1: parsed OK             | raw[:60]='{"name": "Jane Doe", "email": "jane@acme.com", "plan":...'
#   run 2: FAILED to parse: ...  | raw[:60]='Here is the extracted contact:\n{"name": "Jane Doe"...'
#   run 3: parsed OK             | raw[:60]='{"name": "Jane Doe", ...'
#
# The lesson is invariant even if your exact runs all happen to parse: the vendor paths
# and the grammar-constrained path GUARANTEE a valid record; the unconstrained baseline
# only HOPES for one. In production, "hopes for" means a repair loop you don't want.
# -----------------------------------------------------------------------------
