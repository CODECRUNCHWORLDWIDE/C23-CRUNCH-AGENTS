# Week 4 — Quiz

Thirteen questions. Take it with your lecture notes closed. Aim for 11/13 before moving to Week 5. Answer key is at the bottom — don't peek.

---

**Q1.** When the Anthropic API returns `stop_reason == "tool_use"`, what has the model actually done?

- A) Run your tool and returned its output.
- B) Emitted a structured *request* to call a tool, with `name` and `input`, and stopped — your code must run the tool and send the result back.
- C) Refused the request for safety reasons.
- D) Reached `max_tokens`.

---

**Q2.** In a `tool_use` content block, what type is `block.input`, and how should you read it?

- A) A JSON string you must `json.loads` and string-match.
- B) A parsed Python dict — use it directly; never raw-string-match the serialized form (escaping varies by model).
- C) A Pydantic model.
- D) The raw bytes of the model's output.

---

**Q3.** You send a `tool_result` whose `tool_use_id` does not match any `tool_use` block in the prior assistant turn. What happens?

- A) The model ignores it and continues.
- B) The result is silently dropped.
- C) The next request returns a 400 — every `tool_use` needs exactly one matching `tool_result`, and vice versa.
- D) The model retries the tool automatically.

---

**Q4.** A single assistant turn emits two `tool_use` blocks (a Paris-weather and a Tokyo-weather). What must your next user turn contain?

- A) One `tool_result` for the first call; send the second later.
- B) Two `tool_result` blocks, one per `tool_use_id`, both in the same user turn.
- C) A single combined `tool_result` with both answers concatenated.
- D) Nothing — the model already has both answers.

---

**Q5.** Which part of a tool definition is portable across Anthropic, OpenAI, Gemini, and Ollama/Qwen?

- A) The content-block names (`tool_use`, `tool_result`).
- B) The result-message format.
- C) The JSON Schema describing the arguments (`input_schema` / `parameters`).
- D) Nothing — every vendor is entirely different.

---

**Q6.** Your tool's `description` is vague and the model won't call it on `tool_choice: auto`. What is the *first* fix?

- A) Force the tool with `tool_choice: {"type": "tool", "name": ...}`.
- B) Rewrite the description to be specific about *when* to use the tool and what each argument means — forcing a tool to make it behave is a smell.
- C) Add more tools.
- D) Switch to a bigger model.

---

**Q7.** What is the difference between JSON-mode / `output_config.format` and grammar-constrained decoding (`outlines`/`xgrammar`)?

- A) They're identical.
- B) JSON-mode constrains the *response* on a vendor API; grammar-constrained decoding masks the *sampler* on a model you run, so the decoder is structurally incapable of emitting an invalid token.
- C) Grammar-constrained decoding only works on frontier APIs.
- D) JSON-mode is slower than grammar decoding in all cases.

---

**Q8.** Why is `def calculator(expr): return eval(expr)` a catastrophic tool implementation?

- A) `eval` is slow.
- B) The argument comes from an untrusted client (the model, possibly steered by an injected document), so `eval` is arbitrary remote-code execution — e.g. `__import__('os').system('rm -rf ~')`.
- C) `eval` can't do arithmetic.
- D) Nothing is wrong; `eval` is the standard pattern.

---

**Q9.** A file-read tool is confined to a sandbox. Which check correctly rejects `path="../../etc/passwd"` *and* a symlink that points outside the sandbox?

- A) `if ".." in path: reject` — substring check.
- B) `os.path.realpath(os.path.join(SANDBOX, path))` then verify it equals `SANDBOX` or starts with `SANDBOX + os.sep` — realpath resolves `..` and symlinks before the check.
- C) `if path.startswith("/etc"): reject`.
- D) `open(path)` inside a `try/except`.

---

**Q10.** A web-fetch tool resolves the host and checks the IP, but uses `follow_redirects=True`. What attack still gets through?

- A) None — the IP check is sufficient.
- B) An allowed public URL that returns a `302` redirect to `http://169.254.169.254/` (the cloud metadata endpoint) — redirects must not be followed blindly, or re-validated at each hop.
- C) A `file://` URL — but those are always blocked.
- D) A DNS query.

---

**Q11.** Which MCP transport is the **current default for remote** servers in 2026?

- A) stdio.
- B) SSE (Server-Sent Events).
- C) streamable HTTP.
- D) gRPC.

---

**Q12.** You validate the model's tool `input` against the JSON Schema and it fails (a required field is missing). What's the right move?

- A) Crash the loop with the exception.
- B) Run the tool anyway with whatever's there.
- C) Return a `tool_result` with `is_error: true` and the error message — a capable model reads it and re-emits the call correctly on the next turn.
- D) Silently substitute a default and never tell the model.

---

**Q13.** Why can't you make a `run_python` tool safe by tightening its `input_schema`?

- A) JSON Schema doesn't support strings.
- B) Arbitrary code is arbitrary code — no schema constrains what Python *does*; safety comes from *isolation* (container/microVM, no network, resource limits), not argument validation.
- C) You can — `additionalProperties: false` makes it safe.
- D) `run_python` tools are always safe by default.

---

## Answer key

<details>
<summary>Click to reveal answers</summary>

1. **B** — A tool call is a *request*. The model emits `name` + `input` and stops; your code runs the tool and returns a `tool_result`. The model never executes anything. (Lecture 1 §2.)
2. **B** — `block.input` is a parsed dict. Current models may escape Unicode/slashes differently, so never string-match the serialized form — trust the parsed object. (Lecture 1 §2.)
3. **C** — Every `tool_use` needs exactly one matching `tool_result` (and vice versa) in the next turn; a mismatch is a 400. (Lecture 1 §2.)
4. **B** — Parallel tool use: one `tool_result` per `tool_use_id`, all in the same user turn. Order doesn't matter; the IDs do the matching. (Lecture 1 §4.)
5. **C** — The JSON Schema is portable. The envelope, result format, and reliability are not. (Lecture 1 §5.)
6. **B** — Fix the description first; forcing a tool to make it behave papers over the real bug. (Lecture 1 §3; Lecture 2 §5 decision tree.)
7. **B** — JSON-mode is vendor-side response constraint; grammar-constrained decoding masks the sampler on a local model so invalid tokens cannot be emitted at all. (Lecture 2 §1.)
8. **B** — The argument is untrusted input; `eval` on it is RCE. The AST-whitelist evaluator is the safe pattern. (Lecture 1 §8.)
9. **B** — `realpath` resolves `..` and symlinks *before* the prefix check; the `+ os.sep` guard avoids the `agent_sandbox_evil` prefix bug. (Lecture 2 §3.1.)
10. **B** — Blind redirect-following defeats the IP check (SSRF-via-redirect). Set `follow_redirects=False` or re-validate every hop. (Lecture 2 §3.2.)
11. **C** — streamable HTTP is the current remote default; stdio is local; SSE is the legacy remote transport. (Lecture 2 §2.2.)
12. **C** — Return `is_error: true` with the message; a capable model self-corrects on the next turn. Crashing or running-anyway are both wrong. (Lecture 1 §6.)
13. **B** — No schema constrains program behavior; the Python sandbox is defended by isolation, not validation. Three tools are validated; this one is isolated. (Lecture 2 §3.3; mini-project special case.)

</details>

---

If you scored under 9, re-read the lecture sections cited in the answers you missed. If you scored 11 or higher, you're ready for the [homework](./homework.md).
