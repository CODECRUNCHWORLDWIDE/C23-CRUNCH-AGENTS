# Week 4 Homework

Six problems that revisit the week's topics and force the tool-calling discipline into your fingers. The full set should take about **5 hours**. Work in your Week 4 Git repository (the same workspace as the exercises and the `crunch_tools` mini-project) so every problem produces at least one commit you can point to at the Phase I architecture review in Week 6.

The headline deliverable is **Problem 4 — the tool-call accuracy benchmark write-up**, called out explicitly in the syllabus. Treat it as the artifact a reviewer reads, not a journal entry.

Each problem includes:

- A short **problem statement**.
- **Acceptance criteria** so you know when you're done.
- A **hint** if you get stuck.
- An **estimated time**.

Have `ANTHROPIC_API_KEY` exported and `ollama serve` running with `qwen2.5:7b-instruct` pulled. Problems 1, 2, 4, and 6 run against both paths.

---

## Problem 1 — The schema audit table

**Problem statement.** Take the four tools from the mini-project (calculator, read_file, fetch_url, run_python). For **each**, write a row in `notes/week-04/schema-audit.md` with these columns:

| Tool | Required fields | Has `enum`? | `additionalProperties: false`? | Every property described? | Strict-eligible? |
|---|---|---|---|---|---|

The **Strict-eligible** column is your judgement: a tool is strict-eligible only if its schema is fully closed (`additionalProperties: false`, every property in `required` or handled, no numeric/length bounds, no recursion). For any tool you mark *not* strict-eligible, give the one-line reason.

**Acceptance criteria.**

- `notes/week-04/schema-audit.md` exists with one row per tool (four rows).
- Each schema actually validates: run `jsonschema.Draft202012Validator.check_schema(schema)` on each and paste the (clean) result.
- At least one tool is correctly marked *not* strict-eligible with a reason, or you argue all four are eligible and why.
- Committed.

**Hint.** `run_python` is the interesting row — its schema is trivially strict-eligible (just a `code` string), but strict mode does *nothing* for its safety, which lives in isolation. Note that distinction in the reason column.

**Estimated time.** 40 minutes.

---

## Problem 2 — Make a stubborn model call the tool

**Problem statement.** Write a tool whose description is *deliberately vague* (`{"name": "compute", "description": "computes things"}` wrapping the calculator). Ask `claude-haiku-4-5` and `qwen2.5:7b-instruct` "what is 4096 / 64?" and record whether each calls the tool or answers from memory. Then rewrite the description to be specific ("Evaluate an arithmetic expression for any exact numeric computation the user requests; use this rather than computing in your head"). Re-run both. Capture before/after.

**Acceptance criteria.**

- `notes/week-04/description-matters.md` shows the vague and specific descriptions and, for each model, whether it called the tool before and after.
- You state, in one sentence, why the description is "the prompt" for tool selection.
- Committed.

**Hint.** The small model is *more* sensitive to a vague description than the frontier model — it's likelier to skip the tool and answer (often wrongly) from memory. That gap is exactly why description quality matters more, not less, on local models.

**Estimated time.** 40 minutes.

---

## Problem 3 — Structured extraction, vendor vs grammar

**Problem statement.** Reuse the `Contact` model from Exercise 2. Extract the same record (a) with `messages.parse()` on `claude-opus-4-8` and (b) with `outlines` grammar-constrained decoding on the local Qwen. Then run an **unconstrained** local extraction (plain "reply with JSON") **20 times** and count how many produce parseable, schema-valid JSON. Report the failure rate.

**Acceptance criteria.**

- `notes/week-04/extraction-guarantee.md` records the parsed record from both constrained paths (they should match on the fields).
- It reports the unconstrained local failure count out of 20 (e.g. "3/20 failed to parse — JSON wrapped in prose twice, a missing field once").
- You state why the constrained paths have a **guarantee** and the unconstrained path only has a **hope**.
- Committed.

**Hint.** If your local unconstrained run happens to pass 20/20, that doesn't disprove the point — run it 50 times, or use a harder schema (add a nested field). The guarantee/hope distinction holds regardless of one lucky batch.

**Estimated time.** 1 hour.

---

## Problem 4 — The tool-call accuracy benchmark (headline deliverable)

**Problem statement.** This is the syllabus deliverable. Using your mini-project registry (or a minimal version with calculator + read_file + fetch_url), build a **fixed benchmark** of at least 25 tasks across classes (`calc`, `file`, `web`, `multi`, `security`), each with a deterministic expected answer. Run all tasks through `claude-opus-4-8` (or `claude-haiku-4-5`) and `qwen2.5:7b-instruct` over the *same* registry. Write a one-page benchmark report at `notes/week-04/tool-accuracy-benchmark.md` against this template:

1. **Setup** — one paragraph: the tools, the two models, the task count and class mix, the grading rule (final answer contains expected; security task = refused counts as pass).
2. **Results table** — passed/total, accuracy, avg turns, avg latency, est cost per model.
3. **Per-class breakdown** — where the local model is weak (usually `multi` and `security`).
4. **Failure analysis** — quote two actual local-model failures and name the cause (hallucinated tool name? malformed args? skipped a parallel call?).
5. **The routing implication** — one sentence: given this gap, which task classes would you route to the frontier model and which to local?

**Acceptance criteria.**

- `notes/week-04/tool-accuracy-benchmark.md` exists, fits on roughly one page, and hits all five headings.
- The results are from real runs, with the grading method documented — not estimated.
- At least two real local-model failures are quoted with a named cause.
- The routing implication is concrete (names task classes), not "use the better model when needed."
- Committed.

**Hint.** Make the grader deterministic and string-based (`expected in final_answer.lower()`); a fuzzy LLM-judge here would add noise you don't need yet (that's Week 12). The `security` class is the cheap insight: a frontier model usually refuses the path-traversal task cleanly, a local one sometimes tries to comply — and your *tool* refuses regardless, which is the whole point of hardening.

**Estimated time.** 1 hour.

---

## Problem 5 — Break a tool, then defend it

**Problem statement.** Take the **naive** `fetch_url` (no SSRF guard). Write a single adversarial prompt that, given a tool-using agent, would make the model fetch `http://169.254.169.254/latest/meta-data/` (frame it as a legitimate-looking task — "check the status page at this internal URL"). Show the naive tool would comply. Then drop in the hardened `fetch_url` and show the same prompt is refused at the *tool* layer regardless of what the model decides.

**Acceptance criteria.**

- `notes/week-04/ssrf-defense.md` contains the adversarial prompt, the naive tool's behavior (it would fetch the metadata endpoint), and the hardened tool's refusal.
- You note that the defense is at the **tool**, not the **model** — the model can be talked into anything; the tool refusing is what actually protects you. This is the Week 17 thesis, previewed.
- Committed.

**Hint.** You don't need to actually be on a cloud box with a live metadata endpoint — the hardened tool refuses `169.254.169.254` by IP class before any request goes out, which you can prove offline. For the naive side, show the code path that *would* make the request (you can stub the actual `httpx.get` so you don't hang).

**Estimated time.** 45 minutes.

---

## Problem 6 — One registry, prove the model swap

**Problem statement.** Take your registry + two adapters. Write a script `swap_demo.py` that runs the *identical* question ("Read notes.txt, then tell me 17 * 23") through the Anthropic adapter and the Ollama adapter with **no other code change** — the only difference is which adapter you pass. Capture both traces side by side.

**Acceptance criteria.**

- `notes/week-04/model-swap.md` shows both traces (the `tool_use`/`tool_result` sequence) for the same question on both models.
- You confirm the registry and the loop were byte-identical between the two runs; only the adapter argument changed.
- You note in one sentence the rule: **the tool surface must not know which model is behind it** — that's what makes a model swap a one-line change.
- Committed.

**Hint.** The classic self-own: a vendor wire quirk (Ollama's `arguments`-as-string) leaks into your loop, so you patch the loop instead of the adapter, and now the loop "knows" about Ollama. Keep the quirk in the adapter; the loop stays vendor-blind.

**Estimated time.** 35 minutes.

---

## Time budget recap

| Problem | Estimated time |
|--------:|--------------:|
| 1 — Schema audit table | 40 min |
| 2 — Make a stubborn model call the tool | 40 min |
| 3 — Extraction: vendor vs grammar | 1 h 0 min |
| 4 — Tool-call accuracy benchmark (headline) | 1 h 0 min |
| 5 — Break a tool, then defend it | 45 min |
| 6 — One registry, prove the model swap | 35 min |
| **Total** | **~5 h 0 min** |

When you've finished all six, push your repo and make sure the `crunch_tools` [mini-project](./mini-project/README.md) is in the same workspace — Week 5 imports it. Then take the [quiz](./quiz.md) with your notes closed.

---

## Rubric (for graders and self-assessment)

This homework is graded on the canonical four-axis weekly rubric, specialized to Week 4.

| Axis | Weight | What "meets" looks like |
|---|---:|---|
| **Correctness** | 30% | The schemas validate, the tools dispatch correctly, the model-swap script runs identically on both vendors, and the benchmark grades tasks correctly. |
| **Engineering quality** | 25% | Hardened tools (no `eval`/`exec` on model input, sandbox-confined file reads, SSRF-guarded fetches), one registry, vendor code isolated to adapters, sensible error handling. |
| **Measurement** | 25% | Problem 4's benchmark reports accuracy, cost, latency, and a per-class breakdown with the grading method documented; Problem 3 reports a real failure rate. A number, not a vibe. |
| **Write-up** | 20% | The benchmark report and the defense notes explain the design choice, the metric, and at least one observed failure mode, in prose a reviewer can follow. |

Graders are instructed to **fail vibes-only submissions** — a benchmark with no documented grading method, or a "defense" with no demonstrated attack, is not a "meets." The headline deliverable (Problem 4) must hit all five template headings to pass.
