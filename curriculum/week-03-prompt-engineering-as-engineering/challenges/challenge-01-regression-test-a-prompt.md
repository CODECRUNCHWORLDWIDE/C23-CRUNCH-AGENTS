# Challenge 1 — Regression-Test a Prompt

**Time estimate:** ~90 minutes.

## Problem statement

You inherit a customer-support triage prompt that is *visibly failing* in production: it misroutes multi-issue tickets, leaks its own instructions when a user asks it to, and is inconsistent on the ambiguous middle. Your job is the syllabus hands-on lab: **build a `promptfoo` test harness with 30 golden examples, iterate through six prompt versions, commit each to git with its measured pass rate, and deliver a regression-tested prompt with reproducible scores.**

This mirrors the real skill exactly. In production nobody accepts "I rewrote the prompt and it's better now." They accept "v6 passes 28/30, here is the diff from v5, here is the SHA you can revert to, here is the command to reproduce the score." This challenge is that deliverable, rehearsed once so you can do it for real.

## The starting prompt (use this as v1)

This is your `prompts/support-triage.v1.txt` — deliberately weak. Do **not** improve it before measuring; you must baseline the failure first.

```
You are a helpful support bot. Read the ticket and say what kind of issue it is.
Categories: billing, technical, account, other.
```

## The task: rules

1. **Build a 30-example golden set.** Create `tests/golden-30.yaml` (or a CSV promptfoo reads). Each example is a ticket + an assertion on the expected property. Span the space:
   - ~18 clear single-category cases (≥4 per category) — the easy middle; a regression here must scream.
   - ~8 ambiguous / multi-issue cases — where versions actually differ (e.g., "double-charged and can't log in" → `billing`).
   - ~4 adversarial cases — injection / instruction-leak attempts that must be refused (assert the output does **not** reveal the prompt and routes to `other`).
2. **Wire a `promptfooconfig.yaml`** with the prompt versions as `prompts:`, a provider (`anthropic:messages:claude-haiku-4-5`, or an Ollama provider if you have no key), and the golden set as `tests:`. The pass rate comes from `npx promptfoo eval -c promptfooconfig.yaml`. *(You may instead drive the run with your own `exercise-02` harness if you prefer Python — but the golden set and the per-version pass rate are non-negotiable.)*
3. **Iterate six versions, v1 → v6.** Each version is a separate file (`support-triage.v1.txt` … `v6.txt`) and a separate **git commit**, with the **measured pass rate in the commit message**. Each version must address a *named* failure cluster from the previous version's results — not a random reword.
4. **Enforce the regression gate.** A version that drops a previously-passing case is a regression: record it, and either fix it before moving on or explicitly revert. Your final v6 must not regress any case that v5 passed.
5. **Deliver a reproducible score report.** A `SCORES.md` table (version, pass count, rate, delta, commit SHA) plus the exact command to reproduce each number. Anyone with your repo runs one command and gets your matrix.

## Suggested iteration arc (yours may differ — the arc is the point)

| Version | What you change | Typical failure it targets |
|---|---|---|
| **v1** | the weak starter, as given | baseline — measure, don't fix yet |
| **v2** | tighten output contract: "exactly one lowercase word, nothing else" | format noise, extra words |
| **v3** | add the category set with one-line definitions | confusion at category boundaries |
| **v4** | add 3–4 few-shot examples covering the hard boundaries | ambiguous single-category cases |
| **v5** | add the multi-issue rule (primary financial impact → billing) | misrouted multi-issue tickets |
| **v6** | add the injection-refusal rule + an example | instruction-leak / jailbreak cases |

Note: real iteration is not monotonic — you may *regress* on one version (e.g., v3 over-tightens and breaks an edge case). **That is good data.** Record the regression, show the gate catching it, and recover. A clean six-step climb with no regression anywhere is suspicious; a reviewer will assume you didn't test the hard cases.

## Deliverables

- [ ] `prompts/support-triage.v1.txt` … `v6.txt` — six versions, six commits, pass rate in each commit message.
- [ ] `tests/golden-30.yaml` (or `.csv`) — 30 golden examples: ~18 clear, ~8 ambiguous/multi-issue, ~4 adversarial, each with an assertion (not an exact-string-only match for the refusal cases).
- [ ] `promptfooconfig.yaml` — runnable with `npx promptfoo eval`; provider, prompts, tests wired.
- [ ] `SCORES.md` — the version × pass-rate table with deltas, SHAs, and the reproduce command; at least one row shows a regression caught (or a documented near-miss).
- [ ] A 1-paragraph **structured-review note** on the v5→v6 diff using the `exercise-01` checklist (output contract by example? rules non-contradictory? refusal covered? regression-free?).
- [ ] Committed and pushed to a repo named `c23-week-03-support-triage-<yourhandle>`.

## Acceptance criteria

- [ ] The golden set has **30 examples** across the three bands, with the ~4 adversarial cases asserting *refusal as a property* (no instruction leak), not a fixed string.
- [ ] **Six versions**, each a **separate commit**, each commit message carrying the **measured pass rate**.
- [ ] v6's pass rate is **materially higher** than v1's and v6 **does not regress** any case v5 passed.
- [ ] At least one intermediate version's regression (or a documented one you avoided) appears in `SCORES.md` — proving the gate is doing work, not decoration.
- [ ] The score is **reproducible**: `git checkout <sha> && npx promptfoo eval -c promptfooconfig.yaml` (or your Python harness equivalent) reproduces the recorded rate.

## The trap (read after a first attempt)

The trap is **iterating against the model's outputs instead of against the golden set.** It's tempting to run the prompt, read three replies, tweak, and call it better — the exact string-literal habit the week argues against. If you do that, you'll fix the case in front of you and silently break two you fixed earlier, and you'll never know, because you never re-ran the full 30. The whole point of the harness is that *every* version runs against *all* 30, so "better" means "higher rate AND no regression," computed, not eyeballed. If your `SCORES.md` shows a smooth climb with no version ever dipping, ask yourself honestly whether your golden set actually includes the hard cases — a suite that never goes red isn't testing anything.

## Stretch

- **Add a cost column.** For each version record median tokens-in/out (your `toklab` instinct). A v6 that gains 4 points but doubles token cost is a trade-off a reviewer should see, not a free win.
- **Bisect a planted regression.** Deliberately ship a v7 that breaks two cases, then use the harness + `git bisect` to mechanically find the version that broke them. Prompt regressions are real; finding them by tooling, not by memory, is the production skill.
- **Promote through a registry.** Put v6 behind Langfuse prompt management with label `production`, then practice rolling back to v5 by moving the label — no redeploy. That's the runtime half of the pipeline (Lecture 2 §4).

## Why this matters

The Phase I capstone milestone requires, in the syllabus's own words, "prompts versioned in git; promptfoo regression tests committed." This challenge *is* that requirement, scoped to one prompt. Every applied-AI-engineering role will, in some form, ask "how do you know your prompt change didn't break something?" — and the answer they're listening for is exactly this: a golden set, a pass rate, a regression gate, a SHA. Not "I tested a few and it looked good."
