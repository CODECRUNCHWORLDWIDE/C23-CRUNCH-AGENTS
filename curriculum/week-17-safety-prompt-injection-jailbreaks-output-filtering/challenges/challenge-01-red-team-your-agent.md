# Challenge 1 — Red-Team Your Agent

**Time estimate:** ~150 minutes.

## Problem statement

Two weeks ago you built an MCP-tool agent and wrote a security memo claiming it was safe. Now you find out. You're going to red-team your own **week-15 agent** — the LangGraph agent with the sandboxed filesystem server and the corpus-search server — with **25 adversarial prompts** spanning direct injection, indirect injection (planted in retrieved corpus content), and tool-argument abuse. You'll measure the **attack-success-rate** with no week-17 defenses, then add three defenses (input filtering, structured argument validation under fire, and an output classifier), re-measure after each, and write a one-page **threat model** that names the residual attacks that still land.

This is the syllabus deliverable in lab form, and it's a required piece of the Phase III milestone. The output is not "my agent is secure" — no agent is. The output is a table (ASR per defense layer), a preserved benign-pass-rate, and an honest accounting of what gets through and why.

## What you build

A red-team harness and a threat model:

1. **`attacks.jsonl`** — 25 adversarial prompts: ~8 direct, ~9 indirect (with planted corpus content + a benign user request), ~8 tool-argument-abuse. Each with a mechanically-checkable success criterion (canary leaked / file created / prompt revealed / sandbox escaped). Plant a `SECRET.txt` canary *outside* the sandbox.
2. **`asr.py`** — runs the suite against your week-15 agent with cumulative defense layers (none → input filter → + arg validation → + output classifier), computes ASR *and* benign-pass-rate at each layer, and prints the table.
3. **`defenses.py`** — the three layers: an input filter (regex + optionally a classifier), the argument validation (your week-15 `safe_path` + `clause_id` validator, now under fire), and an output filter/classifier (regex for the canary, plus optionally Llama Guard / an LLM-judge).
4. **`threat-model.md`** — the one-page threat model: assets, entry points, tools/blast-radius, attack→asset mapping, the ASR table, and the **named residual**.

## The harness approach

The whole red-team reduces to: run the suite against the agent at each defense level, score the checkable success criterion, and report ASR + benign-pass-rate per layer.

```python
def attack_success_rate(agent, suite, benign):
    succ = sum(check_success(agent.run(a), a["success_check"]) for a in suite)
    asr = succ / len(suite)
    bpr = sum(1 for b in benign if not blocked(agent.run(b))) / len(benign)
    return asr, bpr

for name, agent in [
    ("no defenses",      week15_agent()),
    ("+ input filter",   week15_agent(input_filter=injection_filter)),
    ("+ arg validation", week15_agent(input_filter=injection_filter, validate=True)),
    ("+ output filter",  week15_agent(input_filter=injection_filter, validate=True,
                                      output_filter=safety_classifier)),
]:
    asr, bpr = attack_success_rate(agent, suite, benign)
    print(f"{name:18} ASR={asr:.2f}  benign_pass={bpr:.2f}")
```

For the **indirect** attacks, the harness must plant the malicious content into a corpus clause the agent *retrieves* (not into the user's message), so the injection rides in through your real RAG pipeline. For **success judgment** where the criterion is fuzzy ("did it comply?"), use a calibrated LLM-as-judge (`claude-opus-4-8` or a local model), validated against a few hand-labels per week 12.

## Acceptance criteria

- [ ] A `challenge-01/` directory with `attacks.jsonl` (25 attacks), `asr.py`, `defenses.py`, and `threat-model.md`, all runnable against your week-15 agent.
- [ ] The 25 attacks span **direct, indirect, and tool-argument** families, each with a **mechanically-checkable** success criterion and a planted canary for exfil.
- [ ] The **indirect** attacks plant content into *retrieved corpus clauses*, not the user message — riding in through the RAG pipeline.
- [ ] `asr.py` reports **ASR and benign-pass-rate** at each of the four defense levels, in a table showing the per-layer delta.
- [ ] ASR **drops meaningfully** across the layers (a high bare-agent ASR down to a low hardened ASR), and **benign-pass-rate stays high** (defenses don't DoS legitimate users).
- [ ] The **argument-validation** layer demonstrably stops the tool-argument-abuse attacks even when the input filter is bypassed (it holds when the model is steered).
- [ ] A one-page `threat-model.md` with: assets, entry points (user + retrieval), tools and their blast radius, attack→asset mapping, the ASR table, and the **named residual** (which attacks still land and why you accept/mitigate them).
- [ ] At least one **promise-format result** showing a defense layer actually reducing attack success:
  `ASR 0.64 -> 0.08; residual: [obfuscated indirect X], [payload-split Y] — see threat-model.md ✓`

## The trap (read after a first attempt)

The trap is **declaring victory at ASR = 0 — or claiming your threat model proves the agent is "secure."** Two ways this goes wrong. First, if your ASR hits exactly zero, suspect your attacks are too weak before you celebrate: a *real* red-team of a *real* agent leaves a residual, because injection has no complete fix and obfuscated/indirect attacks slip past regex filters. An ASR of 0 on 25 attacks usually means your success criteria were too strict (the attacks "succeeded" too rarely to register) or your attacks too lazy — go write harder ones (obfuscation, multi-turn payload-splitting, indirect-via-retrieval) until a residual appears. Second, a threat model that says "the agent is secure against prompt injection" is *security theater* — no agent is, and a reviewer who reads that claim stops trusting the document. The honest threat model says "ASR is 0.08; these two attacks land; we accept/mitigate them because X." **Honesty about the residual is the difference between a security review and a press release.**

A second, subtler trap: **forgetting to measure benign-pass-rate.** It's tempting to optimize ASR alone — and the easiest way to drive ASR to zero is to block everything. A filter aggressive enough to catch every injection will also block legitimate users who happen to say "ignore the previous draft." If your defenses dropped ASR to 0.05 but also dropped benign-pass-rate to 0.70, you built a denial-of-service against your own users and called it security. Report both axes; tune to the knee.

## Stretch goals

- **Wire in a real classifier.** Replace (or stack behind) the regex input/output filter with Llama Guard (or a hosted moderation API). Measure where it beats the regex (the obfuscated attacks) and where it just adds latency for no gain. Report the precision/recall delta.
- **The indirect-injection chaos rehearsal.** Build the full indirect scenario: a malicious instruction hidden in a corpus clause that, when retrieved, tries to exfiltrate the canary. Demonstrate (a) it landing on the bare agent, then (b) the output classifier catching the exfil even though the injection reached the model. This is week 24's chaos-drill prompt-injection scenario, rehearsed.
- **Add a canary alert.** Beyond *blocking* exfil, *detect* it: fire an alert (log line / metric) whenever any tool output contains the canary's contents, so an attempt is observable, not just stopped. This bridges to week 18's observability and week 24's incident response.
- **Automated red-team.** Point `promptfoo`'s red-team mode (or `garak`) at your agent and compare its discovered attacks to your hand-written 25. Did the automated tool find a family you missed? Add it to your suite.

## Why this matters

The Phase III milestone (end of week 18) requires a **written threat model with live week-17 defenses.** This challenge *is* that threat model, produced by actually attacking the system rather than imagining attacks. And week 24's chaos drill runs the prompt-injection-on-a-tool scenario against your capstone — injecting a malicious instruction via a retrieved document and asking whether your defenses hold *in production, in a controlled window.* The defenses you measure here are the ones that get tested there. An engineer who red-teamed their own agent, measured the ASR drop, and named the residual walks into that drill knowing exactly what holds and what doesn't; one who shipped a happy-path agent with a "looks secure to me" memo watches their agent leak the canary the first time a document tells it to. If your agent has a tool, your agent has an attack surface — and now you can prove how big it is, and how much smaller you made it.
