# Week 17 — Safety: Prompt Injection, Jailbreaks, and Output Filtering

Welcome to the week you attack your own agent. Two weeks ago you built an MCP tool surface and wrote a security memo claiming it was safe. This week you find out if you were right — by writing 25 adversarial prompts, measuring how many succeed, adding defenses, and measuring again. Safety in an LLM-backed product is not a feeling or a checklist; it's an **attack-success-rate** you drive down with measured defenses, the same way you drove down retrieval error and judged a fine-tune. By Friday you will be able to look at any agent with a tool and state, with evidence, what its attack surface is, which attacks land, and what each defense bought you.

This is week 5 of **Phase III — Agents & Orchestration**, and it's the week the syllabus's recurring warning comes due: *if your agent has a tool, your agent has an attack surface.* You built that tool surface in week 15. You learned in week 15's security review that a tool is RCE and validated arguments accordingly. This week you stop trusting that the defenses work and *prove* it — red-teaming your own week-15 MCP-tool agent with direct injection, indirect injection (via a poisoned retrieved document), and tool-argument abuse, then hardening it with input filtering, structured argument validation, and an output classifier, and re-measuring the attack-success-rate.

The one sentence to internalize before you read another line:

> **If your agent has a tool, your agent has an attack surface. Threat-model it.**

Here's why that's the whole week. A frozen chatbot with no tools can, at worst, say something it shouldn't. An *agent* with tools can *do* something it shouldn't — read a file it wasn't supposed to, run a query it wasn't authorized for, exfiltrate data through an outbound call — because a prompt injection can steer the model into calling those tools with hostile arguments. The model is the soft underbelly: it follows instructions, and an attacker's whole game is getting *their* instructions into the model's context, whether by typing them (direct) or planting them in a document the model retrieves (indirect). The defenses are layered — no single one is sufficient — and the discipline is to *measure* each layer's contribution, not to bolt on a regex and declare victory.

There's a corollary worth taping to your monitor:

> **Prompt injection is the SQL injection of the LLM era — except there is no parameterized query that fully solves it.** You defend in depth, you measure, and you never assume one filter is enough.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** prompt injection as the dominant LLM security issue, and distinguish **direct injection** (the attacker controls the user input) from **indirect injection** (the attacker poisons a document/tool-result the model later ingests).
- **Recognize** the jailbreak surface — role-play attacks, instruction-override ("ignore previous instructions"), encoding/obfuscation, many-shot, prefix-injection — and why a model that *wants* to be helpful is hard to fully lock down.
- **Threat-model** an agentic tool surface using the **OWASP LLM Top 10** as the catalog: which entries apply (LLM01 prompt injection, insecure output handling, excessive agency, …) and what the blast radius of each is.
- **Build** layered defenses — input filtering (detect/strip injection attempts), structured tool-argument validation (the week-15 discipline, now under attack), and **output filtering** (regex, classifier-based, and LLM-judge moderation).
- **Survey** the moderation toolchain — **Llama Guard** (open classifier), **OpenAI Moderation**, **Perspective** — and place each on the input/output-filtering map.
- **Measure** safety as engineering: compute attack-success-rate over a fixed adversarial set, harden, re-measure, and report the delta — and measure **hallucination** as a related reliability signal.
- **Red-team** your own week-15 MCP-tool agent: 25 adversarial prompts (direct, indirect, tool-argument abuse), an ASR before/after hardening, and a written threat model.
- **Build** the `crunchguard` toolkit: an adversarial prompt suite, a layered-defense pipeline, an ASR harness, and a threat-model document.

## Prerequisites

This week assumes you have completed **C23 weeks 1–16**, or have equivalent fluency. Specifically:

- You finished **week 15** and have the `crunchmcp` toolkit (filesystem + corpus servers) wired into a LangGraph agent. **This week attacks that exact agent** — if it's broken, fix it first; the red-team lab has nothing to attack without it.
- You remember **week 15's security review** — a tool is RCE, validate every argument, resolve-then-contain for paths. This week you put those defenses *under fire* and measure whether they hold.
- You internalized the **measurement discipline** from weeks 8, 12, and 16 — a gold set, a metric, a before/after. This week the metric is attack-success-rate, and the discipline is identical: never claim a defense works without the number.
- Python 3.12 on Linux, macOS, or WSL2; a virtualenv you can `pip install` into; the LangGraph agent and MCP servers from week 15 importable.

You do **not** need a GPU for the core lab (the defenses are filters and validators; an open classifier like Llama Guard *prefers* a GPU but has a CPU/hosted path, and a CPU-only path using simpler classifiers is documented). One consumption path uses Claude (`claude-opus-4-8`) as both an attack-judge and an LLM-as-judge output filter; an open-only fallback (a local model from week 6) is documented for every lab, so the week is completable with no vendor credentials.

## Topics covered

- **Prompt injection, the dominant issue:** why the model can't reliably separate "instructions from the developer" from "instructions in the data," and why that makes injection the defining LLM vulnerability.
- **Direct vs indirect injection:** direct (attacker types the malicious instruction) vs indirect (attacker plants it in a webpage, document, email, or tool result the agent later reads) — and why indirect is scarier in a RAG/agent system.
- **The jailbreak surface:** role-play ("you are DAN"), instruction-override, obfuscation/encoding, many-shot, payload splitting, and prefix-injection — the taxonomy of how a refusal gets bypassed.
- **The OWASP LLM Top 10:** the canonical threat catalog; LLM01 (prompt injection), insecure output handling, excessive agency, and the tool/agent-relevant entries, with their blast radii.
- **Layered defense:** input filtering (detect/neutralize injection), structured tool-argument validation (the RCE discipline under attack), least-privilege capability scoping, and human-in-the-loop consent gates.
- **Output filtering:** regex (cheap, brittle), classifier-based (**Llama Guard**, **OpenAI Moderation**, **Perspective**), and LLM-as-judge moderation (flexible, slower) — and where each fits.
- **Hallucination measurement:** the related reliability signal — grounding/faithfulness as a defense against confidently-wrong output.
- **Red-teaming methodology:** building the adversarial set, the attack-success-rate metric, the harden-and-re-measure loop, and the written threat model.

## Weekly schedule

The schedule below adds up to approximately **36 hours**. Treat it as a target, not a contract.

| Day       | Focus                                                          | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|---------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Prompt injection; direct vs indirect; the jailbreak surface   |    2h    |    1.5h   |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     5.5h    |
| Tuesday   | OWASP LLM Top 10; threat modeling; the attack exercises       |    1h    |    2.5h   |     1h     |    0.5h   |   1h     |     0h       |    0h      |     6h      |
| Wednesday | Layered defenses; output filtering (regex/classifier/judge)   |    2h    |    1.5h   |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     6.5h    |
| Thursday  | Red-team methodology; ASR; building the guard pipeline        |    1h    |    1.5h   |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6.5h    |
| Friday    | The red-team run + threat model; defense clinic               |    0h    |    0h     |     1h     |    0.5h   |   1h     |     3h       |    0.5h    |     6h      |
| Saturday  | Mini-project deep work (attack + harden + re-measure)         |    0h    |    0h     |     0h     |    0h     |   0h     |     3h       |    0h      |     3h      |
| Sunday    | Quiz, review, threat-model polish                             |    0h    |    0h     |     0h     |    1h     |   0h     |     1h       |    0h      |     2h      |
| **Total** |                                                               | **6h**   | **7h**    | **4h**     | **3.5h**  | **5h**   | **12h**      | **2h**     | **36h**     |

## How to navigate this week

| File | What's inside |
|------|---------------|
| [README.md](./README.md) | This overview (you are here) |
| [resources.md](./resources.md) | The OWASP LLM Top 10, the injection/jailbreak references, the moderation-tool docs, and the red-teaming methodology reading |
| [lecture-notes/01-prompt-injection-and-the-jailbreak-surface.md](./lecture-notes/01-prompt-injection-and-the-jailbreak-surface.md) | Prompt injection (direct vs indirect), the jailbreak taxonomy, the OWASP LLM Top 10, and threat-modeling a tool surface |
| [lecture-notes/02-defenses-output-filtering-and-red-teaming.md](./lecture-notes/02-defenses-output-filtering-and-red-teaming.md) | Layered input defenses, output filtering (regex/classifier/judge), the moderation toolchain, hallucination, and the red-team measurement loop |
| [exercises/README.md](./exercises/README.md) | Index of the three exercises |
| [exercises/exercise-01-write-attacks.md](./exercises/exercise-01-write-attacks.md) | Write a taxonomy of 15 adversarial prompts (direct, indirect, tool-argument) and predict which land |
| [exercises/exercise-02-build-an-injection-filter.py](./exercises/exercise-02-build-an-injection-filter.py) | Build input + output filters, measure precision/recall on a labeled set, and confront the false-positive trade-off |
| [exercises/exercise-03-measure-attack-success-rate.py](./exercises/exercise-03-measure-attack-success-rate.py) | Run an adversarial suite against a toy agent, compute ASR, add defenses, re-measure the delta |
| [challenges/README.md](./challenges/README.md) | Index of the weekly challenge |
| [challenges/challenge-01-red-team-your-agent.md](./challenges/challenge-01-red-team-your-agent.md) | Red-team your week-15 MCP-tool agent: 25 attacks, ASR before/after three defenses, a written threat model |
| [quiz.md](./quiz.md) | 13 questions with a hidden answer key |
| [homework.md](./homework.md) | Six problems including the one-page agent threat model |
| [mini-project/README.md](./mini-project/README.md) | The `crunchguard` toolkit — adversarial suite, layered defenses, ASR harness, and threat model |

## The "attack landed — then it didn't" promise

C23 uses a recurring marker for every exercise that ends in a defense *measured to actually reduce attack success*:

```
$ python -m crunchguard asr --agent week15 --suite attacks.jsonl
                          attack_success_rate   benign_pass_rate
no defenses                      0.64                 1.00
+ input filter                   0.40                 0.98
+ arg validation                 0.16                 0.98
+ output classifier              0.08                 0.97
--------------------------------------------------------------
ASR 0.64 -> 0.08 across three defense layers. 2 attacks still land
(see threat-model.md). Benign traffic barely affected (1.00 -> 0.97).
```

If a "defense" drops ASR by zero, it's theater — strip it. If it drops ASR but also tanks benign traffic to 0.70, you've built a denial-of-service against your own users. The point of week 17 is to make safety a *measured* property — ASR down, benign-pass-rate up — with a number for every layer, and an honest list of the attacks that *still land*. A threat model that claims zero residual risk is a threat model that didn't measure.

## Stretch goals

If you finish the regular work early and want to push further:

- Read the **OWASP LLM Top 10** (2025) end to end until you can name the blast radius of each entry for *your* agent: <https://genai.owasp.org/llm-top-10/>. Then map each of your 25 attacks to the OWASP entry it exploits.
- Stand up **Llama Guard** (or a hosted equivalent) as a real input *and* output classifier, measure its precision/recall on your adversarial set, and chart where it beats your regex filter and where it adds latency for no gain.
- Build the **indirect-injection scenario fully**: plant a malicious instruction inside a corpus clause your agent retrieves, and demonstrate (a) the attack landing with no defense, then (b) your output classifier catching the exfiltration attempt even though the injection reached the model.
- Add a **canary** to your filesystem sandbox (a planted secret file) and an **alert** that fires if any tool output contains the canary's contents — so an exfiltration attempt is *detected*, not just blocked. This is the chaos-drill (week 24) prompt-injection scenario, rehearsed.

## Up next

Week 18 instruments the whole Phase III system with observability — OpenTelemetry Gen-AI conventions, Langfuse, Arize Phoenix, trace-driven debugging — and closes the phase. Your threat model from this week, your defenses, and your ASR number are part of the **Phase III milestone** (a multi-agent system with MCP tools, a written threat model, and full tracing). And in week 24's chaos drill, the prompt-injection-on-a-tool scenario injects a malicious instruction via a retrieved document and asks whether *this week's* defenses hold — so the harder you red-team now, the calmer that 4 AM drill is. Push your `crunchguard` threat model before you start week 18; the milestone wants it.

---

*If you find errors in this material, please open an issue or send a PR. Future learners will thank you.*
