# Week 17 — Resources

Every resource here is **free** or has a free tier. The OWASP LLM Top 10 is open. Llama Guard is an open model on Hugging Face; OpenAI Moderation and Perspective have free tiers. The injection/jailbreak literature is on arXiv. This week needs no paid GPU for the core lab (the defenses are filters and validators); an open classifier prefers a GPU but has hosted/CPU paths, all documented.

The attack landscape moves *fast* — new jailbreaks appear weekly. The *concepts* (direct vs indirect injection, the layered-defense posture, attack-success-rate as the metric, the OWASP catalog) are stable. When a specific attack or tool page 404s, the methodology — write attacks, measure ASR, harden, re-measure — carries over regardless of the day's specific exploit.

This week attacks your **week-15 MCP-tool agent**. The `crunchmcp` toolkit and the LangGraph consumer come from there; the resources below assume you have that agent to red-team.

## Required reading (work it into your week)

- **OWASP Top 10 for LLM Applications (2025)** — the canonical threat catalog. `LLM01: Prompt Injection`, insecure output handling, excessive agency, and the tool/agent-relevant entries are the spine of the week's threat model. Read LLM01 twice:
  <https://genai.owasp.org/llm-top-10/>
- **Anthropic — mitigating prompt injection / jailbreaks** — practical guidance on defending Claude-based agents: input/output handling, the operator-vs-user instruction boundary, and why layered defense beats a single filter:
  <https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-prompt-injection>
- **Simon Willison — prompt injection (the canonical write-ups)** — the clearest explanation of *why* prompt injection is hard (the model can't separate trusted instructions from untrusted data) and why "just tell it to ignore injections" doesn't work:
  <https://simonwillison.net/series/prompt-injection/>
- **Llama Guard model card** — the open input/output safety classifier you'll wire in as a defense layer; read the taxonomy of categories it classifies:
  <https://huggingface.co/meta-llama/Llama-Guard-3-8B>

## The injection & jailbreak references

- **Indirect prompt injection (Greshake et al., 2023)** — *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* The paper that named the indirect threat: a malicious instruction planted in retrieved content. The scariest vector in a RAG/agent system:
  <https://arxiv.org/abs/2302.12173>
- **Universal and Transferable Adversarial Attacks (Zou et al., 2023)** — the GCG jailbreak suffix paper; why some jailbreaks transfer across models and why robustness is hard:
  <https://arxiv.org/abs/2307.15043>
- **Many-shot jailbreaking (Anthropic, 2024)** — flooding the context with fake dialogue to erode a refusal; a long-context-era jailbreak:
  <https://www.anthropic.com/research/many-shot-jailbreaking>
- **The jailbreak taxonomy (a survey)** — role-play, obfuscation, payload-splitting, prefix-injection, and the rest of the bypass families, in one place:
  <https://genai.owasp.org/llmrisk/llm01-prompt-injection/>

## Defenses & output filtering (have these open on Wednesday)

- **Llama Guard 3** — `meta-llama/Llama-Guard-3-8B`. The open classifier for input *and* output moderation; GPU-preferred, hosted/CPU paths exist. The classifier leg of your defense pipeline:
  <https://huggingface.co/meta-llama/Llama-Guard-3-8B>
- **OpenAI Moderation API** — a free hosted content classifier; a second classifier to compare against Llama Guard on your set:
  <https://platform.openai.com/docs/guides/moderation>
- **Perspective API** — Jigsaw's toxicity/threat classifier; the third classifier option, strong on the toxicity axis:
  <https://perspectiveapi.com/>
- **NeMo Guardrails** — NVIDIA's policy framework for input/output rails and dialog flow constraints; the "policy-as-config" approach you'll meet again in week 20:
  <https://github.com/NVIDIA/NeMo-Guardrails>
- **Rebuff / LLM-injection-filter patterns** — the input-filtering pattern (detect-and-neutralize injection attempts before they reach the model); a reference for Exercise 2's filter:
  <https://github.com/protectai/rebuff>

## Hallucination measurement (the related reliability signal)

- **Ragas — faithfulness / groundedness** — the metric that catches confidently-wrong output by checking whether claims are supported by the retrieved context; the same metric from week 8/12, now framed as a safety signal:
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/>
- **TruthfulQA / hallucination benchmarks** — for literacy on how hallucination is measured at the benchmark level (you won't run these in full, but know the shape):
  <https://github.com/sylinrl/TruthfulQA>

## Red-teaming methodology

- **Anthropic — red-teaming language models** — the methodology: build an adversarial set, measure attack success, iterate. The discipline this week applies to *your* agent:
  <https://www.anthropic.com/research/red-teaming-language-models-to-reduce-harms-methods-scaling-behaviors-and-lessons-learned>
- **`garak` (LLM vulnerability scanner)** — an open automated red-teaming tool; a stretch-goal way to fuzz your agent with a battery of known attacks:
  <https://github.com/NVIDIA/garak>
- **`promptfoo` red-team mode** — your week-3 prompt-testing tool also runs adversarial red-team suites; the harness you already know, pointed at attacks:
  <https://www.promptfoo.dev/docs/red-team/>

## Model used in the judge/filter path

- **`claude-opus-4-8`** — used two ways this week: as an **attack-judge** (did the attack succeed? a calibrated LLM-as-judge, week-12 discipline) and as an **LLM-as-judge output filter** (is this output safe to return?). An open-only fallback (a local model from week 6 for the judge, plus a simpler classifier for filtering) is documented for every lab — the week is completable with no Anthropic key.

## Tools you'll use this week

- **Your week-15 `crunchmcp` agent** — the target. The LangGraph agent + filesystem/corpus MCP servers you red-team.
- **`transformers`** — to run Llama Guard locally (or a hosted endpoint if no GPU).
- **A regex/keyword filter** — the cheap, brittle first layer you build in Exercise 2.
- **An LLM judge** (`claude-opus-4-8` or a local model) — for the calibrated attack-success judgment and the LLM-as-judge output filter.
- **`promptfoo` / `garak` (optional)** — automated red-team tooling for the stretch goals.

## A note on the corpus and the attack target

The red-team target is the **week-15 agent over the legal corpus + filesystem sandbox**. That keeps the attacks *checkable*: an exfiltration attack either gets the planted out-of-sandbox secret (`SECRET.txt`) into the model's output or it doesn't; a tool-argument-abuse attack either escapes the sandbox or it doesn't. The indirect-injection scenario plants a malicious instruction *inside a corpus clause* the agent retrieves — so the attack rides in through your own RAG pipeline, which is exactly the real-world indirect vector. Concrete, planted, and measurable: that's how you make "did the defense work?" a number rather than a vibe.

## A note on ethics and scope

This week teaches you to attack *your own* systems to defend them — red-teaming is a defensive discipline. Every attack you write targets your own agent and your own planted canaries. Do not point these techniques at systems you don't own or have explicit authorization to test. The goal is a harder-to-break agent, not a how-to for breaking other people's; the OWASP framing throughout is *defensive threat modeling*, which is exactly the framing a security review wants.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Prompt injection** | Getting your instructions into the model's context so it follows you, not the developer. |
| **Direct injection** | The attacker controls the user input and types the malicious instruction. |
| **Indirect injection** | The attacker plants the instruction in a document/tool-result the model later reads. |
| **Jailbreak** | Bypassing a model's safety refusal (role-play, override, obfuscation, many-shot, …). |
| **Attack-success-rate (ASR)** | Fraction of an adversarial set that succeeds; the safety metric you drive down. |
| **Benign-pass-rate** | Fraction of *legitimate* traffic that still gets through after defenses — don't tank it. |
| **Input filtering** | Detect/neutralize injection attempts before they reach the model. |
| **Output filtering** | Check the model's output before returning it (regex / classifier / LLM-judge). |
| **Llama Guard** | An open input/output safety classifier (a moderation model). |
| **OWASP LLM Top 10** | The canonical catalog of LLM application security risks; LLM01 is prompt injection. |
| **Excessive agency** | An agent with more tool power than it needs — a larger blast radius when attacked. |
| **Layered defense** | Multiple independent defenses; no single one is sufficient (defense in depth). |
| **Canary** | A planted secret whose appearance in output proves an exfiltration attack landed. |
| **Threat model** | A written account of who attacks, how, what's at risk, and which defenses hold. |

---

*If a link 404s, please open an issue so we can replace it.*
