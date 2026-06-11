# Week 3 — Resources

Every resource here is **free** to read. The tool docs (promptfoo, Langfuse, Claude Code, Cursor) are published openly; the papers are on arXiv; the prompt-injection references are open posts and specs. No paywalled books are linked. Where a link is to a vendor's docs that move over time, the title is given so you can re-find it if the URL drifts.

Pin yourself to **2026-current** model IDs when you write code: the hosted frontier model in this course is Anthropic Claude — `claude-opus-4-8` (most capable), `claude-sonnet-4-6` (balanced), `claude-haiku-4-5` (fast/cheap) — and the local model is whatever Ollama tag you pulled (`qwen2.5:7b`, `llama3.2:3b`). On these 2026 models there is **no assistant-prefill** and **no `budget_tokens`**: you steer with the `system` kwarg and, for reasoning, with `thinking={"type":"adaptive"}` plus `output_config={"effort": "..."}`. Do not copy older prompt-API patterns from blog posts; they may 400.

## Required reading (work it into your week)

- **promptfoo — Getting Started + Configuration reference.** The harness you build the lab on. Read the `promptfooconfig.yaml` schema (providers, prompts, tests, assertions) Monday and keep it open all week:
  <https://www.promptfoo.dev/docs/getting-started/> · config: <https://www.promptfoo.dev/docs/configuration/guide/>
- **promptfoo — Assertions & metrics.** How a test "passes": `contains`, `equals`, `llm-rubric`, `javascript`/`python` asserts, and named metrics. This is where your pass rate comes from:
  <https://www.promptfoo.dev/docs/configuration/expected-outputs/>
- **Langfuse — Prompt Management.** Labelled prompt versions, `production`/`latest` labels, pulling a prompt by name+label at runtime, and rolling back a bad version without a deploy:
  <https://langfuse.com/docs/prompts/get-started>
- **Anthropic — Prompt engineering overview.** The vendor's own guidance on system prompts, examples, and (current) chain-of-thought — read it against the lecture's honesty caveat, not as gospel:
  <https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview>

## The papers (read for the finding, not for the math)

You will not implement these. You read them so the claims behind "use CoT" and "use self-consistency" are yours — including the parts that complicate the advice.

- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (Wei et al., 2022). The paper that started it. Read §1–3 — what CoT is, where it helped, and the crucial caveat that the lift concentrated on *large* models and *multi-step* tasks:
  <https://arxiv.org/abs/2201.11903>
- **Self-Consistency Improves Chain of Thought Reasoning in Language Models** (Wang et al., 2022). Sample N reasoning paths, take the majority answer. Read the method and the cost discussion — the lift is real and it is N× the tokens:
  <https://arxiv.org/abs/2203.11171>
- **Large Language Models Can Be Easily Distracted by Irrelevant Context** (Shi et al., 2023). The empirical counterweight: more "reasoning" context is not always better, and CoT does not rescue a model from distraction:
  <https://arxiv.org/abs/2302.00093>
- **Language Models Don't Always Say What They Think** (Turpin et al., 2023). The honesty caveat in one paper: the visible chain-of-thought is **not** always a faithful account of how the model reached its answer. Required reading for the lecture's "CoT told honestly" section:
  <https://arxiv.org/abs/2305.04388>
- **The Unreasonable Effectiveness of Few-Shot Examples** — read Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020), §3 only, for the original few-shot result that the pattern is built on:
  <https://arxiv.org/abs/2005.14165>

## The jailbreak / prompt-injection surface (read as an attacker would)

This is previewed this week and threat-modeled in depth in Week 17. Read these now so the vocabulary is yours.

- **OWASP Top 10 for LLM Applications — LLM01: Prompt Injection.** The canonical framing of the dominant LLM security issue; direct vs indirect injection:
  <https://genai.owasp.org/llmrisk/llm01-prompt-injection/>
- **Simon Willison — "Prompt injection" series.** The clearest plain-English explanation of why the flat token stream makes this hard to fix, and why "just tell the model to ignore injections" doesn't work:
  <https://simonwillison.net/series/prompt-injection/>
- **Greshake et al. — "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection"** (2023). The indirect-injection paper; read the threat model, skip the experiments for now:
  <https://arxiv.org/abs/2302.12173>

## Agentic dev tools (the spec-then-implement loop)

- **Claude Code — documentation.** The agentic CLI you run the spec-then-implement loop in; read the "common workflows" and the `CLAUDE.md` / settings sections:
  <https://docs.claude.com/en/docs/claude-code/overview>
- **Cursor — documentation.** The editor counterpart; read the Rules and the Agent sections so you can compare the two loops honestly:
  <https://docs.cursor.com/>
- **Anthropic — "Claude Code best practices."** How to run a disciplined spec-then-implement loop (write the spec, let the agent implement, review the diff) rather than vibe-coding:
  <https://www.anthropic.com/engineering/claude-code-best-practices>

## API & SDK references (open all week)

- **Anthropic Python SDK** — `client.messages.create`, the `system` kwarg, `usage` accounting, and the `thinking` / `output_config` reasoning controls (no `budget_tokens`, no assistant-prefill on 2026 models):
  <https://github.com/anthropics/anthropic-sdk-python>
- **Anthropic — Messages API reference** — the exact shape of `messages`, `system`, `output_config`, and the content blocks you read with `next(b.text for b in resp.content if b.type=="text")`:
  <https://docs.claude.com/en/api/messages>
- **Ollama API reference** — the local fallback path for every lab; `/api/chat` with a `system` message and the `options` knobs you'll sample over for self-consistency:
  <https://github.com/ollama/ollama/blob/main/docs/api.md>

## Talks & posts worth your time (free, no signup)

- **"Prompt engineering is dead, long live prompt engineering"** — the recurring industry argument that prompting matters *more*, not less, once you treat it as engineering. Read any well-argued version of this take to inoculate yourself against the "just use a bigger model" reflex.
- **promptfoo — "Why we built a prompt testing tool."** The motivation post; read it for the framing of prompts-as-code that this whole week shares:
  <https://www.promptfoo.dev/docs/intro/>
- **Langfuse — "Prompt management in production."** Why a runtime prompt registry with labelled versions beats redeploying for every wording change:
  <https://langfuse.com/docs/prompts/get-started>

## Tools you'll use this week

- **`promptfoo`** — `npx promptfoo eval -c promptfooconfig.yaml`, `npx promptfoo view`. The Node CLI regression harness. You write YAML + Python providers, not JavaScript.
- **`anthropic` (Python SDK)** — `pip install anthropic`. The hosted path for the harness and the CoT exercise.
- **`langfuse` (Python SDK)** — `pip install langfuse`. The runtime prompt registry; optional this week, used in the homework's versioning problem.
- **`git`** — your prompt version control. This week the commit log *is* the deliverable: one commit per prompt version, the pass rate in the message.
- **Claude Code / Cursor** — the agentic dev loop for `exercise-01`. Either works; the exercise asks you to run a spec-then-implement loop and diff the iterations.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Prompt-as-code** | Treating the prompt as a versioned, diffable, testable file — not a string literal you tweak in prod. |
| **Golden example** | A fixed (input, expected-property) pair in your test set. The pass/fail unit of a prompt regression suite. |
| **Pass rate** | Fraction of golden examples a prompt version satisfies. The number that makes "better" a claim, not a vibe. |
| **Regression** | A prompt change that breaks a case that previously passed. The thing your suite exists to catch. |
| **System prompt** | The `system` kwarg: durable role/policy/format instructions, separate from the user's turn. |
| **Few-shot** | Including input→output exemplars in the prompt to steer behavior by demonstration, not description. |
| **Chain-of-thought (CoT)** | Prompting the model to produce intermediate reasoning before the answer. Helps on some tasks, not all. |
| **Self-consistency** | Sample N reasoning paths at temperature > 0, take the majority answer. Trades N× cost for accuracy. |
| **Role prompting** | "You are a world-class expert..." — mostly theater; test it, don't assume it. |
| **Prompt injection** | Untrusted input that hijacks the instruction stream because the model sees one flat token stream. |
| **Direct injection** | The user's own input carries the attack ("ignore previous instructions..."). |
| **Indirect injection** | The attack rides in on retrieved/tool content the user didn't write. The dangerous one. |
| **promptfoo** | Node CLI prompt regression harness; `promptfooconfig.yaml` declares providers, prompts, tests, asserts. |
| **Langfuse prompt mgmt** | A runtime prompt registry with labelled versions (`production`/`latest`) and rollback without a deploy. |
| **Spec-then-implement** | Write the prompt spec first, implement against it, diff iterations, review against a checklist. |

---

*If a link 404s, please open an issue so we can replace it.*
