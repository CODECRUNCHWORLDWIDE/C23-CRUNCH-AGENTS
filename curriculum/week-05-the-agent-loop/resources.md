# Week 5 — Resources

Every resource here is **free**. The ReAct paper is on arXiv. The Anthropic agent docs and the `claude-agent-sdk` are open. The OpenAI Agents SDK, AWS Strands, and Google ADK are open-source. Ollama is open-source. No paywalled books are linked.

Model IDs and SDK versions move; the loop doesn't. The IDs current in 2026 are `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5` for Anthropic and `qwen2.5:7b-instruct` for the local path. Swap SKUs in a later cohort; the reason→act→observe cycle is stable.

## Required reading (work it into your week)

- **Building effective agents** — Anthropic's foundational post on agent patterns (workflows vs agents, the building blocks). Read it Monday and again Thursday:
  <https://www.anthropic.com/research/building-effective-agents>
- **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022) — the paper that named the pattern you'll hand-roll:
  <https://arxiv.org/abs/2210.03629>
- **Tool use — implement the agentic loop** — Anthropic docs on the manual loop, `stop_reason` handling, and the tool runner:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use>
- **Anthropic Agent SDK / `claude-agent-sdk`** — the Claude-native agent loop you compare against your hand-rolled one:
  <https://platform.claude.com/docs/en/agents-and-tools>
- **Effort & adaptive thinking** — how thinking depth interacts with agentic loops and cost (the budgets story has a model-parameter side):
  <https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking>

## The patterns (read the lineage)

- **Plan-and-Solve / plan-and-execute** — the "plan first, then act" pattern lineage:
  <https://arxiv.org/abs/2305.04091>
- **Reflexion** — language agents that self-critique and retry; the reflection pattern's research root:
  <https://arxiv.org/abs/2303.11366>
- **Self-Consistency** — sampling multiple reasoning paths and voting; relevant background for why a critique pass can help:
  <https://arxiv.org/abs/2203.11171>
- **Toolformer** — the tool-use lineage that underpins why models can call tools at all:
  <https://arxiv.org/abs/2302.04761>

## The 2026 agent SDK landscape (survey level — skim the quickstarts)

- **Anthropic `claude-agent-sdk` (Python)** — the Claude-native managed loop; sessions, tools, the agent runtime:
  <https://github.com/anthropics/claude-agent-sdk-python>
- **Anthropic Python SDK tool runner** — `client.beta.messages.tool_runner` with `@beta_tool`; the in-SDK loop:
  <https://github.com/anthropics/anthropic-sdk-python>
- **OpenAI Agents SDK** — the OpenAI-native agent loop and handoffs:
  <https://openai.github.io/openai-agents-python/>
- **AWS Strands Agents** — AWS-native agent surface:
  <https://github.com/strands-agents/sdk-python>
- **Google ADK (Agent Development Kit)** — Google-native agent surface:
  <https://google.github.io/adk-docs/>

> Read these to *place* them, not to master them. The week's thesis is that the loop is the same underneath; the SDKs differ in ergonomics, observability, and how much they hide. You'll have an informed opinion by Thursday.

## API references (the ones you'll have open all week)

- **Anthropic Python SDK** — `messages.create`, streaming, usage accounting (`response.usage` for the token budget):
  <https://github.com/anthropics/anthropic-sdk-python>
- **Anthropic — token counting** — `messages.count_tokens` for pre-flight token budgeting (never `tiktoken` for Claude):
  <https://platform.claude.com/docs/en/build-with-claude/token-counting>
- **Anthropic — pricing** — the per-token rates you multiply by for the cost budget:
  <https://platform.claude.com/docs/en/about-claude/pricing>
- **Ollama — API** — the local `/api/chat` (and OpenAI-compatible `/v1`) endpoints the local agent calls:
  <https://github.com/ollama/ollama/blob/main/docs/api.md>

## Talks & posts worth your time (free, no signup)

- **Anthropic — "Building effective agents" (the post, again)** — re-read the "when to use an agent" section once you've built one:
  <https://www.anthropic.com/research/building-effective-agents>
- **Anthropic engineering blog** — posts on agent harness design, context management, and the loop:
  <https://www.anthropic.com/engineering>
- **LangChain — "Plan-and-execute agents"** — a concrete write-up of the pattern (read critically; you're not using the framework yet):
  <https://blog.langchain.dev/planning-agents/>

## Tools you'll use this week

- **`anthropic`** — `pip install anthropic`. The frontier loop and `response.usage` for budgets.
- **`claude-agent-sdk`** — `pip install claude-agent-sdk`. The managed loop you compare against.
- **`ollama`** — the local loop; `ollama pull qwen2.5:7b-instruct`.
- **`crunch_tools`** (your Week 4 mini-project) — the tool registry the agent loops over.
- **`tiktoken`** — *not* for Claude token counts (use `messages.count_tokens`); fine for a rough local estimate.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Agent loop** | model call → if `tool_use`, run tools → feed results → repeat → until `end_turn`. |
| **ReAct** | Reason + Act + observe: the model interleaves thinking, tool calls, and observations. |
| **Plan-and-execute** | Make a numbered plan first, then execute the steps. |
| **Reflection / self-critique** | The agent critiques its own draft and optionally revises. |
| **Step budget** | Max number of model turns before the loop force-terminates. |
| **Token budget** | Cumulative token ceiling across the run. |
| **Time budget** | Wall-clock ceiling for the whole run. |
| **Cost budget** | Dollar ceiling, computed from token usage × price. |
| **Infinite tool-call loop** | The canonical failure: the agent calls tools forever without converging. |
| **Trace** | The ordered record of every thought, action, observation, and budget tick. |
| **`stop_reason=tool_use`** | The model wants a tool run; the loop continues. |
| **`stop_reason=end_turn`** | The model is done; the loop terminates with an answer. |
| **Agent SDK** | A library (e.g. `claude-agent-sdk`) that runs the loop for you. |
| **Pass rate** | Fraction of benchmark tasks the agent solved correctly. |

---

*If a link 404s, please open an issue so we can replace it.*
