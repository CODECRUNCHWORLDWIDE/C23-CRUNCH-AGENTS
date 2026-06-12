# C23 · Crunch Agents — AI Agent Systems Engineering

> Code Crunch Club · Crunch Labs tier · sub-brand **Agents** (`#A855F7`)
> 24 weeks · ~864 hours · GPL-3.0
> Track home: `C23-CRUNCH-AGENTS/`

Twenty-four weeks on the modern AI engineering stack: large language models served locally and at scale, retrieval-augmented generation over private corpora, tool-using agents with safety rails, multi-agent orchestration graphs, evaluation pipelines that catch regressions before they ship, and the production runbook for an LLM-backed product on a Friday night. By the end you will have shipped a multi-agent research assistant served from your own vLLM cluster, instrumented with OpenTelemetry Gen-AI conventions, evaluated with Ragas and a calibrated LLM-as-judge, and survived a chaos drill in which a GPU node dies, a tool is prompt-injected, and a vector index corrupts — all in the same week.

This track is deliberately distinct from **C5 (Crunch AI · Data Science)**. C5 owns classical machine learning through deep learning: feature engineering, gradient boosting, CNNs, transformers as architectures, PyTorch fundamentals. C23 picks up where C5 leaves the room — LLMs as components in a system, not as research artifacts. The discipline here is *systems engineering on top of models you mostly do not train*. If C5 makes you a model builder, C23 makes you the engineer who keeps an agentic product alive in production.

---

## Who this is for

Four personas, all welcome, all stretched:

1. **The Python developer who wants to ship LLM products.** You have shipped Flask, FastAPI, or Django apps. You have called the OpenAI API and felt the shape of a real product underneath, but the gap between *demo* and *production* is fog. You want the durable patterns — retrieval as engineering, evaluation as engineering, the agent loop — and you want them taught with the same rigor as backend or systems work.
2. **The ML engineer who knows classical but not modern agentic.** You came up on scikit-learn, XGBoost, PyTorch, maybe Triton. You can train a model, calibrate a probability, fight a leakage bug. What you do not have, yet, is the muscle memory for RAG, MCP, vLLM, agent graphs, tool-use safety, and the production operating model of an LLM-backed product. C23 hands you that muscle memory.
3. **The SRE bridging into AI infrastructure.** You run platforms — K8s, Prometheus, on-call, autoscaling, blue/green. The new pager is GPU pools and token budgets. You want to learn vLLM cluster topology, NeMo Inference, continuous batching, KV-cache economics, eval-in-prod, and what a runbook for an agentic system actually says.
4. **The researcher transitioning to applied engineering.** You have a Ph.D. or near-Ph.D. on a model topic, and you can read a paper a week. The applied side — testable prompts, observability, retrieval pipelines, cost engineering, the messy reality of shipping — is the gap. C23 closes it without insulting your background.

If you can read and write Python comfortably, run Docker without flinching, and survive a Linux command line, you are ready. If you cannot, take **C1** (Convos) and **C14** (Linux) first.

---

## What you will be able to do at the end

Concrete capabilities on day 168:

1. Reason about LLMs as systems components — tokens, context windows, KV cache, attention compute, sampling parameters — and pick the right knob for the right product symptom.
2. Treat prompts as code: version them, diff them, write tests for them, gate releases on regression, and resist prompt injection with layered defenses.
3. Stand up local inference on commodity hardware with **Ollama** and **llama.cpp**, and scale to a multi-GPU **vLLM** cluster with continuous batching, paged attention, and speculative decoding.
4. Operate **NVIDIA NeMo Inference** for production serving and **NeMo Framework** for fine-tuning at scale; choose between **vLLM**, **SGLang**, **TGI**, and **TensorRT-LLM** with reasons, not vibes.
5. Build a retrieval-augmented system end to end: chunking strategies (token-window, semantic-paragraph, recursive, late chunking), open embeddings (**BGE**, **GTE**, **jina**), rerankers (**bge-reranker**, **Cohere**, **ColBERT**), hybrid search, GraphRAG, agentic RAG.
6. Choose, deploy, and operate a vector store — **pgvector**, **Qdrant**, **Weaviate**, **Milvus**, **Chroma** — and defend the choice in an architecture review.
7. Process real documents into retrievable knowledge: PDFs through **Unstructured** and **MinerU**, OCR with **Tesseract** and **Surya**, table extraction, multimodal pages, HTML at scale.
8. Design and ship tool-using agents with **function calling**, **JSON-mode**, and **grammar-constrained decoding**; expose a tool surface over the **Model Context Protocol (MCP)**; sandbox code execution safely.
9. Orchestrate multi-agent systems with **LangGraph**, **Mastra**, and **Inngest**: supervisor, swarm, and hierarchical graph patterns; understand the failure modes of each and when **AutoGen** or **CrewAI** are the wrong choice.
10. Build memory systems with three tiers — episodic, semantic, procedural — and budget context windows like cache: summarization, eviction, hybrid vector + knowledge-graph stores.
11. Fine-tune open-weights models with **LoRA / QLoRA** on **Axolotl** or **Unsloth**, run **DPO** for preference alignment, and decide when fine-tuning is the wrong answer (usually — prompt and retrieve first).
12. Instrument an agentic system with **OpenTelemetry Gen-AI semantic conventions**, **Langfuse** (open), **LangSmith**, and **Arize Phoenix**; trace every agent run, account for every token, hold every step to a latency budget.
13. Evaluate like an engineer: golden sets, **Ragas** for retrieval, **DeepEval** and **promptfoo** for prompt regression, calibrated LLM-as-judge, offline plus online eval, eval-in-prod with shadow traffic.
14. Run an agentic product on-call: model routing for cost, semantic cache, prompt compression, batching, blue/green model deploys, canary by user cohort, and a real postmortem of a real incident.

---

## Prerequisites

| Required | Helpful | Not required |
| --- | --- | --- |
| **C1 — Code Crunch Convos** (Python fluency) | **C5 — Crunch AI · Data Science** (classical ML + deep learning) | A four-year CS degree |
| Comfortable on a Linux shell | **C17 — Crunch Pro Python** (async, perf, packaging) | A Ph.D. in ML |
| Docker and `docker compose` basics | Some prior LLM API experience (OpenAI, Anthropic, Gemini) | Production ML experience |
| Git + GitHub workflow | Comfort reading a research paper without panicking | A GPU on day one |

To be honest about the C5 question: **C5 helps but is not required.** What C5 buys you is intuition for tokenization, attention, training loops, loss curves, and PyTorch idioms — useful when we get to fine-tuning in weeks 16–17. What you actually need on day one is Python fluency, Linux comfort, and the ability to run Docker. We re-derive the LLM internals from a systems perspective in weeks 1–3 so you do not need C5 to follow.

C17 (Pro Python) helps because real agent code is async, contended, and performance-sensitive. If you have not taken it, the async chapters in weeks 4–6 will feel harder but remain doable.

Hardware: a recent laptop (16 GB RAM minimum, 32 GB recommended) covers the first half of the course. Weeks 7+ benefit from a **single 24 GB GPU** (RTX 3090 / 4090 used or new, or a rented A10 / L4 cloud GPU at ~$0.50–$1.00 per hour). H100/H200 access is **not assumed** but is covered conceptually and exercised on rented spot instances for two specific labs (vLLM scale and NeMo Inference). CPU-only paths are documented for every lab; you will be slower but unblocked.

---

## Program at a glance — four phases

| Phase | Weeks | Title | Focus | Capstone milestone |
| --- | --- | --- | --- | --- |
| I | 1–6 | Foundations | LLM internals, prompt engineering, first agent, local inference bring-up | Working ReAct agent on a local 7B model |
| II | 7–12 | RAG & Memory Systems | Chunking, embeddings, rerankers, vector stores, memory tiers, multimodal, evaluation | Hybrid-retrieval system with Ragas eval on a private corpus |
| III | 13–18 | Agents & Orchestration | LangGraph, Mastra, MCP, multi-agent graphs, safety, observability | Supervisor-led multi-agent system with MCP tools and full tracing |
| IV | 19–24 | Production AI & Capstone | vLLM cluster, NeMo Inference, autoscaling, cost, eval-in-prod, capstone, on-call drill | Production agentic research assistant + chaos drill postmortem |

Detailed week-by-week breakdown lives in [`SYLLABUS.md`](./SYLLABUS.md). Design rationale (why this ordering, why open-source-first, what we will and will not teach) lives in [`CHARTER.md`](./CHARTER.md).

---

## Weekly cadence

| Day | Block | Typical content |
| --- | --- | --- |
| Mon | Lecture (2h) | Concept intro, paper-of-the-week excerpt, reference architecture walk |
| Mon | Lab (3h) | Guided exercise with a runnable starter repo |
| Wed | Lecture (2h) | Deeper dive, code review of Monday's lab, failure-mode tour |
| Wed | Lab (3h) | Open-ended mini-project sprint |
| Fri | Studio (4h) | Office hours, debugging clinic, eval-run reviews, trace reading |
| Sun | Quiz (~30m) + reading | Auto-graded; covers the week's lectures and one short paper |

Self-paced cohorts compress to ~12h/week; full-time cohorts run ~36h/week. Each week ships one mini-project, one quiz, and one logged evaluation run.

A weekly **paper club** — 30 minutes, voluntary, instructor-led — picks one paper a week from a curated list (RAG, agent, eval, serving). The course does not assume you read papers fluently; by week 24, you will.

---

## Hardware expectations

| Phase | Minimum | Recommended | Stretch |
| --- | --- | --- | --- |
| I (weeks 1–6) | Laptop, 16 GB RAM | Laptop, 32 GB RAM + Ollama on CPU/Metal | 24 GB GPU (3090/4090) |
| II (weeks 7–12) | Laptop + free Ollama models | 24 GB GPU for local embeddings + reranker | Cloud A10 / L4 for one weekend |
| III (weeks 13–18) | Laptop + a small open model via Ollama | 24 GB GPU for tool-using 7B/13B | Cloud H100 spot for the multi-agent capstone slice |
| IV (weeks 19–24) | Rented A10 (~$0.50/h) | Rented H100 for ~12 hours during the cluster lab | Multi-GPU node for NeMo Inference |

Every lab has a **CPU-only fallback** and a **rented-GPU recipe** with cost ceilings (~$30 covers the full course on rented compute). Local installs target Linux, macOS (Apple Silicon via Metal), and WSL2; ROCm paths for AMD GPUs are documented but secondary.

---

## Recommended pre/post tracks

```text
C1 (Code Crunch Convos · Python)
        |
        v
C5 (Crunch AI · Data Science)              <-- helpful, not required
        |
        v
C17 (Crunch Pro Python · async / perf)     <-- helpful, not required
        |
        v
*** C23 (Crunch Agents — AI Agent Systems Engineering) ***
        |
        +--> C18 / C19 (Crunch GCP / AWS)
        |       to scale your agentic system on a real cloud
        |
        +--> C22 (Crunch Mesh — distributed systems)
        |       to harden the surrounding service architecture
        |
        +--> C15 (Crunch DevOps)
                if you are the one on-call for the GPU pool
```

If you are skipping classical ML entirely, the workable path is **C1 → C17 → C23**. You will lose the intuition-from-the-bottom that C5 buys, but you will not be blocked. If you are coming from research or industry ML, you can enter C23 directly with C1-equivalent Python.

After C23, the most common landing roles are **applied AI engineer**, **agentic systems engineer**, **AI platform engineer**, **LLM SRE**, and **founding engineer at an AI-first company**.

---

## What this track is not

Three things C23 deliberately is not, to set expectations:

- **It is not a deep-learning course.** We re-derive the LLM parts that matter for systems engineering (tokenization, attention, KV cache, sampling). We do not derive backprop, train a transformer from scratch, or teach optimizer theory. That is C5's job.
- **It is not a framework tour.** We teach **LangGraph** and **Mastra** in depth because they earn their keep at the multi-agent stage; we name and critique the others (CrewAI, AutoGen, classic LangChain `LLMChain`) honestly. The unit of value is the underlying pattern, not the import statement.
- **It is not a vendor course.** Every lab has an open-weights, self-hosted path. Vendor APIs (OpenAI, Anthropic, Gemini, Bedrock) are taught as the production scale path and the frontier-capability path, never as the only path.

---

## A word on the field

The agentic / LLM-product field moves on weekly news cycles and quarterly architecture shifts. C23 is built to **survive that motion**. The spine — prompt-as-code, the agent loop, retrieval as engineering, memory as a budget, evaluation as engineering, observability as engineering, MCP as the open protocol, vLLM as the open serving primitive, cost engineering at token granularity — is the part we will still teach in 2028 with the same emphasis. The perimeter — specific frameworks, model names, leaderboard scores, vendor SKUs — rotates every cohort. We update the roster, not the spine.

If you take this course and a year later your favorite framework is deprecated, you should be able to swap it out in a weekend. That is the contract. The course is the engineering, not the import.

---

## Frequently asked, briefly

**"I have shipped a LangChain demo. Is this for me?"** Yes, especially. The course will refit the way you think about the loop and the eval.

**"I have a Ph.D. on diffusion models. Is this for me?"** Yes. The applied-engineering side — production serving, eval, observability, on-call — is what the academic track usually skips.

**"I cannot afford a GPU."** Roughly half the labs run on CPU or Apple Silicon. The other half have a rented-cloud recipe (~$30 of compute covers the course). No lab is gated on personal hardware.

**"How much TypeScript do I need?"** Enough to read week 14's Mastra lab. We do not assume fluency; Python is the primary teaching language.

**"Is the capstone open-source?"** Yes — it lives under GPL-3.0 like the rest of the track. You keep authorship and may relicense your fork; the academy reference implementation stays GPL-3.0.

---

## License & maintainers

- **License:** GPL-3.0 (see [`LICENSE`](./LICENSE)).
- **Maintainers:** Code Crunch Club curriculum council, Agents working group.
- **Status:** active — content versioned per cohort; the field moves and we move with it.

Issues, errata, and curriculum proposals: open an issue on this repository or on the master curriculum repo. Pull requests welcome under the contribution guide.


---

<!-- CCWW:AUTO-INDEX:START — generated by scripts/restructure_course_repos.py; edit ABOVE this marker -->

## Course at a glance

| Section | Count |
| --- | --- |
| Curriculum entries | 25 |
| Projects | 0 |
| Past sessions | 0 |

## Curriculum

- [SYLLABUS](curriculum/SYLLABUS.md)
- [week 01 llm as a system component](curriculum/week-01-llm-as-a-system-component/README.md)
- [week 02 tokens context and sampling](curriculum/week-02-tokens-context-and-sampling/README.md)
- [week 03 prompt engineering as engineering](curriculum/week-03-prompt-engineering-as-engineering/README.md)
- [week 04 tool calling and structured output](curriculum/week-04-tool-calling-and-structured-output/README.md)
- [week 05 the agent loop](curriculum/week-05-the-agent-loop/README.md)
- [week 06 local inference bring up](curriculum/week-06-local-inference-bring-up/README.md)
- [week 07 embeddings and vector search](curriculum/week-07-embeddings-and-vector-search/README.md)
- [week 08 chunking and document processing](curriculum/week-08-chunking-and-document-processing/README.md)
- [week 09 reranking hybrid search structured retrieval](curriculum/week-09-reranking-hybrid-search-structured-retrieval/README.md)
- [week 10 vector stores in production](curriculum/week-10-vector-stores-in-production/README.md)
- [week 11 memory systems and context budgeting](curriculum/week-11-memory-systems-and-context-budgeting/README.md)
- [week 12 multimodal rag and evaluation](curriculum/week-12-multimodal-rag-and-evaluation/README.md)
- [week 13 langgraph and the graph pattern](curriculum/week-13-langgraph-and-the-graph-pattern/README.md)
- [week 14 mastra inngest and typescript agent stacks](curriculum/week-14-mastra-inngest-and-typescript-agent-stacks/README.md)
- [week 15 mcp the cross vendor tool protocol](curriculum/week-15-mcp-the-cross-vendor-tool-protocol/README.md)
- [week 16 fine tuning at the modern scale](curriculum/week-16-fine-tuning-at-the-modern-scale/README.md)
- [week 17 safety prompt injection jailbreaks output filtering](curriculum/week-17-safety-prompt-injection-jailbreaks-output-filtering/README.md)
- [week 18 observability for agentic systems](curriculum/week-18-observability-for-agentic-systems/README.md)
- [week 19 vllm in production](curriculum/week-19-vllm-in-production/README.md)
- [week 20 nemo inference and the nvidia stack](curriculum/week-20-nemo-inference-and-the-nvidia-stack/README.md)
- [week 21 cost engineering and model routing](curriculum/week-21-cost-engineering-and-model-routing/README.md)
- [week 22 capstone sprint a architecture retrieval memory](curriculum/week-22-capstone-sprint-a-architecture-retrieval-memory/README.md)
- [week 23 capstone sprint b agents mcp eval serving](curriculum/week-23-capstone-sprint-b-agents-mcp-eval-serving/README.md)
- [week 24 chaos drill eval in prod on call](curriculum/week-24-chaos-drill-eval-in-prod-on-call/README.md)

## In this course

- **Community** — [community/](community/)
- **Curriculum** — [curriculum/](curriculum/)
- **Projects** — [projects/](projects/)
- **Resources** — [resources/](resources/)
- **Past sessions** — [past-sessions/](past-sessions/)

<!-- CCWW:AUTO-INDEX:END -->
