# C23 · Crunch Agents — Charter

> Design rationale for the AI Agent Systems Engineering track.
> Version 1.0 · ratified 2026-05-13.

---

## Why this track exists

The Code Crunch academy already teaches AI. **C5 · Crunch AI · Data Science** owns classical machine learning through deep learning: feature engineering, gradient boosting, tree ensembles, CNNs, RNNs, the transformer as an architecture, PyTorch as a research and training tool. That track ends where the modern AI engineering discipline begins.

The modern discipline is not training models. It is **building systems on top of models you mostly do not train**: retrieval pipelines over private data, agents that call tools, multi-agent orchestration graphs, evaluation harnesses that gate releases, observability for non-deterministic systems, local inference clusters serving open-weights models, cost engineering at token granularity, safety against prompt injection. The failure modes are distinct from classical ML. The on-call experience is distinct. The hiring market for these skills is distinct.

C23 exists because that surface is too large to bolt onto C5, and because treating it as a footnote inside a deep-learning course produces engineers who can train a transformer but cannot ship one. The split is editorial as much as technical: C5 is the model builder's track; C23 is the systems engineer's track on top of models.

---

## Why 24 weeks

The honest answer is that the surface area is large and shifts every quarter. A 12-week or 15-week course on agents and RAG ends up either superficial (a tour of frameworks) or punishing (a forced march that drops eval, observability, or production entirely). Both failure modes exist in the market. We have seen the syllabi.

Twenty-four weeks gives us four phases of roughly six weeks each:

1. **Foundations** (1–6) — LLM internals as systems, prompt engineering as engineering, the agent loop, local inference. The base camp.
2. **RAG & Memory Systems** (7–12) — retrieval done with measurement, memory designed as a budget, multimodal pipelines, Ragas. The thing most products actually are.
3. **Agents & Orchestration** (13–18) — LangGraph, Mastra, MCP, safety, observability, the fine-tune-or-not decision. The thing most product roadmaps want to be.
4. **Production AI & Capstone** (19–24) — vLLM, NeMo Inference, cost, eval-in-prod, the capstone, the chaos drill. The thing that wakes you up at 3 AM.

Anything shorter forces one of these to be a footnote. Phases I and II are the durable spine; phases III and IV are where the field is moving fastest and where the cohort-to-cohort updates concentrate. Twenty-four weeks lets us keep the spine deep while refreshing the perimeter.

---

## Topic ordering: prompt-as-code → first agent → RAG → memory → multi-agent → production

Most courses on this material get the ordering backwards. They start with a framework — LangChain, AutoGen, CrewAI — and teach the agent loop as a side effect. Three months later, the student has a `crew.kickoff()` call that works on the demo and falls over on the first real input, because the underlying loop was never internalized.

C23 inverts that. We teach in the order that produces durable understanding:

1. **Prompt-as-code first** (week 3). Before you build anything, you must be able to version a prompt, diff a prompt, and write a regression test for a prompt. Without this, every later layer is built on sand.
2. **A hand-rolled agent loop** (week 5). Before you reach for LangGraph, you write the loop yourself — ~150 lines, no framework. You must be able to read an agent trace and explain exactly which step failed and why. Frameworks are leverage, but only if you can read what they generated.
3. **Retrieval done as engineering** (weeks 7–10). Embeddings, chunking, hybrid search, reranking, vector stores — each measured, each A/B'd, each defended in a memo. RAG is mostly chunking and reranking; both can be measured.
4. **Memory as a budget** (week 11). The three tiers — episodic, semantic, procedural — and the discipline of treating context like a cache with eviction.
5. **Multi-agent orchestration** (weeks 13–14). Only now does a framework earn its keep. LangGraph and Mastra solve the problems you have already felt.
6. **Safety, observability, fine-tuning** (weeks 15–18). Threat modeling, OTel Gen-AI tracing, fine-tuning as a debugging tool of last resort.
7. **Production** (weeks 19–24). vLLM clusters, NeMo Inference, cost engineering, eval-in-prod, the chaos drill.

This ordering produces engineers who can debug their own systems. The reverse ordering produces engineers who can debug `crew.kickoff()`.

---

## Open-source-first, vendor-aware

The course teaches local inference before vendor APIs. Week 1 calls vendors only as a comparison; week 6 brings up **Ollama**, **llama.cpp**, and **vLLM** on the student's own hardware (or a rented L4 / A10). The student feels the cost of a million tokens before they spend $1 of someone else's API budget.

The open-source spine of this course is:

- **Inference:** Ollama, llama.cpp, vLLM, SGLang, TGI, TensorRT-LLM, NVIDIA NeMo Inference (the NVIDIA stack is open-licensed even when NVIDIA-specific).
- **Models:** Llama (Meta), Qwen (Alibaba), Mistral, Gemma (Google), Phi (Microsoft), DeepSeek — open-weights with real licenses, read carefully.
- **Frameworks:** LangGraph (Python), Mastra (TypeScript), Inngest, the Anthropic `claude-agent-sdk` (open source), the OpenAI Agents SDK (open source).
- **Retrieval:** pgvector, Qdrant, Weaviate, Milvus, Chroma; open embeddings (BGE, GTE, jina, nomic, E5); open rerankers (bge-reranker, ColBERT).
- **Evaluation:** Ragas, DeepEval, promptfoo, TruLens.
- **Observability:** Langfuse (open), Arize Phoenix (open), OpenTelemetry Gen-AI semantic conventions.
- **MCP and the OpenClaw family:** the Model Context Protocol is the open cross-vendor tool protocol. Around it, an emerging ecosystem of open-source Claude-compatible runtimes, MCP gateways, and self-hosted agent loops has accreted — referred to in this curriculum as the **OpenClaw family**: open agent runtimes that speak MCP natively, open MCP servers for common tool surfaces, and community-maintained Claude-compatible loops. The course teaches this ecosystem first and Anthropic's hosted Claude second.

The vendor side — OpenAI, Anthropic, Google, Amazon Bedrock — is taught as the **production scale path** and the **frontier-capability path**, never as the only path. We are vendor-aware, not vendor-loyal. When Claude 4 or GPT-5 is the right tool for the hard route in a routing layer, we use it; we route the easy 80% to a self-hosted 7B because we know that 80% does not need a frontier model.

This stance is not ideology. It is engineering. An engineer in Lagos, Lahore, La Paz, or Lansing should be able to take this course and build the same systems as an engineer at a frontier lab — on rented commodity GPUs, with open-weights models, with open frameworks. Vendor lock-in is the opposite of that promise.

---

## Relationship to other tracks

| Track | Relationship to C23 |
| --- | --- |
| **C1 — Code Crunch Convos (Python)** | Hard prerequisite. C23 is written for engineers who already write Python without struggling. |
| **C5 — Crunch AI · Data Science** | Recommended but not required. C5 buys you classical ML and PyTorch intuition; C23 re-derives the LLM-specific systems concepts we need. |
| **C17 — Crunch Pro Python** | Recommended. Async, performance, packaging, and C-extension awareness all pay off in agent code. |
| **C14 — Crunch Linux** | Helpful, especially for the local inference and cluster phases. |
| **C18 / C19 — Crunch GCP / AWS** | Excellent follow-ons. C23 stays cloud-agnostic; if you are going to run an agentic product at scale, learn the cloud you are running it on. |
| **C22 — Crunch Mesh** | Excellent follow-on. The service architecture *around* an agentic system is a microservices system; C22 hardens that surface. |
| **C24 — Crunch Robotics** | Spiritually adjacent. Robotics agents share the loop discipline of LLM agents; the failure modes differ but the orchestration patterns overlap. |

**Pathway C (the AI engineer pathway)** is the canonical route: `C1 → C5 → C17 → C23`. A shorter pathway, `C1 → C17 → C23`, works for engineers who do not need classical ML and want the modern stack directly.

---

## What we will and will not teach

**We will teach** the durable concepts: the agent loop, retrieval as engineering, memory as a budget, evaluation as engineering, observability as engineering, prompt-as-code, MCP as the open tool protocol, vLLM as the open serving primitive, the discipline of cost accounting at token granularity, threat modeling for tool surfaces.

**We will teach** specific tools as the current best instance of each concept, with the explicit acknowledgement that the tool roster will rotate. LangGraph today, possibly something else in three years. Qdrant today, possibly something else next year. The concept is durable; the tool is the cohort's instance of it.

**We will not teach** every framework. There are a dozen agent frameworks; we teach LangGraph and Mastra in depth, name and critique the others, and trust the student to evaluate a new one from first principles. The same applies to vector stores, embedding models, eval libraries, and tracing tools.

**We will not teach** "agents will replace software engineers." The course is the opposite of that pitch. It teaches students how to be the engineers that ship and maintain the systems other people call "agents," and how to read past the marketing for what is actually under the hood.

---

## The honest framing

This is a fast-moving field. A syllabus written in 2026 will not be the syllabus shipped in 2028. The shape of the field — open-weights catching up to closed, retrieval being mostly chunking and reranking, agents being mostly loops with budgets, evaluation being mostly golden sets with calibrated judges — has been stable for ~18 months at the time of this charter and is likely to remain stable for another 24 months. The specific frameworks, model names, and benchmark scores will rotate.

The course is built to **survive that rotation**:

- The durable concepts get the deepest coverage.
- The specific tools are taught as instances of the concept, not as the concept.
- Every cohort gets a refresh of the perimeter (frameworks, model rosters, eval libraries, vendor SKUs); the spine moves only on major shifts.
- The reading list is short and opinionated: papers and specs that have held up. New entries are added when they earn it, not when they trend.

We will get some calls wrong. When a course you trusted picks the wrong framework, the way you survive is by having taught the underlying concept well enough that the framework swap is a weekend's work, not a re-enrollment. C23 is built on that contract.

---

## License & maintainers

- **License:** GPL-3.0 (see [`LICENSE`](./LICENSE)).
- **Maintainers:** Code Crunch Club curriculum council, Agents working group.
- **Ratified:** 2026-05-13.

Charter amendments require a council vote and an open issue with a 14-day review window.

*Signed, the Agents working group — Code Crunch Club curriculum council.*
