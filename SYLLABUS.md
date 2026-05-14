# C23 · Crunch Agents — Syllabus

> 24 weeks · ~864 hours full-time (~432 hours self-paced) · Crunch Labs tier · GPL-3.0
> Sub-brand **Agents** · accent `#A855F7` (synth-violet)
> Prereqs: **C1** + (**C5** *or* equivalent) + **C17** helpful; Linux + Docker comfort required

---

## Course statistics

| | |
| --- | --- |
| **Phases** | 4 (Foundations · RAG & Memory Systems · Agents & Orchestration · Production AI & Capstone) |
| **Weeks** | 24 |
| **Hours (full-time)** | ~864 |
| **Hours (self-paced)** | ~432 |
| **Mini-projects** | 24 (one per week) |
| **Quizzes** | 24 |
| **Capstones** | 1 (production agentic research assistant) |
| **Chaos drills** | 3 (GPU node loss · prompt-injection on a tool · vector index corruption) |
| **License** | GPL-3.0 |

---

## Phase I · Foundations (weeks 1–6)

The goal of Phase I is to make you fluent in LLMs as system components, fluent in prompts as engineering artifacts, fluent in the agent loop, and fluent in standing up local inference on your own hardware. By week 6 you have a working ReAct agent talking to a local 7B model, and you can read its trace and tell a friend exactly why it failed on the third question.

### Week 1 · The LLM as a System Component

- **Topics:** What an LLM actually is from the outside (tokenizer → context window → forward pass → sampling); decoder-only transformer at a systems level (we re-derive the parts that matter, skip the math); the 2026 model landscape — open-weights (Llama 4, Qwen 3, Mistral, Gemma 3, DeepSeek) versus closed-weights (GPT-5 class, Claude 4 class, Gemini 2.5 class); model licensing realities.
- **Lecture (Mon):** "The model is a function from tokens to a distribution over tokens — everything else is sampling, scheduling, and engineering."
- **Lecture (Wed):** "How to read a model card without falling for the benchmark." Reading model cards for Llama, Qwen, Mistral; the gap between leaderboard score and product fit; the licensing reality (commercial use, royalty thresholds, derivative works).
- **Hands-on lab:** Call the same prompt against `gpt-4o-mini`, `claude-3.5-sonnet`, `gemini-2.0-flash`, and a local `qwen2.5:7b` via Ollama. Diff outputs token-by-token. Plot latency, tokens-in, tokens-out, cost. Write a 1-page memo on which to use for what.
- **Mini-project:** Build a CLI tool `llmpick` that takes a `--prompt`, a `--budget`, and a `--latency-target`, queries N models in parallel, and recommends one with reasons.
- **Reading:** Llama 4 model card; Qwen 3 model card; one paragraph from *Attention Is All You Need* per day (yes, really — it gets easier).
- **Skills earned:** Reading a model card, picking a model for a job, reasoning about cost and license.

### Week 2 · Tokens, Context, and Sampling

- **Topics:** Tokenization (BPE, SentencePiece, tiktoken, the Llama tokenizer); context windows and the price of long context; KV cache and why streaming feels fast; sampling parameters (temperature, top-p, top-k, repetition penalty, min-p); structured outputs (JSON-mode, grammar-constrained decoding with `outlines`, `guidance`, `xgrammar`); beam search and why nobody uses it.
- **Lecture:** "Temperature is not creativity; top-p is not diversity. Both are knobs on a distribution."
- **Hands-on lab:** Build a side-by-side tokenization explorer for three open tokenizers; instrument an Ollama session to log per-request token counts and timings; write a script that produces structured JSON output with grammar-constrained decoding via `outlines` and assert its schema.
- **Skills earned:** Token-accurate cost estimation, structured-output engineering, grammar-constrained decoding.

### Week 3 · Prompt Engineering as Engineering

- **Topics:** The prompt is code: version it, diff it, test it; system / user / assistant role separation; few-shot patterns, chain-of-thought (with the 2024 honesty that CoT is not always helpful), self-consistency; role-prompting failure modes; the jailbreak surface; prompt versioning with **promptfoo** and **Langfuse** prompt management; spec-then-implement loops with **Claude Code** and **Cursor**.
- **Lecture:** "If you cannot diff it and test it, it is not a prompt — it is a wish."
- **Hands-on lab:** Take a poorly-performing prompt against a customer-support dataset; build a `promptfoo` test harness with 30 golden examples; iterate through 6 prompt versions; commit each to git with measured pass rate; deliver a regression-tested prompt with reproducible scores.
- **Skills earned:** Prompt versioning, prompt regression testing, structured prompt review.

### Week 4 · Tool Calling and Structured Output

- **Topics:** Function calling across vendors (OpenAI tools, Anthropic tool_use, Gemini function calling); the open-source equivalent on Llama and Qwen models; **MCP** as the cross-vendor protocol — its three transports (stdio, SSE, streamable HTTP); JSON-mode versus grammar-constrained decoding; tool-use safety surface (a tool is a remote-code-execution primitive).
- **Lecture:** "A tool call is a request to take action in the world. Treat it like a remote API call from an untrusted client — because that is what it is."
- **Hands-on lab:** Build a 4-tool agent (calculator, file-read, web-fetch, Python sandbox) that runs against (a) Anthropic's API with `tool_use`, (b) a local Qwen 2.5 7B through Ollama with the same tool schema; measure tool-call accuracy on a fixed 50-task benchmark; write a defense for each tool against malicious arguments.
- **Skills earned:** Cross-vendor tool calling, MCP awareness, tool-input sanitization.

### Week 5 · The Agent Loop

- **Topics:** ReAct (reason + act + observe), plan-and-execute, reflection, self-critique; when each works and when it fails; the "infinite tool-call loop" failure mode; budgets — step budget, token budget, time budget, cost budget; the **Anthropic Claude SDK** (`claude-agent-sdk`), **OpenAI Agents SDK**, **AWS Strands Agents**, **Google ADK** at a survey level.
- **Lecture:** "Most agent failures are not model failures — they are loop failures, budget failures, or tool failures."
- **Hands-on lab:** Implement a ReAct agent from scratch in ~150 lines of Python — no framework — against a local Qwen 7B. Run it against a 25-task agent benchmark (math, web lookup, code execution). Then re-implement with `claude-agent-sdk` against Claude 3.5 Sonnet. Compare pass rate, cost, and code surface area.
- **Skills earned:** Hand-rolling an agent loop, reading agent traces, comparing SDKs honestly.

### Week 6 · Local Inference Bring-Up

- **Topics:** Local inference stack overview — **Ollama** for fast iteration, **llama.cpp** for portability and CPU/Metal, **vLLM** for throughput on CUDA, **SGLang** for structured workloads, **TensorRT-LLM** for NVIDIA-optimized serving, **TGI** for the HF ecosystem; quantization formats — **GGUF**, **AWQ**, **GPTQ**, **bitsandbytes**; speculative decoding; KV-cache reuse; **Apple MLX** as a Mac-native path.
- **Lecture:** "The fastest token is the one you do not generate. Quantization, batching, and KV reuse are not optimizations — they are the product."
- **Hands-on lab:** Bring up the same 7B model on Ollama, llama.cpp, and vLLM (rented L4 or A10, ~$1 in compute). Run a fixed 100-prompt benchmark on each. Plot tokens/sec, p50/p95 latency, VRAM. Quantize to Q4_K_M, AWQ, and FP16; chart the trade-offs.
- **Skills earned:** Standing up local inference, picking a quantization, reading a perf chart honestly.

**Phase I capstone milestone:** A working ReAct agent on a local 7B model, with a tool surface and a measurable benchmark score. Code committed, prompts versioned, traces logged.

---

## Phase II · RAG & Memory Systems (weeks 7–12)

Phase II is about giving the model knowledge it did not have at training time. By week 12 you have a hybrid-retrieval system over a real corpus, evaluated with Ragas, with memory tiers that survive a multi-turn conversation.

### Week 7 · Embeddings and Vector Search

- **Topics:** Embeddings as compressed semantic representations; open embedding families — **BGE** (BAAI), **GTE** (Alibaba), **jina-embeddings-v3**, **nomic-embed-text**, **E5-Mistral**; vendor embeddings (OpenAI `text-embedding-3`, Cohere `embed-v3`, Voyage); the **MTEB** leaderboard and how to read it skeptically; vector similarity (cosine, dot, Euclidean) and why it matters less than you think; ANN indexes (HNSW, IVF, ScaNN).
- **Lecture:** "Embedding choice rarely changes the system; retrieval-strategy choice almost always does."
- **Hands-on lab:** Embed a 50-page legal corpus with three open embeddings (BGE-large, GTE-large, jina-v3) and one vendor (OpenAI `text-embedding-3-large`); index each in pgvector; run a 40-query benchmark; report top-1 / top-5 / MRR per embedding. Defend a choice in a 1-page memo.
- **Skills earned:** Embedding model selection, vector index basics, MTEB literacy.

### Week 8 · Chunking and Document Processing

- **Topics:** Chunking strategies — fixed token-window, semantic-paragraph, recursive, sliding-window with overlap, late chunking; document extraction with **Unstructured**, **MinerU**, **LlamaParse**, **PyMuPDF**; OCR with **Tesseract** and **Surya**; table extraction; multimodal page handling (text + figure + caption); chunk-size as a hyperparameter.
- **Lecture:** "Chunking is the part of RAG that determines whether the rest of RAG is doing anything."
- **Hands-on lab — A/B harness:** Build a chunking-strategy A/B harness comparing **token-window (512/1024 tokens)** vs **semantic-paragraph** vs **Recursive (LangChain-style)** vs **late chunking** over a 50-page legal corpus. Use the same embedding (BGE-large) and the same vector store (pgvector). Report retrieval **MRR**, **Recall@5**, and answer **faithfulness** delta. Pick a winner with reasons.
- **Skills earned:** Document ingestion pipeline, chunking A/B methodology, evaluation as engineering.

### Week 9 · Reranking, Hybrid Search, and Structured Retrieval

- **Topics:** Why dense retrieval is not enough; BM25 and the role of lexical search; **hybrid search** (dense + sparse, RRF fusion); rerankers — **BAAI bge-reranker-v2**, **Cohere reranker**, **ColBERT** and its late-interaction variant; structured retrieval (SQL+LLM, text-to-SQL); query rewriting; HyDE.
- **Lecture:** "A reranker is the cheapest meaningful win in RAG. Use one."
- **Hands-on lab:** Take last week's pipeline, add BM25 (Tantivy or Elasticsearch), add reciprocal-rank fusion, add a bge-reranker-v2 reranker. Measure the lift at each step. Then add a HyDE query rewriter. Chart the cumulative lift on the same 40-query set.
- **Skills earned:** Hybrid retrieval, reranking, query rewriting, the discipline of measuring each layer.

### Week 10 · Vector Stores in Production

- **Topics:** Vector store landscape — **pgvector** (Postgres-native, the default), **Qdrant** (Rust, fast, filtered search), **Weaviate** (graph-leaning, generative search), **Milvus** (massive scale), **Chroma** (developer ergonomics, smaller scale); filtered ANN, metadata indexes, hybrid stores; operational realities — backup, replication, rebuild after schema change, eviction; **GraphRAG** (Microsoft's pattern) and knowledge-graph hybrids; **Agentic RAG** patterns where the agent chooses retrievers.
- **Lecture:** "Pick the vector store with the operational story you can live with at 2 AM, not the one with the best benchmark."
- **Hands-on lab:** Run the same hybrid-retrieval pipeline against pgvector, Qdrant, and Weaviate. Measure ingest throughput, query latency, and operational complexity (lines of config, time-to-recover from a simulated index loss). Write a 1-page architecture memo.
- **Skills earned:** Vector store selection, operational reasoning, GraphRAG awareness.

### Week 11 · Memory Systems and Context Budgeting

- **Topics:** The three memory tiers — **episodic** (turn history), **semantic** (vector + knowledge-graph), **procedural** (tool histories, learned behaviors); summarization strategies (rolling, hierarchical, map-reduce); context-window budgeting like a cache; memory eviction (LRU, salience-weighted); **MemGPT** / **Letta** patterns; hybrid vector + KG memory; long-context model limits (the "lost in the middle" effect).
- **Lecture:** "Context is the most expensive cache on the planet. Spend it like one."
- **Hands-on lab:** Build a chat agent with three memory tiers — a rolling summary of recent turns (episodic), a vector store of user facts (semantic), and a tool-history log (procedural). Run a 40-turn conversation benchmark with a memory regression test (does the agent still remember the user's project name in turn 38?). Compare against a no-memory baseline.
- **Skills earned:** Memory architecture, context budgeting, eviction policy design.

### Week 12 · Multimodal RAG and Evaluation

- **Topics:** Vision-language models — **LLaVA**, **Qwen2-VL**, **Phi-Vision**, **InternVL**; image embeddings (**CLIP**, **SigLIP**); multimodal RAG over PDFs with figures; ASR with **Whisper** and **whisper.cpp**; TTS with **Piper** and **XTTS**; image generation with **SDXL** and **Flux** as adjacent capabilities; **Ragas** as the standard retrieval eval; **DeepEval**, **promptfoo**, **TruLens** at a survey level; LLM-as-judge with calibration.
- **Lecture:** "If you cannot measure it, you cannot ship it. RAG without Ragas is a vibe."
- **Hands-on lab:** Build a Ragas evaluation suite for your Phase II pipeline — faithfulness, context recall, context precision, answer relevancy. Add an LLM-as-judge with a calibration step (10 human-labeled examples). Plot the four Ragas metrics against three pipeline variants. Identify which metric improved most for which change.
- **Skills earned:** Multimodal RAG, the Ragas vocabulary, calibrated LLM-as-judge.

**Phase II capstone milestone:** A hybrid-retrieval system with chunking A/B'd, embeddings picked with reasons, a reranker in place, three memory tiers, and a Ragas evaluation report.

---

## Phase III · Agents & Orchestration (weeks 13–18)

Phase III stacks tools, graphs, and safety on top of the retrieval system. By week 18 you have a multi-agent supervisor system exposing a tool surface over MCP, with safety rails, full observability, and a written threat model.

### Week 13 · LangGraph and the Graph Pattern

- **Topics:** Why frameworks now — when the loop becomes a graph; **LangGraph** state graphs, conditional edges, checkpoints, persistence; supervisor, swarm, and hierarchical patterns; comparison with **AutoGen** and **CrewAI** (and an honest critique — CrewAI's role-play abstraction is appealing but leaks); when a state machine beats an agent.
- **Lecture:** "When the agent gets a fourth tool, you graduate from a loop to a graph. Reach for LangGraph before the loop gets a fourth `if`."
- **Hands-on lab:** Re-implement the week-5 ReAct agent as a LangGraph state graph with explicit nodes (plan, retrieve, execute, critique). Add a persistence layer (SQLite checkpointer) so the agent survives a process kill. Run the same 25-task benchmark; compare lines of code, observability, and resumability.
- **Skills earned:** LangGraph fluency, state-graph design, agent persistence.

### Week 14 · Mastra, Inngest, and TypeScript Agent Stacks

- **Topics:** **Mastra** as a TypeScript-first agent framework (workflows, agents, memory, evals); **Inngest** for event-driven agents and durable execution; **Trigger.dev** for background agent jobs; **Temporal** as an agent orchestrator at scale; comparing Python-first (LangGraph) versus TypeScript-first (Mastra) stacks honestly; when each is the right pick.
- **Lecture:** "Your agent platform is your durability platform. If it cannot resume from step 7 after a crash, it is not production."
- **Hands-on lab:** Build the same supervisor agent in Mastra (TypeScript) and LangGraph (Python). Wire one of them to Inngest for event-driven invocation (a new file in S3 triggers a research run). Compare developer ergonomics, type safety, and the resume-after-crash story.
- **Skills earned:** Mastra fluency, durable execution, polyglot agent design.

### Week 15 · MCP — The Cross-Vendor Tool Protocol

- **Topics:** The **Model Context Protocol** in depth — server, client, transport modes (stdio, SSE, streamable HTTP), tool / resource / prompt primitives; writing an MCP server in Python and TypeScript; consuming MCP from Claude Desktop, Cursor, and a programmatic client; the emerging open-source Claude-compatible / MCP-server runtime family — **the OpenClaw family of tooling** — covering open agent runtimes that speak MCP natively (open-source MCP gateways, self-hosted MCP servers, community-maintained Claude-compatible loops); security review of an MCP server (a tool is RCE).
- **Lecture:** "MCP is the USB-C of agent tooling. It is not the future — it is the present, and it is open."
- **Hands-on lab:** Write two MCP servers — one filesystem tool surface, one custom domain tool (e.g., a private-corpus search) — using the official `mcp` Python SDK. Expose them over stdio and over streamable HTTP. Consume them from a LangGraph agent. Run a basic security review: argument validation, path traversal, and rate limiting. Document the tool surface.
- **Skills earned:** Writing MCP servers, MCP transports, tool-surface security review, OpenClaw-ecosystem literacy.

### Week 16 · Fine-Tuning at the Modern Scale

- **Topics:** When **not** to fine-tune (almost always — try prompt and retrieve first); when to fine-tune (domain vocabulary, output style, latency); **LoRA**, **QLoRA**, **DoRA**; **SFT**, **DPO**, **ORPO**, **KTO**; the **NeMo Framework** for serious training; **Axolotl** and **Unsloth** for accessible fine-tuning; **RLHF** and **RLAIF** at survey depth (we will not run a full RLHF pipeline); dataset engineering for SFT.
- **Lecture:** "Fine-tuning is a debugging tool of last resort. The model is rarely the problem — the prompt, the retrieval, or the eval is."
- **Hands-on lab:** Build a SFT dataset of 500 examples for a narrow domain (e.g., converting natural language to a custom DSL). Fine-tune **Qwen2.5-7B** with **Unsloth** on a single 24 GB GPU (or rented A10). Evaluate against the base model on a held-out 50-example test. Decide whether the fine-tune was worth it.
- **Skills earned:** SFT dataset design, LoRA training, honest cost/benefit on fine-tuning.

### Week 17 · Safety — Prompt Injection, Jailbreaks, and Output Filtering

- **Topics:** Prompt injection as the dominant LLM security issue; direct vs indirect injection; jailbreak surface; output filtering (regex, classifier-based, LLM-judge); content moderation (**Llama Guard**, **OpenAI Moderation**, **Perspective**); hallucination measurement; red-teaming methodology; the **OWASP LLM Top 10**; tool-use threat modeling.
- **Lecture:** "If your agent has a tool, your agent has an attack surface. Threat-model it."
- **Hands-on lab:** Red-team your own week-15 MCP-tool agent. Write 25 adversarial prompts (direct injection, indirect via retrieved document, tool-argument abuse). Measure the attack success rate. Add three defenses: input filtering, structured tool-argument validation, an output classifier. Re-measure. Write a 1-page threat model.
- **Skills earned:** Threat modeling, prompt-injection defense, red-teaming an agent.

### Week 18 · Observability for Agentic Systems

- **Topics:** Tracing every step — **LangSmith**, **Langfuse** (open-source), **Arize Phoenix** (open), **Helicone**; the **OpenTelemetry Gen-AI semantic conventions** for traces and metrics; token-usage accounting per user / per route / per model; latency budgets and SLOs; trace-driven debugging; eval-on-traces (replay a production trace through a new prompt version).
- **Lecture:** "An agentic system without traces is a closed-box. You will eventually re-open it the hard way."
- **Hands-on lab:** Instrument your Phase III system with OpenTelemetry Gen-AI conventions; export to Langfuse (self-hosted) and to Arize Phoenix; build three dashboards — token usage per route, p95 latency per agent step, retrieval-precision over time. Trigger one synthetic failure and find it from the dashboard in under 5 minutes.
- **Skills earned:** OTel Gen-AI conventions, self-hosted Langfuse, trace-driven debugging.

**Phase III capstone milestone:** A multi-agent supervisor system with MCP tool surface, fine-tune-or-not decision documented, threat model written, full OTel tracing.

---

## Phase IV · Production AI & Capstone (weeks 19–24)

Phase IV is about putting it all on a cluster, paying for it, watching it, and surviving an incident. By week 24 you have shipped the capstone, run the chaos drill, and written the postmortem.

### Week 19 · vLLM in Production

- **Topics:** **vLLM** architecture — continuous batching, paged attention, prefix caching; serving config (tensor parallel, pipeline parallel); multi-replica vLLM behind a router; the **LiteLLM** proxy as a vendor-and-self-hosted router; speculative decoding in vLLM; OpenAI-compatible API surface.
- **Lecture:** "Continuous batching is the throughput multiplier that makes self-hosting feasible. Without it, you are paying for idle GPU."
- **Hands-on lab:** Stand up a vLLM server on a rented H100 (~6 hours of compute, ~$12) serving Qwen2.5-14B. Put **LiteLLM** in front. Benchmark throughput at concurrency 1, 8, 32, 128. Compare cost-per-million-tokens against the equivalent vendor API. Write a break-even memo.
- **Skills earned:** vLLM cluster basics, LiteLLM routing, self-hosted economics.

### Week 20 · NeMo Inference and the NVIDIA Stack

- **Topics:** **NVIDIA NeMo Inference** for production serving; **TensorRT-LLM** kernel optimization; **Triton Inference Server** for mixed model fleets; **NeMo Framework** for serious training and customization; **NeMo Guardrails** for safety policies; comparing NeMo to vLLM honestly (NeMo wins on NVIDIA-specific kernel perf and policy tooling; vLLM wins on flexibility, OSS velocity, and operational simplicity).
- **Lecture:** "NVIDIA's stack is the production answer if you are an NVIDIA shop. It is also the most opinionated. Know what you are signing up for."
- **Hands-on lab:** Deploy the same Qwen2.5-14B with NeMo Inference and Triton on the same H100; add a NeMo Guardrails policy that blocks one specific class of prompt injection; benchmark against the vLLM deployment from week 19. Decide which would survive in production for your capstone.
- **Skills earned:** NeMo Inference deploy, Triton serving, NeMo Guardrails as policy.

### Week 21 · Cost Engineering and Model Routing

- **Topics:** Model routing — small model for easy, big model for hard; **semantic cache** with **GPTCache** or a self-built pgvector cache; prompt compression (LLMLingua, summarization); batching — **OpenAI Batch**, **Anthropic Batch**, vLLM continuous batching; per-feature cost accounting; the cost story of self-hosted vs vendor; speculative decoding as a cost lever; the carbon story.
- **Lecture:** "The cheapest token is the one you do not generate. The second cheapest is the one a 7B handles instead of a 70B."
- **Hands-on lab:** Build a routing layer that sends "easy" queries to local Qwen2.5-7B (vLLM) and "hard" queries to Claude 3.5 Sonnet, using a small classifier. Add a semantic cache with a 0.92 cosine threshold. Run a 500-query workload; measure cost reduction versus a Claude-only baseline. Plot the cache-hit rate over time.
- **Skills earned:** Model routing, semantic cache, batched inference, cost dashboards.

### Week 22 · Capstone Sprint A — Architecture, Retrieval, Memory

- **Topics:** Capstone kickoff. Architecture review. Retrieval pipeline build. Memory wiring. The capstone supervisor agent draft.
- **Lecture:** "A capstone is an architecture document with code attached. Write the document first."
- **Hands-on lab:** Build the **Production Agentic Research Assistant** capstone's retrieval and memory layers over a 10 GB private corpus. Land hybrid retrieval (BM25 + dense + reranker). Land episodic + semantic + procedural memory. Produce a Mermaid architecture diagram. Submit a 6-page architecture document.
- **Skills earned:** Architecture-document writing, sprint planning, integrating Phase I–III work.

### Week 23 · Capstone Sprint B — Agents, MCP, Eval, Serving

- **Topics:** Multi-agent supervisor + retrieval-agent + code-agent + writing-agent. MCP tool surface (filesystem, web, calculator, custom). vLLM cluster deploy. Ragas + calibrated LLM-as-judge eval suite. OTel Gen-AI tracing. Cost-tracked routing between local 7B/13B and a frontier vendor model.
- **Lecture:** "The last 10% of an agent is 90% of the engineering. Pick what to drop early."
- **Hands-on lab:** Finish the capstone — supervisor graph in LangGraph, MCP tool surface live, vLLM cluster serving the local tier, vendor API serving the hard tier, full eval suite green on a 100-question gold set, OTel traces flowing to Langfuse and Phoenix. Ship a live deploy URL or a runnable container image.
- **Skills earned:** End-to-end agentic system delivery.

### Week 24 · Chaos Drill, Eval-in-Prod, On-Call

- **Topics:** **Eval-in-prod** with shadow traffic; **blue/green model deploys**; **canary by user cohort**; the on-call runbook for an agentic system; the chaos drill; the postmortem.
- **Lecture:** "You do not know if your system is production until you have lost a node, eaten an attack, and corrupted an index — on purpose, in a controlled window."
- **Hands-on lab — chaos drill:** Run three drills against your capstone in a single 4-hour window. (a) **GPU node loss:** kill one vLLM replica; verify the LiteLLM router fails over to the remaining replicas and to the vendor fallback. (b) **Prompt-injection attack on a tool:** inject a malicious instruction via a retrieved document; verify your week-17 defenses hold; if they do not, write the patch. (c) **Retrieval index corruption:** corrupt 5% of the vector store; measure the impact on Ragas faithfulness; restore from backup; verify recovery time. Write a postmortem.
- **Skills earned:** Eval-in-prod, blue/green deploys, chaos engineering for agents, postmortem writing.

**Phase IV capstone deliverable:** Production agentic research assistant + chaos drill postmortem + 5-minute video walkthrough + interview-prep packet.

---

## Assessment matrix

| Component | Weight | Cadence | Pass bar |
| --- | --- | --- | --- |
| Weekly mini-projects | 30% | Weekly (24 total) | ≥18 passed at "meets" or better |
| Quizzes | 10% | Weekly (24 total) | ≥70% average |
| Phase capstone milestones | 20% | End of phases I, II, III | All three at "meets" |
| Final capstone (system + video + postmortem) | 25% | Weeks 22–24 | "meets" on the public rubric |
| Career engineering pack (interviews + runbook + portfolio) | 10% | Rolling | All three artifacts shipped |
| Citizenship — code reviews, paper-club, studio attendance | 5% | Rolling | Visible participation |

**Rubric** for the final capstone is published at `capstone/RUBRIC.md` and is graded by a sealed-review panel of two instructors plus one external reviewer drawn from the alumni network.

---

## Capstone specification

### Title

**Production Agentic Research Assistant**

### One-paragraph spec

A multi-agent system with a supervisor + retrieval-agent + code-agent + writing-agent; hybrid retrieval over a 10 GB private corpus; memory tiers (episodic + semantic + procedural); MCP-server tool surface (filesystem, web, calculator, custom); served on a self-hosted vLLM cluster with continuous batching; full eval suite (Ragas + LLM-as-judge with calibration); OpenTelemetry Gen-AI semantic conventions for tracing; cost-tracked routing between local 7B/13B and a vendor frontier model; red-team report; chaos-drill postmortem.

### Required deliverables

1. **Live deploy URL** or a `docker compose up`-runnable container image.
2. **Architecture diagram** in Mermaid, committed to the repo, kept in sync with reality.
3. **5-minute video walkthrough** narrated by the student, demonstrating one happy path, one tool call, one retrieval, and one failure mode.
4. **Evaluation report** with Ragas metrics (faithfulness, context recall, context precision, answer relevancy) on a 100-question gold set, plus a calibrated LLM-as-judge score on a 50-question subset.
5. **Cost report** — total cost per query at the median, p95, and p99, broken down by route (local vs vendor) with cache-hit accounting.
6. **Observability dashboard** — three views: token usage by route, latency by agent step, retrieval precision over time.
7. **Red-team report** — 25 adversarial prompts attempted, defense success rate before and after hardening, threat model.
8. **Chaos-drill postmortem** — GPU node loss, prompt-injection on a tool, retrieval index corruption — what failed, what held, time-to-recover, follow-up actions.

### Architecture (default, can vary)

```text
                ┌──────────────┐
                │  Supervisor  │   LangGraph state machine
                └──────┬───────┘
        ┌──────────────┼──────────────┬───────────────┐
        v              v              v               v
 ┌────────────┐ ┌─────────────┐ ┌────────────┐  ┌────────────┐
 │ Retrieval  │ │  Code Exec  │ │  Writing   │  │  Critique  │
 │   Agent    │ │   Agent     │ │   Agent    │  │   Agent    │
 └─────┬──────┘ └──────┬──────┘ └─────┬──────┘  └────────────┘
       │               │              │
       v               v              v
 ┌────────────┐  ┌────────────┐  ┌────────────┐
 │  Hybrid    │  │   MCP      │  │  Memory    │
 │ retrieval  │  │  servers   │  │  tiers     │
 │ (BM25 +    │  │ (FS, web,  │  │ (epi+sem+  │
 │  dense +   │  │  calc,     │  │  proc)     │
 │  reranker) │  │  custom)   │  │            │
 └─────┬──────┘  └────────────┘  └────────────┘
       v
 ┌────────────┐
 │ pgvector / │
 │  Qdrant    │
 └────────────┘

Serving:  vLLM cluster (local 7B/13B)  ──LiteLLM router──┐
Vendor:   Claude / GPT for "hard" routes  ───────────────┘
Tracing:  OpenTelemetry Gen-AI → Langfuse + Arize Phoenix
Eval:     Ragas + calibrated LLM-as-judge (offline + shadow online)
```

### Chaos drill (required, week 24)

In one 4-hour window:

1. **GPU node loss.** Kill one vLLM replica. Verify automatic failover via the LiteLLM router. Measure user-visible impact.
2. **Prompt-injection attack on a tool.** Inject a malicious instruction via a retrieved document. Verify the week-17 defenses hold or write the patch. Document the attack and the defense.
3. **Retrieval index corruption.** Corrupt 5% of the vector store. Measure the Ragas-faithfulness regression. Restore from backup. Verify recovery time and steady-state restoration.

A written postmortem in the standard incident format (timeline, what happened, what worked, what did not, action items) is required.

---

## Career engineering pack

Shipped by graduation, gated on capstone delivery:

1. **`interview-prep/`** — system-design drills (design a RAG product, design a multi-agent system, design an LLM gateway, design an eval pipeline); technical drills (write a chunker, write a ReAct loop, write an MCP server tool, write a routing layer); behavioral drills for AI-first companies.
2. **`production-runbook.md`** — a real, narrative on-call runbook for an LLM-backed product: alerts you respond to, dashboards you read, common incident classes (cost spike, latency spike, hallucination spike, attack), escalation, postmortem template.
3. **`portfolio.md`** — three projects from C23 polished for a recruiter: the capstone, the chunking A/B harness (week 8), the multi-agent supervisor (week 13). Each with one image, two paragraphs, links to repo and video, and a "if I had two more weeks" section.

---

## Tools roster

The course teaches concepts; the tools below are the **current cohort's instances** of those concepts. Expect roughly one in three to rotate per cohort as the field shifts. The roster is published with the syllabus so a student can audit it against the field on day one.

### Local inference

| Tool | Role | Why on the roster |
| --- | --- | --- |
| **Ollama** | Fast local iteration, model registry | Lowest friction; great on Apple Silicon |
| **llama.cpp** | Portable CPU/GPU inference, GGUF | The portable substrate; runs anywhere |
| **vLLM** | High-throughput GPU serving | Continuous batching, paged attention; the open serving primitive |
| **SGLang** | Structured-output-heavy workloads | Wins on grammar-constrained workloads |
| **TensorRT-LLM** | NVIDIA-optimized kernels | When you need every microsecond on H100/H200 |
| **TGI** | HF-ecosystem serving | Pragmatic when you live in HF |
| **NVIDIA NeMo Inference** | Enterprise NVIDIA-stack serving | Policy-first deploys with NeMo Guardrails |
| **Apple MLX** | Mac-native inference | First-class Apple Silicon path |

### Agent frameworks & SDKs

| Tool | Role |
| --- | --- |
| **LangGraph** | Python state-graph orchestration |
| **Mastra** | TypeScript-first agent framework |
| **Inngest** | Event-driven durable execution |
| **Anthropic `claude-agent-sdk`** | Claude-native agent loop |
| **OpenAI Agents SDK** | OpenAI-native agent loop |
| **AWS Strands Agents** | AWS-native agent surface |
| **Google ADK** | Google-native agent surface |
| **AutoGen / CrewAI** | Taught for context; critiqued honestly |
| **Temporal** | Workflow engine repurposed as agent orchestrator at scale |

### Retrieval

| Tool | Role |
| --- | --- |
| **pgvector** | Postgres-native vector store; the default |
| **Qdrant** | Rust-based, filtered ANN |
| **Weaviate** | Generative + graph-leaning |
| **Milvus** | Massive scale |
| **Chroma** | Developer ergonomics |
| **BGE / GTE / jina / nomic-embed / E5** | Open embedding families |
| **bge-reranker-v2 / ColBERT / Cohere reranker** | Rerankers |
| **Unstructured / MinerU / LlamaParse / PyMuPDF** | Document extraction |
| **Tesseract / Surya** | OCR |

### Memory

| Tool | Role |
| --- | --- |
| **Letta (formerly MemGPT)** | Tiered memory with eviction |
| **Zep** | Conversation memory store |
| **Mem0** | Long-term memory layer |
| **pgvector + Postgres KG schema** | Self-built memory tier (preferred path) |

### MCP & the OpenClaw family

| Tool | Role |
| --- | --- |
| **Anthropic MCP Python SDK** | Canonical MCP server/client in Python |
| **MCP TypeScript SDK** | Canonical MCP server/client in TS |
| **Claude Desktop / Cursor** | MCP clients you can test against |
| **Open MCP gateways** (OpenClaw family) | Self-hosted MCP routers and aggregators |
| **Open Claude-compatible runtimes** (OpenClaw family) | Community-maintained MCP-native agent loops |

### Evaluation & observability

| Tool | Role |
| --- | --- |
| **Ragas** | The standard retrieval eval |
| **DeepEval** | Pytest-style LLM evals |
| **promptfoo** | Prompt regression harness |
| **TruLens** | Eval + tracing |
| **Langfuse** (open, self-hostable) | Tracing + prompt management |
| **LangSmith** | Hosted tracing (vendor) |
| **Arize Phoenix** (open) | Tracing + eval-in-prod |
| **OpenTelemetry Gen-AI semconv** | The standard for cross-vendor traces |

### Fine-tuning

| Tool | Role |
| --- | --- |
| **Unsloth** | Single-GPU LoRA/QLoRA; the friendliest entry |
| **Axolotl** | Multi-GPU SFT/DPO; production-credible |
| **NVIDIA NeMo Framework** | Serious training at scale |
| **TRL (HuggingFace)** | SFT/DPO/PPO building blocks |

### Multimodal

| Tool | Role |
| --- | --- |
| **LLaVA / Qwen2-VL / Phi-Vision / InternVL** | Open VLMs |
| **CLIP / SigLIP** | Image embeddings |
| **Whisper / whisper.cpp** | ASR |
| **Piper / XTTS** | TTS |
| **SDXL / Flux** | Image generation (adjacent) |

### Workflow automation

| Tool | Role |
| --- | --- |
| **n8n** | Visual workflow builder, good for agent triggers |
| **Inngest / Trigger.dev** | Background and event-driven agent jobs |
| **Temporal** | Durable workflows at scale |
| **Airflow** | Taught for context and critique |

---

## Weekly cadence (canonical)

| Day | Block | Hours | Typical content |
| --- | --- | --- | --- |
| Mon | Lecture 1 + Lab 1 | 5 | Concept intro, runnable lab starter |
| Tue | Reading + studio time | 3 | Paper-of-the-week + open Slack hours |
| Wed | Lecture 2 + Lab 2 | 5 | Deeper dive, code review, mini-project sprint |
| Thu | Mini-project work | 4 | Solo or pair |
| Fri | Studio | 4 | Office hours, debugging clinic, eval-run reviews |
| Sat | Optional paper club | 1 | Voluntary, instructor-led, 30–60m |
| Sun | Quiz + recap | 1 | Auto-graded, 10 questions |

Self-paced cohorts cover the same content on a 12h/week schedule and finish in ~48 weeks. Full-time cohorts run ~36h/week and finish in 24.

---

## Weekly mini-project rubric (canonical)

Every weekly mini-project is graded on the following four-axis rubric. The same axes apply to the phase capstones and the final capstone.

| Axis | Weight | What "meets" looks like |
| --- | --- | --- |
| **Correctness** | 30% | The system does what the spec says, on the provided test inputs and at least one self-authored test input. |
| **Engineering quality** | 25% | Readable code, sensible structure, error handling, no obvious foot-guns. CI passes. |
| **Measurement** | 25% | A reported metric (Ragas, MRR, accuracy, latency, cost) with the method of measurement documented. Vibes do not count. |
| **Write-up** | 20% | A one-page README that explains the design choice, the metric, and one failure mode the student observed. |

Graders are instructed to **fail vibes-only submissions** — a working demo with no measurement is not a "meets."

---

## Phase capstone milestones (deeper detail)

### Phase I milestone (end of week 6)

- A working ReAct agent against a local 7B model.
- Tool surface: at minimum calculator, file-read, web-fetch.
- A reported score on a 25-task benchmark (mix of math, web lookup, code execution).
- Prompts versioned in git; promptfoo regression tests committed.
- A trace from one run that the student can narrate step-by-step.

### Phase II milestone (end of week 12)

- A hybrid-retrieval system over a real corpus (≥10 documents, ≥100 pages).
- Chunking A/B'd; embedding picked with reasons; reranker in place.
- Three memory tiers wired (episodic, semantic, procedural).
- Ragas report on a 40-question gold set: faithfulness, context recall, context precision, answer relevancy.
- A 6-page architecture memo committed.

### Phase III milestone (end of week 18)

- A multi-agent supervisor system with at least three subordinate agents (retrieval, code, writing).
- MCP tool surface with at least two self-authored MCP servers.
- Threat model written; week-17 defenses live.
- Full OpenTelemetry Gen-AI tracing flowing to Langfuse (self-hosted) and Phoenix.
- A 1-page fine-tune-or-not decision document for the capstone's domain.

### Phase IV milestone (end of week 24)

- The capstone, deployed live or runnable from a single `docker compose up`.
- Full eval suite green on a 100-question gold set.
- Chaos drill executed; postmortem committed.
- 5-minute video walkthrough recorded.
- Career engineering pack shipped.

---

## Failure modes the course is built to prevent

These are the failure modes seen across other agentic-course graduates. C23 is built to prevent each one.

1. **"Framework-first" graduate.** Can run `crew.kickoff()` but cannot read an agent trace. **Prevented by:** week-5 hand-rolled loop, week-13 graph re-implementation, mandatory trace reading.
2. **"Vibes-only eval" graduate.** Ships demos with no measurement. **Prevented by:** rubric weight on measurement, Ragas in week 12, regression-testing in week 3.
3. **"Vendor-locked" graduate.** Cannot operate without OpenAI or Anthropic credentials. **Prevented by:** week-6 local inference bring-up, week-19 vLLM cluster, ~80% of labs run on local or open-weights.
4. **"Security-blind" graduate.** Has never thought about prompt injection. **Prevented by:** week-17 threat-model lab, capstone red-team requirement, chaos-drill prompt-injection scenario.
5. **"Single-model" graduate.** Knows one vendor's SDK. **Prevented by:** week-1 cross-vendor comparison, week-21 routing lab, ~6 different model classes touched across the course.
6. **"No-observability" graduate.** Cannot debug a production issue. **Prevented by:** week-18 OTel lab, mandatory tracing in capstone, dashboard requirement.

---

## FAQ

**Do I need C5 (Crunch AI · Data Science)?**
No. It helps in weeks 16–17 (fine-tuning) and gives you intuition for PyTorch. The course re-derives what you need from a systems perspective. If you do not have it, you will work a little harder during the fine-tune weeks; you will not be blocked.

**Do I need a GPU?**
Not on day one. A 16 GB laptop carries you through week 6. From week 7 onward, a 24 GB GPU (3090/4090) or a rented A10/L4 (~$0.50/h) helps. Two specific labs benefit from H100 access; we use rented spot instances at ~$2–$3/h. The total compute budget is ~$30 if you rent for those specific labs.

**Is this a LangChain course?**
No. We teach LangGraph in week 13 and use it where it earns its keep (state graphs with persistence). We teach Mastra in week 14 as the TypeScript counterpart. We do not teach the classic LangChain `LLMChain` abstraction; it is the wrong unit of composition for production agents.

**Is this a "vibe-coding" course?**
No. The course teaches you to use **Claude Code**, **Cursor**, and **Aider** as agentic dev tools and to run a spec-then-implement loop, because that is how the modern AI engineer works in 2026. But the deliverables are measured systems with measured metrics. You will write tests. You will read traces. You will be on-call.

**How current is the syllabus?**
The spine moves on multi-year timescales; the perimeter (frameworks, model names, vendor SKUs) updates every cohort. The roster above is the current cohort's instance.

**Can I take this part-time?**
Yes — the self-paced cadence is ~12h/week and finishes in ~48 weeks.

**Do I get a certificate?**
On capstone completion + sealed-review pass, yes — a Crunch Labs C23 completion seal, with public portfolio links.

---

## Reading list (durable, not exhaustive)

A short, opinionated list of papers and posts that hold up. The course updates this every cohort; the entries below are the durable spine.

- *Attention Is All You Need* (Vaswani et al., 2017) — for shape, not for math.
- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020).
- *ReAct: Synergizing Reasoning and Acting in Language Models* (Yao et al., 2022).
- *vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention* (Kwon et al., 2023).
- *Lost in the Middle: How Language Models Use Long Contexts* (Liu et al., 2023).
- *Toolformer* / *Gorilla* / *ToolLLM* — tool-use lineage.
- *Constitutional AI* and *RLAIF* — alignment lineage.
- *GraphRAG* (Microsoft, 2024).
- *Late Chunking* (Jina, 2024).
- The Anthropic *Model Context Protocol* spec.
- The OpenTelemetry Gen-AI semantic conventions document.

---

*Licensed GPL-3.0 like the rest of the academy.*
