# Lecture 2 — Memory Tiers and the Supervisor Draft: The Contracts the Agents Plug Into

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can wire the three memory tiers — episodic, semantic, procedural — behind clean read/write interfaces the supervisor and sub-agents will use; write a memory regression test that proves a load-bearing fact survives to a late turn; and draft the supervisor agent and the sub-agent contracts (retrieval, code, writing, critique) as interfaces, so the foundation is shaped for what week 23 plugs into it.

Lecture 1 built the retrieval substrate and the architecture document. This lecture builds the *other half* of the Sprint A foundation — memory — and drafts the agent contracts that the whole capstone is organized around.

Memory is the component students most often under-build, because it's invisible when it works and the demo "seems fine" for a few turns. So this lecture is deliberately concrete: the three tiers' implementations, their interfaces, the regression test that catches the under-building, and the diagnosis path when it fails. By the end you should be able to wire a memory layer that survives a real session, not just a three-turn demo.

The throughline:

> **Memory is what makes the agent more than a stateless function. The three tiers answer three different questions — what have we said, what do we know, what have we done — and an agent that can't answer all three forgets the user between turns.** Get the memory interfaces right and the agents have a coherent state to work against; get them wrong and every agent reinvents memory badly.

The stakes are higher than they look. Without memory, the capstone is a glorified search box — every query starts from zero, the user re-explains their context every message, and the "research assistant" can't actually *assist* with a multi-turn research task because it remembers nothing of the research so far. Memory is what turns a stateless Q&A bot into an assistant that builds on a conversation — and a *production* memory system, as opposed to "stuff the transcript in the context," is what makes that work past the third turn and within a sane token budget. This week's memory work is the difference between a demo and a usable system.

---

## 1. Why three memory tiers, not one

A naive agent has no memory — each turn is a fresh function call, and the user re-explains their project every message. A slightly-less-naive agent stuffs the whole conversation into the context window, which works until the conversation is long, the context is expensive, and the model gets "lost in the middle" (Liu et al., 2023) of a buried history. The production answer (week 11) is *three tiers*, each answering a different question and each managed differently:

**Episodic memory — "what have we said?"** A rolling or hierarchical *summary* of the conversation's turns. Not the raw transcript (too long, too expensive) but a compressed running summary that captures the gist: the user's goal, the decisions made, the current state. As the conversation grows, old turns are summarized into the rolling summary and dropped from the active context. This is the context-budgeting tier — it keeps the conversation's *meaning* in a bounded number of tokens.

**Semantic memory — "what do we know?"** *Facts* about the user and the task, stored as vector-searchable (and optionally graph-structured) entries: "the user's project is named Halibut," "the deadline is March 15," "they prefer Python." These are durable facts that persist across the conversation and can be retrieved by relevance — when a later turn mentions "the project," semantic memory surfaces "Halibut." This is the knowledge tier — it's where the agent *remembers facts*, not just summaries.

**Procedural memory — "what have we done?"** A *log* of tool calls and actions taken: "searched the corpus for X, got these results," "ran this code, it returned Y," "called the calculator with Z." This lets the agent avoid repeating actions, reference prior results, and reason about its own trajectory. This is the action-history tier — it's the agent's record of its own behavior.

The three are *complementary*, not redundant. Episodic answers "what's the conversation about?"; semantic answers "what specific facts do I know?"; procedural answers "what have I already tried?". A coding question late in a session needs all three: the episodic summary (we're debugging a deadlock), the semantic facts (it's a Python asyncio project named Halibut), and the procedural log (I already tried adding a lock and it didn't help). Drop any tier and the agent is missing a piece of the state it needs.

A table to fix the distinctions, because the tiers are easy to blur:

| Tier | Question | Storage | Access pattern | Eviction |
|------|----------|---------|----------------|----------|
| **Episodic** | what have we said? | rolling summary (text) | read the whole bounded summary | summarize old turns away |
| **Semantic** | what do we know? | facts in pgvector (+ KG) | query by relevance (top-k) | rarely evict (facts are durable) |
| **Procedural** | what have we done? | append-only action log | read a recent window | drop old actions past the window |

Notice that the three differ on *every* axis — what they store, how you read them, how they evict. That's why they're separate tiers and not one big memory blob: a rolling summary can't be queried by relevance, a fact base shouldn't be summarized away, an action log needs append-only ordering. Trying to serve all three needs with one mechanism (just stuff everything in a vector store, or just summarize everything) fails on at least one — the fact gets summarized away, or the narrative gets shredded into unconnected vector chunks. The capstone uses three tiers because the three needs are genuinely different, and the table is the proof.

> **The mental model:** episodic is the *narrative* (compressed), semantic is the *knowledge base* (factual), procedural is the *journal* (actions). Three questions, three tiers, three different storage-and-eviction stories.

It's worth distinguishing memory from retrieval (Lecture 1), because they're both "fetch relevant context" and easy to conflate. **Retrieval pulls from the corpus** — the 10 GB of documents, fixed and external, the same for every user. **Memory pulls from the conversation** — what *this* user said, what *this* agent knows and did, specific to *this* session. A research assistant answering "how did Q3 revenue compare to forecast?" retrieves the *documents* (corpus) and reads *memory* (the user prefers tables, we already looked at the actuals). Both feed the prompt; they're different sources. The semantic memory tier *uses* a vector store like retrieval does, which is the source of the confusion — but it stores *conversation facts*, not *corpus chunks*. Keeping them distinct in the architecture matters: retrieval is rebuilt when the corpus changes; memory is per-session and ephemeral (or per-user and persistent). The capstone has both, wired to the same agents but answering different questions — "what's in the documents?" vs "what do I know about this user and task?"

---

## 2. The memory interfaces (the contracts)

The capstone discipline — same as retrieval (Lecture 1 §5) — is that the agents see *interfaces*, not implementations. The memory layer exposes read and write per tier, and the agents call them without knowing whether semantic memory is pgvector or a graph or both:

```python
class Memory:
    def write_episodic(self, turn: Turn) -> None:
        """Add a turn; the tier folds it into the rolling summary as needed."""

    def read_episodic(self) -> str:
        """The current rolling summary of the conversation."""

    def write_semantic(self, fact: Fact) -> None:
        """Store a durable fact (subject, relation, object) + its embedding."""

    def read_semantic(self, query: str, k: int = 5) -> list[Fact]:
        """Retrieve the k facts most relevant to the query (vector + KG)."""

    def write_procedural(self, action: Action) -> None:
        """Log a tool call / action and its result."""

    def read_procedural(self, k: int = 10) -> list[Action]:
        """The recent action history."""
```

Notice the asymmetry in the read interfaces, and it's deliberate: **episodic reads return the *whole* current summary** (it's bounded, that's the point), **semantic reads are *queried* by relevance** (the fact base is large; you retrieve what's relevant), and **procedural reads return the *recent* history** (a window of actions). Three tiers, three access patterns, matching the three questions. An agent assembling its context for a turn calls all three: the episodic summary for the narrative, a semantic query for relevant facts, the recent procedural log for what it's tried — and stitches them into the prompt.

The implementations behind these interfaces are week-11 work, scaled:

- **Episodic** — a summarizer (a cheap model, e.g. Haiku) that folds new turns into a rolling summary, with hierarchical summarization when the summary itself grows long. The eviction policy (what to summarize away) is the context-budget knob.
- **Semantic** — facts as rows in pgvector (the fact text + its embedding) plus optionally a KG schema (subject/relation/object) for graph traversal; `read_semantic` does ANN search (and optional graph walk) for relevance.
- **Procedural** — an append-only log (a table, a list) of `(action, args, result, timestamp)`; `read_procedural` returns the recent window.

> **The discipline:** the agents call `memory.read_semantic(query)`, not `pgvector.search(...)`. The interface hides the tier's mechanism, exactly so the mechanism can evolve (add the KG, swap the summarizer) without touching the agents. Same decoupling lesson as the retrieval interface — the contract is stable, the implementation is free.

### 2b. The episodic tier: rolling summary with a budget

The episodic tier's job is to keep the conversation's *meaning* in a bounded number of tokens, so the active context doesn't grow without limit. The mechanism is a rolling summary maintained by a cheap model:

```python
class EpisodicMemory:
    def __init__(self, summarizer, max_summary_tokens: int = 800):
        self.summarizer = summarizer          # a cheap model (Haiku)
        self.summary = ""
        self.recent_turns: list[Turn] = []     # un-summarized recent turns
        self.max_tokens = max_summary_tokens

    def write_episodic(self, turn: Turn) -> None:
        self.recent_turns.append(turn)
        # when recent turns accumulate, fold them into the rolling summary
        if self._token_count(self.recent_turns) > 1000:
            self.summary = self.summarizer.summarize(
                f"Previous summary:\n{self.summary}\n\n"
                f"New turns:\n{self._format(self.recent_turns)}\n\n"
                "Produce an updated summary capturing goals, decisions, and state."
            )
            self.recent_turns = []             # folded in; drop the raw turns

    def read_episodic(self) -> str:
        # the bounded summary + a few most-recent raw turns for immediacy
        return self.summary + "\n" + self._format(self.recent_turns[-3:])
```

The design points worth naming: the summary is **bounded** (`max_summary_tokens`) — that's the budget; recent turns are kept **raw** for a few turns (immediacy matters — the last exchange shouldn't be lossy-summarized); and the summarization is done by a **cheap model** because it runs frequently (every ~1000 tokens of new conversation) and using the frontier model for it would be a cost leak (week 21). When the summary itself grows toward its bound, **hierarchical summarization** kicks in — summarize the summary — so even a very long conversation stays bounded. The eviction question ("what does the summary keep vs drop?") is the context-budget knob: a summary that drops the user's project name fails the regression test (§3), so the summarizer prompt has to be tuned to preserve load-bearing facts — or, better, those facts live in the *semantic* tier where they're stored explicitly, not at the mercy of a summary.

### 2c. The semantic tier: facts as vectors (and optionally a graph)

The semantic tier stores durable facts and retrieves them by relevance. The minimal implementation is facts-as-vectors in pgvector:

```python
class SemanticMemory:
    def write_semantic(self, fact: Fact) -> None:
        # fact: a (subject, relation, object) triple + its natural-language form
        vec = self.embed(fact.text)            # "the user's project is Halibut"
        self.store.insert(fact.id, fact.text, fact.subject, fact.relation,
                          fact.object, vec)

    def read_semantic(self, query: str, k: int = 5) -> list[Fact]:
        # vector relevance: which stored facts are most relevant to this query?
        q_vec = self.embed(query)
        return self.store.knn(q_vec, k=k)
```

When the user says "my project is Halibut," `write_semantic` stores the fact with its embedding. When a later turn asks "what's my project?", `read_semantic` embeds the query and finds the nearest fact — "the user's project is Halibut" — by cosine similarity. The fact survives because it's stored *as a fact*, independent of the conversation's length or the episodic summary's compression.

The optional graph layer (the "+ KG" in the syllabus) adds *structured* facts as (subject, relation, object) rows you can traverse: "Halibut --uses--> Python", "Halibut --deadline--> March 15". This enables multi-hop queries the pure vector tier can't — "what language is my project in?" can traverse `user --project--> Halibut --uses--> Python` even if no single stored sentence says "the user's project uses Python." The capstone's preferred path (per the syllabus) is the self-built pgvector + KG schema: vector relevance for fuzzy recall, graph traversal for structured multi-hop. You don't *have* to build the graph layer for Sprint A's regression test (vector facts pass it), but the interface (`read_semantic`) is designed so you can add it later without touching the agents — the same decoupling discipline.

The critical extraction question: **when does a fact get written?** The user says many things; not all are durable facts. A `write_semantic` triggered too eagerly fills the tier with noise; too rarely and it misses the project name. The usual answer is a cheap extraction pass (a model call, or a heuristic) that pulls salient facts from each user turn — "the user stated a preference / a name / a constraint / a deadline" — and writes those. Getting the extraction right is what makes the regression test pass: if "Halibut" was never extracted to a fact when the user said it, no `read_semantic` can recall it. The extraction is the tier's quiet failure point.

---

## 3. The memory regression test: does it remember at turn 38?

A memory system that works on turn 2 and forgets by turn 38 is worse than no memory — it's *unreliable* memory, which the agent (and the user) can't trust. So the capstone's memory work is gated on a **regression test** straight from week 11: plant a load-bearing fact early in a long conversation, and assert the agent can still recall it many turns later.

```python
def test_memory_survives_long_conversation(agent):
    agent.chat("My project is called Halibut.")             # turn 1: plant the fact
    for i in range(36):                                      # turns 2-37: filler
        agent.chat(f"Tell me about topic {i}.")
    response = agent.chat("What's my project called?")      # turn 38: recall test
    assert "Halibut" in response                             # the fact survived
```

The test is brutal precisely because it's the failure mode that matters: by turn 38, the raw transcript of turns 1–37 is long gone from the active context (that's the whole reason for the memory tiers). If the fact "Halibut" survived, it's because the *semantic* tier stored it as a durable fact and `read_semantic("what's my project called?")` surfaced it — *not* because it was lucky enough to still be in the context window. The test proves the memory architecture, not the model's context length.

This is also a diagnostic for *which tier* failed when it fails. If the fact is gone at turn 38, walk the tiers: did `write_semantic` capture "Halibut" as a fact when the user said it (or did it only go into the episodic summary, which compresses and may have dropped it)? Did `read_semantic` retrieve it for the recall query (or did the relevance match miss)? The test failing tells you the memory architecture has a hole; *which* tier you check tells you where. A capstone that passes this test has memory you can trust across a real session; one that doesn't has a demo that works for three turns and falls apart on turn four.

> **The discipline:** a load-bearing fact must survive to a late turn, proven by a regression test, not hoped for. The test is the difference between "memory works" (a claim) and "the agent recalls the project name at turn 38" (a measured fact). Same measurement discipline as every C23 week — the memory tiers are a *measured* component, not a vibe.

### 3b. Diagnosing a regression-test failure

When the turn-38 test fails — the agent says "I don't know your project name" — you have a memory architecture bug, and the tiers tell you where to look. Walk the path the fact should have taken:

```
The fact "Halibut" should have:
  1. been EXTRACTED to a semantic fact when the user said it (turn 1)
     -> check: did write_semantic get called with a Halibut fact?
        If NO: the extraction pass missed it. Fix the extractor (§2c).
  2. been STORED in the semantic tier with a usable embedding
     -> check: is the fact in pgvector? Does its embedding look right?
        If NO: the write failed. Check the store path.
  3. been RETRIEVED by read_semantic("what's my project called?")
     -> check: does the query embedding match the fact's embedding above the
        retrieval cutoff?
        If NO: a relevance miss. The query "what's my project" and the fact "the
        user's project is Halibut" should be near; if they're not, check the
        embedding model or add the fact's subject ("project") to its text.
  4. been INCLUDED in the turn-38 context the model saw
     -> check: did the supervisor actually call read_semantic and put the result
        in the prompt?
        If NO: the supervisor isn't assembling semantic facts into context. Wire it.
```

Four checkpoints, four different bugs, four different fixes. The most common failure is checkpoint 1 (the fact was never extracted — it only went into the episodic summary, which compressed it away) or checkpoint 3 (the relevance match missed because the fact's stored text doesn't share enough with the recall query). This is why the test is *diagnostic*, not just pass/fail: a failure points you at a specific tier and a specific stage within it. A capstone team that can walk this path when memory fails is a team that *understands* its memory architecture; one that just shrugs at the failure has memory it doesn't understand, which is memory it can't trust. Run the regression test early and often during Sprint A — the longer you wait, the more turns of memory machinery you have to debug at once.

### 3c. The procedural tier: the agent's journal

The procedural tier is the simplest of the three but easy to skip, and skipping it costs the agent self-awareness. It's an append-only log of actions and their results:

```python
class ProceduralMemory:
    def __init__(self):
        self.log: list[Action] = []

    def write_procedural(self, action: Action) -> None:
        # action: (agent, task, args, result, timestamp)
        self.log.append(action)

    def read_procedural(self, k: int = 10) -> list[Action]:
        return self.log[-k:]                    # the recent action window
```

Why it matters: without procedural memory, an agent re-does work it already did. The supervisor delegates "search the corpus for Q3 revenue" to the retrieval agent; if the result isn't logged, a later step might delegate the *same* search again — wasted tokens, wasted latency, and a confused trajectory. With procedural memory, the supervisor can see "I already retrieved Q3 revenue, here are the results" and build on them instead of repeating. It also enables the *critique* agent (week 23) to reason about the trajectory: "the code agent tried approach X and it failed; suggest Y." The procedural log is the agent's record of its own behavior, and it's what lets a multi-step agent be *coherent* across steps rather than amnesiac between them. In the capstone, every supervisor delegation writes to procedural memory (Lecture 2 §5's `run` loop does exactly this), which is what makes the action history complete.

The three tiers, assembled into a turn's context, are the agent's working state: the episodic summary (the narrative so far), the relevant semantic facts (what we know), and the recent procedural log (what we've done). An agent with all three is stateful and coherent; an agent missing one is forgetful in a specific, diagnosable way — which is exactly what the regression test and the §3b diagnosis are built to catch.

---

### 3d. Run the test continuously, not at the end

A practical note on *when* to run the regression test, because the failure mode is saving it for the end. The turn-38 test exercises every tier — extraction, storage, retrieval, context assembly — so a failure could be in any of them. If you build all three tiers, wire them to the supervisor, and *then* run the test for the first time and it fails, you have to debug four stages at once across three tiers, with no idea which is broken. That's a miserable afternoon.

Instead, run the test (or a smaller version of it) *as you build*:

- After `write_semantic` works: a 3-turn version (plant fact, one filler, recall) should pass — proving extraction + storage + retrieval in isolation, before the episodic tier exists to confuse things.
- After the episodic summarizer works: re-run with more filler turns, confirming the fact survives the summarization (it should, because it's in the *semantic* tier, not the summary).
- After the supervisor assembles context: the full turn-38 test, confirming the assembled context actually includes the recalled fact.

Each step adds one stage to the test's coverage, so a failure points at the stage you just added. This is the same incremental-testing discipline you'd apply to any system — test the unit, then the integration, then the whole — applied to memory. A capstone team that runs the regression test from day one of memory work debugs each tier as they build it; one that runs it once at the end debugs the whole memory architecture under deadline. The test is cheap to run; run it constantly.

## 4. Context budgeting: spending the most expensive cache

Week 11's mantra — "context is the most expensive cache on the planet; spend it like one" — is the memory layer's governing principle, and it's worth restating as a capstone constraint. Every turn, the agent assembles a prompt from: the system prompt, the episodic summary, the retrieved chunks (Lecture 1), the relevant semantic facts, the recent procedural log, and the user's message. That all has to fit in a token budget, and more context is not better — it's more expensive *and* it dilutes the model's attention (lost-in-the-middle).

So the memory layer doesn't just *store*; it *budgets*. The episodic tier compresses to a bounded summary. The semantic read returns the *k most relevant* facts, not all facts. The procedural read returns a *recent window*, not the whole log. Each tier's read interface is implicitly a budget decision — how much of this tier's content earns its place in this turn's context. The default `k`s in the interface (§2) are the budget; tuning them (more semantic facts vs more retrieved chunks vs a longer procedural window) is a context-allocation decision the architecture doc should name. **The memory tiers are how you fit a long, stateful conversation into a bounded, expensive context** — that's their job, and the budgeting is half of it.

This connects forward to the cost work (week 21): the context you assemble each turn *is* the input-token cost of that turn, and prompt caching (the stable system prompt + frozen instruction prefix) applies here. A well-budgeted memory layer is also a cheaper one — fewer tokens per turn — which is another reason the budget knobs matter. The architecture doc should note where prompt caching applies (the stable prefix) and where the volatile content (the per-turn memory assembly) goes (after the cached prefix), exactly the week-21 discipline.

---

## 5. The supervisor draft and the sub-agent contracts

The agents themselves are week-23 work, but Sprint A *drafts their contracts*, because the foundation has to be shaped to receive them. The supervisor is a LangGraph state machine (week 13) that delegates to sub-agents; you draft its interface and the sub-agents' signatures now.

The supervisor's job is *orchestration* — it decides which sub-agent handles what, sequences their work, and assembles the final answer:

```python
class Supervisor:
    def run(self, user_query: str) -> str:
        # 1. assemble context from memory (Lecture 2 §2)
        context = self.assemble_context(user_query)
        # 2. plan: which sub-agents, in what order? (LangGraph nodes)
        plan = self.plan(user_query, context)
        # 3. delegate to sub-agents, each via a clean contract
        for step in plan:
            result = self.delegate(step.agent, step.task, context)
            self.memory.write_procedural(Action(step.agent, step.task, result))
            context = self.update_context(context, result)
        # 4. assemble and return the final answer
        return self.synthesize(context)
```

The sub-agent contracts — the interfaces week 23 implements — are the key Sprint-A design artifact:

```python
# Each sub-agent is a function from (task, context) to a result. The supervisor
# doesn't know HOW each agent works, only this contract. (Drafted now, built wk23.)
def retrieval_agent(task: str, context: Context) -> RetrievalResult:
    """Calls retrieve() (Lecture 1), returns ranked chunks + a synthesis."""

def code_agent(task: str, context: Context) -> CodeResult:
    """Calls MCP tools (fs, python sandbox), returns code + execution result."""

def writing_agent(task: str, context: Context) -> WritingResult:
    """Reads memory + retrieved chunks, returns drafted prose."""

def critique_agent(task: str, context: Context) -> CritiqueResult:
    """Scores another agent's output, returns a pass/fail + improvement notes."""
```

Why draft these now, in Sprint A, when they're built in Sprint B? Because **the foundation has to fit them.** The retrieval agent calls `retrieve()` — so `retrieve()`'s signature (Lecture 1 §5) must match what the agent needs (it does: `query -> ranked_chunks`). The writing agent reads memory — so the memory read interface (§2) must give it what it needs (it does: episodic summary + semantic facts). The procedural memory logs every delegation — so `write_procedural` must accept an agent's action (it does). **Drafting the contracts now is how you verify the foundation's interfaces are the right shape** before week 23 commits to them. If drafting the writing-agent contract reveals it needs a memory read the interface doesn't offer, you found that *now*, cheaply, instead of mid-Sprint-B.

This is the deepest reason the document comes first (Lecture 1 §1): the act of writing the agent contracts *tests* the foundation interfaces. A foundation whose interfaces don't cleanly support the drafted agent contracts is a foundation you'd be rebuilding in week 23. Sprint A's success criterion is exactly this fit: the retrieval and memory interfaces support the drafted agent contracts without strain.

> **The discipline:** draft the consumer (the agent contracts) to validate the producer (the foundation interfaces). If the agents you'll build next week can't cleanly call the interfaces you built this week, the interfaces are wrong — and finding that in Sprint A is cheap, finding it in Sprint B is a re-architecture.

---

### 5b. Why the supervisor is a graph, not a loop

A note on *why* the supervisor is a LangGraph state machine (week 13) and not the simple ReAct loop you wrote in week 5. The capstone has four sub-agents, conditional delegation (retrieve *then* maybe code *then* write *then* critique, with the critique possibly sending work back), and it must survive a process kill mid-task (the persistence requirement). That's exactly the threshold week 13 named: "when the agent gets a fourth tool, you graduate from a loop to a graph."

A loop with four `if` branches for which agent to call next, plus retry logic, plus checkpointing, *is* a state machine — just an implicit, untestable one. LangGraph makes it explicit: nodes (plan, retrieve, code, write, critique, synthesize), conditional edges (critique → back to write if it fails, → synthesize if it passes), and a checkpointer (SQLite) so the supervisor resumes from step 7 after a crash. The architecture doc should show the supervisor as this graph — the Mermaid diagram's supervisor box expands into a sub-graph of nodes and edges — because the *control flow* of a multi-agent system is as much a design artifact as its data flow. You draft the graph's shape now (the nodes and the delegation edges); week 23 implements the nodes against the sub-agent contracts.

The persistence point connects to week 24: a supervisor with a SQLite checkpointer survives a process kill (the chaos drill kills a vLLM replica, but a robust supervisor should also survive its own process dying). Drafting the supervisor as a checkpointed graph now means the resumability is designed in, not bolted on under deadline. The contract you draft this week is "the supervisor is a LangGraph state machine over these nodes, with these delegation edges, checkpointed to SQLite" — and that contract shapes how the foundation's interfaces are called (each node calls `retrieve()` or a memory read or a sub-agent), which is the validation Lecture 2 §5 described.

## 6. Sequencing the rest of the capstone

The architecture doc's build-sequence section (Lecture 1 §2) is where Sprint A's place in the whole becomes a plan. State it explicitly:

- **Sprint A (this week) — done:** the architecture doc + diagram; the 10 GB corpus ingested and hybrid-indexed; `retrieve()` measured on the gold set; the three memory tiers wired and passing the regression test; the supervisor + sub-agent contracts drafted.
- **Sprint B (week 23) — next:** implement the supervisor (LangGraph) and the four sub-agents against the drafted contracts; wire the MCP tool surface (fs, web, calc, custom); deploy the vLLM cluster behind LiteLLM with cost-routing (weeks 19/21); stand up the Ragas + LLM-judge eval suite; flow OTel traces to Langfuse + Phoenix.
- **Week 24 — last:** the chaos drill (kill a vLLM replica, prompt-inject a tool, corrupt the index) and the postmortem.

The dependencies are the point: Sprint B's supervisor *depends on* Sprint A's retrieval and memory interfaces; the chaos drill *depends on* the deployed cluster (Sprint B) and the index recovery numbers (Sprint A's ingest measurements). A realistic plan names these dependencies so nobody starts the supervisor before the interfaces are stable or the chaos drill before there's a system to break. **Sprint A is the foundation the whole rest of the capstone is sequenced onto** — which is why getting its interfaces right is the week's entire job.

A closing note on what "done" means for Sprint A, because the temptation is to over-build. Sprint A is done when: the architecture doc and diagram exist and match the code; the corpus is ingested and hybrid-indexed with a measured Recall@5 on the gold set; the three memory tiers are wired and pass the turn-38 regression test; and the supervisor + sub-agent contracts are drafted (not implemented). That's it — the *foundation*, not the system. You do *not* build the agents this week (Sprint B), do *not* deploy the cluster (Sprint B), do *not* run Ragas (Sprint B). The discipline is to land the foundation cleanly and *stop*, rather than half-building the agents and leaving the foundation's interfaces unvalidated. A clean foundation with drafted contracts is a better Sprint-A deliverable than a half-built system with three agents that don't quite talk to each other — because the clean foundation makes Sprint B fast, and the half-built mess makes Sprint B a cleanup job. Land the foundation, draft the contracts, write the document, and trust that a good foundation is the fastest path to the finished capstone.

---

## 7. Recap

You should now be able to:

- Explain **why three memory tiers** — episodic (the compressed narrative), semantic (the fact base), procedural (the action journal) — answer three complementary questions, and why dropping any tier leaves the agent missing state.
- Wire the **memory interfaces** (read/write per tier) so the agents call `memory.read_semantic(query)`, not the store — decoupling the tier's mechanism from its consumers, the same way the retrieval interface decouples.
- Write the **memory regression test** (a load-bearing fact survives to turn 38) and use its failure to diagnose *which* tier has the hole — proving the memory architecture rather than the model's context length.
- Apply **context budgeting** — each tier's read is a budget decision; the memory layer fits a long stateful conversation into a bounded, expensive context, which is also cheaper (prompt-caching-aligned).
- Draft the **supervisor and sub-agent contracts** (retrieval, code, writing, critique) as interfaces, and understand that drafting the consumers *validates* the foundation's producer interfaces before week 23 commits to them.
- **Sequence the capstone** — Sprint A done, Sprint B next, chaos drill last — with the dependencies named, so the foundation's interfaces are the stable base the rest is built on.

Carry these one-line takeaways into the exercises:

- Three tiers, three questions: episodic (what we said), semantic (what we know), procedural (what we did).
- Agents call memory interfaces (`read_semantic(query)`), never the store — the contract hides the mechanism.
- Episodic is a bounded rolling summary by a cheap model; semantic is facts-as-vectors (optionally + KG); procedural is an append-only journal.
- The regression test (a fact survives to turn 38) proves the architecture, not the model's context length.
- A regression failure is diagnosable: walk extract → store → retrieve → include-in-context to find the broken tier.
- The semantic tier's quiet failure point is *extraction* — a fact never written can never be recalled.
- Context budgeting: each tier's read is a budget decision; the tiers fit a long conversation into a bounded, expensive context.
- Draft the agent contracts now to validate the foundation interfaces; the supervisor is a checkpointed LangGraph state machine.

Next week is **Capstone Sprint B** — you implement the supervisor and the sub-agents against the contracts you drafted this week, on top of the retrieval and memory foundation you landed. Push your `crunchcap` foundation and your architecture document; Sprint B is assembly if the interfaces are right and re-architecture if they're not.

---

## References

- *The capstone specification and rubric*: `C23-CRUNCH-AGENTS/SYLLABUS.md` (Capstone section) and `capstone/RUBRIC.md`
- *Lost in the Middle* — Liu et al., 2023 (why context budgeting and memory summarization matter): <https://arxiv.org/abs/2307.03172>
- *Letta (formerly MemGPT)* — tiered memory with eviction: <https://github.com/letta-ai/letta>
- *LangGraph* — the supervisor's state-machine framework (week 13): <https://langchain-ai.github.io/langgraph/>
- *pgvector* — the semantic-memory store: <https://github.com/pgvector/pgvector>
- *The capstone default architecture diagram* (the supervisor + four agents + hybrid retrieval + memory + serving): `C23-CRUNCH-AGENTS/SYLLABUS.md` (Capstone → Architecture)
