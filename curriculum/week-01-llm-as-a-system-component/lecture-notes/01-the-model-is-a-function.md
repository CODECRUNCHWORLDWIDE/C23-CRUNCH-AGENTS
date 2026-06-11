# Lecture 1 — The Model Is a Function: Tokens In, A Distribution Out

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can describe an LLM as a function from token IDs to next-token logits, name the five stages of the external interface, explain the decoder-only transformer at a systems level without any attention math, and distinguish prefill from decode well enough to predict which knob moves which latency number.

If you remember one sentence from this entire week, remember this one:

> **A large language model is a deterministic function from a sequence of token IDs to a vector of logits over the vocabulary — one score for the single next token. Everything you experience as conversation, memory, creativity, or randomness is built by the code wrapped around that function, not by the function itself.**

This is not a simplification you will outgrow. It is the load-bearing abstraction of the whole field. Senior AI engineers reach for it constantly: when a model "hallucinates," they ask which stage of the wrapper failed; when output is "too random," they look at the sampler, not the weights; when latency spikes, they ask whether it is prefill or decode. The people who never form this model spend their careers surprised. You are going to form it this week.

---

## 1. The external contract

Forget, for a moment, everything about transformers, attention, and training. Treat the model as a black box with a strict interface. Here is the contract, stated precisely:

- **Input:** a sequence of integer **token IDs**, `[t₀, t₁, …, tₙ₋₁]`. Each ID is an index into a fixed **vocabulary** of size `V` (tens of thousands to ~256k entries, depending on the family).
- **Output:** a vector of `V` real numbers called **logits** — one unnormalized score per vocabulary entry — representing the model's scoring of what the *next* token, `tₙ`, should be.

That's it. That is the whole function. Written as a type signature:

```
forward : List[int]  ->  Vector[float, V]
```

A few consequences fall out of this contract immediately, and they matter:

1. **The model outputs a distribution, not a token.** The logits are scores. To get an actual next token you must *choose* one — that is sampling, a separate stage you own. The model never picks. Your code picks.
2. **The model is, at this level, deterministic.** Same token IDs in, same logits out (modulo floating-point nondeterminism from parallel hardware, which is real but small). The randomness you see in chat output is injected by your sampler, not by the weights.
3. **The model has no memory between calls.** It is a pure function of its input sequence. "Conversation history" exists only because your code re-sends the whole history as input every turn. The API is stateless; the statefulness is yours.
4. **The model conditions only on tokens.** Roles ("system," "user," "assistant"), system prompts, tool definitions — all of it is encoded into token IDs by a **chat template** *before* the model sees anything. The model sees one flat sequence of integers. There is no privileged "system channel" at the function level; there is only position in the token stream.

Internalize point 4 especially. When week 17 teaches prompt injection, the entire attack works *because* the model sees a flat token stream with no hard boundary between "trusted instructions" and "untrusted retrieved text." The flatness is not a bug in a vendor's implementation; it is a property of the function.

---

## 2. The five stages of the interface

The bare function is `List[int] -> Vector[float, V]`. But you type text and you read text, so there is machinery on both ends. The full pipeline — from your string to the next string — is five stages. Learn the names; we will assign blame to specific stages all course long.

```
   text                                                      text
    │                                                          ▲
    ▼                                                          │
┌────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐
│ 1 Tokenizer│──▶│ 2 Context     │──▶│ 3 Forward    │──▶│ 4 Sample │──▶│ 5 Detokenizer│
│ text→IDs   │   │   window      │   │   pass       │   │ logits→  │   │   IDs→text   │
│            │   │ (bounded buf) │   │ IDs→logits   │   │  one ID  │   │              │
└────────────┘   └───────────────┘   └──────────────┘   └──────────┘   └──────────────┘
                                                              │
                                                  append chosen ID, loop ▲
```

### Stage 1 — Tokenizer (text → IDs)

The tokenizer is a deterministic, reversible map between text and token IDs. It is **not** a word splitter and **not** a character splitter. Modern tokenizers use **sub-word** units learned by an algorithm like BPE (byte-pair encoding) or SentencePiece. A common word like `" the"` is one token; a rare word like `" antidisestablishmentarianism"` is several; an emoji or a CJK character may be one or several tokens depending on the tokenizer.

The critical systems facts about Stage 1:

- **Token count is what you pay for, not character count.** Vendor pricing is per-token. Your cost model is wrong if it counts characters or words. (Week 2 makes this precise; this week you just respect it.)
- **Tokenizers differ across families.** Llama's tokenizer, Qwen's tokenizer, and the tokenizer behind a frontier API are different maps. The same English sentence is a *different number of tokens* on each. This is why you must never estimate a hosted model's token count with a different model's tokenizer — `tiktoken` (OpenAI's) under-counts Anthropic and Llama tokens, badly on code and non-English text. For a hosted model, use its own counting endpoint:

  ```python
  import anthropic

  client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
  resp = client.messages.count_tokens(
      model="claude-opus-4-8",
      messages=[{"role": "user", "content": "Summarize the incident report."}],
  )
  print(resp.input_tokens)  # the real number, from the real tokenizer
  ```

- **The tokenizer is where "weird" text behavior comes from.** A model that "can't spell a word backwards" or "miscounts the letters in strawberry" is not stupid; it never saw individual letters — it saw sub-word tokens. The tokenizer is the lens, and the lens has a grain.

### Stage 2 — Context window (the bounded buffer)

The context window is the maximum number of token IDs the model can condition on in a single forward pass — prompt tokens plus already-generated tokens, counted together. In 2026 this ranges from ~8k on small local models to **1M tokens** on frontier models like `claude-opus-4-8`.

Three things to hold:

- **It is a hard ceiling, not a soft preference.** Exceed it and the request errors (or, worse, silently truncates, depending on the stack). Your context-assembly code is responsible for staying under it — this is the entire reason weeks 8–11 exist.
- **Long context is expensive in two ways.** The obvious one: more tokens, more money. The subtle one: attention cost grows with sequence length (Stage 3), so a 100k-token prompt costs more *per token* to process than a 1k-token prompt, and the first token comes back slower. "It fits in the window" and "it is cheap" are different claims.
- **Position in the window matters.** Models attend unevenly across a long context — the "lost in the middle" effect (week 11). For now: filling the window is not free, and it is not uniform.

### Stage 3 — Forward pass (IDs → logits)

This is the model. It is the expensive stage — the one the GPU spends its cycles on, the one you pay the vendor for, the one §3–4 of this lecture unpacks. Input: the token IDs in the context window. Output: the logit vector for the next token. We will open this box in §3, but at the interface level it is one call that turns `n` token IDs into `V` logits.

### Stage 4 — Sampling (logits → one ID)

The forward pass hands you `V` logits. You must turn that into exactly one chosen token ID. That choice is **sampling**, and it is *your* stage — owned by your code or your inference server's config, not by the weights.

The simplest sampler is **greedy / argmax**: pick the token with the highest logit. Deterministic, repetitive, sometimes bland. The next simplest is **temperature sampling**: divide the logits by a temperature `T`, convert to probabilities with softmax, and draw randomly. Higher `T` flattens the distribution (more "random"); lower `T` sharpens it (more "confident"); `T → 0` approaches greedy. There is more — top-p, top-k, min-p, repetition penalty — and all of it is week 2's job. The point *this* week:

> **"Creativity," "randomness," and "determinism" are properties of Stage 4, not Stage 3.** The model emits the same logits; your sampler decides whether to be deterministic or surprising. When someone says "the model is too random," the fix is almost always a sampling knob, not a different model.

A note on hosted frontier models: some 2026 frontier models (including the current Anthropic Opus tier, `claude-opus-4-8`) have **removed** the user-facing `temperature` / `top_p` / `top_k` knobs from their API — sending them is an error. The sampling still happens; the vendor just manages it internally and asks you to steer behavior with the prompt instead. This is itself a systems-design statement: the vendor decided the sampler is part of the model's contract, not a customer knob. On open-weights models served by Ollama or vLLM, those knobs are yours. Knowing *who owns Stage 4* for a given deployment is part of reading the model as a component.

### Stage 5 — Detokenizer (IDs → text)

The chosen token ID is mapped back to its text fragment and appended to the output. Then — because the model is **autoregressive** — the whole loop repeats: the new token is appended to the input, Stage 3 runs again on the now-longer sequence, Stage 4 picks the next token, and so on, until the model emits a special end-of-sequence token or you hit your `max_tokens` cap. This loop is why output *streams*: each token is produced one at a time, so you can show it as it arrives.

---

## 3. Opening Stage 3: the decoder-only transformer, at a systems level

Now we open the forward-pass box — but only as far as a systems engineer needs. We are not deriving attention. We are building a mental picture detailed enough to reason about cost, latency, and the KV cache, and no more.

A modern LLM is a **decoder-only transformer**. "Decoder-only" means it has one job: given a sequence, predict the next element. (The original 2017 transformer had an encoder *and* a decoder, for translation. GPT-style models kept only the decoder half. That is the whole etymology.) Picture three parts stacked vertically:

```
token IDs ─▶ ┌─────────────┐
             │  Embedding  │   each ID → a vector (a learned point in space)
             └──────┬──────┘
                    ▼
             ┌─────────────┐
             │  Block 1    │   ┌─ attention: mix information across positions
             │  Block 2    │   │  ┌─ MLP:    transform each position's vector
             │   …         │   └──┘  (repeated N times — "depth")
             │  Block L    │
             └──────┬──────┘
                    ▼
             ┌─────────────┐
             │ Unembedding │   final vector → V logits (scores over the vocab)
             └──────┬──────┘
                    ▼
              next-token logits
```

1. **Embedding.** Each token ID is looked up as a vector — a learned point in a high-dimensional space. "Apple" and "orange" land near each other; "apple" and "Tuesday" land far apart. The model now reasons over vectors, not integers.

2. **The stack of blocks.** This is the bulk of the model. Each block does two things in sequence:
   - **Attention** mixes information *across positions*. For each token, attention lets it "look at" every earlier token and pull in relevant information. This is the part that is `O(n²)` in sequence length — every position can attend to every earlier position — and it is the reason long context is expensive. (Decoder-only models use **causal** attention: a token may attend to earlier tokens and itself, never future tokens. That causality is what makes next-token prediction well-defined.)
   - **MLP** (a small feed-forward network) transforms each position's vector independently, adding the model's "knowledge" — the learned facts and patterns baked into the weights.
   The block is repeated `L` times (`L` = the model's *depth*, e.g. 32, 80, more). Depth and width (the vector size) are the main levers of model size.

3. **Unembedding.** After the last block, each position has a final vector. The unembedding projects the *last* position's vector to `V` logits — the scores for the next token. (During generation we only need the last position's logits; during prefill the model computes them for every position, which matters in §4.)

That is the whole model at this altitude. You can now state the three facts that drive every systems decision downstream:

- **Attention is quadratic in sequence length.** Doubling the prompt roughly quadruples the attention work. This is *why* long context is expensive beyond the linear token cost, and why weeks 8–11 obsess over putting the *right* tokens in the window rather than *all* the tokens.
- **The MLP is where the "knowledge" lives.** Parametric knowledge — what the model "knows" without being told — is in the MLP weights, frozen at training time. This is why a model has a **training cutoff** (§ Lecture 2) and why retrieval (Phase II) exists: to inject knowledge the frozen weights never saw.
- **Generation is autoregressive and therefore sequential.** You cannot generate token 50 before token 49; each depends on the last. This is why output latency is fundamentally serial per request, and why throughput tricks (batching, continuous batching — week 19) work by serving *many* requests' token-50s at once, not by speeding up one request's serial chain.

You now know enough about Stage 3. We will not go deeper into attention until week 13, and even then only for the vocabulary.

---

## 4. Prefill vs decode, and why the KV cache exists

Here is the single most useful systems distinction inside the forward pass — the one that separates engineers who can read a latency chart from those who can't. Generating a response has **two phases**, and they have different performance characteristics, different bottlenecks, and different cost behavior.

### Phase A — Prefill

When a request arrives, the model must process the *entire prompt* to set up for generation. It runs the forward pass over all `n` prompt tokens at once. Two things happen:

- It computes the next-token logits (so it can emit the first output token).
- It computes and stores, for every prompt token and every layer, the attention **keys and values** — the intermediate quantities each future token will need to attend to. These get stored in the **KV cache**.

Prefill processes many tokens in parallel, so it is **compute-bound** — limited by raw GPU FLOPs. Its duration sets your **time-to-first-token (TTFT)**: the "it feels slow to start" delay. A long prompt means a long prefill means a slow first token. This is why a 100k-token context has a noticeably laggy first token even on fast hardware.

### Phase B — Decode

Once prefill is done, generation proceeds one token at a time. For each new token, the model does *not* reprocess the whole sequence — that would be catastrophically wasteful and `O(n²)` per token. Instead it reuses the **KV cache**: the new token attends to the cached keys/values of all prior tokens, and only the single new token's key/value is computed and appended.

Each decode step thus processes essentially *one* token's worth of new compute but must read the *entire* growing KV cache from memory. So decode is **memory-bandwidth-bound**, not compute-bound. Its per-step duration sets your **time-per-output-token (TPOT)**: the "how fast does it stream" rate.

```
PREFILL (once, compute-bound)          DECODE (per token, bandwidth-bound)
┌───────────────────────────┐          ┌─────┐ ┌─────┐ ┌─────┐
│ process all n prompt tokens│          │ tok │ │ tok │ │ tok │ ...
│ build full KV cache        │   ───▶   │ n   │ │ n+1 │ │ n+2 │
│ emit first output token    │          │     │ │     │ │     │
└───────────────────────────┘          └─────┘ └─────┘ └─────┘
        sets TTFT                              sets TPOT (per token)
```

Why this matters as an engineer, concretely:

- **TTFT and TPOT are different numbers and you must measure them separately.** "Latency" is not one number. A model can have fast TTFT but slow TPOT (snappy to start, slow to stream) or the reverse. Your exercise this week (`exercise-03`) measures both against a local model and makes you explain the gap.
- **Streaming feels fast because of TPOT, not TTFT.** When you stream output token-by-token, the user starts reading after one TTFT and then consumes at the TPOT rate. Streaming doesn't make the total faster; it makes the *perceived* latency the TTFT, not the whole response. This is why every chat UI streams.
- **The KV cache is why decode is fast and why long context is expensive in memory.** The cache grows with sequence length; on a GPU it competes with the weights for VRAM. This is the central object of week 19 (PagedAttention exists to manage exactly this). For now: the KV cache is the reason the second token is much faster than the first.
- **Prompt caching (the vendor feature) is a TTFT optimization.** When a vendor lets you cache a long shared prefix, what they are caching is the *prefill KV cache* for that prefix. Reuse it and you skip re-prefilling the shared part — lower TTFT, lower cost. You will measure this in week 21; recognize now that it is the same KV cache, persisted across requests.

You can see these numbers directly on a local model. Ollama's API returns them:

```python
import httpx, json

resp = httpx.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen2.5:7b", "prompt": "Explain TTFT vs TPOT in one sentence.",
          "stream": False},
    timeout=120.0,
).json()

# Durations are in nanoseconds. prompt_eval = prefill; eval = decode.
prefill_s = resp["prompt_eval_duration"] / 1e9
decode_s = resp["eval_duration"] / 1e9
out_tokens = resp["eval_count"]
print(f"prefill (prefill phase): {prefill_s:.3f}s for {resp['prompt_eval_count']} prompt tokens")
print(f"decode  (decode phase): {decode_s:.3f}s for {out_tokens} output tokens")
print(f"TPOT ~= {decode_s / out_tokens * 1000:.1f} ms/token")
```

`prompt_eval_duration` is prefill; `eval_duration` is decode. The first divided by prompt tokens is roughly your prefill rate; the second divided by output tokens is your TPOT. Run it twice — once with a short prompt, once with a 2,000-word prompt — and watch prefill (and thus TTFT) balloon while TPOT stays roughly flat. That single experiment is the lecture.

---

## 5. Putting it together: where production bugs live

Now the payoff. With the function-and-five-stages model, you can localize almost any LLM behavior to a stage. Here is the table senior engineers carry in their heads:

| Symptom | Most likely stage | Why |
|---|---|---|
| "It costs more than I expected." | 1 (tokenizer) + 2 (context) | You counted words, not tokens; or you stuffed the window. |
| "The first token takes forever." | Prefill (3) | Long prompt → long prefill → high TTFT. Cache the prefix or shorten it. |
| "It streams slowly." | Decode (3) | Memory-bandwidth bound; bigger model or contended GPU raises TPOT. |
| "Output is too random / inconsistent." | 4 (sampling) | Temperature/top-p too high, or you expected determinism the sampler doesn't give. |
| "Output is too repetitive / bland." | 4 (sampling) | Greedy or low temperature; or a repetition issue. |
| "It doesn't know about last week's event." | 3 (frozen MLP weights) | Training cutoff. Inject via context (RAG), not by re-prompting harder. |
| "It can't count letters / spell backwards." | 1 (tokenizer) | It never saw letters — only sub-word tokens. |
| "It ignored my system prompt and followed the document." | flat token stream (the contract) | No hard boundary between trusted and untrusted tokens. (Week 17.) |
| "It forgot what I said three turns ago." | your wrapper, not the model | The model is stateless; your history-assembly dropped or truncated it. |

Read that table again at the end of the course. Every phase you study is, in effect, hardening one row of it. Phase II (retrieval) is the "doesn't know about last week" row. Phase III (agents + safety) is the "flat token stream" and "forgot what I said" rows. Phase IV (serving + cost) is the prefill/decode/cost rows. The function-and-five-stages model is the spine the whole course hangs on.

---

## 6. A first uniform call to both worlds

To make the abstraction concrete, here is the same logical operation — "complete this prompt" — against a hosted frontier model and a local open-weights model. Notice that the *interface you care about* is identical (`prompt in, text + token counts + latency out`); only the transport differs. Building this uniform wrapper is exactly the mini-project.

```python
import time
import anthropic
import httpx

PROMPT = "In one sentence, what is an LLM from a systems perspective?"


def call_anthropic(prompt: str) -> dict:
    """Hosted frontier path. Returns text + token usage + wall-clock latency."""
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env
    t0 = time.perf_counter()
    msg = client.messages.create(
        model="claude-haiku-4-5",   # fast/cheap tier for a one-liner
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - t0
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return {
        "text": text,
        "tokens_in": msg.usage.input_tokens,
        "tokens_out": msg.usage.output_tokens,
        "latency_s": elapsed,
    }


def call_ollama(prompt: str, model: str = "qwen2.5:7b") -> dict:
    """Local open-weights path via the Ollama HTTP API."""
    t0 = time.perf_counter()
    r = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120.0,
    ).json()
    elapsed = time.perf_counter() - t0
    return {
        "text": r["response"],
        "tokens_in": r["prompt_eval_count"],
        "tokens_out": r["eval_count"],
        "latency_s": elapsed,
    }


if __name__ == "__main__":
    for name, fn in [("anthropic", call_anthropic), ("ollama", call_ollama)]:
        try:
            out = fn(PROMPT)
            print(f"\n[{name}] {out['latency_s']:.2f}s  "
                  f"in={out['tokens_in']} out={out['tokens_out']}")
            print(f"  {out['text'].strip()[:200]}")
        except Exception as e:  # network down, no key, no Ollama — report, don't crash
            print(f"\n[{name}] unavailable: {e}")
```

Run it. You will see two different latencies, two different token counts (different tokenizers!), and two different text styles — but one interface. That uniformity is the whole point. The course is the engineering, not the import: when the model rotates next cohort, you change a string, not your architecture.

---

## 6b. Five confusions the function model dissolves

The "model is a function, everything else is the wrapper" frame is not just tidy — it resolves five confusions that cost beginners real debugging time. Each one is a wrapper bug masquerading as a model mystery.

1. **"The model contradicted itself between two messages."** The model has no memory across calls; it is stateless (§1). Whatever "self" exists between turns is the conversation history *your code* re-sends. A contradiction is almost always a history-assembly bug — you dropped a turn, truncated the wrong end, or re-ordered messages — not the model "changing its mind." Inspect the exact `messages` array you sent before you blame the weights.

2. **"It used to give the right answer and now it doesn't."** At temperature > 0 the sampler (Stage 4) draws from a distribution; the *same* prompt can yield different tokens on different calls. The model didn't regress; your sampler rolled a different draw. If you need reproducibility for a test, that's a sampling-configuration decision (lower temperature, or in week 2's terms, a tighter truncation), made in your wrapper.

3. **"It can't do basic arithmetic / can't count letters."** The tokenizer (Stage 1) presents text as sub-word pieces, not digits or characters. The model never sees "the third letter"; it sees tokens. Letter- and digit-level tasks are hard *by construction* of the interface, and the fix is to give it a tool (a calculator, code execution — week 4+), not to prompt harder.

4. **"It doesn't know something that happened last month."** Parametric knowledge is frozen in the MLP weights at the training cutoff (§3). "It doesn't know" is the default state for anything past the cutoff, by design. The fix is to put the fact in the context window (retrieval — Phase II), not to re-ask. A model "not knowing" current events is working as specified.

5. **"The first response is slow but the rest stream fine."** Prefill (compute-bound) sets time-to-first-token and scales with prompt length; decode (bandwidth-bound) sets time-per-output-token and is roughly prompt-length-independent (§4). A slow start on a long prompt with normal streaming after is the *expected* shape, not a bug. If you measured a single "latency" number, you conflated two different things.

Notice the pattern: in every case, naming the stage that owns the behavior tells you where the bug lives and what the fix is. That is the entire payoff of the function model — it turns "the AI is being weird" into "the history assembly dropped turn 3," which is a bug you can fix.

---

## 7. Recap

You should now be able to:

- State the model's external contract: `List[int] -> Vector[float, V]` — token IDs in, next-token logits out — and the four consequences (distribution not token; deterministic at the function level; stateless; flat token stream).
- Name the five stages — tokenizer, context window, forward pass, sampling, detokenizer — and assign each a primary responsibility (cost, ceiling, compute, determinism, text).
- Describe the decoder-only transformer as embedding → stacked attention+MLP blocks → unembedding, and explain why attention is quadratic, where knowledge lives, and why generation is serial.
- Distinguish prefill (compute-bound, sets TTFT, builds the KV cache) from decode (bandwidth-bound, sets TPOT, reuses the cache), and read those numbers off Ollama's response.
- Localize a production symptom to a stage using the diagnosis table.

Next: the landscape this function lives in. Which models exist in 2026, how to read a model card without falling for the benchmark, and how to read a license as the engineering constraint it is. Continue to [Lecture 2 — The Landscape, the Cards, and the Licenses](./02-the-landscape-cards-and-licenses.md).

---

## References

- *Attention Is All You Need* (Vaswani et al., 2017): <https://arxiv.org/abs/1706.03762>
- *PagedAttention / vLLM* (Kwon et al., 2023) — prefill/decode and KV cache, §2: <https://arxiv.org/abs/2309.06180>
- *Anthropic Messages API — Models overview* (current model IDs, context windows): <https://docs.claude.com/en/docs/about-claude/models/overview>
- *Anthropic — Token counting* (count tokens with the model's own tokenizer): <https://docs.claude.com/en/docs/build-with-claude/token-counting>
- *Ollama API reference* (`eval_count`, `prompt_eval_count`, `*_duration` fields): <https://github.com/ollama/ollama/blob/main/docs/api.md>
- *Intro to Large Language Models* (Karpathy) — the "from the outside" talk: <https://www.youtube.com/watch?v=zjkBMFhNj_g>
