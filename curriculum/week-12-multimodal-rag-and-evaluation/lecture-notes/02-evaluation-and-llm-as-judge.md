# Lecture 2 — Evaluation and the Calibrated LLM-as-Judge

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can define the four Ragas metrics precisely and state what each one catches that the others miss, run a Ragas `EvaluationDataset` + `evaluate()` with either a local open model or `claude-opus-4-8` as the judge backend, build an LLM-as-judge and *calibrate* it against 10 human labels (agreement, Cohen's kappa, threshold tuning), survey DeepEval/promptfoo/TruLens and know what each adds, and name and defend against the judge biases — self-preference, position, verbosity — that quietly corrupt an uncalibrated eval.

Lecture 1 built a multimodal pipeline. This lecture answers the only question that matters once it's built: **does it actually work?** Not "does the demo look good" — that's an anecdote. Not "did the right chunk come back" — that's a *retrieval* metric, and you've measured it since week 7. The new question is whether the *generated answer* is grounded, complete, on-topic, and built from relevant context — and how much you trust the machine that told you so.

If you remember one sentence from this entire week, remember this one:

> **RAG without Ragas is a vibe.** A pipeline you can't score on faithfulness, context recall, context precision, and answer relevancy is a pipeline you're shipping on faith. Phase II ends when you replace the faith with numbers.

And the corollary that makes the numbers trustworthy:

> **An LLM-as-judge you didn't calibrate is a random number generator with good vocabulary.** Before you believe a judge's 0.81, you check it against human labels and compute the agreement (Cohen's kappa). A judge that doesn't agree with humans isn't measuring what you think it is.

Everything in this lecture is in service of one deliverable — the **Phase II milestone**: a Ragas evaluation report across three pipeline variants, with a calibrated judge behind every number, that names which metric improved most for which change.

---

## Part 1 — The four Ragas metrics, and what each one catches

Ragas is the open-source standard for RAG evaluation. Its power is that it splits "is the answer good?" into four *separable* questions, each measuring one failure mode, each computed by a small LLM-driven procedure. The reason you need all four: **a single number hides which stage failed.** A pipeline can be faithful but irrelevant, or relevant but built on missing context. The four metrics localize the failure.

Lay out the inputs first, because every metric uses a subset of them:

- **question** — the user's query.
- **answer** — what your pipeline generated.
- **contexts** — the retrieved chunks the answer was generated from.
- **reference** — the ground-truth answer (from your gold set).

### 1.1 Faithfulness — is the answer grounded in the context?

**Faithfulness** asks: is *every claim in the answer* supported by the *retrieved context*? It catches **hallucination** — the model asserting things the context never said.

The mechanism (this is what you'll re-implement from scratch in Exercise 2): decompose the answer into atomic **claims**, then for each claim, ask the judge "is this claim supported by the context?" Faithfulness is the fraction of claims that are supported.

```
faithfulness = (# claims in answer supported by context) / (total # claims in answer)
```

Worked example. Context: *"Confidential information must be protected for five years after termination."* Answer: *"Confidentiality lasts five years, and the penalty for breach is $10,000."* Two claims: (1) "lasts five years" — **supported**; (2) "penalty is $10,000" — **not in the context, hallucinated**. Faithfulness = 1/2 = 0.5. A high Recall@5 (the right clause came back) with a low faithfulness is the signature of a generator that retrieved well and then made things up — and *only* an answer-level metric catches it.

### 1.2 Context recall — did retrieval bring back everything the answer needs?

**Context recall** asks: of the claims in the *reference* (ground-truth) answer, how many are attributable to the *retrieved context*? It catches a **retriever that misses** — context that the right answer needs but retrieval didn't return.

```
context_recall = (# reference-answer claims attributable to retrieved context) / (total reference claims)
```

Note the direction: it's computed against the **reference**, not the generated answer. It's the recall analogue you know from week 7, but at the *claim* level and judged by an LLM rather than a clause-id match. If the reference answer needs two facts and retrieval only surfaced one, context recall is 0.5 *no matter what the generator did* — the information simply wasn't available. This is the metric that says "fix the retriever, not the prompt."

### 1.3 Context precision — is the retrieved context relevant, and ranked well?

**Context precision** asks: of the retrieved chunks, how many are actually *relevant* to the question, and are the relevant ones ranked *high*? It catches **noisy retrieval** — a top-k padded with irrelevant chunks, or relevant chunks buried below junk.

Mechanically, Ragas checks each retrieved chunk for relevance to the question (and reference), then computes a rank-weighted precision (relevant chunks near the top score higher). High context precision means your top-k is clean and well-ordered; low means you retrieved relevant context *plus* a lot of distracting noise, which dilutes the generator's prompt (the week-11 context-budget lesson, now measured).

The complementary pair to hold in your head: **context recall** = "did we get *all* the right context?" (a miss problem); **context precision** = "is the context we got *clean and ranked*?" (a noise problem). A retriever can be high-recall/low-precision (gets everything plus garbage) or low-recall/high-precision (clean but incomplete). You need both numbers to know which.

### 1.4 Answer relevancy — does the answer address the question?

**Answer relevancy** asks: does the generated answer actually *address the question asked*, or does it wander, hedge, or answer a different question? It catches **off-topic or evasive answers** — even faithful, well-grounded ones that miss the point.

Ragas computes this cleverly: it asks an LLM to *generate questions that the answer would be a good response to*, embeds those reverse-engineered questions, and measures their cosine similarity to the *original* question. If the answer is on-topic, the reverse-engineered questions look like the real one (high similarity); if the answer wandered, they don't.

```
answer_relevancy = mean cosine_similarity(original_question, reverse_engineered_questions_from_answer)
```

The failure it uniquely catches: an answer that's *faithful* (everything it says is grounded) and *complete* (all context recalled) but *off-topic* — "You asked about the confidentiality duration; here is the full payment schedule." Faithfulness is high (the payment schedule is grounded), context metrics may be fine, and the answer is still useless. Only answer relevancy flags it.

> **The four-metric mental model.** Faithfulness = *did the generator make things up?* Context recall = *did the retriever miss anything?* Context precision = *did the retriever return junk?* Answer relevancy = *did the answer address the question?* Four orthogonal failure modes. A single "quality score" collapses them into a number you can't act on; the four separate scores tell you *which stage to fix*.

---

## Part 2 — The Ragas harness: `EvaluationDataset` + `evaluate()`

Ragas's runtime is two objects: a dataset of samples and an `evaluate()` call that scores them with a chosen LLM and embedding backend. The 2026 shape:

```python
# pip install ragas
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference, ResponseRelevancy,
)

# Each sample carries the four inputs the metrics consume.
samples = [
    {
        "user_input": "How long must confidential information be protected?",
        "response": "Confidential information must be protected for five years after termination.",
        "retrieved_contexts": [
            "9. All confidential information must be protected for five years after termination.",
            "14. Either party may terminate this Agreement upon thirty days written notice.",
        ],
        "reference": "Five years after termination.",
    },
    # ... one dict per gold question (40 of them for the milestone report)
]
dataset = EvaluationDataset.from_list(samples)

result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
        ResponseRelevancy(),
    ],
    llm=judge_llm,          # the judge backend — see Part 3 (local OR Claude)
    embeddings=embed_model, # answer-relevancy needs an embedding model
)
print(result)  # {'faithfulness': 0.88, 'context_recall': 0.91, ...}
```

The load-bearing detail: **`evaluate()` is judge-backend-agnostic.** You pass it an `llm=` wrapper, and Ragas makes its metric sub-calls (decompose claims, check support, reverse-engineer questions) through *that* backend. So the *same metrics* run whether the judge is a local open model or Claude — which is exactly how the course's open-path and frontier-path stay equivalent. That wrapper is the next section.

---

## Part 3 — The judge backend: a local model OR Claude

Ragas wraps any LLM behind a small adapter. The **open path** is a local OpenAI-compatible endpoint (a vLLM or Ollama server running Qwen/Llama); the **frontier path** is Claude. Both produce the *same four metrics*.

### 3.1 The open path — a local OpenAI-compatible endpoint

```python
# Open path: a local vLLM/Ollama server, OpenAI-compatible, wrapped for Ragas.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Point at your local server (no API key needed beyond a placeholder).
local_chat = ChatOpenAI(
    model="qwen2.5:14b",
    base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
    api_key="ollama",                       # placeholder; local server ignores it
)
judge_llm = LangchainLLMWrapper(local_chat)
embed_model = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434/v1",
                     api_key="ollama")
)
# Pass judge_llm and embed_model into evaluate() above. Zero external dependency.
```

### 3.2 The frontier path — Claude as the judge

For the milestone report you want the strongest judge, and that's `claude-opus-4-8`. You can wrap it for Ragas, or — better for the *calibration* work in Part 4 — call it directly with **structured output** so the judge returns a validated, parseable verdict instead of free text you have to regex.

This is the authoritative way to call Claude as a judge. Note the model facts: `claude-opus-4-8`, adaptive thinking via `output_config={"effort":"high"}` (a judge wants reasoning effort), **no** `temperature`/`top_p`/`top_k` (they 400 on Opus 4.8), and structured output via `messages.parse` with a Pydantic schema.

```python
# Frontier judge: claude-opus-4-8 with structured output (validated verdict).
# pip install anthropic pydantic
import anthropic
from pydantic import BaseModel, Field

client = anthropic.Anthropic()


class FaithfulnessVerdict(BaseModel):
    """One judged answer. `score` in [0,1]; `supported`/`total` are the claim counts."""
    supported_claims: int = Field(description="# answer claims supported by the context")
    total_claims: int = Field(description="total # atomic claims in the answer")
    score: float = Field(ge=0.0, le=1.0, description="supported_claims / total_claims")
    reasoning: str = Field(description="one sentence: which claim, if any, was unsupported")


def judge_faithfulness(question: str, answer: str, contexts: list[str]) -> FaithfulnessVerdict:
    ctx = "\n---\n".join(contexts)
    prompt = (
        f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:\n{answer}\n\n"
        "Decompose the answer into atomic claims. For each, decide if it is "
        "SUPPORTED by the context. Report the counts and the score "
        "(supported / total)."
    )
    response = client.messages.parse(
        model="claude-opus-4-8",
        max_tokens=2000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},     # a judge wants reasoning effort
        messages=[{"role": "user", "content": prompt}],
        output_format=FaithfulnessVerdict,     # validated, structured verdict
    )
    return response.parsed_output             # a FaithfulnessVerdict instance
```

`response.parsed_output` is a validated `FaithfulnessVerdict` — `score`, `supported_claims`, `reasoning` — no string parsing, no "the model said 0.8 but wrapped it in prose." That structured-output discipline is what makes the judge *programmable* enough to calibrate, which is the whole next part. The same image-block shape from Lecture 1 §1 plugs in here if the answer is over a *figure* — a multimodal judge is this exact call with an `image` block in the content list.

> **Why Opus 4.8 for the judge specifically.** The judge is the one place you don't economize. A weak judge produces noisy scores that disagree with humans, and a metric you can't trust is worse than no metric (it gives false confidence). Use the most capable model you can afford for the *judge*, even if the *pipeline under test* uses a cheaper generator. And — the rule that recurs in Part 5 — never use the *same* model as both generator and judge, or self-preference bias corrupts the score.

---

## Part 4 — Calibration: making the judge's number mean something

Here is the part that separates an engineer from someone who ran a library. **An LLM-as-judge is a measurement instrument, and you do not trust an uncalibrated instrument.** Before you report "faithfulness 0.88," you check the judge against *humans* on a small labeled set and compute how much they agree. If they don't, your 0.88 is noise.

### 4.1 The 10-label calibration step

The recipe (this is Exercise 3 and a homework problem):

1. Take **10 (question, answer, context) examples** — ideally spanning clearly-good, clearly-bad, and borderline.
2. **Label them yourself** (human ground truth): for each, is the answer faithful/relevant? A binary label (pass/fail) is enough to start; a 0–1 score works too.
3. **Run the judge** on the same 10.
4. **Compare** judge vs. human: raw agreement *and* Cohen's kappa.

Ten is the minimum that's honest, not the ideal — more is better. But ten human labels, collected in twenty minutes, turn a judge from "a black box that emits decimals" into "an instrument with a known agreement rate." That's the difference between a number you can defend at the architecture review and one you can't.

### 4.2 Cohen's kappa — agreement beyond chance

Raw agreement ("the judge and I agreed on 8 of 10") is misleading, because some agreement happens *by chance* — if 90% of your examples are "faithful," a judge that always says "faithful" agrees 90% of the time while measuring nothing. **Cohen's kappa** corrects for that:

```
kappa = (p_observed - p_chance) / (1 - p_chance)
```

where `p_observed` is the raw agreement fraction and `p_chance` is the agreement you'd expect if both rated independently at their observed base rates. Kappa = 1 is perfect agreement; kappa = 0 is chance-level (the judge is worthless even if raw agreement looks high); negative is worse than chance. The interpretation bands you'll cite:

| Kappa | Interpretation |
|---|---|
| < 0.20 | Poor — the judge is barely better than guessing; do **not** ship it |
| 0.21–0.40 | Fair — weak; tune the prompt/threshold or change the judge model |
| 0.41–0.60 | Moderate — usable with caution; report the kappa alongside the metric |
| 0.61–0.80 | Substantial — a trustworthy judge; this is your target |
| 0.81–1.00 | Almost perfect — excellent agreement |

```python
# Cohen's kappa from scratch (and via sklearn). human/judge are 0/1 label lists.
def cohen_kappa(human: list[int], judge: list[int]) -> float:
    n = len(human)
    p_obs = sum(h == j for h, j in zip(human, judge)) / n
    # Chance agreement from each rater's marginal base rates.
    h1, j1 = sum(human) / n, sum(judge) / n
    p_chance = h1 * j1 + (1 - h1) * (1 - j1)
    if p_chance == 1.0:
        return 1.0  # degenerate: everyone labeled the same class
    return (p_obs - p_chance) / (1 - p_chance)

# Equivalent, via the standard library:
# from sklearn.metrics import cohen_kappa_score
# kappa = cohen_kappa_score(human, judge)
```

### 4.3 Threshold tuning — pick the judge's operating point

Your judge emits a *continuous* score (0.0–1.0); your human labels are often *binary* (pass/fail). To compare them you threshold the judge: "score ≥ τ counts as pass." **The threshold τ is a tunable hyperparameter** — and the right τ is the one that maximizes agreement (kappa) with the humans on your calibration set. You sweep it the same way you swept chunk size in week 8:

```python
# Sweep the judge threshold; pick the one that maximizes agreement with humans.
import numpy as np

judge_scores = [...]          # continuous judge outputs on the 10 calibration examples
human_labels = [...]          # binary human ground truth (0/1) on the same 10

best_tau, best_kappa = None, -1.0
for tau in np.linspace(0.1, 0.9, 17):
    judge_binary = [1 if s >= tau else 0 for s in judge_scores]
    k = cohen_kappa(human_labels, judge_binary)
    if k > best_kappa:
        best_kappa, best_tau = k, float(tau)
print(f"calibrated threshold τ={best_tau:.2f}  kappa={best_kappa:.2f}")
# Now: trust the judge ONLY with this τ, and report the kappa next to every metric.
```

The calibrated threshold and its kappa travel *with* the metric forever after. "Faithfulness 0.88" alone is a vibe; "faithfulness 0.88, judged at τ=0.62 with kappa=0.71 against 10 human labels" is **evidence**. That second form is what goes in the milestone report. That's the "the score survived calibration" promise, made concrete.

---

## Part 5 — The judge pitfalls you must defend against

LLM judges have systematic biases. An uncalibrated judge doesn't just add noise — it adds *directional* error. Name them so you can defend against them:

- **Self-preference (self-enhancement) bias.** A judge rates answers generated by its *own model family* higher. The fix is structural: **never use the same model as generator and judge.** If your pipeline generates with Claude, judge with a different model (or vice versa) — or at minimum note the risk. Judging Claude's answers with Claude inflates every score.
- **Position bias.** In *pairwise* comparison ("is answer A or B better?"), judges systematically favor the first (or last) option regardless of content. The fix: run both orders (A-then-B and B-then-A) and average, or use pointwise scoring (score each answer alone) instead of pairwise.
- **Verbosity bias.** Judges rate *longer* answers higher even when the extra length is filler. The fix: probe for it (the README stretch goal — pad correct answers and see if the score rises) and, if present, instruct the judge to ignore length, or normalize.
- **Cost.** A `claude-opus-4-8` judge at `effort=high` over a 40-question gold set × four metrics × three variants is a real bill — that's 480 judged samples, each a reasoning call. Mitigate: use a cheaper judge (`claude-haiku-4-5` or a local model) for the *sweep*, reserve Opus for the final report; cache judgments; and recognize that the calibration set is small (10) so the *expensive* part is the full report, not the calibration.

> **The cardinal rule, restated:** calibrate, then trust *within the calibrated regime only*. A judge calibrated on faithfulness for legal answers is not automatically trustworthy on relevancy for medical answers. The kappa is for *this* judge, *this* metric, *this* kind of answer. Re-calibrate when any of those change.

---

## Part 6 — The tooling survey: DeepEval, promptfoo, TruLens

Ragas is the metric spine. Three other tools sit around it; know what each *adds* so you reach for the right one.

- **DeepEval** — "Pytest for LLMs." It wraps metrics (including faithfulness, relevancy, and a configurable `GEval`) as **assertions you run in CI**: `assert_test(test_case, [FaithfulnessMetric(threshold=0.7)])` fails your build if faithfulness drops below 0.7. Reach for it when you want eval to *gate deploys*, not just produce a report. It's the bridge from "I measured it once" to "every PR is measured."

```python
# DeepEval: eval as a CI assertion (fails the build on a regression).
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def test_faithfulness():
    case = LLMTestCase(
        input="How long is confidentiality?",
        actual_output="Five years after termination.",
        retrieval_context=["Protected for five years after termination."],
    )
    assert_test(case, [FaithfulnessMetric(threshold=0.7)])
```

- **promptfoo** — prompt/model **matrix** testing, config-driven (YAML). Run the *same* eval across many prompts × many models and diff the table: "prompt A on Opus vs. prompt B on a local Qwen, which scores higher on my assertions?" Reach for it when the question is "which prompt/model *configuration* wins?" rather than "is this one pipeline good?" It's the A/B harness of week 8, pointed at prompts and models instead of chunkers.

- **TruLens** — trace-level **feedback functions** instrumented into a *running* app, with a dashboard. Where Ragas scores a batch dataset, TruLens scores *each call as it happens* (groundedness, context relevance, answer relevance) and shows you per-call traces. Reach for it when you want live instrumentation and a UI to drill into individual failures, not a one-shot batch report.

The honest summary: **Ragas** = the standard batch metrics (your spine and milestone). **DeepEval** = those metrics as CI assertions. **promptfoo** = prompt/model matrix testing. **TruLens** = live per-call instrumentation. They overlap; you don't need all four. For this week's milestone, Ragas + a calibrated judge is the deliverable; the others are tools you should *recognize* and reach for when the eval question changes shape.

---

## Part 7 — Recap

You should now be able to:

- Define the **four Ragas metrics** precisely and state what each uniquely catches: **faithfulness** (hallucination — claims unsupported by context), **context recall** (a retriever that misses needed context), **context precision** (noisy/poorly-ranked retrieval), **answer relevancy** (off-topic/evasive answers) — and explain why one collapsed "quality score" hides which stage failed.
- Run a Ragas **`EvaluationDataset` + `evaluate()`** with a **judge-agnostic** backend — a local OpenAI-compatible model (the open path) or `claude-opus-4-8` (the frontier path) — producing the same four metrics either way.
- Call **Claude as a structured-output judge** correctly: `claude-opus-4-8`, adaptive thinking with `output_config={"effort":"high"}`, **no** sampling params, `messages.parse` with a Pydantic schema returning a validated verdict — and know the same image-block shape makes it a *multimodal* judge.
- **Calibrate** a judge against **10 human labels**: compute raw agreement *and* **Cohen's kappa** (agreement beyond chance), read the kappa bands, **sweep the threshold τ** to maximize agreement, and report "metric, at τ=…, kappa=…" — the form that turns a vibe into evidence.
- Name and defend against the **judge biases** — self-preference (don't judge with the generator's model), position (run both orders), verbosity (probe and correct), and cost (cheap judge for the sweep, Opus for the report).
- **Survey** the tooling — Ragas (batch spine), DeepEval (CI assertions), promptfoo (prompt/model matrix), TruLens (live instrumentation) — and reach for the right one when the eval question changes.

Next: the exercises put this on real data — a multimodal answer step (VLM vs. Claude over a figure), the four Ragas metrics implemented from scratch, and a judge-calibration sweep against human labels. Continue to [the exercises](../exercises/README.md).

---

## References

- *Ragas — available metrics (faithfulness, context recall/precision, answer relevancy)*: <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>
- *Ragas — evaluating a RAG application (`EvaluationDataset`, `evaluate`)*: <https://docs.ragas.io/en/stable/getstarted/rag_evaluation/>
- *Ragas — customize the LLM/embeddings backend (bring your own judge)*: <https://docs.ragas.io/en/stable/howtos/customizations/customize_models/>
- *RAGAS: Automated Evaluation of Retrieval Augmented Generation* — Es et al., 2023: <https://arxiv.org/abs/2309.15217>
- *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* — Liu et al., 2023: <https://arxiv.org/abs/2303.16634>
- *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (the bias taxonomy)* — Zheng et al., 2023: <https://arxiv.org/abs/2306.05685>
- *Cohen's kappa (formula and interpretation bands)*: <https://en.wikipedia.org/wiki/Cohen%27s_kappa>
- *Anthropic — vision (the image-block shape for a multimodal judge)*: <https://docs.anthropic.com/en/docs/build-with-claude/vision>
- *DeepEval*: <https://github.com/confident-ai/deepeval> · *promptfoo*: <https://www.promptfoo.dev/> · *TruLens*: <https://www.trulens.org/>
