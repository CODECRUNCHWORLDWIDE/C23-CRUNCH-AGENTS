# Lecture 1 — When to Fine-Tune (Usually: Don't) and the PEFT That Makes It Cheap When You Do

> **Duration:** ~2 hours of reading + hands-on.
> **Outcome:** You can walk a problem through the prompt→retrieve→fine-tune decision ladder and justify the verdict, name the three legitimate reasons to fine-tune versus the many illegitimate ones, explain full fine-tuning vs LoRA vs QLoRA vs DoRA and the memory math that fits a 7B on a 24 GB GPU, and choose SFT over the preference methods (DPO/ORPO/KTO) for the problem you actually have.

If you remember one sentence from this entire week, remember this one:

> **Fine-tuning is a debugging tool of last resort. The model is rarely the problem — the prompt, the retrieval, or the eval is.**

There's a corollary you should tape next to it:

> **A fine-tune is only as good as the eval that judges it and the data that trained it.** No held-out eval, no fine-tune; garbage data, confidently-wrong weights.

And a third, for when the temptation strikes:

> **The cheapness of modern fine-tuning is a trap, not a license.** PEFT made it one click and a few dollars — which means the *only* thing stopping a pointless fine-tune is your discipline, not your budget.

Weeks 13–15 stacked frameworks and protocols on top of *frozen* models — you never touched a weight. This week you can, and the first and most important lesson is *learning to not.* Fine-tuning is powerful, expensive, and over-reached-for; the engineer who knows when to skip it is more valuable than the one who reaches for it reflexively. So this lecture spends its first third talking you *out* of fine-tuning, and the rest making it cheap and competent for the one time in ten it's the right call.

---

## 1. The decision ladder — prompt, then retrieve, then fine-tune

When a model isn't doing what you need, there is an ordering to your options, cheapest and most reversible first:

**Rung 1 — Fix the prompt.** Most "the model can't do X" problems are "the prompt didn't ask for X clearly." Better instructions, few-shot examples, a structured output format, a chain-of-thought nudge. A prompt change is reversible in one commit, takes seconds to test, and you proved in week 3 that prompts are code you version and regression-test. **Exhaust this rung first.** A shocking fraction of fine-tunes in the wild were prompt problems in disguise.

**Rung 2 — Add retrieval.** If the problem is "the model doesn't *know* X" — a fact, a document, a current price — the fix is to *give* it X at inference time, not to bake X into weights. That's RAG, which you spent all of Phase II building. Retrieval is updatable (re-index, don't re-train), auditable (you can show the source), and it doesn't risk regressing other capabilities. **Knowledge gaps are retrieval problems, not fine-tuning problems** — fine-tuning facts into weights is slow, lossy, and stale the moment the fact changes.

**Rung 3 — Fine-tune.** Only when rungs 1 and 2 demonstrably hit a ceiling. And "demonstrably" means *measured*: you have an eval, you tuned the prompt and added retrieval, and the metric plateaued below where you need it. Now — and only now — does changing the weights earn its cost.

The ladder is the whole framework. When someone says "let's fine-tune," the first question is always "what did rungs 1 and 2 score?" If the answer is "we didn't try," you're not ready to fine-tune; you're ready to do your homework.

---

## 2. The three legitimate reasons (and the illegitimate ones)

Fine-tuning is the right call for a small, specific set of problems. There are exactly three legitimate triggers:

**1. Output style / format you can't reliably prompt.** When you need the model to *consistently* produce a specific structure, tone, or format — a particular JSON dialect, a house writing style, a domain's terse conventions — and prompting gets you to 80% but not the 99% you need. Style is *learnable from demonstrations* in a way that's hard to fully specify in a prompt. This is the most common legitimate reason, and it's exactly the DSL task this week: "turn natural language into our custom query language" is a style/format problem where 500 demonstrations beat a paragraph of instructions.

**2. Domain vocabulary the base model doesn't have.** When your domain uses tokens, jargon, or patterns the base model handles poorly because they were rare in pretraining — a specialized legal/medical/code dialect, an internal taxonomy, an unusual symbol set. Fine-tuning can teach the model the *distribution* of your domain in a way few-shot examples can't fully convey. (Note the overlap with retrieval: if it's *facts* you need, retrieve; if it's *fluency in a vocabulary*, fine-tune.)

**3. Latency / cost — getting a small model to do a big model's job.** When a frontier model handles your task well but is too slow or too expensive at your volume, and a 7B *can't* do it with prompting alone — fine-tune the 7B on the frontier model's outputs (distillation) so it does the narrow task well, then serve the cheap fast 7B. This is the "make a 7B handle what a 70B was doing" lever from the cost-engineering week, applied via weights.

Everything else is usually illegitimate. **Fine-tuning to add knowledge** → use retrieval. **Fine-tuning because the prompt is messy** → fix the prompt. **Fine-tuning to "make it smarter"** → fine-tuning makes a model *narrower and more specialized*, not generally smarter; you often *trade away* general capability for the narrow win, which is fine if you measured it and meant to. **Fine-tuning because everyone else does** → measure first.

> **The test:** can you state which of the three legitimate reasons applies, and show that prompt+retrieve hit a ceiling below your target? If not, you're on rung 1 or 2, not rung 3.

---

## 3. The cost of a fine-tune — why it's last-resort

The ladder puts fine-tuning last because it's the most expensive rung, in more ways than compute:

- **Data engineering.** You need a clean, well-formatted dataset — hundreds to thousands of examples — and the quality of those examples *is* the quality of your fine-tune. Building and curating that dataset is real work, often more than the training itself.
- **Compute and iteration speed.** A training run is minutes to hours; a prompt edit is seconds. The slow loop alone makes fine-tuning a poor *first* tool — you can try ten prompts in the time one fine-tune trains.
- **Versioning and serving.** A fine-tune is a *new artifact*. You now have a model (or adapter) to version, store, deploy, and keep in sync with the base it was trained from. Your serving stack (week 19's vLLM) has to load it. That's operational surface a prompt doesn't add.
- **Regression risk.** Fine-tuning on a narrow task can degrade capabilities you didn't measure — catastrophic forgetting, in the extreme. The model that's now great at your DSL might be worse at general reasoning. *You won't know unless you measure it*, which is why the eval (Lecture 2) is non-negotiable.

None of these are reasons to *never* fine-tune. They're reasons to fine-tune *deliberately* — with a measured justification going in and a measured verdict coming out.

---

## 3.5 The "fine-tuning got easy" shift — and why that's a trap

It's worth understanding *why* fine-tuning is over-reached-for in 2026 specifically, because the cause is recent and the trap is real. Two years ago, fine-tuning a useful model genuinely required a cluster and serious expertise — full fine-tuning of even a 7B was an infrastructure project. PEFT (LoRA, QLoRA) and friendly tooling (Unsloth, Axolotl) changed that: now anyone with a 24 GB GPU and an afternoon can fine-tune a 7B. **The barrier to *doing* a fine-tune collapsed.**

Here's the trap: the barrier to *doing* a fine-tune collapsed, but the barrier to *deciding whether you should* did not. The hard part of fine-tuning was never the training loop — it was always (a) knowing whether the problem actually needs weights changed, (b) building a clean dataset, and (c) evaluating honestly. Easy tooling made step (b)'s *mechanics* easier and step's loop trivial, but it did *nothing* for the judgment in (a) and the rigor in (c). So the field filled with fine-tunes that were trivial to run and never should have been run — prompt problems and retrieval problems solved with a $3 training run that didn't beat the free prompt.

This is exactly why the whole first half of this lecture is about *deciding* and only the second half is about *doing*. The doing is now genuinely easy; the deciding is where engineering judgment lives, and easy tooling makes that judgment *more* important, not less — because when fine-tuning is cheap and one click away, the only thing stopping you from doing it pointlessly is the discipline of the ladder. Cheap tools reward the disciplined and punish the reflexive. Be disciplined.

---

## 4. PEFT — what makes fine-tuning a 7B affordable

Now the good news: when you *do* fine-tune, you don't have to retrain the whole model. **Parameter-Efficient Fine-Tuning (PEFT)** trains a tiny fraction of the parameters and freezes the rest — and the headline technique, LoRA, is why a 7B model fits on a 24 GB GPU instead of needing a cluster.

### 4.1 Why full fine-tuning is prohibitive

Full fine-tuning updates *every* weight. The memory cost isn't just the weights — it's the weights *plus* the optimizer state. For a 7B model in 16-bit, that's ~14 GB just for the weights, and the Adam optimizer keeps two extra state tensors per parameter (momentum + variance), roughly *doubling or tripling* the memory for training. Add gradients and activations and full fine-tuning of a 7B wants ~60–80 GB — multiple high-end GPUs. That's why "fine-tune a 7B" sounds expensive: full fine-tuning *is*.

### 4.2 LoRA — freeze the base, learn a low-rank update

**LoRA (Low-Rank Adaptation)** observes that the *update* you want to apply to a weight matrix during fine-tuning is low-rank — it doesn't need the full expressivity of the matrix. So instead of updating the big frozen weight matrix `W` (say `d × d`), LoRA freezes `W` and learns two small matrices `A` (`d × r`) and `B` (`r × d`) whose product `BA` is the update, where `r` (the **rank**) is small — 8, 16, 32. At inference, the effective weight is `W + BA`.

The math: a `4096 × 4096` weight matrix has ~16.8M parameters; its rank-16 LoRA update `A` + `B` has `4096×16 + 16×4096` = ~131K parameters — a **~99% reduction** in trainable parameters for that matrix. Across the model, LoRA typically trains **<1% of the total parameters.** The frozen base never changes; only the small adapter does. That's the whole trick: you're not retraining the model, you're learning a small, additive correction.

Two knobs define a LoRA:

- **Rank (`r`)** — the capacity of the update. Higher rank = more expressive adapter = more parameters to train. 8–16 is plenty for narrow tasks; 32–64 for harder ones. Too low and the adapter can't capture the task; too high and you've thrown away LoRA's efficiency (and risk overfitting your small dataset).
- **Alpha** — a scaling factor applied to the LoRA update (`(alpha/r) · BA`). A common heuristic is `alpha = 2·r`. It controls how strongly the adapter influences the frozen base.

### 4.3 QLoRA — the 24 GB enabler

LoRA shrinks the *trainable* parameters, but the *frozen* base still sits in memory at full size (~14 GB for a 7B in 16-bit). **QLoRA** shrinks that too: it **quantizes the frozen base to 4-bit** (a special NF4 format), so the 7B base occupies ~4–5 GB instead of ~14, and trains the LoRA adapter *on top of* the quantized base. The base is never updated, so the quantization-induced precision loss in the *frozen* weights barely matters — the adapter, kept in higher precision, learns around it. The QLoRA paper showed this loses almost nothing in final quality while cutting memory dramatically.

The arithmetic that matters: 4-bit base (~5 GB) + LoRA adapter + optimizer state for *only the adapter* (tiny) + activations → comfortably under 24 GB for a 7B. **That is why your single RTX 3090/4090, or a rented A10, can fine-tune a 7B this week.** QLoRA is the technique that put fine-tuning in reach of a laptop-budget GPU.

```python
# What QLoRA looks like with Unsloth — the whole memory story in ~5 lines.
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",  # 4-bit frozen base (~5GB)
    max_seq_length=2048,
    load_in_4bit=True,                                   # QLoRA: quantize the base
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank — the capacity knob
    lora_alpha=32,           # alpha ~= 2*r — the scaling
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",   # which matrices get adapters
                    "gate_proj", "up_proj", "down_proj"],
)
# Trainable params: ~40M out of ~7B — under 1%. That's PEFT.
```

### 4.4 DoRA — a LoRA refinement

**DoRA (Weight-Decomposed Low-Rank Adaptation)** decomposes the weight update into a *magnitude* component and a *direction* component, applying LoRA to the direction and learning the magnitude separately. The result is often closer to full fine-tuning quality than vanilla LoRA at the same rank, for a small extra cost. You don't need it for this week's task — LoRA is plenty — but know it exists as the "I want a bit more from my adapter" option, and it's a one-flag change in the modern libraries (`use_dora=True`).

### 4.5 Choosing rank, alpha, and target modules with reasons

The LoRA knobs aren't magic numbers to copy from a tutorial — each has a meaning, and the challenge asks you to set them *with reasons*. Here's how to reason about each:

- **Rank (`r`)** is the adapter's *capacity* — how much the low-rank update can express. For a *narrow* task (NL→DSL, a specific format), a small rank (8–16) is plenty: there isn't much to learn, and a small rank trains faster, uses less memory, and is less prone to overfitting your small dataset. For a *broader* or harder task (a complex style, a rich domain), bump to 32–64. The failure modes bracket it: too-low rank → the adapter *underfits* (can't capture the task, eval plateaus low); too-high rank → you've thrown away LoRA's efficiency *and* given the adapter enough capacity to memorize your small dataset (overfit). **Start at 16 for a narrow task; raise it only if the eval says the adapter is underfitting.**
- **Alpha** scales the LoRA update: the effective update is `(alpha/r) · BA`. The common heuristic `alpha = 2·r` (so 32 for rank 16) is a sane default. Intuitively, alpha controls how *strongly* the adapter pulls the frozen base toward your task; the `alpha/r` ratio is what actually matters, which is why people scale alpha with rank. You rarely need to tune alpha independently for a first fine-tune — set `alpha = 2·r` and move on.
- **Target modules** decide *which* weight matrices get LoRA adapters. The standard choice for a transformer is the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) *and* the MLP projections (`gate_proj`, `up_proj`, `down_proj`). Adapting all of these (the "all linear layers" choice) gives the adapter the most reach and is the modern default for instruction-style SFT. Adapting *only* attention is a smaller, cheaper adapter that sometimes suffices for very narrow tasks. **Default to all linear layers unless you have a memory reason to narrow it.**

The meta-point: these are *hyperparameters you can sweep*, exactly like chunk size in week 8 or `ef_search` in week 7. If your first fine-tune underperforms, the eval tells you which way to move — a low eval with low rank says "raise rank"; an eval that's great on train but poor on held-out says "lower rank or fewer epochs (overfit)." You don't guess the knobs once and pray; you set sane defaults (rank 16, alpha 32, all-linear), measure, and adjust if the held-out number says to. The defaults are a starting point, not a destination.

### 4.6 What the adapter file actually is

When you `save_pretrained` a LoRA, you save the **adapter** — the small `A` and `B` matrices — *not* the whole model. The result is a directory of tens of megabytes (the adapter weights plus a config recording the rank, alpha, target modules, and the base model it was trained from), versus the multi-gigabyte base. This has practical consequences worth internalizing:

- **The adapter is meaningless without its base.** Load the adapter on a *different* base than it was trained from and you get garbage — the low-rank update was learned as a correction to *specific* frozen weights. So you always track the (base, adapter) *pair*, and the adapter's config records which base it expects.
- **You can ship many adapters for one base.** Because the adapter is tiny and the base is shared, you can train *several* narrow adapters (one per task) against the same base and load whichever you need at inference. vLLM (week 19) can even serve multiple LoRA adapters against one loaded base, hot-swapping per request — a powerful pattern for serving many specialized behaviors cheaply.
- **You can "merge" the adapter into the base** to produce a standalone fine-tuned model (`merge_and_unload`), which is convenient for quantizing-and-serving (the GGUF stretch goal) but loses the swap-multiple-adapters flexibility. Merge when you want one self-contained model; keep it separate when you want adapter agility.

The small-portable-adapter is one of LoRA's underrated wins: it makes fine-tunes *cheap to store, share, and serve*, which is part of why PEFT made fine-tuning accessible — you're not shipping a new multi-gigabyte model per task, just a small correction file against a shared base.

### 4.7 Why this matters for the decision, not just the mechanics

It's tempting to treat §4 as "the mechanics" — interesting plumbing, but secondary to the §1–3 decision. It's not, and here's the connection: **the cheapness PEFT buys is exactly what makes the over-reaching temptation worse.** When fine-tuning required a cluster and a week, the cost itself enforced discipline — nobody fine-tuned casually. PEFT removed that natural brake: now a fine-tune is a few dollars and an afternoon, so the *only* thing stopping a pointless fine-tune is the decision discipline of §1.

So the mechanics and the decision are two halves of one lesson. PEFT is what makes the fine-tune *runnable* on your hardware (the engineering enablement), and the ladder is what makes it *worth running* (the judgment). A fine-tuning engineer needs both — knowing *how* to train a LoRA without knowing *when* to do so produces a pile of cheap, pointless adapters; knowing *when* without knowing *how* leaves you unable to execute when the decision says "yes." This week teaches both, in order: decide first (§1–3, §5), then execute cheaply (§4, Lecture 2), then measure honestly (Lecture 2 §3). The cheapness is a gift and a trap — a gift because you can afford to run the experiment, a trap because you can afford to run it *pointlessly*. The discipline is what turns the gift into engineering and avoids the trap.

The one-sentence synthesis of this whole lecture: **fine-tuning is now easy to *do* and still hard to *justify*, so the engineering moved from the training loop to the decision and the eval.** A decade ago, the skill was making training *work*; today the tooling works, and the skill is knowing *whether to* and *whether it helped*. That's why this lecture spent its weight on the ladder, the legitimate reasons, the artifact lifecycle, and the SFT-vs-preference choice — those are the parts that don't have a library to do them for you. Lecture 2 covers the parts that do (the data pipeline, the training loop) plus the part that's the actual deliverable (the honest eval). Carry the decision discipline into it: the cleanest training run in the world is worthless if it answered a question you shouldn't have asked.

A parting image to hold onto: think of fine-tuning the way a senior engineer thinks of *adding a cache* to a system. A cache is easy to add, often tempting, and occasionally exactly right — but it's a new piece of state to invalidate, version, and reason about, so you add it only when you've measured that the simpler thing (a faster query, a better index) isn't enough, and you measure that the cache actually helped. Fine-tuning is the cache of the LLM stack: a powerful, stateful, easy-to-add optimization that you reach for *after* the cheaper options, *with* a measured justification, and *never* reflexively. The engineers who add caches everywhere produce brittle systems full of stale state; the engineers who fine-tune everything produce model fleets full of pointless adapters. Be the one who reaches for it deliberately, measures whether it helped, and is happy to take it back out when the number says it didn't earn its place.

---

## 5. SFT vs the preference methods — which objective?

PEFT (LoRA/QLoRA) is *how* you train cheaply; the **objective** is *what* you train toward. There are two families, and for almost every problem this week and beyond, you want the first.

### 5.1 SFT — supervised fine-tuning on demonstrations

**Supervised Fine-Tuning** is the bread and butter: you have input→output *demonstrations* (here's a natural-language query, here's the correct DSL), and you train the model to reproduce the outputs given the inputs. It's just next-token prediction on your demonstrations, with the loss computed on the *response* tokens. If you can write down "given X, the right answer is Y" for a few hundred examples, SFT is your method. **This is what you'll do this week**, and it's what 90% of applied fine-tuning is.

### 5.2 Preference methods — DPO, ORPO, KTO

The preference family trains on *comparisons* rather than demonstrations: instead of "given X, produce Y," you provide "given X, output Y_preferred is better than output Y_rejected," and the model learns to prefer the better style. Use these when the right answer isn't a single demonstration but a *preference* — "this summary is better than that one," "this tone over that tone." The 2026 lineup:

- **DPO (Direct Preference Optimization)** — aligns to preference pairs directly, no separate reward model, no RL loop. The most popular preference method.
- **ORPO** — folds SFT and preference into one objective with no reference model; appealing when you want both in a single pass.
- **KTO** — works from *unpaired* binary feedback (good/bad) rather than explicit pairs, useful when you can't construct preferred-vs-rejected pairs.

The honest guidance: **you almost certainly want SFT.** Preference methods need *preference data* (pairs of outputs ranked against each other), which is more expensive to produce than demonstrations and only earns its cost when the task is genuinely about *preference* rather than *correctness*. For a DSL task with a correct answer, SFT on demonstrations is exactly right; DPO would be using a preference hammer on a correctness nail. The stretch goal lets you *try* DPO on top of your SFT adapter to feel the difference — but the headline lab is SFT, deliberately.

### 5.3 RLHF / RLAIF — the lineage you read about, not run

**RLHF (Reinforcement Learning from Human Feedback)** and **RLAIF (… from AI Feedback)** are the alignment techniques behind frontier instruction-following: train a reward model from human (or AI) preferences, then use RL (PPO and successors) to optimize the policy against that reward. They're powerful and they're how the big labs align models — and they are *firmly out of scope for a single-GPU week.* You'll read the *Constitutional AI* / RLAIF papers (resources) for literacy, you should be able to say what they *are* and why DPO emerged as a simpler alternative, but you will not run an RLHF pipeline this week. Know the lineage; don't build it on a 24 GB card.

---

## 6. The decision, made concrete

Putting the lecture together, here's the reasoning you run before any fine-tune, and the exercise (Exercise 1) drills exactly this:

1. **State the symptom.** "The model produces invalid DSL 30% of the time."
2. **Climb the ladder.** Did a better prompt fix it? (Try it — measure.) Did few-shot examples fix it? (Try it — measure.) Is it a knowledge gap retrieval would close? (No — it's a format/style problem.)
3. **Name the legitimate reason.** Output style/format the model can't reliably hit by prompt → reason #1. Legitimate.
4. **Confirm the ceiling.** Best prompt + few-shot tops out at 0.42 exact-match; target is 0.85. Ceiling confirmed by measurement.
5. **Pick the objective.** Demonstrations of correct DSL exist (or can be built) → SFT, not DPO.
6. **Pick the method.** 7B on a 24 GB GPU → QLoRA (4-bit base + LoRA adapter).
7. **Commit to the eval.** A held-out test set, exact-match + valid-DSL, base vs fine-tune. *This is the contract* — without it, you can't claim the fine-tune worked.

That sequence — symptom → ladder → reason → ceiling → objective → method → eval — is the whole engineering discipline of fine-tuning. The training run in Lecture 2 is the easy part; *this* is the part that separates a measured decision from a reflex.

---

## 7. The memory math, worked — why the numbers fit (or don't)

Lecture 1's PEFT story leans on "QLoRA fits a 7B in 24 GB." Let's actually do the arithmetic, because a fine-tuning engineer who can't estimate the memory budget will rent the wrong GPU and waste their compute allowance debugging out-of-memory errors. Exercise 3 makes you compute this; here's the reasoning behind it.

A parameter's memory cost during training has four contributors:

1. **The weights themselves.** A 7B model in 16-bit (`bf16`/`fp16`) is `7B × 2 bytes = ~14 GB`. In 4-bit (QLoRA's frozen base) it's `7B × 0.5 bytes = ~3.5–5 GB` (a little more in practice due to quantization metadata).
2. **Gradients.** One gradient per *trainable* parameter, same precision as the param. For full fine-tuning that's another ~14 GB (every param is trainable). For LoRA, gradients exist only for the *adapter* — a fraction of a GB.
3. **Optimizer state.** Adam keeps *two* extra tensors per trainable parameter (first and second moment), typically in `fp32` (4 bytes each). For full fine-tuning that's `7B × 4 × 2 = ~56 GB` — the single biggest line item, and the reason full fine-tuning of a 7B needs a cluster. For LoRA, optimizer state exists only for the adapter — again a fraction of a GB.
4. **Activations.** Intermediate values held for the backward pass; they scale with batch size and sequence length, not param count. A few GB at modest batch/sequence settings.

Now the comparison falls out:

| Regime | Base weights | Grads + optimizer | Total (approx, 7B) | Fits 24 GB? |
|---|---|---|---|---|
| **Full fine-tune (16-bit)** | ~14 GB | ~70 GB (every param) | **~80+ GB** | No — needs a cluster |
| **LoRA (16-bit base)** | ~14 GB | <1 GB (adapter only) | **~15–18 GB** | Tight but yes |
| **QLoRA (4-bit base)** | ~4–5 GB | <1 GB (adapter only) | **~6–10 GB** | Comfortably |

The two insights to carry: **(a)** full fine-tuning's killer isn't the weights, it's the *optimizer state* on every parameter — LoRA eliminates it by making almost nothing trainable. **(b)** QLoRA's extra trick over LoRA is shrinking the *frozen base* (which LoRA still keeps at full size), and since the base never trains, the 4-bit precision loss in it barely matters. The headroom QLoRA buys (24 GB budget, ~8 GB used) is what lets you fine-tune with a generous batch size and sequence length without OOM-ing. When you rent your A10/L4 for the challenge, this is why a 24 GB card is plenty for a 7B QLoRA run and why you'd reach for an H100 only if you wanted full fine-tuning (you don't, this week).

---

## 8. The fine-tune as an artifact in a system

One more framing before the recap, because it's the part new fine-tuners under-appreciate. A fine-tune isn't just a training run that produces a number — it's an **artifact that enters your system and stays there.** The LoRA adapter you train is a file (tens of MB) that has to be:

- **Versioned.** Which base model was it trained from? Which dataset version? Which hyperparameters? A LoRA adapter is meaningless without its base — load it on the wrong base and you get garbage. You version the *pair* (base + adapter) and the dataset that made it.
- **Served.** Your serving stack (week 19's vLLM) has to load the adapter alongside (or merged into) the base. vLLM supports serving LoRA adapters directly, which is convenient — but it's a serving concern you've now taken on.
- **Re-evaluated when anything upstream changes.** If you upgrade the base model, your adapter may not transfer; you re-train and re-evaluate. If your task drifts, the fine-tune that was "worth it" six months ago may no longer be.
- **Monitored for regression.** The fine-tune that wins the narrow task today might silently degrade on inputs you didn't test, especially as your real traffic distribution shifts away from your training distribution.

This operational tail is why the decision ladder puts fine-tuning *last*: a prompt change adds no artifact, no serving concern, no version-pair to track. A fine-tune adds all four. None of this means "never fine-tune" — it means the fine-tune has to *clear a higher bar*, because you're not just paying for the training run, you're signing up for the artifact's whole lifecycle. The "was it worth it?" verdict in Lecture 2 is really "was it worth it *including the lifecycle cost*?" — which is why a +0.03 gain that adds a versioned, served, monitored artifact is so often *not* worth it.

This connects forward: in week 19 you'll serve a fine-tuned model on vLLM, and the LoRA you train this week is a candidate for that. The artifact you produce here doesn't end at the eval — it flows into the serving stack, the cost model, and the on-call runbook. Fine-tuning is a *systems* decision, not just a training decision, and that's the lens the whole week is built around.

---

## 8.5 — The objectives in one more pass, with the data each needs

The SFT-vs-preference distinction (§5) is the choice people most often get wrong, so let's anchor it in the *data* each requires, because the data requirement is what actually decides the choice.

**SFT needs demonstrations.** A demonstration is `(input, correct_output)` — "this query, that DSL." You can write demonstrations whenever you can say what the right answer *is*. For a correctness task (classification, NL→DSL, extraction, format-conformance), demonstrations are natural: you know the right label/query/structure. The data is cheap to produce (you can often generate or programmatically construct it) and easy to validate (run it through your checker). This is why SFT covers ~90% of applied fine-tuning — most tasks have a notion of "the right answer," and demonstrations are how you teach it.

**Preference methods need comparisons.** A preference pair is `(input, preferred_output, rejected_output)` — "for this input, output A is better than output B." You need this when there's *no single right answer*, only a *better* one: this summary reads better than that one, this tone fits the brand better, this response is more helpful. Producing preference data is *more* expensive than demonstrations — you (or annotators) have to *rank* outputs, which requires generating multiple candidates and judging them, and the judgments are subjective. That cost is only worth paying when the task is genuinely about preference and not correctness.

The decision rule, sharpened: **can you write down the right answer? → SFT. Can you only say which of two answers is better? → preference method.** For the DSL task, the right answer is a specific valid query — you can write it down — so SFT. If your task were "write a *good* product description" (no single right answer, only better/worse), preference methods would earn their keep. Reaching for DPO on a correctness task is the classic over-engineering mistake: you pay for expensive preference data to solve a problem demonstrations already solve, and you often get a *worse* result because preference optimization on a task with a clear correct answer is the wrong tool.

A practical sequencing note for when preference *is* warranted: the standard recipe is **SFT first, then preference** (DPO/ORPO on top of the SFT'd model). SFT teaches the model the *task*; the preference pass then *refines* the style/quality. You don't typically jump straight to preference optimization on a base model — you SFT to competence, then align to preference. The stretch goal this week (a DPO pass on top of your SFT adapter) follows exactly this recipe, so you feel the two-stage shape.

---

## 8.7 Common questions, answered

Questions that recur every cohort, because the answers sharpen the judgment:

**"Can't I just fine-tune to add knowledge faster than building RAG?"** No — and this is the most common mistake. Knowledge fine-tuned into weights is slow to update (re-train when the fact changes), lossy (the model may not reliably recall what you trained), stale (frozen at training time), and unauditable (you can't cite the source). Retrieval is updatable, exact, current, and citable. Knowledge → retrieval, every time (§1, §2).

**"How many examples do I really need?"** Fewer than you think, if they're clean. For a narrow task, a few hundred *high-quality* examples beat thousands of noisy ones (Lecture 2). Quality and consistency matter more than count past a low threshold. Start with ~500 curated examples for a narrow task and only scale up if the eval says you're data-starved.

**"LoRA or QLoRA?"** QLoRA on a single GPU — it's LoRA with the frozen base quantized to 4-bit, which is what makes a 7B fit in 24 GB. The quality cost over plain LoRA is negligible (the base is frozen). If you have abundant VRAM and want to skip quantization, plain LoRA is fine, but for this week's hardware, QLoRA (§4.3).

**"Will fine-tuning make my model worse at other things?"** It can — narrow fine-tuning can degrade general capability (regression / catastrophic forgetting). It's *fine* if the model only ever does the narrow task, but you have to *notice* the trade. Spot-check general queries before and after (Lecture 2 §3.3). A fine-tune that wins the narrow task and loses general ability is a *trade*, acceptable only if you meant it.

**"When is the answer 'don't fine-tune' even after I climbed the ladder?"** When the measured fine-tune gain is small relative to the cost (the new artifact's lifecycle, §8). A +0.03 gain over the best prompt, for a versioned-served-monitored artifact, is usually *not worth it* — ship the prompt. The verdict is the deliverable, and "no" is a valid, valuable verdict (Lecture 2 §3.2).

---

## 8.8 The ladder applied to three real symptoms

To cement the decision discipline, here are three symptoms an engineer might bring you, run through the ladder. The exercise drills more; these set the pattern.

**Symptom 1: "The model gives wrong answers about our product's pricing."**
- Rung 1 (prompt)? Won't help — the model doesn't *know* the pricing; a better prompt can't conjure a fact.
- Rung 2 (retrieve)? **Yes.** This is a knowledge gap. Put the pricing doc in the RAG index; the model retrieves and answers from it. Updatable when prices change.
- Verdict: **retrieve, don't fine-tune.** Fine-tuning pricing into weights would be stale the next time prices change.

**Symptom 2: "The model writes our API responses in inconsistent JSON shapes — sometimes nested, sometimes flat."**
- Rung 1 (prompt)? Tried — a schema in the prompt and few-shot examples got it to ~85% consistent, but the 15% drift persists across prompt iterations.
- Rung 2 (retrieve)? No — it's not a knowledge gap; the model *knows* JSON, it just doesn't reliably hit *your* shape.
- Rung 3 (fine-tune)? **Yes** — measured prompt ceiling, output-format problem (legitimate reason #1). Objective: **SFT** on demonstrations of the correct shape. Eval: held-out exact-match on the JSON shape.
- Verdict: **fine-tune (SFT)**, because prompting hit a measured ceiling on a format the model can learn from demonstrations.

**Symptom 3: "Our frontier-model classifier is accurate but too expensive at 5M calls/day."**
- Rung 1 (prompt)? A prompted local 7B gets ~78% — not good enough to replace the frontier model.
- Rung 2 (retrieve)? No — classification has no knowledge gap to fill.
- Rung 3 (fine-tune)? **Yes** — latency/cost problem (legitimate reason #3). Distill: SFT the 7B on the frontier model's labeled outputs, then serve the cheap 7B. Eval: held-out accuracy, fine-tuned 7B vs frontier baseline.
- Verdict: **fine-tune (SFT/distillation)** — *if* the fine-tuned 7B gets close enough to the frontier accuracy to justify the swap. The eval decides; if the 7B can't get close, you keep paying for the frontier model.

The pattern across all three: **the symptom alone doesn't tell you the rung — the *cause* does.** "Wrong answers" was a knowledge gap (retrieve). "Inconsistent format" was a learnable style problem with a prompt ceiling (fine-tune). "Too expensive" was a cost problem a small specialized model could solve (fine-tune). Diagnose the cause, climb the ladder, and let the cheapest sufficient rung win — and when fine-tuning *is* the answer, commit to the held-out eval that proves it cleared the ceiling.

---

## 8.9 How this lecture connects to the rest of C23

Fine-tuning isn't an island — it sits inside the systems you've built and will build, and the decision discipline here threads through the whole course:

- **Week 3 (prompt-as-code)** is *rung 1* of the ladder. The prompt you versioned and regression-tested there is the thing a fine-tune must beat. If you haven't exhausted the prompt, you're not on rung 3.
- **Phase II (RAG)** is *rung 2*. Everything you built — chunking, embeddings, retrieval, memory — is the answer to "the model doesn't *know* something." Fine-tuning is what you reach for when the problem *isn't* a knowledge gap, which is exactly the line retrieval can't cross.
- **Week 19 (vLLM serving)** is where a *worth-it* adapter gets served. vLLM can load LoRA adapters directly, so the artifact you produce this week flows straight into the serving stack — and the lifecycle cost (§8) is *real* precisely because that serving stack has to carry it.
- **Week 21 (cost engineering)** is where reason #3 (latency/cost) pays off: a fine-tuned 7B serving a narrow task instead of a frontier model is exactly the "small model for the easy job" routing lever, realized via weights.
- **The Phase III milestone** wants a fine-tune-or-not decision document for the capstone domain — *this lecture's discipline produces it.*

The throughline is the same one that runs through every C23 week: **measure, don't guess.** Week 7 measured embeddings, week 8 measured chunkers, week 12 measured RAG with Ragas, and this week measures whether a fine-tune cleared the prompt ceiling. The technique changes; the discipline — a baseline, a held-out comparison, a number that justifies the decision — does not. Fine-tuning is just one more variable you tune against a measured target, and the most senior move is knowing when *not* to touch it.

---

## 9. Recap

You should now be able to:

- Walk a problem through the **prompt → retrieve → fine-tune ladder**, exhausting the cheap reversible rungs first and only fine-tuning when they demonstrably hit a measured ceiling.
- Name the **three legitimate reasons** to fine-tune (output style/format, domain vocabulary, latency/cost via a small specialized model) and recognize the illegitimate ones (knowledge → retrieve, messy prompt → fix the prompt, "make it smarter" → fine-tuning narrows, it doesn't broaden).
- Explain the **cost** of a fine-tune — data engineering, slow iteration, a new versioned artifact, regression risk — that makes it last-resort.
- Explain **PEFT**: why full fine-tuning is memory-prohibitive, how **LoRA** freezes the base and learns a low-rank update (rank + alpha as the knobs, <1% of params trained), how **QLoRA** quantizes the frozen base to 4-bit to fit a 7B in 24 GB, and what **DoRA** refines.
- Do the **memory math** that explains why full fine-tuning's optimizer state blows past 24 GB while QLoRA's 4-bit base + tiny adapter fits comfortably — the arithmetic behind "rent a 24 GB card, not an H100."
- Choose the **objective**: **SFT** on demonstrations for correctness/format tasks (what you'll do), **DPO/ORPO/KTO** for genuine preference tasks (surveyed), and **RLHF/RLAIF** as the out-of-scope alignment lineage you read about.
- Set the **LoRA knobs with reasons** — rank as capacity (16 for narrow, raise if underfitting), alpha as scaling (`2·r` default), target modules (all linear by default) — and treat them as sweepable hyperparameters the eval tunes.
- Understand the **adapter as an artifact**: a tiny (base, adapter) pair that's cheap to store/share/serve, meaningless without its base, and that enters the serving stack (week 19) with a real lifecycle cost — which is *why* fine-tuning stays last on the ladder even though it's cheap to run.
- Hold the **synthesis**: fine-tuning is easy to *do* and hard to *justify*, so the engineering moved from the training loop to the decision and the eval — and the cache analogy (a powerful, stateful optimization you reach for last, with measured justification) is the mental model to carry forward.

Next: how to build the dataset that makes or breaks the fine-tune, how to run the LoRA training loop with Unsloth and read the loss curve, how to evaluate honestly against the base model on a held-out set, and the tooling roster (Unsloth/Axolotl/NeMo/TRL). Continue to [Lecture 2 — Data, Training, and Honest Evaluation](./02-data-training-and-honest-evaluation.md).

---

## References

- *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., 2021: <https://arxiv.org/abs/2106.09685>
- *QLoRA: Efficient Finetuning of Quantized LLMs* — Dettmers et al., 2023: <https://arxiv.org/abs/2305.14314>
- *DoRA: Weight-Decomposed Low-Rank Adaptation* — Liu et al., 2024: <https://arxiv.org/abs/2402.09353>
- *DPO: Direct Preference Optimization* — Rafailov et al., 2023: <https://arxiv.org/abs/2305.18290>
- *Constitutional AI / RLAIF* — Bai et al., 2022 (the lineage, not the lab): <https://arxiv.org/abs/2212.08073>
- *Hugging Face PEFT library (`LoraConfig`)*: <https://huggingface.co/docs/peft>
- *Unsloth (the single-GPU training path)*: <https://docs.unsloth.ai/>
- *ORPO: Monolithic Preference Optimization without Reference Model* — Hong et al., 2024: <https://arxiv.org/abs/2403.07691>
