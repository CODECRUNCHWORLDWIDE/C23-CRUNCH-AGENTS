# Week 16 — Resources

Every resource here is **free** or has a free tier. The PEFT/LoRA/QLoRA papers are on arXiv. Unsloth, Axolotl, TRL, and PEFT are open source. The base model (Qwen2.5-7B) and its card live on Hugging Face. The only thing that costs money is GPU time, and the whole week fits in **~$5 of rented compute** (an A10/L4 at ~$0.50–$1.00/h for the training run).

The tooling moves every cohort — Unsloth and Axolotl ship breaking changes regularly. The *concepts* (the prompt→retrieve→fine-tune ladder, LoRA's low-rank update, QLoRA's 4-bit base, SFT data format, the held-out eval) are stable. When a specific API page 404s, search the project's docs for the function name (`FastLanguageModel.from_pretrained`, `SFTTrainer`).

This week sidesteps the orchestration track. The eval discipline (gold set, held-out split, a reported metric) comes from weeks 8 and 12; the tokenizer/chat-template intuition comes from week 2. The resources below assume that grounding.

## Required reading (work it into your week)

- **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., 2021. The paper that started PEFT: freeze the base, learn a low-rank update. Read it until you can explain rank and alpha:
  <https://arxiv.org/abs/2106.09685>
- **QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers et al., 2023. The 24 GB enabler: quantize the frozen base to 4-bit (NF4), train LoRA on top, lose almost nothing. The reason your single GPU can fine-tune a 7B:
  <https://arxiv.org/abs/2305.14314>
- **Unsloth documentation** — the single-GPU-friendly fine-tuning library you'll actually use; read the SFT/LoRA quickstart and the Qwen notebook. `FastLanguageModel`, the chat-template helpers, the save-adapter flow:
  <https://docs.unsloth.ai/>
- **The Hugging Face TRL `SFTTrainer` docs** — the supervised-fine-tuning trainer Unsloth wraps; read the dataset-format and packing sections so you understand what's under Unsloth's hood:
  <https://huggingface.co/docs/trl/sft_trainer>

## The PEFT references

- **Hugging Face PEFT library** — the canonical implementation of LoRA/QLoRA/DoRA; the `LoraConfig` (rank `r`, `lora_alpha`, `target_modules`, dropout) is the vocabulary of the week:
  <https://huggingface.co/docs/peft>
- **DoRA: Weight-Decomposed Low-Rank Adaptation** — Liu et al., 2024. The LoRA refinement that decomposes the update into magnitude + direction; a stretch-goal alternative:
  <https://arxiv.org/abs/2402.09353>
- **`bitsandbytes`** — the 4-bit/8-bit quantization backend QLoRA uses (NF4, double quantization). You rarely call it directly; you should know it's what makes the frozen base fit:
  <https://github.com/bitsandbytes-foundation/bitsandbytes>

## Post-training objectives (SFT you'll do; preference methods you'll survey)

- **DPO: Direct Preference Optimization** — Rafailov et al., 2023. Preference alignment without a separate reward model or RL loop; the stretch goal touches it:
  <https://arxiv.org/abs/2305.18290>
- **ORPO: Monolithic Preference Optimization without Reference Model** — Hong et al., 2024. SFT + preference in one objective, no reference model:
  <https://arxiv.org/abs/2403.07691>
- **TRL — DPO/ORPO/KTO trainers** — the building blocks for preference methods, when you graduate past SFT:
  <https://huggingface.co/docs/trl/dpo_trainer>
- **Constitutional AI / RLAIF** — Bai et al., 2022. The alignment lineage you read about but do not run this week; know what RLHF/RLAIF *are* and why they're out of scope for a single-GPU week:
  <https://arxiv.org/abs/2212.08073>

## Dataset engineering

- **The chat-template / instruction-format reference (HF tokenizers)** — `apply_chat_template` turns your instruction/response pairs into the exact token sequence the model expects. Getting this wrong is the most common silent fine-tune bug:
  <https://huggingface.co/docs/transformers/chat_templating>
- **Alpaca / instruction-tuning dataset format** — the canonical `{"instruction", "input", "output"}` shape and its lineage; you'll adapt it for the DSL task:
  <https://github.com/tatsu-lab/stanford_alpaca#data-release>
- **`datasets` (Hugging Face)** — load/split/map your JSONL into a `Dataset`; the `train_test_split` you use for the held-out set:
  <https://huggingface.co/docs/datasets>

## The training tooling roster (have these open on Wednesday)

- **Unsloth** — `pip install unsloth`. Single-GPU LoRA/QLoRA, 2× faster and lower-memory than vanilla, the friendliest entry. The headline lab uses this:
  <https://github.com/unslothai/unsloth>
- **Axolotl** — config-file-driven multi-GPU SFT/DPO; production-credible when you outgrow a single GPU. Survey-depth this week:
  <https://github.com/axolotl-ai-cloud/axolotl>
- **NVIDIA NeMo Framework** — serious training at scale (multi-node, full fine-tuning, large models). The "when you have a cluster" answer; survey-depth:
  <https://github.com/NVIDIA/NeMo>
- **TRL (Hugging Face)** — `SFTTrainer`, `DPOTrainer`, the building blocks Unsloth and Axolotl wrap. Read this to understand what's underneath:
  <https://github.com/huggingface/trl>

## Evaluation (the held-out-eval spine)

- **The held-out test split** — the discipline from weeks 8/12, applied to generation: never evaluate on data the model trained on. Your `train_test_split` is the firewall:
  <https://huggingface.co/docs/datasets/process#split>
- **LLM-as-judge (Ragas-style calibration)** — for open-ended outputs where exact-match doesn't apply, an LLM judge scores quality; calibrate it against a few human labels (week 12's lesson):
  <https://docs.ragas.io/en/stable/concepts/metrics/>
- **`evaluate` (Hugging Face)** — metric implementations (exact-match, BLEU, ROUGE) for the cases where a string metric fits:
  <https://huggingface.co/docs/evaluate>

## Models you'll use this week

- **`Qwen/Qwen2.5-7B-Instruct`** — the base model for the headline fine-tune. 7B fits in 24 GB under QLoRA; the Instruct variant gives you a working chat template to build on:
  <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- **`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`** — Unsloth's pre-quantized 4-bit checkpoint; loads faster and uses less memory than quantizing the full-precision weights yourself:
  <https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit>
- **A tiny model for the CPU/mechanics path** — e.g. a 0.5B Qwen or similar; you won't get a useful fine-tune from it, but you can run the *whole pipeline* (format → train a few steps → eval) on CPU to learn the mechanics before renting a GPU.

## Tools you'll use this week

- **`unsloth`** — the training library. CUDA-Linux fast path; an MLX path exists for Apple Silicon (secondary).
- **`trl` / `peft` / `transformers` / `datasets`** — the HF stack Unsloth sits on; you import pieces of it directly for eval and data prep.
- **`bitsandbytes`** — the 4-bit backend (pulled in by Unsloth on CUDA).
- **A rented GPU** — A10/L4 (~$0.50–$1.00/h) is plenty for a 7B QLoRA run; the whole week is ~$5. RunPod, Lambda, Vast, or your cloud of choice.
- **A held-out `test.jsonl`** — the firewall between "the model learned" and "the model memorized."

## A note on the task and the corpus

The headline fine-tune is a **natural-language → custom-DSL** task: a narrow, well-defined domain where the base model's prompt ceiling is real and a fine-tune can plausibly clear it. The DSL is small and synthetic (think "turn 'list all contracts signed after 2024 in Delaware' into a query in our toy CONTRACTQL"), which keeps the task *checkable* — a generated DSL string either parses and matches the target or it doesn't, so exact-match and valid-DSL rate are honest, automatable metrics. This is deliberate: a task with an automatable metric lets you measure the fine-tune *without* a human in the loop, which is the difference between a real verdict and a vibe. The legal-corpus flavor keeps continuity with the rest of C23, but the *technique* transfers to any narrow NL→structured task.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Fine-tuning** | Changing a model's weights by further training on your data. Last resort. |
| **SFT** | Supervised Fine-Tuning — train on input→output demonstrations. What you'll do. |
| **DPO / ORPO / KTO** | Preference alignment — train on preferred-vs-rejected pairs. Surveyed, not run. |
| **RLHF / RLAIF** | Reinforcement-learning alignment with human/AI feedback. The lineage; out of scope this week. |
| **PEFT** | Parameter-Efficient Fine-Tuning — train a small fraction of params (LoRA et al.). |
| **LoRA** | Low-Rank Adaptation — freeze the base, learn small low-rank update matrices. |
| **QLoRA** | LoRA on a 4-bit-quantized frozen base — the trick that fits a 7B in 24 GB. |
| **DoRA** | Weight-decomposed LoRA — magnitude + direction; a LoRA refinement. |
| **Rank (`r`)** | LoRA's capacity knob — the rank of the update matrices (8/16/32 typical). |
| **Alpha** | LoRA's scaling factor for the update; often set to ~2× rank. |
| **Adapter** | The trained LoRA weights — a small file you load on top of the base. |
| **Chat template** | The exact token format (roles, special tokens) the model expects; `apply_chat_template`. |
| **Held-out test set** | Data the model never trained on; the firewall against memorization. |
| **Exact-match** | A metric: did the generated output equal the target string exactly? |
| **Loss curve** | Training loss over steps; reads as learning / memorizing / diverging. |

---

*If a link 404s, please open an issue so we can replace it.*
