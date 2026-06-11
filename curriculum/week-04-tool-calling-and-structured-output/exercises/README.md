# Week 4 — Exercises

Three focused drills on real tool calls. Each takes 30–60 minutes. Do them in order — exercise 3 reuses the tool-dispatch mental model you build in 1 and 2. Every exercise runs against **both** a frontier model (`claude-opus-4-8` or `claude-haiku-4-5` to save cost) **and** a local `qwen2.5:7b-instruct` through Ollama, because the whole point of the week is the gap between them.

## Index

1. **[Exercise 1 — Design a tool schema](exercise-01-design-a-tool-schema.md)** — hand-write three tool schemas, validate the model's arguments against them, watch a model self-correct from an `is_error` result, and run the same schema against Claude and Qwen. (~50 min, guided)
2. **[Exercise 2 — Structured extraction three ways](exercise-02-structured-extraction.py)** — extract a typed `Pydantic` record with `messages.parse`, with raw `output_config.format`, and with `outlines` grammar-constrained decoding on the local model. Diff the three. (~45 min, runnable)
3. **[Exercise 3 — Sanitize the tools](exercise-03-sanitize-the-tools.py)** — harden a file-read and a web-fetch tool against path traversal and SSRF; prove the attack succeeds against the naive version and fails against yours. (~45 min, runnable)

## How to work the exercises

- Have your `ANTHROPIC_API_KEY` exported and `ollama serve` running with `qwen2.5:7b-instruct` pulled before you start. `ollama list` should show it.
- **Read `response.stop_reason` after every model call.** `tool_use` means the loop continues; `end_turn` means the model is done. Train the habit of printing it.
- **Print the `tool_use` block's `name` and `input` every time.** The two QoS-blocks habit from the sibling course has an analog here: the `tool_use` request and your `tool_result` are your ground truth. Diff what the model *asked* for against what your tool *did*.
- When a model "won't call the tool," run the §5 decision tree from Lecture 2 before you touch `tool_choice`. Description first, schema second, security last.
- Each runnable exercise (`.py`) ends with an **expected output** block. If your output doesn't match the *shape* (exact numbers vary by model and timing), you're not done.

## Running the Python exercises

The two `.py` files are standalone. Install the deps and run them directly:

```bash
pip install anthropic ollama pydantic jsonschema outlines httpx
export ANTHROPIC_API_KEY=sk-ant-...
python3 exercise-02-structured-extraction.py
```

`exercise-02` will download a 7B model the first time `outlines` runs locally (several GB) — if you can't spare the disk or VRAM, the file has a `SKIP_LOCAL = True` flag that runs only the two vendor paths and still teaches the JSON-mode half.

There are no solutions checked in. The course is open source — solutions live in forks. After you finish, search GitHub for `c23-week-04` to compare.
