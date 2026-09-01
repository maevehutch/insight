# InSight: A Benchmark for Agentic Claim Verification in Interactive Visualizations

---

## Overview

InSight is a benchmark for evaluating multimodal agents on claim verification over interactive visualizations. Given an interactive HTML visualization and a natural-language proposition, an agent must interact with the visualization over multiple turns , clicking, hovering, and navigating, and classify the proposition as **True**, **False**, or **NotEnoughInfo**.

The benchmark environment renders visualizations with playwright and supports evaluation of both open-source* (via HuggingFace) and closed-source (via OpenAI-compatible APIs) vision-language models, with results logged to Weights & Biases.

---

## Dataset

The benchmark data lives in `data/`:

- `data/insight_test.jsonl` — the InSight test set: 500 propositions, balanced across the three classes (True / False / NotEnoughInfo).
- `data/insight_extended.jsonl` — an extended pool of 20,849 additional propositions over the same visualizations.
- `data/html/` — the 297 interactive HTML visualizations referenced by the propositions.

Each JSONL sample contains:

- `source_id`: unique sample identifier
- `html_file`: relative path (e.g. `html/602.html`), resolved under `--html-root`
- `proposition`: the statement to verify, inserted into `[PROPOSITION]` in `prompts/user_first_interaction.txt`
- `class`: gold label in `True` / `False` / `NotEnoughInfo`

---

## Installation

1. **Create a conda environment**

```bash
conda create -n insight python=3.12 -y
conda activate insight
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

3. **Log in to Weights & Biases**

```bash
wandb login
```

---

## Running Inference

### Configure Environment Variables

For **closed-source models**, set the relevant API key:

```bash
export GEMINI_API_KEY=your_key_here   # Google AI Studio
export OPENAI_API_KEY=your_key_here   # OpenAI
```

### Quick Start with Scripts

Edit the variables at the top of the scripts, then run them from the repo root:

- **HuggingFace (local)**: `bash scripts/run_huggingface.sh`
- **Gemini (API)**: `bash scripts/run_gemini.sh`

Notes:

- The scripts activate the `insight` conda env internally (change `CONDA_ENV` inside the script if needed).
- To run only a small subset, set `LIMIT` (and optionally `OFFSET`) inside the script.

### HuggingFace (local)

```bash
python run_benchmark.py \
  --dataset data/insight_test.jsonl \
  --html-root data \
  --model huggingface \
  --model-path MODEL_PATH \
  --do-sample \
  --top-p 0.95 \
  --top-k 20 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  --max-turns 10
```

### UI-TARS (local, UI-TARS prompt + parser)

Use this for UI-TARS-family models that output `Thought:` / `Action:` with `start_box` syntax:

```bash
python run_benchmark.py \
  --dataset data/insight_test.jsonl \
  --html-root data \
  --model huggingface \
  --model-path UI-TARS-7B-DPO \
  --do-sample \
  --top-p 0.70 \
  --top-k 0 \
  --temperature 1.0 \
  --max-turns 10 \
  --prompt-prefix ui_tars_ \
  --parse-mode ui_tars
```

### OpenAI / Gemini (OpenAI-compatible endpoint)

```bash
python run_benchmark.py \
  --dataset data/insight_test.jsonl \
  --html-root data \
  --model openai \
  --api-key $GEMINI_API_KEY \
  --model-name gemini-3-pro-preview \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --max-turns 10 \
  --reasoning-effort high \
  --max-retries 5
```

---

## Arguments

`run_benchmark.py` supports:

**Data & output**

- `--dataset`: Path to the JSONL dataset (e.g. `data/insight_test.jsonl`).
- `--html-root`: Root directory used to resolve each sample's `html_file` (e.g. `data`).
- `--output-dir`: Output root for saved screenshots (default: `output`); writes into `output/<run-name>/<source_id>/...`.
- `--limit` / `--offset`: Run a subset of samples (useful for quick tests).
- `--save-images`: Save screenshots into the output directory (default: false).

**Weights & Biases**

- `--wandb-project`: W&B project name (default: `insight-benchmark`).
- `--wandb-entity`: W&B entity/team (defaults to your W&B default entity).
- `--run-name`: Optional run name; auto-generated from model + turns + timestamp if omitted.
- `--continue` / `--run-id`: Resume an existing W&B run from the last finished sample.

**Model backend**

- `--model`: `huggingface` or `openai`.
- `--model-path`: Local HuggingFace model path (required when `--model huggingface`).
- `--api-key`: API key (required when `--model openai`).
- `--model-name`: Model identifier for OpenAI/Gemini (e.g. `gpt-4o`, `gemini-3-pro-preview`).
- `--base-url`: Base URL for OpenAI-compatible endpoints.
- `--reasoning-effort`: Reasoning effort for the Gemini API (`minimal`/`low`/`medium`/`high`).
- `--max-retries`: Max retries for API calls (default: 5).

**Interaction**

- `--max-turns`: Maximum interaction turns per sample (default: 10).
- `--headless`: Run the browser headless (default: true).

**Prompt / parsing controls**

- `--prompt-prefix`: Prefix for prompt files (default: empty, using `prompts/user_first_interaction.txt`). Example: `--prompt-prefix ui_tars_` uses `prompts/ui_tars_user_first_interaction.txt`. The per-turn update prompt is always `prompts/user_update_prompt.txt`.
- `--parse-mode`: Action parsing mode. `default` expects `<action>...</action>` (Qwen-style prompts); `ui_tars` parses UI-TARS-style actions like `click(start_box='(x, y)')`.

**HuggingFace generation params**

- `--do-sample` / `--no-do-sample`: Enable/disable sampling (default: enabled).
- `--top-p` (default 0.95), `--top-k` (default 20), `--temperature` (default 1.0), `--repetition-penalty` (default 1.0).
- `--default-context-length`: Fallback context window size if the model config doesn't expose it.
- `--max-new-tokens`: Overrides the dynamic context-window-based `max_new_tokens`.

---

## License

This repository is released under the CC BY-NC-SA 4.0 license.
