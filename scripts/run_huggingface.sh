#!/usr/bin/env bash
set -euo pipefail

# =========================
# EDIT THESE VARIABLES
# =========================

# Conda env name
CONDA_ENV="insight"

# Paths (defaults assume you're running from repo root)
DATASET="data/insight_test.jsonl"
HTML_ROOT="data"
OUTPUT_DIR="output"

# W&B
WANDB_PROJECT="insight-benchmark"
RUN_NAME=""     # optional, leave empty for auto
WANDB_ENTITY=""      # optional, leave empty for your default entity
WANDB_RUN_ID=""
CONTINUE_RUN=0

# Interaction
MAX_TURNS=10
LIMIT=""        # optional, e.g. 50
OFFSET=""       # optional, e.g. 0
SAVE_IMAGES=0   # 1 to save screenshots, 0 otherwise

# Local HuggingFace model
MODEL_PATH=""  # e.g. Qwen/Qwen3-VL-4B-Thinking or a local path

# Generation params (HuggingFace)
DO_SAMPLE=1          # 1 => --do-sample, 0 => --no-do-sample
TOP_P=0.95
TOP_K=20
TEMPERATURE=1.0
REPETITION_PENALTY=1.0

# If set, overrides dynamic context-window-based max_new_tokens
MAX_NEW_TOKENS=""            # e.g. 256
# If model config doesn't expose context length, set this (optional)
DEFAULT_CONTEXT_LENGTH=""    # e.g. 32768

# Prompt prefix
PROMPT_PREFIX="" # e.g. "ui_tars_"

# Parse mode
PARSE_MODE="default" # "default" or "ui_tars"

# =========================
# DO NOT EDIT BELOW
# =========================

# Activate conda
set +u
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"
set -u

PYTHON_BIN="python"

CMD=(
  "${PYTHON_BIN}" "run_benchmark.py"
  --dataset "${DATASET}"
  --html-root "${HTML_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --wandb-project "${WANDB_PROJECT}"
  --model "huggingface"
  --model-path "${MODEL_PATH}"
  --max-turns "${MAX_TURNS}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --temperature "${TEMPERATURE}"
  --repetition-penalty "${REPETITION_PENALTY}"
	--parse-mode "${PARSE_MODE}"
	--prompt-prefix "${PROMPT_PREFIX}"
)

if [[ "${DO_SAMPLE}" == "1" ]]; then
  CMD+=(--do-sample)
else
  CMD+=(--no-do-sample)
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${RUN_NAME}" ]]; then
  CMD+=(--run-name "${RUN_NAME}")
fi
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  CMD+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ -n "${DEFAULT_CONTEXT_LENGTH}" ]]; then
  CMD+=(--default-context-length "${DEFAULT_CONTEXT_LENGTH}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ -n "${OFFSET}" ]]; then
  CMD+=(--offset "${OFFSET}")
fi
if [[ "${SAVE_IMAGES}" == "1" ]]; then
  CMD+=(--save-images)
fi
if [[ "${CONTINUE_RUN}" == "1" && -n "${WANDB_RUN_ID}" ]]; then
  CMD+=(--continue)
  CMD+=(--run-id "${WANDB_RUN_ID}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


