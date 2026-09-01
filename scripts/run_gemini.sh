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

# Interaction
MAX_TURNS=10
LIMIT=""        # optional, e.g. 50
OFFSET=""       # optional, e.g. 0
SAVE_IMAGES=0   # 1 to save screenshots, 0 otherwise

# Gemini (OpenAI-compatible endpoint)
API_KEY="${GEMINI_API_KEY:-}"  # set via: export GEMINI_API_KEY=your_key
MODEL_NAME="gemini-3-pro-preview"
BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_RETRIES=5
REASONING_EFFORT="high"   # minimal|low|medium|high

# =========================
# DO NOT EDIT BELOW
# =========================

if [[ -z "${API_KEY}" ]]; then
  echo "Error: GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY=your_key" >&2
  exit 1
fi

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
  --model "openai"
  --api-key "${API_KEY}"
  --model-name "${MODEL_NAME}"
  --base-url "${BASE_URL}"
  --max-retries "${MAX_RETRIES}"
  --reasoning-effort "${REASONING_EFFORT}"
  --max-turns "${MAX_TURNS}"
)

if [[ -n "${RUN_NAME}" ]]; then
  CMD+=(--run-name "${RUN_NAME}")
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

echo "Running: ${CMD[*]}"
"${CMD[@]}"


