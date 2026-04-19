#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PYTHON="python"
MODEL_DIR="models/Qwen3-VL-2B-Instruct"
CHECKPOINT_DIR="runs/hiertext_text_bbox_qwen3vl/final"
TEST_JSONL="data/hiertext/jsonl_max300/test.jsonl"
OUTPUT_JSONL="runs/test_predictions_bbox_2d.jsonl"

DEVICE="auto"
DTYPE="bfloat16"
MAX_PIXELS=50176
MAX_SAMPLES=50
MAX_NEW_TOKENS=192

"${PYTHON}" scripts/generate_hiertext_predictions.py \
  --polygon-mode text \
  --model-dir "${MODEL_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --jsonl-path "${TEST_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-pixels "${MAX_PIXELS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"
