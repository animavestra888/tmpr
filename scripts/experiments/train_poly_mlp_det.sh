#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PYTHON="python"
MODEL_DIR="models/Qwen3-VL-2B-Instruct"
TRAIN_JSONL="data/hiertext/jsonl_max300/train.jsonl"
EVAL_JSONL="data/hiertext/jsonl_max300/validation.jsonl"
OUTPUT_DIR="runs/hiertext_polygon_mlp_det_qwen3vl"
DEEPSPEED_CONFIG="configs/deepspeed_zero2.json"

DEVICE="auto"
DTYPE="bfloat16"
MAX_PIXELS=327680
MAX_EVAL_SAMPLES=128

PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.01
NUM_TRAIN_EPOCHS=1.0
MAX_STEPS=-1
LOGGING_STEPS=10
SAVE_STEPS=200
EVAL_STEPS=200
SAVE_TOTAL_LIMIT=2
POLYGON_DROPOUT=0.1
POLY_DET_LOSS_WEIGHT=1.0
POLY_DET_LOSS_TYPE="l1"
POLY_DET_SOURCE="embedding"
POLY_DET_DROPOUT=0.0

"${PYTHON}" scripts/train_hiertext_paragraphs.py \
  --polygon-mode embedding \
  --polygon-encoder mlp \
  --model-dir "${MODEL_DIR}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --eval-jsonl "${EVAL_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-eval-samples "${MAX_EVAL_SAMPLES}" \
  --max-pixels "${MAX_PIXELS}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --logging-steps "${LOGGING_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --eval-steps "${EVAL_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --polygon-dropout "${POLYGON_DROPOUT}" \
  --poly-det-loss-weight "${POLY_DET_LOSS_WEIGHT}" \
  --poly-det-loss-type "${POLY_DET_LOSS_TYPE}" \
  --poly-det-source "${POLY_DET_SOURCE}" \
  --poly-det-dropout "${POLY_DET_DROPOUT}" \
  --gradient-checkpointing
