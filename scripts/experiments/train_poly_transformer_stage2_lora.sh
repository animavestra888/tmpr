#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

PYTHON="python"
MODEL_DIR="models/Qwen3-VL-2B-Instruct"
TRAIN_JSONL="data/hiertext/jsonl_max300/train.jsonl"
EVAL_JSONL="data/hiertext/jsonl_max300/validation.jsonl"
POLYGON_ADAPTER="runs/hiertext_polygon_transformer_stage1_encoder/final/polygon_adapter.pt"
OUTPUT_DIR="runs/hiertext_polygon_transformer_stage2_lora"
DEEPSPEED_CONFIG="configs/deepspeed_zero2.json"

DEVICE="auto"
DTYPE="bfloat16"
MAX_PIXELS=327680
MAX_EVAL_SAMPLES=128

PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
NUM_TRAIN_EPOCHS=1.0
MAX_STEPS=-1
LOGGING_STEPS=10
SAVE_STEPS=200
EVAL_STEPS=200
SAVE_TOTAL_LIMIT=2
POLYGON_DROPOUT=0.1
TRANSFORMER_D_MODEL=256
TRANSFORMER_LAYERS=2
TRANSFORMER_HEADS=4
TRANSFORMER_FFN_DIM=1024
TRANSFORMER_MAX_POSITIONS=2048
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

if [[ ! -f "${POLYGON_ADAPTER}" ]]; then
  echo "Missing polygon adapter: ${POLYGON_ADAPTER}"
  echo "Run: bash scripts/experiments/train_poly_transformer_stage1_encoder.sh"
  exit 1
fi

"${PYTHON}" scripts/train_hiertext_paragraphs.py \
  --polygon-mode embedding \
  --polygon-encoder transformer \
  --polygon-adapter "${POLYGON_ADAPTER}" \
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
  --transformer-d-model "${TRANSFORMER_D_MODEL}" \
  --transformer-layers "${TRANSFORMER_LAYERS}" \
  --transformer-heads "${TRANSFORMER_HEADS}" \
  --transformer-ffn-dim "${TRANSFORMER_FFN_DIM}" \
  --transformer-max-positions "${TRANSFORMER_MAX_POSITIONS}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --lora-target-modules "${LORA_TARGET_MODULES}" \
  --gradient-checkpointing
