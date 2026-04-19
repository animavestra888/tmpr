#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
source scripts/experiments/common_train_params.sh

: "${POLYGON_ADAPTER:=runs/hiertext_polygon_mlp_stage1_encoder/final/polygon_adapter.pt}"
: "${OUTPUT_DIR:=runs/hiertext_polygon_mlp_stage2_lora}"
: "${LEARNING_RATE:=2e-5}"
set_common_train_defaults
set_polygon_embedding_defaults
set_lora_defaults
set_reproducibility_args

if [[ ! -f "${POLYGON_ADAPTER}" ]]; then
  echo "Missing polygon adapter: ${POLYGON_ADAPTER}"
  echo "Run: bash scripts/experiments/train_poly_mlp_stage1_encoder.sh"
  exit 1
fi

"${PYTHON}" scripts/train_hiertext_paragraphs.py \
  --polygon-mode embedding \
  --embedding-geometry "${EMBEDDING_GEOMETRY}" \
  --polygon-encoder mlp \
  --polygon-adapter "${POLYGON_ADAPTER}" \
  --model-dir "${MODEL_DIR}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --eval-jsonl "${EVAL_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-eval-samples "${MAX_EVAL_SAMPLES}" \
  --max-pixels "${MAX_PIXELS}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --seed "${SEED}" \
  --data-seed "${DATA_SEED}" \
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
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --lora-target-modules "${LORA_TARGET_MODULES}" \
  --gradient-checkpointing \
  "${TRAIN_EXTRA_ARGS[@]}"
