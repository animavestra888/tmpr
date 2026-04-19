#!/bin/bash

# Shared defaults for HierText training experiments.
# Any value can still be overridden before launch, for example:
# LEARNING_RATE=1e-4 EMBEDDING_GEOMETRY=minrect bash scripts/experiments/train_poly_mlp.sh

set_common_train_defaults() {
  : "${PYTHON:=python}"
  : "${MODEL_DIR:=models/Qwen3-VL-2B-Instruct}"
  : "${TRAIN_JSONL:=data/hiertext/jsonl_max300/train.jsonl}"
  : "${EVAL_JSONL:=data/hiertext/jsonl_max300/validation.jsonl}"
  : "${DEEPSPEED_CONFIG:=configs/deepspeed_zero2.json}"

  : "${DEVICE:=auto}"
  : "${DTYPE:=bfloat16}"
  : "${SEED:=42}"
  : "${DATA_SEED:=42}"
  : "${FULL_DETERMINISM:=0}"
  : "${MAX_PIXELS:=327680}"
  : "${MAX_EVAL_SAMPLES:=128}"
  export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"

  : "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
  : "${PER_DEVICE_EVAL_BATCH_SIZE:=1}"
  : "${GRADIENT_ACCUMULATION_STEPS:=8}"
  : "${LEARNING_RATE:=2e-4}"
  : "${WEIGHT_DECAY:=0.01}"
  : "${NUM_TRAIN_EPOCHS:=1.0}"
  : "${MAX_STEPS:=-1}"
  : "${LOGGING_STEPS:=10}"
  : "${SAVE_STEPS:=200}"
  : "${EVAL_STEPS:=200}"
  : "${SAVE_TOTAL_LIMIT:=2}"
}

set_reproducibility_args() {
  TRAIN_EXTRA_ARGS=()
  if [[ "${FULL_DETERMINISM}" == "1" || "${FULL_DETERMINISM}" == "true" ]]; then
    TRAIN_EXTRA_ARGS+=(--full-determinism)
  fi
}

set_polygon_embedding_defaults() {
  : "${EMBEDDING_GEOMETRY:=bbox_corners}"
  : "${POLYGON_DROPOUT:=0.1}"
}

set_transformer_encoder_defaults() {
  : "${TRANSFORMER_D_MODEL:=256}"
  : "${TRANSFORMER_LAYERS:=2}"
  : "${TRANSFORMER_HEADS:=4}"
  : "${TRANSFORMER_FFN_DIM:=1024}"
  : "${TRANSFORMER_MAX_POSITIONS:=2048}"
}

set_lora_defaults() {
  : "${LORA_R:=16}"
  : "${LORA_ALPHA:=32}"
  : "${LORA_DROPOUT:=0.05}"
  : "${LORA_TARGET_MODULES:=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
}

set_detection_head_defaults() {
  : "${POLY_DET_LOSS_WEIGHT:=1.0}"
  : "${POLY_DET_LOSS_TYPE:=l1}"
  : "${POLY_DET_SOURCE:=embedding}"
  : "${POLY_DET_DROPOUT:=0.0}"
}
