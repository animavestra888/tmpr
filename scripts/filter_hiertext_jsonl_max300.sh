#!/bin/bash
set -euo pipefail

PYTHON="python"
INPUT_DIR="data/hiertext/jsonl"
OUTPUT_DIR="data/hiertext/jsonl_max300"
MAX_LINES=300

"${PYTHON}" scripts/filter_hiertext_jsonl_by_lines.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-lines "${MAX_LINES}" \
  --splits train validation test
