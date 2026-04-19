#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

HIER_ROOT="data/hiertext"
HIER_REPO="${HIER_ROOT}/repo"
JSONL_DIR="${HIER_ROOT}/jsonl"
PYTHON="python"

mkdir -p "${HIER_ROOT}"

if [[ ! -d "${HIER_REPO}/.git" ]]; then
  git clone https://github.com/google-research-datasets/hiertext.git "${HIER_REPO}"
else
  echo "HierText repo already exists: ${HIER_REPO}"
fi

for split in train validation test; do
  archive="${HIER_ROOT}/${split}.tgz"
  image_dir="${HIER_ROOT}/${split}"

  if [[ ! -f "${archive}" ]]; then
    if ! command -v aws >/dev/null 2>&1; then
      echo "AWS CLI is required to download missing archive: ${archive}"
      echo "Install it first: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
      exit 1
    fi
    aws s3 --no-sign-request cp "s3://open-images-dataset/ocr/${split}.tgz" "${archive}"
  else
    echo "Archive already exists: ${archive}"
  fi

  if [[ ! -d "${image_dir}" ]]; then
    tar -xzf "${archive}" -C "${HIER_ROOT}"
  else
    echo "Image directory already exists: ${image_dir}"
  fi
done

"${PYTHON}" scripts/export_hiertext_jsonl.py \
  --hiertext-root "${HIER_ROOT}" \
  --gt-root "${HIER_REPO}/gt" \
  --output-dir "${JSONL_DIR}" \
  --path-root "." \
  --splits train validation test
