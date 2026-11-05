#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/Datasets}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m tire_anomaly_system.cli.infer_model \
  --data-dir "${DATA_DIR}" \
  --checkpoint "${CHECKPOINT:-${PROJECT_DIR}/outputs/rcg_mtad/rcg_mtad.pt}" \
  --output-dir "${OUTPUT_DIR:-${PROJECT_DIR}/outputs/inference}" --split "${SPLIT:-test}" \
  --batch-size "${BATCH_SIZE:-32}" --num-workers "${NUM_WORKERS:-0}" --device "${DEVICE:-auto}" "$@"
