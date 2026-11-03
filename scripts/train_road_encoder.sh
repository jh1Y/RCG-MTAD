#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${DATA_DIR:?Set DATA_DIR to the TTSPD dataset root}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m tire_anomaly_system.cli.train_road_encoder \
  --data-dir "${DATA_DIR}" --sensor-subdir "${SENSOR_SUBDIR:-sensor}" \
  --config "${CONFIG:-${PROJECT_DIR}/configs/rcg_mtad.json}" \
  --output-dir "${OUTPUT_DIR:-${PROJECT_DIR}/outputs/road_encoder}" \
  --epochs "${EPOCHS:-200}" --patience "${PATIENCE:-20}" \
  --batch-size "${BATCH_SIZE:-32}" --lr "${LEARNING_RATE:-1e-3}" \
  --seed "${SEED:-42}" --num-workers "${NUM_WORKERS:-0}" --device "${DEVICE:-auto}" "$@"
