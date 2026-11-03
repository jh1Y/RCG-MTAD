#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${DATA_DIR:?Set DATA_DIR to the TTSPD dataset root}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
checkpoint_args=()
if [[ -n "${ROAD_CHECKPOINT:-}" ]]; then
  checkpoint_args+=(--road-checkpoint "${ROAD_CHECKPOINT}")
fi
if [[ -n "${INITIAL_TIRE_CHECKPOINT:-}" ]]; then
  checkpoint_args+=(--initial-tire-checkpoint "${INITIAL_TIRE_CHECKPOINT}")
fi
"${PYTHON_BIN}" -m tire_anomaly_system.cli.train_model \
  --data-dir "${DATA_DIR}" --sensor-subdir "${SENSOR_SUBDIR:-sensor}" \
  --config "${CONFIG:-${PROJECT_DIR}/configs/rcg_mtad.json}" "${checkpoint_args[@]}" \
  --output-dir "${OUTPUT_DIR:-${PROJECT_DIR}/outputs/rcg_mtad}" \
  --epochs "${EPOCHS:-200}" --patience "${PATIENCE:-20}" \
  --batch-size "${BATCH_SIZE:-32}" --lr "${LEARNING_RATE:-1e-3}" \
  --road-lr "${ROAD_LEARNING_RATE:-1e-4}" --road-loss-weight "${ROAD_LOSS_WEIGHT:-1.0}" \
  --seed "${SEED:-42}" --num-workers "${NUM_WORKERS:-0}" --device "${DEVICE:-auto}" "$@"
