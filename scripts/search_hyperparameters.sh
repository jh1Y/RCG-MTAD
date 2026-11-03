#!/usr/bin/env bash
# Shared implementation for the two public search_*.sh entry points.
set -euo pipefail
STAGE="${1:?Expected road or joint}"
shift
if [[ "${STAGE}" != "road" && "${STAGE}" != "joint" ]]; then
  echo "Expected stage road or joint, got ${STAGE}" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Discrete candidates within the supplied table ranges (324 combinations).
EPOCHS_GRID="${EPOCHS_GRID:-50,100,150,200}"
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-32,64,128}"
WEIGHT_DECAY_GRID="${WEIGHT_DECAY_GRID:-1e-5,1e-4,1e-3}"
INITIAL_LR_GRID="${INITIAL_LR_GRID:-1e-5,1e-4,1e-3}"
MIN_LR_GRID="${MIN_LR_GRID:-0,1e-6,1e-5}"

source_args=()
if [[ -n "${SAMPLES:-}" ]]; then
  source_args+=(--samples "${SAMPLES}")
else
  : "${DATA_DIR:?Set DATA_DIR to the existing four-state TTSPD dataset root, or set SAMPLES}"
  source_args+=(--data-dir "${DATA_DIR}")
fi
if [[ -n "${DATA_ROOT:-}" ]]; then
  source_args+=(--data-root "${DATA_ROOT}")
fi

options=()
if [[ "${RESUME:-1}" == "0" ]]; then options+=(--no-resume); fi
if [[ "${CONTINUE_ON_ERROR:-1}" == "0" ]]; then options+=(--no-continue-on-error); fi
if [[ "${INCLUDE_REFERENCE:-1}" == "0" ]]; then options+=(--no-include-reference); fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then options+=(--dry-run); fi
if [[ "${RECURSIVE_IMAGES:-0}" == "1" ]]; then options+=(--recursive-images); fi

if [[ "${STAGE}" == "joint" ]]; then
  if [[ -n "${ROAD_CHECKPOINT:-}" ]]; then
    options+=(--road-checkpoint "${ROAD_CHECKPOINT}")
  elif [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
    options+=(--from-scratch)
  else
    options+=(--road-search-result "${ROAD_SEARCH_RESULT:-${PROJECT_DIR}/outputs/hyperparameter_search/road/best.json}")
  fi
  if [[ -n "${INITIAL_TIRE_CHECKPOINT:-}" ]]; then
    options+=(--initial-tire-checkpoint "${INITIAL_TIRE_CHECKPOINT}")
  fi
  if [[ "${FREEZE_ROAD_ENCODER:-0}" == "1" ]]; then options+=(--freeze-road-encoder); fi
fi

"${PYTHON_BIN}" -m tire_anomaly_system.cli.search_hyperparameters \
  --stage "${STAGE}" "${source_args[@]}" \
  --sensor-subdir "${SENSOR_SUBDIR:-sensor}" \
  --config "${CONFIG:-${PROJECT_DIR}/configs/rcg_mtad.json}" \
  --output-dir "${OUTPUT_DIR:-${PROJECT_DIR}/outputs/hyperparameter_search/${STAGE}}" \
  --mode "${SEARCH_MODE:-random}" --trials "${TRIALS:-30}" \
  --epochs-grid "${EPOCHS_GRID}" --batch-size-grid "${BATCH_SIZE_GRID}" \
  --weight-decay-grid "${WEIGHT_DECAY_GRID}" --lr-grid "${INITIAL_LR_GRID}" --min-lr-grid "${MIN_LR_GRID}" \
  --search-seed "${SEARCH_SEED:-42}" --seed "${SEED:-42}" \
  --road-loss-weight "${ROAD_LOSS_WEIGHT:-1.0}" \
  --num-workers "${NUM_WORKERS:-0}" --device "${DEVICE:-auto}" "${options[@]}" "$@"
