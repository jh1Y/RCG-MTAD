#!/usr/bin/env bash
# RCG-MTAD: select complete-model hyperparameters using the chosen road weights.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/search_hyperparameters.sh" joint "$@"
