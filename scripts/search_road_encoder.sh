#!/usr/bin/env bash
# RMPNet: independently select road-encoder hyperparameters.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/search_hyperparameters.sh" road "$@"
