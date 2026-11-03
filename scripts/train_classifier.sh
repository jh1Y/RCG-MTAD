#!/usr/bin/env bash
# The old tire-only command now trains the complete model from raw samples.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_model.sh" "$@"
