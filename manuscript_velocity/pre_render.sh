#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PIPELINE_ARGS=()
if [[ "${CAPACITY_SEM_DEMO:-}" == "1" ]]; then
    PIPELINE_ARGS+=(--demo)
fi

if [[ -z "${CAPACITY_SEM_SKIP_PIPELINE:-}" ]]; then
    echo "=== Running Capacity-SEM pipeline ==="
    python "$ROOT_DIR/src/pipeline.py" run_all "${PIPELINE_ARGS[@]}"
    echo "=== Pipeline complete ==="
fi

DATA_DIR="$SCRIPT_DIR/data"
FIG_DIR="$SCRIPT_DIR/figures"
DIAG_DIR="$ROOT_DIR/data_work/diagnostics"
ROOT_FIG_DIR="$ROOT_DIR/figures"

mkdir -p "$DATA_DIR" "$FIG_DIR"

shopt -s nullglob

rm -f "$DATA_DIR"/*.csv
if [[ -d "$DIAG_DIR" ]]; then
    cp "$DIAG_DIR"/*.csv "$DATA_DIR"/
fi

rm -f "$FIG_DIR"/*.png
if [[ -d "$ROOT_FIG_DIR" ]]; then
    cp "$ROOT_FIG_DIR"/*.png "$FIG_DIR"/
fi

shopt -u nullglob
