#!/usr/bin/env bash
# GPU0 D2 unit: F05 + PartTokenResidualAdapter (PTA).
#
# Same frozen parent, same clean070 subset and same local view as D1. Fixed
# parameters come from docs/f05_focus_protocol_20260924.md section 5.2 and
# configs/f05_focus/D2_pta.yaml: bottleneck 64, residual scale 0.25, dropout
# 0.1, part_top_patches 8, part_temperature 0.07, backbone frozen.
#
# CACHE_BATCH_SIZE must match the batch size used to build the reference caches
# (artifacts/f05_focus/f05_val_logits.pt and f05_val_m1_crop160.pt). The trainer
# enforces this against cache execution metadata and then runs a fail-closed
# bit-exact reference audit; see
# results/f05_focus_d1_reference_audit_failure_20260925.json for why a mismatch
# fails.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
STAGE_DIR="${STAGE_DIR:-/home/lux1/noise/artifacts/stages/repechage/20260921}"
C0_CHECKPOINT="${C0_CHECKPOINT:-$REPO_ROOT/outputs/f05_focus/C0_F05_CUDA/seed42/checkpoints/best.pt}"
CLEAN070="${CLEAN070:-$REPO_ROOT/artifacts/f05_focus/clean070.csv}"
VAL_CSV="${VAL_CSV:-$STAGE_DIR/val_dev.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/f05_focus}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_ROOT/artifacts/f05_focus}"
DEVICE="${DEVICE:-cuda:0}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
LOG="${LOG:-$OUTPUT_ROOT/gpu0_d2_queue.log}"
# Optional: block until another GPU0 queue releases the GPU. The value is a file
# holding the PID of the queue to wait for.
WAIT_PID_FILE="${WAIT_PID_FILE:-}"

export PYTHONPATH="$REPO_ROOT/reproducibility/aegis_f1${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT" "$ARTIFACT_DIR"
exec >>"$LOG" 2>&1

step() { echo; echo "===== $(date '+%F %T') $* ====="; }

step "check required assets"
for path in "$C0_CHECKPOINT" "$CLEAN070" "$VAL_CSV" \
            "$ARTIFACT_DIR/f05_val_logits.pt" "$ARTIFACT_DIR/f05_val_m1_crop160.pt"; do
  [[ -f "$path" ]] || { echo "missing: $path" >&2; exit 2; }
done

if [[ -n "$WAIT_PID_FILE" && -f "$WAIT_PID_FILE" ]]; then
  WAIT_PID="$(cat "$WAIT_PID_FILE")"
  step "waiting for pid $WAIT_PID to release the GPU"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  step "pid $WAIT_PID released the GPU"
fi

step "D2 train cache (part tokens)"
python3 -m aegis_clip.cli.cache_part_token_adapter_features \
  --checkpoint "$C0_CHECKPOINT" \
  --split-csv "$CLEAN070" \
  --output "$ARTIFACT_DIR/D2_train_cache_crop160.pt" \
  --crop-size 160 \
  --top-patches 5 \
  --part-top-patches 8 \
  --part-temperature 0.07 \
  --batch-size "$CACHE_BATCH_SIZE" \
  --num-workers 4

step "D2 validation cache (part tokens)"
python3 -m aegis_clip.cli.cache_part_token_adapter_features \
  --checkpoint "$C0_CHECKPOINT" \
  --split-csv "$VAL_CSV" \
  --output "$ARTIFACT_DIR/D2_val_cache_crop160.pt" \
  --crop-size 160 \
  --top-patches 5 \
  --part-top-patches 8 \
  --part-temperature 0.07 \
  --batch-size "$CACHE_BATCH_SIZE" \
  --num-workers 4

step "D2 PTA adapter training"
TRAIN_COUNT=$(python3 -c 'import pandas as pd; print(len(pd.read_csv("'"$CLEAN070"'")))')
python3 -m aegis_clip.cli.train_part_token_adapter \
  --parent-checkpoint "$C0_CHECKPOINT" \
  --train-cache "$ARTIFACT_DIR/D2_train_cache_crop160.pt" \
  --validation-cache "$ARTIFACT_DIR/D2_val_cache_crop160.pt" \
  --output-dir "$OUTPUT_ROOT/D2_PTA" \
  --center-reference "$ARTIFACT_DIR/f05_val_logits.pt" \
  --m1-reference "$ARTIFACT_DIR/f05_val_m1_crop160.pt" \
  --expected-train-samples "$TRAIN_COUNT" \
  --expected-cache-batch-size "$CACHE_BATCH_SIZE" \
  --bottleneck-dim 64 \
  --residual-scale 0.25 \
  --dropout 0.1 \
  --batch-size 1024 \
  --max-epochs 20 \
  --patience 5 \
  --device "$DEVICE"

step "GPU0 D2 unit complete"
python3 - "$OUTPUT_ROOT/gpu0_d2_status.json" "$CACHE_BATCH_SIZE" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "completed",
            "unit": "D2",
            "adapter": "PartTokenResidualAdapter",
            "bottleneck_dim": 64,
            "residual_scale": 0.25,
            "dropout": 0.1,
            "part_top_patches": 8,
            "part_temperature": 0.07,
            "cache_batch_size": int(sys.argv[2]),
        },
        indent=2,
    )
    + "\n"
)
PY
