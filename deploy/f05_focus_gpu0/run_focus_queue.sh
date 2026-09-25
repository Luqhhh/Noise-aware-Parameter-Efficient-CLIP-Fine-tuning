#!/usr/bin/env bash
# GPU0 focus queue: A0 -> P0 -> D1 O3 adapter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
STAGE_DIR="${STAGE_DIR:-/home/lux1/noise/artifacts/stages/repechage/20260921}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/lux1/noise/train}"
TEST_ROOT="${TEST_ROOT:-/home/lux1/noise/test}"
C0_CHECKPOINT="${C0_CHECKPOINT:-$REPO_ROOT/outputs/f05_focus/C0_F05_CUDA/seed42/checkpoints/best.pt}"
CLEAN070="${CLEAN070:-$REPO_ROOT/artifacts/f05_focus/clean070.csv}"
VAL_CSV="${VAL_CSV:-$STAGE_DIR/val_dev.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/f05_focus}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_ROOT/artifacts/f05_focus}"
DEVICE="${DEVICE:-cuda:0}"
SUMMARY="${SUMMARY:-$REPO_ROOT/results/f05_focus_summary.csv}"
LOG="${LOG:-$OUTPUT_ROOT/gpu0_queue.log}"

export PYTHONPATH="$REPO_ROOT/reproducibility/aegis_f1${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT" "$ARTIFACT_DIR"
exec >>"$LOG" 2>&1

step() { echo; echo "===== $(date '+%F %T') $* ====="; }

step "check required assets"
for path in "$C0_CHECKPOINT" "$CLEAN070" "$VAL_CSV"; do
  [[ -f "$path" ]] || { echo "missing: $path" >&2; exit 2; }
done

step "A0 attention-local probe"
python3 "$REPO_ROOT/scripts/run_a0_m1_probe.py" \
  --checkpoint "$C0_CHECKPOINT" \
  --device "$DEVICE" \
  --crop-size 224 \
  --top-patches 5 \
  --cache-output "$ARTIFACT_DIR/f05_attention_cache.pt" \
  --output-dir "$OUTPUT_ROOT/A0_M1" \
  --summary "$SUMMARY"

step "P0 center validation logits"
python3 -m aegis_clip.cli.cache_validation_logits \
  --checkpoint "$C0_CHECKPOINT" \
  --view-mode center \
  --output "$ARTIFACT_DIR/f05_val_logits.pt" \
  --batch-size 128 \
  --num-workers 4

step "P0 prior probe"
python3 "$REPO_ROOT/scripts/run_p0_prior_probe.py" \
  --validation-logits "$ARTIFACT_DIR/f05_val_logits.pt" \
  --strengths 0.25,0.50,0.75,1.00 \
  --output-dir "$OUTPUT_ROOT/P0_PRIOR" \
  --bias-output "$ARTIFACT_DIR/prior_bias.pt" \
  --summary "$SUMMARY"

step "D1 M1 crop160 reference"
python3 -m aegis_clip.cli.cache_validation_logits \
  --checkpoint "$C0_CHECKPOINT" \
  --view-mode attention_local_global \
  --crop-size 160 \
  --top-patches 5 \
  --output "$ARTIFACT_DIR/f05_val_m1_crop160.pt" \
  --batch-size 128 \
  --num-workers 4

step "D1 train cache"
python3 -m aegis_clip.cli.cache_local_adapter_features \
  --checkpoint "$C0_CHECKPOINT" \
  --split-csv "$CLEAN070" \
  --output "$ARTIFACT_DIR/D1_train_cache_crop160.pt" \
  --crop-size 160 \
  --top-patches 5 \
  --batch-size 64 \
  --num-workers 4

step "D1 validation cache"
python3 -m aegis_clip.cli.cache_local_adapter_features \
  --checkpoint "$C0_CHECKPOINT" \
  --split-csv "$VAL_CSV" \
  --output "$ARTIFACT_DIR/D1_val_cache_crop160.pt" \
  --crop-size 160 \
  --top-patches 5 \
  --batch-size 64 \
  --num-workers 4

step "D1 O3 adapter training"
TRAIN_COUNT=$(python3 -c 'import pandas as pd; print(len(pd.read_csv("'"$CLEAN070"'")))')
python3 -m aegis_clip.cli.train_local_feature_adapter \
  --parent-checkpoint "$C0_CHECKPOINT" \
  --train-cache "$ARTIFACT_DIR/D1_train_cache_crop160.pt" \
  --validation-cache "$ARTIFACT_DIR/D1_val_cache_crop160.pt" \
  --output-dir "$OUTPUT_ROOT/D1_O3" \
  --center-reference "$ARTIFACT_DIR/f05_val_logits.pt" \
  --m1-reference "$ARTIFACT_DIR/f05_val_m1_crop160.pt" \
  --expected-train-samples "$TRAIN_COUNT" \
  --bottleneck-dim 32 \
  --residual-scale 0.25 \
  --dropout 0.1 \
  --batch-size 1024 \
  --max-epochs 20 \
  --patience 5 \
  --device "$DEVICE"

step "GPU0 focus queue complete"
python3 - "$OUTPUT_ROOT/gpu0_queue_status.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"status":"completed","queue":["A0","P0","D1"]}, indent=2)+"\n")
PY
