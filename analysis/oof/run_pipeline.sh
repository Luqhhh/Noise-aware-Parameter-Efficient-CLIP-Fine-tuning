#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
PYTHON=/home/clairvoyant/.venvs/noise-clip/bin/python
cd "$ROOT"

run_with_heartbeat() {
  local name="$1"
  local log_path="$2"
  shift 2
  "$@" > "$log_path" 2>&1 &
  local child_pid=$!
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep 30
    if kill -0 "$child_pid" 2>/dev/null; then
      echo "$name running; log_bytes=$(stat -c %s "$log_path")"
    fi
  done
  local status=0
  wait "$child_pid" || status=$?
  echo "$name exit_status=$status"
  if [[ "$status" -ne 0 ]]; then
    tail -20 "$log_path"
    return "$status"
  fi
  echo "$name complete"
}

mkdir -p outputs/phase3/oof

run_with_heartbeat \
  feature_cache \
  outputs/phase3/oof/cache_features.log \
  "$PYTHON" scripts/cache_features.py --config configs/oof_cache.yaml

run_with_heartbeat \
  three_fold_oof \
  outputs/phase3/oof/run_oof.log \
  "$PYTHON" -m analysis.oof.run_oof \
    --assignments outputs/phase3/oof/fold_assignments.csv \
    --cache-dir cache/preliminary/clip_vit_b32_openai \
    --duplicate-scan outputs/duplicate_scan.json \
    --output-dir outputs/phase3/oof \
    --epochs 50 \
    --q 0.5 \
    --seed 42 \
    --device cuda

echo "OOF pipeline complete"