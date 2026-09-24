#!/usr/bin/env bash
# Sync F05-focus assets from GPU0 to GPU1.
#
# Default is a dry run.  Required environment variables:
#   GPU1_HOST   SSH target, e.g. user@gpu1-host
#   GPU1_ROOT   repository/workspace root on GPU1, e.g. /workspace/noise
#
# Optional:
#   GPU0_ROOT   source root, defaults to this repository worktree
#   INCLUDE_F05_CHECKPOINT=1  sync $F05_CHECKPOINT if set
#   INCLUDE_F05_VAL_LOGITS=1  sync artifacts/f05_focus/f05_val_logits.pt
#   INCLUDE_HP_MANIFEST=1     sync artifacts/f05_focus/hp_noise_manifest.csv
#   INCLUDE_CLEAN070=1        sync artifacts/f05_focus/clean070.csv
#   INCLUDE_F05_FOCUS_OUTPUTS=1 sync outputs/f05_focus (small JSON/CSV evidence)
set -euo pipefail

EXECUTE=0
if [[ "${1:-}" == "--execute" ]]; then
  EXECUTE=1
elif [[ "${1:-}" == "--dry-run" || -z "${1:-}" ]]; then
  EXECUTE=0
else
  echo "usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

GPU1_HOST="${GPU1_HOST:?set GPU1_HOST=user@gpu1-host}"
GPU1_ROOT="${GPU1_ROOT:?set GPU1_ROOT=/workspace/noise}"
GPU0_ROOT="${GPU0_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SSH_OPTS=${SSH_OPTS:--o ServerAliveInterval=30 -o ServerAliveCountMax=6}
RSYNC_OPTS=${RSYNC_OPTS:--a --partial --human-readable --info=progress2 --mkpath}

run_rsync() {
  local source="$1"
  local destination="$2"
  if [[ ! -e "$source" ]]; then
    echo "MISSING source: $source" >&2
    return 1
  fi
  local command=(rsync "${RSYNC_OPTS}" -e "ssh ${SSH_OPTS}" "$source" "$GPU1_HOST:$destination")
  if [[ "$EXECUTE" -eq 0 ]]; then
    printf 'DRY-RUN: '
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

run_rsync "$GPU0_ROOT/train/" "$GPU1_ROOT/train/"
run_rsync "$GPU0_ROOT/test/" "$GPU1_ROOT/test/"
run_rsync "$GPU0_ROOT/artifacts/stages/repechage/20260921/" \
  "$GPU1_ROOT/artifacts/stages/repechage/20260921/"
run_rsync "$GPU0_ROOT/outputs/rematch750_npu/RM_LP/seed42/checkpoints/best.pt" \
  "$GPU1_ROOT/outputs/rematch750_npu/RM_LP/seed42/checkpoints/best.pt"

if [[ "${INCLUDE_F05_CHECKPOINT:-0}" == "1" ]]; then
  F05_CHECKPOINT="${F05_CHECKPOINT:?set F05_CHECKPOINT when INCLUDE_F05_CHECKPOINT=1}"
  run_rsync "$F05_CHECKPOINT" "$GPU1_ROOT/outputs/f05_focus/F05/best.pt"
fi
if [[ "${INCLUDE_F05_VAL_LOGITS:-0}" == "1" ]]; then
  run_rsync "$GPU0_ROOT/artifacts/f05_focus/f05_val_logits.pt" \
    "$GPU1_ROOT/artifacts/f05_focus/f05_val_logits.pt"
fi
if [[ "${INCLUDE_HP_MANIFEST:-0}" == "1" ]]; then
  run_rsync "$GPU0_ROOT/artifacts/f05_focus/hp_noise_manifest.csv" \
    "$GPU1_ROOT/artifacts/f05_focus/hp_noise_manifest.csv"
fi
if [[ "${INCLUDE_CLEAN070:-0}" == "1" ]]; then
  run_rsync "$GPU0_ROOT/artifacts/f05_focus/clean070.csv" \
    "$GPU1_ROOT/artifacts/f05_focus/clean070.csv"
fi
if [[ "${INCLUDE_F05_FOCUS_OUTPUTS:-0}" == "1" ]]; then
  run_rsync "$GPU0_ROOT/outputs/f05_focus/" "$GPU1_ROOT/outputs/f05_focus/"
fi

echo "sync plan complete (execute=$EXECUTE)"
