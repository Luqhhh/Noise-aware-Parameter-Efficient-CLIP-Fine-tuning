#!/usr/bin/env bash
# Run the 750-class OOF/quality asset generation on GPU1, immediately.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TRAIN_CSV="${TRAIN_CSV:-$REPO_ROOT/artifacts/stages/repechage/20260921/train_dev.csv}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/artifacts/stages/repechage/20260921/features}"
TRAIN_ROOT="${TRAIN_ROOT:-$REPO_ROOT/train}"
OOF_DIR="${OOF_DIR:-$REPO_ROOT/outputs/f05_focus/oof}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_ROOT/artifacts/f05_focus}"
DEVICE="${DEVICE:-cuda:0}"
LOG="${LOG:-$REPO_ROOT/outputs/f05_focus/oof_quality_gpu1.log}"

mkdir -p "$(dirname "$LOG")"
echo "[gpu1-oof] repo=$REPO_ROOT"
echo "[gpu1-oof] train_csv=$TRAIN_CSV"
echo "[gpu1-oof] cache_dir=$CACHE_DIR"
echo "[gpu1-oof] output=$OOF_DIR"
echo "[gpu1-oof] device=$DEVICE"

WAIT_FOR_C0=0 \
SKIP_L05_WAIT=1 \
TRAIN_CSV="$TRAIN_CSV" \
CACHE_DIR="$CACHE_DIR" \
TRAIN_ROOT="$TRAIN_ROOT" \
OOF_DIR="$OOF_DIR" \
ARTIFACT_DIR="$ARTIFACT_DIR" \
DEVICE="$DEVICE" \
bash "$REPO_ROOT/scripts/run_f05_focus_quality_chain.sh" 2>&1 | tee "$LOG"
