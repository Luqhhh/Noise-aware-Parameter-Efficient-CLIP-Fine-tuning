#!/usr/bin/env bash
# Wait for the GPU0 C0 run and the OOF/quality chain, then start L05 locally.
#
# L05: R02 384px global full fine-tune plus confidence-gated attention-local
# supervision from epoch 5 with local_supervision_weight = 0.25.
# The Python runner auto-falls back through microbatch candidates on OOM and
# exports the final submission package to the Windows desktop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
STAGE_DIR="${STAGE_DIR:-/home/lux1/noise/artifacts/stages/repechage/20260921}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/lux1/noise/train}"
TEST_ROOT="${TEST_ROOT:-/home/lux1/noise/test}"
RM_LP_CHECKPOINT="${RM_LP_CHECKPOINT:-$REPO_ROOT/outputs/f05_focus/local_parent/RM_LP/seed42/checkpoints/best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/f05_focus_l05}"
RUNTIME_DIR="${RUNTIME_DIR:-$OUTPUT_ROOT/_runtime_configs}"
DESKTOP_DIR="${DESKTOP_DIR:-/mnt/c/Users/lqh22/Desktop}"
DEVICE="${DEVICE:-cuda:0}"
EXPERIMENT_ID="${EXPERIMENT_ID:-RM_V5_L05_CUDA_LOCAL}"
MICROBATCH_CANDIDATES="${MICROBATCH_CANDIDATES:-8,4,2}"

mkdir -p "$OUTPUT_ROOT"
echo "[l05-wait] waiting for C0 and OOF/quality chain to finish"
while pgrep -f "resume_c0_gpu0.py" >/dev/null || \\
      pgrep -f "run_f05_focus_quality_chain.sh" >/dev/null; do
  sleep 60
done

echo "[l05-wait] GPU is expected free; starting L05"
exec python3 "$REPO_ROOT/scripts/run_l05_local_retrain.py" \
  --auto-microbatch \
  --microbatch-candidates "$MICROBATCH_CANDIDATES" \
  --device "$DEVICE" \
  --stage-dir "$STAGE_DIR" \
  --train-root "$TRAIN_ROOT" \
  --test-root "$TEST_ROOT" \
  --rm-lp-checkpoint "$RM_LP_CHECKPOINT" \
  --output-root "$OUTPUT_ROOT" \
  --runtime-dir "$RUNTIME_DIR" \
  --experiment-id "$EXPERIMENT_ID" \
  --desktop-dir "$DESKTOP_DIR"
