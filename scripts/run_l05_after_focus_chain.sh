#!/usr/bin/env bash
# Wait for GPU0 C0, then run L05 before the OOF/quality chain.
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
rm -f "$OUTPUT_ROOT/L05_DONE" "$OUTPUT_ROOT/L05_FAILED"

echo "[l05-wait] waiting for C0 to finish"
while pgrep -f "resume_c0_gpu0.py" >/dev/null || \
      pgrep -f "aegis_clip.cli.rematch train --config .*C0_cuda_0.yaml" >/dev/null; do
  sleep 60
done
echo "[l05-wait] C0 finished; starting L05 before OOF/quality"

set +e
python3 "$REPO_ROOT/scripts/run_l05_local_retrain.py" \
  >> "$OUTPUT_ROOT/l05_train.log" 2>&1 \
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
L05_STATUS=$?
set -e

if [[ "$L05_STATUS" -eq 0 ]]; then
  touch "$OUTPUT_ROOT/L05_DONE"
  echo "[l05-wait] L05 completed; OOF/quality chain may proceed"
else
  touch "$OUTPUT_ROOT/L05_FAILED"
  echo "[l05-wait] L05 failed with status=$L05_STATUS; OOF/quality chain may proceed" >&2
fi
exit "$L05_STATUS"
