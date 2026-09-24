#!/bin/bash
# Sequential lane dispatcher for REMATCH750_SEARCH_V5.
# Usage: run_rematch750_search_v5_lane_queue.sh <wait_pid> <physical_npu> <trial...>
set -u
WAIT_PID="$1"; DEVICE="$2"; shift 2
TRIALS=("$@")
ROOT="/workspace/noise-v5"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
cd "$ROOT" || exit 1
export PYTHONPATH=reproducibility/aegis_f1
export OMP_NUM_THREADS=4
export REMATCH_CODE_COMMIT="${REMATCH_CODE_COMMIT:-unknown}"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
LANE_LOG="outputs/rematch750_search_v5/launch/lane_npu${DEVICE}.log"
mkdir -p outputs/rematch750_search_v5/launch
for TRIAL in "${TRIALS[@]}"; do
  echo "=== lane npu${DEVICE} trial ${TRIAL} $(date -u +%FT%TZ)" >> "$LANE_LOG"
  /workspace/noise-npu-venv/bin/python scripts/run_rematch750_search_v5_queue.py \
    --trials "$TRIAL" --device npu:0 --workers 8 --prefetch-factor 2 \
    --lock "outputs/rematch750_search_v5/queue/lane_npu${DEVICE}_${TRIAL}.lock" \
    >> "$LANE_LOG" 2>&1 || true
done
