#!/bin/bash
# Sequential lane dispatcher: ET trials first, then V5 trials.
# Usage: run_et_then_v5_lane.sh <wait_pid> <physical_npu> <"ET01 ET02"> <"DF01 DF02"> <"Q01 V04">
set -u
WAIT_PID=${1:?wait_pid}
DEVICE=${2:?physical_npu}
ET_TRIALS=${3:-}
DF_TRIALS=${4:-}
V5_TRIALS=${5:-}
ET_ROOT=/workspace/noise-et
V5_ROOT=/workspace/noise-v5
PY=/workspace/noise-npu-venv/bin/python

while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
export OMP_NUM_THREADS=4
export REMATCH_CODE_COMMIT="${REMATCH_CODE_COMMIT:-xjn/rematch750-f05-transfer-aligned@d850291}"

if [ -n "$ET_TRIALS" ]; then
  cd "$ET_ROOT" || exit 1
  export PYTHONPATH="$ET_ROOT/reproducibility/aegis_f1"
  mkdir -p outputs/xjn/rematch750_f05_transfer/launch
  ET_LOG="outputs/xjn/rematch750_f05_transfer/launch/lane_npu${DEVICE}.log"
  for T in $ET_TRIALS; do
    echo "=== ET lane npu${DEVICE} trial ${T} $(date -u +%FT%TZ) ===" >> "$ET_LOG"
    "$PY" scripts/run_rematch750_f05_transfer.py --device npu:0 --trials "$T" --execute >> "$ET_LOG" 2>&1 || true
  done
fi

if [ -n "$DF_TRIALS" ]; then
  cd "$ET_ROOT" || exit 1
  export PYTHONPATH="$ET_ROOT/reproducibility/aegis_f1"
  mkdir -p outputs/xjn/rematch750_decay_filter/launch
  DF_LOG="outputs/xjn/rematch750_decay_filter/launch/lane_npu${DEVICE}.log"
  for T in $DF_TRIALS; do
    echo "=== DF lane npu${DEVICE} trial ${T} $(date -u +%FT%TZ) ===" >> "$DF_LOG"
    "$PY" scripts/run_rematch750_decay_filter.py --device npu:0 --trials "$T" --execute >> "$DF_LOG" 2>&1 || true
  done
fi

if [ -n "$V5_TRIALS" ]; then
  cd "$V5_ROOT" || exit 1
  export PYTHONPATH="$V5_ROOT/reproducibility/aegis_f1"
  mkdir -p outputs/rematch750_search_v5/launch
  V5_LOG="outputs/rematch750_search_v5/launch/lane_npu${DEVICE}.log"
  for T in $V5_TRIALS; do
    echo "=== V5 lane npu${DEVICE} trial ${T} $(date -u +%FT%TZ) ===" >> "$V5_LOG"
    "$PY" scripts/run_rematch750_search_v5_queue.py --trials "$T" --device npu:0 --workers 8 --prefetch-factor 2 --lock "outputs/rematch750_search_v5/queue/lane_npu${DEVICE}_${T}.lock" >> "$V5_LOG" 2>&1 || true
  done
fi
