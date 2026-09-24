#!/bin/bash
# Wait for the V5 NPU6 H overflow lane, then run HL00-HL03 sequentially.
set -u
WAIT_PID="$1"; DEVICE="$2"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
cd /workspace/noise-v5 || exit 1
export PYTHONPATH=reproducibility/aegis_f1
export OMP_NUM_THREADS=4
export REMATCH_CODE_COMMIT="${REMATCH_CODE_COMMIT:-unknown}"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
mkdir -p outputs/rematch750_search_v5/launch
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_head_l2sp.py \
  --device npu:0 --trials HL00 HL01 HL02 HL03 --execute \
  > "outputs/rematch750_search_v5/launch/HL00_HL03.npu${DEVICE}.log" 2>&1
