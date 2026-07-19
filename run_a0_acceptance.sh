#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)"

echo "=== A0 Smoke Acceptance ==="
echo ""

echo "── 1. Runtime Audit Tests ──"
python3 -m pytest -q tests/test_runtime_manifest_audit.py

echo ""
echo "── 2. A0 Smoke Tests ──"
python3 -m pytest -q tests/test_a0_smoke.py

echo ""
echo "── 3. A0 Real 20-Batch Smoke ──"
python3 scripts/run_a0_acceptance.py \
  --config configs/nr_ctrl_oof_zero_0001_fixed.yaml \
  --max-batches 20 \
  --output-log logs/a0_real_20batch.log

echo ""
echo "=== ALL DONE ==="
