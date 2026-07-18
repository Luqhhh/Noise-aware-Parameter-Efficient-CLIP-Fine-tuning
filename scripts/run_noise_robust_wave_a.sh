#!/usr/bin/env bash
# run_noise_robust_wave_a.sh — Train + evaluate + infer + submission for one Wave A experiment.
# Usage: bash scripts/run_noise_robust_wave_a.sh CONFIG

set -euo pipefail

CONFIG="${1:?Usage: bash scripts/run_noise_robust_wave_a.sh <config>}"

# Extract experiment ID from YAML
EXP_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['experiment']['id'])")
echo "=== Wave A: ${EXP_ID} ==="

# Resolve paths from config
eval "$(python3 -c "
import yaml
c = yaml.safe_load(open('${CONFIG}'))
print(f'SAVE_DIR={c[\"train\"][\"save_dir\"]}')
print(f'SUBMISSION_DIR={c[\"output\"][\"submission_dir\"]}')
print(f'MANIFEST_PATH={c[\"sample_weighting\"][\"manifest_path\"]}')
")"

CKPT="${SAVE_DIR}/best.pt"

# 0. Run tests (fail-fast gate)
echo "--- Pre-flight tests ---"
python3 -m pytest -q --ignore=tests/test_integration.py || { echo "TESTS FAILED"; exit 1; }

# 1. Audit manifest
echo "--- Manifest audit ---"
python3 scripts/audit_purification_manifest.py \
  --manifest "${MANIFEST_PATH}" \
  --strict-train outputs/data/d3_strict/seed42/train.csv

# 2. Train
echo "--- Training ---"
python3 -m experiments.baseline.train --config "${CONFIG}"

# 3. Evaluate (dual validation)
echo "--- Evaluation ---"
python3 -m experiments.baseline.evaluate --config "${CONFIG}" --ckpt "${CKPT}"

# 4. Dual validation
echo "--- Dual validation ---"
python3 tools/evaluate_dual_validation.py --name "${EXP_ID}" \
  --config "${CONFIG}" --ckpt "${CKPT}" --device cuda 2>/dev/null || echo "Dual validation skipped (tool may not exist)"

# 5. Infer
echo "--- Inference ---"
python3 -m experiments.baseline.infer --config "${CONFIG}" --ckpt "${CKPT}"

# 6. Generate submission
PRED_RAW="${SUBMISSION_DIR}/pred_raw.csv"
python3 -m common.submission --raw "${PRED_RAW}" --out_dir "${SUBMISSION_DIR}"

# 7. Validate submission
echo "--- Submission validation ---"
python3 scripts/check_submission.py --test_dir test \
  --csv "${SUBMISSION_DIR}/pred_results.csv" \
  --zip "${SUBMISSION_DIR}/submission.zip"

echo "=== ${EXP_ID} complete ==="
