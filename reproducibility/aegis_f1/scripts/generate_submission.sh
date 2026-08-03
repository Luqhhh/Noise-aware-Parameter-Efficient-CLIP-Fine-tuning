#!/usr/bin/env bash
# Generate an audited M1+flip submission package from a trained checkpoint.
#
# Usage:
#   generate_submission.sh <checkpoint> <config> <output_dir> [exp_id]
#
# Uses the proven protocol: attention_crop_flip, crop160/top5, local_weight 0.40,
# flip_weight 0.50, temperature 1.5, balanced-prior 0.85.
set -euo pipefail
CHECKPOINT="${1:?checkpoint path required}"
CONFIG="${2:?config path required}"
OUTDIR="${3:?output dir required}"
EXP_ID="${4:-manual}"

mkdir -p "${OUTDIR}"
python3 -m aegis_clip.cli.infer \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --output-dir "${OUTDIR}" \
  --tta horizontal_flip \
  --tta-fusion mean_probabilities \
  --tta-temperature 1.5 \
  --tta-view-weight 0.5 \
  --local-view attention_crop \
  --local-crop-size 160 \
  --local-top-k 5 \
  --local-weight 0.40 \
  --local-temperature 1.5 \
  --acknowledge-tta-risk \
  --acknowledge-local-view-risk \
  --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior \
  --batch-size 128 \
  --overwrite

echo "=== Auditing ${EXP_ID} ==="
python3 -m aegis_clip.cli.audit_submission \
  --config "${CONFIG}" \
  --submission-dir "${OUTDIR}" \
  --allow-tta
