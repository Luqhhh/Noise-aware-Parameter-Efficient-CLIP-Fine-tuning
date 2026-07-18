#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
PY=/home/clairvoyant/.venvs/noise-clip/bin/python
CFG=configs/s_elr_base.yaml
OUT=outputs/s_elr_base/seed42
CKPT=$OUT/checkpoints/best.pt

cd "$ROOT"
mkdir -p "$OUT/checkpoints" "$OUT/submissions" "$OUT/submissions_tta" "$OUT/logs"

while systemctl --user is-active --quiet s-elr-base-seed42.service; do
  sleep 300
done

test -f "$CKPT"

"$PY" -m experiments.baseline.evaluate --config "$CFG" --ckpt "$CKPT"
"$PY" -m experiments.baseline.infer --config "$CFG" --ckpt "$CKPT"
"$PY" -m common.submission --raw "$OUT/submissions/pred_raw.csv" --out_dir "$OUT/submissions"
/home/clairvoyant/.venvs/noise-clip/bin/python scripts/check_submission.py --test_dir test --csv "$OUT/submissions/pred_results.csv" --zip "$OUT/submissions/submission.zip"

PYTHONPATH=. "$PY" scripts/evaluate_tta.py --config "$CFG" --checkpoint "$CKPT" --tta horizontal_flip --output-dir "$OUT/tta_eval"
PYTHONPATH=. "$PY" scripts/infer_tta.py --config "$CFG" --checkpoint "$CKPT" --tta horizontal_flip --output-dir "$OUT/submissions_tta"
"$PY" -m common.submission --raw "$OUT/submissions_tta/pred_raw.csv" --out_dir "$OUT/submissions_tta"
/home/clairvoyant/.venvs/noise-clip/bin/python scripts/check_submission.py --test_dir test --csv "$OUT/submissions_tta/pred_results.csv" --zip "$OUT/submissions_tta/submission.zip"

touch "$OUT/POSTPROCESS_COMPLETE"
echo "S_ELR_BASE post-processing complete."
