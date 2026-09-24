#!/usr/bin/env bash
# Build the local 750-class OOF/quality assets, N1 manifest and clean070 subset.
#
# This chain intentionally waits for the GPU0 C0 training to finish before it
# starts the OOF pass, because both stages need the same 8 GiB GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TRAIN_CSV="${TRAIN_CSV:-/home/lux1/noise/artifacts/stages/repechage/20260921/train_dev.csv}"
CACHE_DIR="${CACHE_DIR:-/home/lux1/noise/artifacts/stages/repechage/20260921/features}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/lux1/noise/train}"
OOF_DIR="${OOF_DIR:-$REPO_ROOT/outputs/f05_focus/oof}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_ROOT/artifacts/f05_focus}"
C0_RUN_DIR="${C0_RUN_DIR:-$REPO_ROOT/outputs/f05_focus/C0_F05_CUDA/seed42}"
C0_CONFIG="${C0_CONFIG:-$REPO_ROOT/outputs/f05_focus/_runtime_configs/C0_cuda_0.yaml}"
WAIT_FOR_C0="${WAIT_FOR_C0:-1}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$OOF_DIR" "$ARTIFACT_DIR"

if [[ "$WAIT_FOR_C0" == "1" ]]; then
  echo "[quality-chain] waiting for C0 to finish"
  while pgrep -f "aegis_clip.cli.rematch train --config .*C0_cuda_0.yaml" >/dev/null || \
        pgrep -f "resume_c0_gpu0.py" >/dev/null; do
    sleep 60
  done
  if [[ ! -f "$C0_RUN_DIR/checkpoints/selected_report.json" ]]; then
    echo "[quality-chain] selected_report missing; generating it from best.pt"
    env PYTHONPATH="$REPO_ROOT/reproducibility/aegis_f1" \
      python3 - "$C0_CONFIG" "$C0_RUN_DIR" <<'REPORT_PY'
import sys
from pathlib import Path

from aegis_clip.cli.rematch import per_class_report
from aegis_clip.config import load_config

config = load_config(sys.argv[1])
run_dir = Path(sys.argv[2]).resolve()
per_class_report(config, run_dir)
print(f"wrote {run_dir / 'checkpoints/selected_report.json'}")
REPORT_PY
  fi
  if [[ ! -f "$C0_RUN_DIR/checkpoints/selected_report.json" ]]; then
    echo "[quality-chain] C0 stopped without selected_report.json: $C0_RUN_DIR" >&2
    exit 1
  fi
  echo "[quality-chain] C0 finished"
fi

# run_oof opens image_path values directly; canonical CSVs use train/... .
if [[ ! -e "$REPO_ROOT/train" ]]; then
  ln -s "$TRAIN_ROOT" "$REPO_ROOT/train"
fi

echo "[quality-chain] OOF/quality: start"
python3 "$REPO_ROOT/scripts/build_rematch750_quality_asset.py" \
  --train-csv "$TRAIN_CSV" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OOF_DIR" \
  --folds 3 \
  --seed 42 \
  --epochs 30 \
  --device "$DEVICE" \
  --flip-batch-size 64 \
  --flip-workers 2
echo "[quality-chain] OOF/quality: done"

echo "[quality-chain] N1 manifest: start"
python3 "$REPO_ROOT/scripts/build_hp_noise_manifest.py" \
  --sample-quality "$OOF_DIR/sample_quality.csv" \
  --oof-logits "$OOF_DIR/oof_logits.pt" \
  --output "$ARTIFACT_DIR/hp_noise_manifest.csv"
echo "[quality-chain] N1 manifest: done"

# Preregistered mapping for D1/D2: clean_probability := OOF soft_weight.
echo "[quality-chain] clean070: start"
python3 - "$OOF_DIR/sample_quality.csv" "$ARTIFACT_DIR/oof_clean_probability.csv" <<'PY'
import sys
import pandas as pd

quality_path, output_path = sys.argv[1], sys.argv[2]
quality = pd.read_csv(quality_path)
required = {"image_path", "soft_weight"}
missing = required - set(quality.columns)
if missing:
    raise SystemExit(f"sample_quality is missing: {sorted(missing)}")
frame = pd.DataFrame(
    {
        "image_path": quality["image_path"].astype(str),
        "clean_probability": quality["soft_weight"].astype(float),
    }
)
frame.to_csv(output_path, index=False)
print(f"wrote {output_path} rows={len(frame)}")
PY

python3 "$REPO_ROOT/scripts/build_f05_clean_subset.py" \
  --source-csv "$TRAIN_CSV" \
  --quality "$ARTIFACT_DIR/oof_clean_probability.csv" \
  --threshold 0.70 \
  --output "$ARTIFACT_DIR/clean070.csv"
echo "[quality-chain] clean070: done"

python3 - "$ARTIFACT_DIR/quality_chain_status.json" \
  "$OOF_DIR/sample_quality.csv" "$OOF_DIR/oof_logits.pt" \
  "$ARTIFACT_DIR/hp_noise_manifest.csv" "$ARTIFACT_DIR/clean070.csv" <<'STATUS_PY'
import hashlib
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
artifacts = {
    "sample_quality": sys.argv[2],
    "oof_logits": sys.argv[3],
    "hp_noise_manifest": sys.argv[4],
    "clean070": sys.argv[5],
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


status = {"status": "completed", "artifacts": {}}
for name, raw_path in artifacts.items():
    path = Path(raw_path)
    status["artifacts"][name] = str(path)
    status["artifacts"][name + "_sha256"] = digest(path) if path.is_file() else None
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(status, ensure_ascii=False, indent=2))
STATUS_PY
echo "[quality-chain] all done"
