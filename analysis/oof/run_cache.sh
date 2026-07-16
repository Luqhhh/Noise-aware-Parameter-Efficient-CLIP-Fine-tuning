#!/usr/bin/env bash
set -euo pipefail

exec /home/clairvoyant/.venvs/noise-clip/bin/python \
  scripts/cache_features.py \
  --config configs/b2_gce05.yaml \
  > outputs/phase3/oof/cache_features.log 2>&1
