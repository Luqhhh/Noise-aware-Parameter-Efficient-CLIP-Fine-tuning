#!/usr/bin/env bash
# Evaluate a trained child checkpoint with M1+flip sweep on the val set.
# Usage: evaluate_child.sh <checkpoint> <config> <output_json>
set -euo pipefail
CHECKPOINT="${1:?checkpoint required}"
CONFIG="${2:?config required}"
OUTPUT="${3:?output json required}"

python3 -m aegis_clip.cli.sweep_localization \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --crop-sizes 160 --top-ks 5 \
  --local-weights 0.35,0.40,0.45 \
  --flip-weights 0.4,0.5,0.6 \
  --temperature 1.5 --include-horizontal-flip \
  --batch-size 48 --device cuda --overwrite

echo "=== Best fusion summary ==="
python3 - "${OUTPUT}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
best = None
for c in d.get('candidates', []):
    if c.get('crop_size') != 160: continue
    for f in c.get('fusions', []):
        k = (f.get('clean_core_micro', 0), f.get('raw_micro', 0), f.get('local_weight'), f.get('flip_weight'))
        if best is None or k[0] > best[0]:
            best = k
print('best by clean-core: clean=%.4f raw=%.4f local_w=%s flip_w=%s' % best)
PY
