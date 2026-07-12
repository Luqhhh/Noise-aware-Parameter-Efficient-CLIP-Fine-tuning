#!/bin/bash
cd /home/lux1/noise
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== F0_STRICT ==="
python3 -m experiments.augmentation.train --config configs/f0_strict.yaml \
    --init-checkpoint outputs/d3_strict/seed42/checkpoints/best.pt
F0=$(python3 -c "import json; print(json.load(open('outputs/f0_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
log "F0 done: $F0"

log "=== F1_STRICT ==="
python3 -m experiments.augmentation.train --config configs/f1_strict.yaml \
    --init-checkpoint outputs/d3_strict/seed42/checkpoints/best.pt
F1=$(python3 -c "import json; print(json.load(open('outputs/f1_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
log "F1 done: $F1"

log "=== ALL DONE ==="
log "E0=0.7045  D3=0.7066  F0=$F0  F1=$F1"
