#!/bin/bash
# D3 → F0 → F1 on rebuilt seed 42 split.
cd /home/lux1/noise

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_exp() {
    local label="$1"; shift
    log "========== $label START =========="
    python3 -m experiments.augmentation.train "$@"
    if [ $? -ne 0 ]; then
        log "ERROR: $label failed"
        exit 1
    fi
    log "$label DONE"
}

# Phase 2: D3_STRICT
run_exp "Phase 2: D3_STRICT" --config configs/d3_strict.yaml
D3=$(python3 -c "import json; print(json.load(open('outputs/d3_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
log "D3: $D3  (vs E0 0.7045 → $(python3 -c "print($D3 - 0.7045)")pp)"

# Phase 3: F0_STRICT
run_exp "Phase 3: F0_STRICT" --config configs/f0_strict.yaml \
    --init-checkpoint outputs/d3_strict/seed42/checkpoints/best.pt
F0=$(python3 -c "import json; print(json.load(open('outputs/f0_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
log "F0: $F0  (vs D3: $(python3 -c "print($F0 - $D3)")pp)"

# Phase 4: F1_STRICT
run_exp "Phase 4: F1_STRICT" --config configs/f1_strict.yaml \
    --init-checkpoint outputs/d3_strict/seed42/checkpoints/best.pt
F1=$(python3 -c "import json; print(json.load(open('outputs/f1_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
log "F1: $F1  (vs D3: $(python3 -c "print($F1 - $D3)")pp)"

log "========== ALL DONE =========="
log "E0=0.7045  D3=$D3  F0=$F0  F1=$F1"
