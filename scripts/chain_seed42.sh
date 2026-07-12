#!/bin/bash
# Chain all 4 strict experiments on rebuilt seed 42 split.
# Run: nohup bash scripts/chain_seed42.sh > outputs/chain_seed42.log 2>&1 &
cd /home/lux1/noise

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_exp() {
    local label="$1"; shift
    log "========== $label START =========="
    python3 -m experiments.augmentation.train "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "ERROR: $label failed with exit code $rc"
        exit $rc
    fi
    log "$label DONE"
}

# Phase 1: E0_STRICT
run_exp "Phase 1/4: E0_STRICT" --config configs/e0_strict.yaml
E0=$(python3 -c "import json; print(json.load(open('outputs/e0_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
log "E0_STRICT: best_val_acc=$E0"

# Phase 2: D3_STRICT
run_exp "Phase 2/4: D3_STRICT" --config configs/d3_strict.yaml
D3=$(python3 -c "import json; print(json.load(open('outputs/d3_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
log "D3_STRICT: best_val_acc=$D3"
log "D3 vs E0 delta: $(python3 -c "print($D3 - $E0)")"

# Phase 3: F0_STRICT
run_exp "Phase 3/4: F0_STRICT" --config configs/f0_strict.yaml \
    --init-checkpoint outputs/d3_strict/seed42/checkpoints/best.pt
F0=$(python3 -c "import json; print(json.load(open('outputs/f0_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
log "F0_STRICT: best_val_acc=$F0"

# Phase 4: F1_STRICT
run_exp "Phase 4/4: F1_STRICT" --config configs/f1_strict.yaml \
    --init-checkpoint outputs/d3_strict/seed42/checkpoints/best.pt
F1=$(python3 -c "import json; print(json.load(open('outputs/f1_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
log "F1_STRICT: best_val_acc=$F1"

log "========== RE-EVALUATION =========="
python3 scripts/reevaluate_strict.py

log "========== ALL DONE =========="
log "E0=$E0  D3=$D3  F0=$F0  F1=$F1"
