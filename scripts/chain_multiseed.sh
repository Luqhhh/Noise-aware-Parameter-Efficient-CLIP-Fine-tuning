#!/bin/bash
# Chain E0-strict + D3-strict for seeds 3407 and 2026.
# Run AFTER seed 42 chain completes.
# Usage: nohup bash scripts/chain_multiseed.sh > outputs/chain_multiseed.log 2>&1 &
set -euo pipefail
cd /home/lux1/noise

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

for seed in 3407 2026; do
  log "========== E0_STRICT seed=$seed =========="
  python3 -m experiments.augmentation.train --config configs/e0_strict.yaml --seed-override $seed
  E0=$(python3 -c "import json; print(json.load(open('outputs/e0_strict/seed${seed}/checkpoints/eval_results.json'))['best_val_acc'])")
  log "E0 seed=$seed done: best_val_acc=$E0"

  log "========== D3_STRICT seed=$seed =========="
  python3 -m experiments.augmentation.train --config configs/d3_strict.yaml --seed-override $seed
  D3=$(python3 -c "import json; print(json.load(open('outputs/d3_strict/seed${seed}/checkpoints/eval_results.json'))['best_val_acc'])")
  log "D3 seed=$seed done: best_val_acc=$D3"
  log "D3 vs E0 seed=$seed delta: $(python3 -c "print($D3 - $E0)")"
done

log "========== Paired Delta Computation =========="
python3 -c "
import json
from common.evaluation import compute_paired_deltas

e0_results = {}
d3_results = {}
for seed in [42, 3407, 2026]:
    try:
        e0_results[seed] = json.load(open(f'outputs/e0_strict/seed{seed}/checkpoints/eval_results.json'))['best_val_acc']
        d3_results[seed] = json.load(open(f'outputs/d3_strict/seed{seed}/checkpoints/eval_results.json'))['best_val_acc']
    except FileNotFoundError:
        pass

print(f'Seeds with results: {sorted(e0_results.keys())}')
deltas = compute_paired_deltas(e0_results, d3_results)
print(f'Paired deltas (D3 - E0):')
for seed, delta in sorted(deltas['deltas'].items()):
    print(f'  seed {seed}: {delta*100:+.4f}pp')
print(f'  Mean:  {deltas[\"mean_delta\"]*100:+.4f}pp')
print(f'  Std:   {deltas[\"std_delta\"]*100:.4f}pp')
print(f'  Wins:  {deltas[\"confirmation_wins\"]}')
print(f'  Gate +0.30pp: {\"PASS\" if deltas[\"mean_delta\"] > 0.003 else \"FAIL\"}')
"

log "========== ALL MULTI-SEED DONE =========="
