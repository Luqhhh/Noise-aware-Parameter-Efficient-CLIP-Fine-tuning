#!/bin/bash
# Batch multi-seed training for strict experiments.
# Runs E0-strict + D3-strict for seeds 3407 and 2026.
# Total estimated time: ~8 hours (4 × ~2 hours each).
#
# Usage:
#     bash scripts/run_multiseed.sh 2>&1 | tee outputs/multiseed_training.log
#

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENTS=(
    "E0-strict seed 3407:configs/e0_strict.yaml:3407"
    "D3-strict seed 3407:configs/d3_strict.yaml:3407"
    "E0-strict seed 2026:configs/e0_strict.yaml:2026"
    "D3-strict seed 2026:configs/d3_strict.yaml:2026"
)

echo "================================================================"
echo "Multi-Seed Training Batch"
echo "Started: $(date)"
echo "Experiments: ${#EXPERIMENTS[@]}"
echo "================================================================"

for entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r label config seed <<< "$entry"
    echo ""
    echo "================================================================"
    echo "  $label"
    echo "  Config: $config"
    echo "  Seed: $seed"
    echo "  Started: $(date)"
    echo "================================================================"

    python3 -m experiments.augmentation.train \
        --config "$config" \
        --seed-override "$seed"

    echo ""
    echo "  $label — DONE at $(date)"

    # Print best result
    SAVE_DIR=$(python3 -c "
import yaml
c = yaml.safe_load(open('$config'))
d = c['train']['save_dir'].replace('seed42', 'seed$seed')
print(d)
")
    if [ -f "$SAVE_DIR/eval_results.json" ]; then
        python3 -c "
import json
d = json.load(open('$SAVE_DIR/eval_results.json'))
print(f'  Best val acc: {d[\"best_val_acc\"]:.4f}')
print(f'  Actual epochs: {d.get(\"actual_epochs_run\", \"?\")}')
"
    fi
done

echo ""
echo "================================================================"
echo "All experiments complete: $(date)"
echo "================================================================"

# Compute paired deltas
echo ""
echo "Computing paired deltas..."
python3 -c "
import json
from common.evaluation import compute_paired_deltas

# Load results for all seeds
e0_results = {}
d3_results = {}
for seed in [42, 3407, 2026]:
    e0_path = f'outputs/e0_strict/seed{seed}/checkpoints/eval_results.json'
    d3_path = f'outputs/d3_strict/seed{seed}/checkpoints/eval_results.json'
    try:
        e0_results[seed] = json.load(open(e0_path))['best_val_acc']
        d3_results[seed] = json.load(open(d3_path))['best_val_acc']
    except FileNotFoundError:
        pass

if len(e0_results) >= 1 and len(d3_results) >= 1:
    deltas = compute_paired_deltas(e0_results, d3_results)
    print('Paired deltas (D3 - E0):')
    for seed, delta in sorted(deltas['deltas'].items()):
        print(f'  seed {seed}: {delta*100:+.4f}pp')
    print(f'  Mean:  {deltas[\"mean_delta\"]*100:+.4f}pp')
    print(f'  Std:   {deltas[\"std_delta\"]*100:.4f}pp')
    print(f'  Wins:  {deltas[\"confirmation_wins\"]}')
    print(f'  Gate +0.30pp: {\"PASS\" if deltas[\"mean_delta\"] > 0.003 else \"FAIL\"}')
"
