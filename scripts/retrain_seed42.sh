#!/bin/bash
# Batch retraining: all 4 strict experiments on rebuilt seed 42 split.
# Total estimated time: ~8 hours.
#
# Usage:
#     bash scripts/retrain_seed42.sh 2>&1 | tee outputs/retrain_seed42.log
#

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "Seed 42 Strict Experiment Retraining (SHA-256 Dedup Split)"
echo "Started: $(date)"
echo "================================================================"
echo ""
echo "Split:        outputs/master_splits/seed42/"
echo "  Train:      $(wc -l < outputs/master_splits/seed42/train.csv)"
echo "  Val:        $(wc -l < outputs/master_splits/seed42/val.csv)"
echo "  SHA dedup:  true"
echo "D3 Cleaning:  outputs/d3_strict/seed42/"
echo "  Train clean: $(wc -l < outputs/d3_strict/seed42/train.csv)"
echo ""

# ── Phase 1: E0-strict (frozen CLIP + linear head, master split) ──
echo "================================================================"
echo "Phase 1: E0_STRICT"
echo "  Config: configs/e0_strict.yaml"
echo "  Init:   OpenAI CLIP ViT-B/32"
echo "  Started: $(date)"
echo "================================================================"

python3 -m experiments.augmentation.train --config configs/e0_strict.yaml

E0_ACC=$(python3 -c "import json; print(json.load(open('outputs/e0_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
echo "E0_STRICT complete: best_val_acc = $E0_ACC"

# ── Phase 2: D3-strict (frozen CLIP + linear head, train-only dedup) ──
echo ""
echo "================================================================"
echo "Phase 2: D3_STRICT"
echo "  Config: configs/d3_strict.yaml"
echo "  Init:   OpenAI CLIP ViT-B/32"
echo "  Started: $(date)"
echo "================================================================"

python3 -m experiments.augmentation.train --config configs/d3_strict.yaml

D3_ACC=$(python3 -c "import json; print(json.load(open('outputs/d3_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
echo "D3_STRICT complete: best_val_acc = $D3_ACC"
D3_DELTA=$(python3 -c "print($D3_ACC - $E0_ACC)")
echo "D3 vs E0 delta: $D3_DELTA"

# ── Phase 3: F0-strict (frozen continue from D3-strict) ──
echo ""
echo "================================================================"
echo "Phase 3: F0_STRICT"
echo "  Config: configs/f0_strict.yaml"
echo "  Init:   outputs/d3_strict/seed42/checkpoints/best.pt"
echo "  Started: $(date)"
echo "================================================================"

python3 -m experiments.augmentation.train --config configs/f0_strict.yaml \
    --init-checkpoint outputs/d3_strict/seed42/checkpoints/best.pt

F0_ACC=$(python3 -c "import json; print(json.load(open('outputs/f0_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
echo "F0_STRICT complete: best_val_acc = $F0_ACC"
F0_DELTA=$(python3 -c "print($F0_ACC - $D3_ACC)")
echo "F0 vs D3 delta: $F0_DELTA"

# ── Phase 4: F1-strict (ln_post+proj unfrozen from D3-strict) ──
echo ""
echo "================================================================"
echo "Phase 4: F1_STRICT"
echo "  Config: configs/f1_strict.yaml"
echo "  Init:   outputs/d3_strict/seed42/checkpoints/best.pt"
echo "  Started: $(date)"
echo "================================================================"

python3 -m experiments.augmentation.train --config configs/f1_strict.yaml \
    --init-checkpoint outputs/d3_strict/seed42/checkpoints/best.pt

F1_ACC=$(python3 -c "import json; print(json.load(open('outputs/f1_strict/seed42/checkpoints/eval_results.json'))['best_val_acc'])")
echo "F1_STRICT complete: best_val_acc = $F1_ACC"
F1_DELTA=$(python3 -c "print($F1_ACC - $D3_ACC)")
echo "F1 vs D3 delta: $F1_DELTA"

# ── Phase 5: Re-evaluation with per-class metrics ──
echo ""
echo "================================================================"
echo "Phase 5: Per-Class Re-Evaluation"
echo "================================================================"

python3 scripts/reevaluate_strict.py

# ── Phase 6: Summary ──
echo ""
echo "================================================================"
echo "Seed 42 Retraining Complete: $(date)"
echo "================================================================"
echo ""
echo "Summary:"
echo "  Experiment   Val Acc    vs E0      vs D3      Epoch0 Gate"
echo "  ─────────────────────────────────────────────────────────"

for exp_id dir in \
    "E0_STRICT:e0_strict" \
    "D3_STRICT:d3_strict" \
    "F0_STRICT:f0_strict" \
    "F1_STRICT:f1_strict"; do

    IFS=':' read -r eid edir <<< "$exp_id"
    eval_path="outputs/$edir/seed42/checkpoints/eval_results.json"
    if [ -f "$eval_path" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$eval_path')); print(d['best_val_acc'])")
        EPS=$(python3 -c "import json; d=json.load(open('$eval_path')); print(d.get('actual_epochs_run','?'))")
        E0G=$(python3 -c "import json; d=json.load(open('$eval_path')); print(d.get('epoch0_val_acc','N/A'))")
        echo "  $eid  $ACC  ?         ?          epoch0=$E0G  epochs=$EPS"
    fi
done

echo ""
echo "Compared to old (leaked) split results:"
echo "  E0: 69.09% → ?"
echo "  D3: 69.32% → ?"
echo "  F0: 69.33% → ?"
echo "  F1: 69.43% → ?"
