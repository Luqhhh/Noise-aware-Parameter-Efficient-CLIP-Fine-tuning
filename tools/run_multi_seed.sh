#!/bin/bash
# Sequential multi-seed training for R0-2
# Runs D3 and B2 for seeds 2026 and 3407 back-to-back.
# Launched from tools/run_multi_seed.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ── Step 1: D3 seed 2026 ──
log "=== Step 1/4: D3 seed 2026 ==="
python3 -m experiments.baseline.train \
    --config configs/d3_strict.yaml \
    --seed-override 2026 \
    --experiment-id D3_STRICT \
    --mode dev \
    --head-type linear \
    --augmentation-preset a0 \
    --allow-overwrite
log "D3 seed 2026 done."

# ── Step 2: D3 seed 3407 ──
log "=== Step 2/4: D3 seed 3407 ==="
python3 -m experiments.baseline.train \
    --config configs/d3_strict.yaml \
    --seed-override 3407 \
    --experiment-id D3_STRICT \
    --mode dev \
    --head-type linear \
    --augmentation-preset a0 \
    --allow-overwrite
log "D3 seed 3407 done."

# ── Step 3: B2 seed 2026 ──
log "=== Step 3/4: B2 seed 2026 ==="
python3 -m experiments.baseline.train \
    --config configs/b2_gce07.yaml \
    --seed-override 2026 \
    --experiment-id B2_GCE07 \
    --mode dev \
    --head-type linear \
    --augmentation-preset a0 \
    --allow-overwrite
log "B2 seed 2026 done."

# ── Step 4: B2 seed 3407 ──
log "=== Step 4/4: B2 seed 3407 ==="
python3 -m experiments.baseline.train \
    --config configs/b2_gce07.yaml \
    --seed-override 3407 \
    --experiment-id B2_GCE07 \
    --mode dev \
    --head-type linear \
    --augmentation-preset a0 \
    --allow-overwrite
log "B2 seed 3407 done."

log "=== All 4 training jobs complete ==="
