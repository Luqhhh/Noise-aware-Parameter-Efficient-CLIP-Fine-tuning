#!/usr/bin/env bash
# Post-P1 pipeline: checkpoint averaging → CR-0 → CR-1 → CR-2
set -euo pipefail

REPO="/home/lux1/noise/reproducibility/aegis_f1"
cd "$REPO"

P1_OUTDIR="outputs/P1_A2_STRICT_EPOCH_CKPTS/seed42/checkpoints"
LOG_DIR="outputs/phase4/logs"
mkdir -p "$LOG_DIR" "outputs/phase4/p1_averaging"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"; }

# ── Step 0: Wait for P1 training to finish ─────────────────────────────
log "Waiting for P1 training to complete (checking for epoch_6.pt)..."
while [ ! -f "$P1_OUTDIR/epoch_6.pt" ]; do
    sleep 30
done
log "P1 training complete! Found epoch_6.pt"

# Verify all epoch checkpoints exist
for ep in 1 2 3 4 5 6; do
    if [ ! -f "$P1_OUTDIR/epoch_${ep}.pt" ]; then
        log "ERROR: epoch_${ep}.pt missing!"
        exit 1
    fi
done
log "All 6 epoch checkpoints verified"

# ── Step 1: Checkpoint Averaging ────────────────────────────────────────
log "=== P1: Checkpoint Averaging ==="

AVG_OUTDIR="outputs/phase4/p1_averaging"
CONFIG="configs/p1_a2_strict_epochs.yaml"

# SWA-1: epochs 2-6 equal weight
log "Running SWA-1 (epochs 2-6 equal)..."
python3 -m aegis_clip.cli.average_checkpoints \
    --config "$CONFIG" \
    --checkpoints "$P1_OUTDIR"/epoch_{2,3,4,5,6}.pt \
    --scheme equal \
    --output "$AVG_OUTDIR/swa1_epoch2_6.pt" \
    --eval --selection-metric clean_core_micro \
    2>&1 | tee "$LOG_DIR/swa1.log"
log "SWA-1 done"

# SWA-2: epochs 2-4 equal weight
log "Running SWA-2 (epochs 2-4 equal)..."
python3 -m aegis_clip.cli.average_checkpoints \
    --config "$CONFIG" \
    --checkpoints "$P1_OUTDIR"/epoch_{2,3,4}.pt \
    --scheme equal \
    --output "$AVG_OUTDIR/swa2_epoch2_4.pt" \
    --eval --selection-metric clean_core_micro \
    2>&1 | tee "$LOG_DIR/swa2.log"
log "SWA-2 done"

# SWA-3: epochs 3-6 equal weight
log "Running SWA-3 (epochs 3-6 equal)..."
python3 -m aegis_clip.cli.average_checkpoints \
    --config "$CONFIG" \
    --checkpoints "$P1_OUTDIR"/epoch_{3,4,5,6}.pt \
    --scheme equal \
    --output "$AVG_OUTDIR/swa3_epoch3_6.pt" \
    --eval --selection-metric clean_core_micro \
    2>&1 | tee "$LOG_DIR/swa3.log"
log "SWA-3 done"

# SWA-4: greedy soup from epoch 2
log "Running SWA-4 (greedy soup from epoch 2)..."
python3 -m aegis_clip.cli.average_checkpoints \
    --config "$CONFIG" \
    --checkpoints "$P1_OUTDIR"/epoch_{2,3,4,5,6}.pt \
    --scheme greedy_soup \
    --output "$AVG_OUTDIR/swa4_greedy_soup.pt" \
    --eval --selection-metric clean_core_micro \
    2>&1 | tee "$LOG_DIR/swa4.log"
log "SWA-4 done"

log "=== P1 averaging complete ==="

# ── Step 2: CR-0 Baseline ──────────────────────────────────────────────
log "=== P2: CR-0 Baseline ==="
python3 -m aegis_clip.cli.train \
    --config configs/cr0_baseline.yaml \
    2>&1 | tee "$LOG_DIR/cr0.log"
log "CR-0 complete"

# ── Step 3: CR-1 Hard Gate ──────────────────────────────────────────────
log "=== P2: CR-1 Hard Gate (clean≥0.70) ==="
python3 -m aegis_clip.cli.train \
    --config configs/cr1_hard_gate.yaml \
    2>&1 | tee "$LOG_DIR/cr1.log"
log "CR-1 complete"

# ── Step 4: CR-2 Soft Gate ─────────────────────────────────────────────
log "=== P2: CR-2 Soft Gate ==="
python3 -m aegis_clip.cli.train \
    --config configs/cr2_soft_gate.yaml \
    2>&1 | tee "$LOG_DIR/cr2.log"
log "CR-2 complete"

log "=== ALL PIPELINE STEPS COMPLETE ==="
