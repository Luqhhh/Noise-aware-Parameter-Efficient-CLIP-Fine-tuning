#!/usr/bin/env bash
set -euo pipefail

# Smoke test: end-to-end pipeline on a tiny dataset
# Generates 5-class tiny dataset, runs full train->eval->infer->submit->check pipeline

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TINY_TRAIN="data/tiny/train"
TINY_TEST="data/tiny/test"
TINY_CONFIG="configs/tiny_smoke_test.yaml"
TINY_OUTPUT="outputs/tiny_smoke_test"

echo "=========================================="
echo "Step 1: Generate tiny dataset"
echo "=========================================="
python tools/make_tiny_dataset.py \
    --train_dir "$TINY_TRAIN" \
    --test_dir "$TINY_TEST" \
    --num_classes 5 \
    --images_per_class 4 \
    --num_test 3

echo ""
echo "=========================================="
echo "Step 2: Check data"
echo "=========================================="
python scripts/check_data.py \
    --train_dir "$TINY_TRAIN" \
    --test_dir "$TINY_TEST" \
    --log_dir "$TINY_OUTPUT/logs"

echo ""
echo "=========================================="
echo "Step 3: Split data"
echo "=========================================="
python scripts/split_data.py \
    --train_dir "$TINY_TRAIN" \
    --val_ratio 0.25 \
    --seed 42 \
    --split_dir "$TINY_OUTPUT/splits"

echo ""
echo "=========================================="
echo "Step 4: Train (1 epoch)"
echo "=========================================="
python -m experiments.baseline.train --config <(cat <<EOF
data:
  seed: 42
  train_dir: $TINY_TRAIN
  test_dir: $TINY_TEST
  split_dir: $TINY_OUTPUT/splits
  val_ratio: 0.25
model:
  clip_model_name: ViT-B/32
  num_classes: 5
  freeze_clip: true
  feature_dim: 512
train:
  device: cuda
  batch_size: 2
  epochs: 1
  lr: 0.001
  weight_decay: 0.0001
  warmup_epochs: 0
  amp: false
  scheduler: cosine
  num_workers: 0
  save_dir: $TINY_OUTPUT/checkpoints
  max_grad_norm: 1.0
eval:
  batch_size: 2
output:
  log_dir: $TINY_OUTPUT/logs
  submission_dir: $TINY_OUTPUT/submissions
EOF
)

echo ""
echo "=========================================="
echo "Step 5: Evaluate"
echo "=========================================="
python -m experiments.baseline.evaluate --config <(cat <<EOF
data:
  seed: 42
  train_dir: $TINY_TRAIN
  test_dir: $TINY_TEST
  split_dir: $TINY_OUTPUT/splits
  val_ratio: 0.25
model:
  clip_model_name: ViT-B/32
  num_classes: 5
  freeze_clip: true
  feature_dim: 512
train:
  device: cuda
  num_workers: 0
eval:
  batch_size: 2
output:
  log_dir: $TINY_OUTPUT/logs
  submission_dir: $TINY_OUTPUT/submissions
EOF
) --ckpt "$TINY_OUTPUT/checkpoints/best.pt"

echo ""
echo "=========================================="
echo "Step 6: Inference"
echo "=========================================="
python -m experiments.baseline.infer --config <(cat <<EOF
data:
  seed: 42
  train_dir: $TINY_TRAIN
  test_dir: $TINY_TEST
  split_dir: $TINY_OUTPUT/splits
  val_ratio: 0.25
model:
  clip_model_name: ViT-B/32
  num_classes: 5
  freeze_clip: true
  feature_dim: 512
train:
  device: cuda
  num_workers: 0
eval:
  batch_size: 2
output:
  log_dir: $TINY_OUTPUT/logs
  submission_dir: $TINY_OUTPUT/submissions
EOF
) --ckpt "$TINY_OUTPUT/checkpoints/best.pt"

echo ""
echo "=========================================="
echo "Step 7: Generate submission"
echo "=========================================="
python -m common.submission \
    --raw "$TINY_OUTPUT/submissions/pred_raw.csv" \
    --out_dir "$TINY_OUTPUT/submissions"

echo ""
echo "=========================================="
echo "Step 8: Check submission"
echo "=========================================="
python scripts/check_submission.py \
    --test_dir "$TINY_TRAIN" \
    --csv "$TINY_OUTPUT/submissions/pred_results.csv" \
    --zip "$TINY_OUTPUT/submissions/submission.zip"

echo ""
echo "=========================================="
echo "Success: Smoke test PASSED!"
echo "=========================================="
