"""
Integration test: end-to-end smoke test of the full pipeline.

Exercises the full pipeline on a tiny synthetic dataset:
    1. Generate tiny dataset (5 classes, 4 images/class train, 3 test)
    2. Generate canonical class mapping
    3. Split data
    4. Train 1 epoch (smoke test the training loop)
    5. Run inference
    6. Generate submission
    7. Validate submission coverage

Uses subprocess to run CLI commands. Skipped if CLIP is not available.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tiny_dataset():
    """Generate tiny dataset and yield paths to train/test dirs."""
    import tempfile

    train_dir = Path(tempfile.mkdtemp(prefix="tiny_train_"))
    test_dir = Path(tempfile.mkdtemp(prefix="tiny_test_"))

    # Run make_tiny_dataset
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "make_tiny_dataset.py"),
            "--train_dir", str(train_dir),
            "--test_dir", str(test_dir),
            "--num_classes", "5",
            "--images_per_class", "4",
            "--num_test", "3",
            "--seed", "42",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        pytest.fail(f"make_tiny_dataset failed:\n{result.stderr}")

    yield train_dir, test_dir

    # Cleanup
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def tiny_config(tmp_path_factory, tiny_dataset):
    """Create a config YAML for the tiny dataset smoke test."""
    train_dir, test_dir = tiny_dataset
    output_dir = tmp_path_factory.mktemp("tiny_output")
    split_dir = output_dir / "splits"
    log_dir = output_dir / "logs"
    save_dir = output_dir / "checkpoints"
    submission_dir = output_dir / "submissions"

    config_yaml = f"""
data:
  seed: 42
  split_seed: 42
  train_seed: 42
  train_dir: {train_dir}
  test_dir: {test_dir}
  split_dir: {split_dir}
  val_ratio: 0.25
  expected_num_classes: 5

model:
  clip_model_name: ViT-B/32
  num_classes: 5
  freeze_clip: true
  feature_dim: 512
  use_cached_features: false

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
  save_dir: {save_dir}
  max_grad_norm: 1.0

eval:
  batch_size: 2

output:
  log_dir: {log_dir}
  submission_dir: {submission_dir}
"""
    config_path = output_dir / "tiny_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_yaml)

    return config_path, output_dir, train_dir, test_dir


@pytest.mark.skipif(
    not shutil.which("python3") and not shutil.which("python"),
    reason="Python interpreter not found",
)
def test_full_pipeline_smoke(tiny_config):
    """Run the full pipeline: split -> train -> infer -> submit -> verify."""
    config_path, output_dir, train_dir, test_dir = tiny_config
    split_dir = output_dir / "splits"
    save_dir = output_dir / "checkpoints"
    submission_dir = output_dir / "submissions"

    # --- Step 1: Generate canonical class mapping ---
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"""
import json, sys
sys.path.insert(0, '{REPO_ROOT}')
from common.class_mapping import generate_class_mapping, save_class_mapping

class_to_idx, idx_to_class = generate_class_mapping(
    train_dir='{train_dir}',
    expected_num_classes=5,
)
save_class_mapping(
    output_dir='{split_dir}',
    class_to_idx=class_to_idx,
    idx_to_class=idx_to_class,
)
print("Class mapping generated:", json.dumps(class_to_idx))
""",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        pytest.fail(f"Class mapping failed:\n{result.stderr}\n{result.stdout}")
    assert "Class mapping generated" in result.stdout

    # --- Step 2: Split data ---
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "split_data.py"),
            "--train_dir", str(train_dir),
            "--val_ratio", "0.25",
            "--seed", "42",
            "--split_dir", str(split_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        pytest.fail(f"split_data failed:\n{result.stderr}\n{result.stdout}")
    # Logging goes to stderr; just check return code and file existence

    # Verify split files exist
    for fname in ["train.csv", "val.csv", "class_to_idx.json", "idx_to_class.json"]:
        assert (split_dir / fname).exists(), f"Missing split file: {fname}"

    # --- Step 3: Train 1 epoch (smoke test) ---
    # Check if CLIP is available
    clip_check = subprocess.run(
        [sys.executable, "-c", "import clip; print('CLIP available')"],
        capture_output=True, text=True,
    )
    if clip_check.returncode != 0:
        pytest.skip("CLIP package not available")

    result = subprocess.run(
        [
            sys.executable, "-m", "experiments.baseline.train",
            "--config", str(config_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,  # 5 minute timeout
    )
    if result.returncode != 0:
        pytest.fail(f"Training failed:\n{result.stderr}\n{result.stdout}")

    # Verify checkpoints exist
    assert (save_dir / "best.pt").exists(), "best.pt not created"
    assert (save_dir / "last.pt").exists(), "last.pt not created"

    # --- Step 4: Run inference ---
    result = subprocess.run(
        [
            sys.executable, "-m", "experiments.baseline.infer",
            "--config", str(config_path),
            "--ckpt", str(save_dir / "best.pt"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"Inference failed:\n{result.stderr}\n{result.stdout}")

    raw_csv = submission_dir / "pred_raw.csv"
    assert raw_csv.exists(), "pred_raw.csv not created"

    # --- Step 5: Generate submission ---
    result = subprocess.run(
        [
            sys.executable, "-m", "common.submission",
            "--raw", str(raw_csv),
            "--out_dir", str(submission_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        pytest.fail(f"Submission generation failed:\n{result.stderr}\n{result.stdout}")

    results_csv = submission_dir / "pred_results.csv"
    zip_path = submission_dir / "submission.zip"
    assert results_csv.exists(), "pred_results.csv not created"
    assert zip_path.exists(), "submission.zip not created"

    # --- Step 6: Validate submission ---
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_submission.py"),
            "--test_dir", str(test_dir),
            "--csv", str(results_csv),
            "--zip", str(zip_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Submission validation failed:\n{result.stderr}\n{result.stdout}"
        )

    # Verify coverage
    from scripts.check_submission import get_test_image_names

    test_names = get_test_image_names(Path(test_dir))
    with open(results_csv) as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == len(test_names), (
        f"Expected {len(test_names)} predictions, got {len(lines)}"
    )

    # --- Final: Verify pred_results.csv format ---
    import re
    pattern = re.compile(r"^.+\.(jpg|jpeg|png|bmp|webp), \d{4}$")
    for i, line in enumerate(lines, start=1):
        assert pattern.match(line), f"Line {i} format mismatch: {line!r}"
