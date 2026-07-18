"""Test class label mapping correctness."""

import json
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_mapping_roundtrip():
    """class_to_idx and idx_to_class should be inverses of each other."""
    split_dir = Path("outputs/baselines/baseline/splits")
    if not split_dir.exists():
        return  # Skip if splits haven't been generated

    with open(split_dir / "class_to_idx.json") as f:
        class_to_idx = json.load(f)
    with open(split_dir / "idx_to_class.json") as f:
        idx_to_class = json.load(f)

    # Every class_name -> index -> class_name should roundtrip
    for class_name, idx in class_to_idx.items():
        assert (
            idx_to_class[str(idx)] == class_name
        ), f"Roundtrip failed: {class_name} -> {idx} -> {idx_to_class[str(idx)]}"

    # Every index -> class_name -> index should roundtrip
    for idx_str, class_name in idx_to_class.items():
        assert class_to_idx[class_name] == int(
            idx_str
        ), f"Roundtrip failed: {idx_str} -> {class_name} -> {class_to_idx[class_name]}"


def test_infer_uses_idx_to_class():
    """Verify infer.py uses idx_to_class for pred_label, not raw index."""
    import inspect

    from experiments.baseline import infer as infer_module

    source = inspect.getsource(infer_module.run_inference)
    # Should NOT contain the old pattern
    assert (
        'f"{int(pred_idx):04d}"' not in source
    ), "infer.py still formats pred_idx directly instead of using idx_to_class"
    # Should contain idx_to_class usage
    assert "idx_to_class" in source, "infer.py run_inference does not use idx_to_class"
