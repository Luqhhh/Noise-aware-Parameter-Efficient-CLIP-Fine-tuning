"""The submission label-range check must be stage-agnostic."""

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _check_csv_module():
    spec = importlib.util.spec_from_file_location(
        "check_submission_under_test", REPO / "scripts" / "check_submission.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(tmp_path, label):
    path = tmp_path / "pred_results.csv"
    path.write_text(f"test_0000_00000.jpg, {label}\n", encoding="utf-8")
    return path


def test_1500_class_label_range_accepted(tmp_path):
    module = _check_csv_module()
    csv_path = _write_csv(tmp_path, "1499")
    all_ok, errors = module.check_csv(
        csv_path, {"test_0000_00000.jpg"}, num_classes=1500
    )
    assert all_ok
    assert any("0000, 1499" in error for error in errors)


def test_label_above_range_rejected_for_1500_classes(tmp_path):
    module = _check_csv_module()
    csv_path = _write_csv(tmp_path, "1500")
    all_ok, errors = module.check_csv(
        csv_path, {"test_0000_00000.jpg"}, num_classes=1500
    )
    assert not all_ok
    assert any("out of range" in error for error in errors)
