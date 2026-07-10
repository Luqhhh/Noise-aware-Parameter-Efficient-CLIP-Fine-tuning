"""Test canonical class mapping."""
import pytest
import tempfile
from pathlib import Path
from common.class_mapping import (
    validate_class_directory_names,
    generate_canonical_mapping,
    load_or_generate_mapping,
)


def make_dummy_train_dir(base_dir, class_names):
    """Create dummy class directories."""
    train_dir = Path(base_dir) / "train"
    train_dir.mkdir(parents=True)
    for name in class_names:
        (train_dir / name).mkdir()
    return train_dir


def test_validate_valid_names():
    """4-digit names pass validation."""
    validate_class_directory_names(["0000", "0001", "0499"])


def test_validate_invalid_length():
    """Non-4-length name -> ValueError."""
    with pytest.raises(ValueError, match="Invalid"):
        validate_class_directory_names(["000"])


def test_validate_invalid_non_digit():
    """Non-numeric name -> ValueError."""
    with pytest.raises(ValueError, match="Invalid"):
        validate_class_directory_names(["abcd"])


def test_generate_canonical_mapping():
    """Generate mapping from dummy train dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        class_names = ["0000", "0001", "0002"]
        train_dir = make_dummy_train_dir(tmpdir, class_names)
        c2i, i2c = generate_canonical_mapping(train_dir, expected_num_classes=3)
        assert c2i == {"0000": 0, "0001": 1, "0002": 2}
        assert i2c == {"0": "0000", "1": "0001", "2": "0002"}


def test_generate_wrong_count():
    """Class count mismatch -> ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = make_dummy_train_dir(tmpdir, ["0000", "0001"])
        with pytest.raises(ValueError, match="Expected 5"):
            generate_canonical_mapping(train_dir, expected_num_classes=5)


def test_load_or_generate_creates_new():
    """When no mapping exists, generate one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = make_dummy_train_dir(tmpdir, ["0000", "0001"])
        meta_dir = Path(tmpdir) / "metadata"
        c2i, i2c = load_or_generate_mapping(
            meta_dir, train_dir, expected_num_classes=2
        )
        assert c2i == {"0000": 0, "0001": 1}
        assert (meta_dir / "class_to_idx.json").exists()


def test_load_or_generate_reuses_existing():
    """When mapping exists and matches, reuse it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = make_dummy_train_dir(tmpdir, ["0000", "0001"])
        meta_dir = Path(tmpdir) / "metadata"
        c2i_1, _ = load_or_generate_mapping(
            meta_dir, train_dir, expected_num_classes=2
        )
        c2i_2, _ = load_or_generate_mapping(
            meta_dir, train_dir, expected_num_classes=2
        )
        assert c2i_1 == c2i_2


def test_load_or_generate_detects_mismatch():
    """When train dir changes, detect inconsistency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = make_dummy_train_dir(tmpdir, ["0000", "0001"])
        meta_dir = Path(tmpdir) / "metadata"
        load_or_generate_mapping(meta_dir, train_dir, expected_num_classes=2)

        # Change train dir
        import shutil
        shutil.rmtree(train_dir)
        train_dir2 = make_dummy_train_dir(tmpdir, ["0000", "0002"])
        with pytest.raises(ValueError, match="inconsistent"):
            load_or_generate_mapping(meta_dir, train_dir2, expected_num_classes=2)
