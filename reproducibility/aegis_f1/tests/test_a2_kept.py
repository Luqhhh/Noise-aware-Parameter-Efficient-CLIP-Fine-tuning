import pandas as pd
import pytest

from aegis_clip.a2_kept import prepare_a2_kept_split


def _write_inputs(tmp_path, *, overlap=False, relabel=False):
    split = pd.DataFrame(
        {
            "image_path": ["train_dedup/0000/a.jpg", "train_dedup/0001/b.jpg"],
            "class_name": ["0000", "0001"],
            "label": [0, 1],
        }
    )
    purification = pd.DataFrame(
        {
            "image_path": ["train/0000/a.jpg", "train/0001/b.jpg"],
            "original_label": [0, 1],
            "training_label": [1 if relabel else 0, 1],
            "sample_weight": [1.0, 0.0],
            "training_role": ["clean", "rejected"],
        }
    )
    validation = pd.DataFrame(
        {
            "image_path": ["train/0000/a.jpg" if overlap else "train/0001/c.jpg"],
            "label": [0 if overlap else 1],
        }
    )
    paths = [tmp_path / name for name in ("split.csv", "purification.csv", "val.csv")]
    for frame, path in zip((split, purification, validation), paths):
        frame.to_csv(path, index=False)
    return paths


def test_prepare_a2_kept_split_is_exact_and_audited(tmp_path):
    paths = _write_inputs(tmp_path)
    result = prepare_a2_kept_split(*paths, tmp_path / "out", expected_classes=1)
    kept = pd.read_csv(tmp_path / "out" / "a2_kept_train.csv")
    assert result["kept_count"] == 1
    assert result["rejected_count"] == 1
    assert kept["image_path"].tolist() == ["train_dedup/0000/a.jpg"]


def test_prepare_a2_kept_split_rejects_overlap_and_relabel(tmp_path):
    with pytest.raises(ValueError, match="overlaps validation"):
        prepare_a2_kept_split(
            *_write_inputs(tmp_path, overlap=True), tmp_path / "overlap", expected_classes=1
        )
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="hard relabels"):
        prepare_a2_kept_split(
            *_write_inputs(other, relabel=True), other / "out", expected_classes=1
        )
