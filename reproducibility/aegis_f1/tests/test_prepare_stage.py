import json

import pandas as pd

from aegis_clip.cli.prepare_stage import prepare_stage


def test_prepare_stage_is_group_safe_and_reproducible(tmp_path) -> None:
    train_root = tmp_path / "official_train"
    for class_index in range(3):
        class_dir = train_root / f"{class_index:04d}"
        class_dir.mkdir(parents=True)
        for image_index in range(10):
            (class_dir / f"{image_index:03d}.jpg").write_bytes(
                f"class={class_index};image={image_index}".encode()
            )

    first = tmp_path / "first"
    second = tmp_path / "second"
    arguments = {
        "train_root": train_root,
        "stage": "preliminary",
        "seed": 42,
        "val_ratio": 0.2,
        "expected_classes": 3,
        "expected_samples": 30,
        "hash_workers": 2,
    }
    first_manifest = prepare_stage(output_dir=first, **arguments)
    second_manifest = prepare_stage(output_dir=second, **arguments)

    assert first_manifest["train_count"] + first_manifest["val_count"] == 30
    assert first_manifest["test_data_used"] is False
    assert first_manifest["external_data"] is False
    assert first_manifest["train_csv_sha256"] == second_manifest["train_csv_sha256"]
    assert first_manifest["val_csv_sha256"] == second_manifest["val_csv_sha256"]

    groups = json.loads((first / "content_groups.json").read_text())
    train = pd.read_csv(first / "train.csv")
    val = pd.read_csv(first / "val.csv")
    train_hashes = {
        groups["/".join(path.split("/")[-2:])] for path in train["image_path"]
    }
    val_hashes = {
        groups["/".join(path.split("/")[-2:])] for path in val["image_path"]
    }
    assert train_hashes.isdisjoint(val_hashes)
    assert set(train["label"]) == {0, 1, 2}
    assert set(val["label"]) == {0, 1, 2}
