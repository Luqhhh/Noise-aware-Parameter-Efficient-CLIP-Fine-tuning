import json

import pandas as pd
import torch

from aegis_clip.a2_gate import prepare_a2_fullfit, prepare_a2_gate


def _write_split(path, rows) -> None:
    pd.DataFrame(rows, columns=["image_path", "label"]).to_csv(path, index=False)


def test_prepare_a2_gate_builds_disjoint_partitions_and_union_trust(tmp_path) -> None:
    a2_train = tmp_path / "a2_train.csv"
    a2_val = tmp_path / "a2_val.csv"
    aegis_train = tmp_path / "aegis_train.csv"
    aegis_val = tmp_path / "aegis_val.csv"
    _write_split(
        a2_train,
        [("train_dedup/0000/a.jpg", 0), ("train_dedup/0001/b.jpg", 1)],
    )
    _write_split(a2_val, [("train/0000/c.jpg", 0), ("train/0001/d.jpg", 1)])
    _write_split(
        aegis_train,
        [("train/0000/c.jpg", 0), ("train/0001/d.jpg", 1)],
    )
    _write_split(
        aegis_val,
        [
            ("train/0000/a.jpg", 0),
            ("train/0001/b.jpg", 1),
            ("train/0000/e.jpg", 0),
        ],
    )
    # e.jpg is also in A2 validation, so it becomes the cross-audit partition.
    frame = pd.read_csv(a2_val)
    frame.loc[len(frame)] = ["train/0000/e.jpg", 0]
    frame.to_csv(a2_val, index=False)

    paths = ["0000/a.jpg", "0001/b.jpg", "0000/c.jpg", "0001/d.jpg", "0000/e.jpg"]
    groups = {path: f"g{index}" for index, path in enumerate(paths)}
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(json.dumps(groups), encoding="utf-8")
    trust_path = tmp_path / "trust.pt"
    torch.save(
        {
            "paths": paths,
            "clean_probability": torch.ones(5),
            "pseudo_label": torch.full((5,), -1),
            "pseudo_confidence": torch.zeros(5),
            "correction_alpha": torch.zeros(5),
            "metadata": {"method": "test"},
        },
        trust_path,
    )
    rejected = tmp_path / "rejected.txt"
    rejected.write_text("train_dedup/0000/a.jpg\n", encoding="utf-8")

    manifest = prepare_a2_gate(
        a2_train_csv=a2_train,
        a2_val_csv=a2_val,
        aegis_train_csv=aegis_train,
        aegis_val_csv=aegis_val,
        content_groups_json=groups_path,
        trust_bundle_path=trust_path,
        a2_rejected_paths=rejected,
        output_dir=tmp_path / "out",
        clean_threshold=0.70,
        expected_classes=1,
    )

    assert manifest["partitions"] == {
        "adapt": 2,
        "evaluation": 2,
        "cross_audit": 1,
    }
    assert manifest["a2_rejected_in_adapt"] == 1
    combined = torch.load(
        tmp_path / "out" / "cvt_a2_union.pt", weights_only=False
    )
    assert combined["clean_probability"].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert set(pd.read_csv(tmp_path / "out" / "adapt_train.csv")["image_path"]) == {
        "train/0000/a.jpg",
        "train/0001/b.jpg",
    }


def test_prepare_a2_fullfit_physically_removes_rejects_and_adds_clean_val(
    tmp_path,
) -> None:
    a2_train = tmp_path / "a2_train.csv"
    a2_val = tmp_path / "a2_val.csv"
    _write_split(
        a2_train,
        [("train_dedup/0000/a.jpg", 0), ("train_dedup/0001/b.jpg", 1)],
    )
    _write_split(a2_val, [("train/0000/c.jpg", 0), ("train/0001/d.jpg", 1)])
    paths = ["0000/a.jpg", "0001/b.jpg", "0000/c.jpg", "0001/d.jpg"]
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(
        json.dumps({path: f"g{index}" for index, path in enumerate(paths)}),
        encoding="utf-8",
    )
    trust_path = tmp_path / "trust.pt"
    torch.save(
        {
            "paths": paths,
            "clean_probability": torch.tensor([0.9, 0.1, 0.8, 0.5]),
            "pseudo_label": torch.full((4,), -1),
            "pseudo_confidence": torch.zeros(4),
            "correction_alpha": torch.zeros(4),
        },
        trust_path,
    )
    rejected = tmp_path / "rejected.txt"
    rejected.write_text("train_dedup/0000/a.jpg\n", encoding="utf-8")

    manifest = prepare_a2_fullfit(
        a2_train_csv=a2_train,
        a2_val_csv=a2_val,
        content_groups_json=groups_path,
        trust_bundle_path=trust_path,
        a2_rejected_paths=rejected,
        output_dir=tmp_path / "fullfit",
        clean_threshold=0.70,
        expected_classes=2,
    )

    assert manifest["a2_replay_after_reject"] == 1
    assert manifest["a2_val_added_clean"] == 1
    assert manifest["a2_val_content_conflicts_excluded"] == 0
    assert set(
        pd.read_csv(tmp_path / "fullfit" / "fullfit_train.csv")["image_path"]
    ) == {"train/0001/b.jpg", "train/0000/c.jpg"}
    combined = torch.load(
        tmp_path / "fullfit" / "a2_fixed_fullfit_trust.pt", weights_only=False
    )
    assert torch.allclose(
        combined["clean_probability"], torch.tensor([0.0, 1.0, 0.8, 0.5])
    )


def test_prepare_a2_fullfit_excludes_heldout_content_collisions(tmp_path) -> None:
    a2_train = tmp_path / "a2_train.csv"
    a2_val = tmp_path / "a2_val.csv"
    _write_split(a2_train, [("train/0000/a.jpg", 0), ("train/0001/b.jpg", 1)])
    _write_split(a2_val, [("train/0000/c.jpg", 0), ("train/0001/d.jpg", 1)])
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(
        json.dumps(
            {
                "0000/a.jpg": "duplicate",
                "0001/b.jpg": "train-only",
                "0000/c.jpg": "duplicate",
                "0001/d.jpg": "val-only",
            }
        ),
        encoding="utf-8",
    )
    trust_path = tmp_path / "trust.pt"
    torch.save(
        {
            "paths": ["0000/a.jpg", "0001/b.jpg", "0000/c.jpg", "0001/d.jpg"],
            "clean_probability": torch.ones(4),
            "pseudo_label": torch.full((4,), -1),
            "pseudo_confidence": torch.zeros(4),
            "correction_alpha": torch.zeros(4),
        },
        trust_path,
    )
    rejected = tmp_path / "rejected.txt"
    rejected.write_text("", encoding="utf-8")

    manifest = prepare_a2_fullfit(
        a2_train_csv=a2_train,
        a2_val_csv=a2_val,
        content_groups_json=groups_path,
        trust_bundle_path=trust_path,
        a2_rejected_paths=rejected,
        output_dir=tmp_path / "fullfit",
        expected_classes=2,
    )

    selected = set(pd.read_csv(tmp_path / "fullfit" / "fullfit_train.csv")["image_path"])
    assert "train/0000/c.jpg" not in selected
    assert "train/0001/d.jpg" in selected
    assert manifest["a2_train_val_overlapping_content_groups"] == 1
    assert manifest["a2_val_content_conflicts_excluded"] == 1
