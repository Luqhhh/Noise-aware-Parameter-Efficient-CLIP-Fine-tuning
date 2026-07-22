import json

import pandas as pd

from aegis_clip.cli.audit_label_structure import audit_label_structure


def test_audit_label_structure_separates_consensus_and_strict_tiers(tmp_path):
    quality = pd.DataFrame(
        {
            "image_path": ["train/0000/a.jpg", "train/0001/b.jpg", "train/0002/c.jpg"],
            "original_label": [0, 1, 2],
            "oof_top1": [1, 1, 0],
            "p_top1": [0.95, 0.90, 0.92],
            "top1_margin": [0.80, 0.70, 0.75],
            "prototype_top1": [1, 1, 2],
            "knn_top1": [1, 1, 0],
            "knn_top1_agreement": [0.80, 0.90, 0.70],
            "flip_consistency": [1.0, 1.0, 1.0],
            "duplicate_conflict_flag": [False, False, False],
        }
    )
    issues = pd.DataFrame(
        {"index": [0, 2], "selected": [True, True]}
    )
    a2 = pd.DataFrame(
        {
            "image_path": quality["image_path"],
            "training_role": ["rejected", "clean", "clean"],
        }
    )
    quality_path = tmp_path / "quality.csv"
    issues_path = tmp_path / "issues.csv"
    a2_path = tmp_path / "a2.csv"
    output_path = tmp_path / "audit.json"
    quality.to_csv(quality_path, index=False)
    issues.to_csv(issues_path, index=False)
    a2.to_csv(a2_path, index=False)

    result = audit_label_structure(
        quality_path=quality_path,
        issues_path=issues_path,
        a2_manifest_path=a2_path,
        output_path=output_path,
        num_classes=3,
    )

    assert result["oof_disagreement"]["count"] == 2
    assert result["moderate_relabel"]["count"] == 2
    assert result["strict_relabel"]["count"] == 1
    assert result["a2_rejected_overlap"]["with_strict_relabel"] == 1
    assert result["strict_relabel_flow"]["l1_count_shift"] == 2
    assert json.loads(output_path.read_text())["sample_count"] == 3
