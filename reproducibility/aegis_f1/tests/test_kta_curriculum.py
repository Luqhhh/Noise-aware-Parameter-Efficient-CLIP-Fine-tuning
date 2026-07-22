import json

import pandas as pd
import torch

from aegis_clip.kta_curriculum import build_kta_curriculum_bundle


def test_build_kta_curriculum_bundle_rescues_only_strict_consensus(tmp_path):
    paths = ["train/0000/a.jpg", "train/0001/b.jpg", "train/0002/c.jpg"]
    base = {
        "paths": paths,
        "clean_probability": torch.tensor([0.0, 0.4, 0.8]),
        "pseudo_label": torch.tensor([2, 2, 1]),
        "pseudo_confidence": torch.tensor([0.9, 0.8, 0.7]),
        "correction_alpha": torch.tensor([0.4, 0.3, 0.2]),
    }
    base_path = tmp_path / "base.pt"
    torch.save(base, base_path)
    quality = pd.DataFrame(
        {
            "image_path": paths,
            "original_label": [0, 1, 2],
            "oof_top1": [1, 1, 0],
            "p_top1": [0.95, 0.95, 0.95],
            "top1_margin": [0.80, 0.80, 0.80],
            "prototype_top1": [1, 1, 2],
            "knn_top1": [1, 1, 0],
            "knn_top1_agreement": [0.80, 0.80, 0.80],
            "flip_consistency": [1.0, 1.0, 1.0],
            "duplicate_conflict_flag": [False, False, False],
        }
    )
    issues = pd.DataFrame({"index": [0, 2], "selected": [True, True]})
    quality_path = tmp_path / "quality.csv"
    issues_path = tmp_path / "issues.csv"
    output_path = tmp_path / "kta.pt"
    manifest_path = tmp_path / "manifest.json"
    quality.to_csv(quality_path, index=False)
    issues.to_csv(issues_path, index=False)

    manifest = build_kta_curriculum_bundle(
        base_bundle_path=base_path,
        quality_path=quality_path,
        issues_path=issues_path,
        output_bundle_path=output_path,
        output_manifest_path=manifest_path,
    )
    payload = torch.load(output_path, map_location="cpu", weights_only=False)

    assert manifest["strict_corrected"] == 1
    assert manifest["rescued_a2_rejects"] == 1
    assert manifest["trusted_original"] == 1
    assert torch.allclose(
        payload["clean_probability"], torch.tensor([1.0, 1.0, 0.8])
    )
    assert payload["pseudo_label"].tolist() == [1, -1, -1]
    assert payload["correction_alpha"].tolist() == [1.0, 0.0, 0.0]
    assert json.loads(manifest_path.read_text())["strict_corrected"] == 1
