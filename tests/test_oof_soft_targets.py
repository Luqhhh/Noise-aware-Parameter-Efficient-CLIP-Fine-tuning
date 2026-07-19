import pandas as pd
import pytest
import torch

from common.oof_targets import OOFSoftTargetProvider


def _write_inputs(tmp_path):
    logits_path = tmp_path / "oof_logits.pt"
    torch.save(
        {
            "sample_ids": ["s1", "s2"],
            "logits": torch.tensor([[4.0, 0.0], [0.0, 3.0]]),
            "folds": torch.tensor([0, 1]),
        },
        logits_path,
    )
    quality_path = tmp_path / "sample_quality.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "image_path": [
                "train_dedup/0000/a.jpg",
                "train_dedup/0001/b.jpg",
            ],
            "soft_weight": [0.8, 0.4],
        }
    ).to_csv(quality_path, index=False)
    return logits_path, quality_path


def test_oof_targets_map_stable_image_keys_and_return_probabilities(tmp_path):
    logits_path, quality_path = _write_inputs(tmp_path)
    provider = OOFSoftTargetProvider(
        str(logits_path), str(quality_path), min_weight=0.6, max_weight=1.0
    )

    targets, weights = provider.get_batch(
        ["/different/root/0000/a.jpg", "/different/root/0001/b.jpg"],
        device=torch.device("cpu"),
        temperature=2.0,
    )

    assert targets.shape == (2, 2)
    assert torch.allclose(targets.sum(dim=1), torch.ones(2))
    assert weights.tolist() == [0.8, 0.6]
    assert targets[0, 0] > targets[0, 1]
    assert targets[1, 1] > targets[1, 0]


def test_oof_targets_fail_closed_on_missing_path(tmp_path):
    logits_path, quality_path = _write_inputs(tmp_path)
    provider = OOFSoftTargetProvider(str(logits_path), str(quality_path))

    with pytest.raises(KeyError, match="missing OOF soft target"):
        provider.get_batch(["0002/missing.jpg"], torch.device("cpu"), 1.0)
