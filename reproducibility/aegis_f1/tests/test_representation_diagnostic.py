import torch

from aegis_clip.representation_diagnostic import diagnose_representation_shift


def _cache(predictions: list[int]) -> dict[str, object]:
    logits = torch.full((8, 4), -2.0)
    logits[torch.arange(8), torch.tensor(predictions)] = 2.0
    return {
        "paths": [f"v{index}.jpg" for index in range(8)],
        "labels": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        "clean_probability": torch.ones(8),
        "logits": logits,
    }


def test_representation_diagnostic_counts_persistent_recovery() -> None:
    a2_center = _cache([0, 1, 1, 0, 2, 0, 3, 0])
    a2_m1 = _cache([0, 1, 1, 0, 2, 0, 3, 0])
    a2_m3 = _cache([0, 1, 1, 0, 2, 0, 3, 0])
    n3_center = _cache([0, 0, 1, 0, 2, 2, 3, 0])
    n3_m3 = _cache([0, 0, 1, 1, 2, 2, 3, 0])
    train = {
        "labels": torch.tensor([0, 1, 1, 2, 2, 2, 3, 3]),
        "paths": [f"t{index}.jpg" for index in range(8)],
    }

    report = diagnose_representation_shift(
        a2_center,
        a2_m1,
        a2_m3,
        n3_center,
        n3_m3,
        train,
        num_classes=4,
    )

    assert report["previously_persistent_errors"]["a2_center_m1_m3_all_wrong"] == 4
    assert report["previously_persistent_errors"]["recovered_by_n3_m3"] == 3
    assert report["transitions"]["a2_m3_to_n3_m3"]["net_correct"] == 3
    assert report["class_effects"]["classes_improved"] == 3
