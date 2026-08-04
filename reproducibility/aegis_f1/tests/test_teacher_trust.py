import torch

from aegis_clip.teacher_trust import augment_teacher_trust


def _base(size: int) -> dict:
    return {
        "paths": [f"0000/{index}.jpg" for index in range(size)],
        "clean_probability": torch.full((size,), 0.2),
        "pseudo_label": torch.full((size,), -1, dtype=torch.long),
        "pseudo_confidence": torch.zeros(size),
        "correction_alpha": torch.zeros(size),
        "diagnostics": {"preserved": torch.arange(size)},
    }


def test_teacher_correction_requires_two_view_confident_disagreement() -> None:
    base = _base(3)
    labels = torch.tensor([0, 0, 0])
    center = torch.tensor([[0.0, 10.0], [0.0, 10.0], [0.0, 2.0]])
    flip = torch.tensor([[0.0, 10.0], [10.0, 0.0], [0.0, 2.0]])
    output, audit = augment_teacher_trust(
        base,
        labels,
        center,
        flip,
        maximum_class_fraction=1.0,
    )
    assert audit["accepted_teacher_corrections"] == 1
    assert output["pseudo_label"].tolist() == [1, -1, -1]
    assert output["correction_alpha"].tolist() == [0.5, 0.0, 0.0]
    assert output["diagnostics"]["preserved"].tolist() == [0, 1, 2]


def test_teacher_does_not_replace_existing_oof_correction() -> None:
    base = _base(1)
    base["pseudo_label"][0] = 1
    base["correction_alpha"][0] = 0.3
    output, audit = augment_teacher_trust(
        base,
        torch.tensor([0]),
        torch.tensor([[0.0, 10.0]]),
        torch.tensor([[0.0, 10.0]]),
        maximum_class_fraction=1.0,
    )
    assert audit["accepted_teacher_corrections"] == 0
    assert torch.equal(output["correction_alpha"], base["correction_alpha"])


def test_teacher_applies_source_and_target_class_caps() -> None:
    base = _base(6)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    center = torch.tensor([[0.0, 10.0]] * 3 + [[10.0, 0.0]] * 3)
    output, audit = augment_teacher_trust(
        base,
        labels,
        center,
        center,
        maximum_class_fraction=0.34,
    )
    assert audit["maximum_per_class_limit"] == 2
    assert audit["accepted_teacher_corrections"] == 4
    assert int((output["correction_alpha"] > 0).sum()) == 4
