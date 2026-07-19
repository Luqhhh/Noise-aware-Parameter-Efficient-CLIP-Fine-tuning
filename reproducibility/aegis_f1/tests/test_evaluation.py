from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from aegis_clip.evaluation import evaluate


class CountingImageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        *,
        images: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        self.calls += 1
        encoded = (
            images.flatten(1)[:, :2].float()
            if images is not None
            else features[:, :2].float()
        )
        logits = encoded
        return (logits, encoded) if return_features else logits


def _batch(*, images: bool) -> dict[str, torch.Tensor]:
    batch = {
        "label": torch.tensor(0),
        "clean_probability": torch.tensor(1.0),
        "pseudo_label": torch.tensor(0),
        "correction_alpha": torch.tensor(0.0),
    }
    if images:
        batch["images"] = torch.tensor(
            [[[1.0, 0.0], [0.0, 0.0]]]
        )
    else:
        batch["features"] = torch.tensor([1.0, 0.0])
    return batch


def test_horizontal_flip_tta_runs_two_image_forwards() -> None:
    model = CountingImageModel()
    metrics = evaluate(
        model,
        DataLoader([_batch(images=True)], batch_size=1),
        device=torch.device("cpu"),
        num_classes=2,
        use_amp=False,
        selector_metric="raw_micro",
        tta_mode="horizontal_flip",
    )
    assert model.calls == 2
    assert metrics["inference_mode"] == "horizontal_flip"
    assert metrics["tta_fusion"] == "mean_logits"


def test_horizontal_flip_tta_rejects_cached_features() -> None:
    model = CountingImageModel()
    try:
        evaluate(
            model,
            DataLoader([_batch(images=False)], batch_size=1),
            device=torch.device("cpu"),
            num_classes=2,
            use_amp=False,
            tta_mode="horizontal_flip",
        )
    except ValueError as exc:
        assert "online image batches" in str(exc)
    else:
        raise AssertionError("cached-feature TTA must fail closed")


class _LeftRightModel(torch.nn.Module):
    def forward(
        self,
        *,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = images.flatten(start_dim=1).float()
        logits = features[:, :2]
        encoded = F.normalize(features[:, :2], dim=1)
        assert return_features
        return logits, encoded


def test_clean_core_and_flip_agreement_are_measured_independently() -> None:
    records = [
        {
            "images": torch.tensor([[[1.0, 0.0]]]),
            "label": torch.tensor(0),
            "clean_probability": torch.tensor(0.9),
            "pseudo_label": torch.tensor(0),
            "correction_alpha": torch.tensor(0.0),
        },
        {
            "images": torch.tensor([[[1.0, 1.0]]]),
            "label": torch.tensor(0),
            "clean_probability": torch.tensor(0.2),
            "pseudo_label": torch.tensor(0),
            "correction_alpha": torch.tensor(0.0),
        },
        {
            "images": torch.tensor([[[0.0, 2.0]]]),
            "label": torch.tensor(1),
            "clean_probability": torch.tensor(0.8),
            "pseudo_label": torch.tensor(1),
            "correction_alpha": torch.tensor(0.0),
        },
    ]
    metrics = evaluate(
        _LeftRightModel(),
        DataLoader(records, batch_size=3),
        torch.device("cpu"),
        num_classes=2,
        use_amp=False,
        selector_metric="clean_core_micro",
        clean_core_threshold=0.8,
        measure_flip_consistency=True,
    )

    assert metrics["clean_core_samples"] == 2
    assert metrics["clean_core_micro"] == pytest.approx(1.0)
    assert metrics["flip_prediction_agreement"] == pytest.approx(1.0 / 3.0)
    assert metrics["selector"] == pytest.approx(1.0)
