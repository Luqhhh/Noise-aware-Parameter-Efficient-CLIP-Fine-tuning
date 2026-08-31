from __future__ import annotations

import json

import torch

from aegis_clip.cli.sweep_prior_strength import sweep_prior_strength


def _validation_cache(tmp_path, num_samples: int, num_classes: int) -> str:
    generator = torch.Generator().manual_seed(3)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    logits = torch.randn(num_samples, num_classes, generator=generator)
    payload = {
        "logits": logits,
        "labels": labels,
        "clean_probability": torch.ones(num_samples),
        "pseudo_labels": labels.clone(),
        "correction_alpha": torch.zeros(num_samples),
        "paths": [f"img_{index}.jpg" for index in range(num_samples)],
    }
    path = tmp_path / "validation_logits.pt"
    torch.save(payload, path)
    return str(path)


def test_sweep_prior_strength_fits_on_validation_and_freezes(tmp_path) -> None:
    cache = _validation_cache(tmp_path, num_samples=200, num_classes=8)
    config_path = sweep_prior_strength(
        cache,
        tmp_path / "prior",
        strengths=(0.0, 0.5, 1.0),
        selector_metric="raw_micro",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["format_version"] == 1
    assert config["num_classes"] == 8
    assert len(config["bias"]) == 8
    assert config["strength"] in (0.0, 0.5, 1.0)
    assert config["test_data_used"] is False
    assert config["fitted_on"] == "current_stage_validation"
    assert set(config["sweep"]) == {
        "strength_0",
        "strength_0.5",
        "strength_1",
    }
