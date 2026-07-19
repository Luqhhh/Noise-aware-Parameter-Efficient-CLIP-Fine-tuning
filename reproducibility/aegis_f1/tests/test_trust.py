import torch

from aegis_clip.trust import (
    TrustBuildConfig,
    build_cross_fitted_trust,
    cap_classwise_corrections,
)


def test_classwise_correction_cap() -> None:
    labels = torch.tensor([0] * 10 + [1] * 10)
    alpha = torch.linspace(0.01, 1.0, 20)
    result = cap_classwise_corrections(alpha, labels, maximum_rate=0.2)
    assert int((result[labels == 0] > 0).sum()) == 2
    assert int((result[labels == 1] > 0).sum()) == 2


def test_cross_fitted_trust_covers_every_sample() -> None:
    generator = torch.Generator().manual_seed(5)
    classes = 3
    per_class = 18
    dimension = 8
    centers = torch.nn.functional.normalize(
        torch.randn(classes, dimension, generator=generator), dim=1
    )
    labels = torch.arange(classes).repeat_interleave(per_class)
    features = centers[labels] + 0.08 * torch.randn(
        classes * per_class, dimension, generator=generator
    )
    paths = [f"{int(label):04d}/{index}.jpg" for index, label in enumerate(labels)]
    config = TrustBuildConfig(
        folds=3,
        probe_epochs=1,
        probe_batch_size=64,
        maximum_class_correction_rate=0.2,
    )
    bundle, summary = build_cross_fitted_trust(
        features,
        labels,
        paths,
        num_classes=classes,
        config=config,
        device="cpu",
    )
    assert len(bundle["paths"]) == classes * per_class
    assert (bundle["diagnostics"]["fold_id"] >= 0).all()
    assert torch.isfinite(bundle["clean_probability"]).all()
    assert 0.0 <= summary["mean_clean_probability"] <= 1.0
