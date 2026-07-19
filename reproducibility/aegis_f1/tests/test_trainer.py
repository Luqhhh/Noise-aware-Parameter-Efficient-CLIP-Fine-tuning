import torch

from aegis_clip.trainer import _select_training_forward


def test_frozen_cached_batch_is_reported_as_cached_forward() -> None:
    features = torch.randn(4, 512)
    key, selected, cached = _select_training_forward(
        peft=False,
        input_key="features",
        mixed_inputs=features,
        mixed_reference=features,
        mix_lambda=1.0,
    )
    assert key == "features"
    assert selected is features
    assert cached is True


def test_frozen_unmixed_online_batch_reuses_frozen_features() -> None:
    images = torch.randn(4, 3, 8, 8)
    features = torch.randn(4, 512)
    key, selected, cached = _select_training_forward(
        peft=False,
        input_key="images",
        mixed_inputs=images,
        mixed_reference=features,
        mix_lambda=1.0,
    )
    assert key == "features"
    assert selected is features
    assert cached is True


def test_mixup_and_peft_never_skip_online_visual_forward() -> None:
    images = torch.randn(4, 3, 8, 8)
    features = torch.randn(4, 512)
    for peft, mix_lambda in ((False, 0.7), (True, 1.0)):
        key, selected, cached = _select_training_forward(
            peft=peft,
            input_key="images",
            mixed_inputs=images,
            mixed_reference=features,
            mix_lambda=mix_lambda,
        )
        assert key == "images"
        assert selected is images
        assert cached is False
