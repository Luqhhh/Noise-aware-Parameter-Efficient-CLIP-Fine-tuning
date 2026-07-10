"""Test CLIP utilities."""
import pytest
import torch
from common.clip_utils import (
    ALLOWED_MODEL_NAME,
    ALLOWED_PRETRAINED_SOURCE,
    load_openai_clip,
    encode_frozen_clip_features,
)


def test_load_openai_clip_rejects_wrong_model():
    """Non-ViT-B/32 model name -> ValueError."""
    with pytest.raises(ValueError, match="ViT-B/32"):
        load_openai_clip(torch.device("cpu"), model_name="RN50")


def test_load_openai_clip_rejects_wrong_source():
    """Non-openai pretrained source -> ValueError."""
    with pytest.raises(ValueError, match="OpenAI"):
        load_openai_clip(torch.device("cpu"), pretrained_source="laion")


def test_load_openai_clip_accepts_defaults():
    """Default args should not raise."""
    device = torch.device("cpu")
    model, preprocess = load_openai_clip(device)
    assert model is not None
    assert preprocess is not None


def test_encode_frozen_clip_features_output_shape():
    """Output should be (batch, 512), L2-normalized."""
    device = torch.device("cpu")
    model, preprocess = load_openai_clip(device)
    model = model.float()  # ensure float32

    dummy_images = torch.randn(4, 3, 224, 224)
    features = encode_frozen_clip_features(model, dummy_images, device, use_amp=False)

    assert features.shape == (4, 512)
    norms = features.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_encode_frozen_clip_features_no_grad():
    """Encoding should not track gradients."""
    device = torch.device("cpu")
    model, preprocess = load_openai_clip(device)
    model = model.float()

    dummy_images = torch.randn(4, 3, 224, 224)
    features = encode_frozen_clip_features(model, dummy_images, device)
    assert not features.requires_grad
