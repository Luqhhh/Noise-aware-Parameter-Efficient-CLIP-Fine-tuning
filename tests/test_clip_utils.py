"""Test CLIP utilities."""
import builtins
import sys
from unittest.mock import patch

import pytest
import torch
from common.clip_utils import (
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


@pytest.mark.integration
def test_load_openai_clip_accepts_defaults():
    """Default args should not raise."""
    device = torch.device("cpu")
    model, preprocess = load_openai_clip(device)
    assert model is not None
    assert preprocess is not None


@pytest.mark.integration
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


@pytest.mark.integration
def test_encode_frozen_clip_features_no_grad():
    """Encoding should not track gradients."""
    device = torch.device("cpu")
    model, preprocess = load_openai_clip(device)
    model = model.float()

    dummy_images = torch.randn(4, 3, 224, 224)
    features = encode_frozen_clip_features(model, dummy_images, device)
    assert not features.requires_grad


@pytest.mark.integration
def test_encode_frozen_clip_features_with_amp():
    """AMP-enabled encoding should not crash and produce correct shape/normalization."""
    device = torch.device("cpu")
    model, preprocess = load_openai_clip(device)

    dummy_images = torch.randn(2, 3, 224, 224)
    features = encode_frozen_clip_features(model, dummy_images, device, use_amp=True)

    assert features.shape == (2, 512)
    norms = features.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)
    assert not features.requires_grad


def test_load_openai_clip_missing_clip_package():
    """ImportError is raised when clip is not available."""
    # Remove any cached clip module entries so we force a re-import
    for mod in list(sys.modules.keys()):
        if "clip" in mod:
            sys.modules.pop(mod, None)

    # Patch builtins.__import__ to make "import clip" raise ImportError
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "clip" or (isinstance(name, str) and name.startswith("clip.")):
            raise ImportError("No module named 'clip'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="pip install"):
            load_openai_clip(torch.device("cpu"))
