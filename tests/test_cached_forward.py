from types import SimpleNamespace

import torch
import torch.nn as nn

from experiments.baseline.model import CLIPLinearClassifier
from experiments.cosine.model import CosineClassifier


class DummyVisual(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=1)
        self.proj = nn.Linear(8, feature_dim)

    def forward(self, images):
        x = self.conv1(images)
        x = x.mean(dim=(2, 3))
        return self.proj(x)


def make_dummy_clip():
    return SimpleNamespace(visual=DummyVisual())


def test_linear_cached_forward_shape_and_grad():
    model = CLIPLinearClassifier(
        clip_model=make_dummy_clip(),
        num_classes=5,
        feature_dim=512,
        freeze_clip=True,
    )

    features = torch.randn(4, 512)
    logits = model.forward_features(features)

    assert logits.shape == (4, 5)

    logits.sum().backward()
    assert model.classifier.weight.grad is not None


def test_linear_online_and_feature_forward_match():
    model = CLIPLinearClassifier(
        clip_model=make_dummy_clip(),
        num_classes=5,
        feature_dim=512,
        freeze_clip=True,
    )
    model.eval()

    images = torch.randn(4, 3, 16, 16)

    with torch.no_grad():
        features = model.encode_image(images)
        online_logits = model(images)
        cached_logits = model.forward_features(features)

    torch.testing.assert_close(
        online_logits,
        cached_logits,
        rtol=1e-5,
        atol=1e-6,
    )


def test_cosine_cached_forward_shape_and_grad():
    model = CosineClassifier(
        clip_model=make_dummy_clip(),
        num_classes=5,
        feature_dim=512,
        freeze_clip=True,
        init_scale=10.0,
        learnable_scale=True,
    )

    features = torch.randn(4, 512)
    logits = model.forward_features(features)

    assert logits.shape == (4, 5)

    logits.sum().backward()
    assert model.weight.grad is not None
    assert model.logit_scale.grad is not None


def test_cosine_online_and_feature_forward_match():
    model = CosineClassifier(
        clip_model=make_dummy_clip(),
        num_classes=5,
        feature_dim=512,
        freeze_clip=True,
        init_scale=10.0,
        learnable_scale=True,
    )
    model.eval()

    images = torch.randn(4, 3, 16, 16)

    with torch.no_grad():
        features = model.encode_image(images)
        online_logits = model(images)
        cached_logits = model.forward_features(features)

    torch.testing.assert_close(
        online_logits,
        cached_logits,
        rtol=1e-5,
        atol=1e-6,
    )
