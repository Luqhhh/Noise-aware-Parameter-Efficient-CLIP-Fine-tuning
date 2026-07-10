"""Test cosine classifier."""
import pytest
import torch
from experiments.cosine.model import CosineClassifier


class MockVisual(torch.nn.Module):
    """Mock CLIP visual encoder."""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 768, 1)  # Minimal conv for dtype check

    def forward(self, x):
        return torch.randn(x.size(0), 512)  # Return features directly


class MockCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = MockVisual()


def test_cosine_no_bias():
    """Cosine classifier should have no bias parameter."""
    model = CosineClassifier(MockCLIP(), num_classes=10, feature_dim=512)
    assert model.weight is not None
    assert not hasattr(model, 'bias')


def test_cosine_fixed_scale():
    """Fixed scale: logit_scale should be a buffer, not parameter."""
    model = CosineClassifier(
        MockCLIP(), num_classes=10, feature_dim=512,
        init_scale=10.0, learnable_scale=False
    )
    assert not isinstance(model.logit_scale, torch.nn.Parameter)
    assert model.logit_scale.item() == 10.0
    # clamp_scale should be no-op (returns None, value unchanged)
    model.clamp_scale()
    assert model.logit_scale.item() == 10.0


def test_cosine_learnable_scale():
    """Learnable scale: logit_scale should be a Parameter."""
    model = CosineClassifier(
        MockCLIP(), num_classes=10, feature_dim=512,
        init_scale=10.0, learnable_scale=True
    )
    assert isinstance(model.logit_scale, torch.nn.Parameter)
    assert model.logit_scale.item() == 10.0


def test_cosine_clamp():
    """Clamping should work when scale is learnable."""
    model = CosineClassifier(
        MockCLIP(), num_classes=10, feature_dim=512,
        init_scale=10.0, learnable_scale=True
    )
    # Manually set scale to extreme values
    with torch.no_grad():
        model.logit_scale.fill_(200.0)
    model.clamp_scale()
    assert model.logit_scale.item() == 100.0

    with torch.no_grad():
        model.logit_scale.fill_(0.1)
    model.clamp_scale()
    assert model.logit_scale.item() == 1.0


def test_cosine_init_scale_validation():
    """Invalid init_scale -> ValueError."""
    with pytest.raises(ValueError, match="positive"):
        CosineClassifier(MockCLIP(), num_classes=10, init_scale=-1.0)
    with pytest.raises(ValueError, match="<= 100"):
        CosineClassifier(MockCLIP(), num_classes=10, init_scale=200.0)


def test_cosine_forward_shape():
    """Forward pass should produce (B, num_classes) logits."""
    model = CosineClassifier(MockCLIP(), num_classes=10, feature_dim=512)
    model = model.float()
    images = torch.randn(4, 3, 224, 224)
    logits = model(images)
    assert logits.shape == (4, 10)


def test_cosine_param_groups():
    """Check param group structure."""
    model = CosineClassifier(
        MockCLIP(), num_classes=10, feature_dim=512,
        learnable_scale=True
    )
    groups = model.get_param_groups(lr=0.001, weight_decay=0.0001)
    assert len(groups) == 2  # weight group + scale group
    # Scale group has lower lr, no wd
    assert groups[1]["lr"] == 0.0001
    assert groups[1]["weight_decay"] == 0.0


def test_cosine_param_groups_fixed_scale():
    """Fixed scale: only weight group, no scale group."""
    model = CosineClassifier(
        MockCLIP(), num_classes=10, feature_dim=512,
        learnable_scale=False
    )
    groups = model.get_param_groups(lr=0.001, weight_decay=0.0001)
    assert len(groups) == 1  # Only weight group


def test_cosine_train_mode_keeps_backbone_eval():
    """When freeze_clip=True, calling train() should keep visual in eval."""
    model = CosineClassifier(MockCLIP(), freeze_clip=True)
    model.train()
    assert model.training  # Classifier head in train mode
    assert not model.visual.training  # Backbone stays in eval
