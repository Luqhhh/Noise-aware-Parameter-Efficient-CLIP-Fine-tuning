"""Tests for partial CLIP visual encoder unfreezing (D1).

Verifies:
  - freeze_clip=True: all visual frozen, classifier trainable
  - unfreeze_last_n_blocks: correct blocks unfrozen
  - train_ln_post / train_visual_proj: toggle per-component
  - train(mode): frozen blocks stay in eval, unfrozen in train mode
  - Invalid n values raise ValueError
"""

import torch
import torch.nn as nn

from experiments.baseline.model import CLIPLinearClassifier


def _make_mock_visual():
    """Build a minimal CLIP-like visual encoder for testing freeze/unfreeze.

    Structure mirrors CLIP ViT: conv1, transformer.resblocks (12 blocks),
    ln_post, proj. Each block uses Linear+LayerNorm (no BatchNorm) to
    match real CLIP ViT behavior.
    """
    class MockTransformerBlock(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, 2, batch_first=True)
            self.ln_1 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
            self.ln_2 = nn.LayerNorm(dim)

        def forward(self, x):
            # Simple residual path
            attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x

    visual = nn.Module()
    visual.conv1 = nn.Conv2d(3, 64, 3, padding=1)
    visual.transformer = nn.Module()
    visual.transformer.resblocks = nn.ModuleList(
        [MockTransformerBlock(64) for _ in range(12)]
    )
    visual.ln_post = nn.LayerNorm(64)
    visual.proj = nn.Parameter(torch.randn(64, 64))

    return visual


class MockCLIP(nn.Module):
    """Thin wrapper providing .visual attribute matching CLIP model."""

    def __init__(self):
        super().__init__()
        self.visual = _make_mock_visual()


def _make_model(mock_clip=None, **kwargs):
    """Build a CLIPLinearClassifier with test defaults.

    Args:
        mock_clip: Optional MockCLIP instance.
        **kwargs: Override for CLIPLinearClassifier.__init__ params.
    """
    if mock_clip is None:
        mock_clip = MockCLIP()
    defaults = dict(
        num_classes=500,
        feature_dim=64,
        freeze_clip=True,
        unfreeze_last_n_blocks=0,
        train_ln_post=False,
        train_visual_proj=False,
    )
    defaults.update(kwargs)
    return CLIPLinearClassifier(clip_model=mock_clip, **defaults)


# ── freeze_clip=True: all visual frozen ──────────────────────────────

class TestFreezeAll:
    def test_all_visual_frozen_when_freeze_clip_true(self):
        model = _make_model(freeze_clip=True)
        for name, param in model.visual.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_classifier_trainable_when_freeze_clip_true(self):
        model = _make_model(freeze_clip=True)
        for name, param in model.classifier.named_parameters():
            assert param.requires_grad, f"classifier.{name} must be trainable"


# ── unfreeze_last_n_blocks ──────────────────────────────────────────

class TestUnfreezeLastN:
    def test_n0_all_blocks_frozen(self):
        model = _make_model(freeze_clip=False, unfreeze_last_n_blocks=0)
        blocks = model.visual.transformer.resblocks
        for i, block in enumerate(blocks):
            for param in block.parameters():
                assert not param.requires_grad, f"block {i} should be frozen"

    def test_n1_only_last_block_trainable(self):
        model = _make_model(freeze_clip=False, unfreeze_last_n_blocks=1)
        blocks = model.visual.transformer.resblocks
        # First 11 frozen
        for i, block in enumerate(blocks[:-1]):
            for param in block.parameters():
                assert not param.requires_grad, f"block {i} should be frozen"
        # Last block trainable
        for param in blocks[-1].parameters():
            assert param.requires_grad, "last block must be trainable"

    def test_n2_last_two_blocks_trainable(self):
        model = _make_model(freeze_clip=False, unfreeze_last_n_blocks=2)
        blocks = model.visual.transformer.resblocks
        # First 10 frozen
        for i, block in enumerate(blocks[:-2]):
            for param in block.parameters():
                assert not param.requires_grad, f"block {i} should be frozen"
        # Last 2 trainable
        for block in blocks[-2:]:
            for param in block.parameters():
                assert param.requires_grad, "last 2 blocks must be trainable"

    def test_invalid_n_raises_valueerror(self):
        import pytest
        with pytest.raises(ValueError, match="unfreeze_last_n_blocks"):
            _make_model(freeze_clip=False, unfreeze_last_n_blocks=13)

    def test_negative_n_raises_valueerror(self):
        import pytest
        with pytest.raises(ValueError, match="unfreeze_last_n_blocks"):
            _make_model(freeze_clip=False, unfreeze_last_n_blocks=-1)


# ── ln_post and visual.proj toggles ──────────────────────────────────

class TestLnPostAndProj:
    def test_ln_post_trainable_when_enabled(self):
        model = _make_model(
            freeze_clip=False, train_ln_post=True, train_visual_proj=False
        )
        for param in model.visual.ln_post.parameters():
            assert param.requires_grad, "ln_post must be trainable"

    def test_ln_post_frozen_when_disabled(self):
        model = _make_model(
            freeze_clip=False, train_ln_post=False, train_visual_proj=False
        )
        for param in model.visual.ln_post.parameters():
            assert not param.requires_grad, "ln_post must be frozen"

    def test_proj_trainable_when_enabled(self):
        model = _make_model(
            freeze_clip=False, train_ln_post=False, train_visual_proj=True
        )
        assert model.visual.proj.requires_grad, "proj must be trainable"

    def test_proj_frozen_when_disabled(self):
        model = _make_model(
            freeze_clip=False, train_ln_post=False, train_visual_proj=False
        )
        assert not model.visual.proj.requires_grad, "proj must be frozen"


# ── train() mode behavior ────────────────────────────────────────────

class TestTrainMode:
    def test_frozen_clip_visual_stays_in_eval(self):
        model = _make_model(freeze_clip=True)
        model.train()
        assert not model.visual.training, "frozen visual must stay in eval"

    def test_partial_unfreeze_unfrozen_block_in_train_mode(self):
        model = _make_model(
            freeze_clip=False,
            unfreeze_last_n_blocks=1,
            train_ln_post=False,
            train_visual_proj=False,
        )
        model.train()
        blocks = model.visual.transformer.resblocks
        assert blocks[-1].training, "last block must be in train mode"

    def test_partial_unfreeze_frozen_block_in_eval_mode(self):
        model = _make_model(
            freeze_clip=False,
            unfreeze_last_n_blocks=1,
            train_ln_post=False,
            train_visual_proj=False,
        )
        model.train()
        blocks = model.visual.transformer.resblocks
        assert not blocks[0].training, "frozen block must be in eval mode"

    def test_ln_post_train_mode_when_unfrozen(self):
        model = _make_model(
            freeze_clip=False,
            unfreeze_last_n_blocks=0,
            train_ln_post=True,
            train_visual_proj=False,
        )
        model.train()
        assert model.visual.ln_post.training, "ln_post must be in train mode"

    def test_eval_mode_resets_all(self):
        model = _make_model(
            freeze_clip=False,
            unfreeze_last_n_blocks=1,
            train_ln_post=True,
            train_visual_proj=False,
        )
        model.train()
        model.eval()
        blocks = model.visual.transformer.resblocks
        assert not blocks[-1].training, "unfrozen block must be in eval after .eval()"
        assert not model.visual.ln_post.training, "ln_post must be in eval after .eval()"
