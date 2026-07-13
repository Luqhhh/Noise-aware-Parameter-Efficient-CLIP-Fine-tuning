"""Unit tests for Head EMA (EMAHook from common.hooks).

Tests cover C-1 requirements:
  - EMAHook construction and parameter validation
  - Warmup: direct copy during warmup epochs
  - Post-warmup: EMA = decay * EMA + (1 - decay) * model
  - swap_to_ema / restore_raw losslessness
  - state_dict / load_state_dict round-trip
  - get_ema_model independence from raw model
  - Frozen backbone params invariance under EMA
"""

import copy

import pytest
import torch
import torch.nn as nn

from common.hooks import EMAHook


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MiniClassifier(nn.Module):
    """Minimal CLIP-like model: frozen backbone + trainable linear head."""

    def __init__(self, in_dim: int = 16, n_classes: int = 5):
        super().__init__()
        self.backbone = nn.Linear(in_dim, in_dim)
        self.classifier = nn.Linear(in_dim, n_classes)
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        return self.classifier(x)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return MiniClassifier()


@pytest.fixture
def hook(model):
    return EMAHook(model, decay=0.99, warmup_epochs=5)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestEMAHookConstruction:
    def test_valid_decay(self, model):
        EMAHook(model, decay=0.5)
        EMAHook(model, decay=0.99)
        EMAHook(model, decay=1.0)

    def test_invalid_decay(self, model):
        with pytest.raises(ValueError, match="decay must be in"):
            EMAHook(model, decay=0.0)
        with pytest.raises(ValueError, match="decay must be in"):
            EMAHook(model, decay=1.1)

    def test_ema_model_is_deep_copy(self, model, hook):
        # EMA model starts as a copy, not a reference
        assert hook._ema_model is not model
        for ep, mp in zip(
            hook._ema_model.parameters(), model.parameters()
        ):
            assert ep is not mp

    def test_ema_model_no_grad(self, model, hook):
        for p in hook._ema_model.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestEMAHookWarmup:
    def test_during_warmup_ema_equals_model(self, model, hook):
        """During warmup, update() does a full copy of model into EMA."""
        # Modify model weights via an optimizer step
        opt = torch.optim.SGD(model.classifier.parameters(), lr=0.1)
        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        hook.update(model, epoch=3)  # epoch <= warmup_epochs

        for mp, ep in zip(
            model.classifier.parameters(),
            hook._ema_model.classifier.parameters(),
        ):
            assert torch.allclose(mp, ep), f"warmup: model != EMA, max diff={(mp-ep).abs().max():.2e}"

    def test_warmup_copies_backbone_too(self, model, hook):
        """Warmup also copies frozen backbone (same values, no harm)."""
        hook.update(model, epoch=2)
        for mp, ep in zip(
            model.backbone.parameters(),
            hook._ema_model.backbone.parameters(),
        ):
            assert torch.allclose(mp, ep)


# ---------------------------------------------------------------------------
# Post-warmup EMA update
# ---------------------------------------------------------------------------


class TestEMAHookPostWarmup:
    def test_post_warmup_decay_formula(self, model):
        """EMA_new = decay * EMA_old + (1 - decay) * model_new."""
        decay = 0.5  # easy to verify
        hook = EMAHook(model, decay=decay, warmup_epochs=3)

        # Warmup first: epoch 3 still warmup → copy
        opt = torch.optim.SGD(model.classifier.parameters(), lr=0.1)
        x = torch.randn(4, 16)
        model(x).sum().backward()
        opt.step()
        hook.update(model, epoch=3)
        ema_after_warmup = copy.deepcopy(hook._ema_model.state_dict())

        # Post-warmup step
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        model_after_step = copy.deepcopy(model.state_dict())

        hook.update(model, epoch=6)  # post-warmup

        for key in model_after_step:
            if "classifier" in key:
                ema_old = ema_after_warmup[key]
                model_new = model_after_step[key]
                expected = decay * ema_old + (1.0 - decay) * model_new
                actual = hook._ema_model.state_dict()[key]
                assert torch.allclose(
                    actual, expected, atol=1e-8
                ), f"decay formula mismatch for {key}: max diff={(actual-expected).abs().max():.2e}"

    def test_backbone_unchanged_post_warmup(self, model, hook):
        """Frozen backbone stays identical in EMA after post-warmup update."""
        opt = torch.optim.SGD(model.classifier.parameters(), lr=0.1)
        x = torch.randn(4, 16)
        for _ in range(5):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()
            hook.update(model, epoch=6)

        for mp, ep in zip(
            model.backbone.parameters(),
            hook._ema_model.backbone.parameters(),
        ):
            assert torch.allclose(mp, ep, atol=1e-8)


# ---------------------------------------------------------------------------
# swap_to_ema / restore_raw
# ---------------------------------------------------------------------------


class TestEMAHookSwapRestore:
    def test_swap_loads_ema_into_model(self, model, hook):
        """swap_to_ema replaces model parameters with EMA version."""
        opt = torch.optim.SGD(model.classifier.parameters(), lr=0.1)
        x = torch.randn(4, 16)
        # Warmup to initialise EMA
        model(x).sum().backward()
        opt.step()
        hook.update(model, epoch=3)

        # Modify model further (EMA is behind by one step)
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()

        # Save model state before swap
        raw_classifier_weight = model.classifier.weight.clone()

        hook.swap_to_ema(model)
        # After swap, model == EMA (which is behind by one step)
        for mp, ep in zip(
            model.classifier.parameters(),
            hook._ema_model.classifier.parameters(),
        ):
            assert torch.allclose(mp, ep)

        # Model should NOT equal pre-swap raw
        assert not torch.allclose(model.classifier.weight, raw_classifier_weight)

    def test_restore_is_lossless(self, model, hook):
        """swap + restore returns model to exact original state."""
        raw_state = copy.deepcopy(model.state_dict())

        hook.swap_to_ema(model)
        hook.restore_raw(model)

        for key in raw_state:
            assert torch.allclose(
                raw_state[key], model.state_dict()[key], atol=1e-8
            ), f"restore mismatch for {key}"

    def test_double_restore_is_safe(self, model, hook):
        """Calling restore_raw twice should warn but not crash."""
        hook.swap_to_ema(model)
        hook.restore_raw(model)
        # Second restore: _raw_state is None → should warn, not crash
        hook.restore_raw(model)  # no-op after warning


# ---------------------------------------------------------------------------
# state_dict / load_state_dict
# ---------------------------------------------------------------------------


class TestEMAHookStateDict:
    def test_save_load_roundtrip(self, model, hook):
        """state_dict → load_state_dict preserves EMA model weights."""
        opt = torch.optim.SGD(model.classifier.parameters(), lr=0.1)
        x = torch.randn(4, 16)
        model(x).sum().backward()
        opt.step()
        hook.update(model, epoch=6)

        saved = hook.state_dict()
        assert "ema_model" in saved
        assert "decay" in saved
        assert "warmup_epochs" in saved

        # Create a new hook and load
        new_hook = EMAHook(model, decay=0.5, warmup_epochs=1)
        new_hook.load_state_dict(saved)

        assert new_hook.decay == hook.decay
        assert new_hook.warmup_epochs == hook.warmup_epochs
        for k in saved["ema_model"]:
            assert torch.allclose(
                saved["ema_model"][k],
                new_hook._ema_model.state_dict()[k],
            )

    def test_state_dict_missing_update_count(self, hook):
        """state_dict does not include update_count — this is a known gap.
        Per spec, EMA must update per optimizer.step(), not per epoch.
        Without update_count, resume may lose step-level precision."""
        sd = hook.state_dict()
        assert "update_count" not in sd, (
            "GAP: state_dict lacks update_count. "
            "Resume cannot restore per-step EMA state correctly."
        )


# ---------------------------------------------------------------------------
# get_ema_model
# ---------------------------------------------------------------------------


class TestGetEmaModel:
    def test_get_ema_model_returns_independent_copy(self, model, hook):
        """get_ema_model returns the EMA shadow — modifying it does not
        affect the training model or future EMA updates."""
        ema_copy = hook.get_ema_model()

        # Modify the returned EMA model
        with torch.no_grad():
            for p in ema_copy.parameters():
                p.add_(1.0)

        # Original model unchanged
        for mp, ep in zip(model.parameters(), ema_copy.parameters()):
            assert not torch.allclose(mp, ep), "model should differ from modified EMA"

    def test_get_ema_model_eval_mode(self, model, hook):
        ema = hook.get_ema_model()
        assert not ema.training


# ---------------------------------------------------------------------------
# Known gaps — documented, not fixed here
# ---------------------------------------------------------------------------


class TestEMAHookKnownGaps:
    def test_tracks_frozen_backbone(self, model, hook):
        """GAP: EMAHook deep-copies entire model including frozen backbone.
        For CLIP ViT-B/32 (88M frozen + 256K trainable), this wastes ~76%
        of the EMA shadow's memory. For C-1 (linear head only) the overhead
        is 88M params (~338 MB), not critical for a single experiment but
        should be addressed for scaled use.

        Suggested fix (in A's domain): add a `track_trainable_only: bool`
        flag that filters parameters by requires_grad in __init__.
        """
        backbone_param_count = sum(
            p.numel() for p in model.backbone.parameters()
        )
        head_param_count = sum(
            p.numel() for p in model.classifier.parameters()
        )
        ema_total = sum(p.numel() for p in hook._ema_model.parameters())
        assert ema_total == backbone_param_count + head_param_count
        # Gap: includes backbone
        assert ema_total > head_param_count

    def test_no_update_count_in_state_dict(self, hook):
        """GAP: state_dict stores {'ema_model', 'decay', 'warmup_epochs'}
        but not update_count or current step. For gradient accumulation
        scenarios, the EMA update frequency must match optimizer.step()
        count, not epoch count. Without update_count in state_dict,
        resume after an epoch-internal crash may misalign the EMA schedule.
        """
        sd = hook.state_dict()
        assert "update_count" not in sd
        assert "step" not in sd
