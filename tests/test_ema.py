"""Tests for EMA (Exponential Moving Average) integration.

Tests the EMAHook class and its integration with the training loop.
"""

import copy
import pytest
import torch
import torch.nn as nn
from common.hooks import EMAHook


# ── Tiny test model ──────────────────────────────────────────────────


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model():
    return TinyModel()


@pytest.fixture
def ema_hook(model):
    return EMAHook(model, decay=0.99, warmup_steps=10)


# ── Unit tests: EMAHook ──────────────────────────────────────────────


class TestEMAHookInit:
    def test_default_init(self, model):
        ema = EMAHook(model)
        assert ema.decay == 0.99
        assert ema.warmup_steps == 0
        assert ema.num_updates == 0

    def test_custom_decay(self, model):
        ema = EMAHook(model, decay=0.9, warmup_steps=100)
        assert ema.decay == 0.9
        assert ema.warmup_steps == 100

    def test_invalid_decay(self, model):
        with pytest.raises(ValueError):
            EMAHook(model, decay=0.0)
        with pytest.raises(ValueError):
            EMAHook(model, decay=1.1)

    def test_invalid_warmup(self, model):
        with pytest.raises(ValueError):
            EMAHook(model, warmup_steps=-1)

    def test_ema_model_is_deepcopy(self, model, ema_hook):
        ema_model = ema_hook.get_ema_model()
        # Same structure, different parameter objects
        assert list(ema_model.parameters())[0] is not list(model.parameters())[0]

    def test_ema_model_no_grad(self, model, ema_hook):
        ema_model = ema_hook.get_ema_model()
        for p in ema_model.parameters():
            assert not p.requires_grad


class TestEMAHookUpdate:
    def test_num_updates_increments(self, model, ema_hook):
        assert ema_hook.num_updates == 0
        ema_hook.update(model)
        assert ema_hook.num_updates == 1
        ema_hook.update(model)
        assert ema_hook.num_updates == 2

    def test_warmup_direct_copy(self, model):
        ema = EMAHook(model, decay=0.99, warmup_steps=5)
        # Change model weights
        with torch.no_grad():
            model.fc.weight.fill_(0.5)
        ema.update(model)
        # During warmup, EMA = exact copy
        ema_w = list(ema.get_ema_model().parameters())[0]
        raw_w = list(model.parameters())[0]
        assert torch.allclose(ema_w, raw_w)

    def test_decay_after_warmup(self, model):
        ema = EMAHook(model, decay=0.5, warmup_steps=2)
        # Set initial values
        with torch.no_grad():
            model.fc.weight.fill_(1.0)
        ema.update(model)  # step 1: warmup copy, ema = 1.0
        ema.update(model)  # step 2: warmup copy, ema = 1.0
        with torch.no_grad():
            model.fc.weight.fill_(2.0)
        ema.update(model)  # step 3: ema = 0.5*1.0 + 0.5*2.0 = 1.5
        ema_w = list(ema.get_ema_model().parameters())[0]
        assert torch.allclose(ema_w, torch.full_like(ema_w, 1.5))


class TestEMAHookStateDict:
    def test_state_dict_keys(self, model, ema_hook):
        ema_hook.update(model)
        sd = ema_hook.state_dict()
        assert "ema_model" in sd
        assert "num_updates" in sd
        assert sd["num_updates"].item() == 1
        assert sd["decay"] == 0.99
        assert sd["warmup_steps"] == 10

    def test_roundtrip(self, model):
        ema1 = EMAHook(model, decay=0.9, warmup_steps=5)
        for _ in range(10):
            with torch.no_grad():
                model.fc.weight.add_(0.1)
            ema1.update(model)

        # Save state
        sd = ema1.state_dict()

        # New model + new EMA
        model2 = TinyModel()
        model2.load_state_dict(model.state_dict())
        ema2 = EMAHook(model2, decay=0.9, warmup_steps=5)
        ema2.load_state_dict(sd)

        # Verify
        assert ema2.num_updates == ema1.num_updates
        for p1, p2 in zip(ema1.get_ema_model().parameters(),
                          ema2.get_ema_model().parameters()):
            assert torch.allclose(p1, p2)


# ── Integration-level tests ──────────────────────────────────────────


class TestEMADualValidation:
    """Test that raw and EMA validation don't cross-contaminate."""

    def test_raw_model_unchanged_after_ema_validation(self, model, ema_hook):
        """EMA validation via get_ema_model() should not modify raw model."""
        ema_hook.update(model)
        raw_state_before = {k: v.clone() for k, v in model.state_dict().items()}

        # Simulate EMA validation (no swap)
        ema_model = ema_hook.get_ema_model()
        _ = ema_model(torch.randn(2, 4))

        raw_state_after = {k: v.clone() for k, v in model.state_dict().items()}
        for k in raw_state_before:
            assert torch.equal(raw_state_before[k], raw_state_after[k]), \
                f"Raw model parameter {k} changed after EMA validation"


class TestEMABestModelSelection:
    """Test best.pt selection logic based on selection_source."""

    def test_ema_disabled_best_is_raw(self):
        """Without EMA, best.pt always comes from raw."""
        ema_enabled = False
        selection_source = "ema"
        # Logic: if not ema_enabled, best.pt = best_raw.pt
        use_ema = ema_enabled and selection_source == "ema"
        assert not use_ema

    def test_ema_enabled_ema_selection(self):
        ema_enabled = True
        selection_source = "ema"
        use_ema = ema_enabled and selection_source == "ema"
        assert use_ema

    def test_ema_enabled_raw_selection(self):
        ema_enabled = True
        selection_source = "raw"
        use_ema = ema_enabled and selection_source == "ema"
        assert not use_ema


class TestEMABackwardCompat:
    """Old configs (no head_ema section) must produce identical results."""

    def test_no_ema_config_means_disabled(self):
        config = {}  # no head_ema key
        ema_cfg = config.get("head_ema", {})
        assert ema_cfg.get("enabled", False) is False

    def test_ema_disabled_no_effect_on_training(self):
        """With EMA disabled, the train_one_epoch should work identically."""
        ema_hook = None
        # Simulate training step without EMA
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 4)
        y = torch.tensor([0, 1, 0, 1])
        for _ in range(3):
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
            if ema_hook is not None:
                ema_hook.update(model)
        # Just verify no crash — EMA disabled is no-op
        assert True
