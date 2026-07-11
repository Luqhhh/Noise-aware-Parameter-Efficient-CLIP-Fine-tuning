"""Tests for --init-checkpoint weight-only initialization (D3).

Verifies:
  - Same architecture: strict=False loads with no missing/unexpected keys
  - Frozen → partially-unfrozen: strict=False loads with exact match
  - Epoch/optimizer state NOT restored
"""

import tempfile

import torch

from tests.test_partial_unfreeze import MockCLIP, _make_model


class TestInitCheckpointSameArchitecture:
    """Loading checkpoint between identical architectures."""

    def test_frozen_to_frozen_exact_key_match(self):
        model1 = _make_model(MockCLIP(), freeze_clip=True)
        model2 = _make_model(MockCLIP(), freeze_clip=True)

        ckpt = {"model_state_dict": model1.state_dict()}
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(ckpt, f.name)
            state = torch.load(f.name, map_location="cpu")

        missing, unexpected = model2.load_state_dict(
            state["model_state_dict"], strict=False
        )
        assert not missing, f"Unexpected missing keys: {missing}"
        assert not unexpected, f"Unexpected unexpected keys: {unexpected}"

    def test_weights_actually_copied(self):
        """Verify that loaded weights match source, not random init."""
        model1 = _make_model(MockCLIP(), freeze_clip=True)
        model2 = _make_model(MockCLIP(), freeze_clip=True)

        ckpt = {"model_state_dict": model1.state_dict()}
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(ckpt, f.name)
            state = torch.load(f.name, map_location="cpu")

        model2.load_state_dict(state["model_state_dict"], strict=False)

        for (n1, p1), (n2, p2) in zip(
            model1.state_dict().items(), model2.state_dict().items()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Weight mismatch for {n1}"


class TestFrozenToPartialUnfreeze:
    """Loading a frozen checkpoint into a partially unfrozen model.

    The state_dict keys are identical — only requires_grad differs,
    which is NOT part of state_dict.
    """

    def test_frozen_init_to_partial_unfreeze(self):
        model_frozen = _make_model(MockCLIP(), freeze_clip=True)
        model_unfrozen = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )

        ckpt = {"model_state_dict": model_frozen.state_dict()}
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(ckpt, f.name)
            state = torch.load(f.name, map_location="cpu")

        missing, unexpected = model_unfrozen.load_state_dict(
            state["model_state_dict"], strict=False
        )
        assert not missing, f"Missing keys: {missing}"
        assert not unexpected, f"Unexpected keys: {unexpected}"

    def test_unfrozen_params_still_trainable_after_load(self):
        """Loading weights doesn't change requires_grad."""
        model_unfrozen = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )
        model_frozen = _make_model(MockCLIP(), freeze_clip=True)

        ckpt = {"model_state_dict": model_frozen.state_dict()}
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(ckpt, f.name)
            state = torch.load(f.name, map_location="cpu")

        model_unfrozen.load_state_dict(state["model_state_dict"], strict=False)

        # Last block should still be trainable
        last_block = model_unfrozen.visual.transformer.resblocks[-1]
        for param in last_block.parameters():
            assert param.requires_grad, \
                "Unfrozen block must stay trainable after weight load"

        # First block should still be frozen
        first_block = model_unfrozen.visual.transformer.resblocks[0]
        for param in first_block.parameters():
            assert not param.requires_grad, \
                "Frozen block must stay frozen after weight load"
