"""Tests for common.losses module."""

import torch
import torch.nn as nn
import pytest

from common.losses import build_loss, LabelSmoothingLoss, GCELoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logits_target():
    """Return (logits, targets) with shape (N=8, n_classes=5)."""
    rng = torch.Generator().manual_seed(42)
    logits = torch.randn(8, 5, generator=rng)
    targets = torch.randint(0, 5, (8,), generator=rng)
    return logits, targets


# ---------------------------------------------------------------------------
# Cross Entropy
# ---------------------------------------------------------------------------


class TestCrossEntropy:
    def test_ce_identical_to_pytorch(self, logits_target):
        logits, targets = logits_target
        cfg = {"loss": {"name": "cross_entropy"}}
        custom_loss = build_loss(cfg)
        reference = nn.CrossEntropyLoss()

        l1 = custom_loss(logits, targets)
        l2 = reference(logits, targets)

        assert torch.isclose(l1, l2, atol=1e-8).item(), (
            f"custom CE {l1.item()} != torch CE {l2.item()}"
        )

    def test_ce_reduction_none(self, logits_target):
        logits, targets = logits_target
        cfg = {"loss": {"name": "cross_entropy", "reduction": "none"}}
        loss_fn = build_loss(cfg)
        loss = loss_fn(logits, targets)

        assert loss.shape == (logits.size(0),), f"Expected ({logits.size(0)},), got {loss.shape}"

    def test_ce_reduction_sum(self, logits_target):
        logits, targets = logits_target
        cfg_none = {"loss": {"name": "cross_entropy", "reduction": "none"}}
        cfg_sum = {"loss": {"name": "cross_entropy", "reduction": "sum"}}
        loss_none = build_loss(cfg_none)(logits, targets)
        loss_sum = build_loss(cfg_sum)(logits, targets)

        assert torch.isclose(loss_sum, loss_none.sum(), atol=1e-8).item()


# ---------------------------------------------------------------------------
# Label Smoothing
# ---------------------------------------------------------------------------


class TestLabelSmoothing:
    def test_label_smoothing_epsilon_zero_equals_ce(self, logits_target):
        logits, targets = logits_target
        cfg_ls = {"loss": {"name": "label_smoothing", "epsilon": 0.0}}
        cfg_ce = {"loss": {"name": "cross_entropy"}}
        ls_loss = build_loss(cfg_ls)(logits, targets)
        ce_loss = build_loss(cfg_ce)(logits, targets)

        assert torch.isclose(ls_loss, ce_loss, atol=1e-8).item(), (
            f"label smoothing (eps=0) {ls_loss.item()} != CE {ce_loss.item()}"
        )

    def test_label_smoothing_epsilon_half_uniform(self):
        """At epsilon=0.5 with 2 classes, verify target distribution and loss manually.

        Formula: q_y = 1 - eps, q_other = eps / (C - 1).
        With eps=0.5, C=2: q_y = 0.5, q_other = 0.5 / 1 = 0.5 (uniform).
        """
        logits = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])
        loss_fn = LabelSmoothingLoss(epsilon=0.5, reduction="none")
        loss = loss_fn(logits, targets)

        # Manually compute per-sample.
        eps = 0.5
        # q_y = 1 - eps = 0.5, q_other = eps / (C-1) = 0.5
        soft = torch.softmax(logits, dim=1)

        # Sample 0: logits [1.0, 0.0] → softmax [0.7311, 0.2689]
        #   expected loss = -(0.5 * ln(0.7311) + 0.5 * ln(0.2689))
        expected_sample0 = -(
            0.5 * torch.log(soft[0, 0]) + 0.5 * torch.log(soft[0, 1])
        )
        # Sample 1: logits [0.0, 2.0] → softmax [0.1192, 0.8808]
        expected_sample1 = -(
            0.5 * torch.log(soft[1, 0]) + 0.5 * torch.log(soft[1, 1])
        )

        assert loss.shape == (2,)
        assert torch.isclose(loss[0], expected_sample0, atol=1e-8).item()
        assert torch.isclose(loss[1], expected_sample1, atol=1e-8).item()

    def test_label_smoothing_reduction_none(self, logits_target):
        logits, targets = logits_target
        cfg = {"loss": {"name": "label_smoothing", "reduction": "none"}}
        loss_fn = build_loss(cfg)
        loss = loss_fn(logits, targets)

        assert loss.shape == (logits.size(0),), f"Expected ({logits.size(0)},), got {loss.shape}"

    def test_label_smoothing_epsilon_range(self):
        LabelSmoothingLoss(epsilon=0.0)
        LabelSmoothingLoss(epsilon=0.5)
        LabelSmoothingLoss(epsilon=1.0)

        with pytest.raises(ValueError, match="epsilon must be in"):
            LabelSmoothingLoss(epsilon=-0.1)
        with pytest.raises(ValueError, match="epsilon must be in"):
            LabelSmoothingLoss(epsilon=1.1)

    def test_label_smoothing_illegal_reduction(self):
        with pytest.raises(ValueError, match="reduction must be"):
            LabelSmoothingLoss(epsilon=0.1, reduction="sum_mean")

    def test_label_smoothing_bad_logits_shape(self):
        loss_fn = LabelSmoothingLoss(epsilon=0.1)
        # 3-D logits not allowed
        with pytest.raises(ValueError, match="logits must be 2-D"):
            loss_fn(torch.randn(4, 5, 5), torch.randint(0, 5, (4,)))

    def test_label_smoothing_bad_targets_shape(self):
        loss_fn = LabelSmoothingLoss(epsilon=0.1)
        with pytest.raises(ValueError, match="targets must be 1-D"):
            loss_fn(torch.randn(4, 5), torch.randint(0, 5, (4, 1)))

    def test_label_smoothing_batch_size_mismatch(self):
        loss_fn = LabelSmoothingLoss(epsilon=0.1)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            loss_fn(torch.randn(4, 5), torch.randint(0, 5, (3,)))

    def test_label_smoothing_target_out_of_range(self):
        loss_fn = LabelSmoothingLoss(epsilon=0.1)
        with pytest.raises(ValueError, match="targets must be in"):
            loss_fn(torch.randn(4, 5), torch.tensor([0, 1, 2, 5]))

    def test_label_smoothing_single_class(self):
        # 1 class is invalid for label smoothing
        with pytest.raises(ValueError, match="at least 2 classes"):
            loss_fn = LabelSmoothingLoss(epsilon=0.1)
            loss_fn(torch.randn(4, 1), torch.zeros(4, dtype=torch.long))

    def test_label_smoothing_sum_reduction(self, logits_target):
        """Verify sum reduction = manual sum of per-sample losses."""
        logits, targets = logits_target
        cfg_none = {"loss": {"name": "label_smoothing", "reduction": "none"}}
        cfg_sum = {"loss": {"name": "label_smoothing", "reduction": "sum"}}
        loss_none = build_loss(cfg_none)(logits, targets)
        loss_sum = build_loss(cfg_sum)(logits, targets)

        assert torch.isclose(loss_sum, loss_none.sum(), atol=1e-8).item()


# ---------------------------------------------------------------------------
# Generalized Cross Entropy
# ---------------------------------------------------------------------------


class TestGCE:
    def test_gce_q_equals_1_is_mae(self):
        """When q=1.0, GCE = (1 - py) / 1 = 1 - py."""
        logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, -1.0]])
        targets = torch.tensor([0, 2])
        loss_fn = GCELoss(q=1.0, reduction="none")
        loss = loss_fn(logits, targets)

        # Manual computation.
        probs = torch.softmax(logits, dim=1)
        py = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        expected = 1.0 - py

        assert torch.allclose(loss, expected, atol=1e-8), (
            f"GCE (q=1) {loss} != 1-py {expected}"
        )

    def test_gce_clamps_low_probability(self):
        """py very small is clamped to probability_epsilon; no inf/nan."""
        logits = torch.tensor([[-100.0, 100.0]])
        targets = torch.tensor([0])
        loss_fn = GCELoss(q=0.7, probability_epsilon=1e-7, reduction="none")
        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss).any(), "Loss contains NaN"
        assert not torch.isinf(loss).any(), "Loss contains Inf"
        # py should be clamped to 1e-7: loss = (1 - (1e-7)^0.7) / 0.7
        assert loss.item() < 1.0 / 0.7, "Loss not bounded as expected"

    def test_gce_q_close_to_zero(self):
        """Small q (0.1) works without nan."""
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss_fn = GCELoss(q=0.1, reduction="none")
        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss).any(), "Loss contains NaN"
        assert not torch.isinf(loss).any(), "Loss contains Inf"
        assert loss.shape == (4,)


# ---------------------------------------------------------------------------
# build_loss — edge cases
# ---------------------------------------------------------------------------


class TestBuildLoss:
    def test_build_loss_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown loss name"):
            build_loss({"loss": {"name": "nonexistent"}})

    def test_build_loss_extra_params_raises(self):
        with pytest.raises(ValueError, match="CE loss takes no extra params"):
            build_loss({"loss": {"name": "cross_entropy", "foo": "bar"}})

    def test_build_loss_defaults_to_ce_when_no_loss_key(self):
        """Empty config → default CE."""
        loss_fn = build_loss({})
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_build_loss_defaults_to_ce_when_empty_loss_dict(self):
        """loss: {} → default CE."""
        loss_fn = build_loss({"loss": {}})
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_build_loss_default_ce_identical_to_pytorch(self, logits_target):
        """Default (no config) CE matches torch CE."""
        logits, targets = logits_target
        l1 = build_loss({})(logits, targets)
        l2 = nn.CrossEntropyLoss()(logits, targets)
        assert torch.isclose(l1, l2, atol=1e-8).item()

    def test_build_loss_label_smoothing_extra_params_raises(self):
        with pytest.raises(ValueError, match="Unknown label_smoothing params"):
            build_loss(
                {"loss": {"name": "label_smoothing", "epsilon": 0.1, "bad": True}}
            )


# ---------------------------------------------------------------------------
# B0: Gradient regression — same logits → same gradient
# ---------------------------------------------------------------------------


class TestCEGradientRegression:
    """B0 requirement: custom CE gradient must match torch CE (1e-8 tol)."""

    def test_ce_gradient_identical_to_pytorch(self, logits_target):
        logits, targets = logits_target

        # Reference: torch CE
        ref_logits = logits.clone().requires_grad_(True)
        ref_loss = nn.CrossEntropyLoss()(ref_logits, targets)
        ref_loss.backward()

        # Custom: build_loss CE
        custom_logits = logits.clone().requires_grad_(True)
        custom_loss = build_loss({"loss": {"name": "cross_entropy"}})(custom_logits, targets)
        custom_loss.backward()

        assert torch.allclose(
            custom_logits.grad, ref_logits.grad, atol=1e-8
        ), (
            f"Gradient mismatch: max diff = "
            f"{(custom_logits.grad - ref_logits.grad).abs().max().item():.2e}"
        )

    def test_label_smoothing_eps0_gradient_equals_ce(self, logits_target):
        """LS with epsilon=0 must produce identical gradients to CE."""
        logits, targets = logits_target

        ref_logits = logits.clone().requires_grad_(True)
        nn.CrossEntropyLoss()(ref_logits, targets).backward()

        ls_logits = logits.clone().requires_grad_(True)
        LabelSmoothingLoss(epsilon=0.0)(ls_logits, targets).backward()

        assert torch.allclose(
            ls_logits.grad, ref_logits.grad, atol=1e-8
        ), (
            f"LS(eps=0) gradient mismatch: max diff = "
            f"{(ls_logits.grad - ref_logits.grad).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# B0: 1-epoch smoke test — end-to-end training step with custom loss
# ---------------------------------------------------------------------------


class TestOneEpochSmoke:
    """Verify that build_loss CE trains identically to raw nn.CrossEntropyLoss
    over a full optimizer step (forward + backward + update)."""

    def test_one_step_loss_and_gradient_match(self):
        """One optimizer step with identical init → same weights, same loss."""
        torch.manual_seed(42)
        n_classes = 10

        # Two identical linear models
        model_ref = nn.Linear(16, n_classes)
        model_custom = nn.Linear(16, n_classes)
        model_custom.load_state_dict(model_ref.state_dict())

        # Identical optimizers
        opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)
        opt_custom = torch.optim.SGD(model_custom.parameters(), lr=0.01)

        # Random batch
        x = torch.randn(32, 16)
        targets = torch.randint(0, n_classes, (32,))

        # Reference step
        opt_ref.zero_grad()
        loss_ref = nn.CrossEntropyLoss()(model_ref(x), targets)
        loss_ref.backward()
        opt_ref.step()

        # Custom CE step
        opt_custom.zero_grad()
        loss_custom = build_loss({"loss": {"name": "cross_entropy"}})(model_custom(x), targets)
        loss_custom.backward()
        opt_custom.step()

        # Losses must match
        assert torch.allclose(loss_ref, loss_custom, atol=1e-8), (
            f"Loss mismatch: {loss_ref.item():.10f} vs {loss_custom.item():.10f}"
        )

        # Weights must match after one step
        for (n_ref, p_ref), (n_cust, p_cust) in zip(
            model_ref.named_parameters(), model_custom.named_parameters()
        ):
            assert torch.allclose(p_ref, p_cust, atol=1e-8), (
                f"Weight mismatch for {n_ref}: "
                f"max diff = {(p_ref - p_cust).abs().max().item():.2e}"
            )
