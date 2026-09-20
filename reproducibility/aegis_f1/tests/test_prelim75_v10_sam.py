"""Synthetic CPU checks for the fixed PRELIM75 v10 standard-SAM contract."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from aegis_clip.prelim75_sam import (
    EXPECTED_INFERENCE,
    EXPECTED_SAM,
    RHO,
    V10Schedule,
    _materialize_parameters,
    _validate_plan_schema,
    apply_standard_sam_perturbation,
    build_v10_scheduler,
    global_l2_gradient_norm,
    restore_standard_sam_parameters,
    scheduler_reference,
    standard_sam_update,
)


CONFIG = Path(__file__).resolve().parents[3] / "configs/prelim75_v10_sam.yaml"


def _parameter(value=1.0):
    return torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))


def _set_grad(parameter, value):
    parameter.grad = torch.full_like(parameter, float(value))


def test_global_norm_single_parameter():
    parameter = _parameter()
    _set_grad(parameter, 3.0)
    assert global_l2_gradient_norm([parameter]) == 3.0


def test_global_norm_is_shared_across_parameters():
    first, second = _parameter(), _parameter()
    _set_grad(first, 3.0); _set_grad(second, 4.0)
    assert global_l2_gradient_norm([first, second]) == 5.0


def test_empty_parameter_list_rejected():
    with pytest.raises(ValueError, match="at least one"):
        global_l2_gradient_norm([])


def test_frozen_parameter_rejected():
    parameter = _parameter(); parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="frozen"):
        _materialize_parameters([parameter])


def test_duplicate_parameter_alias_rejected():
    parameter = _parameter()
    with pytest.raises(ValueError, match="aliases"):
        _materialize_parameters([parameter, parameter])


def test_shared_storage_alias_rejected():
    base = torch.ones(2, dtype=torch.float32)
    first = torch.nn.Parameter(base[:1])
    second = torch.nn.Parameter(base[1:])
    with pytest.raises(ValueError, match="share storage"):
        _materialize_parameters([first, second])


def test_non_fp32_parameter_rejected():
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="FP32"):
        _materialize_parameters([parameter])


def test_missing_gradient_rejected():
    with pytest.raises(ValueError, match="missing"):
        global_l2_gradient_norm([_parameter()])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_gradient_rejected(bad):
    parameter = _parameter(); _set_grad(parameter, bad)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        global_l2_gradient_norm([parameter])


def test_zero_gradient_norm_rejected():
    parameter = _parameter(); _set_grad(parameter, 0.0)
    with pytest.raises(ValueError, match="positive"):
        global_l2_gradient_norm([parameter])


def test_standard_perturbation_uses_one_global_radius():
    first, second = _parameter(1.0), _parameter(2.0)
    _set_grad(first, 3.0); _set_grad(second, 4.0)
    state = apply_standard_sam_perturbation([first, second])
    assert math.isclose(first.item(), 1.03, rel_tol=0, abs_tol=2e-7)
    assert math.isclose(second.item(), 2.04, rel_tol=0, abs_tol=2e-7)
    assert abs(state.actual_radius - RHO) <= 5e-6
    restore_standard_sam_parameters(state)


def test_restore_is_bitwise_exact_copy():
    parameter = _parameter(123.456)
    original = parameter.detach().clone()
    _set_grad(parameter, 0.333)
    state = apply_standard_sam_perturbation([parameter])
    restore_standard_sam_parameters(state)
    assert torch.equal(parameter.detach(), original)


def test_double_restore_rejected():
    parameter = _parameter(); _set_grad(parameter, 1.0)
    state = apply_standard_sam_perturbation([parameter])
    restore_standard_sam_parameters(state)
    with pytest.raises(RuntimeError, match="already restored"):
        restore_standard_sam_parameters(state)


@pytest.mark.parametrize("rho", [0.0, -0.1, float("nan")])
def test_invalid_rho_rejected(rho):
    parameter = _parameter(); _set_grad(parameter, 1.0)
    with pytest.raises(ValueError, match="rho"):
        apply_standard_sam_perturbation([parameter], rho=rho)


@pytest.mark.parametrize("epsilon", [0.0, -1e-12, float("nan")])
def test_invalid_epsilon_rejected(epsilon):
    parameter = _parameter(); _set_grad(parameter, 1.0)
    with pytest.raises(ValueError, match="epsilon"):
        apply_standard_sam_perturbation([parameter], epsilon=epsilon)


def test_radius_failure_restores_original_value():
    parameter = _parameter(1.0); original = parameter.detach().clone(); _set_grad(parameter, 1.0)
    with pytest.raises(RuntimeError, match="radius"):
        apply_standard_sam_perturbation([parameter], radius_tolerance=1e-12)
    assert torch.equal(parameter.detach(), original)


def test_second_pass_exception_restores_original_value():
    parameter = _parameter(1.0)
    original = parameter.detach().clone()
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)

    def backward(pass_id):
        if pass_id == 2:
            raise KeyboardInterrupt("synthetic interruption")
        loss = (parameter.square()).sum()
        loss.backward()
        return loss.detach()

    with pytest.raises(KeyboardInterrupt):
        standard_sam_update([parameter], optimizer, backward)
    assert torch.equal(parameter.detach(), original)


def test_g1_and_g2_are_separate_gradients():
    parameter = _parameter(1.0)
    optimizer = torch.optim.AdamW([parameter], lr=0.0)

    def backward(_pass_id):
        loss = 0.5 * (parameter - 3.0).square().sum()
        loss.backward()
        return loss.detach()

    result = standard_sam_update([parameter], optimizer, backward)
    assert result["first_gradient_norm"] == pytest.approx(2.0)
    assert result["second_gradient_norm"] == pytest.approx(2.05, abs=2e-6)
    assert result["loss_increase"] > 0


def test_adamw_step_is_applied_at_original_parameters():
    parameter = _parameter(1.0)
    expected = _parameter(1.0)
    optimizer = torch.optim.AdamW([parameter], lr=0.01, betas=(0.9, 0.999), weight_decay=0.1)
    reference = torch.optim.AdamW([expected], lr=0.01, betas=(0.9, 0.999), weight_decay=0.1)

    def backward(_pass_id):
        loss = 0.5 * (parameter - 3.0).square().sum()
        loss.backward()
        return loss.detach()

    # g2 at theta+e is -2.05; use that gradient on the original reference theta.
    expected.grad = torch.tensor([-2.05])
    reference.step()
    standard_sam_update([parameter], optimizer, backward)
    assert parameter.detach().item() == pytest.approx(expected.detach().item(), abs=2e-7)


def test_first_and_second_gradients_are_not_averaged():
    parameter = _parameter(1.0)
    optimizer = torch.optim.AdamW([parameter], lr=0.01, betas=(0.0, 0.0), eps=1e-8, weight_decay=0.0)

    def backward(_pass_id):
        loss = 0.5 * (parameter - 3.0).square().sum()
        loss.backward()
        return loss.detach()

    result = standard_sam_update([parameter], optimizer, backward)
    assert result["second_gradient_norm"] != pytest.approx(
        (result["first_gradient_norm"] + result["second_gradient_norm"]) / 2
    )


def test_gradient_clipping_applies_to_g2_only():
    parameter = _parameter(1.0)
    optimizer = torch.optim.AdamW([parameter], lr=0.0)

    def backward(_pass_id):
        loss = 100.0 * parameter.square().sum()
        loss.backward()
        return loss.detach()

    result = standard_sam_update([parameter], optimizer, backward, clip_norm=1.0)
    assert result["preclip_second_gradient_norm"] > 1.0
    assert result["gradient_clipped"] is True


def test_one_sam_call_creates_one_adamw_state_entry():
    parameter = _parameter(1.0)
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)

    def backward(_pass_id):
        loss = parameter.square().sum(); loss.backward(); return loss.detach()

    standard_sam_update([parameter], optimizer, backward)
    assert len(optimizer.state) == 1
    assert int(optimizer.state[parameter]["step"]) == 1


def test_chunked_full_denominator_gradient_matches_full_batch():
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[0.0], [1.0], [1.5], [2.0]])
    weights = torch.tensor([1.0, 0.5, 0.0, 2.0])
    full = _parameter(0.25)
    chunked = _parameter(0.25)
    denominator = weights.sum()
    (((x * full - y).square().flatten() * weights).sum() / denominator).backward()
    for start in (0, 2):
        loss = ((x[start:start+2] * chunked - y[start:start+2]).square().flatten()
                * weights[start:start+2]).sum() / denominator
        loss.backward()
    torch.testing.assert_close(full.grad, chunked.grad, atol=0, rtol=0)


def test_scheduler_registered_boundaries():
    reference = scheduler_reference()
    assert reference["total_updates"] == 9678
    assert reference["boundaries"]["1M"]["multiplier"] == pytest.approx(0.7525)
    assert reference["boundaries"]["2M"]["multiplier"] == pytest.approx(0.2575)
    assert reference["boundaries"]["3M"]["multiplier"] == pytest.approx(0.01)


def test_scheduler_optimizer_then_scheduler_alignment():
    parameter = _parameter()
    optimizer = torch.optim.AdamW([{"name": "backbone", "params": [parameter], "lr": 1e-6}])
    spec = V10Schedule()
    scheduler = build_v10_scheduler(optimizer, spec)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step(); scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6 * spec.multiplier(1))


def test_scheduler_rejects_position_beyond_horizon():
    with pytest.raises(ValueError, match="outside"):
        V10Schedule().multiplier(9679)


def test_config_keeps_registered_standard_sam_contract():
    plan = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert plan["sam"] == EXPECTED_SAM
    assert plan["protocol"]["sample_epochs"] == [19, 20, 21]
    assert plan["protocol"]["total_updates_per_candidate"] == 9678


def test_schema_rejects_rho_change():
    plan = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(plan); changed["sam"]["rho"] = 0.1
    with pytest.raises(ValueError, match="sam"):
        _validate_plan_schema(changed)


def test_inference_numerics_preserve_v9_replay_boundary():
    assert EXPECTED_INFERENCE == {
        "batch_size": 64,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "float32_matmul_precision": "highest",
    }
