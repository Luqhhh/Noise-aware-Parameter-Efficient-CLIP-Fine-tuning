import torch

from aegis_clip.balanced_transport import (
    balanced_transport_prediction,
    converged_balanced_transport_prediction,
    hard_balance_diagnostics,
    paired_change_summary,
    uniform_target_counts,
    v1_gate_decision,
    v2_gate_decision,
)


def test_uniform_target_counts_support_fractional_balanced_marginal() -> None:
    counts = uniform_target_counts(5, 2)
    assert torch.equal(counts, torch.tensor([2.5, 2.5]))
    assert float(counts.sum()) == 5.0


def test_balanced_transport_reduces_a_shared_prediction_bias() -> None:
    logits = torch.tensor(
        [
            [8.0, 0.0],
            [7.0, 0.0],
            [6.0, 0.0],
            [5.0, 0.0],
            [4.0, 3.0],
            [0.0, 8.0],
        ]
    )
    baseline = logits.argmax(dim=1)
    prediction, sinkhorn = balanced_transport_prediction(logits)
    bare = hard_balance_diagnostics(baseline, num_classes=2)
    candidate = hard_balance_diagnostics(prediction, num_classes=2)
    assert candidate["prediction_count_cv"] < bare["prediction_count_cv"]
    assert sinkhorn["maximum_row_absolute_error"] < 1.0e-4
    assert sinkhorn["maximum_column_absolute_error"] < 1.0e-4


def test_converged_balanced_transport_records_fixed_stopping_rule() -> None:
    logits = torch.tensor(
        [[8.0, 0.0], [7.0, 0.0], [6.0, 0.0], [0.0, 8.0]]
    )
    prediction, sinkhorn = converged_balanced_transport_prediction(
        logits,
        minimum_iterations=10,
        maximum_iterations=500,
        check_interval=5,
    )
    assert prediction.shape == (4,)
    assert sinkhorn["converged"] is True
    assert sinkhorn["maximum_row_absolute_error"] <= 1.0e-5
    assert sinkhorn["maximum_column_absolute_error"] <= 1.0e-5


def test_paired_change_summary_reports_net_repairs() -> None:
    summary = paired_change_summary(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 1, 1, 1]),
    )
    assert summary["wrong_to_correct"] == 1
    assert summary["correct_to_wrong"] == 1
    assert summary["net_correct"] == 0


def _report(
    *,
    clean_delta: float,
    trusted_delta: float = 0.0,
    raw_delta: float = 0.0,
    cv_reduction: float = 0.5,
) -> dict[str, object]:
    return {
        "delta_pp": {
            "clean_core_micro": clean_delta,
            "trusted_macro": trusted_delta,
            "raw_micro": raw_delta,
        },
        "prediction_count_cv_relative_reduction": cv_reduction,
        "transport": {"prediction_empty_classes": 0},
        "sinkhorn": {
            "maximum_row_absolute_error": 1.0e-6,
            "maximum_column_absolute_error": 1.0e-5,
        },
    }


def test_v1_gate_requires_cross_checkpoint_accuracy_gain() -> None:
    passed = v1_gate_decision(
        {
            "f1_m1": _report(clean_delta=0.25),
            "a2_m1": _report(clean_delta=0.15),
        }
    )
    assert passed["passed"] is True

    failed = v1_gate_decision(
        {
            "f1_m1": _report(clean_delta=0.19),
            "a2_m1": _report(clean_delta=0.15),
        }
    )
    assert failed["passed"] is False
    assert "f1_m1_clean_core_micro_delta_pp" in failed["failed_checks"]


def test_v1_gate_rejects_accuracy_for_superficial_balance() -> None:
    result = v1_gate_decision(
        {
            "f1_m1": _report(clean_delta=0.25, raw_delta=-0.11),
            "a2_m1": _report(clean_delta=0.15),
        }
    )
    assert result["passed"] is False
    assert "f1_m1_raw_micro_delta_pp" in result["failed_checks"]


def test_v2_gate_requires_solver_convergence() -> None:
    f1 = _report(clean_delta=0.25)
    a2 = _report(clean_delta=0.15)
    f1["sinkhorn"]["converged"] = True
    a2["sinkhorn"]["converged"] = False
    result = v2_gate_decision({"f1_m1": f1, "a2_m1": a2})
    assert result["passed"] is False
    assert "a2_m1_sinkhorn_converged" in result["failed_checks"]

