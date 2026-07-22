import pytest

from aegis_clip.n3_gate import evaluate_n3_gate


def _metrics(
    clean: float,
    trusted: float,
    raw: float,
    drift: float,
    flip: float,
) -> dict[str, float | int]:
    return {
        "clean_core_micro": clean,
        "clean_core_samples": 100,
        "trusted_macro": trusted,
        "raw_micro": raw,
        "mean_feature_drift": drift,
        "flip_prediction_agreement": flip,
    }


def _complementary(clean_delta: float = 0.3) -> dict:
    return {
        "delta_pp": {
            "clean_core_micro": clean_delta,
            "trusted_macro": 0.2,
            "raw_micro": 0.1,
        },
        "fusion_transition_clean_core": {"net_correct": 2},
        "global_path_max_abs_logit_difference": 0.0,
        "global_path_prediction_agreement": 1.0,
    }


def test_n3_gate_accepts_all_boundaries() -> None:
    report = evaluate_n3_gate(
        _metrics(0.80, 0.75, 0.60, 0.0, 0.90),
        _metrics(0.805, 0.75, 0.598, 0.0075, 0.898),
        _metrics(0.8025, 0.74, 0.59, 0.0, 0.89),
        _complementary(0.25),
    )
    assert report["overall_passed"] is True
    assert report["selected_inference"] == "complementary_flip_local_global"


def test_n3_gate_closes_when_control_margin_fails() -> None:
    report = evaluate_n3_gate(
        _metrics(0.80, 0.75, 0.60, 0.0, 0.90),
        _metrics(0.805, 0.76, 0.61, 0.001, 0.91),
        _metrics(0.803, 0.74, 0.59, 0.0, 0.89),
        _complementary(),
    )
    assert report["training_gate"]["passed"] is False
    assert report["overall_passed"] is False


def test_n3_gate_rejects_misaligned_cohorts() -> None:
    initial = _metrics(0.80, 0.75, 0.60, 0.0, 0.90)
    candidate = _metrics(0.81, 0.76, 0.61, 0.001, 0.91)
    candidate["clean_core_samples"] = 99
    with pytest.raises(ValueError, match="cohorts differ"):
        evaluate_n3_gate(
            initial,
            candidate,
            _metrics(0.80, 0.74, 0.59, 0.0, 0.89),
            _complementary(),
        )
