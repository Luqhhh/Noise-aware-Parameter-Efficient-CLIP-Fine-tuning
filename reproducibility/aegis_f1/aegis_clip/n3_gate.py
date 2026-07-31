"""Deterministic acceptance gate for the preregistered N3 experiment."""

from __future__ import annotations

from typing import Any


def _minimum(name: str, observed: float, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "rule": ">=",
        "observed": observed,
        "threshold": threshold,
        "passed": observed + 1.0e-9 >= threshold,
    }


def _maximum(name: str, observed: float, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "rule": "<=",
        "observed": observed,
        "threshold": threshold,
        "passed": observed <= threshold + 1.0e-9,
    }


def evaluate_n3_gate(
    initial: dict[str, Any],
    candidate: dict[str, Any],
    j0_control: dict[str, Any],
    complementary: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen training thresholds and the fixed M3 inference gate."""
    if int(initial["clean_core_samples"]) != int(candidate["clean_core_samples"]):
        raise ValueError("initial and candidate clean-core cohorts differ")
    if int(j0_control["clean_core_samples"]) != int(candidate["clean_core_samples"]):
        raise ValueError("J0 and N3 clean-core cohorts differ")
    training_checks = [
        _minimum(
            "clean_core_gain_vs_A2_pp",
            100.0
            * (float(candidate["clean_core_micro"]) - float(initial["clean_core_micro"])),
            0.50,
        ),
        _minimum(
            "clean_core_gain_vs_J0_pp",
            100.0
            * (
                float(candidate["clean_core_micro"])
                - float(j0_control["clean_core_micro"])
            ),
            0.25,
        ),
        _minimum(
            "trusted_macro_gain_vs_A2_pp",
            100.0
            * (float(candidate["trusted_macro"]) - float(initial["trusted_macro"])),
            0.0,
        ),
        _minimum(
            "raw_micro_gain_vs_A2_pp",
            100.0 * (float(candidate["raw_micro"]) - float(initial["raw_micro"])),
            -0.20,
        ),
        _maximum(
            "feature_drift_percent",
            100.0 * float(candidate["mean_feature_drift"]),
            0.75,
        ),
        _minimum(
            "flip_agreement_gain_vs_A2_pp",
            100.0
            * (
                float(candidate["flip_prediction_agreement"])
                - float(initial["flip_prediction_agreement"])
            ),
            -0.20,
        ),
    ]
    delta = complementary["delta_pp"]
    transition = complementary["fusion_transition_clean_core"]
    inference_checks = [
        _minimum(
            "M3_clean_core_gain_vs_N3_center_pp",
            float(delta["clean_core_micro"]),
            0.25,
        ),
        _minimum(
            "M3_trusted_macro_gain_vs_N3_center_pp",
            float(delta["trusted_macro"]),
            0.0,
        ),
        _minimum(
            "M3_raw_micro_gain_vs_N3_center_pp",
            float(delta["raw_micro"]),
            0.0,
        ),
        _minimum(
            "M3_clean_core_net_corrections",
            float(transition["net_correct"]),
            1.0,
        ),
        _maximum(
            "M3_global_path_max_abs_logit_difference",
            float(complementary["global_path_max_abs_logit_difference"]),
            0.0,
        ),
        _minimum(
            "M3_global_path_prediction_agreement",
            float(complementary["global_path_prediction_agreement"]),
            1.0,
        ),
    ]
    training_passed = all(check["passed"] for check in training_checks)
    inference_passed = all(check["passed"] for check in inference_checks)
    return {
        "experiment": "N3_A2_ADAPTFORMER_GATE",
        "training_gate": {
            "passed": training_passed,
            "checks": training_checks,
        },
        "fixed_M3_inference_gate": {
            "passed": inference_passed,
            "checks": inference_checks,
        },
        "overall_passed": training_passed and inference_passed,
        "selected_inference": (
            "complementary_flip_local_global"
            if training_passed and inference_passed
            else None
        ),
        "test_data_used_for_selection": False,
        "parameter_scan": False,
    }
