#!/usr/bin/env python3
"""Round-0 P0 probe: validation-fitted uniform prior bias.

The bias is fitted once with IPF on F05 validation logits.  Only the additive
strength is swept, and only on validation.  The test set is never read by this
script.  The selected frozen vector is written to ``prior_bias.pt`` for a
separate, source-bound test-application step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
for location in (str(ROOT), str(AEGIS_ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

from aegis_clip.balanced_inference import prediction_metrics  # noqa: E402
from aegis_clip.prior_alignment import fit_prior_bias, apply_prior_bias  # noqa: E402
from aegis_clip.runtime import sha256_file  # noqa: E402
from f05_focus_summary import append_summary_row  # noqa: E402


def _parse_strengths(value: str) -> tuple[float, ...]:
    strengths = tuple(
        float(item) for item in value.split(",") if item.strip()
    )
    if not strengths:
        raise ValueError("at least one strength is required")
    if any(not 0.0 <= item <= 1.0 for item in strengths):
        raise ValueError("strengths must lie in [0, 1]")
    if len(set(strengths)) != len(strengths):
        raise ValueError("strengths must be unique")
    return strengths


def _status(delta_macro_pp: float, delta_micro_pp: float) -> str:
    if delta_macro_pp >= 1.00:
        return "breakthrough"
    if delta_macro_pp >= 0.50 and delta_micro_pp >= 0.0:
        return "strong"
    if delta_macro_pp >= 0.30 and delta_micro_pp >= -0.10:
        return "promote"
    if delta_macro_pp > 0.0:
        return "weak"
    return "fail"


def _metrics(
    logits: torch.Tensor,
    payload: dict[str, Any],
    num_classes: int,
) -> dict[str, float | int]:
    return prediction_metrics(
        logits.argmax(dim=1),
        labels=payload["labels"],
        clean_probability=payload["clean_probability"],
        pseudo_labels=payload["pseudo_labels"],
        correction_alpha=payload["correction_alpha"],
        num_classes=num_classes,
        clean_core_threshold=0.70,
    )


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    validation_path = Path(args.validation_logits).expanduser().resolve()
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    payload = torch.load(validation_path, map_location="cpu", weights_only=False)
    required = {
        "logits",
        "labels",
        "clean_probability",
        "pseudo_labels",
        "correction_alpha",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"validation logits are missing keys: {sorted(missing)}"
        )
    logits = torch.as_tensor(payload["logits"]).float()
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("validation logits must be a non-empty [N,C] tensor")
    if not torch.isfinite(logits).all():
        raise ValueError("validation logits contain non-finite values")
    num_classes = int(logits.shape[1])

    bias, fit_report = fit_prior_bias(
        logits,
        max_iterations=int(args.iterations),
        tolerance=float(args.tolerance),
        damping=float(args.damping),
    )
    bare_metrics = _metrics(logits, payload, num_classes)
    strengths = _parse_strengths(args.strengths)
    sweep: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for strength in strengths:
        corrected = apply_prior_bias(logits, bias, strength=float(strength))
        metrics = _metrics(corrected, payload, num_classes)
        delta_macro_pp = 100.0 * (
            float(metrics["raw_macro"]) - float(bare_metrics["raw_macro"])
        )
        delta_micro_pp = 100.0 * (
            float(metrics["raw_micro"]) - float(bare_metrics["raw_micro"])
        )
        candidate = {
            "strength": float(strength),
            "metrics": metrics,
            "delta_macro_pp": delta_macro_pp,
            "delta_micro_pp": delta_micro_pp,
            "changed_predictions": int(
                (logits.argmax(1) != corrected.argmax(1)).sum()
            ),
            "status": _status(delta_macro_pp, delta_micro_pp),
        }
        sweep.append(candidate)
        if best is None or (
            candidate["metrics"]["raw_macro"],
            candidate["metrics"]["raw_micro"],
        ) > (
            best["metrics"]["raw_macro"],
            best["metrics"]["raw_micro"],
        ):
            best = candidate
    assert best is not None

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = {
        "experiment_id": "P0_PRIOR",
        "direction": "prior",
        "validation_logits": str(validation_path),
        "validation_logits_sha256": sha256_file(validation_path),
        "num_classes": num_classes,
        "target_prior": "uniform",
        "fit_report": fit_report,
        "bare": bare_metrics,
        "sweep": sweep,
        "best": best,
        "passed_promotion": bool(
            best["delta_macro_pp"] >= 0.30 and best["delta_micro_pp"] >= -0.10
        ),
        "selection_rule": "best raw_macro then raw_micro on validation only",
        "test_data_used": False,
    }
    probe_path = output_dir / "probe.json"
    probe_path.write_text(
        json.dumps(probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    bias_path = Path(args.bias_output).expanduser().resolve()
    bias_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "method": "validation_fitted_ipf_bias_to_uniform",
            "bias": bias,
            "selected_strength": float(best["strength"]),
            "num_classes": num_classes,
            "fit_report": fit_report,
            "validation_logits": str(validation_path),
            "validation_logits_sha256": sha256_file(validation_path),
            "test_data_used": False,
        },
        bias_path,
    )
    append_summary_row(
        args.summary,
        {
            "experiment_id": "P0",
            "parent_sha": "",
            "seed": "",
            "macro": best["metrics"]["raw_macro"],
            "micro": best["metrics"]["raw_micro"],
            "delta_macro": best["delta_macro_pp"],
            "delta_micro": best["delta_micro_pp"],
            "changed_predictions": best["changed_predictions"],
            "selected_epoch": "",
            "reject_rate": "",
            "platform_score": "",
            "status": best["status"],
        },
    )
    return probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-logits", required=True)
    parser.add_argument(
        "--strengths", default="0.25,0.50,0.75,1.00"
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument(
        "--output-dir", default="outputs/f05_focus/P0_PRIOR"
    )
    parser.add_argument(
        "--bias-output", default="artifacts/f05_focus/prior_bias.pt"
    )
    parser.add_argument(
        "--summary", default="results/f05_focus_summary.csv"
    )
    args = parser.parse_args()
    probe = run_probe(args)
    print(json.dumps(probe["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
