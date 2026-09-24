#!/usr/bin/env python3
"""Apply a validation-fitted frozen prior bias to already-cached test logits.

This script is deliberately post-hoc and deterministic:

1. read test logits and names;
2. read the frozen bias selected on validation;
3. add ``strength * bias`` once;
4. write corrected logits and an optional submission CSV.

It performs no test-batch fitting, no soft-marginal matching, and no label or
model update.
"""

from __future__ import annotations

import argparse
import csv
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

from aegis_clip.prior_alignment import apply_prior_bias  # noqa: E402
from aegis_clip.runtime import atomic_json_dump, sha256_file  # noqa: E402


def _payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("logit payload must be a mapping")
    if "logits" not in payload:
        raise ValueError("logit payload must contain logits")
    names = payload.get("names")
    if names is None:
        names = payload.get("paths")
    if names is None:
        raise ValueError("logit payload must contain names or paths")
    return payload


def _load_bias(path: Path) -> tuple[torch.Tensor, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("prior bias payload must be a mapping")
    if "bias" not in payload:
        raise ValueError("prior bias payload must contain bias")
    bias = torch.as_tensor(payload["bias"], dtype=torch.float32).flatten()
    strength = float(payload.get("selected_strength", 1.0))
    if not torch.isfinite(bias).all():
        raise ValueError("prior bias contains non-finite values")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("prior bias strength must be in [0, 1]")
    return bias, strength


def _idx_to_class(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("class mapping must be a JSON mapping")
    result: dict[int, str] = {}
    for key, value in payload.items():
        result[int(key)] = str(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-logits", required=True)
    parser.add_argument("--bias", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-mapping")
    parser.add_argument(
        "--strength",
        type=float,
        help="override the validation-selected strength; not selected on test",
    )
    args = parser.parse_args()

    test_path = Path(args.test_logits).expanduser().resolve()
    bias_path = Path(args.bias).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _payload(test_path)
    bias, stored_strength = _load_bias(bias_path)
    strength = (
        float(args.strength)
        if args.strength is not None
        else float(stored_strength)
    )
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    logits = torch.as_tensor(payload["logits"]).float()
    if logits.ndim != 2 or logits.shape[1] != bias.numel():
        raise ValueError("test logits and bias class dimensions do not align")
    names = [str(name) for name in payload.get("names", payload.get("paths"))]
    if len(names) != logits.shape[0]:
        raise ValueError("test logits and names have different lengths")
    corrected = apply_prior_bias(logits, bias, strength=strength)
    corrected_path = output_dir / "corrected_test_logits.pt"
    torch.save(
        {
            "format_version": 1,
            "source_logits": str(test_path),
            "source_logits_sha256": sha256_file(test_path),
            "bias": str(bias_path),
            "bias_sha256": sha256_file(bias_path),
            "strength": float(strength),
            "test_data_used": False,
            "names": names,
            "logits": corrected,
        },
        corrected_path,
    )

    changed_predictions = int(
        (logits.argmax(dim=1) != corrected.argmax(dim=1)).sum()
    )
    report: dict[str, Any] = {
        "method": "frozen_validation_fitted_prior_bias",
        "source_logits": str(test_path),
        "source_logits_sha256": sha256_file(test_path),
        "bias": str(bias_path),
        "bias_sha256": sha256_file(bias_path),
        "strength": float(strength),
        "strength_source": "validation_only",
        "test_data_used_for_fit_or_selection": False,
        "changed_predictions": changed_predictions,
        "corrected_logits": str(corrected_path),
        "corrected_logits_sha256": sha256_file(corrected_path),
    }
    if args.class_mapping:
        mapping_path = Path(args.class_mapping).expanduser().resolve()
        mapping = _idx_to_class(mapping_path)
        csv_path = output_dir / "pred_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for name, index in zip(names, corrected.argmax(dim=1).tolist()):
                if int(index) not in mapping:
                    raise ValueError(f"predicted index {index} missing from class mapping")
                writer.writerow([name, str(mapping[int(index)]).zfill(4)])
        report["class_mapping"] = str(mapping_path)
        report["class_mapping_sha256"] = sha256_file(mapping_path)
        report["prediction_csv"] = str(csv_path)
        report["prediction_csv_sha256"] = sha256_file(csv_path)
    atomic_json_dump(report, output_dir / "apply_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
