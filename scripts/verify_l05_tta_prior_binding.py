#!/usr/bin/env python3
"""Verify the L05 TTA+prior artifact chain without touching the test set.

Checks, in order:

1. the protected candidate package still matches the hashes recorded in the
   protected-package registry (files are only read, never written),
2. the validation branch cache re-binds to the checkpoint, class mapping, frozen
   split, content-group partition, resolution/preprocessing/precision and TTA
   rule,
3. the frozen-recipe bias fitted from that cache reproduces the recorded local
   validation metrics,
4. the full-fit bias SHA-256 is recorded so a future seed cannot reuse it
   without an explicit binding mismatch.

Outputs a machine-readable audit record.  Reads no test image and runs no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
if str(AEGIS_ROOT) not in sys.path:
    sys.path.insert(0, str(AEGIS_ROOT))

from aegis_clip.balanced_inference import prediction_metrics  # noqa: E402
from aegis_clip.config import load_config  # noqa: E402
from aegis_clip.runtime import sha256_file  # noqa: E402
from aegis_clip.tta_prior_binding import (  # noqa: E402
    apply_fixed_recipe,
    checkpoint_identity,
    fit_bound_prior,
    resolved_recipe,
    validation_cache_identity,
    verify_prior_record,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val-branch-cache", required=True)
    parser.add_argument(
        "--protected-registry",
        default="results/f05_focus_l05_protected_packages.json",
    )
    parser.add_argument(
        "--recorded-result",
        default="results/f05_focus_l05_tta_prior_platform_20260925.json",
    )
    parser.add_argument("--output", default="results/f05_focus_l05_binding_audit_20260925.json")
    return parser.parse_args()


def _verify_protected(registry_path: Path) -> dict:
    if not registry_path.is_file():
        return {"registry": str(registry_path), "status": "missing"}
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    packages = []
    for entry in registry.get("protected_packages", []):
        package_dir = (ROOT / entry["repo_path"]).resolve()
        files = {}
        ok = package_dir.is_dir()
        for name, expected in entry.get("files", {}).items():
            path = package_dir / name
            present = path.is_file()
            actual = sha256_file(path) if present else None
            match = present and actual == expected
            files[name] = {"present": present, "sha256": actual, "matches_record": match}
            ok = ok and match
        packages.append(
            {
                "candidate": entry.get("candidate"),
                "repo_path": str(package_dir),
                "status": "verified" if ok else "MISMATCH",
                "files": files,
            }
        )
    return {
        "registry": str(registry_path),
        "status": "verified" if all(p["status"] == "verified" for p in packages) else "MISMATCH",
        "packages": packages,
    }


def main() -> int:
    args = _parse_args()
    config = load_config(Path(args.config).expanduser().resolve())
    cache_path = Path(args.val_branch_cache).expanduser().resolve()
    checkpoint = checkpoint_identity(args.checkpoint, config)
    cache_identity = validation_cache_identity(cache_path, config, checkpoint)
    recipe = resolved_recipe(enforce_fixed=True)

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    bias, report, record = fit_bound_prior(payload, cache_identity, recipe=recipe)
    verify_prior_record(record, cache_identity, recipe=recipe, bias=bias)
    corrected = apply_fixed_recipe(payload, bias, recipe=recipe)
    metrics = prediction_metrics(
        corrected.argmax(dim=1),
        labels=payload["labels"],
        clean_probability=payload["clean_probability"],
        pseudo_labels=payload["pseudo_labels"],
        correction_alpha=payload["correction_alpha"],
        num_classes=int(cache_identity["num_classes"]),
        clean_core_threshold=0.70,
    )

    recorded_path = Path(args.recorded_result).expanduser()
    recorded = (
        json.loads(recorded_path.read_text(encoding="utf-8"))
        if recorded_path.is_file()
        else None
    )
    recorded_metrics = None
    reproduces = None
    if recorded:
        if "local" in recorded:
            recorded_metrics = {
                "macro": recorded["local"]["selected_raw_macro"],
                "micro": recorded["local"]["selected_raw_micro"],
            }
        else:
            recorded_metrics = {
                "macro": recorded["local_validation"]["selected_raw_macro"],
                "micro": recorded["local_validation"]["selected_raw_micro"],
            }
        tolerance = 5.0e-7 if "local" not in recorded else 1.0e-9
        reproduces = (
            abs(float(metrics["raw_macro"]) - float(recorded_metrics["macro"])) <= tolerance
            and abs(float(metrics["raw_micro"]) - float(recorded_metrics["micro"])) <= tolerance
        )

    audit = {
        "schema_version": 1,
        "diagnostic": "l05_tta_prior_binding_audit",
        "test_data_used": False,
        "protected_packages": _verify_protected(Path(args.protected_registry).expanduser()),
        "recipe": recipe,
        "checkpoint": checkpoint,
        "cache_identity": cache_identity,
        "full_validation_fit": {
            "macro": float(metrics["raw_macro"]),
            "micro": float(metrics["raw_micro"]),
            "bias_sha256": record["bias_sha256"],
            "bias_fit_iterations": int(report["iterations"]),
            "fit_scope": record["fit_scope"],
            "prior_record": record,
        },
        "recorded_result": {
            "path": str(recorded_path),
            "expected": recorded_metrics,
            "reproduces_recorded_metrics": reproduces,
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["protected_packages"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "macro": audit["full_validation_fit"]["macro"],
                "micro": audit["full_validation_fit"]["micro"],
                "bias_sha256": audit["full_validation_fit"]["bias_sha256"],
                "reproduces_recorded_metrics": reproduces,
            },
            indent=2,
        )
    )
    print(f"wrote {output}")
    if audit["protected_packages"]["status"] != "verified" or reproduces is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
