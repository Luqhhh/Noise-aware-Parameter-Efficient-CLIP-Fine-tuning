#!/usr/bin/env python3
"""Fixed-protocol evaluation for an L05 continuation candidate.

Reports exactly the two protocols the search is allowed to compare against the
existing L05 winner, with no temperature or prior-strength scan:

* ``center`` -- the single center view,
* ``decode`` -- horizontal flip, ``mean_probabilities``, T=1.4, plus a prior bias
  fitted on THIS stage's validation cache only, applied at the frozen strength 0.60.

It also records corrections and regressions against both the center view and the
flip-without-prior view.  The test set is never read; the bias is fitted on the
validation cache and only its SHA-256 is recorded.  Cache/checkpoint/class-mapping/
split/group identity is re-verified by ``aegis_clip.tta_prior_binding`` before any
bias is fitted.
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
from aegis_clip.prior_alignment import apply_prior_bias  # noqa: E402
from aegis_clip.tta import fuse_paired_logits  # noqa: E402
from aegis_clip.tta_prior_binding import (  # noqa: E402
    checkpoint_identity,
    resolved_recipe,
    validation_cache_identity,
    verify_prior_record,
    fit_bound_prior,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val-branch-cache", required=True)
    parser.add_argument("--temperature", type=float, default=1.4)
    parser.add_argument("--prior-strength", type=float, default=0.60)
    parser.add_argument("--fusion", default="mean_probabilities")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _metrics(prediction, payload, num_classes):
    return prediction_metrics(
        prediction,
        labels=payload["labels"],
        clean_probability=payload["clean_probability"],
        pseudo_labels=payload["pseudo_labels"],
        correction_alpha=payload["correction_alpha"],
        num_classes=int(num_classes),
        clean_core_threshold=0.70,
    )


def _changes(reference, candidate, labels):
    reference_correct = reference == labels
    candidate_correct = candidate == labels
    return {
        "corrections": int(((~reference_correct) & candidate_correct).sum()),
        "regressions": int((reference_correct & (~candidate_correct)).sum()),
        "changed": int((reference != candidate).sum()),
    }


def main() -> int:
    args = _parse_args()
    config = load_config(Path(args.config).expanduser().resolve())
    cache_path = Path(args.val_branch_cache).expanduser().resolve()
    checkpoint = checkpoint_identity(args.checkpoint, config)
    cache_identity = validation_cache_identity(cache_path, config, checkpoint)
    recipe = resolved_recipe(
        fusion=str(args.fusion),
        temperature=float(args.temperature),
        prior_strength=float(args.prior_strength),
        enforce_fixed=True,
    )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    labels = torch.as_tensor(payload["labels"]).long().cpu()
    num_classes = int(cache_identity["num_classes"])

    center_prediction = torch.as_tensor(payload["original_logits"]).float().argmax(dim=1)
    fused = fuse_paired_logits(
        torch.as_tensor(payload["original_logits"]).float(),
        torch.as_tensor(payload["flip_logits"]).float(),
        mode=recipe["fusion"],
        temperature=recipe["temperature"],
    )
    flip_prediction = fused.argmax(dim=1)

    bias, fit_report, prior_record = fit_bound_prior(
        payload, cache_identity, recipe=recipe
    )
    verify_prior_record(prior_record, cache_identity, recipe=recipe, bias=bias)
    corrected = apply_prior_bias(
        fused, bias, strength=float(recipe["prior_strength"])
    )
    decode_prediction = corrected.argmax(dim=1)

    center_metrics = _metrics(center_prediction, payload, num_classes)
    flip_metrics = _metrics(flip_prediction, payload, num_classes)
    decode_metrics = _metrics(decode_prediction, payload, num_classes)

    result = {
        "schema_version": 1,
        "protocol": {
            "center": "center view only",
            "decode": (
                f"{recipe['tta']}:{recipe['fusion']}:t={recipe['temperature']:g}:"
                f"validation_fitted_prior={recipe['prior_strength']:g}"
            ),
        },
        "test_data_used": False,
        "temperature_scanned": False,
        "prior_strength_scanned": False,
        "checkpoint": checkpoint["checkpoint"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "training_config_sha256": checkpoint["training_config_sha256"],
        "binding": cache_identity,
        "center": {
            "macro": float(center_metrics["raw_macro"]),
            "micro": float(center_metrics["raw_micro"]),
        },
        "decode": {
            "macro": float(decode_metrics["raw_macro"]),
            "micro": float(decode_metrics["raw_micro"]),
            "flip_no_prior_macro": float(flip_metrics["raw_macro"]),
            "flip_no_prior_micro": float(flip_metrics["raw_micro"]),
            "prior_strength": float(recipe["prior_strength"]),
            "prior_bias_sha256": prior_record["bias_sha256"],
            "prior_fit_iterations": int(fit_report["iterations"]),
            "vs_center": _changes(center_prediction, decode_prediction, labels),
            "vs_flip_no_prior": _changes(flip_prediction, decode_prediction, labels),
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("center", "decode")}, ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
