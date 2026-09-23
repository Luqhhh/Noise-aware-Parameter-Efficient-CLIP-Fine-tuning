#!/usr/bin/env python3
"""Convert the current-split OOF quality asset into twelve C-group bundles."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

C_PARAMS: dict[str, dict[str, Any]] = {
    "C01": {"mode": "soft", "rho": 0.2, "gate": 0.8, "fraction": 0.20},
    "C02": {"mode": "soft", "rho": 0.2, "gate": 0.9, "fraction": 0.20},
    "C03": {"mode": "soft", "rho": 0.4, "gate": 0.8, "fraction": 0.20},
    "C04": {"mode": "soft", "rho": 0.4, "gate": 0.9, "fraction": 0.20},
    "C05": {"mode": "hard", "rho": 1.0, "gate": 0.90, "fraction": 0.05},
    "C06": {"mode": "hard", "rho": 1.0, "gate": 0.90, "fraction": 0.10},
    "C07": {"mode": "hard", "rho": 1.0, "gate": 0.95, "fraction": 0.05},
    "C08": {"mode": "hard", "rho": 1.0, "gate": 0.95, "fraction": 0.10},
    "C09": {"mode": "unlabel", "rho": 1.0, "gate": 0.9, "fraction": 0.10, "consistency": 0.5},
    "C10": {"mode": "unlabel", "rho": 1.0, "gate": 0.9, "fraction": 0.10, "consistency": 1.0},
    "C11": {"mode": "unlabel", "rho": 1.0, "gate": 0.9, "fraction": 0.20, "consistency": 0.5},
    "C12": {"mode": "unlabel", "rho": 1.0, "gate": 0.9, "fraction": 0.20, "consistency": 1.0},
}
KNN_CONSISTENCY_THRESHOLD = 0.60


def select_repair(quality: pd.DataFrame, *, gate: float, fraction: float) -> np.ndarray:
    labels = quality["original_label"].to_numpy()
    eligible = (
        (quality["p_top1"].to_numpy(dtype=np.float64) >= float(gate))
        & (quality["knn_agreement"].to_numpy(dtype=np.float64) >= KNN_CONSISTENCY_THRESHOLD)
        & (quality["oof_top1"].to_numpy() != labels)
    )
    selected = np.zeros(len(quality), dtype=bool)
    for label in sorted(set(labels.tolist())):
        class_rows = np.flatnonzero((labels == label) & eligible)
        if class_rows.size == 0:
            continue
        cap = int(math.floor(float(fraction) * int((labels == label).sum())))
        if cap <= 0:
            continue
        ordered = class_rows[
            np.argsort(-quality["p_top1"].to_numpy(dtype=np.float64)[class_rows], kind="stable")
        ]
        selected[ordered[:cap]] = True
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    quality_dir = Path(args.quality_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    quality = pd.read_csv(
        quality_dir / "sample_quality.csv", dtype={"sample_id": str, "image_path": str}
    )
    oof = torch.load(quality_dir / "oof_logits.pt", map_location="cpu", weights_only=False)
    logits_by_id = {
        str(sample_id): logits
        for sample_id, logits in zip(oof["sample_ids"], oof["logits"].float())
    }
    logits = torch.stack([logits_by_id[str(sample_id)] for sample_id in quality["sample_id"]])
    probabilities = torch.softmax(logits, dim=1)
    original_labels = quality["original_label"].to_numpy(dtype=np.int64)
    report = {}
    for trial_id, spec in C_PARAMS.items():
        selected = select_repair(quality, gate=float(spec["gate"]), fraction=float(spec["fraction"]))
        clean = np.ones(len(quality), dtype=np.float32)
        pseudo_label = original_labels.copy()
        pseudo_confidence = np.zeros(len(quality), dtype=np.float32)
        correction_alpha = np.zeros(len(quality), dtype=np.float32)
        pseudo_label[selected] = quality["oof_top1"].to_numpy(dtype=np.int64)[selected]
        pseudo_confidence[selected] = quality["p_top1"].to_numpy(dtype=np.float32)[selected]
        correction_alpha[selected] = float(spec["rho"])
        if spec["mode"] == "unlabel":
            clean[selected] = float(spec["consistency"])
        payload: dict[str, Any] = {
            "paths": quality["image_path"].astype(str).tolist(),
            "clean_probability": torch.tensor(clean),
            "pseudo_label": torch.tensor(pseudo_label, dtype=torch.long),
            "pseudo_confidence": torch.tensor(pseudo_confidence),
            "correction_alpha": torch.tensor(correction_alpha),
            "metadata": {
                "trial_id": trial_id,
                "source": str(quality_dir / "sample_quality.csv"),
                "mode": spec["mode"],
                "rho": spec["rho"],
                "gate": spec["gate"],
                "fraction": spec["fraction"],
                "knn_consistency_threshold": KNN_CONSISTENCY_THRESHOLD,
                "selected_count": int(selected.sum()),
                "selected_fraction": float(selected.mean()),
            },
        }
        if spec["mode"] == "soft":
            soft = torch.zeros(len(quality), probabilities.shape[1], dtype=torch.float16)
            soft[selected] = probabilities[selected].to(torch.float16)
            payload["pseudo_soft"] = soft
        torch.save(payload, output / f"{trial_id}.pt")
        report[trial_id] = {
            **payload["metadata"],
            "selected_by_class": {
                str(int(label)): int(selected[original_labels == label].sum())
                for label in sorted(set(original_labels.tolist()))
            },
        }
    (output / "quality_bundles.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
