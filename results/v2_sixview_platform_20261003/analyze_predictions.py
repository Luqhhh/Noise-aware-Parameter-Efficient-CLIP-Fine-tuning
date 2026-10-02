"""Read-only diagnostic over already delivered submission CSVs.

No model, no test inference, no labels. It quantifies how far each package's
hard argmax distribution is from the uniform test prior, how many predictions
the fixed 200-iteration bias flipped inside each same-checkpoint pair, and how
much the delivered packages differ from each other. Output is a JSON record.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PACKAGES = {
    "v1_768_sixview_raw": "/home/lux1/noise-worktrees/v1_768_full_test_bias_20261001/outputs/codex/v1_768_full_test_bias_20261001/submission_raw/pred_results.csv",
    "v1_768_sixview_bias": "/home/lux1/noise-worktrees/v1_768_full_test_bias_20261001/outputs/codex/v1_768_full_test_bias_20261001/submission/pred_results.csv",
    "v1_512_sixview_raw": "/home/lux1/noise-worktrees/v1_512_test_bias_control_20261001/outputs/codex/v1_512_test_bias_control_20261001/submission_raw/pred_results.csv",
    "v1_512_sixview_bias": "/home/lux1/noise-worktrees/v1_512_test_bias_control_20261001/outputs/codex/v1_512_test_bias_control_20261001/submission/pred_results.csv",
    "v2_sixview_raw": "/mnt/c/Users/lqh22/Downloads/v2_continuation_20261001/submission_raw_sixview/pred_results.csv",
    "v2_sixview_bias": "/mnt/c/Users/lqh22/Downloads/v2_continuation_20261001/submission_bias_sixview/pred_results.csv",
}
PAIRS = (("v1_768_sixview_raw", "v1_768_sixview_bias"),
         ("v1_512_sixview_raw", "v1_512_sixview_bias"),
         ("v2_sixview_raw", "v2_sixview_bias"))
COMPARISONS = (("v1_768_sixview_raw", "v2_sixview_raw"),
               ("v1_768_sixview_bias", "v2_sixview_bias"),
               ("v1_512_sixview_raw", "v1_768_sixview_raw"))
OUT = Path(__file__).resolve().parent / "prediction_distribution_diagnostic.json"


def read_labels(path):
    names, labels = [], []
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            names.append(row[0])
            labels.append(int(row[1]))
    return names, np.asarray(labels, dtype=np.int64)


def main():
    labels, names = {}, None
    for key, path in PACKAGES.items():
        rows, values = read_labels(path)
        if names is None:
            names = rows
        elif rows != names:
            raise ValueError(f"image order differs for {key}")
        labels[key] = values
    total, classes = len(names), 750
    expected = total / classes
    stats = {}
    for key, values in labels.items():
        counts = np.bincount(values, minlength=classes).astype(float)
        stats[key] = dict(
            rows=total,
            classes_used=int((counts > 0).sum()),
            argmax_min=int(counts.min()), argmax_max=int(counts.max()),
            argmax_std=round(float(counts.std()), 4),
            expected_per_class=round(expected, 4),
            max_over_expected=round(float(counts.max() / expected), 4),
            chi2_vs_uniform=round(float(((counts - expected) ** 2 / expected).sum()), 1),
        )
    record = dict(
        record_id="V2_SIXVIEW_PLATFORM_20261003_prediction_distribution",
        note="read-only, no labels, no inference; only the delivered CSVs",
        packages=stats,
        same_checkpoint_bias_flips={f"{a} -> {b}": int((labels[a] != labels[b]).sum()) for a, b in PAIRS},
        cross_package_disagreement={f"{a} vs {b}": int((labels[a] != labels[b]).sum()) for a, b in COMPARISONS},
    )
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
