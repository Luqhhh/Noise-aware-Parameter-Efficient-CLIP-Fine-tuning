"""Complete holdout corrections/regressions and training-defined class slices."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from v2.plan import dump, json_read, require, sha
from .plan import ARMS, read_rows, verify


def paired_metrics(labels, original, candidate, classes, groups):
    labels, original, candidate = [np.asarray(x, dtype=np.int64) for x in (labels, original, candidate)]
    require(labels.ndim == 1 and labels.shape == original.shape == candidate.shape and len(labels) > 0,
            "Pair must cover the same nonempty validation rows")
    require(all(((x >= 0) & (x < classes)).all() for x in (labels, original, candidate)), "Invalid pair labels")
    a, b = original == labels, candidate == labels
    repaired, regressed = ~a & b, a & ~b
    counts = np.bincount(labels, minlength=classes)
    ca = np.bincount(labels[a], minlength=classes)
    cb = np.bincount(labels[b], minlength=classes)

    def summarize(class_ids):
        mask = np.isin(labels, class_ids)
        present = [c for c in class_ids if counts[c] > 0]
        ma = float(np.mean(ca[present] / counts[present])) if present else None
        mb = float(np.mean(cb[present] / counts[present])) if present else None
        rows = int(mask.sum())
        micro_a, micro_b = (float(a[mask].mean()), float(b[mask].mean())) if rows else (None, None)
        return dict(rows=rows, defined_classes=len(class_ids), classes_present=len(present),
                    original=dict(macro=ma, micro=micro_a), v1_supervision=dict(macro=mb, micro=micro_b),
                    delta_macro_pp=None if ma is None else 100 * (mb - ma),
                    delta_micro_pp=None if micro_a is None else 100 * (micro_b - micro_a),
                    corrections=int((repaired & mask).sum()), regressions=int((regressed & mask).sum()),
                    net=int((repaired & mask).sum() - (regressed & mask).sum()),
                    prediction_changes=int(((original != candidate) & mask).sum()))

    slices = dict(all=summarize(list(range(classes))),
                  tail=summarize(groups["tail_classes"]), other=summarize(groups["other_classes"]),
                  few_support=summarize(groups["few_support_classes"]))
    full, tail = slices["all"], slices["tail"]
    signal = (full["net"] > 0 and full["delta_macro_pp"] > 0 and
              tail["delta_macro_pp"] is not None and tail["delta_macro_pp"] > 0)
    return dict(slices=slices, decision="signal_requires_review" if signal else "no_support_for_ladder",
                automatic_ladder=False, platform_gain_known=False, clean_labels_known=False)


def compare(plan_path, run_root, output):
    plan_path, run_root, output = Path(plan_path).resolve(), Path(run_root).resolve(), Path(output).resolve()
    plan = verify(plan_path)
    require(not output.exists(), "Refusing to overwrite a pair comparison")
    terminal = json_read(run_root / "status.json")
    require(terminal["status"] == "complete" and terminal["plan_sha256"] == sha(plan_path), "Pair is incomplete")
    rows = read_rows(plan_path.parent / "val_manifest.csv")
    labels = np.array([r["label"] for r in rows])
    predictions, traces, initial = {}, {}, {}
    for arm in ARMS:
        root = run_root / arm
        status = json_read(root / "status.json")
        require(status["status"] == "complete" and status["updates"] == plan["updates_per_arm"] and
                status["plan_sha256"] == sha(plan_path) and status["weights"] == "last_ema",
                "Arm did not reach the fixed last-EMA endpoint")
        require(sha(root / "selected.pt") == status["checkpoint_sha256"], "Selected checkpoint changed")
        require(sha(root / "predictions.npz") == status["predictions_sha256"], "Validation predictions changed")
        require(sha(root / "steps.jsonl") == status["trace_sha256"], "Paired execution trace changed")
        with np.load(root / "predictions.npz", allow_pickle=False) as data:
            require(np.array_equal(data["labels"], labels) and
                    np.array_equal(data["image_paths"], [r["image_path"] for r in rows]),
                    "Validation labels, paths or ordering mismatch")
            predictions[arm] = data["predictions"].copy()
        traces[arm] = (root / "steps.jsonl").read_bytes()
        initial[arm] = status["initial_model_sha256"]
    require(traces[ARMS[0]] == traces[ARMS[1]], "Arm draws/mixing/LRs/optimizer updates diverged")
    require(initial[ARMS[0]] == initial[ARMS[1]], "Arm initial weights differ")
    groups = json_read(plan_path.parent / "groups.json")
    a, b = predictions[ARMS[0]], predictions[ARMS[1]]
    report = paired_metrics(labels, a, b, len(plan["classes"]), groups)
    output.mkdir(parents=True)
    with (output / "paired.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label", "original_prediction", "v1_prediction", "correction", "regression"])
        for row, pa, pb in zip(rows, a, b):
            truth = row["label"]
            writer.writerow([row["image_path"], truth, int(pa), int(pb), int(pa != truth and pb == truth),
                             int(pa == truth and pb != truth)])
    with (output / "per_class.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "train_rows", "train_groups", "active_original_rows", "active_target_rows",
                         "val_rows", "original_correct", "v1_correct", "corrections", "regressions", "net"])
        for c, name in enumerate(plan["classes"]):
            mask = labels == c
            ca, cb = (a == labels) & mask, (b == labels) & mask
            repairs, losses = int((~ca & cb & mask).sum()), int((ca & ~cb & mask).sum())
            writer.writerow([name, groups["train_rows"][c], groups["train_groups"][c],
                             groups["active_original_rows"][c], groups["active_target_rows"][c],
                             int(mask.sum()), int(ca.sum()), int(cb.sum()), repairs, losses, repairs - losses])
    report.update(status="local_result", experiment_id=plan["experiment_id"], plan_sha256=sha(plan_path),
                  run_root=str(run_root), selected_weights="last_ema", paired_trace_verified=True,
                  checkpoint_sha256={arm: json_read(run_root / arm / "status.json")["checkpoint_sha256"] for arm in ARMS},
                  paired_csv_sha256=sha(output / "paired.csv"), per_class_sha256=sha(output / "per_class.csv"),
                  platform_metrics=None)
    dump(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(args.plan, args.run_root, args.output)
    print(report["slices"])


if __name__ == "__main__":
    main()
