#!/usr/bin/env python3
"""Summarize archived content scores and L05 supervision; never encode or train."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quantiles(values):
    ordered = sorted(values)
    result = {}
    for fraction in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]:
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        result[str(fraction)] = (
            ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
            if upper != lower else ordered[lower]
        ) if ordered else None
    return result


def confidence_bin(py, boundaries):
    low, high = boundaries
    return "low" if py < low else "middle" if py < high else "high"


def supervision(rows, config, total_gradient=None):
    bins = Counter({key: 0 for key in ["low", "middle", "high"]})
    q = config["gce_q"]
    gradient = 0.0
    bin_gradient = Counter({key: 0.0 for key in bins})
    bin_scale = Counter({key: 0.0 for key in bins})
    for row in rows:
        py = row["py"]
        key = confidence_bin(py, config["confidence_boundaries"])
        bins[key] += 1
        contribution = 2 * py ** q * (1 - py)
        gradient += contribution
        bin_gradient[key] += contribution
        bin_scale[key] += py ** q
    return {
        "rows": len(rows),
        "original_label_errors": sum(not row["correct"] for row in rows),
        "py_bins": dict(bins),
        "py_bin_logit_gradient_l1_sum": dict(bin_gradient),
        "py_bin_mean_gce_scale": {key: bin_scale[key] / count if count else None for key, count in bins.items()},
        "py_lt_001": sum(row["py"] < 0.01 for row in rows),
        "py_quantiles": quantiles([row["py"] for row in rows]),
        "mean_gce_scale": sum(row["py"] ** q for row in rows) / len(rows) if rows else None,
        "final_center_logit_gradient_l1_sum": gradient,
        "final_center_logit_gradient_l1_share": gradient / total_gradient if total_gradient else None,
    }


def load_joined(archive, evidence_path, p0_path, rule_path, descriptions_path):
    manifest = json.loads((archive / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        require(sha(archive / name) == expected, "Changed archive: " + name)
    rule = json.loads(rule_path.read_text())
    frozen = json.loads((archive / "frozen_summary.json").read_text())
    p0 = json.loads(p0_path.read_text())
    require(sha(rule_path) == frozen["rule_sha256"], "Frozen rule changed")
    require(sha(descriptions_path) == rule["descriptions_sha256"], "Descriptions changed")
    require(sha(evidence_path) == frozen["evidence_sha256"], "Evidence hash differs from content archive")
    require(sha(evidence_path) == p0["archived_artifacts"][evidence_path.name], "Evidence hash differs from P0")
    require(frozen["feature_manifest_sha256"] == p0["identity"]["feature_manifest_sha256"], "Stage mismatch")
    phrases = json.loads(descriptions_path.read_text())["N"]
    with gzip.open(archive / "frozen_selection.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    by_path = {}
    for row in rows:
        path = row["image_path"]
        require(path not in by_path, "Duplicate content path: " + path)
        require(row["split"] in ["train", "val"], "Unexpected split")
        require(type(row["label"]) is int and 0 <= row["label"] < p0["identity"]["num_classes"], "Invalid label")
        require(type(row["selected"]) is bool, "Invalid membership")
        require(all(math.isfinite(row[key]) and -1 <= row[key] <= 1 for key in ["sB", "sA", "sN"]), "Invalid cosine")
        require(math.isfinite(row["margin"]), "Invalid margin")
        require(abs(row["margin"] - (row["sN"] - max(row["sB"], row["sA"]))) <= 1e-7, "Invalid protected score")
        require(row["selected"] == (row["margin"] >= rule["minimum_margin"]), "Membership changed")
        require(row["non_target_type"] in phrases, "Unknown content description")
        by_path[path] = row
    seen = set()
    with gzip.open(evidence_path, "rt") as stream:
        for evidence in csv.DictReader(stream):
            path = evidence["image_path"]
            require(path not in seen, "Duplicate evidence path: " + path)
            require(path in by_path, "Extra evidence path: " + path)
            seen.add(path)
            row = by_path[path]
            require(row["split"] == evidence["split"] and row["label"] == int(evidence["original_label"])
                    and row["content_group"] == evidence["content_group"], "Join identity mismatch: " + path)
            py = float(evidence["center_label_probability"])
            require(math.isfinite(py) and 0 <= py <= 1, "Invalid probability")
            prediction = int(evidence["l05_fixed_decode_prediction"] if row["split"] == "val" else evidence["l05_prediction"])
            require(0 <= prediction < p0["identity"]["num_classes"], "Invalid prediction")
            row.update(py=py, prediction=prediction, correct=prediction == row["label"])
    require(seen == set(by_path), "Missing evidence paths")
    for split, size_key in [("train", "training_samples"), ("val", "validation_samples")]:
        require(sum(row["split"] == split for row in rows) == p0["identity"][size_key], "Split count changed")
        for selected, name in [(True, "selected"), (False, "remaining")]:
            subset = [row for row in rows if row["split"] == split and row["selected"] == selected]
            expected = frozen["groups"][split + "_" + name]
            require(len(subset) == expected["rows"] and sum(not row["correct"] for row in subset) == expected["original_label_errors"], "Frozen group changed")
    require(sum(row["split"] == "val" and not row["correct"] for row in rows) == p0["baseline_errors"], "L05 validation errors changed")
    return rows, p0["identity"], rule


def recall(rows):
    counts, correct = Counter(), Counter()
    for row in rows:
        counts[row["label"]] += 1
        correct[row["label"]] += int(row["correct"])
    return {
        "rows": len(rows), "classes_present": len(counts),
        "original_label_micro": sum(correct.values()) / len(rows) if rows else None,
        "original_label_macro": sum(correct[label] / count for label, count in counts.items()) / len(counts) if counts else None,
        "original_label_errors": len(rows) - sum(correct.values()),
    }


def analyze(rows, identity, rule, config):
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    selected = [row for row in train if row["selected"]]
    total_gradient = supervision(train, config)["final_center_logit_gradient_l1_sum"]
    thresholds = []
    for threshold in config["diagnostic_margin_thresholds"]:
        subset = [row for row in train if row["margin"] >= threshold]
        thresholds.append(dict(minimum_margin=threshold, fraction=len(subset) / len(train),
                               frozen_selection=threshold == rule["minimum_margin"],
                               **supervision(subset, config, total_gradient)))
    ranked = sorted(train, key=lambda row: (-row["margin"], row["image_path"]))
    ranks = []
    for fraction in config["diagnostic_top_fractions"]:
        size = math.ceil(len(train) * fraction)
        subset = ranked[:size]
        cutoff = subset[-1]["margin"]
        ranks.append(dict(requested_fraction=fraction, rows=size, minimum_margin=cutoff,
                          rows_including_all_cutoff_ties=sum(row["margin"] >= cutoff for row in train),
                          already_selected=sum(row["selected"] for row in subset)))
    groups = {"train_all": supervision(train, config, total_gradient),
              "train_selected": supervision(selected, config, total_gradient),
              "train_remaining": supervision([row for row in train if not row["selected"]], config, total_gradient)}
    by_class_train, by_class_val = {}, {}
    for source, target in [(train, by_class_train), (val, by_class_val)]:
        for row in source:
            target.setdefault(row["label"], []).append(row)
    classes = []
    for label in range(identity["num_classes"]):
        tr, va = by_class_train.get(label, []), by_class_val.get(label, [])
        targets = [row for row in tr if row["selected"]]
        item = {"class": label, "train_rows": len(tr), "selected_rows": len(targets),
                "selected_rate": len(targets) / len(tr) if tr else 0,
                "selected_high_py_rows": sum(row["py"] >= config["confidence_boundaries"][1] for row in targets)}
        for key, subset in [("all", va), ("selected", [row for row in va if row["selected"]]),
                            ("remaining", [row for row in va if not row["selected"]]),
                            ("bio_dominant", [row for row in va if row["sB"] >= max(row["sA"], row["sN"])])]:
            item["val_" + key + "_rows"] = len(subset)
            item["val_" + key + "_recall"] = recall(subset)["original_label_micro"]
        classes.append(item)
    concentration = []
    for size in config["concentration_top_classes"]:
        top = sorted(classes, key=lambda item: (-item["selected_rows"], item["class"]))[:size]
        count = sum(item["selected_rows"] for item in top)
        concentration.append(dict(top_classes=size, selected_rows=count,
                                  share_of_selected=count / len(selected) if selected else None))
    rate_slices = []
    for minimum in config["class_rate_boundaries"]:
        subset = [item for item in classes if item["selected_rate"] >= minimum]
        rate_slices.append(dict(minimum_class_rate=minimum, classes=len(subset),
                               selected_rows=sum(item["selected_rows"] for item in subset),
                               train_rows=sum(item["train_rows"] for item in subset)))
    high_classes = {item["class"] for item in classes if item["selected_rate"] >= config["validation_class_slice_minimum_training_rate"]}
    validation = {}
    for name, labels in [("high_training_content_rate", high_classes),
                         ("other_training_content_rate", set(range(identity["num_classes"])) - high_classes)]:
        subset = [row for row in val if row["label"] in labels]
        validation[name] = dict(training_classes=sorted(labels), all=recall(subset),
            selected=recall([row for row in subset if row["selected"]]),
            remaining=recall([row for row in subset if not row["selected"]]),
            bio_dominant=recall([row for row in subset if row["sB"] >= max(row["sA"], row["sN"])]))
    types = {phrase: supervision([row for row in selected if row["non_target_type"] == phrase], config, total_gradient)
             for phrase in sorted({row["non_target_type"] for row in train})}
    naive = [row for row in train if row["sN"] - row["sB"] >= rule["minimum_margin"]]
    summary = dict(experiment=config["experiment"], identity=identity,
        scoring="sN-max(sB,sA); maximum normalized cosine per group", frozen_minimum_margin=rule["minimum_margin"],
        training_selected_fraction=len(selected) / len(train),
        selected_training_classes=sum(item["selected_rows"] > 0 for item in classes),
        selected_content_groups=len({row["content_group"] for row in selected}),
        naive_without_protection_rows=len(naive), protected_from_naive_rule_rows=sum(not row["selected"] for row in naive),
        margin_quantiles=quantiles([row["margin"] for row in train]),
        diagnostic_thresholds=thresholds, diagnostic_top_fractions=ranks, supervision=groups,
        selected_content_types=types, concentration=concentration, class_rate_slices=rate_slices,
        highest_rate_classes=sorted(classes, key=lambda item: (-item["selected_rate"], item["class"]))[:10],
        validation_class_slices=validation, validation_all=recall(val),
        training_started_this_analysis=False, images_reviewed_this_analysis=0, mask_created_this_analysis=False,
        limitations=["Scores and types are content proxies, not contamination probabilities or labels.",
          "Unselected and bio-dominant proxies are not clean-photo ground truth; class differences are observational.",
          "py is the final L05 center snapshot, not actual training history. Validation correctness uses fixed L05 decoding.",
          "GCE logit-gradient L1 = 2*py**q*(1-py); confidence or gradient share does not establish harmful supervision.",
          "Existing four-epoch paired observation is separate; its six-epoch primary endpoint was not completed.",
          "No causal or platform improvement claim; no automatic training authorization from this report."])
    return summary, classes


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--p0-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Output exists; use a new directory")
    config = json.loads(args.config.read_text())
    rule_path = ROOT / "configs/p75_semantic_content_probe_20260929/frozen_rule.json"
    descriptions_path = rule_path.with_name("development.json")
    rows, identity, rule = load_joined(args.archive, args.evidence, args.p0_summary, rule_path, descriptions_path)
    summary, classes = analyze(rows, identity, rule, config)
    with (args.archive / "frozen_class_coverage.csv").open() as stream:
        coverage = list(csv.DictReader(stream))
    require(len(coverage) == len(classes), "Class coverage size changed")
    for expected, actual in zip(coverage, classes):
        require(int(expected["class"]) == actual["class"] and int(expected["raw"]) == actual["train_rows"]
                and int(expected["selected"]) == actual["selected_rows"], "Class coverage differs")
    frozen = json.loads((args.archive / "frozen_summary.json").read_text())
    require(math.isclose(summary["supervision"]["train_selected"]["final_center_logit_gradient_l1_share"],
                         frozen["groups"]["train_selected"]["final_logit_l1_share"], rel_tol=1e-10), "GCE archive differs")
    args.out.mkdir(parents=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(args.out / "class_statistics.csv", classes)
    write_csv(args.out / "selected_supervision.csv", [dict(split=row["split"], image_path=row["image_path"],
        label=row["label"], content_group=row["content_group"], margin=row["margin"],
        non_target_type=row["non_target_type"], center_py=row["py"],
        confidence_bin=confidence_bin(row["py"], config["confidence_boundaries"]),
        prediction=row["prediction"], original_label_correct=row["correct"]) for row in rows if row["selected"]])
    source_paths = [args.archive / "manifest.json", args.archive / "frozen_selection.jsonl.gz",
                    args.archive / "frozen_summary.json", args.evidence, args.p0_summary, args.config,
                    rule_path, descriptions_path, Path(__file__).resolve()]
    manifest = {"inputs": {str(path.resolve().relative_to(ROOT)): sha(path) for path in source_paths},
                "outputs": {path.name: sha(path) for path in sorted(args.out.iterdir())},
                "joined_rows": len(rows), "duplicate_or_missing_paths": 0,
                "training_started_this_analysis": False}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ["experiment", "training_selected_fraction",
        "selected_training_classes", "concentration", "class_rate_slices", "supervision"]}, indent=2))


if __name__ == "__main__":
    main()
