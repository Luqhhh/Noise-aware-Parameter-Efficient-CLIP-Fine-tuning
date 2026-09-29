"""Boundary and identity checks for offline content/supervision diagnostics."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("content_analysis", ROOT / "scripts/analyze_p75_content_contamination.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/p75_content_contamination_analysis_20260929.json").read_text())


def row(path, label, py, sb, sa, sn, split="train", correct=True):
    margin = sn - max(sb, sa)
    return dict(image_path=path, label=label, py=py, sB=sb, sA=sa, sN=sn,
                margin=margin, selected=margin >= 0.05, split=split,
                content_group=path, non_target_type="map", correct=correct,
                prediction=label if correct else 1 - label)


def test_confidence_boundaries_and_gce_gradient(config):
    rows = [dict(py=py, correct=True) for py in [0, 0.3, 0.7, 1]]
    metrics = analysis.supervision(rows, config)
    assert metrics["py_bins"] == {"low": 1, "middle": 1, "high": 2}
    expected = 2 * 0.3 ** 0.5 * 0.7 + 2 * 0.7 ** 0.5 * 0.3
    assert metrics["final_center_logit_gradient_l1_sum"] == pytest.approx(expected)
    assert metrics["py_bin_logit_gradient_l1_sum"]["low"] == 0
    assert metrics["py_bin_logit_gradient_l1_sum"]["middle"] > metrics["py_bin_logit_gradient_l1_sum"]["high"]


def test_protection_class_concentration_and_validation_proxy(config):
    rows = [row("map", 0, 0.8, 0.1, 0.1, 0.2),
            row("anatomy", 0, 0.5, 0.1, 0.25, 0.2),
            row("photo", 1, 0.95, 0.3, 0.1, 0.1),
            row("valbio", 0, 0.2, 0.3, 0.1, 0.1, "val", False),
            row("valmap", 0, 0.7, 0.1, 0.1, 0.2, "val")]
    summary, classes = analysis.analyze(rows, {"num_classes": 2}, {"minimum_margin": 0.05}, config)
    assert summary["naive_without_protection_rows"] == 2
    assert summary["protected_from_naive_rule_rows"] == 1
    assert summary["selected_training_classes"] == 1
    assert classes[0]["selected_rate"] == 0.5
    assert classes[0]["val_all_recall"] == 0.5
    assert classes[0]["val_bio_dominant_recall"] == 0
    assert classes[1]["val_all_recall"] is None
    assert summary["concentration"][0]["share_of_selected"] == 1
    assert summary["validation_class_slices"]["high_training_content_rate"]["bio_dominant"]["original_label_errors"] == 1
    assert summary["supervision"]["train_selected"]["py_bins"]["high"] == 1


def test_rank_cutoff_reports_ties(config):
    rows = [row(path, 0, 0.8, 0.1, 0.1, 0.2) for path in ["a", "b", "c"]]
    summary, _ = analysis.analyze(rows, {"num_classes": 1}, {"minimum_margin": 0.05}, config)
    top = summary["diagnostic_top_fractions"][0]
    assert top["rows"] == 1
    assert top["rows_including_all_cutoff_ties"] == 3


def fixture_archive(tmp_path, evidence_rows):
    archive = tmp_path / "archive"
    archive.mkdir()
    descriptions = tmp_path / "descriptions.json"
    descriptions.write_text(json.dumps({"N": ["map"]}))
    rule = tmp_path / "rule.json"
    rule.write_text(json.dumps({"minimum_margin": 0.05, "descriptions_sha256": analysis.sha(descriptions)}))
    rows = [row("a", 0, 0.8, 0.1, 0.1, 0.2), row("b", 1, 0.2, 0.3, 0.1, 0.1, "val", False)]
    with gzip.open(archive / "frozen_selection.jsonl.gz", "wt") as stream:
        for item in rows:
            stream.write(json.dumps({key: value for key, value in item.items() if key not in ["py", "correct", "prediction"]}) + "\n")
    evidence_path = tmp_path / "evidence.csv.gz"
    with gzip.open(evidence_path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evidence_rows[0]))
        writer.writeheader()
        writer.writerows(evidence_rows)
    groups = {"train_selected": dict(rows=1, original_label_errors=0),
              "train_remaining": dict(rows=0, original_label_errors=0),
              "val_selected": dict(rows=0, original_label_errors=0),
              "val_remaining": dict(rows=1, original_label_errors=1)}
    frozen = dict(rule_sha256=analysis.sha(rule), evidence_sha256=analysis.sha(evidence_path),
                  feature_manifest_sha256="current-stage", groups=groups)
    (archive / "frozen_summary.json").write_text(json.dumps(frozen))
    (archive / "manifest.json").write_text(json.dumps({"files": {path.name: analysis.sha(path) for path in archive.iterdir()}}))
    p0_path = tmp_path / "p0.json"
    p0_path.write_text(json.dumps(dict(archived_artifacts={evidence_path.name: analysis.sha(evidence_path)},
        identity=dict(num_classes=2, feature_manifest_sha256="current-stage", training_samples=1, validation_samples=1), baseline_errors=1)))
    return archive, evidence_path, p0_path, rule, descriptions


def evidence(path, split, label, py, prediction):
    return dict(image_path=path, split=split, original_label=label, content_group=path,
                center_label_probability=py, l05_prediction=prediction, l05_fixed_decode_prediction=prediction)


@pytest.mark.parametrize("failure", ["duplicate", "missing", "label", "nan"])
def test_join_rejects_invalid_evidence(tmp_path, failure):
    records = [evidence("a", "train", 0, 0.8, 0), evidence("b", "val", 1, 0.2, 0)]
    if failure == "duplicate":
        records.append(records[0])
    elif failure == "missing":
        records.pop()
    elif failure == "label":
        records[0]["original_label"] = 1
    else:
        records[0]["center_label_probability"] = "nan"
    with pytest.raises(ValueError):
        analysis.load_joined(*fixture_archive(tmp_path, records))


def test_join_and_tampered_archive(tmp_path):
    records = [evidence("a", "train", 0, 0.8, 0), evidence("b", "val", 1, 0.2, 0)]
    inputs = fixture_archive(tmp_path, records)
    rows, _, _ = analysis.load_joined(*inputs)
    assert rows[0]["py"] == 0.8 and not rows[1]["correct"]
    (inputs[0] / "frozen_summary.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed archive"):
        analysis.load_joined(*inputs)
