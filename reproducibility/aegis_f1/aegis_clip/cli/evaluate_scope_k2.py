"""Evaluate all six SCOPE-K2 methods with conditional grouped nested OOF."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from scipy.stats import binomtest

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.scope_cache import (
    load_scope_cache,
    replicate_semantic_sha256,
    semantic_sha256,
    validate_evidence_cache,
    validate_parent_cache,
)
from aegis_clip.scope_crossfit import (
    METHODS,
    cluster_bootstrap_delta,
    fit_full_scope_deployment,
    method_metrics,
    promotion_gate,
    run_conditional_nested_oof,
    validate_fold_artifact,
)
from aegis_clip.scope_protocol import load_scope_protocol, verify_scope_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-cache", type=Path, required=True)
    parser.add_argument("--evidence-cache", type=Path, required=True)
    parser.add_argument("--replicate-parent-cache", type=Path, required=True)
    parser.add_argument("--replicate-evidence-cache", type=Path, required=True)
    parser.add_argument("--fold-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _threshold_payload(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    payload.pop("switch_mask", None)
    return payload


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "method": result.method,
        "beta_by_outer_fold": list(result.beta_by_outer_fold),
        "oof_thresholds": [_threshold_payload(value) for value in result.oof_thresholds],
        "mapped_thresholds": [_threshold_payload(value) for value in result.mapped_thresholds],
        "failure_reasons": list(result.failure_reasons),
    }


def _replicate_audit(parent1: dict, parent2: dict, evidence1: dict, evidence2: dict) -> dict[str, Any]:
    parent_semantic_1 = semantic_sha256(parent1)
    parent_semantic_2 = semantic_sha256(parent2)
    evidence_replicate_1 = replicate_semantic_sha256(evidence1)
    evidence_replicate_2 = replicate_semantic_sha256(evidence2)
    checks = {
        "parent_semantic_equal": parent_semantic_1 == parent_semantic_2,
        "evidence_replicate_semantic_equal": evidence_replicate_1 == evidence_replicate_2,
        "evidence1_bound_to_parent1_file": evidence1["parent_cache_sha256"] == parent1["_cache_sha256"],
        "evidence2_bound_to_parent2_file": evidence2["parent_cache_sha256"] == parent2["_cache_sha256"],
        "evidence_parent_semantic_equal": evidence1["parent_semantic_sha256"] == evidence2["parent_semantic_sha256"] == parent_semantic_1,
    }
    if not all(checks.values()):
        raise ValueError(f"run1/run2 replicate audit failed: {checks}")
    return {
        "checks": checks,
        "parent_file_sha256": [parent1["_cache_sha256"], parent2["_cache_sha256"]],
        "parent_semantic_sha256": parent_semantic_1,
        "evidence_file_sha256": [evidence1["_cache_sha256"], evidence2["_cache_sha256"]],
        "evidence_instance_semantic_sha256": [evidence1["_semantic_sha256"], evidence2["_semantic_sha256"]],
        "evidence_replicate_semantic_sha256": evidence_replicate_1,
    }


def _model_audit(evidence: dict, protocol: Any) -> dict[str, Any]:
    classifier = evidence["classifier_space_audit"]
    antisymmetry = evidence["antisymmetry_audit"]
    fixed = protocol.fixed["evidence"]
    checks = {
        "classifier_base": float(classifier["base_max_abs_error"]) <= float(fixed["linear_gate_atol"]),
        "classifier_dual": float(classifier["dual_max_abs_error"]) <= float(fixed["linear_gate_atol"]),
        "canonical_antisymmetry_bitwise": antisymmetry.get("canonical_bitwise") is True,
        "independent_antisymmetry": float(antisymmetry["independent_max_abs_error"]) <= float(fixed["antisymmetry_atol"]),
        "all_weight_norms_valid": bool(torch.as_tensor(evidence["weight_norm_valid"]).all()),
        "node_count_49": evidence["grid_shape"] == [7, 7],
        "edge_count_84": tuple(torch.as_tensor(evidence["edges"]).shape) == (84, 2),
    }
    return {"checks": checks, "passed": all(checks.values()), "classifier": classifier, "antisymmetry": antisymmetry}


def _write_metrics_csv(path: Path, metrics: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for method, values in metrics.items():
        for fold in values["folds"]:
            rows.append({
                "method": method,
                **fold,
                "pooled_correct": values["correct"],
                "pooled_accuracy": values["accuracy"],
                "clean_correct": values["clean_correct"],
                "clean_accuracy": values["clean_accuracy"],
                "corrections": values["corrections"],
                "regressions": values["regressions"],
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path, metrics: dict[str, dict[str, Any]], decision: dict[str, Any],
    bootstrap: dict[str, Any],
) -> None:
    lines = [
        "# SCOPE-K2 conditional grouped nested OOF report",
        "",
        "> Conditional grouped nested OOF given the frozen FULLFT_DUAL parent. "
        "The parent was trained with validation overlap; these folds isolate only beta, "
        "threshold, duplicate groups, ablations, and the promotion decision.",
        "",
        "| Method | Raw correct | Raw accuracy | Delta (pp) | Clean correct | Clean accuracy | Clean delta (pp) | Switches | Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = metrics[method]
        lines.append(
            f"| {method} | {value['correct']} / {value['rows']} | {100*value['accuracy']:.6f}% | "
            f"{100*value['delta_accuracy']:+.6f} | {value['clean_correct']} / {value['clean_rows']} | "
            f"{100*value['clean_accuracy']:.6f}% | {100*value['clean_delta_accuracy']:+.6f} | "
            f"{value['switches']} | {value['corrections']-value['regressions']:+d} |"
        )
    lines.extend([
        "", "## Paired cluster bootstrap", "",
        f"SCOPE minus Parent: point={100*bootstrap['point']:+.6f}pp, "
        f"95% CI=[{100*bootstrap['lower']:+.6f}, {100*bootstrap['upper']:+.6f}]pp, "
        f"draws={bootstrap['draws']}, seed={bootstrap['seed']}.",
        "", "## Promotion gates", "",
    ])
    for name, passed in decision["gates"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend([
        "", f"Final decision: **{'PROMOTE' if decision['promoted'] else 'REJECT / EXACT PARENT FALLBACK'}**.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    protocol = load_scope_protocol(args.config)
    verify_scope_assets(protocol)
    parent1 = load_scope_cache(args.parent_cache)
    parent2 = load_scope_cache(args.replicate_parent_cache)
    evidence1 = load_scope_cache(args.evidence_cache)
    evidence2 = load_scope_cache(args.replicate_evidence_cache)
    validate_parent_cache(parent1, protocol, "validation")
    validate_parent_cache(parent2, protocol, "validation")
    validate_evidence_cache(evidence1, parent1, protocol, "validation")
    validate_evidence_cache(evidence2, parent2, protocol, "validation")
    replicate_audit = _replicate_audit(parent1, parent2, evidence1, evidence2)
    model_audit = _model_audit(evidence1, protocol)
    if not model_audit["passed"]:
        raise ValueError(f"classifier/grid/antisymmetry audit failed: {model_audit['checks']}")

    group_map = json.loads(protocol.assets.group_artifact.read_text(encoding="utf-8"))
    groups = [str(group_map[path]) for path in parent1["paths"]]
    folds = load_scope_cache(args.fold_artifact)
    validate_fold_artifact(folds, parent1, groups, protocol)

    results = {
        method: run_conditional_nested_oof(
            parent1, evidence1, folds, method,
            protocol.fixed["beta_solver"], protocol.fixed["threshold"],
        )
        for method in METHODS
    }
    metrics = {method: method_metrics(result, parent1) for method, result in results.items()}
    for method, value in metrics.items():
        changed = value["corrections"] + value["regressions"]
        value["mcnemar_exact_p"] = float(binomtest(value["corrections"], changed, 0.5).pvalue) if changed else 1.0
    labels = torch.as_tensor(parent1["label"], dtype=torch.int64)
    parent_correct = results["parent"].predictions.eq(labels)
    scope_correct = results["scope"].predictions.eq(labels)
    bootstrap = cluster_bootstrap_delta(
        parent_correct, scope_correct, groups, labels, folds["outer_fold_id"],
        draws=int(protocol.fixed["bootstrap"]["draws"]),
        seed=int(protocol.fixed["bootstrap"]["seed"]),
        quantile_method=str(protocol.fixed["bootstrap"]["quantile_method"]),
    )
    scope = metrics["scope"]
    gate_input = {
        "raw_parent_correct": metrics["parent"]["correct"],
        "raw_scope_correct": scope["correct"], "raw_total": scope["rows"],
        "clean_parent_correct": metrics["parent"]["clean_correct"],
        "clean_scope_correct": scope["clean_correct"], "clean_total": scope["clean_rows"],
        "corrections": scope["corrections"], "regressions": scope["regressions"],
        "fold_deltas": [value["delta_correct"] for value in scope["folds"]],
        "bootstrap_lower": bootstrap.lower,
        "pace_raw_correct": metrics["pace"]["correct"],
        "no_topology_raw_correct": metrics["no_topology"]["correct"],
        "pace_clean_correct": metrics["pace"]["clean_correct"],
        "no_topology_clean_correct": metrics["no_topology"]["clean_correct"],
        "audits_passed": all(replicate_audit["checks"].values()) and model_audit["passed"],
    }
    gate_decision = promotion_gate(gate_input)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    deployment_path: str | None = None
    deployment_error: str | None = None
    if gate_decision.promoted:
        try:
            fitted = fit_full_scope_deployment(parent1, evidence1, groups, protocol)
            deployment = {
                "schema": "scope_deployment_v1",
                "method": "scope",
                "formula": "eta=parent_margin+beta*scope_evidence",
                "beta": fitted["beta"], "intercept": 0.0,
                "beta_by_full_oof_fold": fitted["beta_by_full_oof_fold"],
                "threshold": _threshold_payload(fitted["threshold"]),
                "mapped_threshold": _threshold_payload(fitted["mapped_threshold"]),
                "eligible_count": fitted["eligible_count"],
                "pair_training_count": fitted["pair_training_count"],
                "oof_score_sha256": fitted["oof_score_sha256"],
                "refit_score_sha256": fitted["refit_score_sha256"],
                "parent_cache_sha256": parent1["_cache_sha256"],
                "parent_semantic_sha256": semantic_sha256(parent1),
                "evidence_cache_sha256": evidence1["_cache_sha256"],
                "evidence_semantic_sha256": evidence1["_semantic_sha256"],
                "fold_artifact_sha256": folds["_cache_sha256"],
                "protocol_sha256": sha256_file(protocol.config_path),
                "view_order": evidence1["view_order"], "view_weights": evidence1["view_weights"],
                "edge_sha256": evidence1["edges_sha256"],
                "gate_spec": protocol.fixed["evidence"],
                "lineage": parent1["lineage"],
            }
            deployment_file = output / "deployment.json"
            atomic_json_dump(deployment, deployment_file)
            deployment_file.chmod(0o444)
            deployment_path = str(deployment_file)
        except Exception as error:  # fail closed after a nominal outer promotion
            deployment_error = str(error)
            gate_decision = promotion_gate(dict(gate_input, audits_passed=False))

    decision = {
        "schema": "scope_decision_v1",
        "promoted": gate_decision.promoted,
        "terminal_submission": "scope_k2" if gate_decision.promoted else "parent_fallback",
        "terminal_submission_checker": "pending_external_checker",
        "gates": gate_decision.gates,
        "gate_evidence": gate_input,
        "conditional_parent": True,
        "conditional_parent_limitation": "FULLFT_DUAL final parent used validation_overlap_with_training=true",
        "deployment": deployment_path,
        "deployment_error": deployment_error,
        "fallback": {
            "csv": str(protocol.assets.fallback_csv), "csv_sha256": protocol.assets.fallback_csv_sha256,
            "zip": str(protocol.assets.fallback_zip), "zip_sha256": protocol.assets.fallback_zip_sha256,
            "manifest": str(protocol.assets.fallback_manifest), "manifest_sha256": protocol.assets.fallback_manifest_sha256,
        },
        "replicate_audit": replicate_audit,
        "model_audit": model_audit,
        "fold_artifact": {"path": str(args.fold_artifact.resolve()), "sha256": folds["_cache_sha256"], "semantic_sha256": folds["_semantic_sha256"]},
    }
    atomic_json_dump(decision, output / "decision.json")
    atomic_json_dump(metrics, output / "metrics.json")
    atomic_json_dump({method: _result_payload(value) for method, value in results.items()}, output / "crossfit.json")
    atomic_json_dump(asdict(bootstrap), output / "bootstrap.json")
    atomic_json_dump({"replicate": replicate_audit, "model": model_audit}, output / "audit.json")
    _write_metrics_csv(output / "fold_metrics.csv", metrics)
    _write_report(output / "report.md", metrics, decision, asdict(bootstrap))
    print(json.dumps({
        "promoted": decision["promoted"], "gates": decision["gates"],
        "scope": metrics["scope"], "bootstrap": asdict(bootstrap),
        "output_dir": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
