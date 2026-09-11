"""Fail-closed parent-child split lineage audit for AEGIS checkpoint swaps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import sha256_file


class LineageAuditError(ValueError):
    """Raised when the parent-child split lineage is violated."""


def _load_split(path: str | Path) -> dict[str, int]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise LineageAuditError(f"Split CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"image_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise LineageAuditError(f"Split CSV missing columns: {sorted(missing)}")

    records: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        identity = canonical_sample_path(str(row.image_path))
        label = int(row.label)
        if identity in records:
            raise LineageAuditError(f"Duplicate canonical sample: {identity}")
        records[identity] = label
    return records


def run_lineage_audit(
    config: dict[str, Any],
    *,
    child_train_csv: str,
    child_val_csv: str,
    checkpoint_path: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run a fail-closed split lineage audit.

    All critical mismatches raise ``LineageAuditError``.
    The audit artifact is always written before raising.
    """
    lineage = config.get("lineage", {})
    if not lineage.get("enabled", False):
        raise LineageAuditError("lineage.enabled must be true")

    parent_train_path = Path(lineage["parent_train_csv"])
    parent_val_path = Path(lineage["parent_val_csv"])
    child_train_path = Path(child_train_csv)
    child_val_path = Path(child_val_csv)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise LineageAuditError(f"Parent checkpoint not found: {checkpoint}")

    parent_train = _load_split(parent_train_path)
    parent_val = _load_split(parent_val_path)
    child_train = _load_split(child_train_path)
    child_val = _load_split(child_val_path)

    parent_train_keys = set(parent_train)
    parent_val_keys = set(parent_val)
    child_train_keys = set(child_train)
    child_val_keys = set(child_val)
    allow_parent_val_in_child_train = bool(
        lineage.get("allow_parent_val_in_child_train", False)
    )
    allow_parent_train_in_child_val = bool(
        lineage.get("allow_parent_train_in_child_val", False)
    )
    exact_parent_fullfit = (
        child_train_keys == parent_train_keys | parent_val_keys
        and child_val_keys == parent_val_keys
    )
    exact_overlapping_parent_replay = (
        child_train_keys == parent_train_keys
        and child_val_keys == parent_val_keys
        and parent_val_keys <= parent_train_keys
    )

    label_mismatches = sorted(
        key
        for key in (set(parent_train) | set(parent_val))
        & (set(child_train) | set(child_val))
        if (
            parent_train.get(key, parent_val.get(key))
            != child_train.get(key, child_val.get(key))
        )
    )

    audit = {
        "protocol_valid": True,
        "parent_experiment_id": lineage["parent_experiment_id"],
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": sha256_file(checkpoint),
        "parent_train_csv": str(parent_train_path),
        "parent_train_csv_sha256": sha256_file(parent_train_path),
        "parent_val_csv": str(parent_val_path),
        "parent_val_csv_sha256": sha256_file(parent_val_path),
        "child_train_csv": str(child_train_path),
        "child_train_csv_sha256": sha256_file(child_train_path),
        "child_val_csv": str(child_val_path),
        "child_val_csv_sha256": sha256_file(child_val_path),
        "parent_train_count": len(parent_train),
        "parent_val_count": len(parent_val),
        "child_train_count": len(child_train),
        "child_val_count": len(child_val),
        "child_val_in_parent_train": len(child_val_keys & parent_train_keys),
        "child_train_in_parent_val": len(child_train_keys & parent_val_keys),
        "child_train_equals_parent_train": child_train_keys == parent_train_keys,
        "child_val_equals_parent_val": child_val_keys == parent_val_keys,
        "allow_parent_val_in_child_train": allow_parent_val_in_child_train,
        "exact_parent_fullfit": exact_parent_fullfit,
        "allow_parent_train_in_child_val": allow_parent_train_in_child_val,
        "exact_overlapping_parent_replay": exact_overlapping_parent_replay,
        "label_mismatches": len(label_mismatches),
        "label_mismatch_examples": label_mismatches[:10],
    }

    errors: list[str] = []
    if allow_parent_train_in_child_val and not exact_overlapping_parent_replay:
        errors.append(
            "overlapping parent replay requires unchanged child train and val splits, "
            "with the parent val split fully contained in parent train"
        )
    if audit["child_val_in_parent_train"] and not allow_parent_train_in_child_val:
        errors.append(
            f"{audit['child_val_in_parent_train']} child validation samples "
            f"were seen by parent training"
        )
    if allow_parent_val_in_child_train and not exact_parent_fullfit:
        errors.append(
            "full-fit child train must exactly equal parent train union parent val, "
            "and child diagnostic val must equal parent val"
        )
    if audit["child_train_in_parent_val"] and not allow_parent_val_in_child_train:
        errors.append(
            f"{audit['child_train_in_parent_val']} child training samples "
            f"overlap parent validation"
        )
    if (
        not allow_parent_val_in_child_train
        and lineage.get("require_same_train", True)
        and not audit["child_train_equals_parent_train"]
    ):
        errors.append("child train split differs from parent train split")
    if lineage.get("require_same_val", True) and not audit["child_val_equals_parent_val"]:
        errors.append("child val split differs from parent val split")
    if label_mismatches:
        errors.append(
            f"{len(label_mismatches)} labels differ for shared canonical samples"
        )

    audit["errors"] = errors
    audit["protocol_valid"] = not errors
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if errors:
        raise LineageAuditError("Lineage audit failed: " + "; ".join(errors))
    return audit


def audit_declared_scope_graph(
    nodes: list[dict[str, Any]],
    *,
    target_id: str,
    stage: str,
    dataset_id: str,
    class_mapping_sha256: str,
    evaluation_groups: set[str],
    official_test_groups: set[str],
    require_independent: bool = True,
) -> dict[str, Any]:
    """Audit transitive *declared* group exposure without authorizing execution.

    Content-group IDs must come from the caller's bound manifests. Empty exposure
    lists mean explicitly none; missing lists mean unknown. Encoding by a frozen
    model is not fitting, but its ancestors' fitting/selection still propagates.
    This checks declaration consistency, not the truth of producer provenance.
    It cannot unlock the formal reproduction runner by itself.
    """
    if not stage or not dataset_id or not class_mapping_sha256:
        raise LineageAuditError('stage, dataset and class mapping binding required')
    lookup: dict[str, dict[str, Any]] = {}
    for node in nodes:
        identity = node.get('artifact_id')
        if not isinstance(identity, str) or not identity or identity in lookup:
            raise LineageAuditError('unique nonempty artifact_id required')
        lookup[identity] = node
    errors: list[str] = []
    unknown: list[str] = []
    visited: set[str] = set()
    active: set[str] = set()
    learned: set[str] = set()
    selected: set[str] = set()
    encoded: set[str] = set()
    exposures: list[dict[str, Any]] = []

    def visit(identity: str, path: list[str]) -> None:
        if identity in active:
            errors.append(f'cycle: {" -> ".join(path + [identity])}')
            return
        if identity in visited:
            return
        if identity not in lookup:
            unknown.append(f'missing ancestor: {identity}')
            return
        node = lookup[identity]
        active.add(identity)
        chain = path + [identity]
        for key, expected in [('stage', stage), ('dataset_id', dataset_id),
                              ('class_mapping_sha256', class_mapping_sha256)]:
            if node.get(key) != expected:
                errors.append(f'{identity}: {key} mismatch or missing')
        if node.get('scope') not in {'development_fit', 'calibration_fit',
                                      'development_evaluation', 'final_fit',
                                      'overlap_diagnostic', 'official_test', 'synthetic_dryrun'}:
            errors.append(f'{identity}: unknown artifact scope')
        if node.get('scope') == 'synthetic_dryrun':
            errors.append(f'{identity}: synthetic asset in official scope graph')
        if node.get('scope') == 'final_fit' and require_independent:
            errors.append(f'{identity}: final_fit cannot certify development independence')
        if not node.get('producer_record'):
            unknown.append(f'{identity}: missing producer record')
        local: dict[str, set[str]] = {}
        for key, union in [('learned_from_groups', learned),
                           ('selected_using_groups', selected),
                           ('encoded_groups', encoded)]:
            values = node.get(key)
            if not isinstance(values, list) or not all(isinstance(x, str) and x for x in values):
                unknown.append(f'{identity}: {key} unknown or invalid')
                local[key] = set()
            else:
                local[key] = set(values)
                union.update(values)
        fit = local['learned_from_groups'] | local['selected_using_groups']
        test_overlap = fit & official_test_groups
        evaluation_overlap = fit & evaluation_groups
        if test_overlap:
            errors.append(f'{identity}: official test influenced fitting/selection ({len(test_overlap)} groups)')
        if evaluation_overlap and require_independent:
            errors.append(f'{identity}: evaluation influenced fitting/selection ({len(evaluation_overlap)} groups)')
        exposures.append({'artifact_id': identity, 'dependency_path': chain,
                          'evaluation_learned_groups': len(local['learned_from_groups'] & evaluation_groups),
                          'evaluation_selected_groups': len(local['selected_using_groups'] & evaluation_groups),
                          'evaluation_encoded_groups': len(local['encoded_groups'] & evaluation_groups)})
        parents = node.get('parent_artifact_ids')
        if not isinstance(parents, list) or not all(isinstance(x, str) and x for x in parents):
            unknown.append(f'{identity}: parents unknown or invalid')
        else:
            for parent in parents:
                visit(parent, chain)
        active.remove(identity)
        visited.add(identity)

    visit(target_id, [])
    return {
        'schema_version': 1,
        'audit_kind': 'declared_transitive_scope_consistency',
        'target_id': target_id,
        'status': 'blocked' if errors or unknown else 'checks_passed',
        'source_authenticity_verified': False,
        'authorizes_formal_execution': False,
        'evaluation_role': 'development_evaluation' if require_independent else 'overlap_diagnostic',
        'independent_under_declared_graph': not errors and not unknown and not ((learned | selected) & evaluation_groups),
        'visited_artifacts': sorted(visited),
        'evaluation_group_count': len(evaluation_groups),
        'evaluation_learned_groups': len(learned & evaluation_groups),
        'evaluation_selected_groups': len(selected & evaluation_groups),
        'evaluation_encoded_groups': len(encoded & evaluation_groups),
        'exposures': exposures,
        'errors': errors,
        'unknown': unknown,
    }
