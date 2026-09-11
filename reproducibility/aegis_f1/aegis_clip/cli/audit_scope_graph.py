"""Read-only transitive declaration audit; does not execute recipe nodes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from aegis_clip.lineage import audit_declared_scope_graph, LineageAuditError
from aegis_clip.runtime import sha256_file


def load_bound_groups(record: dict, base: Path) -> set[str]:
    path = Path(record['path'])
    if not path.is_absolute():
        path = base / path
    if not record.get('sha256') or sha256_file(path) != record['sha256']:
        raise LineageAuditError('group-set file hash mismatch')
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not all(isinstance(x, str) and x for x in values):
        raise LineageAuditError('group-set file must contain a list of group IDs')
    if len(values) != len(set(values)):
        raise LineageAuditError('group-set file contains duplicate group IDs')
    return set(values)


def audit_manifest(path: str | Path) -> dict:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1:
        raise LineageAuditError('scope manifest schema_version must be 1')
    evaluation = load_bound_groups(manifest['evaluation_groups'], path.parent)
    test = load_bound_groups(manifest['official_test_groups'], path.parent)
    nodes = []
    for source in manifest['nodes']:
        node = dict(source)
        for field in ('learned_from_groups', 'selected_using_groups', 'encoded_groups'):
            # Unknown exposure stays unknown, never silently becomes an empty set.
            if isinstance(node.get(field), dict):
                node[field] = sorted(load_bound_groups(node[field], path.parent))
        nodes.append(node)
    report = audit_declared_scope_graph(
        nodes, target_id=manifest['target_id'], stage=manifest['stage'],
        dataset_id=manifest['dataset_id'], class_mapping_sha256=manifest['class_mapping_sha256'],
        evaluation_groups=evaluation, official_test_groups=test,
        require_independent=manifest['evaluation_role'] == 'development_evaluation',
    )
    if manifest['evaluation_role'] not in {'development_evaluation', 'overlap_diagnostic'}:
        raise LineageAuditError('explicit evaluation role required')
    report['manifest_sha256'] = sha256_file(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    report = audit_manifest(args.manifest)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'checks_passed' else 2)


if __name__ == '__main__':
    main()
