"""Freeze the SCOPE-K2 5x3 grouped nested OOF artifact exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_clip.runtime import sha256_file
from aegis_clip.scope_cache import load_scope_cache, validate_parent_cache
from aegis_clip.scope_crossfit import freeze_fold_artifact
from aegis_clip.scope_protocol import load_scope_protocol, verify_scope_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-cache", type=Path, required=True)
    parser.add_argument("--group-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_scope_protocol(args.config)
    verify_scope_assets(protocol)
    group_path = args.group_artifact.expanduser().resolve()
    if group_path != protocol.assets.group_artifact or sha256_file(group_path) != protocol.assets.group_artifact_sha256:
        raise ValueError("group artifact is not the frozen protocol asset")
    group_map = json.loads(group_path.read_text(encoding="utf-8"))
    if not isinstance(group_map, dict):
        raise ValueError("group artifact root must be a path-to-group mapping")
    expected = protocol.fixed["group_expected"]
    if len(group_map) != int(expected["expected_total_rows"]):
        raise ValueError("group artifact row count mismatch")
    if len(set(str(value) for value in group_map.values())) != int(expected["expected_unique_groups"]):
        raise ValueError("group artifact unique-group count mismatch")
    parent = load_scope_cache(args.parent_cache)
    validate_parent_cache(parent, protocol, "validation")
    missing = [path for path in parent["paths"] if path not in group_map]
    if missing:
        raise ValueError(f"validation path is missing from group artifact: {missing[0]}")
    groups = [str(group_map[path]) for path in parent["paths"]]
    artifact = freeze_fold_artifact(parent, groups, protocol, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": artifact["_cache_sha256"],
        "semantic_sha256": artifact["_semantic_sha256"],
        "rows": len(artifact["paths"]),
        "outer_folds": sorted(artifact["outer_fold_id"].unique().tolist()),
        "inner_folds_per_outer": {
            str(fold): sorted(ids[ids >= 0].unique().tolist())
            for fold, ids in artifact["inner_fold_id"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
