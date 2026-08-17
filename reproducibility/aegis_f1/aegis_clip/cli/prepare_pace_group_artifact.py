"""Freeze the one-time PACE-K2 exact-byte duplicate group artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from aegis_clip.pace_protocol import (
    PacePreflightError,
    build_exact_group_artifact,
    freeze_group_sha_in_config,
    load_pace_protocol,
)
from aegis_clip.runtime import atomic_json_dump


def prepare_pace_group_artifact(
    config_path: str | Path,
    *,
    hash_workers: int,
) -> dict[str, Any]:
    """Double-build, audit, and freeze the protocol group artifact."""
    protocol = load_pace_protocol(config_path, allow_unfrozen_group=True)
    first = build_exact_group_artifact(protocol, hash_workers=int(hash_workers))
    first_bytes = protocol.group_artifact.output_path.read_bytes()
    second = build_exact_group_artifact(protocol, hash_workers=int(hash_workers))
    second_bytes = protocol.group_artifact.output_path.read_bytes()
    if first.map_sha256 != second.map_sha256 or first_bytes != second_bytes:
        raise PacePreflightError(
            "independent exact-group builds produced different bytes or SHA-256"
        )

    ignore = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--",
            str(protocol.group_artifact.output_path),
        ],
        cwd=protocol.config_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if ignore.returncode == 0:
        raise PacePreflightError("PACE group artifact path is ignored by Git")
    if ignore.returncode != 1:
        detail = ignore.stderr.strip() or ignore.stdout.strip()
        raise PacePreflightError(
            f"git check-ignore failed with exit code {ignore.returncode}: {detail}"
        )

    freeze_group_sha_in_config(protocol.config_path, first.map_sha256)
    frozen = load_pace_protocol(protocol.config_path)
    report = json.loads(
        frozen.group_artifact.report_path.read_text(encoding="utf-8")
    )
    report.update(
        {
            "first_sha256": first.map_sha256,
            "second_sha256": second.map_sha256,
            "independent_builds_match": True,
            "git_check_ignore_exit_code": int(ignore.returncode),
            "group_artifact_state": frozen.group_artifact.state,
            "frozen_group_sha256": frozen.group_artifact.sha256,
        }
    )
    atomic_json_dump(report, frozen.group_artifact.report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--hash-workers", type=int, default=8)
    args = parser.parse_args()
    result = prepare_pace_group_artifact(
        args.config,
        hash_workers=args.hash_workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
