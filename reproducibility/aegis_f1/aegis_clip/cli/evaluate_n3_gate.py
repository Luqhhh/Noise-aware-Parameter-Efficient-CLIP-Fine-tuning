"""Persist the strict N3 training and fixed-M3 inference decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_clip.n3_gate import evaluate_n3_gate
from aegis_clip.runtime import atomic_json_dump, sha256_file


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--j0-control", required=True)
    parser.add_argument("--complementary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "initial": args.initial,
            "candidate": args.candidate,
            "j0_control": args.j0_control,
            "complementary": args.complementary,
        }.items()
    }
    report = evaluate_n3_gate(
        _load(paths["initial"]),
        _load(paths["candidate"]),
        _load(paths["j0_control"]),
        _load(paths["complementary"]),
    )
    report["lineage"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    destination = Path(args.output).resolve()
    atomic_json_dump(report, destination)
    print(destination)


if __name__ == "__main__":
    main()
