"""Persist N3 representation-shift diagnostics from aligned logit caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aegis_clip.representation_diagnostic import diagnose_representation_shift
from aegis_clip.runtime import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("a2-center", "a2-m1", "a2-m3", "n3-center", "n3-m3", "train-cache"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "a2_center": Path(args.a2_center).resolve(),
        "a2_m1": Path(args.a2_m1).resolve(),
        "a2_m3": Path(args.a2_m3).resolve(),
        "n3_center": Path(args.n3_center).resolve(),
        "n3_m3": Path(args.n3_m3).resolve(),
        "train_cache": Path(args.train_cache).resolve(),
    }
    caches = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    report = diagnose_representation_shift(**caches)
    report["lineage"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    destination = Path(args.output).resolve()
    atomic_json_dump(report, destination)
    print(destination)


if __name__ == "__main__":
    main()
