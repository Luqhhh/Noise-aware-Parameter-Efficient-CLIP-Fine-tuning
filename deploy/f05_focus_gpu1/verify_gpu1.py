#!/usr/bin/env python3
"""Fail-closed deployment check for the GPU1 focus worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

REQUIRED_STAGE_FILES = (
    "class_to_idx.json",
    "dataset_manifest.json",
    "train_dev.csv",
    "val_dev.csv",
    "features/features.pt",
    "features/image_paths.json",
    "features/manifest.json",
)


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _check(ok: bool, detail: str) -> dict[str, Any]:
    return {"ok": bool(ok), "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir")
    parser.add_argument("--train-root")
    parser.add_argument("--test-root")
    parser.add_argument("--rm-lp-checkpoint")
    parser.add_argument("--f05-checkpoint")
    parser.add_argument("--asset-root")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {"checks": {}}
    checks = report["checks"]
    checks["python"] = _check(
        True,
        f"{os.sys.executable}",
    )

    try:
        import torch

        checks["torch"] = _check(True, torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        checks["cuda"] = _check(
            cuda_available or not args.require_cuda,
            f"available={cuda_available}, count={torch.cuda.device_count()}",
        )
    except Exception as exc:  # pragma: no cover - deployment host dependent
        checks["torch"] = _check(False, repr(exc))

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
        checks["branch"] = _check(
            branch == "focus/f05-four-lines",
            branch,
        )
    except Exception as exc:
        checks["branch"] = _check(False, repr(exc))

    if args.stage_dir:
        stage = _path(args.stage_dir)
        missing = [name for name in REQUIRED_STAGE_FILES if not (stage / name).is_file()]
        checks["stage_dir"] = _check(not missing, f"{stage}; missing={missing}")
        if stage.is_dir():
            checks["stage_manifest"] = _check(
                (stage / "dataset_manifest.json").is_file(),
                str(stage / "dataset_manifest.json"),
            )
    else:
        checks["stage_dir"] = _check(False, "--stage-dir not provided")

    for name, value in (
        ("train_root", args.train_root),
        ("test_root", args.test_root),
    ):
        path = _path(value)
        checks[name] = _check(
            bool(path and path.is_dir() and any(path.iterdir())),
            str(path) if path else f"--{name.replace('_', '-')} not provided",
        )

    rm_lp = _path(args.rm_lp_checkpoint)
    checks["rm_lp_checkpoint"] = _check(
        bool(rm_lp and rm_lp.is_file()),
        str(rm_lp) if rm_lp else "--rm-lp-checkpoint not provided",
    )

    if args.f05_checkpoint:
        f05 = _path(args.f05_checkpoint)
        checks["f05_checkpoint"] = _check(
            bool(f05 and f05.is_file()),
            str(f05),
        )
    else:
        checks["f05_checkpoint"] = {
            "ok": True,
            "detail": "not provided; required only for A0/D1/D2",
        }

    asset_root = Path(args.asset_root).expanduser().resolve() if args.asset_root else Path(__file__).resolve().parents[2]
    checksum_path = Path(__file__).resolve().parent / "data_checksums.json"
    if checksum_path.is_file():
        checksum_payload = json.loads(checksum_path.read_text(encoding="utf-8"))
        expected = checksum_payload.get("sha256", {})
        bad: list[str] = []
        for relative, digest in expected.items():
            candidate = asset_root / relative
            if not candidate.is_file():
                bad.append(f"{relative}:missing")
                continue
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if actual != digest:
                bad.append(f"{relative}:sha256")
        checks["canonical_asset_checksums"] = _check(
            not bad,
            f"root={asset_root}; failures={bad[:5]}",
        )

    passed = all(bool(item["ok"]) for item in checks.values())
    report["passed"] = passed
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
