"""Snapshot atomically replaced best checkpoints during a long training run."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _snapshot(source: Path, destination: Path) -> tuple[int, str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="best.", suffix=".tmp", dir=destination
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        payload = torch.load(temporary, map_location="cpu", weights_only=False)
        epoch = int(payload["epoch"])
        digest = _sha256(temporary)
        output = destination / f"best_epoch_{epoch:03d}.pt"
        os.replace(temporary, output)
        return epoch, digest, output
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0.0:
        raise ValueError("interval must be positive")

    last_identity: tuple[int, int] | None = None
    while True:
        if args.source.exists():
            stat = args.source.stat()
            identity = (stat.st_mtime_ns, stat.st_size)
            if identity != last_identity:
                epoch, digest, output = _snapshot(args.source, args.destination)
                print(
                    f"snapshotted epoch={epoch} sha256={digest} path={output}",
                    flush=True,
                )
                last_identity = identity
        if args.once or not _process_exists(args.pid):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
