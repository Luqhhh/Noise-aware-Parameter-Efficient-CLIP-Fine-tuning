#!/usr/bin/env python3
"""Run-id isolated, non-destructive queue for REMATCH750_SEARCH_V5.

The V4 queue moved and deleted active run directories.  V5 uses a lease and a
separate immutable runtime directory per run, never touches another run's
directory, and records every transition in a JSONL ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))

CONFIG_DIR = ROOT / "configs/rematch750_search_v5"
RESULT_DIR = ROOT / "results/rematch750_search_v5"
QUEUE_DIR = ROOT / "outputs/rematch750_search_v5/queue"
LEDGER = RESULT_DIR / "ledger.jsonl"
MANIFEST = ROOT / "search_manifest.json"


def _now() -> int:
    return int(time.time())


def _append_ledger(row: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    row = {"at_unix": _now(), **row}
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _new_run_root(trial_id: str) -> Path:
    """Create an exclusive run directory without mutating any existing one."""
    parent = QUEUE_DIR / _slug(trial_id)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for suffix in range(1000):
        candidate = parent / f"{stamp}-pid{os.getpid()}-{suffix:03d}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not allocate an exclusive run directory for {trial_id}")


def _write_lease(run_root: Path, trial_id: str) -> None:
    lease = {
        "trial_id": trial_id,
        "pid": os.getpid(),
        "started_at_unix": _now(),
        "status": "running",
    }
    (run_root / "LEASE.json").write_text(
        json.dumps(lease, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_runtime_config(source: Path, run_root: Path, device: str) -> Path:
    """Resolve a source config into an absolute-path, run-root-bound copy."""
    from aegis_clip.config import load_config, public_config

    config = load_config(source)
    public = public_config(config)
    public["output"]["root"] = str(run_root)
    public["train"]["device"] = device
    public["project"]["trial_id"] = str(public["project"].get("trial_id", source.stem))
    runtime = run_root / "runtime_config.yaml"
    runtime.write_text(
        yaml.safe_dump(public, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return runtime


def _selected_report(run_root: Path) -> Path:
    return run_root / "checkpoints/selected_report.json"


def _best_checkpoint(run_root: Path) -> Path:
    return run_root / "checkpoints/best.pt"


def _execute_one(trial_id: str, source_config: Path, device: str) -> int:
    from aegis_clip.config import load_config
    from aegis_clip.rematch_search_v5 import MechanismBlockedError, validate_declaration

    declaration = validate_declaration(load_config(source_config))
    if declaration["status"] != "implemented":
        _append_ledger(
            {
                "trial_id": trial_id,
                "status": "blocked_implementation"
                if declaration["status"].startswith("blocked")
                else declaration["status"],
                "detail": declaration,
            }
        )
        print(f"SKIP {trial_id}: {declaration['status']}", flush=True)
        return 0

    # Re-run declaration with the executable gate before creating any run dir.
    try:
        validate_declaration(load_config(source_config), require_implemented=True)
    except MechanismBlockedError as exc:
        _append_ledger({"trial_id": trial_id, "status": "blocked_implementation", "detail": str(exc)})
        print(f"SKIP {trial_id}: {exc}", flush=True)
        return 0

    run_root = _new_run_root(trial_id)
    _write_lease(run_root, trial_id)
    runtime_config = _write_runtime_config(source_config, run_root, device)
    queue_log = run_root / "queue.log"
    command = [
        sys.executable,
        "-m",
        "aegis_clip.cli.rematch",
        "train",
        "--config",
        str(runtime_config),
    ]
    environment = dict(os.environ)
    environment.setdefault("PYTHONPATH", str(ROOT / "reproducibility/aegis_f1"))
    _append_ledger({"trial_id": trial_id, "status": "planned", "run_root": str(run_root)})
    started = time.time()
    with queue_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
    elapsed = time.time() - started

    if return_code != 0 or not _best_checkpoint(run_root).is_file():
        _append_ledger(
            {
                "trial_id": trial_id,
                "status": "failed_runtime",
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "run_root": str(run_root),
                "queue_log": str(queue_log),
            }
        )
        print(f"FAIL {trial_id}: rc={return_code}", flush=True)
        return 1

    if not _selected_report(run_root).is_file():
        _append_ledger(
            {
                "trial_id": trial_id,
                "status": "audit_incomplete",
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "run_root": str(run_root),
                "reason": "best.pt exists but selected_report.json is missing",
            }
        )
        print(f"AUDIT_INCOMPLETE {trial_id}", flush=True)
        return 1

    metrics = json.loads(_selected_report(run_root).read_text(encoding="utf-8"))
    _append_ledger(
        {
            "trial_id": trial_id,
            "status": "training_complete",
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "run_root": str(run_root),
            "selected_report": str(_selected_report(run_root)),
            "selected_epoch": metrics.get("selected_epoch"),
            "raw_macro": metrics.get("raw_macro"),
            "raw_micro": metrics.get("raw_micro"),
        }
    )
    print(f"DONE {trial_id}: epoch={metrics.get('selected_epoch')}", flush=True)
    return 0


def main() -> int:
    manifest = _load_manifest()
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", nargs="+", default=manifest.get("first_wave", []))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--lock", default=str(QUEUE_DIR / "queue.lock"))
    args = parser.parse_args()

    lock_path = Path(args.lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        stale = lock_path.read_text(encoding="utf-8").strip()
        raise SystemExit(f"queue lock exists: {stale}; refusing to guess whether it is live")
    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.close(fd)
    try:
        failures = 0
        for trial_id in args.trials:
            source = CONFIG_DIR / f"{trial_id}.yaml"
            if not source.is_file():
                _append_ledger({"trial_id": trial_id, "status": "failed_runtime", "reason": "config_missing"})
                failures += 1
                continue
            failures += int(_execute_one(trial_id, source, args.device) != 0)
        return 1 if failures else 0
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
