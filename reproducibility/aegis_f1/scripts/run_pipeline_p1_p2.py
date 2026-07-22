#!/usr/bin/env python3
"""Post-P1 pipeline: checkpoint averaging → CR-0 → CR-1 → CR-2.

Run from reproducibility/aegis_f1 directory.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
P1_CKPT_DIR = REPO / "outputs" / "P1_A2_STRICT_EPOCH_CKPTS" / "seed42" / "checkpoints"
AVG_OUTDIR = REPO / "outputs" / "phase4" / "p1_averaging"
LOG_DIR = REPO / "outputs" / "phase4" / "logs"
CONFIG = "configs/p1_a2_strict_epochs.yaml"


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "pipeline.log", "a") as f:
        f.write(line + "\n")


def run(cmd: list[str], logfile: str | None = None) -> int:
    log(f"RUN: {' '.join(cmd)}")
    if logfile:
        path = LOG_DIR / logfile
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    else:
        proc = subprocess.run(cmd, cwd=REPO)
    return proc.returncode


def check_ckpt(epoch: int) -> bool:
    return (P1_CKPT_DIR / f"epoch_{epoch}.pt").exists()


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    AVG_OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Wait for P1 ─────────────────────────────────────────
    log("Waiting for P1 training to complete...")
    while not check_ckpt(6):
        completed = sum(1 for ep in range(1, 7) if check_ckpt(ep))
        log(f"  {completed}/6 epoch checkpoints found, sleeping 60s...")
        time.sleep(60)

    log("P1 training complete! Verifying all 6 checkpoints...")
    for ep in range(1, 7):
        if not check_ckpt(ep):
            log(f"FATAL: epoch_{ep}.pt missing!")
            sys.exit(1)
    log("All 6 epoch checkpoints verified ✓")

    # ── Step 1: Checkpoint Averaging ────────────────────────────────
    log("=" * 50)
    log("P1: Checkpoint Averaging")

    swa_configs = [
        ("swa1_epoch2_6", "equal", [2, 3, 4, 5, 6]),
        ("swa2_epoch2_4", "equal", [2, 3, 4]),
        ("swa3_epoch3_6", "equal", [3, 4, 5, 6]),
        ("swa4_greedy_soup", "greedy_soup", [2, 3, 4, 5, 6]),
    ]

    for name, scheme, epochs in swa_configs:
        log(f"Running {name} ({scheme}, epochs {epochs})...")
        ckpt_paths = [str(P1_CKPT_DIR / f"epoch_{ep}.pt") for ep in epochs]
        out_path = str(AVG_OUTDIR / f"{name}.pt")
        rc = run(
            [
                sys.executable, "-m", "aegis_clip.cli.average_checkpoints",
                "--config", CONFIG,
                "--checkpoints", *ckpt_paths,
                "--scheme", scheme,
                "--output", out_path,
                "--eval",
                "--selection-metric", "clean_core_micro",
            ],
            logfile=f"{name}.log",
        )
        if rc != 0:
            log(f"WARNING: {name} exited with code {rc}")
        else:
            log(f"{name} complete → {out_path}")

    log("P1 averaging complete")

    # ── Step 2-4: CR experiments ────────────────────────────────────
    cr_configs = [
        ("cr0_baseline", "CR-0 Baseline (no routing)"),
        ("cr1_hard_gate", "CR-1 Hard Gate (clean≥0.70)"),
        ("cr2_soft_gate", "CR-2 Soft Gate"),
    ]

    for config_name, desc in cr_configs:
        log("=" * 50)
        log(f"P2: {desc}")
        rc = run(
            [sys.executable, "-m", "aegis_clip.cli.train",
             "--config", f"configs/{config_name}.yaml"],
            logfile=f"{config_name}.log",
        )
        if rc != 0:
            log(f"ERROR: {config_name} failed with code {rc}")
            # Continue to next experiment even if one fails
        else:
            log(f"{config_name} complete")

    log("=" * 50)
    log("ALL PIPELINE STEPS COMPLETE")


if __name__ == "__main__":
    main()
