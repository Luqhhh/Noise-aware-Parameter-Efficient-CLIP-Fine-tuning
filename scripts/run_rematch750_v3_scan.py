"""Run the bounded V3 candidates sequentially on one already selected NPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/codex/rematch750_v3_tradeoff"
RESULTS = ROOT / "results/rematch750_v3"
STATUS = RESULTS / "scan_status.json"
CONFIGS = {
    "RM_V3_B64": "rematch750_v3_b64.npu0.local.yaml",
    "RM_V3_B128_E16": "rematch750_v3_b128_e16.npu0.local.yaml",
    "RM_V3_B1024_E16_LR4": "rematch750_v3_b1024_e16_lr4.npu0.local.yaml",
    "RM_V3_B64_LR2": "rematch750_v3_b64_lr2.npu0.local.yaml",
}
BASELINE_MACRO = 0.7190471887588501
MIN_FREE_BYTES = 50 * 2**30


def write_status(**fields: object) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = {"updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(STATUS)


def wait_for_pid(pid: int) -> None:
    while True:
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            return
        write_status(stage="waiting_for_b64", pid=pid)
        time.sleep(30)


def wait_for_idle_npu(candidate: str) -> None:
    probe = (
        "import torch,torch_npu; "
        "free,total=torch.npu.mem_get_info(0); print(free,total,flush=True)"
    )
    while True:
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=40)
        if result.returncode:
            raise RuntimeError(f"NPU memory probe failed: {result.stderr}")
        free, total = map(int, result.stdout.split())
        if free >= MIN_FREE_BYTES:
            write_status(stage="npu_idle", candidate=candidate, free_bytes=free, total_bytes=total)
            return
        write_status(stage="waiting_for_idle_npu", candidate=candidate, free_bytes=free, total_bytes=total)
        time.sleep(60)


def audit(candidate: str) -> dict:
    config = ROOT / "configs" / CONFIGS[candidate]
    output = RESULTS / f"{candidate}.json"
    log = RESULTS / f"{candidate}_audit.log"
    write_status(stage="auditing", candidate=candidate)
    with log.open("w") as f:
        subprocess.run(
            [sys.executable, "scripts/audit_rematch750_v2.py", "--config", str(config), "--output", str(output)],
            cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True,
        )
    report = json.loads(output.read_text())
    if report["status"] != "passed":
        raise RuntimeError(f"Audit did not pass for {candidate}")
    write_status(stage="audited", candidate=candidate, raw_macro=report["raw_macro"],
                 raw_micro=report["raw_micro"], train_cli_seconds=report["train_cli_seconds"])
    return report


def train(candidate: str) -> dict:
    config = ROOT / "configs" / CONFIGS[candidate]
    run = OUT / candidate / "seed42"
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run}")
    wait_for_idle_npu(candidate)
    write_status(stage="training", candidate=candidate)
    with (OUT / f"{candidate}_launch.log").open("w") as f:
        subprocess.run(
            [sys.executable, "-u", "-m", "aegis_clip.cli.rematch", "train", "--config", str(config)],
            cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True,
        )
    return audit(candidate)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait-for-b64-pid", type=int, required=True)
    args = p.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0":
        raise ValueError("This bound scan requires ASCEND_RT_VISIBLE_DEVICES=0")
    if not (ROOT / "configs" / CONFIGS["RM_V3_B64"]).is_file():
        raise FileNotFoundError("B64 bound configuration is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        wait_for_pid(args.wait_for_b64_pid)
        b64 = audit("RM_V3_B64")
        b128 = train("RM_V3_B128_E16")
        train("RM_V3_B1024_E16_LR4")
        if b64["raw_macro"] < BASELINE_MACRO - 0.001 and b128["raw_macro"] < BASELINE_MACRO - 0.001:
            train("RM_V3_B64_LR2")
        write_status(stage="all_runs_audited")
    except Exception as exc:
        write_status(stage="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
