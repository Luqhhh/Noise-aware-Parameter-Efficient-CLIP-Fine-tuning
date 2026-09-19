#!/usr/bin/env python3
"""Complete v8 delivery with the audited repr-address compatibility entry point."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "outputs/prelim75_v8_20260919/execution_source"
sys.path.insert(0, str(SNAPSHOT))

from aegis_clip.prelim75_source_bias import deliver_v8, load_v8_plan  # noqa: E402
from aegis_clip.runtime import atomic_json_dump, sha256_file  # noqa: E402
import aegis_clip.prelim75_source_bias as delivery_module  # noqa: E402


_original_run = subprocess.run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", required=True, choices=["B0", "B1"])
    args = parser.parse_args()
    plan = load_v8_plan(args.config)
    candidate_dir = Path(plan["output"]) / args.candidate
    receipt = candidate_dir / "protocol_repair_receipt.json"
    if receipt.exists():
        raise FileExistsError(f"Existing repair receipt is never overwritten: {receipt}")

    def run(command, *positional, **kwargs):
        command = list(command)
        if command[1:4] == ["-u", "-m", "aegis_clip.cli.infer"]:
            command = [
                command[0],
                "-u",
                str(ROOT / "scripts/infer_prelim75_v8_protocol_repair.py"),
                *command[4:],
            ]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(SNAPSHOT)
            environment["PRELIM75_V8_PROTOCOL_REPAIR_RECEIPT"] = str(receipt)
            kwargs["env"] = environment
        return _original_run(command, *positional, **kwargs)

    delivery_module.subprocess.run = run
    report = deliver_v8(plan, args.candidate)
    if not receipt.is_file():
        raise RuntimeError("Audited protocol repair did not produce its receipt")
    report["protocol_repair"] = {
        "reason": "process_local_function_address_in_preprocess_repr",
        "receipt": str(receipt),
        "receipt_sha256": sha256_file(receipt),
        "failed_delivery_before_test_iteration": True,
        "model_or_bias_changed": False,
    }
    atomic_json_dump(report, candidate_dir / "candidate_report.json")
    atomic_json_dump(report, candidate_dir / "status.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
