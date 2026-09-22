"""One bounded NPU LP -> FT migration validation, with strict dataset parity.

Run from the repository root after sourcing CANN and selecting the allocated NPU.
This rebuilds current-stage assets and does not submit anything to the platform.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def audit_ft():
    import torch
    base = ROOT / "outputs/rematch750_npu"
    with (ROOT / "artifacts/stages/repechage/20260921_npu/train_dev.csv").open() as handle:
        train_count = sum(1 for _ in csv.DictReader(handle))
    parent = torch.load(base / "RM_LP/seed42/checkpoints/best.pt", map_location="cpu", weights_only=False)
    records = {}
    for name in ("best", "last"):
        checkpoint = torch.load(base / f"RM_FT_NPU/seed42/checkpoints/{name}.pt",
                                map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        steps = sorted({int(value["step"]) for value in checkpoint["optimizer_state_dict"]["state"].values()
                        if "step" in value})
        batch_size = checkpoint["config"]["train"]["batch_size"]
        expected = checkpoint["epoch"] * ((train_count + batch_size - 1) // batch_size)
        frozen = ("visual.conv1.weight", "visual.positional_embedding")
        record = dict(epoch=checkpoint["epoch"], global_step=checkpoint["global_step"],
                      scheduler_steps=checkpoint["scheduler_state_dict"]["last_epoch"],
                      optimizer_step_values=steps,
                      all_tensors_finite=all(bool(torch.isfinite(v).all()) for v in state.values()),
                      frozen_unchanged=all(torch.equal(state[k], parent["model_state_dict"][k]) for k in frozen),
                      npu_rng_saved="npu" in checkpoint["rng_state"],
                      raw_macro=checkpoint["metrics"]["raw_macro"],
                      raw_micro=checkpoint["metrics"]["raw_micro"])
        assert record["global_step"] == record["scheduler_steps"] == expected
        assert steps == [expected]
        assert record["all_tensors_finite"] and record["frozen_unchanged"] and record["npu_rng_saved"]
        records[name] = record
    assert records["last"]["epoch"] == 8
    records["best"]["macro_delta_vs_gpu_pp"] = (records["best"]["raw_macro"] - 0.718362) * 100
    if records["best"]["macro_delta_vs_gpu_pp"] <= -2:
        raise RuntimeError(f"NPU baseline regressed by >=2pp: {records}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    commands = [
        ("prepare", "lp"), ("verify", "lp"), ("cache", "lp"),
        ("train", "lp"), ("train", "ft"), ("infer", "ft"),
    ]
    if not args.execute:
        print(json.dumps(commands))
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT / "reproducibility/aegis_f1"))
    logs = ROOT / "outputs/rematch750_npu/migration"
    logs.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "results/rematch750_npu_migration_execution.json"
    report_path.parent.mkdir(exist_ok=True)
    if report_path.exists():
        raise FileExistsError("Migration report exists; inspect it before rerunning any stage")
    report = dict(status="running", stages=[], platform_submitted=False)

    def save():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)

    save()
    try:
        for action, candidate in commands:
            command = [sys.executable, "-u", "-m", "aegis_clip.cli.rematch", action,
                       "--config", f"configs/rematch750_{candidate}_npu.yaml"]
            entry = dict(action=action, candidate=candidate, command=command, status="running")
            report["stages"].append(entry)
            save()
            print(f"START {action} {candidate}", flush=True)
            start = time.monotonic()
            with (logs / f"{action}_{candidate}.log").open("w") as handle:
                completed = subprocess.run(command, cwd=ROOT, env=env,
                                           stdout=handle, stderr=subprocess.STDOUT)
            entry.update(seconds=time.monotonic() - start, returncode=completed.returncode,
                         status="passed" if completed.returncode == 0 else "failed")
            save()
            if completed.returncode:
                raise RuntimeError(f"Stage failed: {action} {candidate}; see {logs}")
            if action == "prepare":
                original = json.loads((ROOT / "results/rematch750_20260921_dataset_manifest.json").read_text())
                rebuilt = json.loads((ROOT / "artifacts/stages/repechage/20260921_npu/dataset_manifest.json").read_text())
                compared = sorted(set(original) - {"train_root", "test_root"})
                mismatches = [key for key in compared if original[key] != rebuilt.get(key)]
                report["dataset_parity"] = dict(compared_keys=compared, mismatches=mismatches)
                save()
                if mismatches:
                    raise RuntimeError(f"NPU dataset differs from the frozen GPU split: {mismatches}")
            if action == "train" and candidate == "ft":
                report["checkpoint_audit"] = audit_ft()
                save()
            print(f"PASS {action} {candidate}: {entry['seconds']:.1f}s", flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
