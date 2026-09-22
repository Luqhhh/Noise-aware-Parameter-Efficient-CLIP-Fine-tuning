"""Two-epoch acceptance of the recipe-preserving NPU throughput configuration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/rematch750_ft_npu_tuned_acceptance.yaml"
RUN = ROOT / "outputs/npu_tuning/RM_FT_NPU_TUNED_E2/seed42"


def audit():
    import torch
    from aegis_clip.config import load_config
    from aegis_clip.rematch_assets import validate_checkpoint
    config = load_config(ROOT / CONFIG)
    with open(config["data"]["train_csv"]) as f:
        count = sum(1 for _ in csv.DictReader(f))
    expected = 2 * ((count + config["train"]["batch_size"] - 1) // config["train"]["batch_size"])
    parent = torch.load(config["train"]["init_checkpoint"], map_location="cpu", weights_only=False)
    records = {}
    for name in ("best", "last"):
        path = RUN / f"checkpoints/{name}.pt"
        validate_checkpoint(path, config)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        steps = sorted({int(v["step"]) for v in checkpoint["optimizer_state_dict"]["state"].values() if "step" in v})
        record = dict(epoch=checkpoint["epoch"], global_step=checkpoint["global_step"],
                      scheduler_steps=checkpoint["scheduler_state_dict"]["last_epoch"],
                      optimizer_step_values=steps,
                      all_tensors_finite=all(bool(torch.isfinite(v).all()) for v in state.values()),
                      frozen_unchanged=all(torch.equal(state[k], parent["model_state_dict"][k])
                                           for k in ("visual.conv1.weight", "visual.positional_embedding")),
                      npu_rng_saved="npu" in checkpoint["rng_state"],
                      raw_macro=checkpoint["metrics"]["raw_macro"], raw_micro=checkpoint["metrics"]["raw_micro"])
        assert record["epoch"] == 2
        assert record["global_step"] == record["scheduler_steps"] == expected
        assert steps == [expected]
        assert record["all_tensors_finite"] and record["frozen_unchanged"] and record["npu_rng_saved"]
        records[name] = record
    reloaded = json.loads((RUN / "checkpoints/best_evaluation.json").read_text())
    assert all(reloaded[k] == records["best"][k] for k in ("raw_macro", "raw_micro"))
    records["reload_metrics_exact"] = True
    baseline = json.loads((ROOT / "results/rematch750_npu_migration_execution.json").read_text())
    reference = baseline["checkpoint_audit"]["best"]
    records["reference_npu_epoch2_macro"] = reference["raw_macro"]
    records["macro_delta_vs_original_npu_pp"] = 100 * (records["best"]["raw_macro"] - reference["raw_macro"])
    assert records["macro_delta_vs_original_npu_pp"] > -2
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    if not args.execute:
        print(f"train then infer with {CONFIG}; audit 8364 updates and exact reload metrics")
        return
    os.chdir(ROOT)
    if args.wait_pid:
        process = Path(f"/proc/{args.wait_pid}/cmdline")
        while process.exists() and b"outputs/npu_tuning/phase5.py" in process.read_bytes():
            time.sleep(2)
    for name in ("confirm32_pinned", "confirm32_gce"):
        evidence = json.loads((ROOT / f"results/npu_tuning/{name}.json").read_text())
        assert evidence["status"] == "passed", name
    path = ROOT / "results/rematch750_npu_tuned_acceptance.json"
    if path.exists():
        raise FileExistsError(path)
    report = dict(status="running", candidate="RM_FT_NPU_TUNED_E2", stages=[],
                  platform_submitted=False, completed_epochs=0, schedule_epochs=8,
                  source_manifest="results/rematch750_npu_tuning_source_manifest.json",
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(path)

    save()
    try:
        for action in ("train", "infer"):
            command = [sys.executable, "-u", "-m", "aegis_clip.cli.rematch", action, "--config", CONFIG]
            entry = dict(action=action, command=command, status="running")
            report["stages"].append(entry)
            save()
            print("START", action, flush=True)
            start = time.monotonic()
            with (ROOT / f"outputs/npu_tuning/acceptance_{action}.log").open("w") as f:
                result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
            entry.update(seconds=time.monotonic()-start, returncode=result.returncode,
                         status="passed" if result.returncode == 0 else "failed")
            save()
            result.check_returncode()
            if action == "train":
                report["checkpoint_audit"] = audit()
                report["completed_epochs"] = 2
                save()
            print("PASS", action, flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
