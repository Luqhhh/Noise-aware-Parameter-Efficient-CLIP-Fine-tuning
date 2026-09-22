"""Finish an already-running migration at the user-authorized epoch boundary.

This preserves the original eight-epoch schedule/config and checkpoints. It
stops only the specified supervisor and trainer, after both epoch checkpoints
and bindings are saved, then audits, reloads/evaluates, and packages predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/rematch750_npu/RM_FT_NPU/seed42"
CONFIG = ROOT / "configs/rematch750_ft_npu.yaml"


def check_process(pid, expected):
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if expected not in command:
        raise RuntimeError(f"Unexpected process {pid}: {command}")


def audit(config, epoch):
    import torch
    from aegis_clip.rematch_assets import validate_checkpoint

    parent = torch.load(config["train"]["init_checkpoint"], map_location="cpu", weights_only=False)
    with open(config["data"]["train_csv"]) as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    expected = epoch * ((count + config["train"]["batch_size"] - 1) // config["train"]["batch_size"])
    records = {}
    for name in ("best", "last"):
        path = RUN / f"checkpoints/{name}.pt"
        validate_checkpoint(path, config)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        steps = sorted({int(v["step"]) for v in ckpt["optimizer_state_dict"]["state"].values() if "step" in v})
        record = dict(epoch=ckpt["epoch"], global_step=ckpt["global_step"],
                      scheduler_steps=ckpt["scheduler_state_dict"]["last_epoch"],
                      optimizer_step_values=steps,
                      all_tensors_finite=all(bool(torch.isfinite(v).all()) for v in state.values()),
                      frozen_unchanged=all(torch.equal(state[k], parent["model_state_dict"][k])
                                           for k in ("visual.conv1.weight", "visual.positional_embedding")),
                      npu_rng_saved="npu" in ckpt["rng_state"],
                      raw_macro=ckpt["metrics"]["raw_macro"], raw_micro=ckpt["metrics"]["raw_micro"])
        assert record["epoch"] == epoch
        assert record["global_step"] == record["scheduler_steps"] == expected
        assert steps == [expected]
        assert record["all_tensors_finite"] and record["frozen_unchanged"] and record["npu_rng_saved"]
        records[name] = record
    reference = json.loads((ROOT / "results/rematch750_npu_epoch2_reference.json").read_text())
    assert reference["epoch"] == epoch
    records["gpu_same_epoch_reference"] = reference
    records["macro_delta_vs_gpu_same_epoch_pp"] = 100 * (records["best"]["raw_macro"] - reference["raw_macro"])
    assert records["macro_delta_vs_gpu_same_epoch_pp"] > -2
    return records


def reload_evaluate(config):
    import torch
    from torch.utils.data import DataLoader
    from aegis_clip.checkpoint import build_from_checkpoint
    from aegis_clip.data import OnlineImageDataset
    from aegis_clip.device import resolve_device
    from aegis_clip.evaluation import evaluate
    from aegis_clip.features import FrozenFeatureStore
    from aegis_clip.runtime import atomic_json_dump, seed_worker, set_seed

    device = resolve_device(config["train"]["device"])
    set_seed(config["project"]["seed"], deterministic=True)
    model, preprocess, ckpt = build_from_checkpoint(RUN / "checkpoints/best.pt", device, config_override=config)
    f = config["features"]
    store = FrozenFeatureStore(f["tensor_path"], f["paths_path"], f["manifest_path"], expected_dim=512)
    dataset = OnlineImageDataset(config["data"]["val_csv"], config["data"]["train_root"], preprocess, store)
    loader = DataLoader(dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False,
                        num_workers=4, timeout=120, pin_memory=False, worker_init_fn=seed_worker)
    with open(config["data"]["train_csv"]) as handle:
        labels = [int(row["label"]) for row in csv.DictReader(handle)]
    counts = torch.bincount(torch.tensor(labels), minlength=config["model"]["num_classes"])
    metrics = evaluate(model, loader, device=device, num_classes=config["model"]["num_classes"],
                       class_counts=counts, use_amp=True, selector_metric="raw_macro",
                       drift_penalty=0.0, measure_flip_consistency=False)
    for key in ("raw_macro", "raw_micro"):
        assert abs(metrics[key] - ckpt["metrics"][key]) < 1e-7, (key, metrics[key], ckpt["metrics"][key])
    atomic_json_dump(metrics, RUN / "checkpoints/best_evaluation.json")
    return {k: metrics[k] for k in ("raw_macro", "raw_micro", "samples")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--trainer-pid", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    check_process(args.supervisor_pid, "scripts/run_rematch_npu_migration.py --execute")
    check_process(args.trainer_pid, "aegis_clip.cli.rematch train --config configs/rematch750_ft_npu.yaml")
    os.kill(args.supervisor_pid, signal.SIGTERM)
    path = ROOT / "results/rematch750_npu_migration_execution.json"
    report = json.loads(path.read_text())
    snapshot = path.with_name("rematch750_npu_before_scope_change.json")
    if snapshot.exists():
        raise FileExistsError(snapshot)
    snapshot.write_text(json.dumps(report, indent=2) + "\n")
    report["scope_change"] = dict(reason="User explicitly chose two-epoch migration acceptance",
        completed_ft_epochs=2, configured_schedule_epochs=8, full_eight_epoch_comparison=False,
        command=[sys.executable, *sys.argv], trainer_pid=args.trainer_pid)

    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    save()
    try:
        log = ROOT / "outputs/rematch750_npu/migration/train_ft.log"
        deadline = time.monotonic() + 7200
        while "Selected checkpoint at epoch 2 " not in log.read_text():
            check_process(args.trainer_pid, "aegis_clip.cli.rematch train --config configs/rematch750_ft_npu.yaml")
            if time.monotonic() > deadline:
                raise TimeoutError("Epoch-two checkpoint deadline exceeded")
            time.sleep(2)
        for name in ("best", "last"):
            meta = json.loads((RUN / f"checkpoints/{name}.binding.json").read_text())
            assert meta["epoch"] == 2
        # SIGSTOP first prevents any later checkpoint writes while auditing.
        check_process(args.trainer_pid, "aegis_clip.cli.rematch train --config configs/rematch750_ft_npu.yaml")
        os.kill(args.trainer_pid, signal.SIGSTOP)
        os.kill(args.trainer_pid, signal.SIGTERM)
        os.kill(args.trainer_pid, signal.SIGCONT)
        time.sleep(5)
        entry = report["stages"][-1]
        assert entry["action"] == "train" and entry["candidate"] == "ft"
        entry.update(status="stopped_at_user_requested_checkpoint", completed_epochs=2,
                     termination_signal="SIGTERM", full_eight_epochs_completed=False)
        save()
        from aegis_clip.config import load_config
        from aegis_clip.cli.rematch import per_class_report
        config = load_config(CONFIG)
        report["checkpoint_audit"] = audit(config, 2)
        save()
        per_class_report(config, RUN)
        report["checkpoint_reload_evaluation"] = reload_evaluate(config)
        save()
        command = [sys.executable, "-u", "-m", "aegis_clip.cli.rematch", "infer", "--config", str(CONFIG)]
        entry = dict(action="infer", candidate="ft", command=command, status="running")
        report["stages"].append(entry)
        save()
        start = time.monotonic()
        with (ROOT / "outputs/rematch750_npu/migration/infer_ft.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode, seconds=time.monotonic()-start,
                     status="passed" if result.returncode == 0 else "failed")
        result.check_returncode()
        report["status"] = "passed_two_epoch_acceptance"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
