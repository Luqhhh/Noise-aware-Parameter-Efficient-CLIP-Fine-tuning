"""One CPU-only average of full_576 RAW epochs 3-5; no window/source search."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aegis_clip.v1_strategy import WeightAverage
from .plan import dump, json_read, require, sha, verify_prepared


def checked_snapshot(path, plan, last, epoch):
    sidecar = json_read(path.with_suffix(".binding.json"))
    require(sidecar["sha256"] == sha(path), "Raw snapshot checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload["binding"] == sidecar["binding"], "Raw snapshot binding mismatch")
    stable = {k: v for k, v in last["binding"].items() if k not in ("complete", "completed_epochs")}
    actual = {k: v for k, v in payload["binding"].items() if k not in ("complete", "completed_epochs")}
    require(stable == actual and actual["stage"] == "full_576", "Cannot mix stages, branches, parents or trajectories")
    require(payload["epoch"] == epoch-1 and payload["global_step"] == epoch*plan["stages"]["full_576"]["steps_per_epoch"]
            and payload["binding"]["completed_epochs"] == epoch and payload.get("weight_source") == "raw",
            "Wrong fixed epoch/raw source")
    require(payload["config"] == last["config"] and payload["num_classes"] == last["num_classes"]
            and payload["image_size"] == last["image_size"] == 576, "Snapshot model/recipe differs")
    state = payload["model"]
    reference = last["model"]
    require(state.keys() == reference.keys() and all(value.shape == reference[name].shape and
            value.dtype == reference[name].dtype and value.is_floating_point() and torch.isfinite(value).all()
            for name, value in state.items()), "Raw snapshot schema or numeric state differs")
    return state


def export(plan_path, output):
    from .runtime import read_checkpoint, save_checkpoint
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = verify_prepared(plan_path)
    workspace = plan_path.parent
    source = workspace / "runs/full_576"
    require(not output.is_relative_to(workspace), "Use a separate export directory, outside the original workspace")
    require(not output.exists(), "Refusing to overwrite an export")
    paths = [source / f"epoch_{epoch:02d}_raw.pt" for epoch in (3, 4, 5)]
    require(all(p.is_file() and p.with_suffix(".binding.json").is_file() for p in paths),
            "Missing RAW full_576 epochs 3-5; do not retrain to manufacture snapshots")
    last = read_checkpoint(source / "last.pt", plan, "full_576")
    require(last["epoch"] == 4 and last["metrics"]["chosen"] == "raw", "Original final RAW candidate is incomplete")
    average, records = None, []
    for epoch, path in zip((3, 4, 5), paths):
        state = checked_snapshot(path, plan, last, epoch)
        if average is None:
            average = WeightAverage(state)
        average.update(state)
        records.append(dict(epoch=epoch, path=str(path), sha256=sha(path)))
    output.mkdir(parents=True)
    payload = {k: last[k] for k in ("binding", "config", "epoch", "global_step", "num_classes", "image_size")}
    payload.update(model=average.state, metrics=dict(chosen="raw"),
        experiment_id="V2_FULL_LAST3_SWA", weight_source="raw", average_epochs=[3, 4, 5],
        sources=records, original_last=str(source / "last.pt"), original_last_sha256=sha(source / "last.pt"))
    save_checkpoint(output / "selected.pt", payload)
    dump(output / "report.json", dict(status="exported_not_evaluated", experiment_id="V2_FULL_LAST3_SWA",
         plan_sha256=sha(plan_path), sources=records, checkpoint_sha256=sha(output / "selected.pt"),
         training_started=False, inference_mode="single_checkpoint", platform_score=None))
    return output / "selected.pt"


def read_export(path, plan, workspace):
    from .runtime import read_checkpoint
    path, workspace = Path(path).resolve(), Path(workspace).resolve()
    sidecar = json_read(path.with_suffix(".binding.json"))
    require(sidecar["sha256"] == sha(path), "SWA export checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    last_path = workspace / "runs/full_576/last.pt"
    last = read_checkpoint(last_path, plan, "full_576")
    require(payload["binding"] == sidecar["binding"] == last["binding"], "SWA/current final trajectory differs")
    require(payload.get("experiment_id") == "V2_FULL_LAST3_SWA" and payload.get("weight_source") == "raw"
            and payload.get("average_epochs") == [3, 4, 5] and payload.get("epoch") == 4
            and payload.get("original_last") == str(last_path) and payload.get("original_last_sha256") == sha(last_path),
            "Only the fixed final RAW3-5 export is allowed")
    require(payload["config"] == last["config"] and payload["num_classes"] == last["num_classes"]
            and payload["image_size"] == last["image_size"] == 576 and payload["global_step"] == last["global_step"]
            and payload["model"].keys() == last["model"].keys() and all(
                value.shape == last["model"][name].shape and value.dtype == last["model"][name].dtype
                and torch.isfinite(value).all() for name, value in payload["model"].items()), "Wrong SWA schema")
    require(len(payload["sources"]) == 3, "SWA source window incomplete")
    for epoch, record in zip((3, 4, 5), payload["sources"]):
        expected_path = last_path.parent / f"epoch_{epoch:02d}_raw.pt"
        require(record["epoch"] == epoch and record["path"] == str(expected_path)
                and record["sha256"] == sha(expected_path), "SWA source changed")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export(args.plan, args.output))


if __name__ == "__main__":
    main()
