"""CPU export of averaged weights from one saved v1 training trajectory."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import yaml

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import fingerprint, load_artifact, save_artifact
from aegis_clip.v1_strategy import WeightAverage


def export_swa(context, output_dir, *, training_dir=None, start_epoch=None, end_epoch=None,
               weight_source="ema"):
    """Average a complete epoch window; source artifacts remain read-only.

    The original training binding is retained so the output can use the existing
    calibrate/infer pipeline. Export provenance records the distinct weight policy.
    """
    cfg = context.config
    start = cfg["train"]["swa_start"] if start_epoch is None else start_epoch
    end = cfg["train"]["epochs"] if end_epoch is None else end_epoch
    if not 1 <= start < end <= cfg["train"]["epochs"]:
        raise ValueError("SWA needs at least two epochs within the original fixed schedule")
    if weight_source not in ("ema", "raw"):
        raise ValueError("SWA weight source must be ema or raw")
    source = Path(training_dir or Path(cfg["output"]["root"]) / "training").resolve()
    output = Path(output_dir).resolve()
    if output == source.parent or output.is_relative_to(source.parent):
        raise ValueError("Export output must be outside the original run directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Export output is not empty: {output}")
    paths = [source / f"epoch_{epoch:02d}.pt" for epoch in range(start, end + 1)]
    missing = [p.name for p in paths if not p.is_file() or not p.with_suffix(".sha256.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Complete SWA epoch window {start}-{end} is unavailable in {source}: "
                                + ", ".join(missing))
    trajectory = fingerprint(dict(binding=context.binding, seed=cfg["project"]["seed"]))
    average, target_hash, schema, records = None, None, None, []
    started = time.monotonic()
    for epoch, path in zip(range(start, end + 1), paths):
        payload = load_artifact(path, context.binding)
        if (payload.get("epoch") != epoch or payload.get("trajectory") != trajectory
                or payload.get("classes") != context.classes
                or payload.get("model_config") != cfg["model"]):
            raise ValueError(f"Epoch, trajectory, architecture or classes disagree: {path}")
        if payload.get("optimizer") is None or payload.get("rng") is None:
            raise ValueError(f"SWA inputs must be complete training epoch checkpoints: {path}")
        training = payload.get("effective_training_config", {})
        if training.get("seed") != cfg["project"]["seed"] or any(
                training.get(key) != value for key, value in cfg["train"].items()):
            raise ValueError(f"Effective training recipe disagrees: {path}")
        checkpoint_targets = payload.get("targets_sha256")
        if not isinstance(checkpoint_targets, str) or not checkpoint_targets:
            raise ValueError(f"Checkpoint has no training target binding: {path}")
        if target_hash is None:
            target_hash = checkpoint_targets
        elif checkpoint_targets != target_hash:
            raise ValueError(f"SWA inputs use different training targets: {path}")
        raw = payload.get("raw_state")
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"Checkpoint has no raw trainable weights: {path}")
        if weight_source == "ema":
            ema = payload.get("ema")
            if (not isinstance(ema, dict) or ema.get("count", 0) < 1
                    or not cfg["train"]["ema_decay"] or ema.get("decay") != cfg["train"]["ema_decay"]):
                raise ValueError(f"Checkpoint has no matching updated EMA weights: {path}")
            state = ema.get("state")
        else:
            state = raw
        if not isinstance(state, dict) or state.keys() != raw.keys():
            raise ValueError(f"SWA trainable parameter keys disagree: {path}")
        if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
               or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError(f"SWA weights must be finite floating-point tensors: {path}")
        current_schema = {name: (tuple(value.shape), value.dtype) for name, value in state.items()}
        if schema is None:
            schema = current_schema
            average = WeightAverage(state)
        elif current_schema != schema:
            raise ValueError(f"SWA parameter shapes or dtypes disagree: {path}")
        if any(tuple(state[name].shape) != tuple(raw[name].shape) for name in state):
            raise ValueError(f"EMA and raw parameter shapes disagree: {path}")
        average.update(state)
        records.append(dict(epoch=epoch, path=str(path), sha256=sha256_file(path)))
        del payload, state, raw
    epochs = list(range(start, end + 1))
    selected = output / "selected.pt"
    policy = f"swa_{weight_source}"
    report = dict(status="exported_not_evaluated", strategy_name="v1", strategy_origin="team_original",
        device="cpu", training_started=False, weight_source=weight_source, selected_policy=policy,
        epochs=epochs, checkpoint=str(selected), source_training_dir=str(source),
        sources=records, trajectory=trajectory, targets_sha256=target_hash,
        exporter_sha256=sha256_file(__file__), binding=context.binding,
        local_score=None, platform_score=None, inference_mode="single_checkpoint",
        elapsed_seconds=time.monotonic() - started)
    # No fitting state can resume training, and no old calibration is carried over.
    save_artifact(selected, dict(epoch=end, trajectory=trajectory, selected_state=average.state,
        selected_policy=policy, swa_epochs=epochs, swa_source=weight_source,
        classes=context.classes, model_config=cfg["model"], targets_sha256=target_hash,
        optimizer=None, scheduler=None, scaler=None, rng=None, selected_metrics=None,
        export_provenance=report), context.binding)
    recipe = copy.deepcopy({key: value for key, value in cfg.items() if key != "_config_path"})
    recipe["output"]["root"] = str(output)
    (output / "config.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    atomic_json_dump(report, output / "export_report.json")
    return selected
