"""Team strategy mechanisms with CPU-testable logical batches and stage guards."""
from __future__ import annotations


import numpy as np
import torch

from .plan import require

from . import training_utils
from .training_utils import build_view


def sample_weights(labels, classes):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=classes)
    require(len(counts) == classes and (counts > 0).all(), "Training split would empty a class")
    return torch.as_tensor(1.0 / np.sqrt(counts[labels]), dtype=torch.double)


def logical_backward(model, images, targets, micro_batch_size, autocast, supervision_mass=None):
    """One logical batch, one mix operation, mean CE, many micro forwards."""
    require(micro_batch_size > 0 and len(images) == len(targets), "Invalid logical batch")
    if supervision_mass is not None:
        require(torch.isfinite(supervision_mass).all() and float(supervision_mass) > 0,
                "Invalid ND-CW logical supervision mass")
    loss_sum = 0.0
    for start in range(0, len(images), micro_batch_size):
        end = min(start + micro_batch_size, len(images))
        with autocast():
            loss = training_utils.soft_cross_entropy(model(images[start:end]), targets[start:end])
            loss = loss * ((end - start) / (len(images) if supervision_mass is None else supervision_mass))
        loss.backward()
        loss_sum += float(loss.detach())
    return loss_sum


def check_checkpoint(payload, plan, stage):
    b = payload.get("binding", {})
    require(b.get("experiment_id") == plan["experiment_id"] and b.get("data_version") == plan["data_version"],
            "Checkpoint belongs to another experiment/phase")
    manifests = [v for k, v in plan["inputs"].items()
                 if k.replace("\\", "/").endswith("/train_manifest.csv")]
    require(manifests == [b.get("manifest_sha256")],
            "Checkpoint manifest identity mismatch")
    require(b.get("stage") == stage and b.get("strategy_revision") == plan["strategy_revision"] and
            b.get("official_sha256") == plan["recipe"]["official_sha256"], "Wrong checkpoint lineage")
    require(b.get("complete") is True and b.get("completed_epochs") == plan["stages"][stage]["epochs"],
            "Stage checkpoint is incomplete")
    require(b.get("stage_config_sha256") == plan["stages"][stage]["config_sha256"], "Wrong stage configuration")
    require(payload.get("num_classes") == plan["recipe"]["num_classes"] and
            payload.get("image_size") == plan["stages"][stage]["image_size"], "Wrong class count/resolution")
    require(0 <= payload.get("epoch", -1) < plan["stages"][stage]["epochs"] and
            payload.get("global_step") == (payload["epoch"] + 1) * plan["stages"][stage]["steps_per_epoch"],
            "Wrong checkpoint update endpoint")
    enabled = plan["recipe"]["ema_enabled"]
    chosen = payload.get("metrics", {}).get("chosen")
    require(payload.get("ema_enabled") is enabled and
            chosen in (("raw", "ema") if enabled else ("raw",)), "Checkpoint EMA/selection policy mismatch")
    if enabled:
        model, ema = payload.get("model", {}), payload.get("ema", {})
        require(bool(model) and model.keys() == ema.keys() and
                all(model[k].shape == ema[k].shape for k in model), "Incomplete EMA model state")
        require(payload.get("ema_decay") == plan["recipe"]["ema_decay"] and
                payload.get("ema_updates") == payload["global_step"], "Wrong EMA decay/update endpoint")
    else:
        require("ema" not in payload and payload.get("ema_updates") == 0, "EMA state present in a disabled stage")
