"""Transferred mechanisms with CPU-testable logical batches and stage guards."""
from __future__ import annotations


import numpy as np
import torch

from .plan import require

from . import reference
from .reference import build_view


def sample_weights(labels, classes):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=classes)
    require(len(counts) == classes and (counts > 0).all(), "Training split would empty a class")
    return torch.as_tensor(1.0 / np.sqrt(counts[labels]), dtype=torch.double)


def logical_backward(model, images, targets, micro_batch_size, autocast):
    """One source logical batch, one mix operation, mean CE, many micro forwards."""
    require(micro_batch_size > 0 and len(images) == len(targets), "Invalid logical batch")
    loss_sum = 0.0
    for start in range(0, len(images), micro_batch_size):
        end = min(start + micro_batch_size, len(images))
        with autocast():
            loss = reference.soft_cross_entropy(model(images[start:end]), targets[start:end])
            loss = loss * ((end - start) / len(images))
        loss.backward()
        loss_sum += float(loss.detach())
    return loss_sum


def check_checkpoint(payload, plan, stage):
    b = payload.get("binding", {})
    require(b.get("experiment_id") == plan["experiment_id"] and b.get("data_version") == plan["data_version"],
            "Checkpoint belongs to another experiment/phase")
    manifests = [v for k, v in plan["inputs"].items() if k.endswith("/train_manifest.csv")]
    require(manifests == [b.get("manifest_sha256")],
            "Checkpoint manifest identity mismatch")
    require(b.get("stage") == stage and b.get("source_commit") == plan["source_commit"] and
            b.get("official_sha256") == plan["recipe"]["official_sha256"], "Wrong checkpoint lineage")
    require(b.get("complete") is True and b.get("completed_epochs") == plan["stages"][stage]["epochs"],
            "Stage checkpoint is incomplete")
    require(b.get("stage_config_sha256") == plan["stages"][stage]["config_sha256"], "Wrong stage configuration")
    require(payload.get("num_classes") == plan["recipe"]["num_classes"] and
            payload.get("image_size") == plan["stages"][stage]["image_size"], "Wrong class count/resolution")
    require(0 <= payload.get("epoch", -1) < plan["stages"][stage]["epochs"] and
            payload.get("global_step") == (payload["epoch"] + 1) * plan["stages"][stage]["steps_per_epoch"],
            "Wrong checkpoint update endpoint")
