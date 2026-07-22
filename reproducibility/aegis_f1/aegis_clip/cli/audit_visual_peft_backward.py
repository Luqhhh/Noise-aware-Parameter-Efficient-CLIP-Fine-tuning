"""Run one real-image backward pass and audit the visual-PEFT freeze boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aegis_clip.checkpoint import load_initial_weights
from aegis_clip.config import load_config
from aegis_clip.data import OnlineImageDataset, TrustBundle
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.model import build_model
from aegis_clip.runtime import atomic_json_dump, seed_worker, set_seed, sha256_file
from aegis_clip.trainer import _per_sample_loss, _training_preprocess


def audit_visual_peft_backward(
    config_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int,
    num_workers: int,
) -> Path:
    config = load_config(config_path)
    seed = int(config["project"].get("seed", 42))
    set_seed(seed, deterministic=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for the real visual-PEFT backward audit")
    device = torch.device("cuda")
    model, preprocess = build_model(config, device)
    source = Path(config["train"]["init_checkpoint"]).resolve()
    load_initial_weights(model, source, device)
    train_preprocess = _training_preprocess(
        preprocess,
        config["data"],
        input_resolution=int(config["model"].get("input_resolution", 224)),
    )
    feature_store = FrozenFeatureStore(
        tensor_path=config["features"]["tensor_path"],
        paths_path=config["features"]["paths_path"],
        manifest_path=config["features"].get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    trust_bundle = TrustBundle(config["trust"]["bundle_path"])
    dataset = OnlineImageDataset(
        config["data"]["train_csv"],
        config["data"]["train_root"],
        train_preprocess,
        feature_store,
        trust_bundle,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=seed_worker,
    )
    batch = next(iter(loader))
    images = batch["images"].to(device, non_blocking=True)
    labels = batch["label"].long().to(device, non_blocking=True)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    use_amp = bool(config["train"].get("amp", True))
    amp_initial_scale = float(
        config["train"].get("amp_initial_scale", 65536.0)
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=amp_initial_scale,
    )
    loss_config = config["loss"]
    audit_epoch = max(1, int(loss_config.get("ce_warmup_epochs", 0)) + 1)
    with torch.autocast(device_type="cuda", enabled=use_amp):
        logits = model(images=images)
        targets = F.one_hot(
            labels,
            num_classes=int(config["model"]["num_classes"]),
        ).to(dtype=logits.dtype)
        loss = _per_sample_loss(
            logits,
            targets,
            loss_config,
            epoch=audit_epoch,
        ).mean()
    scaler.scale(loss).backward()

    trainable = []
    frozen_with_gradient = []
    changed_without_step = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if not parameter.requires_grad:
            if gradient is not None:
                frozen_with_gradient.append(name)
            continue
        nonzero = bool(gradient is not None and torch.count_nonzero(gradient))
        finite = bool(gradient is not None and torch.isfinite(gradient).all())
        trainable.append(
            {
                "name": name,
                "parameters": parameter.numel(),
                "gradient_present": gradient is not None,
                "gradient_nonzero": nonzero,
                "gradient_finite": finite,
            }
        )
        if not torch.equal(parameter.detach(), before[name]):
            changed_without_step.append(name)

    peft_marker = (
        ".visual_prompt."
        if model.peft_mode == "visual_prompt"
        else ".adaptmlp."
    )
    adapter_rows = [row for row in trainable if peft_marker in row["name"]]
    head_rows = [row for row in trainable if row["name"].startswith("classifier.")]
    if not adapter_rows or not head_rows:
        raise RuntimeError("audit did not find both visual PEFT and classifier parameters")
    if not any(row["gradient_nonzero"] for row in adapter_rows):
        raise RuntimeError("no visual PEFT parameter received a non-zero gradient")
    if not all(row["gradient_nonzero"] for row in head_rows):
        raise RuntimeError("not every classifier parameter received a non-zero gradient")
    invalid_gradients = [
        row
        for row in trainable
        if not row["gradient_present"] or not row["gradient_finite"]
    ]
    if invalid_gradients:
        details = ", ".join(
            f"{row['name']}[present={row['gradient_present']},"
            f"finite={row['gradient_finite']},nonzero={row['gradient_nonzero']}]"
            for row in invalid_gradients
        )
        raise RuntimeError(
            "trainable parameters have missing or non-finite gradients: " + details
        )
    if frozen_with_gradient:
        raise RuntimeError("frozen parameters received gradients")
    if changed_without_step:
        raise RuntimeError("parameters changed even though no optimizer step was executed")

    output_path = Path(output_path).resolve()
    report = {
        "status": "pass",
        "batch_size": int(images.shape[0]),
        "loss": float(loss.detach()),
        "loss_name": str(loss_config["name"]),
        "loss_audit_epoch": audit_epoch,
        "amp_enabled": use_amp,
        "amp_initial_scale": amp_initial_scale,
        "logits_shape": list(logits.shape),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "trainable_parameters": sum(row["parameters"] for row in trainable),
        "adapter_parameters": sum(row["parameters"] for row in adapter_rows),
        "classifier_parameters": sum(row["parameters"] for row in head_rows),
        "adapter_tensors_with_nonzero_gradient": sum(
            row["gradient_nonzero"] for row in adapter_rows
        ),
        "adapter_tensors": len(adapter_rows),
        "classifier_tensors_with_nonzero_gradient": sum(
            row["gradient_nonzero"] for row in head_rows
        ),
        "classifier_tensors": len(head_rows),
        "frozen_parameters_with_gradient": frozen_with_gradient,
        "parameters_changed_without_optimizer_step": changed_without_step,
        "expected_zero_gradient_note": (
            "all visual prompt tensors should receive gradients"
            if model.peft_mode == "visual_prompt"
            else "zero-initialized Up makes Down/LN gradients zero on the first backward pass"
        ),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file(source),
        "effective_model_spec": model.effective_spec(),
        "trainable_tensors": trainable,
    }
    atomic_json_dump(report, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    print(
        audit_visual_peft_backward(
            args.config,
            args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )


if __name__ == "__main__":
    main()
