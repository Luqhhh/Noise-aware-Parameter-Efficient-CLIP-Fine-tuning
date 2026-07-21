"""Unified training loop for cached heads and online visual PEFT."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aegis_clip.checkpoint import (
    load_initial_weights,
    resume_checkpoint,
    save_checkpoint,
)
from aegis_clip.config import public_config
from aegis_clip.data import (
    CachedFeatureDataset,
    OnlineImageDataset,
    TrustBundle,
    load_class_mapping,
)
from aegis_clip.evaluation import evaluate, format_metrics
from aegis_clip.features import FrozenFeatureStore
from aegis_clip.losses import (
    AdaptiveLossCap,
    EarlyLearningRegularizer,
    classwise_suspicion_mask,
    class_prior_adjusted_logits,
    corrected_targets,
    mixup,
    project_conflicting_gradients,
    soft_cross_entropy,
    soft_generalized_cross_entropy,
)
from aegis_clip.model import AegisCLIP, build_model
from aegis_clip.runtime import (
    atomic_json_dump,
    configure_logging,
    environment_manifest,
    seed_worker,
    set_seed,
    sha256_file,
)


def _promotion_decision(
    epoch0: dict[str, Any],
    best: dict[str, Any],
    best_epoch: int,
    promotion_config: dict[str, Any],
) -> dict[str, Any]:
    """Determine whether the LoRA-trained model exceeds the epoch-0 parent."""
    selector_gain = float(best["selector"]) - float(epoch0["selector"])
    raw_gain = float(best["raw_micro"]) - float(epoch0["raw_micro"])
    checks = {
        "trained_epoch_selected": best_epoch >= 1,
        "selector_gain": selector_gain >= float(
            promotion_config.get("minimum_selector_gain", 0.001)
        ),
        "raw_micro_floor": raw_gain >= float(
            promotion_config.get("minimum_raw_micro_gain", -0.001)
        ),
        "drift_budget": float(best["mean_feature_drift"]) <= float(
            promotion_config.get("maximum_mean_feature_drift", 0.01)
        ),
        "class_coverage": int(best["predicted_class_count"]) == int(
            promotion_config.get("required_predicted_class_count", 500)
        ),
    }
    return {
        "passed": all(checks.values()),
        "best_epoch": int(best_epoch),
        "selector_gain": selector_gain,
        "raw_micro_gain": raw_gain,
        "checks": checks,
    }


def train(
    config: dict[str, Any],
    *,
    resume: str | None = None,
    init_checkpoint: str | None = None,
    overwrite: bool = False,
) -> Path:
    project = config["project"]
    data_config = config["data"]
    feature_config = config["features"]
    model_config = config["model"]
    train_config = config["train"]
    evaluation_config = config["evaluation"]
    output_config = config["output"]

    seed = int(project.get("seed", 42))
    set_seed(seed, deterministic=bool(train_config.get("deterministic", True)))
    device = torch.device(
        train_config.get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )
    run_dir = (
        Path(output_config["root"])
        / str(project["experiment_id"])
        / f"seed{seed}"
    )
    if run_dir.exists() and not (resume or overwrite):
        raise FileExistsError(
            f"Run directory exists: {run_dir}. Use --overwrite or --resume."
        )
    if overwrite and not resume and run_dir.exists():
        shutil.rmtree(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_dir, "train")

    class_to_idx, _ = load_class_mapping(data_config["class_mapping"])
    num_classes = int(model_config["num_classes"])
    if len(class_to_idx) != num_classes:
        raise ValueError(
            f"Class mapping has {len(class_to_idx)} classes, expected {num_classes}"
        )

    # --- Lineage audit (fail-close before any model or data loading) ---
    require_lineage = bool(train_config.get("require_lineage_for_init_checkpoint", False))
    if require_lineage:
        source = init_checkpoint or train_config.get("init_checkpoint")
        if not source:
            raise ValueError(
                "require_lineage_for_init_checkpoint=true but no init_checkpoint"
            )
        from aegis_clip.lineage import run_lineage_audit

        run_lineage_audit(
            config,
            child_train_csv=data_config["train_csv"],
            child_val_csv=data_config["val_csv"],
            checkpoint_path=source,
            output_path=run_dir / "split_lineage_audit.json",
        )

    feature_store = FrozenFeatureStore(
        tensor_path=feature_config["tensor_path"],
        paths_path=feature_config["paths_path"],
        manifest_path=feature_config.get("manifest_path"),
        expected_dim=int(model_config.get("feature_dim", 512)),
    )
    trust_bundle = (
        TrustBundle(config["trust"]["bundle_path"])
        if config.get("trust", {}).get("enabled", False)
        else None
    )

    model, preprocess = build_model(config, device)
    visual_peft = model.visual_requires_grad
    train_preprocess = _training_preprocess(preprocess, data_config)
    use_cached = bool(model_config.get("use_cached_training", not visual_peft))
    if visual_peft and use_cached:
        raise ValueError("Visual PEFT cannot use cached-only training")
    train_dataset = _build_dataset(
        use_cached=use_cached,
        split_csv=data_config["train_csv"],
        image_root=data_config["train_root"],
        preprocess=train_preprocess,
        feature_store=feature_store,
        trust_bundle=trust_bundle,
    )
    val_dataset = _build_dataset(
        use_cached=not visual_peft,
        split_csv=data_config["val_csv"],
        image_root=data_config["train_root"],
        preprocess=preprocess,
        feature_store=feature_store,
        trust_bundle=trust_bundle,
    )
    elr_config = config.get("elr", {})
    elr_regularizer = (
        EarlyLearningRegularizer(
            num_examples=len(train_dataset),
            num_classes=num_classes,
            momentum=float(elr_config.get("momentum", 0.9)),
            target_weight=float(elr_config.get("target_weight", 3.0)),
            warmup_epochs=int(elr_config.get("warmup_epochs", 5)),
            ramp_epochs=int(elr_config.get("ramp_epochs", 5)),
            epsilon=float(elr_config.get("epsilon", 1.0e-7)),
        ).to(device)
        if elr_config.get("enabled", False)
        else None
    )
    class_counts = torch.bincount(
        torch.as_tensor(train_dataset.labels, dtype=torch.long),
        minlength=num_classes,
    ).to(device=device, dtype=torch.float32)
    if (class_counts == 0).any():
        missing = torch.nonzero(class_counts == 0).flatten().tolist()
        raise ValueError(f"Training split is missing classes: {missing[:10]}")
    dual_gce_config = config["loss"].get("dual_gce", {})
    suspicious_mask = None
    if dual_gce_config.get("enabled", False):
        if trust_bundle is None:
            raise ValueError("dual_gce requires an OOF trust bundle")
        trust_scores = torch.stack(
            [
                trust_bundle.values_for(path, label)["clean_probability"]
                for path, label in zip(train_dataset.paths, train_dataset.labels)
            ]
        )
        suspicious_mask = classwise_suspicion_mask(
            torch.as_tensor(train_dataset.labels),
            trust_scores,
            float(dual_gce_config.get("suspicious_fraction", 0.2)),
        ).to(device)
    prior_tau = float(config["loss"].get("class_prior_adjustment_tau", 0.0))
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = int(train_config.get("num_workers", 4))
    timeout = int(train_config.get("loader_timeout", 120 if workers else 0))
    loader_options = {
        "num_workers": workers,
        "pin_memory": bool(train_config.get("pin_memory", True)),
        "timeout": timeout,
        "worker_init_fn": seed_worker,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = int(
            train_config.get("prefetch_factor", 1)
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(evaluation_config.get("batch_size", 256)),
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    groups = model.parameter_groups(
        head_lr=float(train_config["head_lr"]),
        head_weight_decay=float(train_config.get("head_weight_decay", 1.0e-4)),
        backbone_lr=float(train_config.get("backbone_lr", 0.0)),
        backbone_weight_decay=float(
            train_config.get("backbone_weight_decay", 1.0e-4)
        ),
    )
    optimizer = torch.optim.AdamW(groups)
    epochs = int(train_config["epochs"])
    schedule_epochs = int(train_config.get("schedule_epochs", epochs))
    total_steps = schedule_epochs * len(train_loader)
    warmup_steps = int(train_config.get("lr_warmup_epochs", 0)) * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine(step, warmup_steps, total_steps),
    )
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    cap_config = config["loss"].get("adaptive_cap", {})
    adaptive_cap = (
        AdaptiveLossCap(
            quantile=float(cap_config.get("quantile", 0.90)),
            momentum=float(cap_config.get("momentum", 0.90)),
            minimum=float(cap_config.get("minimum", 0.05)),
            maximum=float(cap_config.get("maximum", 10.0)),
        )
        if cap_config.get("enabled", False)
        else None
    )

    start_epoch = 1
    global_step = 0
    best_selector = -math.inf
    if resume:
        state = resume_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            adaptive_cap=adaptive_cap,
            elr_regularizer=elr_regularizer,
            data_generator=generator,
        )
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_selector = float(state["best_selector"])
        logger.info("Resumed from epoch %d", state["epoch"])
    elif init_checkpoint or train_config.get("init_checkpoint"):
        source = init_checkpoint or train_config["init_checkpoint"]
        state = load_initial_weights(model, source, device)
        logger.info("Initialised weights from %s (epoch=%s)", source, state.get("epoch"))

    # --- Epoch-0 baseline evaluation ---
    if not resume and (init_checkpoint or train_config.get("init_checkpoint")):
        epoch0_metrics = evaluate(
            model,
            val_loader,
            device=device,
            num_classes=num_classes,
            use_amp=use_amp,
            drift_budget=float(evaluation_config.get("drift_budget", 0.01)),
            drift_penalty=float(evaluation_config.get("drift_penalty", 0.5)),
            selector_metric=str(
                evaluation_config.get("selector_metric", "proxy_macro")
            ),
            clean_core_threshold=float(
                evaluation_config.get("clean_core_threshold", 0.70)
            ),
            measure_flip_consistency=bool(
                evaluation_config.get("measure_flip_consistency", False)
            ),
        )
        atomic_json_dump(epoch0_metrics, checkpoint_dir / "epoch0_evaluation.json")
        logger.info(
            "Epoch 0 baseline | %s",
            format_metrics(epoch0_metrics),
        )
        best_selector = float(epoch0_metrics["selector"])

        save_checkpoint(
            checkpoint_dir / "epoch0.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=0,
            global_step=0,
            best_selector=best_selector,
            config=config,
            metrics=epoch0_metrics,
            adaptive_cap_state=adaptive_cap.state_dict() if adaptive_cap else None,
            data_generator_state=generator.get_state(),
            elr_state_dict=(
                elr_regularizer.state_dict() if elr_regularizer is not None else None
            ),
        )
        shutil.copy2(checkpoint_dir / "epoch0.pt", checkpoint_dir / "best.pt")
        logger.info("Epoch-0 checkpoint saved as initial best")
        epoch0_saved = epoch0_metrics
    else:
        epoch0_saved = None

    effective_spec = model.effective_spec()
    _validate_effective_spec(effective_spec, model.peft_mode)
    atomic_json_dump(public_config(config), checkpoint_dir / "resolved_config.json")
    atomic_json_dump(effective_spec, checkpoint_dir / "effective_model_spec.json")
    atomic_json_dump(environment_manifest(), checkpoint_dir / "environment.json")
    logger.info("Effective model: %s", json.dumps(effective_spec, ensure_ascii=False))

    log_path = log_dir / "metrics.csv"
    _prepare_metrics_csv(log_path, resume=bool(resume))
    patience = int(train_config.get("early_stop_patience", 0))
    stale_epochs = 0
    first_step_audited = global_step > 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "examples": 0,
            "correct": 0,
            "projection_count": 0,
            "projection_cosine": 0.0,
            "feature_drift": 0.0,
            "cached_forward_examples": 0,
            "elr_regularization": 0.0,
        }
        for batch in train_loader:
            labels = batch["label"].to(device, non_blocking=True)
            batch_indices = batch["index"].to(device, non_blocking=True)
            batch_suspicious = (
                suspicious_mask[batch_indices]
                if suspicious_mask is not None
                else None
            )
            clean = batch["clean_probability"].to(device).float()
            pseudo = batch["pseudo_label"].to(device).long()
            correction = batch["correction_alpha"].to(device).float()
            if epoch <= int(config["trust"].get("correction_start_epoch", 0)):
                correction = torch.zeros_like(correction)
            targets = corrected_targets(labels, pseudo, correction, num_classes)
            minimum_weight = float(config["trust"].get("minimum_sample_weight", 0.25))
            if config.get("trust", {}).get("enabled", False) and epoch >= int(
                config["trust"].get("weighting_start_epoch", 1)
            ):
                weights = minimum_weight + (1.0 - minimum_weight) * clean
                selection_threshold = config["trust"].get("selection_threshold")
                if selection_threshold is not None:
                    rejected_weight = float(
                        config["trust"].get("rejected_sample_weight", 0.0)
                    )
                    weights = torch.where(
                        clean >= float(selection_threshold),
                        weights,
                        torch.full_like(weights, rejected_weight),
                    )
            else:
                weights = torch.ones_like(clean)

            input_key = "features" if "features" in batch else "images"
            original_inputs = batch[input_key].to(device, non_blocking=True)
            mixed_inputs, mixed_targets, mixed_weights, mix_lambda, mix_permutation = mixup(
                original_inputs,
                targets,
                weights,
                alpha=float(config["loss"].get("mixup_alpha", 0.0)),
                probability=float(config["loss"].get("mixup_probability", 0.0)),
                generator=generator,
            )
            reference = batch["reference_features"].to(device).float()
            mixed_reference = (
                mix_lambda * reference
                + (1.0 - mix_lambda) * reference[mix_permutation]
            )
            forward_key, forward_inputs, used_cached_forward = (
                _select_training_forward(
                    peft=visual_peft,
                    input_key=input_key,
                    mixed_inputs=mixed_inputs,
                    mixed_reference=mixed_reference,
                    mix_lambda=mix_lambda,
                )
            )
            before = (
                _snapshot_trainable(model)
                if not first_step_audited
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                arguments = {forward_key: forward_inputs}
                logits, encoded = model(**arguments, return_features=True)
                training_logits = class_prior_adjusted_logits(
                    logits, class_counts, prior_tau
                )
                per_sample = _per_sample_loss(
                    training_logits,
                    mixed_targets,
                    config["loss"],
                    epoch,
                    batch_suspicious,
                )
                mixed_clean = (
                    mix_lambda * clean
                    + (1.0 - mix_lambda) * clean[mix_permutation]
                )
                trusted_loss_mask = mixed_clean >= float(
                    config["trust"].get("anchor_threshold", 0.80)
                )
                cap_start = int(cap_config.get("start_epoch", 1))
                if adaptive_cap is not None and epoch >= cap_start:
                    per_sample = adaptive_cap(per_sample, trusted_loss_mask)
                classification_loss = (
                    per_sample * mixed_weights
                ).sum() / mixed_weights.sum().clamp_min(1.0e-8)
                loss = classification_loss
                elr_value = logits.new_zeros(())
                elr_weight = 0.0
                if elr_regularizer is not None:
                    elr_value = elr_regularizer.update_and_loss(
                        batch["index"].to(device, non_blocking=True), logits
                    )
                    elr_weight = elr_regularizer.rampup_weight(epoch)
                    loss = loss + elr_weight * elr_value
                distill_weight = (
                    float(config["loss"].get("feature_distillation_weight", 0.0))
                    if model.peft_mode != "frozen"
                    else 0.0
                )
                drift = 1.0 - F.cosine_similarity(
                    encoded.float(), F.normalize(mixed_reference, dim=1), dim=1
                )
                if distill_weight > 0.0:
                    loss = loss + distill_weight * drift.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            head_grad = _gradient_norm(model.classifier.parameters())
            visual_grad = _gradient_norm(model.visual.parameters())

            projection_config = config["trust"].get("gradient_projection", {})
            projection_interval = int(projection_config.get("interval", 0))
            anchor_mask = clean >= float(
                config["trust"].get("anchor_threshold", 0.80)
            )
            if (
                projection_config.get("enabled", False)
                and epoch >= int(projection_config.get("start_epoch", 1))
                and projection_interval > 0
                and global_step % projection_interval == 0
                and int(anchor_mask.sum()) >= int(
                    projection_config.get("minimum_anchor_samples", 4)
                )
            ):
                anchor_parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    anchor_arguments = {
                        input_key: original_inputs[anchor_mask]
                    }
                    anchor_logits = model(**anchor_arguments)
                    anchor_training_logits = class_prior_adjusted_logits(
                        anchor_logits, class_counts, prior_tau
                    )
                    anchor_targets = F.one_hot(
                        labels[anchor_mask], num_classes=num_classes
                    ).float()
                    anchor_loss = _per_sample_loss(
                        anchor_training_logits,
                        anchor_targets,
                        config["loss"],
                        epoch,
                        (
                            batch_suspicious[anchor_mask]
                            if batch_suspicious is not None
                            else None
                        ),
                    ).mean()
                anchor_gradients = torch.autograd.grad(
                    anchor_loss,
                    anchor_parameters,
                    allow_unused=True,
                )
                projection = project_conflicting_gradients(
                    anchor_parameters, anchor_gradients
                )
                if projection["projected"]:
                    totals["projection_count"] += 1
                totals["projection_cosine"] += float(projection["cosine"])

            max_grad_norm = float(train_config.get("max_grad_norm", 1.0))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if not first_step_audited:
                _audit_first_step(
                    model=model,
                    before=before or {},
                    head_grad=head_grad,
                    visual_grad=visual_grad,
                )
                first_step_audited = True
                logger.info(
                    "First-step audit passed: head_grad=%.6f visual_grad=%.6f",
                    head_grad,
                    visual_grad,
                )

            batch_size = labels.numel()
            totals["loss"] += float(loss.detach()) * batch_size
            totals["examples"] += batch_size
            totals["correct"] += int(
                (logits.detach().argmax(dim=1) == mixed_targets.argmax(dim=1)).sum()
            )
            totals["feature_drift"] += float(drift.detach().sum())
            totals["elr_regularization"] += float(elr_value.detach()) * batch_size
            if used_cached_forward:
                totals["cached_forward_examples"] += batch_size
            global_step += 1

        train_metrics = {
            "train_loss": totals["loss"] / totals["examples"],
            "train_accuracy": totals["correct"] / totals["examples"],
            "train_feature_drift": totals["feature_drift"] / totals["examples"],
            "train_cached_forward_fraction": (
                totals["cached_forward_examples"] / totals["examples"]
            ),
            "projection_count": int(totals["projection_count"]),
            "train_elr_regularization": (
                totals["elr_regularization"] / totals["examples"]
            ),
            "train_elr_weight": (
                elr_regularizer.rampup_weight(epoch)
                if elr_regularizer is not None
                else 0.0
            ),
        }
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            num_classes=num_classes,
            use_amp=use_amp,
            drift_budget=float(evaluation_config.get("drift_budget", 0.01)),
            drift_penalty=float(evaluation_config.get("drift_penalty", 0.5)),
            selector_metric=str(
                evaluation_config.get("selector_metric", "proxy_macro")
            ),
            clean_core_threshold=float(
                evaluation_config.get("clean_core_threshold", 0.70)
            ),
            measure_flip_consistency=bool(
                evaluation_config.get("measure_flip_consistency", False)
            ),
        )
        metrics = {**train_metrics, **val_metrics}
        if epoch0_saved is not None:
            metrics["delta_vs_epoch0_selector"] = (
                float(val_metrics["selector"]) - float(epoch0_saved["selector"])
            )
            metrics["delta_vs_epoch0_raw_micro"] = (
                float(val_metrics["raw_micro"]) - float(epoch0_saved["raw_micro"])
            )
            metrics["delta_vs_epoch0_clean_core_micro"] = (
                float(val_metrics["clean_core_micro"])
                - float(epoch0_saved["clean_core_micro"])
            )
        logger.info("Epoch %d | %s", epoch, format_metrics(metrics))
        _append_metrics_csv(log_path, epoch, metrics, optimizer)

        selector = float(val_metrics["selector"])
        improved = selector > best_selector
        if improved:
            best_selector = selector
            stale_epochs = 0
        else:
            stale_epochs += 1
        common = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_selector=best_selector,
            config=config,
            metrics=metrics,
            adaptive_cap_state=adaptive_cap.state_dict() if adaptive_cap else None,
            data_generator_state=generator.get_state(),
            elr_state_dict=(
                elr_regularizer.state_dict() if elr_regularizer is not None else None
            ),
        )
        save_checkpoint(checkpoint_dir / "last.pt", **common)
        if improved:
            save_checkpoint(checkpoint_dir / "best.pt", **common)
            logger.info("New best clean-proxy selector: %.6f", best_selector)
        if patience > 0 and stale_epochs >= patience:
            logger.info("Early stopping after %d stale epochs", stale_epochs)
            break

    best_path = checkpoint_dir / "best.pt"
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    final_metrics = evaluate(
        model,
        val_loader,
        device=device,
        num_classes=num_classes,
        use_amp=use_amp,
        drift_budget=float(evaluation_config.get("drift_budget", 0.01)),
        drift_penalty=float(evaluation_config.get("drift_penalty", 0.5)),
        selector_metric=str(
            evaluation_config.get("selector_metric", "proxy_macro")
        ),
        clean_core_threshold=float(
            evaluation_config.get("clean_core_threshold", 0.70)
        ),
        measure_flip_consistency=bool(
            evaluation_config.get("measure_flip_consistency", False)
        ),
    )
    atomic_json_dump(final_metrics, checkpoint_dir / "best_evaluation.json")

    # --- Promotion decision ---
    if epoch0_saved is not None:
        best_epoch = int(best["epoch"])
        promotion = _promotion_decision(
            epoch0_saved,
            final_metrics,
            best_epoch,
            config.get("promotion", {}),
        )
        atomic_json_dump(promotion, checkpoint_dir / "promotion.json")
        logger.info(
            "Promotion: %s selector_gain=%.6f raw_gain=%.6f",
            "PASS" if promotion["passed"] else "FAIL",
            promotion["selector_gain"],
            promotion["raw_micro_gain"],
        )

    atomic_json_dump(
        {
            "experiment_id": project["experiment_id"],
            "seed": seed,
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": sha256_file(best_path),
            "train_csv_sha256": sha256_file(data_config["train_csv"]),
            "val_csv_sha256": sha256_file(data_config["val_csv"]),
            "trust_bundle_sha256": (
                sha256_file(config["trust"]["bundle_path"])
                if trust_bundle is not None
                else None
            ),
            "split_lineage_audit_sha256": (
                sha256_file(run_dir / "split_lineage_audit.json")
                if (run_dir / "split_lineage_audit.json").exists()
                else None
            ),
            "epoch0_evaluation_sha256": (
                sha256_file(checkpoint_dir / "epoch0_evaluation.json")
                if (checkpoint_dir / "epoch0_evaluation.json").exists()
                else None
            ),
            "promotion_sha256": (
                sha256_file(checkpoint_dir / "promotion.json")
                if (checkpoint_dir / "promotion.json").exists()
                else None
            ),
            "metrics": final_metrics,
            "effective_model_spec": model.effective_spec(),
        },
        checkpoint_dir / "artifact_manifest.json",
    )
    return best_path


def _build_dataset(
    *,
    use_cached: bool,
    split_csv: str,
    image_root: str,
    preprocess: Any,
    feature_store: FrozenFeatureStore,
    trust_bundle: TrustBundle | None,
) -> torch.utils.data.Dataset:
    if use_cached:
        return CachedFeatureDataset(split_csv, feature_store, trust_bundle)
    return OnlineImageDataset(
        split_csv, image_root, preprocess, feature_store, trust_bundle
    )


def _training_preprocess(preprocess: Any, data_config: dict[str, Any]) -> Any:
    preset = str(data_config.get("train_augmentation", "clip_center_crop"))
    if preset == "clip_center_crop":
        return preprocess
    if preset != "weak_rrc_flip":
        raise ValueError(f"Unsupported training augmentation: {preset}")
    try:
        from torchvision.transforms import (
            Compose,
            InterpolationMode,
            RandomHorizontalFlip,
            RandomResizedCrop,
        )
    except ImportError as exc:
        raise ImportError("weak_rrc_flip requires torchvision") from exc
    transforms = list(getattr(preprocess, "transforms", []))
    if len(transforms) < 2:
        raise ValueError("Cannot derive CLIP tensor conversion and normalization")
    return Compose(
        [
            RandomResizedCrop(
                224,
                scale=(0.70, 1.0),
                ratio=(0.85, 1.15),
                interpolation=InterpolationMode.BICUBIC,
            ),
            RandomHorizontalFlip(p=0.5),
            transforms[-2],
            transforms[-1],
        ]
    )


def _select_training_forward(
    *,
    peft: bool,
    input_key: str,
    mixed_inputs: torch.Tensor,
    mixed_reference: torch.Tensor,
    mix_lambda: float,
) -> tuple[str, torch.Tensor, bool]:
    """Reuse frozen features only when an online pixel batch was not mixed."""
    if not peft and input_key == "features":
        return "features", mixed_inputs, True
    if not peft and input_key == "images" and mix_lambda == 1.0:
        return "features", mixed_reference, True
    return input_key, mixed_inputs, False


def _per_sample_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_config: dict[str, Any],
    epoch: int,
    suspicious_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if epoch <= int(loss_config.get("ce_warmup_epochs", 0)):
        return soft_cross_entropy(logits, targets)
    if loss_config["name"] == "cross_entropy":
        return soft_cross_entropy(logits, targets)
    dual_gce = loss_config.get("dual_gce", {})
    if dual_gce.get("enabled", False):
        if suspicious_mask is None or suspicious_mask.numel() != logits.shape[0]:
            raise ValueError("dual_gce requires one suspicion flag per sample")
        clean = soft_generalized_cross_entropy(
            logits,
            targets,
            q=float(dual_gce.get("clean_q", loss_config.get("gce_q", 0.5))),
            epsilon=float(loss_config.get("epsilon", 1.0e-7)),
        )
        suspicious = soft_generalized_cross_entropy(
            logits,
            targets,
            q=float(dual_gce.get("suspicious_q", 1.0)),
            epsilon=float(loss_config.get("epsilon", 1.0e-7)),
        )
        return torch.where(suspicious_mask.bool(), suspicious, clean)
    return soft_generalized_cross_entropy(
        logits,
        targets,
        q=float(loss_config.get("gce_q", 0.5)),
        epsilon=float(loss_config.get("epsilon", 1.0e-7)),
    )


def _warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1.0e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _gradient_norm(parameters: Any) -> float:
    values = [
        parameter.grad.detach().float().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt())


def _snapshot_trainable(model: AegisCLIP) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _audit_first_step(
    *,
    model: AegisCLIP,
    before: dict[str, torch.Tensor],
    head_grad: float,
    visual_grad: float,
) -> None:
    if not math.isfinite(head_grad) or head_grad <= 0.0:
        raise RuntimeError(f"Invalid first-step head gradient: {head_grad}")
    if model.visual_requires_grad and (
        not math.isfinite(visual_grad) or visual_grad <= 0.0
    ):
        raise RuntimeError(f"PEFT configured but visual gradient is {visual_grad}")
    changed = []
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        delta = (parameter.detach().cpu() - before[name]).abs().max()
        if float(delta) > 0.0:
            changed.append(name)
    if not any(name.startswith("classifier.") for name in changed):
        raise RuntimeError("First optimizer step did not change classifier parameters")
    if model.visual_requires_grad and not any(
        name.startswith("visual.") for name in changed
    ):
        raise RuntimeError("PEFT configured but first step changed no visual parameter")
    if model.peft_mode == "feature_adapter" and not any(
        name.startswith("feature_adapter.") for name in changed
    ):
        raise RuntimeError("Feature adapter configured but no adapter parameter changed")


def _validate_effective_spec(spec: dict[str, Any], peft_mode: str) -> None:
    if peft_mode == "frozen" and spec["visual_requires_grad"]:
        raise RuntimeError("Frozen mode has trainable visual parameters")
    if peft_mode in {"visual_ln", "ln_post_proj", "visual_lora"} and not spec[
        "visual_requires_grad"
    ]:
        raise RuntimeError("Visual PEFT mode has no trainable visual parameters")
    if peft_mode == "visual_lora" and not any(
        "parametrizations" in name for name in spec["trainable_names"]
    ):
        raise RuntimeError("Visual LoRA mode has no trainable low-rank parameters")
    if peft_mode == "feature_adapter":
        if spec["visual_requires_grad"]:
            raise RuntimeError("Feature adapter must keep visual parameters frozen")
        if not any(
            name.startswith("feature_adapter.")
            for name in spec["trainable_names"]
        ):
            raise RuntimeError("Feature adapter mode has no trainable adapter parameters")


def _prepare_metrics_csv(path: Path, resume: bool) -> None:
    if resume and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_feature_drift",
                "train_cached_forward_fraction",
                "projection_count",
                "train_elr_regularization",
                "train_elr_weight",
                "raw_micro",
                "raw_macro",
                "trusted_micro",
                "trusted_macro",
                "proxy_micro",
                "proxy_macro",
                "clean_core_micro",
                "clean_core_macro",
                "clean_core_samples",
                "flip_prediction_agreement",
                "mean_feature_drift",
                "selector",
                "head_lr",
                "visual_lr",
            ]
        )


def _append_metrics_csv(
    path: Path,
    epoch: int,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> None:
    learning_rates = {
        group.get("name", f"group{index}"): group["lr"]
        for index, group in enumerate(optimizer.param_groups)
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                epoch,
                metrics["train_loss"],
                metrics["train_accuracy"],
                metrics["train_feature_drift"],
                metrics["train_cached_forward_fraction"],
                metrics["projection_count"],
                metrics["train_elr_regularization"],
                metrics["train_elr_weight"],
                metrics["raw_micro"],
                metrics["raw_macro"],
                metrics["trusted_micro"],
                metrics["trusted_macro"],
                metrics["proxy_micro"],
                metrics["proxy_macro"],
                metrics["clean_core_micro"],
                metrics["clean_core_macro"],
                metrics["clean_core_samples"],
                metrics["flip_prediction_agreement"],
                metrics["mean_feature_drift"],
                metrics["selector"],
                learning_rates.get("head", ""),
                learning_rates.get("visual", ""),
            ]
        )
