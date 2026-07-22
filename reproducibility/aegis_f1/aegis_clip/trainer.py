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
from aegis_clip.features import FrozenFeatureStore, canonical_sample_path
from aegis_clip.losses import (
    AdaptiveLossCap,
    EarlyLearningRegularizer,
    TrustedPrototypeBank,
    active_forgetting_noise_suppression_losses,
    classwise_high_loss_filter,
    classwise_suspicion_mask,
    class_prior_adjusted_logits,
    consensus_conflict_mask,
    corrected_targets,
    double_softmax_cross_entropy,
    mixup,
    noise_tolerant_supervised_contrastive_loss,
    project_conflicting_gradients,
    soft_cross_entropy,
    soft_generalized_cross_entropy,
    smoothstep_damped_loss,
)
from aegis_clip.local_inference import (
    attention_guided_crop,
    logits_with_last_block_attention,
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
from aegis_clip.snscl import (
    StatefulSNSCL,
    classwise_queue_contrastive_loss,
)
from aegis_clip.trust_subspace import (
    OnlineTrustGradientSubspace,
    construct_trust_subspace_gradient,
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
    train_preprocess = _training_preprocess(
        preprocess,
        data_config,
        input_resolution=int(model_config.get("input_resolution", 224)),
    )
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
    _validate_train_val_overlap(
        train_dataset.paths,
        val_dataset.paths,
        allow_overlap=bool(data_config.get("validation_overlap_with_training", False)),
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
    train_clean_scores = None
    if trust_bundle is not None:
        train_clean_scores = torch.stack(
            [
                trust_bundle.values_for(path, label)["clean_probability"]
                for path, label in zip(train_dataset.paths, train_dataset.labels)
            ]
        )
    dual_gce_config = config["loss"].get("dual_gce", {})
    suspicious_mask = None
    if dual_gce_config.get("enabled", False):
        if trust_bundle is None:
            raise ValueError("dual_gce requires an OOF trust bundle")
        suspicious_mask = classwise_suspicion_mask(
            torch.as_tensor(train_dataset.labels),
            train_clean_scores,
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

    snscl_config = config["loss"].get("snscl", {})
    snscl_state = None
    if snscl_config.get("enabled", False):
        if trust_bundle is None:
            raise ValueError("SNSCL requires cross-fitted trust evidence")
        snscl_state = StatefulSNSCL(
            num_samples=len(train_dataset),
            num_classes=num_classes,
            feature_dim=int(model_config.get("feature_dim", 512)),
            hidden_dim=int(snscl_config.get("hidden_dim", 2048)),
            projection_dim=int(snscl_config.get("projection_dim", 128)),
            queue_size=int(snscl_config.get("queue_size", 32)),
            initial_std=float(snscl_config.get("initial_std", 0.05)),
            mean_residual_scale=float(
                snscl_config.get("mean_residual_scale", 0.1)
            ),
        ).to(device)

    subspace_config = config["trust"].get("subspace_projection", {})
    trust_subspace = (
        OnlineTrustGradientSubspace(
            max_rank=int(subspace_config.get("rank", 8)),
            epsilon=float(subspace_config.get("epsilon", 1.0e-12)),
        )
        if subspace_config.get("enabled", False)
        else None
    )
    if snscl_state is not None and trust_subspace is not None:
        raise ValueError("SNSCL and trust-subspace state cannot share one checkpoint")
    training_auxiliary = (
        snscl_state if snscl_state is not None else trust_subspace
    )

    groups = model.parameter_groups(
        head_lr=float(train_config["head_lr"]),
        head_weight_decay=float(train_config.get("head_weight_decay", 1.0e-4)),
        backbone_lr=float(train_config.get("backbone_lr", 0.0)),
        backbone_weight_decay=float(
            train_config.get("backbone_weight_decay", 1.0e-4)
        ),
    )
    if snscl_state is not None:
        groups.append(
            {
                "name": "snscl",
                "params": list(snscl_state.parameters()),
                "lr": float(snscl_config["module_lr"]),
                "weight_decay": float(snscl_config.get("module_weight_decay", 1.0e-4)),
            }
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
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=use_amp,
        init_scale=float(train_config.get("amp_initial_scale", 65536.0)),
    )

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
    active_forgetting_config = config["loss"].get("active_forgetting", {})
    active_forgetting_enabled = bool(
        active_forgetting_config.get("enabled", False)
    )
    attention_local_config = config["loss"].get("attention_local_training", {})
    attention_local_enabled = bool(attention_local_config.get("enabled", False))

    routing_config = config.get("clean_routing", {})
    routing_enabled = bool(routing_config.get("enabled", False))
    routing_mode = str(routing_config.get("mode", "hard"))
    routing_threshold = float(routing_config.get("threshold", 0.70))
    routing_start_epoch = int(routing_config.get("start_epoch", 1))

    proto_config = config.get("prototype_contrastive", {})
    proto_enabled = bool(proto_config.get("enabled", False))
    prototype_bank = (
        TrustedPrototypeBank(
            num_classes=num_classes,
            feature_dim=int(model_config.get("feature_dim", 512)),
            momentum=float(proto_config.get("momentum", 0.99)),
            temperature=float(proto_config.get("temperature", 0.10)),
            threshold=float(proto_config.get("threshold", 0.80)),
        ).to(device)
        if proto_enabled
        else None
    )
    proto_weight = float(proto_config.get("loss_weight", 0.05))
    proto_start_epoch = int(proto_config.get("start_epoch", 1))

    dynamic_config = config.get("dynamic_trust", {})
    dynamic_enabled = bool(dynamic_config.get("enabled", False))
    dynamic_refresh_epoch = int(dynamic_config.get("refresh_epoch", 2))
    dynamic_clean: torch.Tensor | None = None  # populated at refresh epoch, shape [N]

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
            training_auxiliary=training_auxiliary,
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

    if (
        not resume
        and start_epoch == 1
        and bool(evaluation_config.get("record_initial", False))
    ):
        initial_metrics = evaluate(
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
        atomic_json_dump(initial_metrics, checkpoint_dir / "initial_evaluation.json")
        logger.info("Initial checkpoint | %s", format_metrics(initial_metrics))

    cyclic_config = config["loss"].get("cyclic_filter", {})
    cyclic_enabled = bool(cyclic_config.get("enabled", False))
    cyclic_filter_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
    cycle_epochs = int(cyclic_config.get("cycle_epochs", 15))
    if cyclic_enabled:
        if train_clean_scores is None:
            raise ValueError("Cyclic filtering requires trust scores")
        cyclic_filter_mask = _refresh_cyclic_filter(
            model=model,
            train_dataset=train_dataset,
            device=device,
            use_amp=use_amp,
            loss_config=config["loss"],
            epoch=max(start_epoch, 1),
            clean_scores=train_clean_scores,
            cyclic_config=cyclic_config,
            num_classes=num_classes,
        )
        logger.info(
            "Initial cyclic filter selected %d/%d samples",
            int(cyclic_filter_mask.sum()),
            len(cyclic_filter_mask),
        )

    for epoch in range(start_epoch, epochs + 1):
        # --- Dynamic Trust Refresh (P4) ---
        if dynamic_enabled and epoch == dynamic_refresh_epoch:
            logger.info("Dynamic trust refresh at epoch %d...", epoch)
            model.eval()
            all_idx, all_scores = [], []
            for batch in train_loader:
                images = batch["images"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                indices = batch["index"]
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits_orig, _ = model(images=images, return_features=True)
                    logits_flip, _ = model(
                        images=torch.flip(images, dims=(3,)), return_features=True
                    )
                probs = (F.softmax(logits_orig.float(), dim=1) + F.softmax(logits_flip.float(), dim=1)) / 2.0
                mean_conf, pred = probs.max(dim=1)
                flip_agree = (logits_orig.argmax(dim=1) == logits_flip.argmax(dim=1))
                clean_mask = (mean_conf >= 0.80) & flip_agree & (pred == labels)
                hard_mask = ((mean_conf >= 0.50) & ~clean_mask) | ~flip_agree
                new_clean = torch.where(
                    clean_mask,
                    torch.tensor(1.0),
                    torch.where(hard_mask, torch.tensor(0.5), torch.tensor(0.0)),
                )
                all_idx.append(indices)
                all_scores.append(new_clean.cpu())
            full_idx = torch.cat(all_idx)
            full_scores = torch.cat(all_scores)
            dynamic_clean = torch.zeros(len(train_dataset))
            dynamic_clean[full_idx] = full_scores
            clean_count = int((dynamic_clean >= 0.80).sum())
            hard_count = int(((dynamic_clean >= 0.50) & (dynamic_clean < 0.80)).sum())
            reject_count = int((dynamic_clean < 0.50).sum())
            logger.info(
                "Dynamic refresh done: clean=%d hard=%d reject=%d (total=%d)",
                clean_count, hard_count, reject_count, len(train_dataset),
            )
            model.train()

        model.train()
        epoch_in_cycle = (epoch - 1) % cycle_epochs + 1
        reintroduction_epoch = cyclic_enabled and epoch_in_cycle == cycle_epochs
        active_cyclic_filter = (
            torch.zeros_like(cyclic_filter_mask)
            if reintroduction_epoch
            else cyclic_filter_mask
        ).to(device)
        cyclic_delta = 0.0
        totals = {
            "loss": 0.0,
            "examples": 0,
            "correct": 0,
            "projection_count": 0,
            "projection_cosine": 0.0,
            "feature_drift": 0.0,
            "cached_forward_examples": 0,
            "consensus_conflict_dropped": 0,
            "elr_regularization": 0.0,
            "contrastive_loss": 0.0,
            "snscl_contrastive_loss": 0.0,
            "snscl_kl_loss": 0.0,
            "snscl_usable_anchors": 0,
            "snscl_admitted": 0,
            "fine_active_forgetting": 0.0,
            "fine_negative_learning": 0.0,
            "fine_suspicious_samples": 0,
            "attention_local_classification": 0.0,
            "attention_local_consistency": 0.0,
            "attention_local_agreement": 0.0,
            "attention_local_examples": 0,
            "trust_subspace_steps": 0,
            "trust_subspace_skipped_steps": 0,
            "trust_subspace_basis_updates": 0,
            "trust_subspace_projection_steps": 0,
            "trust_subspace_trusted_examples": 0,
            "trust_subspace_uncertain_examples": 0,
            "trust_subspace_reference_gradient_norm": 0.0,
            "trust_subspace_shared_gradient_norm": 0.0,
            "trust_subspace_uncertain_gradient_norm": 0.0,
            "trust_subspace_projected_gradient_norm": 0.0,
            "trust_subspace_retained_norm_ratio": 0.0,
            "trust_subspace_uncertain_loss": 0.0,
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
            if dynamic_clean is not None and epoch >= dynamic_refresh_epoch:
                clean = dynamic_clean[batch_indices.cpu()].to(device).float()
            pseudo = batch["pseudo_label"].to(device).long()
            pseudo_confidence = batch["pseudo_confidence"].to(device).float()
            correction_evidence = batch["correction_alpha"].to(device).float()
            correction = correction_evidence
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
            conflict_config = config["trust"].get("consensus_conflict", {})
            conflict_mode = str(conflict_config.get("mode", "keep"))
            if conflict_mode not in {"keep", "drop"}:
                raise ValueError(
                    "trust.consensus_conflict.mode must be 'keep' or 'drop'"
                )
            if conflict_mode == "drop":
                conflict_mask = consensus_conflict_mask(
                    labels,
                    pseudo,
                    pseudo_confidence,
                    correction_evidence,
                    minimum_confidence=float(
                        conflict_config.get("minimum_pseudo_confidence", 0.85)
                    ),
                )
                weights = torch.where(
                    conflict_mask, torch.zeros_like(weights), weights
                )
                totals["consensus_conflict_dropped"] += int(conflict_mask.sum())
            if cyclic_enabled:
                weights = torch.where(
                    active_cyclic_filter[batch_indices],
                    torch.zeros_like(weights),
                    weights,
                )

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

            # --- Clean-Routing gate ---
            if routing_enabled and epoch >= routing_start_epoch:
                if routing_mode == "hard":
                    gate = (clean >= routing_threshold).float()
                else:
                    gate = ((clean - 0.5) / 0.5).clamp(0.0, 1.0)
                # Apply mixup to gate so it aligns with mixed inputs
                mixed_gate = (
                    mix_lambda * gate
                    + (1.0 - mix_lambda) * gate[mix_permutation]
                )
            else:
                mixed_gate = None
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
            snscl_projected = None
            snscl_hard_labels = None
            snscl_admission_probabilities = None
            snscl_usable_anchors = 0
            with torch.autocast(device_type=device.type, enabled=use_amp):
                arguments: dict[str, torch.Tensor] = {forward_key: forward_inputs}
                if mixed_gate is not None and forward_key == "images":
                    arguments["gate"] = mixed_gate
                    arguments["reference_features"] = mixed_reference
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
                if cyclic_enabled:
                    per_sample, cyclic_delta = smoothstep_damped_loss(
                        per_sample,
                        maximum_delta=float(cyclic_config["maximum_delta"]),
                        epoch_in_cycle=epoch_in_cycle,
                        cycle_epochs=cycle_epochs,
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
                local_classification_value = logits.new_zeros(())
                local_consistency_value = logits.new_zeros(())
                local_agreement_value = logits.new_zeros(())
                if attention_local_enabled:
                    if input_key != "images" or mix_lambda != 1.0:
                        raise ValueError(
                            "Attention-local training requires unmixed online images"
                        )
                    # Crop selection is a deterministic, label-free teacher path.
                    # Detaching it prevents the non-smooth top-k location from
                    # becoming an accidental optimisation target, while the global
                    # and local classification forwards remain fully trainable.
                    with torch.no_grad():
                        _, _, patch_attention = logits_with_last_block_attention(
                            model, original_inputs
                        )
                        local_inputs = attention_guided_crop(
                            original_inputs,
                            patch_attention.detach(),
                            crop_size=int(
                                attention_local_config.get("crop_size", 160)
                            ),
                            top_patches=int(
                                attention_local_config.get("top_patches", 5)
                            ),
                        )
                    local_logits = model(images=local_inputs)
                    local_training_logits = class_prior_adjusted_logits(
                        local_logits, class_counts, prior_tau
                    )
                    local_per_sample = _per_sample_loss(
                        local_training_logits,
                        targets,
                        config["loss"],
                        epoch,
                        batch_suspicious,
                    )
                    local_classification_value = (
                        local_per_sample * weights
                    ).sum() / weights.sum().clamp_min(1.0e-8)
                    local_weight = float(
                        attention_local_config.get(
                            "local_supervision_weight", 0.5
                        )
                    )
                    classification_loss = (
                        (1.0 - local_weight) * classification_loss
                        + local_weight * local_classification_value
                    )
                    temperature = float(
                        attention_local_config.get("temperature", 1.0)
                    )
                    teacher_probability = F.softmax(
                        training_logits.detach().float() / temperature,
                        dim=1,
                    )
                    local_log_probability = F.log_softmax(
                        local_training_logits.float() / temperature,
                        dim=1,
                    )
                    consistency_per_sample = F.kl_div(
                        local_log_probability,
                        teacher_probability,
                        reduction="none",
                    ).sum(dim=1) * (temperature * temperature)
                    local_consistency_value = (
                        consistency_per_sample * weights
                    ).sum() / weights.sum().clamp_min(1.0e-8)
                    local_agreement_value = (
                        local_logits.detach()
                        .argmax(dim=1)
                        .eq(logits.detach().argmax(dim=1))
                        .float()
                        .mean()
                    )
                    loss = classification_loss + float(
                        attention_local_config.get("consistency_weight", 0.25)
                    ) * local_consistency_value
                fine_active_value = logits.new_zeros(())
                fine_negative_value = logits.new_zeros(())
                fine_suspicious_count = 0
                if active_forgetting_enabled and epoch >= int(
                    active_forgetting_config.get("start_epoch", 1)
                ):
                    fine_suspicious = clean <= float(
                        active_forgetting_config.get(
                            "maximum_clean_probability", 0.05
                        )
                    )
                    (
                        fine_active_value,
                        fine_negative_value,
                        fine_suspicious_count,
                    ) = active_forgetting_noise_suppression_losses(
                        training_logits,
                        labels,
                        fine_suspicious,
                        batch_indices,
                        epoch=epoch,
                        epsilon=float(config["loss"].get("epsilon", 1.0e-7)),
                    )
                    loss = (
                        loss
                        + float(
                            active_forgetting_config.get(
                                "unlearning_weight", 0.001
                            )
                        )
                        * fine_active_value
                        + float(
                            active_forgetting_config.get(
                                "negative_learning_weight", 0.1
                            )
                        )
                        * fine_negative_value
                    )
                contrastive_value = logits.new_zeros(())
                contrastive_config = config["loss"].get("contrastive", {})
                if contrastive_config.get("enabled", False):
                    if input_key != "features" or mix_lambda != 1.0:
                        raise ValueError(
                            "Current contrastive gate requires unmixed cached features"
                        )
                    feature_noise = float(
                        contrastive_config.get("feature_noise_std", 0.01)
                    )
                    stochastic_features = F.normalize(
                        original_inputs.float()
                        + feature_noise * torch.randn_like(original_inputs.float()),
                        dim=1,
                    )
                    _, stochastic_encoded = model(
                        features=stochastic_features, return_features=True
                    )
                    trusted_for_class = clean >= float(
                        contrastive_config.get("trusted_threshold", 0.70)
                    )
                    trusted_for_class = trusted_for_class | (
                        (correction_evidence > 0.0)
                        & (
                            pseudo_confidence
                            >= float(
                                contrastive_config.get(
                                    "pseudo_confidence_threshold", 0.90
                                )
                            )
                        )
                    )
                    contrastive_value = noise_tolerant_supervised_contrastive_loss(
                        encoded,
                        stochastic_encoded,
                        mixed_targets.argmax(dim=1),
                        trusted_for_class,
                        temperature=float(
                            contrastive_config.get("temperature", 0.10)
                        ),
                    )
                    loss = loss + float(
                        contrastive_config.get("weight", 1.0)
                    ) * contrastive_value
                snscl_contrastive_value = logits.new_zeros(())
                snscl_kl_value = logits.new_zeros(())
                if snscl_state is not None:
                    corrected_anchor_labels, snscl_admission_probabilities = (
                        snscl_state.corrected_labels(
                            indices=batch_indices,
                            noisy_labels=labels,
                            reliability=clean,
                            logits=logits,
                            reliability_threshold=float(
                                snscl_config.get("reliability_threshold", 0.5)
                            ),
                            moving_average=float(
                                snscl_config.get("label_moving_average", 0.99)
                            ),
                        )
                    )
                    snscl_hard_labels = corrected_anchor_labels.argmax(dim=1)
                    (
                        snscl_projected,
                        _,
                        _,
                        snscl_kl_value,
                    ) = snscl_state.embedding(encoded)
                    queue_features, queue_labels = snscl_state.queue.snapshot()
                    (
                        snscl_contrastive_value,
                        snscl_usable_anchors,
                    ) = classwise_queue_contrastive_loss(
                        snscl_projected,
                        snscl_hard_labels,
                        queue_features,
                        queue_labels,
                        temperature=float(snscl_config.get("temperature", 0.07)),
                    )
                    loss = (
                        loss
                        + float(snscl_config.get("contrastive_weight", 1.0))
                        * snscl_contrastive_value
                        + float(snscl_config.get("kl_weight", 0.001))
                        * snscl_kl_value
                    )
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
                    if mixed_gate is not None:
                        gate_sum = mixed_gate.sum().clamp_min(1.0)
                        gated_drift = (drift * mixed_gate).sum() / gate_sum
                        loss = loss + distill_weight * gated_drift
                    else:
                        loss = loss + distill_weight * drift.mean()

                proto_loss = encoded.new_zeros(())
                if proto_enabled and epoch >= proto_start_epoch:
                    prototype_bank.update(encoded, labels, clean)
                    proto_loss = prototype_bank.loss(encoded, labels, clean)
                    loss = loss + proto_weight * proto_loss

            subspace_step = None
            subspace_uncertain_value = logits.new_zeros(())
            if trust_subspace is not None:
                trusted_mask = clean >= float(
                    subspace_config["trusted_threshold"]
                )
                uncertain_mask = ~trusted_mask
                trusted_count = int(trusted_mask.sum())
                uncertain_count = int(uncertain_mask.sum())
                denominator = float(labels.numel())
                trusted_float = trusted_mask.to(dtype=per_sample.dtype)
                uncertain_float = uncertain_mask.to(dtype=per_sample.dtype)
                trusted_classification = (
                    per_sample * trusted_float
                ).sum() / denominator
                subspace_uncertain_value = (
                    per_sample * uncertain_float
                ).sum() / denominator
                trusted_reference_loss = trusted_classification + distill_weight * (
                    drift * trusted_float
                ).sum() / denominator
                shared_loss = trusted_classification + distill_weight * drift.mean()
                # The displayed scalar is the shared T0 objective. The T1
                # uncertain contribution is a gradient projection and therefore
                # has no equivalent scalar objective; it is logged separately.
                classification_loss = trusted_classification
                loss = shared_loss
                minimum_trusted = int(
                    subspace_config.get("minimum_trusted_samples", 8)
                )
                minimum_uncertain = int(
                    subspace_config.get("minimum_uncertain_samples", 8)
                )
                if trusted_count >= minimum_trusted:
                    treatment_mode = (
                        str(subspace_config["mode"]) == "project_uncertain"
                    )
                    include_uncertain = (
                        treatment_mode and uncertain_count >= minimum_uncertain
                    )
                    if treatment_mode and not include_uncertain:
                        totals["trust_subspace_skipped_steps"] += 1
                    subspace_step = construct_trust_subspace_gradient(
                        parameters=model.parameters(),
                        trusted_reference_loss=trusted_reference_loss,
                        shared_loss=shared_loss,
                        uncertain_loss=(
                            subspace_uncertain_value if include_uncertain else None
                        ),
                        scaler=scaler,
                        subspace=trust_subspace,
                        update_basis=(
                            global_step
                            % int(subspace_config.get("update_interval", 1))
                            == 0
                        ),
                        include_uncertain=include_uncertain,
                    )
                    totals["trust_subspace_steps"] += 1
                    totals["trust_subspace_basis_updates"] += int(
                        subspace_step.basis_updated
                    )
                    totals["trust_subspace_projection_steps"] += int(
                        subspace_step.projection_applied
                    )
                    totals["trust_subspace_trusted_examples"] += trusted_count
                    totals["trust_subspace_uncertain_examples"] += uncertain_count
                    totals["trust_subspace_reference_gradient_norm"] += (
                        subspace_step.reference_gradient_norm
                    )
                    totals["trust_subspace_shared_gradient_norm"] += (
                        subspace_step.shared_gradient_norm
                    )
                    totals["trust_subspace_uncertain_gradient_norm"] += (
                        subspace_step.uncertain_gradient_norm
                    )
                    totals["trust_subspace_projected_gradient_norm"] += (
                        subspace_step.projected_uncertain_gradient_norm
                    )
                    totals["trust_subspace_retained_norm_ratio"] += (
                        subspace_step.retained_uncertain_norm_ratio
                    )
                else:
                    # Fail closed for the uncertain branch when a shuffled batch
                    # cannot construct a sufficiently supported trusted anchor.
                    scaler.scale(shared_loss).backward()
                    totals["trust_subspace_skipped_steps"] += 1
                totals["trust_subspace_uncertain_loss"] += float(
                    subspace_uncertain_value.detach()
                ) * labels.numel()
            else:
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            head_grad = _gradient_norm(model.classifier.parameters())
            adapter_grad = _gradient_norm(model.feature_adapter.parameters())
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
            parameters_to_clip = list(model.parameters())
            if snscl_state is not None:
                parameters_to_clip.extend(snscl_state.parameters())
            torch.nn.utils.clip_grad_norm_(parameters_to_clip, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            snscl_admitted = 0
            if snscl_state is not None:
                if (
                    snscl_projected is None
                    or snscl_hard_labels is None
                    or snscl_admission_probabilities is None
                ):
                    raise RuntimeError("SNSCL batch state was not constructed")
                snscl_admitted = snscl_state.queue.enqueue(
                    snscl_projected,
                    snscl_hard_labels,
                    snscl_admission_probabilities,
                )

            if not first_step_audited:
                _audit_first_step(
                    model=model,
                    before=before or {},
                    head_grad=head_grad,
                    adapter_grad=adapter_grad,
                    visual_grad=visual_grad,
                )
                first_step_audited = True
                logger.info(
                    "First-step audit passed: head_grad=%.6f adapter_grad=%.6f visual_grad=%.6f",
                    head_grad,
                    adapter_grad,
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
            totals["contrastive_loss"] += (
                float(contrastive_value.detach()) * batch_size
            )
            totals["snscl_contrastive_loss"] += (
                float(snscl_contrastive_value.detach()) * batch_size
            )
            totals["snscl_kl_loss"] += (
                float(snscl_kl_value.detach()) * batch_size
            )
            totals["snscl_usable_anchors"] += int(snscl_usable_anchors)
            totals["snscl_admitted"] += int(snscl_admitted)
            totals["fine_active_forgetting"] += (
                float(fine_active_value.detach()) * fine_suspicious_count
            )
            totals["fine_negative_learning"] += (
                float(fine_negative_value.detach()) * fine_suspicious_count
            )
            totals["fine_suspicious_samples"] += int(fine_suspicious_count)
            if attention_local_enabled:
                totals["attention_local_classification"] += (
                    float(local_classification_value.detach()) * batch_size
                )
                totals["attention_local_consistency"] += (
                    float(local_consistency_value.detach()) * batch_size
                )
                totals["attention_local_agreement"] += (
                    float(local_agreement_value.detach()) * batch_size
                )
                totals["attention_local_examples"] += batch_size
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
            "train_consensus_conflict_dropped": int(
                totals["consensus_conflict_dropped"]
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
            "train_contrastive_loss": (
                totals["contrastive_loss"] / totals["examples"]
            ),
            "train_snscl_contrastive_loss": (
                totals["snscl_contrastive_loss"] / totals["examples"]
            ),
            "train_snscl_kl_loss": (
                totals["snscl_kl_loss"] / totals["examples"]
            ),
            "train_snscl_usable_anchor_fraction": (
                totals["snscl_usable_anchors"] / totals["examples"]
            ),
            "train_snscl_admitted": int(totals["snscl_admitted"]),
            "train_snscl_queue_valid": (
                snscl_state.queue.valid_count if snscl_state is not None else 0
            ),
            "train_fine_active_forgetting": (
                totals["fine_active_forgetting"]
                / max(1, totals["fine_suspicious_samples"])
            ),
            "train_fine_negative_learning": (
                totals["fine_negative_learning"]
                / max(1, totals["fine_suspicious_samples"])
            ),
            "train_fine_suspicious_samples": int(
                totals["fine_suspicious_samples"]
            ),
            "train_attention_local_classification": (
                totals["attention_local_classification"]
                / max(1, totals["attention_local_examples"])
            ),
            "train_attention_local_consistency": (
                totals["attention_local_consistency"]
                / max(1, totals["attention_local_examples"])
            ),
            "train_attention_local_agreement": (
                totals["attention_local_agreement"]
                / max(1, totals["attention_local_examples"])
            ),
            "train_trust_subspace_steps": int(totals["trust_subspace_steps"]),
            "train_trust_subspace_skipped_steps": int(
                totals["trust_subspace_skipped_steps"]
            ),
            "train_trust_subspace_basis_updates": int(
                totals["trust_subspace_basis_updates"]
            ),
            "train_trust_subspace_basis_rank": (
                trust_subspace.rank if trust_subspace is not None else 0
            ),
            "train_trust_subspace_projection_steps": int(
                totals["trust_subspace_projection_steps"]
            ),
            "train_trust_subspace_trusted_examples": int(
                totals["trust_subspace_trusted_examples"]
            ),
            "train_trust_subspace_uncertain_examples": int(
                totals["trust_subspace_uncertain_examples"]
            ),
            "train_trust_subspace_reference_gradient_norm": (
                totals["trust_subspace_reference_gradient_norm"]
                / max(1, totals["trust_subspace_steps"])
            ),
            "train_trust_subspace_shared_gradient_norm": (
                totals["trust_subspace_shared_gradient_norm"]
                / max(1, totals["trust_subspace_steps"])
            ),
            "train_trust_subspace_uncertain_gradient_norm": (
                totals["trust_subspace_uncertain_gradient_norm"]
                / max(1, totals["trust_subspace_steps"])
            ),
            "train_trust_subspace_projected_gradient_norm": (
                totals["trust_subspace_projected_gradient_norm"]
                / max(1, totals["trust_subspace_steps"])
            ),
            "train_trust_subspace_retained_norm_ratio": (
                totals["trust_subspace_retained_norm_ratio"]
                / max(1, totals["trust_subspace_steps"])
            ),
            "train_trust_subspace_uncertain_loss": (
                totals["trust_subspace_uncertain_loss"] / totals["examples"]
            ),
            "train_cyclic_filter_selected": int(cyclic_filter_mask.sum()),
            "train_cyclic_filter_active": int(active_cyclic_filter.sum()),
            "train_cyclic_delta": float(cyclic_delta),
            "train_cyclic_reintroduction": int(reintroduction_epoch),
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

        if cyclic_enabled and reintroduction_epoch and epoch < epochs:
            cyclic_filter_mask = _refresh_cyclic_filter(
                model=model,
                train_dataset=train_dataset,
                device=device,
                use_amp=use_amp,
                loss_config=config["loss"],
                epoch=epoch + 1,
                clean_scores=train_clean_scores,
                cyclic_config=cyclic_config,
                num_classes=num_classes,
            )
            logger.info(
                "Refreshed cyclic filter after epoch %d: selected=%d",
                epoch,
                int(cyclic_filter_mask.sum()),
            )

        selector = float(val_metrics["selector"])
        selection_policy = str(
            evaluation_config.get("selection_policy", "best_selector")
        )
        improved = _checkpoint_is_selected(
            selection_policy=selection_policy,
            selector=selector,
            best_selector=best_selector,
        )
        if improved:
            best_selector = (
                selector if selection_policy == "best_selector" else float(epoch)
            )
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
            training_aux_state=(
                training_auxiliary.state_dict()
                if training_auxiliary is not None
                else None
            ),
        )
        save_checkpoint(checkpoint_dir / "last.pt", **common)
        save_checkpoint(checkpoint_dir / f"epoch_{epoch}.pt", **common)
        if improved:
            save_checkpoint(checkpoint_dir / "best.pt", **common)
            logger.info(
                "Selected checkpoint at epoch %d (policy=%s, selector=%.6f)",
                epoch,
                selection_policy,
                selector,
            )
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


@torch.no_grad()
def _refresh_cyclic_filter(
    *,
    model: AegisCLIP,
    train_dataset: torch.utils.data.Dataset,
    device: torch.device,
    use_amp: bool,
    loss_config: dict[str, Any],
    epoch: int,
    clean_scores: torch.Tensor,
    cyclic_config: dict[str, Any],
    num_classes: int,
) -> torch.Tensor:
    """Rank deterministic, unmixed training losses and refresh the curriculum."""
    was_training = model.training
    model.eval()
    loader = DataLoader(
        train_dataset,
        batch_size=int(cyclic_config.get("scoring_batch_size", 1024)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    losses = torch.empty(len(train_dataset), dtype=torch.float32)
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device).long()
        correction = batch["correction_alpha"].to(device).float()
        targets = corrected_targets(
            labels,
            batch["pseudo_label"].to(device).long(),
            correction,
            num_classes,
        )
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(features=features)
            per_sample = _per_sample_loss(
                logits, targets, loss_config, epoch, suspicious_mask=None
            )
        losses[batch["index"].long()] = per_sample.detach().float().cpu()
    if was_training:
        model.train()
    protection_threshold = float(
        cyclic_config.get("protect_clean_threshold", 0.999)
    )
    eligible = (clean_scores > 0.0) & (clean_scores < protection_threshold)
    return classwise_high_loss_filter(
        losses,
        torch.as_tensor(train_dataset.labels, dtype=torch.long),
        eligible,
        remove_fraction=float(cyclic_config["remove_fraction"]),
        maximum_class_fraction=float(cyclic_config["maximum_class_fraction"]),
        minimum_kept_per_class=int(cyclic_config["minimum_kept_per_class"]),
    )


def _checkpoint_is_selected(
    *,
    selection_policy: str,
    selector: float,
    best_selector: float,
) -> bool:
    """Apply an explicit checkpoint policy without consulting test results.

    ``last_epoch`` is intended for a full-data replay whose epoch count was
    fixed by a preceding development run.  Its overlapping validation metrics
    remain diagnostic only and cannot silently choose the submitted epoch.
    """

    if selection_policy == "last_epoch":
        return True
    if selection_policy == "best_selector":
        return float(selector) > float(best_selector)
    raise ValueError(f"Unsupported selection policy: {selection_policy}")


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


def _validate_train_val_overlap(
    train_paths: list[str],
    val_paths: list[str],
    *,
    allow_overlap: bool,
) -> None:
    """Fail closed when a development selector can see its training samples."""
    train_keys = {canonical_sample_path(path) for path in train_paths}
    val_keys = {canonical_sample_path(path) for path in val_paths}
    overlap = train_keys & val_keys
    if not overlap:
        return
    if not allow_overlap:
        first = min(overlap)
        raise ValueError(
            f"Train/validation path overlap: {len(overlap)}; first={first}"
        )
    missing = val_keys - train_keys
    if missing:
        raise ValueError(
            "Overlapping diagnostic validation must be a subset of training: "
            f"missing={len(missing)}"
        )


def _training_preprocess(
    preprocess: Any,
    data_config: dict[str, Any],
    input_resolution: int = 224,
) -> Any:
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
                int(input_resolution),
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
    if loss_config["name"] == "double_softmax_cross_entropy":
        return double_softmax_cross_entropy(logits, targets)
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
    adapter_grad: float,
    visual_grad: float,
) -> None:
    if not math.isfinite(head_grad) or head_grad <= 0.0:
        raise RuntimeError(f"Invalid first-step head gradient: {head_grad}")
    if model.visual_requires_grad and (
        not math.isfinite(visual_grad) or visual_grad <= 0.0
    ):
        raise RuntimeError(f"PEFT configured but visual gradient is {visual_grad}")
    if model.peft_mode == "feature_adapter" and (
        not math.isfinite(adapter_grad) or adapter_grad <= 0.0
    ):
        raise RuntimeError(f"Invalid first-step adapter gradient: {adapter_grad}")
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
    if peft_mode in {
        "visual_ln",
        "ln_post_proj",
        "visual_lora",
        "visual_mlp_adapter",
        "visual_prompt",
    } and not spec[
        "visual_requires_grad"
    ]:
        raise RuntimeError("Visual PEFT mode has no trainable visual parameters")
    if peft_mode == "visual_lora" and not any(
        "parametrizations" in name for name in spec["trainable_names"]
    ):
        raise RuntimeError("Visual LoRA mode has no trainable low-rank parameters")
    if peft_mode == "visual_mlp_adapter" and not any(
        ".adaptmlp." in name for name in spec["trainable_names"]
    ):
        raise RuntimeError("Visual MLP adapter mode has no trainable adapters")
    if peft_mode == "visual_prompt" and not any(
        ".visual_prompt." in name for name in spec["trainable_names"]
    ):
        raise RuntimeError("Visual prompt mode has no trainable prompt tokens")
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
                "train_consensus_conflict_dropped",
                "projection_count",
                "train_elr_regularization",
                "train_elr_weight",
                "train_snscl_contrastive_loss",
                "train_snscl_kl_loss",
                "train_snscl_usable_anchor_fraction",
                "train_snscl_admitted",
                "train_snscl_queue_valid",
                "train_fine_active_forgetting",
                "train_fine_negative_learning",
                "train_fine_suspicious_samples",
                "train_attention_local_classification",
                "train_attention_local_consistency",
                "train_attention_local_agreement",
                "train_trust_subspace_steps",
                "train_trust_subspace_skipped_steps",
                "train_trust_subspace_basis_updates",
                "train_trust_subspace_basis_rank",
                "train_trust_subspace_projection_steps",
                "train_trust_subspace_trusted_examples",
                "train_trust_subspace_uncertain_examples",
                "train_trust_subspace_reference_gradient_norm",
                "train_trust_subspace_shared_gradient_norm",
                "train_trust_subspace_uncertain_gradient_norm",
                "train_trust_subspace_projected_gradient_norm",
                "train_trust_subspace_retained_norm_ratio",
                "train_trust_subspace_uncertain_loss",
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
                "snscl_lr",
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
                metrics["train_consensus_conflict_dropped"],
                metrics["projection_count"],
                metrics["train_elr_regularization"],
                metrics["train_elr_weight"],
                metrics["train_snscl_contrastive_loss"],
                metrics["train_snscl_kl_loss"],
                metrics["train_snscl_usable_anchor_fraction"],
                metrics["train_snscl_admitted"],
                metrics["train_snscl_queue_valid"],
                metrics["train_fine_active_forgetting"],
                metrics["train_fine_negative_learning"],
                metrics["train_fine_suspicious_samples"],
                metrics["train_attention_local_classification"],
                metrics["train_attention_local_consistency"],
                metrics["train_attention_local_agreement"],
                metrics["train_trust_subspace_steps"],
                metrics["train_trust_subspace_skipped_steps"],
                metrics["train_trust_subspace_basis_updates"],
                metrics["train_trust_subspace_basis_rank"],
                metrics["train_trust_subspace_projection_steps"],
                metrics["train_trust_subspace_trusted_examples"],
                metrics["train_trust_subspace_uncertain_examples"],
                metrics["train_trust_subspace_reference_gradient_norm"],
                metrics["train_trust_subspace_shared_gradient_norm"],
                metrics["train_trust_subspace_uncertain_gradient_norm"],
                metrics["train_trust_subspace_projected_gradient_norm"],
                metrics["train_trust_subspace_retained_norm_ratio"],
                metrics["train_trust_subspace_uncertain_loss"],
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
                learning_rates.get("snscl", ""),
            ]
        )
