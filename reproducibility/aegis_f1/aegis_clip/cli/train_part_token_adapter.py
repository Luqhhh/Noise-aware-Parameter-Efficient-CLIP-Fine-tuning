"""Train R1's part-token residual while preserving F1+M1 at epoch zero."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aegis_clip.checkpoint import _atomic_torch_save
from aegis_clip.cli.train_local_feature_adapter import (
    _classifier_from_checkpoint,
    _delta_pp,
    _gce_from_log_probabilities,
    _prediction_metrics,
    _reference_audit,
)
from aegis_clip.local_feature_adapter import fuse_global_local_log_probabilities
from aegis_clip.part_token_adapter import (
    PartTokenResidualAdapter,
    anchored_classifier_residual_logits,
    validate_part_token_cache,
)
from aegis_clip.runtime import atomic_json_dump, set_seed, sha256_file


def _load_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_part_token_cache(
        payload,
        expected_feature_dim=512,
        expected_num_classes=500,
    )
    return payload


def _evaluate(
    adapter: PartTokenResidualAdapter | None,
    cache: dict[str, Any],
    classifier_weight: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 4096,
) -> tuple[dict[str, float | int], torch.Tensor]:
    global_values = torch.as_tensor(cache["global_logits"]).float()
    cached_local_logits = torch.as_tensor(cache["local_logits"]).float()
    local_features = torch.as_tensor(cache["local_features"]).float()
    part_features = torch.as_tensor(cache["part_features"]).float()
    fused_parts: list[torch.Tensor] = []
    local_prediction_parts: list[torch.Tensor] = []
    drift_sum = 0.0
    norm_drift_sum = 0.0
    agreement_sum = 0
    if adapter is not None:
        adapter.eval()
    with torch.no_grad():
        for start in range(0, len(local_features), int(batch_size)):
            stop = min(start + int(batch_size), len(local_features))
            global_logits = global_values[start:stop].to(device)
            base_features = local_features[start:stop].to(device)
            part_values = part_features[start:stop].to(device)
            if adapter is None:
                local_logits = cached_local_logits[start:stop].to(device)
                adapted_features = base_features
            else:
                adapted_features = adapter(base_features, part_values)
                local_logits = anchored_classifier_residual_logits(
                    cached_local_logits[start:stop].to(device),
                    base_features,
                    adapted_features,
                    classifier_weight,
                )
            fused = fuse_global_local_log_probabilities(global_logits, local_logits)
            fused_parts.append(fused.cpu())
            local_prediction_parts.append(local_logits.argmax(1).cpu())
            drift_sum += float(
                (
                    1.0
                    - F.cosine_similarity(
                        adapted_features,
                        base_features,
                        dim=1,
                    )
                ).sum()
            )
            norm_drift_sum += float(
                (adapted_features.norm(dim=1) - base_features.norm(dim=1))
                .abs()
                .sum()
            )
            agreement_sum += int(
                (global_logits.argmax(1) == local_logits.argmax(1)).sum()
            )
    fused_logits = torch.cat(fused_parts)
    local_predictions = torch.cat(local_prediction_parts)
    metrics = _prediction_metrics(fused_logits.argmax(1), cache)
    local_metrics = _prediction_metrics(local_predictions, cache)
    metrics.update(
        {
            f"local_{name}": value
            for name, value in local_metrics.items()
            if name
            in {
                "clean_core_macro",
                "clean_core_micro",
                "trusted_macro",
                "trusted_micro",
                "raw_macro",
                "raw_micro",
            }
        }
    )
    metrics["local_feature_drift"] = drift_sum / max(len(local_features), 1)
    metrics["local_feature_norm_drift"] = norm_drift_sum / max(
        len(local_features), 1
    )
    metrics["local_global_prediction_agreement"] = agreement_sum / max(
        len(local_features), 1
    )
    metrics["part_local_cosine"] = float(
        F.cosine_similarity(part_features, local_features, dim=1).mean()
    )
    return metrics, fused_logits


def _paired_change_audit(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    cache: dict[str, Any],
) -> dict[str, float | int]:
    labels = torch.as_tensor(cache["labels"]).long()
    clean_mask = torch.as_tensor(cache["clean_probability"]).float() >= 0.70
    baseline_correct = baseline_logits.argmax(1).eq(labels) & clean_mask
    candidate_correct = candidate_logits.argmax(1).eq(labels) & clean_mask
    corrected = int((~baseline_correct & candidate_correct & clean_mask).sum())
    harmed = int((baseline_correct & ~candidate_correct & clean_mask).sum())
    return {
        "clean_core_corrected": corrected,
        "clean_core_harmed": harmed,
        "clean_core_net_corrections": corrected - harmed,
        "clean_core_corrected_to_harmed_ratio": float(
            (corrected + 1.0) / (harmed + 1.0)
        ),
    }


def train_part_token_adapter(
    parent_checkpoint_path: str | Path,
    train_cache_path: str | Path,
    validation_cache_path: str | Path,
    output_dir: str | Path,
    *,
    center_reference_path: str | Path,
    m1_reference_path: str | Path,
    expected_train_samples: int,
    expected_cache_batch_size: int = 128,
    seed: int = 42,
    bottleneck_dim: int = 32,
    residual_scale: float = 0.25,
    dropout: float = 0.1,
    learning_rate: float = 5.0e-4,
    weight_decay: float = 1.0e-4,
    batch_size: int = 1024,
    max_epochs: int = 20,
    patience: int = 5,
    gce_q: float = 0.5,
    local_loss_weight: float = 0.25,
    feature_anchor_weight: float = 2.0,
    device_name: str = "cpu",
) -> Path:
    parent_checkpoint_path = Path(parent_checkpoint_path).resolve()
    train_cache_path = Path(train_cache_path).resolve()
    validation_cache_path = Path(validation_cache_path).resolve()
    output_dir = Path(output_dir).resolve()
    train_cache = _load_cache(train_cache_path)
    validation_cache = _load_cache(validation_cache_path)
    if len(train_cache["paths"]) != int(expected_train_samples):
        raise ValueError("R1 high-clean training cache count is unexpected")
    if float(torch.as_tensor(train_cache["clean_probability"]).min()) < 0.70:
        raise ValueError("R1 training cache contains a sample below clean threshold")
    if set(train_cache["paths"]) & set(validation_cache["paths"]):
        raise ValueError("R1 train and validation caches overlap")
    if train_cache["part_pool_spec"] != validation_cache["part_pool_spec"]:
        raise ValueError("R1 train and validation part pooling specifications differ")
    for name, cache in {"train": train_cache, "validation": validation_cache}.items():
        execution = cache.get("execution")
        if not isinstance(execution, dict) or int(
            execution.get("batch_size", 0)
        ) != int(expected_cache_batch_size):
            raise ValueError(f"R1 {name} cache batch size is not preregistered")
    parent_sha256 = sha256_file(parent_checkpoint_path)
    if {
        str(train_cache["checkpoint_sha256"]),
        str(validation_cache["checkpoint_sha256"]),
    } != {parent_sha256}:
        raise ValueError("R1 caches do not match the parent checkpoint")
    classifier_weight, _, parent_checkpoint = (
        _classifier_from_checkpoint(parent_checkpoint_path)
    )
    center_audit = _reference_audit(
        validation_cache,
        center_reference_path,
        fused=False,
    )
    m1_audit = _reference_audit(
        validation_cache,
        m1_reference_path,
        fused=True,
    )
    reference_audit_passed = bool(
        center_audit["maximum_absolute_logit_difference"] == 0.0
        and center_audit["prediction_agreement"] == 1.0
        and m1_audit["maximum_absolute_logit_difference"] <= 4.0e-6
        and m1_audit["prediction_agreement"] == 1.0
    )
    if not reference_audit_passed:
        raise ValueError(
            f"R1 cache reference audit failed: center={center_audit}, m1={m1_audit}"
        )

    set_seed(int(seed), deterministic=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    adapter = PartTokenResidualAdapter(
        512,
        int(bottleneck_dim),
        residual_scale=float(residual_scale),
        dropout=float(dropout),
    ).to(device)
    epoch_zero_state = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
    }
    classifier_weight = classifier_weight.to(device)
    baseline_metrics, baseline_logits = _evaluate(
        None,
        validation_cache,
        classifier_weight,
        device,
    )
    epoch_zero_metrics, epoch_zero_logits = _evaluate(
        adapter,
        validation_cache,
        classifier_weight,
        device,
    )
    epoch_zero_difference = (baseline_logits - epoch_zero_logits).abs()
    epoch_zero_audit = {
        "maximum_absolute_logit_difference": float(epoch_zero_difference.max()),
        "prediction_agreement": float(
            (
                baseline_logits.argmax(1) == epoch_zero_logits.argmax(1)
            ).float().mean()
        ),
    }
    if epoch_zero_audit != {
        "maximum_absolute_logit_difference": 0.0,
        "prediction_agreement": 1.0,
    }:
        raise RuntimeError(f"R1 zero-initialisation audit failed: {epoch_zero_audit}")

    dataset = TensorDataset(
        torch.as_tensor(train_cache["local_features"]).float(),
        torch.as_tensor(train_cache["part_features"]).float(),
        torch.as_tensor(train_cache["global_logits"]).float(),
        torch.as_tensor(train_cache["local_logits"]).float(),
        torch.as_tensor(train_cache["labels"]).long(),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(max_epochs),
        eta_min=0.0,
    )
    protocol: dict[str, Any] = {
        "seed": int(seed),
        "device": str(device),
        "bottleneck_dim": int(bottleneck_dim),
        "residual_scale": float(residual_scale),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "cache_batch_size": int(expected_cache_batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "gce_q": float(gce_q),
        "local_loss_weight": float(local_loss_weight),
        "feature_anchor_weight": float(feature_anchor_weight),
        "part_pool_spec": copy.deepcopy(train_cache["part_pool_spec"]),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "selection": "best clean-core micro among safety-eligible epochs",
        "parameter_scan": False,
        "center_reference_audit": center_audit,
        "m1_reference_audit": m1_audit,
        "epoch_zero_audit": epoch_zero_audit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = [
        {"epoch": 0, "metrics": epoch_zero_metrics, "audit": epoch_zero_audit}
    ]
    best_selector = float(baseline_metrics["clean_core_micro"])
    best_record: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_logits: torch.Tensor | None = None
    stale_epochs = 0
    for epoch in range(1, int(max_epochs) + 1):
        adapter.train()
        fused_loss_sum = 0.0
        local_loss_sum = 0.0
        anchor_loss_sum = 0.0
        samples = 0
        for (
            local_features,
            part_features,
            global_logits,
            base_local_logits,
            labels,
        ) in loader:
            local_features = local_features.to(device, non_blocking=True)
            part_features = part_features.to(device, non_blocking=True)
            global_logits = global_logits.to(device, non_blocking=True)
            base_local_logits = base_local_logits.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            adapted = adapter(local_features, part_features)
            local_logits = anchored_classifier_residual_logits(
                base_local_logits,
                local_features,
                adapted,
                classifier_weight,
            )
            fused = fuse_global_local_log_probabilities(global_logits, local_logits)
            fused_loss = _gce_from_log_probabilities(fused, labels, float(gce_q))
            local_loss = _gce_from_log_probabilities(
                F.log_softmax(local_logits.float(), dim=1),
                labels,
                float(gce_q),
            )
            anchor_loss = (adapted - local_features).square().sum(dim=1).mean()
            loss = (
                fused_loss
                + float(local_loss_weight) * local_loss
                + float(feature_anchor_weight) * anchor_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            count = labels.numel()
            fused_loss_sum += float(fused_loss.detach()) * count
            local_loss_sum += float(local_loss.detach()) * count
            anchor_loss_sum += float(anchor_loss.detach()) * count
            samples += count
        scheduler.step()
        metrics, candidate_logits = _evaluate(
            adapter,
            validation_cache,
            classifier_weight,
            device,
        )
        finite = all(
            torch.isfinite(torch.tensor(float(value)))
            for value in metrics.values()
            if isinstance(value, (float, int))
        )
        safety_eligible = bool(
            finite
            and int(metrics["prediction_empty_classes"]) == 0
            and float(metrics["trusted_macro"])
            >= float(baseline_metrics["trusted_macro"])
            and float(metrics["raw_micro"])
            >= float(baseline_metrics["raw_micro"]) - 0.001
            and float(metrics["local_feature_drift"]) <= 0.01
        )
        record = {
            "epoch": epoch,
            "fused_train_loss": fused_loss_sum / max(samples, 1),
            "local_train_loss": local_loss_sum / max(samples, 1),
            "feature_anchor_loss": anchor_loss_sum / max(samples, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "residual_parameter_norm": adapter.residual_parameter_norm(),
            "safety_eligible": safety_eligible,
            "metrics": metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        selector = float(metrics["clean_core_micro"])
        if safety_eligible and selector > best_selector:
            best_selector = selector
            best_record = copy.deepcopy(record)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in adapter.state_dict().items()
            }
            best_logits = candidate_logits.clone()
            stale_epochs = 0
        else:
            stale_epochs += 1
        atomic_json_dump(
            {
                "protocol": protocol,
                "baseline": baseline_metrics,
                "best_selector": best_selector,
                "best_record": best_record,
                "history": history,
            },
            output_dir / "history.json",
        )
        if stale_epochs >= int(patience):
            break

    if best_record is None or best_state is None or best_logits is None:
        best_record = history[0]
        best_state = epoch_zero_state
        best_logits = baseline_logits
    best_metrics = best_record["metrics"]
    paired_audit = _paired_change_audit(
        baseline_logits,
        best_logits,
        validation_cache,
    )
    gate = {
        "clean_core_micro_delta_pp": _delta_pp(
            best_metrics,
            baseline_metrics,
            "clean_core_micro",
        ),
        "trusted_macro_delta_pp": _delta_pp(
            best_metrics,
            baseline_metrics,
            "trusted_macro",
        ),
        "raw_micro_delta_pp": _delta_pp(
            best_metrics,
            baseline_metrics,
            "raw_micro",
        ),
        "local_feature_drift": float(best_metrics["local_feature_drift"]),
        **paired_audit,
        "reference_audit_passed": reference_audit_passed,
        "global_path_bit_exact": (
            center_audit["maximum_absolute_logit_difference"] == 0.0
            and center_audit["prediction_agreement"] == 1.0
        ),
        "epoch_zero_m1_reproduced": (
            m1_audit["maximum_absolute_logit_difference"] <= 4.0e-6
            and m1_audit["prediction_agreement"] == 1.0
            and epoch_zero_audit["maximum_absolute_logit_difference"] == 0.0
            and epoch_zero_audit["prediction_agreement"] == 1.0
        ),
    }
    gate["passed"] = bool(
        gate["clean_core_micro_delta_pp"] >= 0.20
        and gate["trusted_macro_delta_pp"] >= 0.0
        and gate["raw_micro_delta_pp"] >= -0.10
        and gate["local_feature_drift"] <= 0.01
        and int(best_metrics["prediction_empty_classes"]) == 0
        and gate["reference_audit_passed"]
        and gate["global_path_bit_exact"]
        and gate["epoch_zero_m1_reproduced"]
    )
    adapter_payload = {
        "format_version": 1,
        "experiment": "R1_F1_M1_PART_TOKEN_RESIDUAL",
        "state_dict": best_state,
        "spec": {
            "feature_dim": 512,
            "bottleneck_dim": int(bottleneck_dim),
            "residual_scale": float(residual_scale),
            "dropout": float(dropout),
            "part_pool_spec": copy.deepcopy(train_cache["part_pool_spec"]),
            "fusion": "1:1_global_part_adapted_local_probability_mean",
            "shared_classifier": True,
        },
        "protocol": protocol,
        "baseline_metrics": baseline_metrics,
        "best_record": best_record,
        "gate": gate,
        "lineage": {
            "parent_checkpoint": str(parent_checkpoint_path),
            "parent_checkpoint_sha256": parent_sha256,
            "train_cache": str(train_cache_path),
            "train_cache_sha256": sha256_file(train_cache_path),
            "validation_cache": str(validation_cache_path),
            "validation_cache_sha256": sha256_file(validation_cache_path),
        },
    }
    _atomic_torch_save(adapter_payload, output_dir / "best_adapter.pt")
    composite = dict(parent_checkpoint)
    composite["part_token_adapter"] = adapter_payload
    _atomic_torch_save(composite, output_dir / "best.pt")
    atomic_json_dump(gate, output_dir / "gate.json")
    return output_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--center-reference", required=True)
    parser.add_argument("--m1-reference", required=True)
    parser.add_argument("--expected-train-samples", type=int, required=True)
    parser.add_argument("--expected-cache-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bottleneck-dim", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gce-q", type=float, default=0.5)
    parser.add_argument("--local-loss-weight", type=float, default=0.25)
    parser.add_argument("--feature-anchor-weight", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        train_part_token_adapter(
            args.parent_checkpoint,
            args.train_cache,
            args.validation_cache,
            args.output_dir,
            center_reference_path=args.center_reference,
            m1_reference_path=args.m1_reference,
            expected_train_samples=args.expected_train_samples,
            expected_cache_batch_size=args.expected_cache_batch_size,
            seed=args.seed,
            bottleneck_dim=args.bottleneck_dim,
            residual_scale=args.residual_scale,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            gce_q=args.gce_q,
            local_loss_weight=args.local_loss_weight,
            feature_anchor_weight=args.feature_anchor_weight,
            device_name=args.device,
        )
    )


if __name__ == "__main__":
    main()
