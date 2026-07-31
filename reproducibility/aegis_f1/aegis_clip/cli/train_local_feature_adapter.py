"""Train O3's local-only bottleneck while leaving F1 bit-exact globally."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aegis_clip.balanced_inference import prediction_metrics
from aegis_clip.checkpoint import _atomic_torch_save
from aegis_clip.local_feature_adapter import (
    BottleneckLocalFeatureAdapter,
    fuse_global_local_log_probabilities,
    validate_local_adapter_cache,
)
from aegis_clip.runtime import atomic_json_dump, set_seed, sha256_file


def _load_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_local_adapter_cache(
        payload, expected_feature_dim=512, expected_num_classes=500
    )
    return payload


def _classifier_from_checkpoint(
    checkpoint_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("Parent checkpoint is incomplete")
    if config.get("model", {}).get("classifier_mode", "linear") != "linear":
        raise ValueError("O3 requires the shared ordinary linear classifier")
    try:
        weight = torch.as_tensor(state["classifier.weight"]).float().cpu()
        bias = torch.as_tensor(state["classifier.bias"]).float().cpu()
    except KeyError as exc:
        raise ValueError("Parent checkpoint lacks the linear classifier") from exc
    if tuple(weight.shape) != (500, 512) or tuple(bias.shape) != (500,):
        raise ValueError("Parent classifier dimensions are unexpected")
    return weight, bias, checkpoint


def _reference_audit(
    cache: dict[str, Any],
    reference_path: str | Path,
    *,
    fused: bool,
) -> dict[str, float]:
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    if list(reference["paths"]) != list(cache["paths"]):
        raise ValueError("Reference path order does not match the O3 validation cache")
    cached_logits = (
        fuse_global_local_log_probabilities(
            torch.as_tensor(cache["global_logits"]),
            torch.as_tensor(cache["local_logits"]),
        )
        if fused
        else torch.as_tensor(cache["global_logits"]).float()
    )
    reference_logits = torch.as_tensor(reference["logits"]).float()
    difference = (reference_logits - cached_logits).abs()
    agreement = (reference_logits.argmax(1) == cached_logits.argmax(1)).float().mean()
    return {
        "maximum_absolute_logit_difference": float(difference.max()),
        "prediction_agreement": float(agreement),
    }


def _prediction_metrics(
    predictions: torch.Tensor, cache: dict[str, Any]
) -> dict[str, float | int]:
    return prediction_metrics(
        predictions,
        labels=cache["labels"],
        clean_probability=cache["clean_probability"],
        pseudo_labels=cache["pseudo_labels"],
        correction_alpha=cache["correction_alpha"],
        num_classes=500,
        clean_core_threshold=0.70,
    )


def _evaluate(
    adapter: BottleneckLocalFeatureAdapter | None,
    cache: dict[str, Any],
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 4096,
) -> tuple[dict[str, float | int], torch.Tensor]:
    global_values = torch.as_tensor(cache["global_logits"]).float()
    cached_local_logits = torch.as_tensor(cache["local_logits"]).float()
    local_features = torch.as_tensor(cache["local_features"]).float()
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
            if adapter is None:
                local_logits = cached_local_logits[start:stop].to(device)
                adapted_features = base_features
            else:
                adapted_features = adapter(base_features)
                local_logits = F.linear(
                    adapted_features,
                    classifier_weight.to(device),
                    classifier_bias.to(device),
                )
            fused = fuse_global_local_log_probabilities(global_logits, local_logits)
            fused_parts.append(fused.cpu())
            local_prediction_parts.append(local_logits.argmax(1).cpu())
            drift_sum += float(
                (1.0 - F.cosine_similarity(adapted_features, base_features, dim=1)).sum()
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
    return metrics, fused_logits


def _gce_from_log_probabilities(
    log_probabilities: torch.Tensor, labels: torch.Tensor, q: float
) -> torch.Tensor:
    probability = log_probabilities.gather(1, labels[:, None]).exp().squeeze(1)
    return ((1.0 - probability.pow(float(q))) / float(q)).mean()


def _delta_pp(
    candidate: dict[str, float | int], baseline: dict[str, float | int], name: str
) -> float:
    return 100.0 * (float(candidate[name]) - float(baseline[name]))


def train_local_feature_adapter(
    parent_checkpoint_path: str | Path,
    train_cache_path: str | Path,
    validation_cache_path: str | Path,
    output_dir: str | Path,
    *,
    center_reference_path: str | Path,
    m1_reference_path: str | Path,
    expected_train_samples: int,
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
        raise ValueError("O3 high-clean training cache count is unexpected")
    if float(torch.as_tensor(train_cache["clean_probability"]).min()) < 0.70:
        raise ValueError("O3 training cache contains a sample below clean threshold")
    if set(train_cache["paths"]) & set(validation_cache["paths"]):
        raise ValueError("O3 train and validation caches overlap")
    parent_sha256 = sha256_file(parent_checkpoint_path)
    if {
        str(train_cache["checkpoint_sha256"]),
        str(validation_cache["checkpoint_sha256"]),
    } != {parent_sha256}:
        raise ValueError("O3 caches do not match the parent checkpoint")
    classifier_weight, classifier_bias, parent_checkpoint = (
        _classifier_from_checkpoint(parent_checkpoint_path)
    )
    center_audit = _reference_audit(
        validation_cache, center_reference_path, fused=False
    )
    m1_audit = _reference_audit(validation_cache, m1_reference_path, fused=True)
    exact_audit = {
        "maximum_absolute_logit_difference": 0.0,
        "prediction_agreement": 1.0,
    }
    if center_audit != exact_audit or m1_audit != exact_audit:
        raise ValueError(
            f"O3 cache reference audit failed: center={center_audit}, m1={m1_audit}"
        )

    set_seed(int(seed), deterministic=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    adapter = BottleneckLocalFeatureAdapter(
        512,
        int(bottleneck_dim),
        residual_scale=float(residual_scale),
        dropout=float(dropout),
    ).to(device)
    classifier_weight = classifier_weight.to(device)
    classifier_bias = classifier_bias.to(device)
    baseline_metrics, baseline_logits = _evaluate(
        None,
        validation_cache,
        classifier_weight,
        classifier_bias,
        device,
    )
    epoch_zero_metrics, epoch_zero_logits = _evaluate(
        adapter,
        validation_cache,
        classifier_weight,
        classifier_bias,
        device,
    )
    epoch_zero_difference = (baseline_logits - epoch_zero_logits).abs()
    epoch_zero_audit = {
        "maximum_absolute_logit_difference": float(epoch_zero_difference.max()),
        "prediction_agreement": float(
            (baseline_logits.argmax(1) == epoch_zero_logits.argmax(1)).float().mean()
        ),
    }
    if epoch_zero_audit["prediction_agreement"] != 1.0:
        raise RuntimeError(f"O3 zero-initialisation audit failed: {epoch_zero_audit}")

    dataset = TensorDataset(
        torch.as_tensor(train_cache["local_features"]).float(),
        torch.as_tensor(train_cache["global_logits"]).float(),
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
        optimizer, T_max=int(max_epochs), eta_min=0.0
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
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "gce_q": float(gce_q),
        "local_loss_weight": float(local_loss_weight),
        "feature_anchor_weight": float(feature_anchor_weight),
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
    stale_epochs = 0
    for epoch in range(1, int(max_epochs) + 1):
        adapter.train()
        fused_loss_sum = 0.0
        local_loss_sum = 0.0
        anchor_loss_sum = 0.0
        samples = 0
        for local_features, global_logits, labels in loader:
            local_features = local_features.to(device, non_blocking=True)
            global_logits = global_logits.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            adapted = adapter(local_features)
            local_logits = F.linear(adapted, classifier_weight, classifier_bias)
            fused = fuse_global_local_log_probabilities(
                global_logits, local_logits
            )
            fused_loss = _gce_from_log_probabilities(fused, labels, float(gce_q))
            local_log_probability = F.log_softmax(local_logits.float(), dim=1)
            local_loss = _gce_from_log_probabilities(
                local_log_probability, labels, float(gce_q)
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
        metrics, _ = _evaluate(
            adapter,
            validation_cache,
            classifier_weight,
            classifier_bias,
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

    if best_record is None or best_state is None:
        best_record = history[0]
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in BottleneckLocalFeatureAdapter(
                512,
                int(bottleneck_dim),
                residual_scale=float(residual_scale),
                dropout=float(dropout),
            ).state_dict().items()
        }
    best_metrics = best_record["metrics"]
    gate = {
        "clean_core_micro_delta_pp": _delta_pp(
            best_metrics, baseline_metrics, "clean_core_micro"
        ),
        "trusted_macro_delta_pp": _delta_pp(
            best_metrics, baseline_metrics, "trusted_macro"
        ),
        "raw_micro_delta_pp": _delta_pp(
            best_metrics, baseline_metrics, "raw_micro"
        ),
        "local_feature_drift": float(best_metrics["local_feature_drift"]),
        "global_path_bit_exact": center_audit == exact_audit,
        "epoch_zero_m1_bit_exact": m1_audit == exact_audit,
        "passed": bool(
            _delta_pp(best_metrics, baseline_metrics, "clean_core_micro") >= 0.20
            and _delta_pp(best_metrics, baseline_metrics, "trusted_macro") >= 0.0
            and _delta_pp(best_metrics, baseline_metrics, "raw_micro") >= -0.10
            and float(best_metrics["local_feature_drift"]) <= 0.01
            and int(best_metrics["prediction_empty_classes"]) == 0
            and center_audit == exact_audit
            and m1_audit == exact_audit
        ),
    }
    adapter_payload = {
        "format_version": 1,
        "experiment": "O3_F1_LOCAL_ONLY_FEATURE_ADAPTER",
        "state_dict": best_state,
        "spec": {
            "feature_dim": 512,
            "bottleneck_dim": int(bottleneck_dim),
            "residual_scale": float(residual_scale),
            "dropout": float(dropout),
            "fusion": "1:1_global_adapted_local_probability_mean",
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
    composite["local_feature_adapter"] = adapter_payload
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
        train_local_feature_adapter(
            args.parent_checkpoint,
            args.train_cache,
            args.validation_cache,
            args.output_dir,
            center_reference_path=args.center_reference,
            m1_reference_path=args.m1_reference,
            expected_train_samples=args.expected_train_samples,
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
