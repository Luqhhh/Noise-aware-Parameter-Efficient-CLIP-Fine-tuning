"""Train the preregistered N1 local linear residual on frozen feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aegis_clip.balanced_inference import prediction_metrics
from aegis_clip.checkpoint import _atomic_torch_save
from aegis_clip.local_residual import LearnedLocalResidualHead, validate_dual_view_cache
from aegis_clip.runtime import atomic_json_dump, set_seed, sha256_file


def _load_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_dual_view_cache(payload, expected_feature_dim=512, expected_num_classes=500)
    return payload


def _metrics(
    head: LearnedLocalResidualHead,
    cache: dict[str, Any],
    device: torch.device,
    *,
    batch_size: int = 4096,
) -> tuple[dict[str, float | int], torch.Tensor]:
    head.eval()
    predictions: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    local = torch.as_tensor(cache["local_features"])
    base = torch.as_tensor(cache["global_logits"])
    with torch.no_grad():
        for start in range(0, len(local), int(batch_size)):
            stop = min(start + int(batch_size), len(local))
            logits = head(
                base[start:stop].to(device, non_blocking=True),
                local[start:stop].to(device, non_blocking=True),
            )
            logits_parts.append(logits.cpu())
            predictions.append(logits.argmax(dim=1).cpu())
    prediction = torch.cat(predictions)
    metrics = prediction_metrics(
        prediction,
        labels=cache["labels"],
        clean_probability=cache["clean_probability"],
        pseudo_labels=cache["pseudo_labels"],
        correction_alpha=cache["correction_alpha"],
        num_classes=500,
        clean_core_threshold=0.70,
    )
    return metrics, torch.cat(logits_parts)


def _checkpoint_payload(
    head: LearnedLocalResidualHead,
    *,
    epoch: int,
    metrics: dict[str, float | int],
    protocol: dict[str, Any],
    train_cache_path: Path,
    validation_cache_path: Path,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "N1_LEARNED_LOCAL_RESIDUAL_HEAD",
        "epoch": int(epoch),
        "head_state_dict": {
            name: value.detach().cpu() for name, value in head.state_dict().items()
        },
        "head_spec": {
            "feature_dim": 512,
            "num_classes": 500,
            "dropout": 0.1,
            "fusion": "global_logits_plus_local_linear_residual",
        },
        "protocol": protocol,
        "metrics": metrics,
        "lineage": {
            "train_cache": str(train_cache_path),
            "train_cache_sha256": sha256_file(train_cache_path),
            "validation_cache": str(validation_cache_path),
            "validation_cache_sha256": sha256_file(validation_cache_path),
        },
    }


def train_local_residual(
    train_cache_path: str | Path,
    validation_cache_path: str | Path,
    output_dir: str | Path,
    *,
    center_reference_path: str | Path | None = None,
    seed: int = 42,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    batch_size: int = 1024,
    max_epochs: int = 30,
    patience: int = 5,
) -> Path:
    train_cache_path = Path(train_cache_path).resolve()
    validation_cache_path = Path(validation_cache_path).resolve()
    output_dir = Path(output_dir).resolve()
    train_cache = _load_cache(train_cache_path)
    validation_cache = _load_cache(validation_cache_path)
    if set(train_cache["paths"]) & set(validation_cache["paths"]):
        raise ValueError("N1 train and validation caches overlap")
    if train_cache["checkpoint_sha256"] != validation_cache["checkpoint_sha256"]:
        raise ValueError("N1 caches were not built from the same checkpoint")
    center_audit: dict[str, float] | None = None
    if center_reference_path is not None:
        reference = torch.load(
            center_reference_path, map_location="cpu", weights_only=False
        )
        if list(reference["paths"]) != list(validation_cache["paths"]):
            raise ValueError("Center reference path order does not match N1 validation")
        difference = (
            torch.as_tensor(reference["logits"]).float()
            - torch.as_tensor(validation_cache["global_logits"]).float()
        ).abs()
        agreement = (
            torch.as_tensor(reference["logits"]).argmax(dim=1)
            == torch.as_tensor(validation_cache["global_logits"]).argmax(dim=1)
        ).float().mean()
        center_audit = {
            "maximum_absolute_logit_difference": float(difference.max()),
            "prediction_agreement": float(agreement),
        }
        if center_audit != {
            "maximum_absolute_logit_difference": 0.0,
            "prediction_agreement": 1.0,
        }:
            raise ValueError(f"N1 global cache is not bit-exact: {center_audit}")

    set_seed(int(seed), deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = LearnedLocalResidualHead(512, 500, dropout=0.1).to(device)
    protocol: dict[str, Any] = {
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss": "cross_entropy",
        "center_reference_audit": center_audit,
        "parameter_scan": False,
    }
    epoch_zero_metrics, epoch_zero_logits = _metrics(
        head, validation_cache, device
    )
    epoch_zero_difference = (
        epoch_zero_logits - torch.as_tensor(validation_cache["global_logits"]).float()
    ).abs()
    epoch_zero_audit = {
        "maximum_absolute_logit_difference": float(epoch_zero_difference.max()),
        "prediction_agreement": float(
            (
                epoch_zero_logits.argmax(dim=1)
                == torch.as_tensor(validation_cache["global_logits"]).argmax(dim=1)
            ).float().mean()
        ),
    }
    if epoch_zero_audit != {
        "maximum_absolute_logit_difference": 0.0,
        "prediction_agreement": 1.0,
    }:
        raise RuntimeError(f"N1 zero initialisation audit failed: {epoch_zero_audit}")

    train_dataset = TensorDataset(
        torch.as_tensor(train_cache["local_features"]),
        torch.as_tensor(train_cache["global_logits"]).float(),
        torch.as_tensor(train_cache["labels"]).long(),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(max_epochs), eta_min=0.0
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = [
        {"epoch": 0, "metrics": epoch_zero_metrics, "audit": epoch_zero_audit}
    ]
    best_selector = float("-inf")
    stale_epochs = 0
    best_path = output_dir / "best.pt"
    for epoch in range(1, int(max_epochs) + 1):
        head.train()
        loss_sum = 0.0
        samples = 0
        for local_features, base_logits, labels in loader:
            local_features = local_features.to(device, non_blocking=True)
            base_logits = base_logits.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(base_logits, local_features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * labels.numel()
            samples += labels.numel()
        scheduler.step()
        metrics, _ = _metrics(head, validation_cache, device)
        selector = float(metrics["clean_core_micro"])
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(samples, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "residual_parameter_norm": head.residual_parameter_norm(),
            "metrics": metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if selector > best_selector:
            best_selector = selector
            stale_epochs = 0
            _atomic_torch_save(
                _checkpoint_payload(
                    head,
                    epoch=epoch,
                    metrics=metrics,
                    protocol=protocol,
                    train_cache_path=train_cache_path,
                    validation_cache_path=validation_cache_path,
                ),
                best_path,
            )
        else:
            stale_epochs += 1
        _atomic_torch_save(
            _checkpoint_payload(
                head,
                epoch=epoch,
                metrics=metrics,
                protocol=protocol,
                train_cache_path=train_cache_path,
                validation_cache_path=validation_cache_path,
            ),
            output_dir / "last.pt",
        )
        atomic_json_dump(
            {
                "protocol": protocol,
                "epoch_zero": history[0],
                "best_selector": best_selector,
                "history": history,
            },
            output_dir / "history.json",
        )
        if stale_epochs >= int(patience):
            break
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--center-reference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    print(
        train_local_residual(
            args.train_cache,
            args.validation_cache,
            args.output_dir,
            center_reference_path=args.center_reference,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
    )


if __name__ == "__main__":
    main()
