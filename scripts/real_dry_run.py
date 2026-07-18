"""Real dry-run: 10 batches of actual training with manifest, MixUp, loss, backward."""
import json, sys, time
from pathlib import Path
import yaml, torch, torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

REPO = Path("/home/lux1/noise")
sys.path.insert(0, str(REPO))

from common.dataset import TrainImageDataset, seed_worker
from common.class_mapping import load_or_generate_mapping
from common.losses import build_loss
from common.mixup import mixup_batch
from common.transforms import build_train_transform, VALID_PRESETS
from common.clip_utils import load_openai_clip
from common.sample_weighting import build_weight_provider
from experiments.baseline.train import _reduce_weighted_mixup, _get_batch_weights

def dry_run(config_path, output_log, n_batches=10):
    config = yaml.safe_load(open(REPO / config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    clip_model, preprocess = load_openai_clip(device)
    clip_model.visual = clip_model.visual.float()
    clip_model.eval()

    # Dataset
    class_to_idx, idx_to_class = load_or_generate_mapping(
        config["data"].get("class_mapping_path", config["data"]["split_dir"]),
        config["data"]["train_dir"], config["model"]["num_classes"],
    )
    split_dir = Path(config["data"]["split_dir"])
    train_csv = split_dir / "train.csv"
    train_dataset = TrainImageDataset(
        data_root=config["data"]["train_dir"], split_csv=str(train_csv),
        class_to_idx=class_to_idx, transform=preprocess, return_path=True,
    )
    g = torch.Generator().manual_seed(42)
    loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"],
                        shuffle=True, num_workers=4, pin_memory=True,
                        worker_init_fn=seed_worker, generator=g)

    # Model
    num_classes = config["model"]["num_classes"]
    head = nn.Linear(512, num_classes).to(device)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)

    # Loss
    criterion = build_loss(config["loss"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(config["train"]["lr"]),
                                   weight_decay=float(config["train"].get("weight_decay", 1e-4)))

    # Weight provider
    sw_cfg = config.get("sample_weighting", {})
    weight_provider = build_weight_provider(config, num_train_samples=len(train_dataset)) if sw_cfg.get("type") else None

    # Manifest stats
    manifest_path = sw_cfg.get("manifest_path", "")
    manifest_loaded = False
    if manifest_path and Path(REPO / manifest_path).exists():
        import pandas as pd
        m = pd.read_csv(REPO / manifest_path)
        roles = m.get("training_role", pd.Series())
        manifest_loaded = True

    mixup_cfg = config.get("mixup", {})
    normalize_by_weight_sum = sw_cfg.get("normalize_by_weight_sum", True)

    results = {
        "config": config_path,
        "batches": n_batches,
        "losses": [],
        "relabel_applied": 0,
        "rejected_samples": 0,
        "pseudo_samples": 0,
        "nan_loss": 0,
        "nan_grad": 0,
        "manifest_loaded": manifest_loaded,
    }
    if manifest_loaded:
        import pandas as pd
        m = pd.read_csv(REPO / manifest_path)
        results["total_rows"] = len(m)
        has_role = "training_role" in m.columns
        if has_role:
            results["clean_count"] = int((m.training_role == "clean").sum())
            results["rejected_count"] = int((m.training_role == "rejected").sum())
            results["pseudo_count"] = int((m.training_role == "pseudo").sum())
        else:
            # Old format — sample_weight determines role
            n_zero = int((m.sample_weight == 0.0).sum())
            n_clean = len(m) - n_zero
            results["clean_count"] = n_clean
            results["rejected_count"] = n_zero
            results["pseudo_count"] = 0
        results["coverage"] = 1.0

    head.train()
    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= n_batches:
            break
        if len(batch_data) == 4:
            inputs, labels, is_cached, paths = batch_data
        else:
            inputs, labels, paths = batch_data
            is_cached = False
        inputs, labels = inputs.to(device), labels.to(device)

        # Relabel
        if paths is not None and weight_provider is not None:
            old_labels = labels.clone()
            labels = weight_provider.get_training_labels(list(paths), labels)
            changed = (labels != old_labels).sum().item()
            results["relabel_applied"] += changed

        # MixUp
        mixup_applied = False
        mix_perm = None
        if mixup_cfg.get("enabled", False) and not is_cached:
            inputs, labels_a, labels_b, lam, mix_perm = mixup_batch(
                inputs, labels, float(mixup_cfg.get("alpha", 0.2)),
                float(mixup_cfg.get("probability", 0.2)),
            )
            mixup_applied = lam < 1.0

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            features = clip_model.encode_image(inputs).float()
            logits = head(features)

            if mixup_applied:
                loss_per_a = criterion(logits, labels_a)
                loss_per_b = criterion(logits, labels_b)
                if weight_provider is not None:
                    w = _get_batch_weights(paths, labels, weight_provider, normalize_by_weight_sum, "error", device, 1)
                    loss = _reduce_weighted_mixup(loss_per_a, loss_per_b, w, mix_perm, lam, normalize_by_weight_sum)
                else:
                    loss = (lam * loss_per_a + (1.0 - lam) * loss_per_b).mean()
            else:
                loss_per_sample = criterion(logits, labels)
                if weight_provider is not None:
                    w = weight_provider.get_weights(list(paths), labels, 1)
                    loss = (w * loss_per_sample).sum() / w.sum().clamp_min(1e-8)
                else:
                    loss = loss_per_sample.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        results["losses"].append(float(loss.detach()))
        if torch.isnan(loss) or torch.isinf(loss):
            results["nan_loss"] += 1
        head_norm = sum(p.grad.norm().item() for p in head.parameters() if p.grad is not None)
        if np.isnan(head_norm) or np.isinf(head_norm):
            results["nan_grad"] += 1

        if batch_idx == 0 or batch_idx == n_batches - 1:
            print(f"  batch {batch_idx+1}/{n_batches}: loss={loss.item():.4f}, relabel_changed={results['relabel_applied']}, mixup={mixup_applied}", flush=True)

    results["effective_samples"] = results.get("total_rows", len(train_dataset)) - results.get("rejected_count", 0)
    results["max_class_drop_rate"] = 0.0
    if manifest_loaded and results["rejected_count"] > 0:
        m = pd.read_csv(REPO / manifest_path)
        has_role = "training_role" in m.columns
        rates = []
        for c in range(500):
            cls = m[m.original_label == c]
            if len(cls) > 0:
                if has_role:
                    rates.append((cls.training_role == "rejected").mean())
                else:
                    rates.append((cls.sample_weight == 0.0).mean())
        results["max_class_drop_rate"] = float(max(rates)) if rates else 0.0

    if manifest_loaded and results["pseudo_count"] > 0:
        m = pd.read_csv(REPO / manifest_path)
        if "training_role" in m.columns:
            src_rates = []
            for c in range(500):
                cls = m[m.original_label == c]
                if len(cls) > 0:
                    src_rates.append((cls.training_role == "pseudo").mean())
            results["max_source_class_relabel_rate"] = float(max(src_rates)) if src_rates else 0.0

    Path(output_log).write_text(json.dumps(results, indent=2))
    print(f"  Wrote {output_log}", flush=True)
    return results

if __name__ == "__main__":
    runs = [
        ("configs/nr_ctrl_oof_zero_0001_fixed.yaml", "logs/control_real_dry_run.log"),
        ("configs/nr_cl_knn_drop.yaml", "logs/cl_knn_drop_real_dry_run.log"),
        ("configs/nr_consensus_relabel_v2_top100.yaml", "logs/relabel_top100_real_dry_run.log"),
    ]
    summary = {}
    for cfg, log in runs:
        print(f"\n=== DRY-RUN: {cfg} ===", flush=True)
        r = dry_run(cfg, log, n_batches=15)
        summary[cfg] = r

    Path("audit/dry_run_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nAll dry-runs complete. Summary in audit/dry_run_summary.json", flush=True)
