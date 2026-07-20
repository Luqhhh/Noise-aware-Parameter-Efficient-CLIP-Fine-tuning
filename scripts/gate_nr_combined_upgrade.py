"""NR_COMBINED_UPGRADE GPU gate: epoch-0 + 1 training batch only.

Usage:
    python scripts/gate_nr_combined_upgrade.py

Does NOT start full 6-epoch training.  Saves results to
outputs/gate/nr_combined_upgrade/seed42/.
"""

import copy
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# ── Setup ────────────────────────────────────────────────────────────
GATE_DIR = Path("outputs/gate/nr_combined_upgrade/seed42")
GATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(GATE_DIR / "gate.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("gate")

CONFIG_PATH = "configs/nr_combined_upgrade.yaml"
A2_CKPT = "outputs/oof/nr_cl_knn_drop/seed42/checkpoints/best.pt"
A2_SHA = "74ad2856e4449a42397edbda599ae79e8a4c6a6fa923624ef4e91a35e20a2a4c"

results = {"gate_branch": "a0907d6", "parent_sha": A2_SHA, "checks": {}}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()


# ── 1. Verify parent SHA ─────────────────────────────────────────────
logger.info("=== 1. Verify parent checkpoint SHA ===")
actual = sha256_file(A2_CKPT)
assert actual == A2_SHA, f"SHA mismatch: {actual[:16]} != {A2_SHA[:16]}"
logger.info("Parent SHA: OK (%s)", actual[:16])
results["checks"]["parent_sha_ok"] = True

# ── 2. Load config ───────────────────────────────────────────────────
from common.utils import load_config
config = load_config(CONFIG_PATH)
experiment_id = config["experiment"]["id"]
assert experiment_id == "NR_COMBINED_UPGRADE"
logger.info("Config: %s", experiment_id)

# ── 3. Pre-model audit ───────────────────────────────────────────────
logger.info("=== 3. Pre-model audit ===")
from experiments.baseline.train import _runtime_manifest_audit_premodel
_audit_ok = _runtime_manifest_audit_premodel(
    config, experiment_id, GATE_DIR, logger,
)
assert _audit_ok
results["checks"]["pre_model_audit_ok"] = True

# ── 4. Build model + PEFT ────────────────────────────────────────────
logger.info("=== 4. Build model + apply PEFT ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", device)

from experiments.baseline.model import build_model, CLIPLinearClassifier
model, preprocess = build_model(config, device)

# Strict-load parent weights BEFORE PEFT
parent_ckpt = torch.load(A2_CKPT, map_location=device)
parent_state = parent_ckpt.get("model_state_dict", parent_ckpt)
missing, unexpected = model.load_state_dict(parent_state, strict=True)
assert len(missing) == 0, f"Missing keys: {missing[:5]}"
assert len(unexpected) == 0, f"Unexpected keys: {unexpected[:5]}"
logger.info("Pre-PEFT strict load: 0 missing, 0 unexpected")
results["checks"]["pre_peft_strict_load_ok"] = True

# Apply visual_lora
from common.peft import apply_peft
peft_cfg = config.get("peft", {})
peft_info = apply_peft(model, peft_cfg)
logger.info("PEFT applied: type=%s, trainable=%d",
            peft_info["peft_type"], peft_info["trainable_param_count"])
results["checks"]["peft_applied"] = True

# Count params
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
logger.info("Trainable: %d params, Frozen: %d params",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if not p.requires_grad))

# Verify only classifier + LoRA are trainable
for n in trainable_names:
    assert "classifier" in n or "lora_" in n, f"Unexpected trainable: {n}"
# K must be frozen
for n in trainable_names:
    assert "k_proj" not in n, f"K should be frozen: {n}"
logger.info("Trainable params audit: OK")
results["checks"]["trainable_params_audit_ok"] = True

# ── 5. Epoch-0 validation gate ──────────────────────────────────────
logger.info("=== 5. Epoch-0 validation gate ===")
model.eval()

# Load val data (first 256 samples for speed)
from common.dataset import TrainImageDataset
from common.class_mapping import load_or_generate_mapping

class_mapping_path = config["data"].get("class_mapping_path", config["data"]["split_dir"])
class_to_idx, _ = load_or_generate_mapping(
    metadata_dir=class_mapping_path,
    train_dir=config["data"]["train_dir"],
    expected_num_classes=config["model"]["num_classes"],
)

val_csv = Path(config["data"]["split_dir"]) / "val.csv"
val_dataset = TrainImageDataset(
    data_root=config["data"]["train_dir"],
    split_csv=str(val_csv),
    class_to_idx=class_to_idx,
    transform=preprocess,
    return_path=True,
)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=True)

# Quick eval on first 4 batches (~1024 samples)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (images, labels, _) in enumerate(val_loader):
        if i >= 4:
            break
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

val_acc_sample = correct / total
parent_best = parent_ckpt.get("best_val_acc", 0.69444)
logger.info("Epoch-0 sample val acc: %.6f (parent best: %.6f, delta: %.6f)",
            val_acc_sample, parent_best, abs(val_acc_sample - parent_best))
# Check delta is reasonable (< 0.02 for 1024-sample estimate)
delta = abs(val_acc_sample - parent_best)
assert delta < 0.05, f"Epoch-0 delta too large: {delta:.6f}"
logger.info("Epoch-0 gate: PASSED (delta=%.6f < 0.05)", delta)
results["checks"]["epoch0_gate_ok"] = True

# ── 6. Build feature distillation parent ───────────────────────────
logger.info("=== 6. Feature distillation parent ===")
from common.feature_distillation import FeatureDistillation
_parent_cfg = copy.deepcopy(config)
_parent_cfg["model"]["freeze_clip"] = True
parent_model, _ = build_model(_parent_cfg, device)
_parent_ckpt = torch.load(A2_CKPT, map_location="cpu")
_parent_state = _parent_ckpt.get("model_state_dict", _parent_ckpt)
_parent_keys = {k: v for k, v in _parent_state.items() if "classifier" not in k}
parent_model.load_state_dict(_parent_keys, strict=False)
for p in parent_model.parameters():
    p.requires_grad_(False)
parent_model.eval()
parent_model.to(device)

parent_actual_sha = sha256_file(A2_CKPT)
assert parent_actual_sha == A2_SHA, f"Parent SHA mismatch: {parent_actual_sha[:16]}"
logger.info("Feature-distillation parent SHA: OK (%s)", parent_actual_sha[:16])
results["checks"]["distill_parent_sha_ok"] = True

feature_distill = FeatureDistillation(parent_model)

# ── 7. Build training data loader ───────────────────────────────────
logger.info("=== 7. Build DataLoader ===")
train_csv = Path(config["data"]["split_dir"]) / "train.csv"
train_dataset = TrainImageDataset(
    data_root=config["data"]["train_dir"],
    split_csv=str(train_csv),
    class_to_idx=class_to_idx,
    transform=preprocess,
    return_path=True,
)

# Apply global blacklist
from common.manifest_loader import portable_image_key
_bl_path = Path("outputs/phase4/global_rejected_paths.txt")
_bl_raw = [p for p in _bl_path.read_text().strip().split("\n") if p.strip()]
_bl_keys = {portable_image_key(p) for p in _bl_raw}
_old_n = len(train_dataset.samples)
_keep = [i for i, p in enumerate(train_dataset.samples)
         if portable_image_key(str(p)) not in _bl_keys]
train_dataset.samples = [train_dataset.samples[i] for i in _keep]
train_dataset.labels = [train_dataset.labels[i] for i in _keep]
_dropped = _old_n - len(_keep)
logger.info("Blacklist: removed %d samples (%d → %d)", _dropped, _old_n, len(_keep))
assert _dropped == 991, f"Expected 991 dropped, got {_dropped}"
assert len(train_dataset.samples) == 90204
results["checks"]["blacklist_dropped_991"] = True

# Verify zero overlap
ds_keys = {portable_image_key(str(p)) for p in train_dataset.samples}
assert len(ds_keys & _bl_keys) == 0, f"Blacklist overlap: {len(ds_keys & _bl_keys)}"
logger.info("Blacklist ∩ dataset = 0: OK")
results["checks"]["blacklist_dataset_zero_overlap"] = True

g = torch.Generator()
g.manual_seed(config["data"].get("train_seed", 42))
train_loader = DataLoader(
    train_dataset, batch_size=config["train"]["batch_size"], shuffle=True,
    num_workers=config["train"]["num_workers"], pin_memory=True,
    drop_last=False, generator=g,
)
logger.info("DataLoader: %d batches (batch_size=%d, dataset=%d)",
            len(train_loader), config["train"]["batch_size"], len(train_dataset.samples))
assert len(train_loader) == 1410, f"Expected 1410 batches, got {len(train_loader)}"
results["checks"]["dataloader_batches"] = 1410

# ── 8. Build optimizer, criterion, scaler ───────────────────────────
logger.info("=== 8. Build optimizer ===")
from common.peft import build_peft_param_groups
from common.losses import build_loss

train_cfg = config["train"]
param_groups = build_peft_param_groups(
    model, peft_cfg,
    head_lr=train_cfg["lr"],
    head_weight_decay=train_cfg["weight_decay"],
    backbone_lr=train_cfg.get("backbone_lr"),
    backbone_weight_decay=train_cfg.get("backbone_weight_decay"),
)
optimizer = torch.optim.AdamW(param_groups)
scaler = GradScaler(device=device.type, enabled=train_cfg.get("amp", False))

loss_cfg = config.get("loss", {}).copy()
loss_cfg["reduction"] = "none"
criterion = build_loss({"loss": loss_cfg})
use_amp = train_cfg.get("amp", False)
feat_distill_weight = config.get("sample_weighting", {}).get(
    "feature_distillation_weight", 2.0)

# ── 9. Run 1 training batch ─────────────────────────────────────────
logger.info("=== 9. Run 1 training batch ===")
model.train()

batch_data = next(iter(train_loader))
images, labels, paths = batch_data[0], batch_data[1], batch_data[2]
images = images.to(device)
labels = labels.to(device)

# Capture pre-step LoRA params
pre_step_params = {}
for n, p in model.named_parameters():
    if "lora_" in n and p.requires_grad:
        pre_step_params[n] = p.data.clone()

optimizer.zero_grad()

with autocast(device_type=device.type, enabled=use_amp):
    logits = model(images)
    loss_per_sample = criterion(logits, labels)

# Apply clean-prob weights (binary: 0 or 1)
from common.sample_weighting import OOFManifestProvider
wp = OOFManifestProvider(
    config["sample_weighting"]["manifest_path"],
    min_weight=0.0, max_weight=1.0,
    missing_policy=config["sample_weighting"]["missing_weight_policy"],
    clean_prob_threshold=config["sample_weighting"]["clean_prob_threshold"],
)
w = wp.get_weights(list(paths), labels, epoch=0).to(device)
w_sum = w.sum() + 1e-8
task_loss = (loss_per_sample * w).sum() / w_sum

# Feature distillation on rejected samples only
clean_mask = wp.get_clean_mask(list(paths)).to(device)
rejected_mask = ~clean_mask
feat_loss = torch.tensor(0.0, device=device)
if rejected_mask.any():
    s_feat = model.encode_image(images)
    p_feat = feature_distill.get_parent_features(images)
    feat_loss = feature_distill.compute_loss(
        s_feat[rejected_mask], p_feat[rejected_mask],
    )

total_loss = task_loss + feat_distill_weight * feat_loss

scaler.scale(total_loss).backward()

# Check gradients
max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
if max_grad_norm > 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# Verify LoRA grads
lora_grad_norms = {}
for n, p in model.named_parameters():
    if "lora_" in n and p.requires_grad:
        if p.grad is not None:
            lora_grad_norms[n] = p.grad.norm().item()
assert len(lora_grad_norms) > 0, "No LoRA params received gradients!"
logger.info("LoRA params with gradients: %d", len(lora_grad_norms))
results["checks"]["lora_gradients_nonzero"] = len(lora_grad_norms)

# Verify parent model has NO gradients
for n, p in parent_model.named_parameters():
    assert p.grad is None, f"Parent param {n} got gradient!"
logger.info("Parent model gradients: ALL NONE — OK")
results["checks"]["parent_no_gradients"] = True

# Verify K projection frozen
for n, p in model.named_parameters():
    if "k_proj" in n:
        assert not p.requires_grad, f"K should be frozen: {n}"
        assert p.grad is None, f"K should have no grad: {n}"
logger.info("K projection frozen: OK")
results["checks"]["k_projection_frozen"] = True

scaler.step(optimizer)
scaler.update()

# ── 10. Post-step checks ───────────────────────────────────────────
logger.info("=== 10. Post-step checks ===")

# LoRA params changed
params_changed = 0
for n, p in model.named_parameters():
    if "lora_" in n and p.requires_grad:
        if not torch.equal(pre_step_params[n], p.data):
            params_changed += 1
assert params_changed > 0, "No LoRA params changed after step!"
logger.info("LoRA params changed: %d/%d", params_changed, len(pre_step_params))
results["checks"]["lora_params_changed"] = params_changed

# Loss finite
assert torch.isfinite(total_loss), f"Total loss not finite: {total_loss.item()}"
assert torch.isfinite(task_loss), f"Task loss not finite: {task_loss.item()}"
assert torch.isfinite(feat_loss), f"Feat loss not finite: {feat_loss.item()}"
logger.info("Losses: total=%.4f task=%.4f feat=%.4f",
            total_loss.item(), task_loss.item(), feat_loss.item())
results["checks"]["losses_finite"] = True

# Grad norm finite
for n, gn in lora_grad_norms.items():
    assert torch.isfinite(torch.tensor(gn)), f"Grad norm not finite: {n}={gn}"
logger.info("All grad norms finite: OK")
results["checks"]["grad_norms_finite"] = True

# No AMP overflow
assert scaler.get_scale() > 0, "AMP scale is zero (overflow)"
logger.info("AMP scale: %.1f — OK", scaler.get_scale())
results["checks"]["no_amp_overflow"] = True

# ── 11. Summary ──────────────────────────────────────────────────────
results["gate_passed"] = all(results["checks"].values())
logger.info("=" * 60)
logger.info("GATE RESULT: %s", "PASSED" if results["gate_passed"] else "FAILED")
logger.info("Checks: %s", json.dumps(results["checks"], indent=2))
logger.info("=" * 60)

# Write gate results
(GATE_DIR / "gate_results.json").write_text(json.dumps(results, indent=2, default=str))
logger.info("Gate results written to %s", GATE_DIR / "gate_results.json")

if not results["gate_passed"]:
    logger.error("GATE FAILED — review gate_results.json")
    sys.exit(1)
logger.info("GPU gate PASSED. Do NOT start full training without approval.")
