"""NR_COMBINED_UPGRADE GPU gate: epoch-0 + 2 training batches.

Precise parent-vs-child logit comparison, lora_A/lora_B grad tracking,
two-batch dynamics verification.  Saves to outputs/gate/... (not oof/).
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
        logging.FileHandler(str(GATE_DIR / "gate.log"), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("gate")

CONFIG_PATH = "configs/nr_combined_upgrade.yaml"
A2_CKPT = "outputs/oof/nr_cl_knn_drop/seed42/checkpoints/best.pt"
A2_SHA = "74ad2856e4449a42397edbda599ae79e8a4c6a6fa923624ef4e91a35e20a2a4c"
GIT_HEAD = "a989932"

results = {"gate_commit": GIT_HEAD, "parent_sha": A2_SHA, "checks": {}}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()


# ── 1. Verify parent SHA ─────────────────────────────────────────────
logger.info("=== 1. Verify parent SHA ===")
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

# Report manifest-level rejected (pre-BL) vs post-BL rejected
from common.manifest_loader import portable_image_key
import pandas as pd
mf = pd.read_csv("outputs/phase/phase3/oof/oof_zero_weight_manifest_thresh0.001.csv")
bl_raw = [p for p in Path("outputs/phase4/global_rejected_paths.txt").read_text().strip().split("\n") if p.strip()]
bl_keys = {portable_image_key(p) for p in bl_raw}
mf_all_rej = int((mf["p_original_label"] < 0.70).sum())
mf_post_bl_rej = sum(1 for _, row in mf.iterrows()
                     if portable_image_key(str(row["image_path"])) not in bl_keys
                     and float(row["p_original_label"]) < 0.70)
logger.info("Manifest rejected (pre-BL): %d  |  post-BL rejected: %d  |  BL: %d",
            mf_all_rej, mf_post_bl_rej, len(bl_keys))
assert mf_all_rej == 40962, f"Expected 40962 pre-BL rejected, got {mf_all_rej}"
assert mf_post_bl_rej == 39971, f"Expected 39971 post-BL rejected, got {mf_post_bl_rej}"
results["checks"]["manifest_rejected_pre_bl_40962"] = True
results["checks"]["manifest_rejected_post_bl_39971"] = True

# ── 4. Build A2 parent model (frozen, no LoRA, for epoch-0 comparison) ──
logger.info("=== 4. Build A2 parent model ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", device)

from experiments.baseline.model import build_model, CLIPLinearClassifier
_parent_cfg_bare = copy.deepcopy(config)
_parent_cfg_bare["model"]["freeze_clip"] = True
_parent_cfg_bare["peft"] = {"type": "linear_head_only"}
parent_model, preprocess = build_model(_parent_cfg_bare, device)
_parent_state = torch.load(A2_CKPT, map_location="cpu").get("model_state_dict")
parent_model.load_state_dict(_parent_state, strict=True)
parent_model.eval()
for p in parent_model.parameters():
    p.requires_grad_(False)
logger.info("A2 parent built: strict load OK, all frozen, eval mode")

# ── 5. Build child model + PEFT ──────────────────────────────────────
logger.info("=== 5. Build child model + apply PEFT ===")
model, _ = build_model(config, device)
# Strict-load parent weights BEFORE PEFT
parent_ckpt_meta = torch.load(A2_CKPT, map_location="cpu")
parent_state = parent_ckpt_meta.get("model_state_dict", parent_ckpt_meta)
missing, unexpected = model.load_state_dict(parent_state, strict=True)
assert len(missing) == 0, f"Missing keys: {missing[:5]}"
assert len(unexpected) == 0, f"Unexpected keys: {unexpected[:5]}"
logger.info("Pre-PEFT strict load: 0 missing, 0 unexpected")
results["checks"]["pre_peft_strict_load_ok"] = True

from common.peft import apply_peft
peft_cfg = config.get("peft", {})
peft_info = apply_peft(model, peft_cfg)
logger.info("PEFT: type=%s, trainable=%d", peft_info["peft_type"], peft_info["trainable_param_count"])

# Verify trainable params
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
for n in trainable_names:
    assert "classifier" in n or "lora_" in n, f"Unexpected trainable: {n}"
    assert "k_proj" not in n, f"K should be frozen: {n}"
logger.info("Trainable params audit: OK")
results["checks"]["trainable_params_audit_ok"] = True

# ── 6. Epoch-0: direct parent-vs-child logit comparison ─────────────
logger.info("=== 6. Epoch-0: direct logit comparison ===")
from common.class_mapping import load_or_generate_mapping
class_mapping_path = config["data"].get("class_mapping_path", config["data"]["split_dir"])
class_to_idx, _ = load_or_generate_mapping(
    metadata_dir=class_mapping_path, train_dir=config["data"]["train_dir"],
    expected_num_classes=config["model"]["num_classes"],
)
from common.dataset import TrainImageDataset
val_csv = Path(config["data"]["split_dir"]) / "val.csv"
val_dataset = TrainImageDataset(
    data_root=config["data"]["train_dir"], split_csv=str(val_csv),
    class_to_idx=class_to_idx, transform=preprocess, return_path=True,
)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

model.eval()
parent_model.eval()
all_max_diff = []
all_mean_diff = []
argmax_mismatches = 0
total_samples = 0

with torch.no_grad():
    for batch_idx, (images, labels, _) in enumerate(val_loader):
        if batch_idx >= 8:  # ~512 samples enough
            break
        images = images.to(device)
        child_logits = model(images).float()
        parent_logits = parent_model(images).float()

        diff = (child_logits - parent_logits).abs()
        all_max_diff.append(diff.max().item())
        all_mean_diff.append(diff.mean().item())

        child_preds = child_logits.argmax(dim=1)
        parent_preds = parent_logits.argmax(dim=1)
        argmax_mismatches += (child_preds != parent_preds).sum().item()
        total_samples += images.size(0)

max_abs_diff = max(all_max_diff)
mean_abs_diff = sum(all_mean_diff) / len(all_mean_diff)
logger.info("Epoch-0 logit comparison (%d samples):", total_samples)
logger.info("  max_abs_diff=%.6e", max_abs_diff)
logger.info("  mean_abs_diff=%.6e", mean_abs_diff)
logger.info("  argmax_mismatch=%d/%d", argmax_mismatches, total_samples)

assert argmax_mismatches == 0, f"Argmax mismatch: {argmax_mismatches}/{total_samples}"
assert max_abs_diff < 0.01, f"Max abs diff too large: {max_abs_diff:.2e}"
logger.info("Epoch-0 gate: PASSED (argmax=0, max_diff=%.2e)", max_abs_diff)
results["checks"]["epoch0_argmax_mismatch_0"] = True
results["checks"]["epoch0_max_abs_diff"] = float(max_abs_diff)
results["checks"]["epoch0_mean_abs_diff"] = float(mean_abs_diff)

# ── 7. Feature distillation parent ───────────────────────────────────
logger.info("=== 7. Feature distillation parent ===")
from common.feature_distillation import FeatureDistillation
distill_parent, _ = build_model(_parent_cfg_bare, device)
distill_parent.load_state_dict(_parent_state, strict=True)
for p in distill_parent.parameters():
    p.requires_grad_(False)
distill_parent.eval()
distill_parent_sha = sha256_file(A2_CKPT)
assert distill_parent_sha == A2_SHA
feature_distill = FeatureDistillation(distill_parent)
results["checks"]["distill_parent_sha_ok"] = True

# ── 8. DataLoader ────────────────────────────────────────────────────
logger.info("=== 8. DataLoader ===")
train_csv = Path(config["data"]["split_dir"]) / "train.csv"
train_dataset = TrainImageDataset(
    data_root=config["data"]["train_dir"], split_csv=str(train_csv),
    class_to_idx=class_to_idx, transform=preprocess, return_path=True,
)
_old_n = len(train_dataset.samples)
_keep = [i for i, p in enumerate(train_dataset.samples)
         if portable_image_key(str(p)) not in bl_keys]
train_dataset.samples = [train_dataset.samples[i] for i in _keep]
train_dataset.labels = [train_dataset.labels[i] for i in _keep]
_dropped = _old_n - len(_keep)
assert _dropped == 991
assert len(train_dataset.samples) == 90204
ds_keys = {portable_image_key(str(p)) for p in train_dataset.samples}
assert len(ds_keys & bl_keys) == 0
logger.info("Blacklist: %d dropped, dataset=%d, overlap=0", _dropped, len(train_dataset.samples))
results["checks"]["blacklist_991_dropped"] = True
results["checks"]["blacklist_dataset_overlap_0"] = True

g = torch.Generator()
g.manual_seed(config["data"].get("train_seed", 42))
train_loader = DataLoader(
    train_dataset, batch_size=config["train"]["batch_size"], shuffle=True,
    num_workers=config["train"]["num_workers"], pin_memory=True,
    drop_last=False, generator=g,
)
assert len(train_loader) == 1410
logger.info("DataLoader: %d batches", len(train_loader))
results["checks"]["dataloader_batches"] = 1410

# ── 9. Optimizer, criterion ──────────────────────────────────────────
from common.peft import build_peft_param_groups
from common.losses import build_loss
from common.sample_weighting import OOFManifestProvider

train_cfg = config["train"]
param_groups = build_peft_param_groups(
    model, peft_cfg, head_lr=train_cfg["lr"],
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
feat_distill_weight = config.get("sample_weighting", {}).get("feature_distillation_weight", 2.0)

wp = OOFManifestProvider(
    config["sample_weighting"]["manifest_path"],
    min_weight=0.0, max_weight=1.0,
    missing_policy=config["sample_weighting"]["missing_weight_policy"],
    clean_prob_threshold=config["sample_weighting"]["clean_prob_threshold"],
)
logger.info("Optimizer: head lr=%.1e wd=%.1e | lora lr=%.1e wd=%.1e",
            train_cfg["lr"], train_cfg["weight_decay"],
            train_cfg.get("backbone_lr", 0), train_cfg.get("backbone_weight_decay", 0))

# Collect lora_A / lora_B param names
lora_a_names = sorted([n for n, p in model.named_parameters()
                        if "lora_A" in n and p.requires_grad])
lora_b_names = sorted([n for n, p in model.named_parameters()
                        if "lora_B" in n and p.requires_grad])
logger.info("LoRA A params: %d, LoRA B params: %d", len(lora_a_names), len(lora_b_names))

# ── 10. Training batch #1 ───────────────────────────────────────────
logger.info("=== 10. Batch #1 ===")
model.train()

batch_iter = iter(train_loader)
batch1 = next(batch_iter)
images1, labels1, paths1 = batch1[0].to(device), batch1[1].to(device), batch1[2]

# Snapshot pre-step params
pre_a1 = {n: model.get_parameter(n).data.clone() for n in lora_a_names}
pre_b1 = {n: model.get_parameter(n).data.clone() for n in lora_b_names}

optimizer.zero_grad()
with autocast(device_type=device.type, enabled=use_amp):
    logits1 = model(images1)
    loss_per_sample1 = criterion(logits1, labels1)
w1 = wp.get_weights(list(paths1), labels1, epoch=0).to(device)
task_loss1 = (loss_per_sample1 * w1).sum() / (w1.sum() + 1e-8)
clean_mask1 = wp.get_clean_mask(list(paths1)).to(device)
rej_mask1 = ~clean_mask1
feat_loss1 = torch.tensor(0.0, device=device)
if rej_mask1.any():
    s_feat1 = model.encode_image(images1)
    p_feat1 = feature_distill.get_parent_features(images1)
    feat_loss1 = feature_distill.compute_loss(s_feat1[rej_mask1], p_feat1[rej_mask1])
total_loss1 = task_loss1 + feat_distill_weight * feat_loss1

scaler.scale(total_loss1).backward()
if train_cfg.get("max_grad_norm", 1.0) > 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])

# Report batch-1 gradients
grad_a1 = {}
grad_b1 = {}
for n in lora_a_names:
    p = model.get_parameter(n)
    grad_a1[n] = p.grad.norm().item() if p.grad is not None else 0.0
for n in lora_b_names:
    p = model.get_parameter(n)
    grad_b1[n] = p.grad.norm().item() if p.grad is not None else 0.0

a_nonzero_1 = sum(1 for v in grad_a1.values() if v > 0)
b_nonzero_1 = sum(1 for v in grad_b1.values() if v > 0)
logger.info("Batch #1 grad norms: lora_A nonzero=%d/12, lora_B nonzero=%d/12",
            a_nonzero_1, b_nonzero_1)

old_scale = scaler.get_scale()
scaler.step(optimizer)
scaler.update()

# Check A/B changes after step
a_changed_1 = sum(1 for n in lora_a_names
                  if not torch.equal(pre_a1[n], model.get_parameter(n).data))
b_changed_1 = sum(1 for n in lora_b_names
                  if not torch.equal(pre_b1[n], model.get_parameter(n).data))
logger.info("Batch #1 param changes: lora_A=%d/12, lora_B=%d/12",
            a_changed_1, b_changed_1)

assert a_changed_1 == 12, f"Expected all 12 lora_A to change, got {a_changed_1}"
assert b_changed_1 == 0, f"Expected 0 lora_B to change (B=random init, A=0 init, first step only A gets grad via B), got {b_changed_1}"
logger.info("Batch #1 dynamics: CORRECT — A changes 12/12, B stays 0/12 (A=0, B~N init)")

results["checks"]["batch1_a_changed_12"] = a_changed_1
results["checks"]["batch1_b_changed_0"] = b_changed_1
results["checks"]["batch1_a_nonzero_grad"] = a_nonzero_1
results["checks"]["batch1_b_nonzero_grad"] = b_nonzero_1

# ── 11. Training batch #2 ───────────────────────────────────────────
logger.info("=== 11. Batch #2 ===")
batch2 = next(batch_iter)
images2, labels2, paths2 = batch2[0].to(device), batch2[1].to(device), batch2[2]

pre_a2 = {n: model.get_parameter(n).data.clone() for n in lora_a_names}
pre_b2 = {n: model.get_parameter(n).data.clone() for n in lora_b_names}

optimizer.zero_grad()
with autocast(device_type=device.type, enabled=use_amp):
    logits2 = model(images2)
    loss_per_sample2 = criterion(logits2, labels2)
w2 = wp.get_weights(list(paths2), labels2, epoch=0).to(device)
task_loss2 = (loss_per_sample2 * w2).sum() / (w2.sum() + 1e-8)
clean_mask2 = wp.get_clean_mask(list(paths2)).to(device)
rej_mask2 = ~clean_mask2
feat_loss2 = torch.tensor(0.0, device=device)
if rej_mask2.any():
    s_feat2 = model.encode_image(images2)
    p_feat2 = feature_distill.get_parent_features(images2)
    feat_loss2 = feature_distill.compute_loss(s_feat2[rej_mask2], p_feat2[rej_mask2])
total_loss2 = task_loss2 + feat_distill_weight * feat_loss2

scaler.scale(total_loss2).backward()
if train_cfg.get("max_grad_norm", 1.0) > 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])

grad_a2 = {}
grad_b2 = {}
for n in lora_a_names:
    p = model.get_parameter(n)
    grad_a2[n] = p.grad.norm().item() if p.grad is not None else 0.0
for n in lora_b_names:
    p = model.get_parameter(n)
    grad_b2[n] = p.grad.norm().item() if p.grad is not None else 0.0

a_nonzero_2 = sum(1 for v in grad_a2.values() if v > 0)
b_nonzero_2 = sum(1 for v in grad_b2.values() if v > 0)
logger.info("Batch #2 grad norms: lora_A nonzero=%d/12, lora_B nonzero=%d/12",
            a_nonzero_2, b_nonzero_2)

scaler.step(optimizer)
scaler.update()

a_changed_2 = sum(1 for n in lora_a_names
                  if not torch.equal(pre_a2[n], model.get_parameter(n).data))
b_changed_2 = sum(1 for n in lora_b_names
                  if not torch.equal(pre_b2[n], model.get_parameter(n).data))
logger.info("Batch #2 param changes: lora_A=%d/12, lora_B=%d/12",
            a_changed_2, b_changed_2)

# Batch-2: B MUST start changing (A is no longer zero, so B gets gradients)
assert b_nonzero_2 > 0, \
    f"Batch #2: B grad norms still all zero! lora_B grad norms: {grad_b2}"
assert b_changed_2 > 0, \
    f"Batch #2: B params unchanged! Expected B to start changing."
logger.info("Batch #2 dynamics: CORRECT — B begins changing (A≠0 after batch 1, B gets grad)")

results["checks"]["batch2_a_nonzero_grad"] = a_nonzero_2
results["checks"]["batch2_b_nonzero_grad"] = b_nonzero_2
results["checks"]["batch2_a_changed"] = a_changed_2
results["checks"]["batch2_b_changed"] = b_changed_2

# ── 12. Post-batch checks ──────────────────────────────────────────
logger.info("=== 12. Final checks ===")

# Losses finite
for label, loss_val in [("batch1_task", task_loss1), ("batch1_feat", feat_loss1),
                         ("batch1_total", total_loss1),
                         ("batch2_task", task_loss2), ("batch2_feat", feat_loss2),
                         ("batch2_total", total_loss2)]:
    assert torch.isfinite(loss_val), f"{label} not finite: {loss_val.item()}"
logger.info("All losses finite: batch1 total=%.4f feat=%.6f | batch2 total=%.4f feat=%.6f",
            total_loss1.item(), feat_loss1.item(), total_loss2.item(), feat_loss2.item())
results["checks"]["losses_finite"] = True

# Parent zero gradients
for n, p in distill_parent.named_parameters():
    assert p.grad is None, f"Parent param {n} got gradient!"
logger.info("Parent gradients: ALL NONE")
results["checks"]["parent_no_gradients"] = True

# K frozen
for n, p in model.named_parameters():
    if "k_proj" in n:
        assert not p.requires_grad and p.grad is None, f"K not frozen: {n}"
logger.info("K projection: frozen and zero-grad")
results["checks"]["k_frozen"] = True

# No AMP overflow
assert scaler.get_scale() > 0, f"AMP overflow! scale={scaler.get_scale()}"
logger.info("AMP scale: %.1f — OK", scaler.get_scale())
results["checks"]["no_amp_overflow"] = True

# ── 13. Summary ──────────────────────────────────────────────────────
# gate_passed: only boolean checks matter; integer counts are informational
_bool_checks = {k: v for k, v in results["checks"].items() if isinstance(v, bool)}
results["gate_passed"] = all(_bool_checks.values())
logger.info("=" * 60)
logger.info("GATE RESULT: %s", "PASSED" if results["gate_passed"] else "FAILED")
logger.info("Checks: %s", json.dumps(results["checks"], indent=2))
logger.info("=" * 60)

(GATE_DIR / "gate_results.json").write_text(json.dumps(results, indent=2, default=str))
logger.info("Gate results written to %s", GATE_DIR / "gate_results.json")

if not results["gate_passed"]:
    logger.error("GATE FAILED — review gate_results.json")
    sys.exit(1)
logger.info("GPU gate PASSED. Do NOT start full training without approval.")
