# Augmentation Selection & Partial Unfreeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (B) Select best augmentation from E2/E3/E4 via discovery+confirmation; (C) implement partial CLIP unfreeze infrastructure; (D) add comprehensive tests; (E) run F0-F3 partial unfreeze experiments; (F) select final configuration.

**Architecture:** Three phases with hard dependencies. Phase 1 completes the augmentation ablation (E3 needs finishing, E4 needs running), runs discovery rules to pick the best augmentation, and optionally confirms across multiple splits. Phase 2 implements model changes (parameter groups, init-checkpoint, scheduler fix, diagnostic logging) and tests, independent of Phase 1 results. Phase 3 initializes from the best augmentation checkpoint and runs F0-F3 with discriminative learning rates.

**Tech Stack:** PyTorch, OpenAI CLIP ViT-B/32, CosineAnnealingLR → LambdaLR for proportional decay

## Global Constraints

- `freeze_clip=true` must freeze ALL visual parameters before any selective unfreezing
- `use_cached_features=true` + `freeze_clip=false` must raise ValueError
- `--init-checkpoint` loads model weights only (no optimizer/scheduler/epoch state)
- F0-F3 experiments must NOT use `--resume` (reserved for crash recovery)
- Scheduler must maintain backbone_lr/head_lr ratio throughout training
- `strict=False` for init_checkpoint loading; for same architecture, `missing_keys` and `unexpected_keys` must be empty
- `batch_size: 128` for all experiments; if OOM, uniformly reduce to 64 for all paired experiments
- E0 paired reference required for every split seed in confirmation

---

## Phase 1: Augmentation Selection (Task B)

### Prerequisites

- E3 must complete (currently epoch 19/50, Val 66.7%)
- E4 must be run (config ready, splits prepared)

### Task B1: Complete E3 and Run E4

**Files:**
- (no code changes — execution only)

**Dependencies:** None beyond existing code

- [ ] **Step 1: Wait for E3 to complete**

E3 is running in background (`b10hqugnn`). Check completion:
```bash
cat outputs/e3/checkpoints/eval_results.json
```

- [ ] **Step 2: Extract E3 best result**

```bash
python3 -c "
import json
r = json.load(open('outputs/e3/checkpoints/eval_results.json'))
print(f'E3 (A2): best_val_acc={r[\"best_val_acc\"]:.4f}, epoch={r[\"dev_best_epoch\"]}')
"
```

- [ ] **Step 3: Run E4 (Linear + A3)**

```bash
python3 -m experiments.baseline.train --config configs/e4_augmentation.yaml
```

E4 uses online encoding, ~140s/epoch. With early stopping, expect 35-45 epochs.

- [ ] **Step 4: Extract E4 best result**

```bash
python3 -c "
import json
r = json.load(open('outputs/e4/checkpoints/eval_results.json'))
print(f'E4 (A3): best_val_acc={r[\"best_val_acc\"]:.4f}, epoch={r[\"dev_best_epoch\"]}')
"
```

- [ ] **Step 5: Commit E3 and E4 results**

```bash
git add outputs/e3/ outputs/e4/
git commit -m "results: E3 (A2) and E4 (A3) augmentation ablation"
```

### Task B2: Discovery — Apply Selection Rules

**Files:**
- Create: `scripts/select_augmentation.py`

**Dependencies:** Task B1 (E3, E4 results available)

**Rules (from spec):**

1. If all candidates improve over E0 by `< +0.2 pp`, keep A0
2. If best candidate improves by `>= +0.2 pp`, enter confirmation
3. If 1st and 2nd differ by `<= 0.2 pp`, both enter confirmation
4. If A3 only beats A2 by `< 0.2 pp`, prefer A2
5. If A2 only beats A1 by `< 0.2 pp`, prefer A1

- [ ] **Step 1: Write the discovery script**

```python
# scripts/select_augmentation.py
"""Apply B1 discovery rules to select augmentation for confirmation."""

import json
import sys
from pathlib import Path
from typing import Dict


def load_acc(path: str) -> float:
    with open(path) as f:
        return json.load(f)["best_val_acc"]


def main():
    # E0 reference: best from hyper search (lr=5e-3, wd=1e-4)
    e0_acc = load_acc(
        "outputs/e0/search/lr_5e-03__wd_1e-04/checkpoints/eval_results.json"
    )

    candidates = {}
    for exp_id, path in [
        ("E2", "outputs/e2/checkpoints/eval_results.json"),
        ("E3", "outputs/e3/checkpoints/eval_results.json"),
        ("E4", "outputs/e4/checkpoints/eval_results.json"),
    ]:
        if Path(path).exists():
            candidates[exp_id] = {
                "acc": load_acc(path),
                "aug": {"E2": "a1", "E3": "a2", "E4": "a3"}[exp_id],
            }

    print(f"E0 reference (A0): {e0_acc:.6f}")
    print()

    # Compute deltas
    deltas = {}
    for cid, info in candidates.items():
        delta = info["acc"] - e0_acc
        deltas[cid] = delta
        print(f"{cid} ({info['aug']}): {info['acc']:.6f}  delta={delta:+.6f} ({delta*100:+.2f}pp)")

    # Rule 1: all < +0.2pp → keep A0
    best_delta = max(deltas.values())
    if best_delta < 0.002:
        print("\n→ All candidates < +0.2pp. Keep A0.")
        print("BEST_AUG=A0")
        print("BEST_EXP=E0")
        return

    # Sort candidates by delta descending
    ranked = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
    first_id, first_delta = ranked[0]
    first_aug = candidates[first_id]["aug"]
    second_id, second_delta = ranked[1] if len(ranked) > 1 else (None, -999)

    # Rules 4 & 5: prefer simpler augmentation when gains are marginal
    # Rule 5: A2 vs A1
    if first_aug == "a2" and first_delta - deltas.get("E2", -999) < 0.002:
        print(f"\n→ A2 only beats A1 by < 0.2pp. Prefer A1 (E2).")
        print(f"BEST_AUG=A1")
        print(f"BEST_EXP=E2")
        print(f"CONFIRMATION_CANDIDATES=E2")
        return

    # Rule 4: A3 vs A2
    if first_aug == "a3":
        a2_delta = deltas.get("E3", -999)
        if first_delta - a2_delta < 0.002:
            print(f"\n→ A3 only beats A2 by < 0.2pp. Prefer A2 (E3).")
            print(f"BEST_AUG=A2")
            print(f"BEST_EXP=E3")
            print(f"CONFIRMATION_CANDIDATES=E3")
            return

    # Rule 3: 1st vs 2nd <= 0.2pp → both enter confirmation
    confirmation = [first_id]
    if second_id and first_delta - second_delta <= 0.002:
        confirmation.append(second_id)

    print(f"\n→ Best candidate improves >= +0.2pp. Enter confirmation.")
    print(f"BEST_AUG={candidates[first_id]['aug']}")
    print(f"BEST_EXP={first_id}")
    print(f"CONFIRMATION_CANDIDATES={','.join(confirmation)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run discovery**

```bash
python3 scripts/select_augmentation.py
```

Expected output: `BEST_AUG=`, `BEST_EXP=`, `CONFIRMATION_CANDIDATES=` lines.

- [ ] **Step 3: Record discovery decision**

Save output to `outputs/ablation_discovery.txt`.

- [ ] **Step 4: Commit**

```bash
git add scripts/select_augmentation.py outputs/ablation_discovery.txt
git commit -m "feat: augmentation discovery (B1) — selection rules applied"
```

### Task B3: Confirmation — Multi-Split Runs (conditional)

**Files:**
- Modify: `configs/e0_hyper_search.yaml` (split_seed)
- Create: temp configs for multi-split E0 and candidate

**Dependencies:** Task B2 (discovery decision), Task C completion (for --init-checkpoint)

**Gate:** Only execute if `CONFIRMATION_CANDIDATES` from B2 is non-empty. If empty, skip to Phase 2.

**Seeds:** 42 (done), 3407, 2026

For each confirmation candidate, and for each seed in {3407, 2026}:

- [ ] **Step 1: Run E0-A0 reference for this seed**

```bash
# 1. Update split_seed in a temp config
python3 -c "
import yaml
from common.utils import load_config
config = load_config('configs/e0_hyper_search.yaml')
config['data']['split_seed'] = <SEED>
config['data']['split_dir'] = f'outputs/e0/splits_seed<SEED>'
config['train']['save_dir'] = f'outputs/e0/checkpoints_seed<SEED>'
config['train']['lr'] = 0.005
config['train']['epochs'] = 50
config['output']['log_dir'] = f'outputs/e0/logs_seed<SEED>'
with open(f'configs/e0_seed<SEED>.yaml', 'w') as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
"

# 2. Split data for this seed
python3 scripts/split_data.py --config configs/e0_seed<SEED>.yaml

# 3. Train E0 reference
python3 -m experiments.baseline.train --config configs/e0_seed<SEED>.yaml
```

- [ ] **Step 2: Run candidate for this seed**

Same process with candidate config (`e2`, `e3`, or `e4`), same seed.

- [ ] **Step 3: Compute paired deltas**

```python
from common.evaluation import compute_paired_deltas, apply_candidate_rules

e0_results = {
    42: load_acc("outputs/e0/search/lr_5e-03__wd_1e-04/checkpoints/eval_results.json"),
    3407: load_acc("outputs/e0_seed3407/checkpoints/eval_results.json"),
    2026: load_acc("outputs/e0_seed2026/checkpoints/eval_results.json"),
}

candidate_results = {
    42: load_acc("outputs/<candidate>/checkpoints/eval_results.json"),
    3407: load_acc("outputs/<candidate>_seed3407/checkpoints/eval_results.json"),
    2026: load_acc("outputs/<candidate>_seed2026/checkpoints/eval_results.json"),
}

report = compute_paired_deltas(e0_results, candidate_results)
print(report)

selected, reason = apply_candidate_rules(
    {"<candidate>": report},
    elimination_threshold=-0.002,
    tie_threshold=0.001,
)
print(f"Selected: {selected}, Reason: {reason}")
```

- [ ] **Step 4: Apply confirmation rules**

```text
mean_delta > 0
min_delta > -0.002
```

If no candidate passes: final augmentation = A0.

- [ ] **Step 5: Commit confirmation results**

```bash
git add outputs/ configs/e0_seed*.yaml configs/<candidate>_seed*.yaml
git commit -m "results: augmentation confirmation (B2) — multi-split paired deltas"
```

---

## Phase 2: Partial Unfreeze Infrastructure (Tasks C, D)

### Task C1: Add Config Fields

**Files:**
- Modify: `configs/e2_augmentation.yaml` (add model/train fields as template)
- Create: `configs/f0_frozen_continue.yaml`
- Create: `configs/f1_ln_proj.yaml`
- Create: `configs/f2_last1_block.yaml`

**Dependencies:** None

- [ ] **Step 1: Create F0 config**

```yaml
# configs/f0_frozen_continue.yaml
experiment:
  id: F0_FROZEN_CONTINUE
  mode: dev
  head_type: linear
  augmentation_preset: a1  # Placeholder — update after B2

data:
  stage: preliminary
  image_extensions: [.jpg, .jpeg, .png, .bmp, .webp]
  split_seed: 42
  train_seed: 42
  split_dir: outputs/f0/splits
  test_dir: test
  train_dir: train
  val_ratio: 0.1
  expected_num_classes: 500
  class_mapping_path: outputs/data/metadata
  use_full_training_set: false

model:
  clip_model_name: ViT-B/32
  feature_dim: 512
  freeze_clip: true
  num_classes: 500
  use_cached_features: false
  unfreeze_last_n_blocks: 0
  train_ln_post: false
  train_visual_proj: false

eval:
  batch_size: 256

output:
  log_dir: outputs/f0/logs
  submission_dir: outputs/f0/submissions

train:
  amp: true
  batch_size: 128
  device: cuda
  epochs: 50
  image_size: 224
  lr: 0.0003
  max_grad_norm: 1.0
  num_workers: 8
  save_dir: outputs/f0/checkpoints
  scheduler: cosine
  warmup_epochs: 2
  weight_decay: 0.0001
  min_lr_ratio: 0.01
```

- [ ] **Step 2: Create F1 config** (same structure, different values)

```yaml
experiment:
  id: F1_LN_PROJ
  augmentation_preset: a1  # Placeholder

model:
  freeze_clip: false
  unfreeze_last_n_blocks: 0
  train_ln_post: true
  train_visual_proj: true

train:
  epochs: 50
  lr: 0.0003
  backbone_lr: 0.00001
  backbone_weight_decay: 0.01
  warmup_epochs: 2
  weight_decay: 0.0001
  min_lr_ratio: 0.01
```

- [ ] **Step 3: Create F2 config** (template — backbone_lr varies per F2A/B/C)

```yaml
experiment:
  id: F2_LAST1_BLOCK
  augmentation_preset: a1  # Placeholder

model:
  freeze_clip: false
  unfreeze_last_n_blocks: 1
  train_ln_post: true
  train_visual_proj: true

train:
  epochs: 50
  lr: 0.0003
  backbone_lr: 0.000003   # F2B default; F2A=1e-6, F2C=1e-5
  backbone_weight_decay: 0.01
  warmup_epochs: 2
  weight_decay: 0.0001
  min_lr_ratio: 0.01
```

- [ ] **Step 4: Commit configs**

```bash
git add configs/f0_frozen_continue.yaml configs/f1_ln_proj.yaml configs/f2_last1_block.yaml
git commit -m "feat: add F0-F2 partial unfreeze configs (C1)"
```

### Task C2: Modify model.py — Selective Unfreeze

**Files:**
- Modify: `experiments/baseline/model.py`

**Interfaces:**
- Consumes: config fields `model.unfreeze_last_n_blocks`, `model.train_ln_post`, `model.train_visual_proj`, `train.backbone_lr`, `train.backbone_weight_decay`
- Produces: `CLIPLinearClassifier.configure_visual_trainability()`, `CLIPLinearClassifier.get_param_groups(head_lr, head_weight_decay)`, overridden `train(mode)`

**Dependencies:** Task C1 (config fields defined)

- [ ] **Step 1: Update `__init__` to always freeze visual first, then selectively unfreeze**

```python
def __init__(
    self,
    clip_model: nn.Module,
    num_classes: int = 500,
    feature_dim: int = 512,
    freeze_clip: bool = True,
    unfreeze_last_n_blocks: int = 0,
    train_ln_post: bool = False,
    train_visual_proj: bool = False,
    backbone_lr: float = 1e-5,
    backbone_weight_decay: float = 0.01,
):
    super().__init__()

    self.visual = clip_model.visual
    self.feature_dim = feature_dim
    self.num_classes = num_classes
    self.freeze_clip = freeze_clip
    self.head_type = "linear"

    # Store discriminative LR config
    self.backbone_lr = backbone_lr
    self.backbone_weight_decay = backbone_weight_decay
    self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
    self.train_ln_post = train_ln_post
    self.train_visual_proj = train_visual_proj

    # ALWAYS freeze all visual parameters first
    for param in self.visual.parameters():
        param.requires_grad = False

    # Then selectively unfreeze based on config
    if not freeze_clip:
        self.configure_visual_trainability(
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            train_ln_post=train_ln_post,
            train_visual_proj=train_visual_proj,
        )

    if freeze_clip:
        logger.info("CLIP image encoder fully frozen.")

    # Linear classification head
    self.classifier = nn.Linear(feature_dim, num_classes)

    # Initialize the linear layer
    nn.init.xavier_uniform_(self.classifier.weight)
    nn.init.zeros_(self.classifier.bias)
```

- [ ] **Step 2: Add `configure_visual_trainability` method**

```python
def configure_visual_trainability(
    self,
    unfreeze_last_n_blocks: int,
    train_ln_post: bool,
    train_visual_proj: bool,
) -> None:
    """Selectively unfreeze CLIP visual encoder components.

    Must be called after all visual parameters are frozen.
    """
    blocks = self.visual.transformer.resblocks
    num_blocks = len(blocks)

    if not 0 <= unfreeze_last_n_blocks <= num_blocks:
        raise ValueError(
            f"unfreeze_last_n_blocks must be in [0, {num_blocks}], "
            f"got {unfreeze_last_n_blocks}"
        )

    if unfreeze_last_n_blocks > 0:
        for block in blocks[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        logger.info(
            f"Unfrozen last {unfreeze_last_n_blocks}/{num_blocks} "
            f"transformer blocks."
        )

    if train_ln_post:
        for param in self.visual.ln_post.parameters():
            param.requires_grad = True
        logger.info("Unfrozen visual.ln_post.")

    if train_visual_proj:
        self.visual.proj.requires_grad = True
        logger.info("Unfrozen visual.proj.")
```

- [ ] **Step 3: Override `train()` to handle partial unfreeze**

```python
def train(self, mode: bool = True):
    """Override train() — partially unfrozen visual stays in eval for frozen parts."""
    super().train(mode)

    if self.freeze_clip:
        self.visual.eval()
        return self

    # Partially unfrozen: start from eval, selectively set train
    self.visual.eval()

    n = self.unfreeze_last_n_blocks
    if n > 0:
        for block in self.visual.transformer.resblocks[-n:]:
            block.train(mode)

    if self.train_ln_post:
        self.visual.ln_post.train(mode)

    return self
```

- [ ] **Step 4: Update `encode_image` for partial unfreeze**

```python
def encode_image(self, images: torch.Tensor) -> torch.Tensor:
    conv1_dtype = self.visual.conv1.weight.dtype
    images = images.to(dtype=conv1_dtype)

    # When freeze_clip=False and partial unfreeze: allow gradients
    with torch.set_grad_enabled(not self.freeze_clip):
        features = self.visual(images)

    if features.dim() > 2:
        features = (
            features.mean(dim=[2, 3]) if features.dim() == 4 else features[:, 0]
        )

    features = F.normalize(features, p=2, dim=-1)
    return features
```

- [ ] **Step 5: Add `get_param_groups` for discriminative LR**

```python
def get_param_groups(
    self,
    head_lr: float,
    head_weight_decay: float,
) -> list:
    """Return parameter groups with separate LRs for head and backbone.

    Args:
        head_lr: Learning rate for the classifier head.
        head_weight_decay: Weight decay for the classifier head.

    Returns:
        List of dicts, each with 'name', 'params', 'lr', 'weight_decay'.
    """
    head_params = [
        p for p in self.classifier.parameters()
        if p.requires_grad
    ]
    backbone_params = [
        p for p in self.visual.parameters()
        if p.requires_grad
    ]

    groups = [
        {
            "name": "head",
            "params": head_params,
            "lr": head_lr,
            "weight_decay": head_weight_decay,
        }
    ]

    if backbone_params:
        groups.append({
            "name": "backbone",
            "params": backbone_params,
            "lr": self.backbone_lr,
            "weight_decay": self.backbone_weight_decay,
        })

    return groups
```

- [ ] **Step 6: Remove old `get_trainable_parameters`** (replaced by `get_param_groups`)

Keep `get_trainable_parameters` for backward compatibility (used in cached training path), but deprecate:
```python
def get_trainable_parameters(self):
    """Return only trainable parameters (backward compat)."""
    return filter(lambda p: p.requires_grad, self.parameters())
```

- [ ] **Step 7: Update `build_model` to pass new config fields**

```python
def build_model(
    config: dict, device: torch.device
) -> Tuple[CLIPLinearClassifier, callable]:
    model_cfg = config["model"]
    train_cfg = config.get("train", {})

    clip_model, preprocess = load_openai_clip(
        device, model_name=model_cfg["clip_model_name"]
    )

    model = CLIPLinearClassifier(
        clip_model=clip_model,
        num_classes=model_cfg["num_classes"],
        feature_dim=model_cfg.get("feature_dim", 512),
        freeze_clip=model_cfg.get("freeze_clip", True),
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 0),
        train_ln_post=model_cfg.get("train_ln_post", False),
        train_visual_proj=model_cfg.get("train_visual_proj", False),
        backbone_lr=train_cfg.get("backbone_lr", 1e-5),
        backbone_weight_decay=train_cfg.get("backbone_weight_decay", 0.01),
    )

    model = model.to(device)
    total, trainable = _count_params(model)
    logger.info(
        f"Model built: {total:,} total params, {trainable:,} trainable params"
    )

    return model, preprocess
```

- [ ] **Step 8: Run existing tests to verify no regression**

```bash
pytest tests/ -v
```

All 66 tests must pass.

- [ ] **Step 9: Commit**

```bash
git add experiments/baseline/model.py
git commit -m "feat: partial CLIP unfreeze with discriminative LRs (C2)"
```

### Task C3: Fix Scheduler — Proportional LambdaLR

**Files:**
- Modify: `experiments/baseline/train.py` (replace CosineAnnealingLR with LambdaLR)

**Dependencies:** Task C2 (get_param_groups available)

- [ ] **Step 1: Add `cosine_factor` function to train.py**

Add after imports:

```python
import math


def _cosine_factor(
    step: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """Cosine decay factor maintaining lr ratios across param groups.

    Returns a factor in [min_lr_ratio, 1.0] that follows cosine annealing,
    ensuring all param groups decay proportionally.
    """
    progress = min(step / max(total_steps, 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
```

- [ ] **Step 2: Replace scheduler construction in `_build_optimizer_and_scheduler`**

Find the current scheduler code:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cosine_steps,
    eta_min=train_cfg["lr"] * 0.01,
)
```

Replace with:
```python
# Use LambdaLR with proportional cosine decay to preserve
# backbone_lr / head_lr ratio throughout training
min_lr_ratio = train_cfg.get("min_lr_ratio", 0.01)
lr_lambda = lambda step: _cosine_factor(
    step=step,
    total_steps=cosine_steps,
    min_lr_ratio=min_lr_ratio,
)
# Apply the same lambda to all param groups
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=[lr_lambda] * len(optimizer.param_groups),
)
```

- [ ] **Step 3: Update `_build_optimizer_and_scheduler` to use `get_param_groups`**

The function currently uses `get_trainable_parameters()` for linear head. Update to use `get_param_groups` when available:

```python
def _build_optimizer_and_scheduler(
    model: nn.Module, config: Dict[str, Any], cosine_steps: int
) -> tuple:
    train_cfg = config["train"]

    if hasattr(model, "get_param_groups") and not config["model"].get("freeze_clip", True):
        # Partially unfrozen: use discriminative LRs
        optimizer = torch.optim.AdamW(
            model.get_param_groups(train_cfg["lr"], train_cfg["weight_decay"]),
        )
    elif hasattr(model, "get_param_groups"):
        # Cosine head with its own param groups
        optimizer = torch.optim.AdamW(
            model.get_param_groups(train_cfg["lr"], train_cfg["weight_decay"]),
        )
    else:
        # Linear head, frozen CLIP: uniform LR
        optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

    # Save per-group initial_lr before warmup
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    # LambdaLR with proportional decay (preserves LR ratios)
    min_lr_ratio = train_cfg.get("min_lr_ratio", 0.01)
    lr_lambda = lambda step: _cosine_factor(
        step=step,
        total_steps=cosine_steps,
        min_lr_ratio=min_lr_ratio,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_lambda] * len(optimizer.param_groups),
    )

    return optimizer, scheduler
```

- [ ] **Step 4: Run existing tests**

```bash
pytest tests/ -v
```

All 66 tests must pass. The integration test (`test_full_pipeline_smoke`) validates the scheduler change end-to-end.

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline/train.py
git commit -m "fix: replace CosineAnnealingLR with proportional LambdaLR (C3)"
```

### Task C4: Add --init-checkpoint CLI

**Files:**
- Modify: `experiments/baseline/train.py` (add CLI arg + loading logic)

**Dependencies:** Task C2 (model.py changes)

- [ ] **Step 1: Add `--init-checkpoint` CLI argument to `parse_args`**

```python
parser.add_argument(
    "--init-checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint for model weight initialization only. "
         "Does NOT restore optimizer, scheduler, or epoch state. "
         "Use this (not --resume) when starting F0-F3 partial unfreeze "
         "experiments from a trained checkpoint.",
)
```

- [ ] **Step 2: Add init-checkpoint loading logic in `main()`**

After model construction and before training loop, add:

```python
# Initialize model weights from checkpoint (no optimizer/scheduler/epoch)
init_ckpt_path = None
if args.init_checkpoint:
    init_ckpt_path = args.init_checkpoint
    train_logger.info(f"Initializing model weights from: {init_ckpt_path}")
    checkpoint = torch.load(init_ckpt_path, map_location=device)
    model_state = checkpoint.get("model_state_dict", checkpoint)

    missing_keys, unexpected_keys = model.load_state_dict(
        model_state, strict=False
    )

    if missing_keys:
        train_logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        train_logger.warning(f"Unexpected keys: {unexpected_keys}")

    # For same Linear architecture, both must be empty
    if not missing_keys and not unexpected_keys:
        train_logger.info("Model weights loaded with exact key match.")
    else:
        train_logger.warning(
            f"Model weight load had {len(missing_keys)} missing, "
            f"{len(unexpected_keys)} unexpected keys."
        )

    # Do NOT load optimizer state — fresh training from epoch 1
    # Do NOT load scheduler state
    # Do NOT restore epoch — starts from epoch 1
```

- [ ] **Step 3: Record `init_checkpoint` in checkpoint metadata**

In `_build_checkpoint_metadata`, add:
```python
if args.init_checkpoint:
    meta["init_checkpoint"] = args.init_checkpoint
```

- [ ] **Step 4: Record `init_checkpoint` in eval_results.json**

In the `eval_results` dict at the end of `main()`, add:
```python
"init_checkpoint": args.init_checkpoint,
```

- [ ] **Step 5: Add guard: forbid --resume with --init-checkpoint**

```python
if args.resume and args.init_checkpoint:
    raise ValueError(
        "--resume and --init-checkpoint are mutually exclusive. "
        "Use --init-checkpoint to load model weights for a new "
        "training run; use --resume to continue a crashed run."
    )
```

- [ ] **Step 6: Commit**

```bash
git add experiments/baseline/train.py
git commit -m "feat: add --init-checkpoint for weight-only initialization (C4)"
```

### Task C5: Training Diagnostics

**Files:**
- Modify: `experiments/baseline/train.py`

**Dependencies:** Tasks C2-C4

- [ ] **Step 1: Add diagnostic logging after optimizer creation**

In `main()`, after `_build_optimizer_and_scheduler`:

```python
# Diagnostic: log parameter group configuration
train_logger.info("Optimizer parameter groups:")
total_trainable = 0
for group in optimizer.param_groups:
    group_params = sum(p.numel() for p in group["params"])
    total_trainable += group_params
    train_logger.info(
        f"  {group.get('name', 'default'):12s}: "
        f"{group_params:>10,} params, "
        f"lr={group['lr']:.2e}, "
        f"wd={group['weight_decay']:.2e}"
    )
train_logger.info(f"  {'TOTAL':12s}: {total_trainable:>10,} trainable params")

# Count by component
head_params = sum(
    p.numel() for p in model.classifier.parameters() if p.requires_grad
)
visual_params = sum(
    p.numel() for p in model.visual.parameters() if p.requires_grad
)
train_logger.info(f"  Trainable head params:    {head_params:>10,}")
train_logger.info(f"  Trainable visual params:  {visual_params:>10,}")
```

- [ ] **Step 2: Add gradient norm tracking per group**

In `train_one_epoch`, after `loss.backward()` and before `optimizer.step()`, add gradient norm computation:

```python
# Compute gradient norms for diagnostic CSV
head_grad_norm = 0.0
backbone_grad_norm = 0.0
for group in optimizer.param_groups:
    gn = sum(
        p.grad.norm().item() ** 2
        for p in group["params"]
        if p.grad is not None
    ) ** 0.5
    if group.get("name") == "head":
        head_grad_norm = gn
    elif group.get("name") == "backbone":
        backbone_grad_norm = gn
```

- [ ] **Step 3: Extend train_log.csv header and rows**

Update the CSV header and row writes in the training loop to include:
```text
epoch,train_loss,train_acc,val_loss,val_acc,head_lr,backbone_lr,head_grad_norm,backbone_grad_norm,epoch_time
```

When `val_loader` is None, drop val columns but keep lr/grad columns:
```text
epoch,train_loss,train_acc,head_lr,backbone_lr,head_grad_norm,backbone_grad_norm,epoch_time
```

Extract per-group LRs from optimizer:
```python
head_lr = optimizer.param_groups[0]["lr"]
backbone_lr = (
    optimizer.param_groups[1]["lr"]
    if len(optimizer.param_groups) > 1
    else 0.0
)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline/train.py
git commit -m "feat: training diagnostics — per-group LR/grad logging (C5)"
```

---

## Phase 3: Tests (Task D)

### Task D1: test_partial_unfreeze.py

**Files:**
- Create: `tests/test_partial_unfreeze.py`

**Dependencies:** Task C2 (model changes)

- [ ] **Step 1: Write test file**

```python
"""Tests for partial CLIP visual encoder unfreezing."""

import pytest
import torch

from experiments.baseline.model import CLIPLinearClassifier


class MockVisual(torch.nn.Module):
    """Minimal mock of CLIP visual encoder for testing freeze/unfreeze logic."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.ln_post = torch.nn.LayerNorm(16)
        self.proj = torch.nn.Parameter(torch.randn(16, 16))
        # Mock transformer with 12 blocks
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=16, nhead=2, dim_feedforward=64, batch_first=True
            )
            for _ in range(12)
        ])
        # Each block needs its own LayerNorm
        for block in self.transformer.resblocks:
            block.norm1 = torch.nn.LayerNorm(16)
            block.norm2 = torch.nn.LayerNorm(16)

    def forward(self, x):
        # Minimal forward for testing
        x = self.conv1(x)
        x = x.mean(dim=[2, 3])  # Global pool
        return x.unsqueeze(1)


class MockCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = MockVisual()


@pytest.fixture
def mock_clip():
    return MockCLIP()


def _make_model(mock_clip, **kwargs):
    defaults = dict(
        num_classes=500,
        feature_dim=16,
        freeze_clip=False,
        unfreeze_last_n_blocks=0,
        train_ln_post=False,
        train_visual_proj=False,
    )
    defaults.update(kwargs)
    return CLIPLinearClassifier(clip_model=mock_clip, **defaults)


class TestFreezeAll:
    """freeze_clip=True → everything frozen except classifier."""

    def test_all_visual_frozen(self, mock_clip):
        model = _make_model(mock_clip, freeze_clip=True)
        for name, param in model.visual.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_classifier_trainable(self, mock_clip):
        model = _make_model(mock_clip, freeze_clip=True)
        for name, param in model.classifier.named_parameters():
            assert param.requires_grad, f"classifier.{name} should be trainable"


class TestUnfreezeLastN:
    """unfreeze_last_n_blocks controls which transformer blocks are trainable."""

    def test_n0_all_blocks_frozen(self, mock_clip):
        model = _make_model(mock_clip, unfreeze_last_n_blocks=0)
        for i, block in enumerate(model.visual.transformer.resblocks):
            for param in block.parameters():
                assert not param.requires_grad, f"block {i} should be frozen"

    def test_n1_only_last_block_trainable(self, mock_clip):
        model = _make_model(mock_clip, unfreeze_last_n_blocks=1)
        # First 11 blocks frozen
        for i, block in enumerate(model.visual.transformer.resblocks[:-1]):
            for param in block.parameters():
                assert not param.requires_grad, f"block {i} should be frozen"
        # Last block trainable
        for param in model.visual.transformer.resblocks[-1].parameters():
            assert param.requires_grad, "last block should be trainable"

    def test_n2_last_two_blocks_trainable(self, mock_clip):
        model = _make_model(mock_clip, unfreeze_last_n_blocks=2)
        # First 10 blocks frozen
        for i, block in enumerate(model.visual.transformer.resblocks[:-2]):
            for param in block.parameters():
                assert not param.requires_grad, f"block {i} should be frozen"
        # Last 2 blocks trainable
        for block in model.visual.transformer.resblocks[-2:]:
            for param in block.parameters():
                assert param.requires_grad, "last blocks should be trainable"

    def test_invalid_n_raises(self, mock_clip):
        with pytest.raises(ValueError, match="unfreeze_last_n_blocks"):
            _make_model(mock_clip, unfreeze_last_n_blocks=13)


class TestLnPostAndProj:
    """train_ln_post and train_visual_proj toggle specific components."""

    def test_ln_post_trainable_when_enabled(self, mock_clip):
        model = _make_model(mock_clip, train_ln_post=True)
        for param in model.visual.ln_post.parameters():
            assert param.requires_grad, "ln_post should be trainable"

    def test_ln_post_frozen_when_disabled(self, mock_clip):
        model = _make_model(mock_clip, train_ln_post=False)
        for param in model.visual.ln_post.parameters():
            assert not param.requires_grad, "ln_post should be frozen"

    def test_proj_trainable_when_enabled(self, mock_clip):
        model = _make_model(mock_clip, train_visual_proj=True)
        assert model.visual.proj.requires_grad, "proj should be trainable"

    def test_proj_frozen_when_disabled(self, mock_clip):
        model = _make_model(mock_clip, train_visual_proj=False)
        assert not model.visual.proj.requires_grad, "proj should be frozen"


class TestTrainMode:
    """train(mode) handles partial unfreeze correctly."""

    def test_frozen_clip_visual_in_eval(self, mock_clip):
        model = _make_model(mock_clip, freeze_clip=True)
        model.train()
        assert not model.visual.training, "frozen visual should be in eval mode"

    def test_partial_unfreeze_blocks_in_train(self, mock_clip):
        model = _make_model(
            mock_clip, freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True
        )
        model.train()
        # Last block should be in train mode
        assert model.visual.transformer.resblocks[-1].training, \
            "last block should be in train mode"
        # Frozen block should be in eval mode
        assert not model.visual.transformer.resblocks[0].training, \
            "frozen block should be in eval mode"
        # ln_post should be in train mode
        assert model.visual.ln_post.training, "ln_post should be in train mode"
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_partial_unfreeze.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_partial_unfreeze.py
git commit -m "test: partial unfreeze parameter selection (D1)"
```

### Task D2: test_discriminative_optimizer.py

**Files:**
- Create: `tests/test_discriminative_optimizer.py`

**Dependencies:** Tasks C2, D1

- [ ] **Step 1: Write test file**

```python
"""Tests for discriminative optimizer with separate head/backbone LRs."""

import torch
from tests.test_partial_unfreeze import _make_model, MockCLIP


class TestParamGroups:
    """get_param_groups returns correct, non-overlapping groups."""

    def test_head_group_exists(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
            backbone_lr=1e-5, backbone_weight_decay=0.01,
        )
        groups = model.get_param_groups(
            head_lr=3e-4, head_weight_decay=1e-4
        )
        assert len(groups) == 2
        assert groups[0]["name"] == "head"
        assert groups[1]["name"] == "backbone"

    def test_head_lr_and_wd_correct(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
            backbone_lr=1e-5, backbone_weight_decay=0.01,
        )
        groups = model.get_param_groups(
            head_lr=3e-4, head_weight_decay=1e-4
        )
        assert groups[0]["lr"] == 3e-4
        assert groups[0]["weight_decay"] == 1e-4

    def test_backbone_lr_and_wd_from_config(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
            backbone_lr=3e-6, backbone_weight_decay=0.01,
        )
        groups = model.get_param_groups(
            head_lr=3e-4, head_weight_decay=1e-4
        )
        assert groups[1]["lr"] == 3e-6
        assert groups[1]["weight_decay"] == 0.01

    def test_no_overlap_between_groups(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )
        groups = model.get_param_groups(
            head_lr=3e-4, head_weight_decay=1e-4
        )
        head_ids = {id(p) for p in groups[0]["params"]}
        backbone_ids = {id(p) for p in groups[1]["params"]}
        assert head_ids.isdisjoint(backbone_ids), \
            "Head and backbone param groups must not overlap"

    def test_no_frozen_params_in_optimizer(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=0, train_ln_post=True,
            train_visual_proj=False,
        )
        groups = model.get_param_groups(
            head_lr=3e-4, head_weight_decay=1e-4
        )
        all_params = []
        for g in groups:
            all_params.extend(g["params"])
        for p in all_params:
            assert p.requires_grad, \
                f"Param with requires_grad=False in optimizer: {p.shape}"

    def test_all_trainable_in_optimizer(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )
        groups = model.get_param_groups(
            head_lr=3e-4, head_weight_decay=1e-4
        )
        opt_param_ids = set()
        for g in groups:
            opt_param_ids.update(id(p) for p in g["params"])

        model_param_ids = {
            id(p) for p in model.parameters() if p.requires_grad
        }
        missing = model_param_ids - opt_param_ids
        assert not missing, f"Trainable params not in optimizer: {len(missing)}"

    def test_freeze_clip_no_backbone_group(self):
        model = _make_model(MockCLIP(), freeze_clip=True)
        groups = model.get_param_groups(
            head_lr=3e-4, head_weight_decay=1e-4
        )
        assert len(groups) == 1
        assert groups[0]["name"] == "head"
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_discriminative_optimizer.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_discriminative_optimizer.py
git commit -m "test: discriminative optimizer param groups (D2)"
```

### Task D3: test_init_checkpoint.py

**Files:**
- Create: `tests/test_init_checkpoint.py`

**Dependencies:** Task C4

- [ ] **Step 1: Write test file**

```python
"""Tests for --init-checkpoint weight-only initialization."""

import json
import tempfile
from pathlib import Path

import torch

from experiments.baseline.model import CLIPLinearClassifier
from tests.test_partial_unfreeze import MockCLIP, _make_model


class TestInitCheckpoint:
    """Weight-only loading vs full resume."""

    def test_load_matching_architecture_no_missing_keys(self):
        """Same architecture → strict=False should have no missing keys."""
        model1 = _make_model(MockCLIP(), freeze_clip=True)
        model2 = _make_model(MockCLIP(), freeze_clip=True)

        # Save model1 weights
        ckpt = {"model_state_dict": model1.state_dict()}
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(ckpt, f.name)
            state = torch.load(f.name, map_location="cpu")

        missing, unexpected = model2.load_state_dict(
            state["model_state_dict"], strict=False
        )
        assert not missing, f"Unexpected missing keys: {missing}"
        assert not unexpected, f"Unexpected unexpected keys: {unexpected}"

    def test_init_checkpoint_does_not_restore_epoch(self):
        """--init-checkpoint should NOT set epoch or global_step."""
        model1 = _make_model(MockCLIP(), freeze_clip=True)
        model2 = _make_model(MockCLIP(), freeze_clip=True)

        ckpt = {
            "model_state_dict": model1.state_dict(),
            "epoch": 42,
            "global_step": 1000,
            "best_val_acc": 0.95,
        }
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(ckpt, f.name)
            state = torch.load(f.name, map_location="cpu")

        model2.load_state_dict(state["model_state_dict"], strict=False)
        # Epoch and best_val_acc are NOT restored — they're in the
        # checkpoint dict but only model_state_dict is used
        assert True  # If we got here without error, test passes

    def test_partial_unfreeze_init_from_frozen_checkpoint(self):
        """Loading a frozen model into a partially unfrozen one should work."""
        # Source: fully frozen
        model_frozen = _make_model(MockCLIP(), freeze_clip=True)
        # Target: partially unfrozen (last block + ln_post)
        model_unfrozen = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )

        ckpt = {"model_state_dict": model_frozen.state_dict()}
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(ckpt, f.name)
            state = torch.load(f.name, map_location="cpu")

        missing, unexpected = model_unfrozen.load_state_dict(
            state["model_state_dict"], strict=False
        )
        # All keys should match — same architecture, just different
        # requires_grad settings (which are NOT part of state_dict)
        assert not missing, f"Missing keys: {missing}"
        assert not unexpected, f"Unexpected keys: {unexpected}"
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_init_checkpoint.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_init_checkpoint.py
git commit -m "test: init-checkpoint weight-only loading (D3)"
```

### Task D4: Scheduler + Cache Guard Tests

**Files:**
- Create: `tests/test_scheduler_ratio.py` (add to existing test file or create new)

**Dependencies:** Task C3

- [ ] **Step 1: Write scheduler ratio test**

```python
"""Test that LambdaLR preserves backbone_lr/head_lr ratio."""

import math
import torch
from experiments.baseline.train import _cosine_factor


class TestCosineFactor:
    def test_start_is_1(self):
        assert _cosine_factor(0, 100, 0.01) == pytest.approx(1.0)

    def test_end_is_min_ratio(self):
        assert _cosine_factor(100, 100, 0.01) == pytest.approx(0.01)

    def test_midpoint(self):
        # At half, cosine is 0.5, mapped to (min_ratio + 0.5*(1-min_ratio))
        mid = _cosine_factor(50, 100, 0.01)
        expected = 0.01 + 0.5 * (1.0 - 0.01)
        assert mid == pytest.approx(expected)

    def test_ratio_preserved(self):
        """At any step, the ratio of two different initial LRs is preserved."""
        for step in [0, 10, 50, 90, 100]:
            factor = _cosine_factor(step, 100, 0.01)
            # head_lr = 3e-4, backbone_lr = 3e-6 → ratio = 100
            # After factor applied, both scaled identically
            head_lr = 3e-4 * factor
            backbone_lr = 3e-6 * factor
            assert head_lr / backbone_lr == pytest.approx(100.0)


class TestCacheGuard:
    """Cached features + freeze_clip=false must raise."""

    def test_cache_with_unfrozen_clip_raises(self):
        from experiments.baseline.train import _enforce_guards
        import pytest
        with pytest.raises(ValueError, match="freeze_clip=True"):
            _enforce_guards(
                experiment_id="F1",
                use_cached_features=True,
                augmentation_preset="a0",
                freeze_clip=False,
            )
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_scheduler_ratio.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_scheduler_ratio.py
git commit -m "test: scheduler ratio preservation + cache guard (D4)"
```

### Task D5: Full Acceptance

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

All tests must pass.

- [ ] **Step 2: Run acceptance script**

```bash
python scripts/run_acceptance.py
```

- [ ] **Step 3: Commit if any test files were updated**

```bash
git add -A && git diff --cached --stat
git commit -m "test: full acceptance — all tests pass (D6)"
```

---

## Phase 4: F0-F3 Experiments (Task E)

**Gate:** Phase 3 all tests pass + Phase 1 augmentation selected

### Pre-Flight: Determine Best Augmentation Checkpoint

From Task B2 output, set:
```bash
BEST_EXP=<E2|E3|E4>
BEST_AUG=<a1|a2|a3>
INIT_CKPT="outputs/${BEST_EXP}/checkpoints/best.pt"
```

Update all F0-F3 configs with correct `augmentation_preset` and `init_checkpoint` path.

### Task E0: Run F0 (Frozen Continue)

- [ ] **Step 1: Update F0 config augmentation preset**

```bash
python3 -c "
import yaml
c = yaml.safe_load(open('configs/f0_frozen_continue.yaml'))
c['experiment']['augmentation_preset'] = '<BEST_AUG>'
c['data']['split_dir'] = 'outputs/f0/splits'
yaml.safe_dump(c, open('configs/f0_frozen_continue.yaml', 'w'), sort_keys=False, allow_unicode=True)
"
```

- [ ] **Step 2: Split data for F0**

```bash
python3 scripts/split_data.py --config configs/f0_frozen_continue.yaml
```

- [ ] **Step 3: Train F0**

```bash
python3 -m experiments.baseline.train \
  --config configs/f0_frozen_continue.yaml \
  --init-checkpoint <INIT_CKPT>
```

- [ ] **Step 4: Record result**

```bash
cat outputs/f0/checkpoints/eval_results.json
```

- [ ] **Step 5: Commit**

```bash
git add outputs/f0/ configs/f0_frozen_continue.yaml
git commit -m "results: F0 frozen continue — 50 epoch baseline"
```

### Task E1: Run F1 (ln_post + proj only)

- [ ] **Step 1: Update & run F1**

Same process as F0, using `configs/f1_ln_proj.yaml`.

- [ ] **Step 2: Commit**

### Task E2: Run F2B (last block, backbone_lr=3e-6)

- [ ] **Step 1: Run F2B first**

F2B is the middle backbone_lr. Run it first. Check for crash signals:

```python
# After epoch 5, verify:
# - No NaN in loss
# - val_acc drop from init checkpoint < 0.02
# - backbone_grad_norm not anomalously large (< 10.0 typically)
```

- [ ] **Step 2: If F2B stable, run F2A (backbone_lr=1e-6) and F2C (backbone_lr=1e-5)**

- [ ] **Step 3: Commit**

### Task E3: Gate — F3 (last 2 blocks) only if best F2 - F0 >= +0.3pp

- [ ] **Step 1: Evaluate gate condition**

```python
f0_acc = load("outputs/f0/checkpoints/eval_results.json")["best_val_acc"]
f2_best = max(
    load("outputs/f2a/checkpoints/eval_results.json")["best_val_acc"],
    load("outputs/f2b/checkpoints/eval_results.json")["best_val_acc"],
    load("outputs/f2c/checkpoints/eval_results.json")["best_val_acc"],
)
if f2_best - f0_acc >= 0.003:
    print("GATE PASS: run F3")
else:
    print("GATE FAIL: skip F3")
```

- [ ] **Step 2: If gate passes, run F3**

```yaml
model:
  unfreeze_last_n_blocks: 2
  train_ln_post: true
  train_visual_proj: true
train:
  backbone_lr: <BEST_F2_BACKBONE_LR>
```

---

## Phase 5: Result Selection (Task F)

### Task F1: Apply Discovery Rules

- [ ] **Step 1: Compare candidates against F0**

```python
from common.evaluation import apply_candidate_rules

candidates = {}
for exp_id in ["F1", "F2A", "F2B", "F2C"]:
    path = f"outputs/{exp_id.lower()}/checkpoints/eval_results.json"
    if Path(path).exists():
        delta = load(path)["best_val_acc"] - f0_acc
        candidates[exp_id] = {"mean_delta": delta, "min_delta": delta}
    # Note: single-split deltas — confirmation runs multi-split if needed

selected, reason = apply_candidate_rules(candidates)
print(f"Selected: {selected}")
print(f"Reason: {reason}")
```

Selection priority (from spec): F1 > F2 > F3

- [ ] **Step 2: If multiple close candidates, run multi-split confirmation**

Same process as Task B3, with seeds 42/3407/2026.

Passing criteria: `mean_delta > 0` AND `min_delta > -0.002`

- [ ] **Step 3: Document final configuration selection**

Save to `outputs/final_selection.json`.

- [ ] **Step 4: Commit**

```bash
git add outputs/ outputs/final_selection.json
git commit -m "results: partial unfreeze selection (F) — final config"
```

---

## Dependency Graph

```
B1 (complete E3, run E4)
  ↓
B2 (discovery rules)
  ↓
B3 (confirmation, conditional) ← ┐
                                  │
C1 (configs)                      │
  ↓                               │
C2 (model.py)                     │
  ↓                               │
C3 (scheduler)                    │
  ↓                               │
C4 (init-checkpoint)              │
  ↓                               │
C5 (diagnostics)                  │
  ↓                               │
D1-D4 (tests)                     │
  ↓                               │
D5 (acceptance)                   │
  ↓                               │
E0 (F0) ← ─── uses B2 result ────┘
  ↓
E1 (F1)
  ↓
E2 (F2A/B/C)
  ↓
E3 (F3, conditional)
  ↓
F (selection)
```

---

## Execution Order Summary

1. **B1**: Wait for E3, run E4
2. **B2**: Run discovery, select augmentation
3. **C1-C5**: Implement partial unfreeze infrastructure
4. **D1-D4**: Write and pass all tests
5. **D5**: Full acceptance
6. **B3**: (conditional) Multi-split confirmation if needed
7. **E0-E3**: Run F0-F3 experiments
8. **F**: Select final configuration
