"""Standard training mechanisms adapted from Palm's MIT-licensed reference.

Source: Palm0palM/aic-noisy-clip @ fcc37813c8bea3f2761707a145b2955ce95c097e
solutions/clip-vitb32-full-finetune/src/aic_clip/train_ft.py and infer_ft.py
Copyright (c) 2026 Palm. License: adjacent LICENSE.
Only mechanism helpers are retained; data, backbone, workflow and budgets are local.
"""
from __future__ import annotations
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as T
ImageFile.LOAD_TRUNCATED_IMAGES = False
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_image(path: Path, decode_cap: int = 0) -> Image.Image:
    with Image.open(path) as im:
        if decode_cap and im.format == 'JPEG':
            im.draft('RGB', (decode_cap, decode_cap))
        im.load()
        try:
            im = ImageOps.exif_transpose(im)
        except (ValueError, TypeError, OSError):
            pass
        return im.convert('RGB').copy()

class ManifestDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform, decode_cap: int = 0):
        self.root = root
        self.records = records
        self.transform = transform
        self.decode_cap = decode_cap

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = load_image(self.root / rec["relative_path"], self.decode_cap)
        return self.transform(image), rec["label"], index

def read_manifest(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "relative_path": row["relative_path"],
            "label": int(row["label"]),
            "index": int(row.get("index", i)),
        }
        for i, row in enumerate(rows)
    ]

def build_train_transform(image_size: int, augment: dict):
    ops = [
        T.RandomResizedCrop(
            image_size,
            scale=tuple(augment.get("rrc_scale", (0.35, 1.0))),
            ratio=tuple(augment.get("rrc_ratio", (3 / 4, 4 / 3))),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(),
    ]
    jitter = float(augment.get("color_jitter", 0.4))
    if jitter > 0:
        ops.append(T.ColorJitter(jitter, jitter, jitter, jitter / 4))
    if augment.get("rand_augment", True):
        ops.append(
            T.RandAugment(
                num_ops=int(augment.get("rand_augment_ops", 2)),
                magnitude=int(augment.get("rand_augment_magnitude", 9)),
                interpolation=T.InterpolationMode.BICUBIC,
            )
        )
    ops += [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    if float(augment.get("random_erase", 0.25)) > 0:
        ops.append(T.RandomErasing(p=float(augment["random_erase"]), value="random"))
    return T.Compose(ops)

def build_eval_transform(image_size: int, resize_ratio: float = 1.14):
    resize = int(round(image_size * resize_ratio))
    return T.Compose(
        [
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )

def build_optimizer(model: nn.Module, cfg: dict):
    """Build AdamW groups, optionally with layer-wise LR decay (LLRD).

    With llrd_gamma < 1 the vision-tower layers get lr * gamma^(n_layers - depth),
    i.e. earlier layers train more slowly. The head always uses lr_head.
    """
    head_ids = {id(p) for p in model.head.parameters()}
    wd = float(cfg["weight_decay"])
    lr_bb = float(cfg["lr_backbone"])
    gamma = float(cfg.get("llrd_gamma", 1.0))
    n_layers = len(model.vision.vision_model.encoder.layers) if gamma != 1.0 else 0

    groups: dict[tuple[float, float], list] = {}

    def add(param, lr, weight_decay):
        key = (round(lr, 12), weight_decay)
        groups.setdefault(key, []).append(param)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        decay = 0.0 if no_decay else wd
        if id(param) in head_ids:
            add(param, float(cfg["lr_head"]), decay)
            continue
        lr = lr_bb
        if gamma != 1.0:
            depth = None
            if "encoder.layers." in name:
                depth = int(name.split("encoder.layers.")[1].split(".")[0])
            elif "pre_layrnorm" in name or "embeddings" in name:
                depth = -1
            if depth is not None:
                lr = lr_bb * (gamma ** (n_layers - 1 - depth if depth >= 0 else n_layers))
        add(param, lr, decay)

    return torch.optim.AdamW(
        [{"params": params, "lr": lr, "weight_decay": decay} for (lr, decay), params in groups.items()],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for key, value in model.state_dict().items():
            stored = self.state[key]
            if stored.dtype.is_floating_point:
                stored.mul_(self.decay).add_(value.detach().float(), alpha=1.0 - self.decay)
            else:
                stored.copy_(value)

def one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    target = torch.full((labels.size(0), num_classes), smoothing / num_classes, device=labels.device)
    target.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing + smoothing / num_classes)
    return target

def rand_bbox(height: int, width: int, lam: float, device):
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = torch.randint(height, (1,), device=device).item(), torch.randint(width, (1,), device=device).item()
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    return y1, y2, x1, x2

def apply_mixup_cutmix(images, targets, mixup_alpha, cutmix_alpha, prob, num_classes):
    if prob <= 0 or random.random() > prob:
        return images, targets, None
    index = torch.randperm(images.size(0), device=images.device)
    if mixup_alpha > 0 and (cutmix_alpha <= 0 or random.random() < 0.5):
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        images = images * lam + images[index] * (1.0 - lam)
    elif cutmix_alpha > 0:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        y1, y2, x1, x2 = rand_bbox(images.size(2), images.size(3), lam, images.device)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (images.size(2) * images.size(3)))
    else:
        return images, targets, None
    return images, lam * targets + (1.0 - lam) * targets[index], index

def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_scheduler(optimizer, epochs: float, warmup_epochs: float, steps_per_epoch: int, min_lr_ratio: float):
    total_steps = max(int(epochs * steps_per_epoch), 1)
    warmup_steps = max(int(warmup_epochs * steps_per_epoch), 1)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def build_view(name, image_size):
    flip = name == 'flip'
    if name in ('center','flip'):
        resize = round(image_size * 1.14)
    elif name.startswith('resize'):
        resize = round(float(name[6:]))
    else:
        raise ValueError(f'Unknown deterministic view: {name}')
    ops = [T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(image_size)]
    if flip:
        ops.append(T.RandomHorizontalFlip(p=1.0))
    return T.Compose(ops + [T.ToTensor(),T.Normalize(CLIP_MEAN,CLIP_STD)])
