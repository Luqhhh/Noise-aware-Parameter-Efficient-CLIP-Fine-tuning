"""Paired random streams, isolated transforms, and predeclared readouts."""
from __future__ import annotations

import contextlib
import hashlib
import random

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms as T

from aegis_clip.v1_pipeline import Images, MEAN, STD

ARMS = ("original_augmentation", "strong_augmentation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def training_transform(arm):
    require(arm in ARMS, "Unknown arm")
    strong = arm == ARMS[1]
    ops = [T.RandomResizedCrop(448, scale=(.35 if strong else .8, 1.),
                              interpolation=T.InterpolationMode.BICUBIC),
           T.RandomHorizontalFlip()]
    if strong:
        ops.append(T.ColorJitter(.5, .5, .5, .125))
    ops.extend([T.RandAugment(2, 7), T.ToTensor(), T.Normalize(MEAN, STD)])
    if strong:
        ops.append(T.RandomErasing(p=.3, value="random"))
    return T.Compose(ops)


@contextlib.contextmanager
def augmentation_rng(seed):
    """Worker-local random scope; never consume sampler/Mixup RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed % 2**32)
        # CPU generator only: dataset workers must not initialize CUDA.
        torch.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class PairedImages(Images):
    def __init__(self, root, rows, arm, epoch, seed):
        super().__init__(root, rows, training_transform(arm))
        self.epoch, self.seed = epoch, seed

    def __getitem__(self, index):
        seed = int.from_bytes(hashlib.sha256(
            f"augmentation:{self.seed}:{self.epoch}:{index}".encode()).digest()[:8], "little") % (2**63-1)
        with augmentation_rng(seed):
            return super().__getitem__(index)


def epoch_order(rows, seed, epoch):
    return torch.randperm(rows, generator=torch.Generator().manual_seed(seed + 100000 * epoch)).tolist()


def mixup_draw(size, seed, epoch, batch):
    seed_sequence = np.random.SeedSequence([seed, epoch, batch, 736])
    rng = np.random.default_rng(seed_sequence)
    return torch.from_numpy(rng.permutation(size)), float(rng.beta(.2, .2))


@torch.no_grad()
def group_excluded_maximum(query, gallery, query_groups, gallery_groups, *,
                           query_chunk=256, gallery_chunk=8192, progress=None):
    require(len(gallery) > 0 and query_chunk > 0 and gallery_chunk > 0, "Invalid neighbour dimensions")
    query, gallery = F.normalize(query.float(), dim=1), F.normalize(gallery.float(), dim=1)
    require(torch.isfinite(query).all() and torch.isfinite(gallery).all(), "Nonfinite features")
    require(query.shape[1] == gallery.shape[1], "Feature dimensions disagree")
    output = []
    for start in range(0, len(query), query_chunk):
        q = query[start:start+query_chunk]
        best = q.new_full((len(q),), -torch.inf)
        for gs in range(0, len(gallery), gallery_chunk):
            similarities = q @ gallery[gs:gs+gallery_chunk].T
            if query_groups is not None:
                require(gallery_groups is not None, "Missing gallery groups")
                similarities.masked_fill_(query_groups[start:start+len(q), None] ==
                                          gallery_groups[None, gs:gs+similarities.shape[1]], -torch.inf)
            best = torch.maximum(best, similarities.max(1).values)
        require(torch.isfinite(best).all(), "No group-excluded training neighbour")
        output.append(best.cpu())
        if progress:
            progress(min(start+len(q), len(query)), len(query))
    return torch.cat(output).numpy()


def metrics(labels, predictions, classes, mask=None):
    mask = np.ones(len(labels), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    labels, predictions = np.asarray(labels)[mask], np.asarray(predictions)[mask]
    counts = np.bincount(labels, minlength=classes)
    correct = np.bincount(labels[predictions == labels], minlength=classes)
    return dict(rows=len(labels), correct=int(correct.sum()),
                micro=float(correct.sum()/len(labels)) if len(labels) else None,
                macro=float((correct[counts > 0]/counts[counts > 0]).mean()) if len(labels) else None,
                covered_classes=int((counts > 0).sum()))


def paired(labels, baseline, candidate, classes, groups):
    result = {}
    for name, mask in groups.items():
        correction = int(((baseline != labels) & (candidate == labels) & mask).sum())
        regression = int(((baseline == labels) & (candidate != labels) & mask).sum())
        result[name] = dict(baseline=metrics(labels, baseline, classes, mask),
                            candidate=metrics(labels, candidate, classes, mask),
                            corrections=correction, regressions=regression, net=correction-regression,
                            changed=int(((baseline != candidate) & mask).sum()))
    return result


def review_decision(control_comparison, parent_comparison):
    conditions = dict(net_vs_control=control_comparison["all"]["net"] >= 75,
                      net_vs_parent=parent_comparison["all"]["net"] >= 75,
                      target_net_vs_control=control_comparison["target"]["net"] >= 25)
    for name, comparison in (("control", control_comparison), ("parent", parent_comparison)):
        row = comparison["all"]
        conditions[f"corrections_regressions_{name}"] = (
            row["corrections"] > 0 and row["corrections"] >= 1.25 * row["regressions"])
    # Explicitly flag regressions in the predeclared complement and tail groups.
    conditions["other_large_groups_not_net_harmed"] = all(
        comparison[group]["net"] >= 0 for comparison in (control_comparison, parent_comparison)
        for group in ("complement", "tail75"))
    return dict(status="supports_review" if all(conditions.values()) else "closed_fixed_recipe",
                conditions=conditions, automatic_full_training=False, platform_gain_known=False)
