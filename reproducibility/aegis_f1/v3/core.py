"""Paired draws, deterministic augmentation, and weighted mixed supervision."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch
from torch.nn import functional as F

from v2 import training_utils
from v2.plan import require


def replay_seed(seed, *parts):
    text = "/".join(map(str, ("v3", seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


@contextmanager
def isolated_cpu_rng(seed):
    """Only CPU RNGs; augmentation/mixing never depends on either arm's loss."""
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)


def paired_draws(labels, classes, seed, epochs, batch_size):
    from v2.core import sample_weights
    weights = sample_weights(labels, classes)  # ORIGINAL class counts for both arms.
    count = len(labels) // batch_size * batch_size
    require(count > 0, "Fewer training rows than the logical batch")
    return np.stack([
        torch.multinomial(weights, len(labels), replacement=True,
                          generator=torch.Generator(device="cpu").manual_seed(
                              replay_seed(seed, "sampler", epoch)))[:count].numpy()
        for epoch in range(1, epochs + 1)
    ])


def mix_supervision(images, probabilities, weights, augment, seed):
    require(images.device.type == "cpu", "Pair mixing must use the common CPU replay")
    require(probabilities.shape[0] == len(images) == len(weights), "Supervision row mismatch")
    require(torch.isfinite(weights).all() and (weights >= 0).all(), "Invalid reliability weights")
    # Weight each original contribution BEFORE Mixup/CutMix. The permutation
    # preserves total mass, including the actual clipped CutMix rectangle area.
    masses = probabilities * weights[:, None]
    with isolated_cpu_rng(seed):
        images, masses, permutation = training_utils.apply_mixup_cutmix(
            images, masses, augment["mixup"], augment["cutmix"],
            augment["mix_prob"], probabilities.shape[1])
    trace = dict(mix_seed=seed, permutation=None if permutation is None else permutation.tolist())
    return images, masses, weights.sum().clamp_min(1e-8), trace


def weighted_backward(model, images, masses, denominator, micro_batch_size, autocast):
    """One logical loss denominator, also for a short final microbatch."""
    require(micro_batch_size > 0 and len(images) == len(masses), "Invalid logical batch")
    require(torch.isfinite(masses).all() and (masses >= 0).all(), "Invalid weighted targets")
    require(torch.isfinite(denominator) and denominator > 0, "Invalid logical denominator")
    total = 0.0
    for start in range(0, len(images), micro_batch_size):
        end = min(start + micro_batch_size, len(images))
        with autocast():
            logits = model(images[start:end]).float()
            loss = -(masses[start:end] * F.log_softmax(logits, dim=1)).sum() / denominator
        require(torch.isfinite(loss), "Nonfinite classification loss")
        loss.backward()
        total += float(loss.detach())
    return total


class PairedImages(torch.utils.data.Dataset):
    """Same image slots, duplicate occurrences and augmentations in both arms."""
    def __init__(self, root, records, transform, draws, seed, epoch):
        self.root, self.records, self.transform = root, records, transform
        self.draws, self.seed, self.epoch = draws, seed, epoch

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, position):
        index = int(self.draws[position])
        row = self.records[index]
        path = (self.root / row["relative_path"]).resolve()
        require(path.is_relative_to(self.root.resolve()), "Image escapes the official training root")
        image = training_utils.load_image(path)
        with isolated_cpu_rng(replay_seed(self.seed, "image", self.epoch, position, index)):
            image = self.transform(image)
        return image, index
