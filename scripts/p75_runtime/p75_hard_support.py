"""Conservative original-label supervision restoration for a fixed support CSV."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_support(config: dict, paths: list[str], labels: list[int]) -> torch.Tensor | None:
    options = config['loss'].get('hard_support', {})
    if not options.get('enabled', False):
        return None
    if options.get('version') != 1 or float(options.get('maximum_weight', -1)) != .5:
        raise ValueError('Only fixed Hard Support v1, maximum weight 0.5, is supported')
    path = Path(options['path'])
    if file_sha256(path) != options['sha256']:
        raise ValueError('Hard Support CSV hash mismatch')
    weights = []
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['image_path', 'label', 'support_weight']:
            raise ValueError('Hard Support CSV columns changed')
        for expected_path, expected_label, row in zip(paths, labels, reader):
            if row['image_path'].removeprefix('train/') != str(expected_path).removeprefix('train/'):
                raise ValueError('Hard Support paths do not match the training order')
            if int(row['label']) != int(expected_label):
                raise ValueError('Hard Support original label mismatch')
            weight = float(row['support_weight'])
            if weight not in (0., .5):
                raise ValueError('Hard Support contains an unregistered weight')
            weights.append(weight)
        if next(reader, None) is not None or len(weights) != len(paths):
            raise ValueError('Hard Support row count mismatch')
    if not weights or sum(value > 0 for value in weights) == 0:
        raise ValueError('Hard Support has no admitted examples')
    return torch.tensor(weights, dtype=torch.float32)


def restore_original_label_loss(base_loss: torch.Tensor, logits: torch.Tensor,
                                targets: torch.Tensor, support: torch.Tensor | None) -> torch.Tensor:
    if support is None:
        return base_loss
    if base_loss.ndim != 1 or support.shape != base_loss.shape:
        raise ValueError('Hard Support loss vectors do not align')
    ce = -(targets.float() * F.log_softmax(logits.float(), dim=1)).sum(dim=1)
    return (1. - support.detach()) * base_loss + support.detach() * ce


def local_quality(local_inputs: torch.Tensor) -> torch.Tensor:
    """Reject spatially blank crops; inspect within-channel spatial variation."""
    if local_inputs.ndim != 4:
        raise ValueError('Local images must have [batch,channels,height,width] shape')
    return local_inputs.float().flatten(2).std(dim=2).mean(dim=1) >= .05


def local_admission(global_confidence: torch.Tensor, gate: float,
                    support: torch.Tensor | None, local_inputs: torch.Tensor) -> torch.Tensor:
    admitted = global_confidence >= gate
    if support is not None:
        admitted = admitted | ((support.detach() > 0) & local_quality(local_inputs))
    return admitted
