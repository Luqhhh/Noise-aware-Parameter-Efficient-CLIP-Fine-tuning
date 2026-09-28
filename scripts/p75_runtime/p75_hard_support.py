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
    for key in ('ce_restore_enabled', 'local_restore_enabled'):
        if type(options.get(key, False)) is not bool:
            raise ValueError(f'Hard Support {key} must be boolean')
    if not (options.get('ce_restore_enabled', False) or
            options.get('local_restore_enabled', False)):
        return None
    if len(paths) != len(labels):
        raise ValueError('Hard Support paths and labels have different lengths')
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
        rows = list(reader)
        if len(rows) != len(paths):
            raise ValueError('Hard Support row count mismatch')
        for expected_path, expected_label, row in zip(paths, labels, rows):
            if row['image_path'].removeprefix('train/') != str(expected_path).removeprefix('train/'):
                raise ValueError('Hard Support paths do not match the training order')
            if int(row['label']) != int(expected_label):
                raise ValueError('Hard Support original label mismatch')
            weight = float(row['support_weight'])
            if weight not in (0., .5):
                raise ValueError('Hard Support contains an unregistered weight')
            weights.append(weight)
    if not weights or sum(value > 0 for value in weights) == 0:
        raise ValueError('Hard Support has no admitted examples')
    return torch.tensor(weights, dtype=torch.float32)


def classification_support(config: dict, support: torch.Tensor | None,
                           epoch: int) -> torch.Tensor | None:
    """CSV presence never enables either mechanism; CE warmup is untouched."""
    options = config['loss'].get('hard_support', {})
    if (not options.get('enabled', False) or
            not options.get('ce_restore_enabled', False) or
            epoch <= int(config['loss'].get('ce_warmup_epochs', 0))):
        return None
    return support


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
                    support: torch.Tensor | None, local_inputs: torch.Tensor,
                    *, restore_enabled: bool = False) -> torch.Tensor:
    admitted = global_confidence >= gate
    if restore_enabled and support is not None:
        admitted = admitted | ((support.detach() > 0) & local_quality(local_inputs))
    return admitted


@torch.no_grad()
def record_classification(totals: dict, logits: torch.Tensor, targets: torch.Tensor,
                          base: torch.Tensor, blended: torch.Tensor,
                          support: torch.Tensor | None) -> None:
    """Training fit diagnostics, never validation selection or label confidence."""
    if support is None:
        return
    mask = support > 0
    count = int(mask.sum())
    if not count:
        return
    probabilities = logits.detach().float().softmax(1)
    labels = targets.argmax(1)
    ce = F.cross_entropy(logits.detach().float(), labels, reduction='none')
    stats = totals.setdefault('supported_ce', dict(examples=0, correct=0,
        label_probability_sum=0., base_loss_sum=0., ce_loss_sum=0., blended_loss_sum=0.))
    stats['examples'] += count
    stats['correct'] += int(((logits.argmax(1) == labels) & mask).sum())
    stats['label_probability_sum'] += float(probabilities[mask, labels[mask]].sum())
    for name, vector in [('base_loss_sum', base), ('ce_loss_sum', ce),
                         ('blended_loss_sum', blended)]:
        stats[name] += float(vector.detach()[mask].sum())
