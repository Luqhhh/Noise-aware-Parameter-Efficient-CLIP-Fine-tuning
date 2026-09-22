"""Per-sample training weights carried by a standalone audited CSV sidecar.

The rematch first round runs with ``trust`` disabled, so strategies that need
per-sample weights cannot route them through :class:`~aegis_clip.data.TrustBundle`.
They carry them here instead, keyed by canonical sample path, and the trainer
applies them only when ``trust.sample_weight_path`` is set.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from aegis_clip.features import canonical_sample_path


def load_sample_weights(
    path: str | Path,
    sample_paths: list[str],
    *,
    weight_column: str = "weight",
) -> torch.Tensor:
    """Load a weight column and align it to ``sample_paths`` order.

    Fails closed on duplicate keys, missing coverage, and out-of-range values, so
    a misaligned sidecar raises instead of silently reweighting the wrong
    samples.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"sample weight sidecar does not exist: {source}")

    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = {"image_path", weight_column}
        missing = required - columns
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")
        table: dict[str, float] = {}
        for row in reader:
            key = canonical_sample_path(row["image_path"])
            if key in table:
                raise ValueError(f"{source} has duplicate sample path: {key}")
            value = float(row[weight_column])
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(
                    f"{source} weight for {key} is outside [0, 1]: {value}"
                )
            table[key] = value

    if not table:
        raise ValueError(f"{source} contains no data rows")

    weights = torch.ones(len(sample_paths), dtype=torch.float32)
    unresolved: list[str] = []
    for index, raw in enumerate(sample_paths):
        key = canonical_sample_path(raw)
        value = table.get(key)
        if value is None:
            unresolved.append(key)
            continue
        weights[index] = value

    if unresolved:
        raise ValueError(
            f"{source} misses {len(unresolved)} of {len(sample_paths)} split "
            f"samples; first={unresolved[:3]}"
        )
    return weights
