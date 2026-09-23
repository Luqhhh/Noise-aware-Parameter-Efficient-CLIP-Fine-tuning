"""Protocol dispatcher for legacy rematch assets and REMATCH750_SEARCH_V4."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aegis_clip import rematch_assets as _legacy
from aegis_clip import rematch_search_v4 as _v4


def validate_dataset(config: dict[str, Any]):
    return _v4.validate_dataset(config) if _v4.is_v4(config) else _legacy.validate_dataset(config)


def expected_binding(config: dict[str, Any], manifest: dict[str, Any]):
    return _v4.expected_binding(config, manifest) if _v4.is_v4(config) else _legacy.expected_binding(config, manifest)


def expected_feature_binding(config: dict[str, Any], manifest: dict[str, Any]):
    return _v4.expected_feature_binding(config, manifest) if _v4.is_v4(config) else _legacy.expected_binding(config, manifest)


def validate_cache(config: dict[str, Any], manifest: dict[str, Any] | None = None):
    return _v4.validate_cache(config, manifest) if _v4.is_v4(config) else _legacy.validate_cache(config, manifest)


def checkpoint_binding(config: dict[str, Any]):
    return _v4.checkpoint_binding(config) if _v4.is_v4(config) else _legacy.checkpoint_binding(config)


def validate_checkpoint(path: str | Path, config: dict[str, Any], *, parent: bool = False):
    if _v4.is_v4(config):
        return _v4.validate_checkpoint(path, config, parent=parent)
    return _legacy.validate_checkpoint(path, config, parent=parent)


def validate_training(config: dict[str, Any], resume: str | None = None, init_checkpoint: str | None = None):
    if _v4.is_v4(config):
        return _v4.validate_training(config, resume=resume, init_checkpoint=init_checkpoint)
    return _legacy.validate_training(config, resume=resume, init_checkpoint=init_checkpoint)
