"""Contract tests for the full-data delivery (RM_FULL_LP -> RM_FULL).

Three things are worth guarding mechanically rather than by review comment:

1. FULL01 must be a *pure data-scope change* of the V3 winner.  The whole point of
   the run is a paired comparison, so any drifted hyper-parameter silently destroys
   its evidential value.  ``test_full01_is_a_paired_data_scope_change_of_v3`` diffs
   the two tracked configs and fails on any key outside the declared allow-list.
2. A full-data FT cannot initialize from the development RM-LP, because
   ``rematch_assets.checkpoint_binding`` binds ``train_csv_sha256``.  That is why
   FULL00 exists; ``test_dev_lp_is_rejected_as_parent_for_full_ft`` proves it against
   the real development checkpoint instead of trusting the comment.
3. Everything must stay fail-closed: ``npu:UNASSIGNED`` in the tracked files, and the
   full-data provenance gate must accept the configs once machine-local roots are
   substituted.

The asset-dependent tests skip on a machine without the repechage artifacts, so this
file stays green on a fresh clone while still running for real where the data lives.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
if str(AEGIS_ROOT) not in sys.path:
    sys.path.insert(0, str(AEGIS_ROOT))

from aegis_clip.config import load_config  # noqa: E402
from aegis_clip.rematch_assets import (  # noqa: E402
    validate_cache,
    validate_checkpoint,
    validate_dataset,
)

CONFIG_DIR = ROOT / "configs" / "rematch750_full_ft"
V3_CONFIG = ROOT / "configs" / "rematch750_v3_b1024_e16_lr4.yaml"
ASSETS = Path(
    os.environ.get(
        "AEGIS_REMATCH_ASSETS", ROOT / "artifacts" / "stages" / "repechage" / "20260921"
    )
)
DEV_LP_CHECKPOINT = Path(
    os.environ.get(
        "AEGIS_REMATCH_LP_CHECKPOINT",
        ROOT / "outputs" / "rematch750" / "RM_LP" / "seed42" / "checkpoints" / "best.pt",
    )
)

# The only fields FULL01 may change relative to the V3 winner.  Anything else drifting
# means the pairing is broken and the platform comparison would be uninterpretable.
FULL01_ALLOWED_DIFFS = frozenset(
    {
        "project.experiment_id",
        "project.full_training",
        "output.root",
        "data.train_csv",
        "data.validation_overlap_with_training",
        "evaluation.selection_policy",
        "evaluation.tiebreak_metric",
        "train.init_checkpoint",
    }
)

_MISSING = object()


def _flatten(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _flatten(value, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, node


def _diff(left, right):
    """Flat keys whose values differ, ignoring private runtime keys."""
    flat_left = {key: value for key, value in _flatten(left) if not key.startswith("_")}
    flat_right = {key: value for key, value in _flatten(right) if not key.startswith("_")}
    return {
        key
        for key in flat_left.keys() | flat_right.keys()
        if flat_left.get(key, _MISSING) != flat_right.get(key, _MISSING)
    }


def _substitute_machine_local(config, assets):
    """Repoint a tracked config at machine-local assets for offline gate validation.

    Only paths that are machine-specific are replaced: the asset directory and the
    image roots recorded in the local manifest.  Split, scope declarations and the
    full-data/full-training declaration are left exactly as tracked.
    """
    manifest = json.loads((assets / "dataset_manifest.json").read_text())
    resolved = copy.deepcopy(config)
    resolved["data"].update(
        train_root=manifest["train_root"],
        test_root=manifest["test_root"],
        dataset_manifest=str(assets / "dataset_manifest.json"),
        class_mapping=str(assets / "class_to_idx.json"),
        train_csv=str(assets / Path(resolved["data"]["train_csv"]).name),
        val_csv=str(assets / "val_dev.csv"),
    )
    resolved["features"].update(
        tensor_path=str(assets / "features" / "features.pt"),
        paths_path=str(assets / "features" / "image_paths.json"),
        manifest_path=str(assets / "features" / "manifest.json"),
    )
    return resolved


@pytest.fixture(scope="module")
def full00():
    return load_config(CONFIG_DIR / "FULL00.yaml")


@pytest.fixture(scope="module")
def full01():
    return load_config(CONFIG_DIR / "FULL01.yaml")


@pytest.fixture(scope="module")
def assets():
    if not (ASSETS / "dataset_manifest.json").exists():
        pytest.skip(f"repechage artifacts absent: {ASSETS}")
    return ASSETS


def test_tracked_configs_are_fail_closed(full00, full01):
    for config in (full00, full01):
        assert config["train"]["device"] == "npu:UNASSIGNED"
        assert config["model"]["backbone"] == "ViT-B/32"
        assert config["model"]["pretrained"] == "openai"
        assert config["trust"]["enabled"] is False
        assert config["longtail"]["loss_reweighting"] == "none"
        assert config["longtail"]["balanced_softmax_tau"] == 0.0


def test_tracked_configs_declare_full_data_scope(full00, full01):
    for config in (full00, full01):
        assert config["project"]["full_training"] is True
        assert config["data"]["validation_overlap_with_training"] is True
        assert Path(config["data"]["train_csv"]).name == "full_train.csv"
        assert Path(config["data"]["val_csv"]).name == "val_dev.csv"
        # The former validation split is part of training now, so its metrics are
        # diagnostic only and must never be allowed to pick the submitted epoch.
        assert config["evaluation"]["selection_policy"] == "last_epoch"
        assert config["evaluation"]["tiebreak_metric"] is None


def test_full00_is_shaped_as_a_legal_parent(full00, full01):
    # validate_checkpoint(parent=True) accepts only these two experiment ids.
    assert full00["project"]["experiment_id"] == "RM_FULL_LP"
    assert full00["model"]["peft_mode"] == "frozen"
    assert full00["model"]["use_cached_training"] is True
    assert not full00["train"].get("init_checkpoint")
    # The dev LP recipe the parent was built from: same epoch count and horizon.
    assert full00["train"]["epochs"] == 20
    assert full00["train"]["schedule_epochs"] == 20

    assert full01["project"]["experiment_id"] == "RM_FULL"
    assert full01["model"]["peft_mode"] == "full_finetune"
    assert full01["model"]["use_cached_training"] is False
    assert full01["train"]["epochs"] == 16
    assert full01["train"]["schedule_epochs"] == 16


def test_full01_parent_is_the_full_lp_not_the_dev_lp(full00, full01):
    parent = Path(full01["train"]["init_checkpoint"])
    assert parent.name == "best.pt"
    assert parent.parent.name == "checkpoints"
    assert parent.parent.parent.name == "seed42"
    assert parent.parent.parent.parent.name == "RM_FULL_LP"
    # Same output root as FULL00, so one invocation produces both.
    assert parent.parent.parent.parent.parent == Path(full00["output"]["root"])
    assert full00["output"]["root"] == full01["output"]["root"]


def test_full01_is_a_paired_data_scope_change_of_v3(full01):
    v3 = load_config(V3_CONFIG)
    assert v3["project"]["experiment_id"] == "RM_V3_B1024_E16_LR4"
    assert not v3["project"].get("full_training")
    assert v3["data"]["validation_overlap_with_training"] is False

    drifted = _diff(v3, full01) - FULL01_ALLOWED_DIFFS
    assert not drifted, (
        "FULL01 must differ from the V3 winner only in the declared data-scope fields; "
        f"unexpected differences: {sorted(drifted)}"
    )
    # Guard the allow-list itself: every declared field must really differ, so the
    # list cannot silently rot into a blanket exemption.
    changed = _diff(v3, full01)
    assert FULL01_ALLOWED_DIFFS <= changed, (
        "declared differences that no longer differ: "
        f"{sorted(FULL01_ALLOWED_DIFFS - changed)}"
    )


def test_full_data_configs_pass_the_provenance_gate(full00, full01, assets):
    for config in (full00, full01):
        manifest = validate_dataset(_substitute_machine_local(config, assets))
        assert manifest["train_samples"] == config["data"]["expected_official_train_samples"]
        assert manifest["stage"] == "repechage"
        # The frozen-feature cache is keyed to full_train.csv order, so both the
        # probe and the fine-tune read the same audited cache.
        validate_cache(_substitute_machine_local(config, assets), manifest)


def test_dev_lp_is_rejected_as_parent_for_full_ft(full01, assets):
    if not DEV_LP_CHECKPOINT.exists():
        pytest.skip(f"development RM-LP checkpoint absent: {DEV_LP_CHECKPOINT}")

    # Control: the same checkpoint is a legal parent of a dev-scope fine-tune, which
    # proves the rejection below comes from the data scope and not from the checkpoint.
    dev_variant = _substitute_machine_local(full01, assets)
    dev_variant["data"]["train_csv"] = str(assets / "train_dev.csv")
    dev_variant["project"]["full_training"] = False
    dev_variant["data"]["validation_overlap_with_training"] = False
    validate_checkpoint(DEV_LP_CHECKPOINT, dev_variant, parent=True)

    # Treatment: the full-data fine-tune must not reuse it.  checkpoint_binding binds
    # train_csv_sha256, so FULL00 (RM_FULL_LP) is a hard prerequisite, not a nicety.
    with pytest.raises(ValueError, match="checkpoint data lineage mismatch"):
        validate_checkpoint(
            DEV_LP_CHECKPOINT, _substitute_machine_local(full01, assets), parent=True
        )
