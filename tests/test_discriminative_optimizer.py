"""Tests for discriminative optimizer param groups (D2).

Verifies:
  - Head and backbone groups are non-overlapping
  - LRs and weight decays match config
  - All trainable params in optimizer, no frozen params in optimizer
  - freeze_clip=True yields head-only group
"""

import torch

from tests.test_partial_unfreeze import MockCLIP, _make_model


# ── Param group structure ────────────────────────────────────────────

class TestParamGroupStructure:
    def test_head_group_always_exists(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        assert len(groups) >= 1
        assert groups[0]["name"] == "head"

    def test_backbone_group_exists_when_visual_unfrozen(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        assert len(groups) == 2
        assert groups[1]["name"] == "backbone"

    def test_no_backbone_group_when_freeze_clip(self):
        model = _make_model(MockCLIP(), freeze_clip=True)
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        assert len(groups) == 1
        assert groups[0]["name"] == "head"

    def test_no_backbone_group_when_all_visual_frozen(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=0, train_ln_post=False,
            train_visual_proj=False,
        )
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        assert len(groups) == 1


# ── LR and weight decay ──────────────────────────────────────────────

class TestLearningRates:
    def test_head_lr_from_arg(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        assert groups[0]["lr"] == 3e-4

    def test_head_wd_from_arg(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        assert groups[0]["weight_decay"] == 1e-4

    def test_backbone_lr_from_init(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
            backbone_lr=3e-6, backbone_weight_decay=0.01,
        )
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        assert groups[1]["lr"] == 3e-6

    def test_backbone_wd_from_init(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
            backbone_lr=3e-6, backbone_weight_decay=0.01,
        )
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        assert groups[1]["weight_decay"] == 0.01


# ── Param coverage ────────────────────────────────────────────────────

class TestParamCoverage:
    def test_no_overlap_between_head_and_backbone(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        if len(groups) < 2:
            return  # No backbone group to compare
        head_ids = {id(p) for p in groups[0]["params"]}
        bb_ids = {id(p) for p in groups[1]["params"]}
        assert head_ids.isdisjoint(bb_ids), \
            "Head and backbone param groups must not overlap"

    def test_all_trainable_params_in_optimizer(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=1, train_ln_post=True,
        )
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        optimizer_param_ids = set()
        for g in groups:
            optimizer_param_ids.update(id(p) for p in g["params"])

        model_trainable_ids = {
            id(p) for p in model.parameters() if p.requires_grad
        }
        missing = model_trainable_ids - optimizer_param_ids
        assert not missing, \
            f"{len(missing)} trainable params missing from optimizer"

    def test_no_frozen_params_in_optimizer(self):
        model = _make_model(
            MockCLIP(), freeze_clip=False,
            unfreeze_last_n_blocks=0, train_ln_post=True,
            train_visual_proj=False,
        )
        groups = model.get_param_groups(head_lr=3e-4, head_weight_decay=1e-4)
        for g in groups:
            for p in g["params"]:
                assert p.requires_grad, \
                    f"Param with requires_grad=False in optimizer group {g['name']}"
