"""Synthetic CPU tests for the PRELIM75 v6 online-geometry wrapper."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis_clip.localization import extract_attention_crops
from aegis_clip.prelim75_fusion import fusion_gce_terms
from aegis_clip.prelim75_joint import crop_with_bound_boxes
from aegis_clip.prelim75_online_geometry import (
    D0_MINIMUM_MATERIAL_GROUPS,
    _accumulate_box_drift,
    _drift_accumulator,
    _drift_summary,
    _empty_positive_safety,
    _forward_global_with_captured_tokens,
    attention_box,
    attention_boxes_by_scale,
    attention_boxes_for_all_scales,
    attention_crop_boxes,
    extract_online_local_views,
    forward_train_with_detached_attention,
    geometry_comparison,
    rank_d0_groups,
    replay_last_block_attention,
)


class _TinyBlock(nn.Module):
    def __init__(self, width: int = 8, heads: int = 2) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.0)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.ln_1(values)
        attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        values = values + attended
        return values + self.mlp(self.ln_2(values))


class _TinyVisual(nn.Module):
    def __init__(self, width: int = 8, sequence: int = 6) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_TinyBlock(width)])
        self.embed = nn.Linear(3 * 4 * 4, width)
        self.pos = nn.Parameter(torch.randn(sequence, width) * 0.02)
        self.proj = nn.Linear(width, width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        tokens = torch.tanh(self.embed(images.flatten(1))).unsqueeze(0)
        tokens = tokens.expand(self.pos.shape[0], batch, self.pos.shape[1]).clone()
        tokens = tokens + self.pos.unsqueeze(1)
        for block in self.transformer.resblocks:
            tokens = block(tokens)
        return F.normalize(self.proj(tokens[0]), dim=-1)


class _TinyModel(nn.Module):
    def __init__(self, width: int = 8, classes: int = 3) -> None:
        super().__init__()
        self.visual = _TinyVisual(width=width)
        self.classifier = nn.Linear(width, classes)

    def forward(self, *, images=None, return_features=False):
        features = self.visual(images)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def _tiny_fixture(batch: int = 4):
    torch.manual_seed(7)
    model = _TinyModel()
    images = torch.randn(batch, 3, 4, 4)
    return model, images


def _hook_count(model: nn.Module) -> int:
    return sum(len(module._forward_pre_hooks) for module in model.modules())


def test_wrapper_preserves_native_global_graph_and_detaches_attention():
    model, images = _tiny_fixture()
    model.train()
    native_logits, native_features = model(images=images, return_features=True)
    logits, features, attention = forward_train_with_detached_attention(model, images)
    torch.testing.assert_close(native_logits, logits)
    torch.testing.assert_close(native_features, features)
    assert logits.requires_grad and features.requires_grad
    assert not attention.requires_grad and attention.grad_fn is None
    assert _hook_count(model) == 0
    (logits.sum() + features.sum()).backward()
    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.grad.abs().sum() > 0
    assert any(parameter.grad is not None for parameter in model.visual.proj.parameters())
    assert attention.shape[0] == images.shape[0] and attention.shape[-1] == 5


def test_replay_does_not_consume_rng_or_mutate_module_state():
    model, images = _tiny_fixture()
    with torch.no_grad():
        _, _, tokens = _forward_global_with_captured_tokens(model, images)
        block = model.visual.transformer.resblocks[-1]
        state_before = {name: value.detach().clone() for name, value in block.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        replay = replay_last_block_attention(block, tokens)
        state_after = {name: value.detach().clone() for name, value in block.state_dict().items()}
        rng_after = torch.get_rng_state()
    assert torch.equal(rng_before, rng_after)
    assert all(torch.equal(state_before[name], state_after[name]) for name in state_before)
    assert replay.shape[-1] == tokens.shape[0] - 1


def test_replay_rejects_stochastic_or_stateful_blocks():
    model, images = _tiny_fixture()
    block = model.visual.transformer.resblocks[-1]
    block.attn.dropout = 0.1
    with pytest.raises(RuntimeError, match="Stochastic"):
        forward_train_with_detached_attention(model, images)
    block.attn.dropout = 0.0
    block.batch_norm_probe = nn.BatchNorm1d(8)
    with pytest.raises(RuntimeError, match="BatchNorm"):
        forward_train_with_detached_attention(model, images)


def test_hook_is_removed_when_global_forward_raises(monkeypatch):
    model, images = _tiny_fixture()
    before = _hook_count(model)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic global failure")

    monkeypatch.setattr(model, "forward", boom)
    with pytest.raises(RuntimeError, match="synthetic"):
        forward_train_with_detached_attention(model, images)
    assert _hook_count(model) == before
    assert not hasattr(model, "_v6_active_capture")


def test_unsupported_visual_structure_is_rejected():
    class _BadModel(nn.Module):
        def forward(self, images=None, return_features=False):
            raise AssertionError("must not be called")

    with pytest.raises(ValueError, match="CLIP ViT"):
        forward_train_with_detached_attention(_BadModel(), torch.randn(1, 3, 4, 4))


def test_attention_crop_boxes_match_existing_extractor():
    torch.manual_seed(11)
    images = torch.randn(4, 3, 224, 224)
    attention = torch.rand(4, 2, 49)
    attention = attention / attention.sum(-1, keepdim=True)
    for scale_size in (112, 128, 144, 160):
        _, reference = extract_attention_crops(images, attention, crop_size=scale_size, top_k=5)
        actual = attention_crop_boxes(images, attention, crop_size=scale_size, top_k=5)
        assert [tuple(map(int, box)) for box in reference] == actual


def test_grouped_online_views_restore_row_order_and_ignore_old_boxes():
    torch.manual_seed(13)
    images = torch.randn(5, 3, 224, 224)
    attention = torch.rand(5, 2, 49)
    attention = attention / attention.sum(-1, keepdim=True)
    scale_ids = torch.tensor([2, 0, 3, 1, 2])
    local, boxes = extract_online_local_views(images, attention, scale_ids, top_k=5)
    expected = torch.empty_like(images)
    expected_boxes = [None] * len(images)
    for scale_id in (0, 1, 2, 3):
        rows = torch.nonzero(scale_ids == scale_id, as_tuple=False).flatten()
        crops, group_boxes = extract_attention_crops(
            images[rows], attention[rows], crop_size=(112, 128, 144, 160)[scale_id], top_k=5
        )
        expected[rows] = crops
        for row, box in zip(rows.tolist(), group_boxes):
            expected_boxes[row] = box
    assert torch.equal(local, expected)
    assert boxes == expected_boxes
    assert attention_boxes_by_scale(images, attention, scale_ids, top_k=5) == expected_boxes

    # The same call with mutated attention changes the online boxes; the
    # function has no old-box argument at all, so no cache can influence G1.
    mutated = attention.clone()
    mutated[:, :, :] = 0.0
    mutated[:, :, 0] = 1.0
    mutated_boxes = attention_boxes_by_scale(images, mutated, scale_ids, top_k=5)
    assert mutated_boxes != boxes
    assert attention_boxes_for_all_scales(images, attention, top_k=5).shape == (5, 4, 4)


def test_geometry_comparison_hand_checked_material_predicate_and_iou():
    old = np.zeros((1, 2, 4, 4), dtype=np.int16)
    old[..., 2:] = 112
    new = old.copy()
    new[0, 0, 0, 0] = 8
    new[0, 0, 0, 2] = 120
    comparison = geometry_comparison(old, new)
    assert comparison["material_change_groups"] == 1
    assert comparison["material_change"].tolist() == [True]
    assert comparison["all_boxes_identical"] is False
    changed_iou = (104.0 * 112.0) / 13440.0
    assert math.isclose(comparison["per_scale"]["112"]["iou"]["mean"], (changed_iou + 1.0) / 2.0, rel_tol=0.0, abs_tol=1e-6)
    assert comparison["per_scale"]["160"]["center_distance_px"]["max"] == 0.0


def test_drift_accumulator_matches_geometry_comparison_ratio():
    old = np.zeros((2, 4, 4), dtype=np.int16)
    old[..., 2:] = 160
    new = old.copy()
    new[0, 0] = (8, 0, 168, 160)
    accumulator = _drift_accumulator()
    _accumulate_box_drift(accumulator, torch.as_tensor(old), torch.as_tensor(new))
    summary = _drift_summary(accumulator)
    assert summary["pairs"] == 8
    assert summary["fraction_boxes_ge_8px"] == 1.0 / 8.0
    assert summary["groups_with_material_change"] == 1


def test_rank_d0_groups_representative_sorting_and_insufficient_rejection():
    data = {
        "paths": ["b/2.jpg", "a/1.jpg", "c/3.jpg", "d/4.jpg", "e/5.jpg"],
        "group_ids": ["g2", "g1", "g2", "g3", "g1"],
        "weights": torch.tensor([1.0, 1.0, 0.5, 0.0, 1.0]),
    }
    first = rank_d0_groups(data, group_count=2)
    assert first["status"] == "selected"
    assert first["available_groups"] == 2
    assert [row["train_index"] for row in first["selected"]] == [0, 1]
    assert first["selection_sha256"] == rank_d0_groups(data, group_count=2)["selection_sha256"]
    insufficient = rank_d0_groups(data, group_count=3)
    assert insufficient["status"] == "insufficient"
    assert insufficient["selected_count"] == 2
    assert D0_MINIMUM_MATERIAL_GROUPS == 205


def test_grouped_scale_validation_and_empty_batch_helpers():
    images = torch.randn(2, 3, 224, 224)
    attention = torch.rand(2, 2, 49)
    with pytest.raises(ValueError, match="registered"):
        attention_boxes_by_scale(images, attention, torch.tensor([4, 0]))
    assert attention_crop_boxes(torch.empty(0, 3, 224, 224), torch.empty(0, 2, 49), crop_size=160) == []


def test_empty_positive_classification_term_is_zero_and_finite():
    global_logits = torch.randn(3, 4)
    local_logits = torch.randn(3, 4)
    targets = torch.softmax(torch.randn(3, 4), dim=1)
    terms = fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
    safety = _empty_positive_safety(terms, torch.ones(3))
    assert safety == {
        "classification_zero": True,
        "classification_finite": True,
        "anchor_proxy_finite": True,
    }


def test_attention_box_alias_and_invalid_shapes_raise():
    images = torch.randn(2, 3, 224, 224)
    attention = torch.rand(2, 2, 49)
    assert attention_box(images, attention, crop_size=160, top_k=5) == attention_crop_boxes(
        images, attention, crop_size=160, top_k=5
    )
    with pytest.raises(ValueError):
        attention_crop_boxes(images, attention, crop_size=800, top_k=5)
    with pytest.raises(ValueError):
        attention_crop_boxes(images, attention[:, :, :15], crop_size=160, top_k=5)


def test_forced_same_boxes_give_same_local_view_and_fusion_objective():
    torch.manual_seed(17)
    images = torch.randn(3, 3, 224, 224)
    attention = torch.rand(3, 1, 49)
    attention = attention / attention.sum(-1, keepdim=True)
    boxes = attention_crop_boxes(images, attention, crop_size=112, top_k=5)
    old_local = crop_with_bound_boxes(images, torch.as_tensor(boxes, dtype=torch.int64))
    online_local, reported = extract_online_local_views(
        images, attention, torch.zeros(3, dtype=torch.long),
        local_scales=(112, 128, 144, 160), top_k=5,
    )
    assert torch.equal(old_local, online_local)
    assert reported == boxes
    global_logits = torch.randn(3, 4)
    local_logits = torch.randn(3, 4)
    targets = torch.softmax(torch.randn(3, 4), dim=1)
    terms = fusion_gce_terms(global_logits, local_logits, targets, blend=0.5)
    assert torch.isfinite(terms["objective"]).all()
