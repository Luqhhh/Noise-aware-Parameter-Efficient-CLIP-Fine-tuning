"""Focused contract tests for REMATCH750_SEARCH_V4.

These tests cover the versioned protocol entry point, generated manifest
coverage, and the new numerical operators without requiring the NPU assets.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from aegis_clip.config import load_config
from aegis_clip.losses import (
    cutmix,
    label_smoothing_targets,
    soft_cross_entropy,
    symmetric_cross_entropy,
)
from aegis_clip.longtail import EpochAwareSampler
from aegis_clip.model import AegisCLIP, CosineClassifier
from aegis_clip.trainer import _training_preprocess


ROOT = Path(__file__).resolve().parents[3]


def test_search_manifest_has_expected_trial_coverage():
    manifest = json.loads((ROOT / "search_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "rematch750_search_v4"
    assert manifest["status"] == "planned_not_executed"
    assert len(manifest["trials"]) == 82
    families = {trial["trial_id"][0] for trial in manifest["trials"]}
    assert families == {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert manifest["first_wave"] == [
        "A01", "A02", "A04", "A07", "B02", "B08",
        "D02", "D08", "E01", "E03", "C01", "F01",
    ]
    by_id = {trial["trial_id"]: trial for trial in manifest["trials"]}
    assert by_id["A01"]["implementation_status"] == "implemented"
    assert by_id["C01"]["implementation_status"] == "pending_quality_asset"
    assert by_id["F01"]["implementation_status"] == "pending_gradient_accumulation"
    assert by_id["G01"]["implementation_status"] == "pending_sam_integration"
    assert by_id["H01"]["implementation_status"] == "pending_v3_feature_cache"


def test_all_generated_configs_load_under_v4_protocol():
    paths = sorted((ROOT / "configs/rematch750_search_v4").glob("*.yaml"))
    assert len(paths) == 82
    for path in paths:
        config = load_config(path)
        assert config["project"]["protocol"] == "rematch750_search_v4"
        assert config["train"]["device"] == "npu:UNASSIGNED"


def test_label_smoothing_targets_are_probability_distributions():
    labels = torch.tensor([0, 1, 2])
    targets = label_smoothing_targets(labels, num_classes=3, smoothing=0.1)
    assert targets.shape == (3, 3)
    assert torch.allclose(targets.sum(dim=1), torch.ones(3))
    assert targets[0, 0] == pytest.approx(1.0 - 0.1 + 0.1 / 3.0)
    assert targets[0, 1] == pytest.approx(0.1 / 3.0)


def test_symmetric_cross_entropy_has_finite_gradients():
    logits = torch.randn(4, 5, requires_grad=True)
    targets = label_smoothing_targets(torch.tensor([0, 1, 2, 3]), 5, 0.0)
    loss = symmetric_cross_entropy(
        logits, targets, ce_weight=0.1, rce_weight=1.0, minimum_probability=1e-4
    )
    assert loss.shape == (4,)
    loss.mean().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(loss).all()


def test_cutmix_mixes_by_exact_retained_area():
    torch.manual_seed(0)
    inputs = torch.randn(6, 3, 16, 16)
    targets = label_smoothing_targets(torch.arange(6), 6, 0.0)
    weights = torch.ones(6)
    mixed, mixed_targets, mixed_weights, lam, permutation = cutmix(
        inputs, targets, weights, alpha=1.0, probability=1.0, generator=torch.Generator().manual_seed(1)
    )
    assert mixed.shape == inputs.shape
    assert 0.0 <= lam <= 1.0
    assert torch.allclose(mixed_targets, lam * targets + (1.0 - lam) * targets[permutation])
    assert torch.allclose(mixed_weights, lam * weights + (1.0 - lam) * weights[permutation])
    changed = (mixed != inputs).any(dim=1).any(dim=1)
    assert changed.any()


def test_randaugment_preset_preserves_clip_tensor_shape():
    preprocess = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_preprocess = _training_preprocess(
        preprocess,
        {
            "train_augmentation": "weak_rrc_flip_randaugment",
            "randaugment_num_ops": 2,
            "randaugment_magnitude": 5,
        },
        input_resolution=224,
    )
    image = Image.new("RGB", (320, 240), "red")
    tensor = train_preprocess(image)
    assert tensor.shape == (3, 224, 224)
    assert torch.isfinite(tensor).all()


def test_cosine_classifier_scales_cosine_similarity():
    classifier = CosineClassifier(8, 4, initial_scale=20.0, trainable_scale=True)
    features = torch.randn(3, 8)
    logits = classifier(features)
    expected = 20.0 * (
        torch.nn.functional.normalize(features, dim=-1)
        @ torch.nn.functional.normalize(classifier.weight, dim=-1).t()
    )
    assert torch.allclose(logits, expected, atol=1e-5)
    assert classifier.scale.requires_grad


def _tiny_visual(block_count: int = 3) -> nn.Module:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.resblocks = nn.ModuleList([Block() for _ in range(block_count)])

    class Visual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.positional_embedding = nn.Parameter(torch.zeros(1, 4))
            self.transformer = Transformer()
            self.ln_post = nn.LayerNorm(4)
            self.proj = nn.Parameter(torch.eye(4))

    return Visual()


def test_partial_full_finetune_unfreezes_last_blocks_and_layer_decay():
    model = AegisCLIP(
        visual=_tiny_visual(3),
        num_classes=2,
        feature_dim=4,
        peft_mode="full_finetune",
        unfreeze_last_n_blocks=2,
    )
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert "visual.transformer.resblocks.2.fc.weight" in trainable
    assert "visual.transformer.resblocks.1.fc.weight" in trainable
    assert "visual.transformer.resblocks.0.fc.weight" not in trainable
    assert "visual.ln_post.weight" in trainable
    assert "visual.proj" in trainable
    assert "classifier.weight" in trainable

    groups = model.parameter_groups(
        4e-4, 1e-4, 1.2e-5, 1e-4, visual_layer_decay=0.5
    )
    visual_groups = [group for group in groups if group["name"].startswith("visual")]
    assert {group["name"] for group in visual_groups} == {"visual_layer", "visual_top"}
    assert min(group["lr"] for group in visual_groups) == pytest.approx(6.0e-6)
    assert max(group["lr"] for group in visual_groups) == pytest.approx(1.2e-5)


def test_epoch_aware_sampler_switches_mode_at_declared_epoch():
    labels = [0, 0, 1, 1, 2, 2]
    counts = torch.tensor([2.0, 2.0, 2.0])
    sampler = EpochAwareSampler(
        labels,
        counts,
        base_mode="none",
        late_mode="class_power",
        late_start_epoch=3,
        class_power_alpha=0.5,
        num_samples=len(labels),
        num_classes=3,
        generator=torch.Generator().manual_seed(0),
    )
    sampler.set_epoch(1)
    assert sampler._active_mode() == "none"
    early = list(iter(sampler))
    assert sorted(early) == list(range(len(labels)))
    sampler.set_epoch(3)
    assert sampler._active_mode() == "class_power"
    late = list(iter(sampler))
    assert len(late) == len(labels)
    assert all(0 <= index < len(labels) for index in late)
