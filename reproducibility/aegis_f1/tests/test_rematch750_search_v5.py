"""Contract tests for REMATCH750_SEARCH_V5 conditional slots."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
import torchvision.transforms as T

from aegis_clip.config import load_config
from aegis_clip.rematch_search_v5 import (
    MechanismBlockedError,
    validate_declaration,
)
from aegis_clip.trainer import _training_preprocess


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs/rematch750_search_v5"


def test_v5_manifest_has_conditional_slot_coverage():
    manifest = json.loads((ROOT / "search_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "rematch750_search_v5"
    assert manifest["status"] == "planned_not_executed"
    assert len(manifest["trials"]) == 68
    assert manifest["budget"] == {
        "visual_training_points": 48,
        "cached_head_points": 12,
        "max_combinations": 8,
        "max_additional_confirmation_training_runs": 6,
    }
    counts = manifest["family_counts"]
    assert counts == {"R": 6, "S": 9, "K": 9, "V": 8, "L": 6, "Q": 6, "N": 4, "H": 12, "X": 8}
    assert manifest["baseline"]["candidate"] == "RM_V4_F05"
    assert manifest["baseline"]["platform_score"] == 65.4711
    assert manifest["baseline"]["platform_correct_predictions"] == 24515
    assert manifest["platform_goal"]["reached"] is False
    assert manifest["platform_goal"]["additional_correct_predictions_needed"] == 1696
    assert manifest["first_wave"] == [
        "R02", "R04", "R01", "R03", "S01", "S02",
        "K02", "K03", "K07", "V02", "L01", "H01",
    ]
    by_id = {trial["trial_id"]: trial for trial in manifest["trials"]}
    assert by_id["R01"]["implementation_status"] == "implemented"
    assert by_id["S02"]["implementation_status"] == "implemented"
    assert by_id["K07"]["implementation_status"] == "implemented"
    assert by_id["L01"]["implementation_status"] == "implemented"
    assert by_id["H01"]["implementation_status"] == "pending_artifact"
    assert by_id["X08"]["implementation_status"] == "blocked_dependency"


def test_all_v5_configs_load_and_declare():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(paths) == 68
    for path in paths:
        config = load_config(path)
        declaration = validate_declaration(config)
        assert declaration["trial_id"] == path.stem
        assert config["project"]["protocol"] == "rematch750_search_v5"
        assert config["train"]["device"] == "npu:UNASSIGNED"


@pytest.mark.parametrize("trial_id", ["H01", "X01", "N01"])
def test_blocked_v5_slots_fail_closed_when_execution_is_requested(trial_id):
    config = load_config(CONFIG_DIR / f"{trial_id}.yaml")
    with pytest.raises(MechanismBlockedError):
        validate_declaration(config, require_implemented=True)


def test_v5_unknown_search_field_is_not_silently_ignored():
    config = load_config(CONFIG_DIR / "R01.yaml")
    config["project"]["search"] = copy.deepcopy(config["project"]["search"])
    config["project"]["search"]["mystery_axis"] = 1
    with pytest.raises(ValueError, match="unknown project.search keys"):
        validate_declaration(config)


def test_v5_rrc_area_bounds_are_forwarded_to_train_transform():
    base = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform = _training_preprocess(
        base,
        {
            "train_augmentation": "weak_rrc_flip",
            "rrc_scale_min": 0.8,
            "rrc_scale_max": 1.0,
            "rrc_ratio_min": 0.85,
            "rrc_ratio_max": 1.15,
        },
        input_resolution=224,
    )
    tensor = transform(Image.new("RGB", (320, 240), "blue"))
    assert tensor.shape == (3, 224, 224)


def test_v5_staged_resolution_is_declared_implemented():
    config = load_config(CONFIG_DIR / "R05.yaml")
    declaration = validate_declaration(config, require_implemented=True)
    assert declaration["status"] == "implemented"
    staged = config["train"]["staged_resolution"]
    assert staged["early_resolution"] == 224
    assert staged["late_resolution"] == 384
    assert staged["switch_epoch"] == 9


def _tiny_keyword_model():
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, *, images, return_features=False, **kwargs):
            logits = self.fc(images)
            return (logits, images) if return_features else logits

    return TinyModel()


def _microbatch_state(model, inputs, labels, loss_config, class_counts):
    from aegis_clip.trainer import _per_sample_loss

    targets = torch.nn.functional.one_hot(labels, 3).float()
    logits = model(images=inputs)
    loss = _per_sample_loss(logits, targets, loss_config, epoch=1).mean()
    return (
        {
            "forward_key": "images",
            "forward_inputs": inputs,
            "mixed_gate": None,
            "mixed_reference": inputs,
            "mixed_targets": targets,
            "mixed_weights": torch.ones(labels.numel()),
            "class_counts": class_counts,
            "prior_tau": 0.0,
            "loss_config": loss_config,
            "epoch": 1,
            "batch_suspicious": None,
            "distill_weight": 0.0,
            "sample_count": int(labels.numel()),
            "first_loss": float(loss.detach()),
        },
        loss,
    )


def test_effective_batch_sam_matches_single_batch_sam():
    import copy
    from aegis_clip.trainer import (
        _sam_accumulated_optimizer_step,
        _sam_optimizer_step,
    )

    torch.manual_seed(7)
    loss_config = {"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 0}
    inputs = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 1, 0])
    class_counts = torch.ones(3)

    model_a = _tiny_keyword_model()
    model_e = copy.deepcopy(model_a)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    opt_e = torch.optim.AdamW(model_e.parameters(), lr=1e-3)

    microbatches = []
    opt_a.zero_grad()
    for start, end in ((0, 3), (3, 5)):
        state, loss = _microbatch_state(
            model_a,
            inputs[start:end],
            labels[start:end],
            loss_config,
            class_counts,
        )
        microbatches.append(state)
        (loss * ((end - start) / len(labels))).backward()

    _sam_accumulated_optimizer_step(
        model_a,
        opt_a,
        microbatches=microbatches,
        cycle_samples=len(labels),
        rho=0.05,
        clip_norm=1.0,
        gsam_alpha=0.0,
        rng_state=None,
        device=torch.device("cpu"),
    )

    def expected_loss():
        from aegis_clip.trainer import _per_sample_loss

        targets = torch.nn.functional.one_hot(labels, 3).float()
        logits = model_e(images=inputs)
        return _per_sample_loss(logits, targets, loss_config, epoch=1).mean()

    _sam_optimizer_step(
        model_e,
        opt_e,
        first_loss=expected_loss(),
        second_pass=expected_loss,
        rho=0.05,
        clip_norm=1.0,
    )
    for parameter_a, parameter_e in zip(model_a.parameters(), model_e.parameters()):
        assert torch.allclose(parameter_a, parameter_e, atol=1e-6, rtol=1e-5)


def test_effective_batch_sam_rho_zero_degenerates_to_adamw():
    import copy
    from aegis_clip.trainer import (
        _per_sample_loss,
        _sam_accumulated_optimizer_step,
    )

    torch.manual_seed(11)
    loss_config = {"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 0}
    inputs = torch.randn(4, 4)
    labels = torch.tensor([0, 1, 1, 2])
    class_counts = torch.ones(3)
    model_s = _tiny_keyword_model()
    model_a = copy.deepcopy(model_s)
    opt_s = torch.optim.AdamW(model_s.parameters(), lr=1e-3)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)

    targets = torch.nn.functional.one_hot(labels, 3).float()
    logits = model_a(images=inputs)
    opt_a.zero_grad()
    (_per_sample_loss(logits, targets, loss_config, epoch=1).mean()).backward()
    opt_a.step()

    microbatches = []
    opt_s.zero_grad()
    for start, end in ((0, 2), (2, 4)):
        state, loss = _microbatch_state(
            model_s, inputs[start:end], labels[start:end], loss_config, class_counts
        )
        microbatches.append(state)
        (loss * ((end - start) / len(labels))).backward()
    _sam_accumulated_optimizer_step(
        model_s,
        opt_s,
        microbatches=microbatches,
        cycle_samples=len(labels),
        rho=0.0,
        clip_norm=1.0,
        rng_state=None,
        device=torch.device("cpu"),
    )
    for parameter_s, parameter_a in zip(model_s.parameters(), model_a.parameters()):
        assert torch.allclose(parameter_s, parameter_a, atol=1e-6, rtol=1e-5)


def test_effective_batch_sam_trainer_uses_one_update_per_effective_batch(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import aegis_clip.trainer as trainer
    from aegis_clip.config import load_config
    from aegis_clip.model import AegisCLIP

    root = Path(__file__).resolve().parents[3]
    cfg = load_config(root / "configs/rematch750_lp.yaml")
    cfg["data"].pop("dataset_manifest", None)
    cfg["model"].update(
        num_classes=3,
        feature_dim=4,
        peft_mode="full_finetune",
        input_resolution=224,
        use_cached_training=False,
    )
    cfg["model"].pop("official_checkpoint", None)
    cfg["train"].update(
        device="cpu",
        epochs=1,
        schedule_epochs=1,
        num_workers=0,
        loader_timeout=0,
        batch_size=2,
        amp=False,
        grad_accum_steps=2,
        log_every_steps=1,
        head_lr=1e-3,
        backbone_lr=1e-4,
        sam={"enabled": True, "mode": "standard_global_l2", "rho": 0.05},
        init_checkpoint=None,
    )
    cfg["loss"].update(
        name="gce",
        gce_q=0.5,
        ce_warmup_epochs=0,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        feature_distillation_weight=0.0,
    )
    cfg["output"]["root"] = str(tmp_path / "out")
    paths = [f"{c:04d}/{j}.jpg" for c in range(3) for j in range(4)]
    labels = [c for c in range(3) for _ in range(4)]
    frame = pd.DataFrame(dict(image_path=paths, label=labels))
    train_root = tmp_path / "train"
    for relative in paths:
        image_path = train_root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40), "red").save(image_path)
    val_mask = frame.index % 4 == 0
    for key, data in (("train_csv", frame[~val_mask]), ("val_csv", frame[val_mask])):
        path = tmp_path / f"{key}.csv"
        data.to_csv(path, index=False)
        cfg["data"][key] = str(path)
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({f"{i:04d}": i for i in range(3)}))
    cfg["data"]["class_mapping"] = str(mapping)
    cfg["data"]["train_root"] = str(train_root)
    cfg["data"]["test_root"] = str(tmp_path / "test")
    cfg["data"]["expected_official_train_samples"] = 12
    cfg["data"]["expected_test_samples"] = 1
    (tmp_path / "test").mkdir(exist_ok=True)
    (tmp_path / "test" / "x.jpg").write_bytes(b"")
    tensor = tmp_path / "features.pt"
    torch.save(torch.randn(len(paths), 4), tensor)
    index = tmp_path / "paths.json"
    index.write_text(json.dumps(paths))
    cfg["features"] = dict(tensor_path=str(tensor), paths_path=str(index))

    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.resblocks = torch.nn.ModuleList([])

        def forward(self, values):
            return values

    class TinyImageVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=32, stride=32)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.positional_embedding = torch.nn.Parameter(torch.zeros(50, 4))
            self.ln_pre = torch.nn.LayerNorm(4)
            self.transformer = TinyTransformer()
            self.ln_post = torch.nn.LayerNorm(4)
            self.proj = torch.nn.Parameter(torch.eye(4))

        def forward(self, images):
            return self.pool(self.conv1(images)).flatten(1)

    class TinyModel(AegisCLIP):
        def __init__(self):
            super().__init__(
                visual=TinyImageVisual(),
                num_classes=3,
                feature_dim=4,
                peft_mode="full_finetune",
                input_resolution=224,
            )

    preprocess = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    monkeypatch.setattr(
        trainer, "build_model", lambda _config, _device: (TinyModel(), preprocess)
    )
    checkpoint = trainer.train(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # 9 train samples; batch 2 * accumulation 2 => 3 effective updates.
    assert state["global_step"] == 3
    assert state.get("metrics", {}).get("successful_optimizer_updates") == 3


def test_feature_anchor_schedule_holds_then_decays():
    from aegis_clip.trainer import _feature_anchor_weight

    loss_cfg = {
        "feature_distillation_weight": 9.0,
        "anchor_schedule": {
            "enabled": True,
            "hold_epochs": 2,
            "start_weight": 2.0,
            "end_epoch": 16,
            "end_weight": 0.2,
        },
    }
    assert _feature_anchor_weight(loss_cfg, 1, "full_finetune") == pytest.approx(2.0)
    assert _feature_anchor_weight(loss_cfg, 2, "full_finetune") == pytest.approx(2.0)
    assert _feature_anchor_weight(loss_cfg, 9, "full_finetune") == pytest.approx(1.1)
    assert _feature_anchor_weight(loss_cfg, 16, "full_finetune") == pytest.approx(0.2)
    assert _feature_anchor_weight(loss_cfg, 18, "full_finetune") == pytest.approx(0.2)
    assert _feature_anchor_weight({}, 5, "frozen") == 0.0


def test_same_view_teacher_resizes_augmented_view_to_224():
    from aegis_clip.trainer import _same_view_teacher_features

    teacher = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )
    images = torch.randn(2, 3, 16, 16)
    features = _same_view_teacher_features(teacher, images, resolution=8)
    assert features.shape == (2, 4)


def test_attention_local_fallback_uses_global_below_confidence_gate():
    from aegis_clip.trainer import _attention_local_fallback_per_sample

    local = torch.tensor([1.0, 2.0])
    global_loss = torch.tensor([3.0, 4.0])
    confidence = torch.tensor([0.60, 0.80])
    result = _attention_local_fallback_per_sample(
        local, global_loss, confidence, gate=0.70
    )
    assert torch.equal(result, torch.tensor([3.0, 2.0]))


def test_v5_newly_unblocked_slots_declare_as_implemented():
    for trial_id in (
        "R05", "R06",
        "V03", "V04", "V07", "V08",
        "Q01", "Q02", "Q03", "Q04", "Q05", "Q06",
    ):
        config = load_config(CONFIG_DIR / f"{trial_id}.yaml")
        declaration = validate_declaration(config, require_implemented=True)
        assert declaration["status"] == "implemented"


def test_v5_letterbox_training_transform_preserves_square_input():
    base = T.Compose(
        [
            T.Resize(320),
            T.CenterCrop(320),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform = _training_preprocess(
        base,
        {
            "train_augmentation": "clip_letterbox",
            "rrc_scale_min": 0.7,
            "rrc_scale_max": 1.0,
        },
        input_resolution=320,
    )
    tensor = transform(Image.new("RGB", (300, 200), "blue"))
    assert tensor.shape == (3, 320, 320)
    assert torch.isfinite(tensor).all()


def test_v5_late_rrc_declaration_has_epoch_window():
    config = load_config(CONFIG_DIR / "V04.yaml")
    low = config["data"]["late_rrc"]
    assert low["enabled"] is True
    assert low["start_epoch"] == 13
    assert low["scale_min"] == pytest.approx(0.9)
    assert low["scale_max"] == pytest.approx(1.0)
    validate_declaration(config, require_implemented=True)


def test_v5_gsam_declaration_requires_fp32_accumulation():
    config = load_config(CONFIG_DIR / "Q01.yaml")
    config = copy.deepcopy(config)
    config["train"]["grad_accum_steps"] = 1
    with pytest.raises(ValueError, match="GSAM requires effective-batch accumulation"):
        validate_declaration(config, require_implemented=True)


def test_gsam_alpha_changes_update_but_stays_finite():
    import copy
    from aegis_clip.trainer import _sam_accumulated_optimizer_step

    torch.manual_seed(23)
    loss_config = {"name": "gce", "gce_q": 0.5, "ce_warmup_epochs": 0}
    inputs = torch.randn(4, 4)
    labels = torch.tensor([0, 1, 1, 2])
    class_counts = torch.ones(3)

    def run(alpha):
        model = _tiny_keyword_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        microbatches = []
        optimizer.zero_grad()
        for start, end in ((0, 2), (2, 4)):
            state, loss = _microbatch_state(
                model, inputs[start:end], labels[start:end], loss_config, class_counts
            )
            microbatches.append(state)
            (loss * ((end - start) / len(labels))).backward()
        update = _sam_accumulated_optimizer_step(
            model,
            optimizer,
            microbatches=microbatches,
            cycle_samples=len(labels),
            rho=0.025,
            clip_norm=1.0,
            gsam_alpha=alpha,
            rng_state=None,
            device=torch.device("cpu"),
        )
        return model, update

    model_zero, update_zero = run(0.0)
    model_gsam, update_gsam = run(0.4)
    assert update_zero["gsam_alpha"] == 0.0
    assert update_gsam["gsam_alpha"] == 0.4
    assert update_gsam["gsam_projection"] is not None
    for parameter in model_gsam.parameters():
        assert torch.isfinite(parameter).all()
    assert any(
        not torch.allclose(p0, p1, atol=1e-7, rtol=1e-6)
        for p0, p1 in zip(model_zero.parameters(), model_gsam.parameters())
    )
