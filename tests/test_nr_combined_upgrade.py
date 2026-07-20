"""Zero-GPU acceptance tests for NR_COMBINED_UPGRADE.

Verifies portable image keys, visual_lora PEFT (with deepcopy-safe
QKV module), clean probability filter, feature distillation, unified
checkpoint loading, config schema, blacklist audit hard gates, and
parent-checkpoint fail-closed behaviour.
"""

import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_dummy_manifest(csv_path: str, n_samples: int = 10):
    rows = []
    for i in range(n_samples):
        p_clean = 0.85 if i < 7 else 0.45
        rows.append({
            "sample_id": f"sample_{i}",
            "image_path": f"train_dedup/00{i:02d}/img_{i}.jpg",
            "original_label": i % 5,
            "training_label": i % 5,
            "sample_weight": p_clean,
            "quality_score": p_clean,
            "source": "oof_zero_floor_thresh0.001",
            "oof_top1": i % 5,
            "p_original_label": p_clean,
            "p_top1": p_clean,
            "prototype_margin": 0.01,
            "knn_agreement": 0.5,
            "flip_consistency": 1.0,
            "duplicate_conflict_flag": False,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _build_clip_model():
    import clip
    cm, prep = clip.load("ViT-B/32", device="cpu")
    cm.visual = cm.visual.float()
    return cm, prep


def _build_classifier(clip_model=None, freeze_clip=False):
    from experiments.baseline.model import CLIPLinearClassifier
    if clip_model is None:
        clip_model, _ = _build_clip_model()
    cm_copy = copy.deepcopy(clip_model)
    return CLIPLinearClassifier(cm_copy, num_classes=500, freeze_clip=freeze_clip)


# ──────────────────────────────────────────────────────────────────────
# 1. Portable image key tests
# ──────────────────────────────────────────────────────────────────────


class TestPortableImageKey(unittest.TestCase):

    def test_991_exact_match_with_dataset(self):
        from common.manifest_loader import portable_image_key
        bl_path = Path("outputs/phase4/global_rejected_paths.txt")
        if not bl_path.exists():
            self.skipTest("global_rejected_paths.txt not found")
        blacklist = bl_path.read_text().strip().split("\n")
        bl_keys = {portable_image_key(p) for p in blacklist}
        self.assertEqual(len(bl_keys), 991)
        train = pd.read_csv("outputs/data/d3_strict/seed42/train.csv")
        ds_keys = {portable_image_key(p) for p in train["image_path"]}
        matched = bl_keys & ds_keys
        self.assertEqual(len(matched), 991)

    def test_portable_key_formats(self):
        from common.manifest_loader import portable_image_key as pk
        self.assertEqual(pk("/home/lux1/noise/train/0001/foo.jpg"), "0001/foo.jpg")
        self.assertEqual(pk("train_dedup/0000/bar.jpg"), "0000/bar.jpg")
        self.assertEqual(pk("0001/foo.jpg"), "0001/foo.jpg")
        self.assertEqual(pk("a/b/train/0250/img.png"), "0250/img.png")

    def test_portable_key_rejects_short_paths(self):
        from common.manifest_loader import portable_image_key as pk
        with self.assertRaises(ValueError):
            pk("just_filename.jpg")

    def test_portable_key_blacklist_no_duplicates(self):
        from common.manifest_loader import portable_image_key
        bl_path = Path("outputs/phase4/global_rejected_paths.txt")
        if not bl_path.exists():
            self.skipTest("global_rejected_paths.txt not found")
        blacklist = bl_path.read_text().strip().split("\n")
        bl_keys = [portable_image_key(p) for p in blacklist]
        self.assertEqual(len(bl_keys), len(set(bl_keys)))

    def test_backslash_path_compatibility(self):
        from common.manifest_loader import portable_image_key as pk
        self.assertEqual(pk("train_dedup\\0001\\img.jpg"), "0001/img.jpg")


# ──────────────────────────────────────────────────────────────────────
# 2. OOFManifestProvider clean-prob threshold tests
# ──────────────────────────────────────────────────────────────────────


class TestOOFManifestCleanProb(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self._tmp.name)
        self.manifest_path = str(self.tmp_root / "test_manifest.csv")
        _make_dummy_manifest(self.manifest_path, n_samples=10)

    def tearDown(self):
        self._tmp.cleanup()

    def test_clean_prob_threshold_binary_weights(self):
        from common.sample_weighting import OOFManifestProvider
        provider = OOFManifestProvider(
            self.manifest_path, clean_prob_threshold=0.70,
            min_weight=0.0, max_weight=1.0,
        )
        paths = [f"train_dedup/00{i:02d}/img_{i}.jpg" for i in range(10)]
        labels = torch.tensor([i % 5 for i in range(10)])
        weights = provider.get_weights(paths, labels, epoch=0)
        for i in range(7):
            self.assertAlmostEqual(weights[i].item(), 1.0, places=5)
        for i in range(7, 10):
            self.assertAlmostEqual(weights[i].item(), 0.0, places=5)

    def test_clean_mask_correct(self):
        from common.sample_weighting import OOFManifestProvider
        provider = OOFManifestProvider(self.manifest_path, clean_prob_threshold=0.70)
        paths = [f"train_dedup/00{i:02d}/img_{i}.jpg" for i in range(10)]
        mask = provider.get_clean_mask(paths)
        self.assertEqual(mask.sum().item(), 7)
        self.assertTrue(mask[0].item())
        self.assertFalse(mask[9].item())

    def test_clean_mask_missing_key_hard_fails(self):
        from common.sample_weighting import OOFManifestProvider
        provider = OOFManifestProvider(self.manifest_path, clean_prob_threshold=0.70)
        paths = ["train_dedup/0099/nonexistent.jpg"]
        with self.assertRaises(KeyError):
            provider.get_clean_mask(paths)

    def test_clean_prob_missing_column_raises(self):
        from common.sample_weighting import OOFManifestProvider
        csv_path = str(self.tmp_root / "no_pcol.csv")
        pd.DataFrame({
            "sample_id": ["s1"],
            "image_path": ["train_dedup/0001/img.jpg"],
            "original_label": [1],
            "training_label": [1],
            "sample_weight": [0.5],
            "quality_score": [0.5],
        }).to_csv(csv_path, index=False)
        with self.assertRaises(ValueError):
            OOFManifestProvider(csv_path, clean_prob_threshold=0.70)


# ──────────────────────────────────────────────────────────────────────
# 3. visual_lora PEFT tests (updated for nn.Module-based QKV)
# ──────────────────────────────────────────────────────────────────────


class TestVisualLoRA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clip_model, cls.preprocess = _build_clip_model()

    def test_visual_lora_is_known_type(self):
        from common.peft import KNOWN_PEFT_TYPES
        self.assertIn("visual_lora", KNOWN_PEFT_TYPES)

    def test_visual_lora_zero_init_equivalence(self):
        from common.peft import apply_peft
        model = _build_classifier(self.clip_model, freeze_clip=False)
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_before = model(dummy_input)
        apply_peft(model, {
            "type": "visual_lora", "lora_last_n_blocks": 4,
            "lora_rank": 8, "lora_alpha": 8,
            "lora_adapt_qv": True, "lora_adapt_out": True,
        })
        model.eval()
        with torch.no_grad():
            out_after = model(dummy_input)
        max_diff = (out_before - out_after).abs().max().item()
        self.assertLess(max_diff, 1e-5,
                        f"LoRA changed output (max diff={max_diff:.2e})")

    def test_visual_lora_only_specified_params_trainable(self):
        from common.peft import apply_peft
        model = _build_classifier(self.clip_model, freeze_clip=False)
        apply_peft(model, {
            "type": "visual_lora", "lora_last_n_blocks": 4,
            "lora_rank": 8, "lora_alpha": 8,
            "lora_adapt_qv": True, "lora_adapt_out": True,
        })
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        for name in trainable:
            ok = "lora_" in name or "classifier" in name
            self.assertTrue(ok, f"Unexpected trainable param: {name}")
        # K must not be trainable
        for name in trainable:
            self.assertNotIn("k_proj", name)
        # LoRA params must exist
        lora_names = [n for n in trainable if "lora_" in n]
        self.assertGreater(len(lora_names), 0)

    def test_visual_lora_gradient_flow(self):
        from common.peft import apply_peft
        model = _build_classifier(self.clip_model, freeze_clip=False)
        apply_peft(model, {
            "type": "visual_lora", "lora_last_n_blocks": 2,
            "lora_rank": 4, "lora_alpha": 4,
            "lora_adapt_qv": True, "lora_adapt_out": True,
        })
        model.train()
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 500, (2,))
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        lora_grads = 0
        for n, p in model.named_parameters():
            if "lora_" in n and p.requires_grad:
                if p.grad is not None:
                    lora_grads += 1
        self.assertGreater(lora_grads, 0, "No LoRA params received gradients")

    def test_visual_lora_block_count(self):
        from common.peft import apply_peft
        model = _build_classifier(self.clip_model, freeze_clip=False)
        apply_peft(model, {
            "type": "visual_lora", "lora_last_n_blocks": 2,
            "lora_rank": 4, "lora_alpha": 4,
            "lora_adapt_qv": True, "lora_adapt_out": True,
        })
        blocks = model.visual.transformer.resblocks
        for i, block in enumerate(blocks):
            is_patched = isinstance(block.attn,
                                    type(blocks[-1].attn))  # _QKVLoRAPatchedAttention
            from common.peft import _QKVLoRAPatchedAttention
            is_qkv = isinstance(block.attn, _QKVLoRAPatchedAttention)
            if i >= len(blocks) - 2:
                self.assertTrue(is_qkv, f"Block {i} should be patched")
            else:
                self.assertFalse(is_qkv, f"Block {i} should NOT be patched")


# ──────────────────────────────────────────────────────────────────────
# 4. DeepCopy safety for visual_lora QKV module
# ──────────────────────────────────────────────────────────────────────


class TestDeepCopyVisualLoRA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clip_model, _ = _build_clip_model()

    def test_deepcopy_produces_independent_lora_params(self):
        from common.peft import apply_peft
        model = _build_classifier(self.clip_model, freeze_clip=False)
        apply_peft(model, {
            "type": "visual_lora", "lora_last_n_blocks": 2,
            "lora_rank": 4, "lora_alpha": 4,
            "lora_adapt_qv": True, "lora_adapt_out": True,
        })

        # Capture original outputs
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_orig = model(x)

        # Deepcopy
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        with torch.no_grad():
            out_copy = model_copy(x)

        # Outputs should be identical
        self.assertTrue(torch.allclose(out_orig, out_copy, atol=1e-5))

        # Modify the COPY's LoRA params
        for n, p in model_copy.named_parameters():
            if "lora_A" in n and p.requires_grad:
                p.data.add_(1.0)

        with torch.no_grad():
            out_copy_modified = model_copy(x)
            out_orig_still = model(x)

        # Copy output should CHANGE
        self.assertFalse(torch.allclose(out_copy, out_copy_modified, atol=1e-3),
                         "Modified copy output should differ from unmodified")

        # Original output should NOT change
        self.assertTrue(torch.allclose(out_orig, out_orig_still, atol=1e-5),
                        "Original model output changed after copy was modified")


# ──────────────────────────────────────────────────────────────────────
# 5. LoRA init contract: A=0, B~N(0, 1/√r)
# ──────────────────────────────────────────────────────────────────────


class TestLoRAInit(unittest.TestCase):

    def test_lora_a_is_zero_init(self):
        from common.lora import LoRALinear
        base = nn.Linear(64, 64)
        lora = LoRALinear(base, r=4, alpha=4)
        self.assertTrue((lora.lora_A == 0).all(),
                        "LoRA A must be zero-initialised per plan")
        self.assertFalse((lora.lora_B == 0).all(),
                         "LoRA B must be random normal (not zero) per plan")

    def test_lora_b_is_normal_with_expected_std(self):
        from common.lora import LoRALinear
        import math
        base = nn.Linear(128, 128)
        r = 8
        lora = LoRALinear(base, r=r, alpha=8)
        expected_std = 1.0 / math.sqrt(r)
        actual_std = lora.lora_B.std().item()
        # Allow 3-sigma tolerance
        self.assertAlmostEqual(actual_std, expected_std, delta=3 * expected_std)

    def test_lora_zero_init_identity_perturbation(self):
        from common.lora import LoRALinear
        base = nn.Linear(64, 64)
        lora = LoRALinear(base, r=4, alpha=8)
        # B @ A should be zero matrix initially
        delta = lora.lora_B @ lora.lora_A
        self.assertTrue((delta == 0).all(),
                        "B@A must be zero at init")


# ──────────────────────────────────────────────────────────────────────
# 6. Unified model loader
# ──────────────────────────────────────────────────────────────────────


class TestModelLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clip_model, cls.preprocess = _build_clip_model()

    def test_build_and_load_linear_head_only_strict_ok(self):
        from common.model_loader import build_and_load_model
        from experiments.baseline.model import build_model
        import tempfile, os

        # Build a clean model, save its state_dict
        model = _build_classifier(self.clip_model, freeze_clip=True)
        sd = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": sd, "epoch": 0, "best_val_acc": 0.5}, f.name)
            tmp_path = f.name

        try:
            config = {"model": {"freeze_clip": True, "num_classes": 500, "clip_model_name": "ViT-B/32"},
                      "peft": {"type": "linear_head_only"}}
            model2, prep, info = build_and_load_model(
                config, tmp_path, torch.device("cpu"),
                build_model_fn=build_model, strict=True,
            )
            self.assertEqual(len(info["missing_keys"]), 0)
            self.assertEqual(len(info["unexpected_keys"]), 0)
            self.assertIsNotNone(info["checkpoint_sha256"])
        finally:
            os.unlink(tmp_path)

    def test_build_and_load_with_visual_lora_strict_roundtrip(self):
        from common.model_loader import build_and_load_model
        from common.peft import apply_peft
        from experiments.baseline.model import build_model
        import tempfile, os

        # Build model with visual_lora, save state_dict
        model = _build_classifier(self.clip_model, freeze_clip=False)
        peft_cfg = {
            "type": "visual_lora", "lora_last_n_blocks": 2,
            "lora_rank": 4, "lora_alpha": 4,
            "lora_adapt_qv": True, "lora_adapt_out": True,
        }
        apply_peft(model, peft_cfg)
        sd = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": sd, "epoch": 0, "best_val_acc": 0.65}, f.name)
            tmp_path = f.name

        try:
            config = {
                "model": {"freeze_clip": False, "num_classes": 500, "clip_model_name": "ViT-B/32"},
                "peft": peft_cfg,
            }
            model2, prep, info = build_and_load_model(
                config, tmp_path, torch.device("cpu"),
                build_model_fn=build_model, strict=True,
            )
            self.assertEqual(len(info["missing_keys"]), 0,
                             f"Missing: {info['missing_keys'][:5]}")
            self.assertEqual(len(info["unexpected_keys"]), 0,
                             f"Unexpected: {info['unexpected_keys'][:5]}")
        finally:
            os.unlink(tmp_path)


# ──────────────────────────────────────────────────────────────────────
# 7. Config schema validation
# ──────────────────────────────────────────────────────────────────────


class TestConfigSchemaValidation(unittest.TestCase):

    def test_nr_combined_upgrade_config_passes_validate_config(self):
        from common.config_schema import validate_config
        from common.utils import load_config
        config = load_config("configs/nr_combined_upgrade.yaml")
        warnings = validate_config(config)
        # No hard errors; warnings are fine
        self.assertIsInstance(warnings, list)

    def test_visual_lora_is_known_peft_type_in_schema(self):
        from common.config_schema import KNOWN_PEFT_TYPES
        self.assertIn("visual_lora", KNOWN_PEFT_TYPES)


# ──────────────────────────────────────────────────────────────────────
# 8. Blacklist audit hard gates
# ──────────────────────────────────────────────────────────────────────


class TestBlacklistAudit(unittest.TestCase):

    def test_data_cardinalities(self):
        """Verify expected counts: train 91195, manifest 91195, BL 991, post-BL 90204."""
        from common.manifest_loader import portable_image_key

        train = pd.read_csv("outputs/data/d3_strict/seed42/train.csv")
        ds_keys = {portable_image_key(p) for p in train["image_path"]}
        self.assertEqual(len(ds_keys), 91195)

        mf = pd.read_csv("outputs/phase/phase3/oof/oof_zero_weight_manifest_thresh0.001.csv")
        mf_keys = {portable_image_key(p) for p in mf["image_path"]}
        self.assertEqual(len(mf_keys), 91195)

        bl_path = Path("outputs/phase4/global_rejected_paths.txt")
        bl_raw = bl_path.read_text().strip().split("\n")
        bl_keys = {portable_image_key(p) for p in bl_raw}
        self.assertEqual(len(bl_keys), 991)

        # intersection
        self.assertEqual(len(bl_keys & ds_keys), 991)
        self.assertEqual(len(bl_keys & mf_keys), 991)

        # post-drop
        post_drop = ds_keys - bl_keys
        self.assertEqual(len(post_drop), 90204)

        # manifest - BL == post-drop dataset
        self.assertEqual(mf_keys - bl_keys, post_drop)

    def test_clean_rejected_partition(self):
        """Verify clean_probe threshold partition: clean=50233, rejected=39971 (post-BL)."""
        from common.manifest_loader import portable_image_key

        bl_path = Path("outputs/phase4/global_rejected_paths.txt")
        bl_keys = {portable_image_key(p)
                   for p in bl_path.read_text().strip().split("\n")}

        mf = pd.read_csv("outputs/phase/phase3/oof/oof_zero_weight_manifest_thresh0.001.csv")

        n_clean = 0
        n_rejected = 0
        for _, row in mf.iterrows():
            key = portable_image_key(str(row["image_path"]))
            if key in bl_keys:
                continue  # excluded by global blacklist
            p_val = float(row["p_original_label"])
            self.assertTrue(0.0 <= p_val <= 1.0, f"p_original_label out of range: {p_val}")
            self.assertTrue(bool(torch.isfinite(torch.tensor(p_val))),
                            f"p_original_label not finite: {p_val}")
            if p_val >= 0.70:
                n_clean += 1
            else:
                n_rejected += 1

        self.assertEqual(n_clean, 50233, f"Expected 50233 clean, got {n_clean}")
        self.assertEqual(n_rejected, 39971, f"Expected 39971 rejected, got {n_rejected}")
        self.assertEqual(n_clean + n_rejected, 90204)

    def test_p_original_label_range(self):
        """All p_original_label values are finite and in [0, 1]."""
        mf = pd.read_csv("outputs/phase/phase3/oof/oof_zero_weight_manifest_thresh0.001.csv")
        p = mf["p_original_label"]
        self.assertTrue(p.notna().all())
        self.assertTrue((p >= 0).all() and (p <= 1).all())
        self.assertTrue(p.apply(lambda x: bool(torch.isfinite(torch.tensor(x)))).all())


# ──────────────────────────────────────────────────────────────────────
# 9. Parent checkpoint fail-closed
# ──────────────────────────────────────────────────────────────────────


class TestParentCheckpointFailClosed(unittest.TestCase):

    def test_a2_artifact_manifest_sha_matches_checkpoint(self):
        """A2 artifact manifest SHA-256 matches actual best.pt file."""
        import json, hashlib
        manifest_path = Path(
            "outputs/oof/nr_cl_knn_drop/seed42/checkpoints/artifact_manifest.json"
        )
        ckpt_path = Path(
            "outputs/oof/nr_cl_knn_drop/seed42/checkpoints/best.pt"
        )
        if not manifest_path.exists() or not ckpt_path.exists():
            self.skipTest("A2 artifacts not available")

        manifest = json.loads(manifest_path.read_text())
        expected_sha = manifest["checkpoint_sha256"]

        h = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        actual_sha = h.hexdigest()

        self.assertEqual(actual_sha, expected_sha,
                         f"A2 checkpoint SHA mismatch: expected {expected_sha[:16]}..., "
                         f"got {actual_sha[:16]}...")
        # Verify it's the full 64-char SHA-256
        self.assertEqual(len(expected_sha), 64)
        self.assertEqual(len(actual_sha), 64)

    def test_split_sha_in_artifact_manifest(self):
        """A2 artifact manifest records split CSV SHA-256."""
        import json
        manifest_path = Path(
            "outputs/oof/nr_cl_knn_drop/seed42/checkpoints/artifact_manifest.json"
        )
        if not manifest_path.exists():
            self.skipTest("A2 artifact manifest not available")
        manifest = json.loads(manifest_path.read_text())
        self.assertIn("train_csv_sha256", manifest)
        self.assertIn("val_csv_sha256", manifest)
        self.assertEqual(len(manifest["train_csv_sha256"]), 64)
        self.assertEqual(len(manifest["val_csv_sha256"]), 64)


# ──────────────────────────────────────────────────────────────────────
# 10. Feature distillation
# ──────────────────────────────────────────────────────────────────────


class TestFeatureDistillation(unittest.TestCase):

    def test_distill_loss_finite_and_scalar(self):
        from common.feature_distillation import FeatureDistillation
        class Dummy(nn.Module):
            def encode_image(self, x):
                return nn.functional.normalize(
                    x.flatten(1)[:, :512].float(), p=2, dim=-1)
        parent = Dummy()
        distill = FeatureDistillation(parent)
        images = torch.randn(4, 3, 224, 224)
        s_feat = Dummy().encode_image(images)
        p_feat = distill.get_parent_features(images)
        loss = distill.compute_loss(s_feat, p_feat)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(loss.ndim, 0)

    def test_teacher_parameters_have_no_grad(self):
        from common.feature_distillation import FeatureDistillation
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(512, 512)
            def encode_image(self, x):
                return self.fc(x.flatten(1)[:, :512])
        parent = Dummy()
        FeatureDistillation(parent)
        for p in parent.parameters():
            self.assertFalse(p.requires_grad)

    def test_parent_features_detached(self):
        from common.feature_distillation import FeatureDistillation
        class Dummy(nn.Module):
            def encode_image(self, x):
                return nn.functional.normalize(
                    x.flatten(1)[:, :512].float(), p=2, dim=-1)
        parent = Dummy()
        distill = FeatureDistillation(parent)
        images = torch.randn(4, 3, 224, 224)
        p_feat = distill.get_parent_features(images)
        self.assertFalse(p_feat.requires_grad)


# ──────────────────────────────────────────────────────────────────────
# 11. Old PEFT regression
# ──────────────────────────────────────────────────────────────────────


class TestOldPEFTRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clip_model, _ = _build_clip_model()

    def test_linear_head_only(self):
        from common.peft import apply_peft
        model = _build_classifier(self.clip_model, freeze_clip=False)
        info = apply_peft(model, {"type": "linear_head_only"})
        self.assertEqual(info["peft_type"], "linear_head_only")

    def test_last_block_lora(self):
        from common.peft import apply_peft
        model = _build_classifier(self.clip_model, freeze_clip=False)
        info = apply_peft(model, {
            "type": "last_block_lora",
            "lora": {"rank": 4, "alpha": 4, "target_block": 11},
        })
        self.assertEqual(info["peft_type"], "last_block_lora")

    def test_unknown_type_raises(self):
        from common.peft import apply_peft
        model = _build_classifier(self.clip_model, freeze_clip=False)
        with self.assertRaises(ValueError):
            apply_peft(model, {"type": "nonexistent_type"})


# ──────────────────────────────────────────────────────────────────────
# 12. Pre-PEFT weight loading & checkpoint round-trip
# ──────────────────────────────────────────────────────────────────────


class TestPrePEFTWeightLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clip_model, _ = _build_clip_model()

    def test_strict_load_before_peft_succeeds(self):
        model = _build_classifier(self.clip_model, freeze_clip=False)
        state_dict = model.state_dict()
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)

    def test_strict_load_after_visual_lora_has_missing_keys(self):
        from common.peft import apply_peft
        model = _build_classifier(self.clip_model, freeze_clip=False)
        parent_state = {k: v.clone() for k, v in model.state_dict().items()}
        apply_peft(model, {
            "type": "visual_lora", "lora_last_n_blocks": 4,
            "lora_rank": 8, "lora_alpha": 8,
            "lora_adapt_qv": True, "lora_adapt_out": True,
        })
        with self.assertRaises(RuntimeError):
            model.load_state_dict(parent_state, strict=True)

    def test_patched_to_patched_strict_roundtrip(self):
        """Save a visual_lora checkpoint and strict-load it into a fresh patched model."""
        from common.peft import apply_peft
        import tempfile, os

        model = _build_classifier(self.clip_model, freeze_clip=False)
        peft_cfg = {
            "type": "visual_lora", "lora_last_n_blocks": 2,
            "lora_rank": 4, "lora_alpha": 4,
            "lora_adapt_qv": True, "lora_adapt_out": True,
        }
        apply_peft(model, peft_cfg)
        sd = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": sd}, f.name)
            tmp_path = f.name

        try:
            model2 = _build_classifier(self.clip_model, freeze_clip=False)
            apply_peft(model2, peft_cfg)
            ckpt = torch.load(tmp_path, map_location="cpu")
            missing, unexpected = model2.load_state_dict(
                ckpt["model_state_dict"], strict=True,
            )
            self.assertEqual(len(missing), 0, f"Missing: {missing[:5]}")
            self.assertEqual(len(unexpected), 0, f"Unexpected: {unexpected[:5]}")
        finally:
            os.unlink(tmp_path)


# ──────────────────────────────────────────────────────────────────────
# 13. CPU dry-run (single batch)
# ──────────────────────────────────────────────────────────────────────


class TestCPUDryRun(unittest.TestCase):

    def test_cpu_single_batch_dry_run(self):
        from common.peft import apply_peft
        from common.feature_distillation import FeatureDistillation
        from experiments.baseline.model import CLIPLinearClassifier

        cm, _ = _build_clip_model()
        cm_copy1 = copy.deepcopy(cm)
        model = CLIPLinearClassifier(cm_copy1, num_classes=500, freeze_clip=False)
        parent_state = {k: v.clone() for k, v in model.state_dict().items()}

        apply_peft(model, {
            "type": "visual_lora", "lora_last_n_blocks": 2,
            "lora_rank": 4, "lora_alpha": 4,
            "lora_adapt_qv": True, "lora_adapt_out": True,
        })

        cm_copy2 = copy.deepcopy(cm)
        parent_model = CLIPLinearClassifier(cm_copy2, num_classes=500, freeze_clip=True)
        parent_model.load_state_dict(parent_state, strict=True)
        for p in parent_model.parameters():
            p.requires_grad_(False)
        parent_model.eval()

        distill = FeatureDistillation(parent_model)
        images = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])

        model.train()
        logits = model(images)
        task_loss = nn.functional.cross_entropy(logits, labels)
        s_feat = model.encode_image(images)
        with torch.no_grad():
            p_feat = distill.get_parent_features(images)
        feat_loss = distill.compute_loss(s_feat, p_feat)
        total_loss = task_loss + 2.0 * feat_loss
        total_loss.backward()

        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0)
        for n, p in parent_model.named_parameters():
            self.assertIsNone(p.grad, f"Parent param {n} should have no grad")

        lora_with_grad = sum(
            1 for n, p in model.named_parameters()
            if "lora_" in n and p.requires_grad and p.grad is not None
        )
        self.assertGreater(lora_with_grad, 0, "No LoRA params received gradients")
        self.assertTrue(torch.isfinite(total_loss).item())


# ──────────────────────────────────────────────────────────────────────
# 14. Config integration
# ──────────────────────────────────────────────────────────────────────


class TestConfigAndIntegration(unittest.TestCase):

    def test_nr_combined_upgrade_config_loads(self):
        from common.utils import load_config
        config = load_config("configs/nr_combined_upgrade.yaml")
        self.assertEqual(config["experiment"]["id"], "NR_COMBINED_UPGRADE")
        self.assertEqual(config["peft"]["type"], "visual_lora")
        self.assertEqual(config["peft"]["lora_last_n_blocks"], 4)
        self.assertEqual(config["sample_weighting"]["clean_prob_threshold"], 0.70)
        self.assertEqual(config["sample_weighting"]["feature_distillation_weight"], 2.0)
        self.assertEqual(config["train"]["epochs"], 6)
        self.assertFalse(config["mixup"]["enabled"])


if __name__ == "__main__":
    unittest.main()
