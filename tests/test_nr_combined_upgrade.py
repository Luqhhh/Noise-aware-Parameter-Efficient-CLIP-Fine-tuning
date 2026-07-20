"""Zero-GPU acceptance tests for NR_COMBINED_UPGRADE.

Verifies portable image keys, visual_lora PEFT, clean probability filter,
feature distillation, and weight-loading correctness without GPU.
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


def _make_dummy_manifest(csv_path: str, n_samples: int = 10,
                         clean_prob_threshold: float = 0.70):
    """Write a minimal OOF manifest CSV for testing."""
    rows = []
    for i in range(n_samples):
        p_clean = 0.85 if i < 7 else 0.45  # 7 clean, 3 rejected
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


# ──────────────────────────────────────────────────────────────────────
# 1. Portable image key tests
# ──────────────────────────────────────────────────────────────────────


class TestPortableImageKey(unittest.TestCase):

    def test_991_exact_match_with_dataset(self):
        """All 991 blacklist entries match dataset via portable keys."""
        from common.manifest_loader import portable_image_key

        bl_path = Path("outputs/phase4/global_rejected_paths.txt")
        if not bl_path.exists():
            self.skipTest("global_rejected_paths.txt not found")

        blacklist = bl_path.read_text().strip().split("\n")
        bl_keys = {portable_image_key(p) for p in blacklist}
        # No duplicates
        self.assertEqual(len(bl_keys), 991, "Duplicate portable keys in blacklist")

        train = pd.read_csv("outputs/data/d3_strict/seed42/train.csv")
        ds_keys = {portable_image_key(p) for p in train["image_path"]}

        matched = bl_keys & ds_keys
        self.assertEqual(len(matched), 991,
                         f"Only {len(matched)}/991 matched in dataset")

    def test_portable_key_formats(self):
        """Various path formats produce correct class_dir/filename."""
        from common.manifest_loader import portable_image_key as pk

        # Absolute paths (captain's machine)
        self.assertEqual(pk("/home/lux1/noise/train/0001/foo.jpg"),
                         "0001/foo.jpg")
        # Relative repo paths
        self.assertEqual(pk("train_dedup/0000/bar.jpg"),
                         "0000/bar.jpg")
        # Local absolute paths
        self.assertEqual(pk("/home/x28639/projects/repo/train_dedup/0499/baz.jpeg"),
                         "0499/baz.jpeg")
        # With extra leading components
        self.assertEqual(pk("a/b/train/0250/img.png"), "0250/img.png")
        # Already minimal
        self.assertEqual(pk("0001/foo.jpg"), "0001/foo.jpg")
        # Three-digit class dir
        self.assertEqual(pk("train/042/img.webp"), "042/img.webp")

    def test_portable_key_rejects_short_paths(self):
        """Paths with < 2 components raise ValueError."""
        from common.manifest_loader import portable_image_key as pk

        with self.assertRaises(ValueError):
            pk("just_filename.jpg")
        with self.assertRaises(ValueError):
            pk("")

    def test_portable_key_blacklist_no_duplicates(self):
        """Blacklist has zero duplicate portable keys."""
        from common.manifest_loader import portable_image_key

        bl_path = Path("outputs/phase4/global_rejected_paths.txt")
        if not bl_path.exists():
            self.skipTest("global_rejected_paths.txt not found")

        blacklist = bl_path.read_text().strip().split("\n")
        bl_keys = [portable_image_key(p) for p in blacklist]
        self.assertEqual(len(bl_keys), len(set(bl_keys)),
                         "Blacklist contains duplicate portable keys")


# ──────────────────────────────────────────────────────────────────────
# 2. OOFManifestProvider clean-prob threshold tests
# ──────────────────────────────────────────────────────────────────────


class TestOOFManifestCleanProb(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self._tmp.name)
        self.manifest_path = str(self.tmp_root / "test_manifest.csv")
        _make_dummy_manifest(self.manifest_path, n_samples=10,
                             clean_prob_threshold=0.70)

    def tearDown(self):
        self._tmp.cleanup()

    def test_clean_prob_threshold_binary_weights(self):
        """Samples with p_original_label >= 0.70 get weight=1.0, else 0.0."""
        from common.sample_weighting import OOFManifestProvider

        provider = OOFManifestProvider(
            self.manifest_path,
            clean_prob_threshold=0.70,
            min_weight=0.0,
            max_weight=1.0,
        )
        paths = [f"train_dedup/00{i:02d}/img_{i}.jpg" for i in range(10)]
        labels = torch.tensor([i % 5 for i in range(10)])
        weights = provider.get_weights(paths, labels, epoch=0)

        # First 7 have p_original_label = 0.85 → clean → weight 1.0
        for i in range(7):
            self.assertAlmostEqual(weights[i].item(), 1.0, places=5,
                                   msg=f"Sample {i} should be clean (weight 1.0)")
        # Last 3 have p_original_label = 0.45 → rejected → weight 0.0
        for i in range(7, 10):
            self.assertAlmostEqual(weights[i].item(), 0.0, places=5,
                                   msg=f"Sample {i} should be rejected (weight 0.0)")

    def test_clean_mask_correct(self):
        """get_clean_mask returns correct boolean mask."""
        from common.sample_weighting import OOFManifestProvider

        provider = OOFManifestProvider(
            self.manifest_path,
            clean_prob_threshold=0.70,
        )
        paths = [f"train_dedup/00{i:02d}/img_{i}.jpg" for i in range(10)]
        mask = provider.get_clean_mask(paths)

        self.assertEqual(mask.sum().item(), 7)  # 7 clean
        self.assertEqual((~mask).sum().item(), 3)  # 3 rejected
        self.assertTrue(mask[0].item())
        self.assertFalse(mask[9].item())

    def test_clean_prob_missing_column_raises(self):
        """Missing p_original_label column raises ValueError."""
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

        with self.assertRaises(ValueError) as ctx:
            OOFManifestProvider(csv_path, clean_prob_threshold=0.70)
        self.assertIn("p_original_label", str(ctx.exception))

    def test_clean_prob_coverage_check(self):
        """All samples in batch have manifest coverage."""
        from common.sample_weighting import OOFManifestProvider

        provider = OOFManifestProvider(
            self.manifest_path,
            clean_prob_threshold=0.70,
        )
        paths = [f"train_dedup/00{i:02d}/img_{i}.jpg" for i in range(10)]
        labels = torch.tensor([i % 5 for i in range(10)])
        # Should not raise — all 10 present in manifest
        weights = provider.get_weights(paths, labels, epoch=0)
        self.assertEqual(len(weights), 10)

    def test_uses_clean_prob_filter_flag(self):
        """uses_clean_prob_filter returns True only when threshold is set."""
        from common.sample_weighting import OOFManifestProvider

        provider_with = OOFManifestProvider(
            self.manifest_path, clean_prob_threshold=0.70,
        )
        self.assertTrue(provider_with.uses_clean_prob_filter())
        self.assertEqual(provider_with.clean_prob_threshold, 0.70)

        provider_without = OOFManifestProvider(self.manifest_path)
        self.assertFalse(provider_without.uses_clean_prob_filter())
        self.assertIsNone(provider_without.clean_prob_threshold)


# ──────────────────────────────────────────────────────────────────────
# 3. visual_lora PEFT tests
# ──────────────────────────────────────────────────────────────────────


class TestVisualLoRA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Build a CLIP model once for all tests in this class."""
        import clip
        cls.clip_model, cls.preprocess = clip.load("ViT-B/32", device="cpu")
        # Convert to float32 for consistency
        cls.clip_model.visual = cls.clip_model.visual.float()

    def _build_classifier_model(self):
        """Wrap CLIP visual encoder with a linear classifier head."""
        from experiments.baseline.model import CLIPLinearClassifier

        clip_copy = copy.deepcopy(self.clip_model)
        return CLIPLinearClassifier(clip_copy, num_classes=500, freeze_clip=False)

    def test_visual_lora_is_known_type(self):
        """visual_lora is a registered PEFT type."""
        from common.peft import KNOWN_PEFT_TYPES
        self.assertIn("visual_lora", KNOWN_PEFT_TYPES)

    def test_visual_lora_zero_init_equivalence(self):
        """After applying visual_lora, model output == pre-LoRA output."""
        from common.peft import apply_peft

        model = self._build_classifier_model()
        model.eval()

        # Capture output before LoRA
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_before = model(dummy_input)

        # Apply visual_lora (zero-init B → identity adapter)
        peft_cfg = {
            "type": "visual_lora",
            "lora_last_n_blocks": 4,
            "lora_rank": 8,
            "lora_alpha": 8,
            "lora_adapt_qv": True,
            "lora_adapt_out": True,
        }
        apply_peft(model, peft_cfg)
        model.eval()

        with torch.no_grad():
            out_after = model(dummy_input)

        # Logits should be identical (within float32 tolerance)
        max_diff = (out_before - out_after).abs().max().item()
        self.assertLess(max_diff, 1e-5,
                        f"LoRA zero-init changed output (max diff={max_diff:.2e})")

    def test_visual_lora_only_specified_params_trainable(self):
        """Only LoRA and classifier parameters require grad."""
        from common.peft import apply_peft

        model = self._build_classifier_model()
        peft_cfg = {
            "type": "visual_lora",
            "lora_last_n_blocks": 4,
            "lora_rank": 8,
            "lora_alpha": 8,
            "lora_adapt_qv": True,
            "lora_adapt_out": True,
        }
        apply_peft(model, peft_cfg)

        trainable_names = [
            n for n, p in model.named_parameters() if p.requires_grad
        ]
        frozen_names = [
            n for n, p in model.named_parameters() if not p.requires_grad
        ]

        # All trainable params must be lora_A/lora_B or classifier
        for name in trainable_names:
            ok = "lora_" in name or "classifier" in name
            self.assertTrue(ok,
                            f"Unexpected trainable param: {name}")

        # K projection must NOT be trainable (Q/V only)
        for name in trainable_names:
            self.assertNotIn("k_proj", name,
                             f"K projection should be frozen, got: {name}")

        # Q and V LoRA must be trainable
        q_lora = [n for n in trainable_names if "_qkv_lora_q" in n]
        v_lora = [n for n in trainable_names if "_qkv_lora_v" in n]
        self.assertGreater(len(q_lora), 0, "No Q LoRA params found")
        self.assertGreater(len(v_lora), 0, "No V LoRA params found")

        # Classifier must be trainable
        cls_params = [n for n in trainable_names if "classifier" in n]
        self.assertGreater(len(cls_params), 0, "Classifier params not trainable")

        # Frozen params should include backbone weights
        backbone_frozen = [n for n in frozen_names if "visual" in n
                           and "lora_" not in n]
        self.assertGreater(len(backbone_frozen), 0,
                           "No frozen backbone params found")

    def test_visual_lora_forward_runs(self):
        """Forward and backward pass complete without error."""
        from common.peft import apply_peft

        model = self._build_classifier_model()
        peft_cfg = {
            "type": "visual_lora",
            "lora_last_n_blocks": 4,
            "lora_rank": 4,
            "lora_alpha": 4,
            "lora_adapt_qv": True,
            "lora_adapt_out": True,
        }
        apply_peft(model, peft_cfg)

        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_labels = torch.randint(0, 500, (2,))

        model.train()
        logits = model(dummy_input)
        loss = nn.functional.cross_entropy(logits, dummy_labels)
        loss.backward()

        # LoRA params should have gradients
        lora_grads = []
        for n, p in model.named_parameters():
            if "lora_" in n and p.requires_grad:
                self.assertIsNotNone(p.grad, f"No grad for trainable {n}")
                lora_grads.append(n)
        self.assertGreater(len(lora_grads), 0)

        # K projection must have no gradient
        for n, p in model.named_parameters():
            if "_qkv_k_proj" in n:
                self.assertIsNone(p.grad, f"K projection got grad: {n}")

    def test_visual_lora_block_count(self):
        """Only the last N blocks are adapted."""
        from common.peft import apply_peft

        model = self._build_classifier_model()
        peft_cfg = {
            "type": "visual_lora",
            "lora_last_n_blocks": 2,
            "lora_rank": 4,
            "lora_alpha": 4,
            "lora_adapt_qv": True,
            "lora_adapt_out": True,
        }
        apply_peft(model, peft_cfg)

        # Check that only blocks 10-11 (last 2 of 12) have LoRA
        blocks = model.visual.transformer.resblocks
        for i, block in enumerate(blocks):
            has_q_lora = hasattr(block.attn, "_qkv_lora_q")
            has_out_lora = hasattr(block.attn.out_proj, "lora_A")
            if i >= len(blocks) - 2:  # last 2 blocks
                self.assertTrue(has_q_lora or has_out_lora,
                                f"Block {i} should have LoRA")
            else:
                self.assertFalse(has_q_lora,
                                 f"Block {i} should NOT have Q LoRA")

    def test_lora_n_blocks_exceeds_total_raises(self):
        """Requesting more blocks than available raises ValueError."""
        from common.peft import apply_peft

        model = self._build_classifier_model()
        peft_cfg = {
            "type": "visual_lora",
            "lora_last_n_blocks": 999,
            "lora_rank": 4,
            "lora_alpha": 4,
            "lora_adapt_qv": True,
            "lora_adapt_out": True,
        }
        with self.assertRaises(ValueError):
            apply_peft(model, peft_cfg)


# ──────────────────────────────────────────────────────────────────────
# 4. Regression: old PEFT configs unchanged
# ──────────────────────────────────────────────────────────────────────


class TestOldPEFTRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import clip
        cls.clip_model, cls.preprocess = clip.load("ViT-B/32", device="cpu")
        cls.clip_model.visual = cls.clip_model.visual.float()

    def _build_model(self):
        from experiments.baseline.model import CLIPLinearClassifier
        clip_copy = copy.deepcopy(self.clip_model)
        return CLIPLinearClassifier(clip_copy, num_classes=500, freeze_clip=False)

    def test_linear_head_only(self):
        from common.peft import apply_peft
        model = self._build_model()
        info = apply_peft(model, {"type": "linear_head_only"})
        self.assertEqual(info["peft_type"], "linear_head_only")
        # Only classifier trainable
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.assertIn("classifier", n)

    def test_last_block_lora(self):
        from common.peft import apply_peft
        model = self._build_model()
        peft_cfg = {
            "type": "last_block_lora",
            "lora": {"rank": 4, "alpha": 4, "target_block": 11},
        }
        info = apply_peft(model, peft_cfg)
        self.assertEqual(info["peft_type"], "last_block_lora")
        self.assertGreater(len(info["lora_layers"]), 0)

    def test_ln_post_and_proj(self):
        from common.peft import apply_peft
        model = self._build_model()
        info = apply_peft(model, {"type": "ln_post_and_proj"})
        self.assertEqual(info["peft_type"], "ln_post_and_proj")
        # ln_post and proj should be trainable
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any("ln_post" in n for n in trainable))

    def test_visual_layernorm_only(self):
        from common.peft import apply_peft
        model = self._build_model()
        info = apply_peft(model, {"type": "visual_layernorm_only"})
        self.assertEqual(info["peft_type"], "visual_layernorm_only")

    def test_unknown_type_raises(self):
        from common.peft import apply_peft
        model = self._build_model()
        with self.assertRaises(ValueError):
            apply_peft(model, {"type": "nonexistent_type"})


# ──────────────────────────────────────────────────────────────────────
# 5. Feature distillation tests
# ──────────────────────────────────────────────────────────────────────


class TestFeatureDistillation(unittest.TestCase):

    def test_distill_loss_finite_and_scalar(self):
        """Distillation loss is finite and has correct shape."""
        from common.feature_distillation import FeatureDistillation

        # Two simple linear "models" for test
        class DummyEncoder(nn.Module):
            def encode_image(self, x):
                return nn.functional.normalize(
                    x.flatten(1)[:, :512].float(), p=2, dim=-1,
                )

        parent = DummyEncoder()
        student = DummyEncoder()
        distill = FeatureDistillation(parent)

        images = torch.randn(4, 3, 224, 224)
        s_feat = student.encode_image(images)
        p_feat = distill.get_parent_features(images)
        loss = distill.compute_loss(s_feat, p_feat)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(loss.ndim, 0)  # scalar

    def test_teacher_parameters_have_no_grad(self):
        """Parent model parameters must have requires_grad=False."""
        from common.feature_distillation import FeatureDistillation

        class DummyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(512, 512)

            def encode_image(self, x):
                return self.fc(x.flatten(1)[:, :512])

        parent = DummyEncoder()
        FeatureDistillation(parent)

        for p in parent.parameters():
            self.assertFalse(p.requires_grad)

    def test_parent_features_detached(self):
        """Parent features have no gradient history."""
        from common.feature_distillation import FeatureDistillation

        class DummyEncoder(nn.Module):
            def encode_image(self, x):
                return nn.functional.normalize(
                    x.flatten(1)[:, :512].float(), p=2, dim=-1,
                )

        parent = DummyEncoder()
        distill = FeatureDistillation(parent)
        images = torch.randn(4, 3, 224, 224)
        p_feat = distill.get_parent_features(images)

        self.assertFalse(p_feat.requires_grad)

    def test_distill_loss_on_subset(self):
        """Distillation loss computed on a subset of samples (rejected only)."""
        from common.feature_distillation import FeatureDistillation

        class DummyEncoder(nn.Module):
            def __init__(self, shift: float = 0.0):
                super().__init__()
                self.shift = shift

            def encode_image(self, x):
                base = nn.functional.normalize(
                    x.flatten(1)[:, :512].float(), p=2, dim=-1,
                )
                return nn.functional.normalize(base + self.shift, p=2, dim=-1)

        parent = DummyEncoder(shift=0.0)
        student = DummyEncoder(shift=0.1)  # deliberately different
        distill = FeatureDistillation(parent)

        images = torch.randn(8, 3, 224, 224)
        s_feat = student.encode_image(images)
        p_feat = distill.get_parent_features(images)

        # Only samples 3-5 (indices 3, 4, 5) are "rejected"
        rejected_idx = torch.tensor([3, 4, 5])
        loss_rejected = distill.compute_loss(
            s_feat[rejected_idx], p_feat[rejected_idx],
        )
        loss_all = distill.compute_loss(s_feat, p_feat)

        self.assertTrue(torch.isfinite(loss_rejected).item())
        # Loss on rejected subset should differ from full set
        self.assertNotAlmostEqual(loss_rejected.item(), loss_all.item(), places=3)


# ──────────────────────────────────────────────────────────────────────
# 6. Config & integration tests
# ──────────────────────────────────────────────────────────────────────


class TestConfigAndIntegration(unittest.TestCase):

    def test_nr_combined_upgrade_config_loads(self):
        """Config file exists and has required sections."""
        from common.utils import load_config

        config = load_config("configs/nr_combined_upgrade.yaml")
        self.assertEqual(config["experiment"]["id"], "NR_COMBINED_UPGRADE")
        self.assertEqual(config["peft"]["type"], "visual_lora")
        self.assertEqual(config["peft"]["lora_last_n_blocks"], 4)
        self.assertEqual(config["peft"]["lora_rank"], 8)
        self.assertEqual(config["peft"]["lora_alpha"], 8)
        self.assertTrue(config["peft"]["lora_adapt_qv"])
        self.assertTrue(config["peft"]["lora_adapt_out"])
        self.assertEqual(config["sample_weighting"]["type"], "oof_manifest")
        self.assertEqual(config["sample_weighting"]["clean_prob_threshold"], 0.70)
        self.assertEqual(config["sample_weighting"]["feature_distillation_weight"], 2.0)
        self.assertEqual(config["train"]["epochs"], 6)
        self.assertEqual(config["train"]["batch_size"], 64)
        self.assertFalse(config["mixup"]["enabled"])

    def test_config_no_mixup(self):
        """MixUp is disabled for LoRA training."""
        from common.utils import load_config

        config = load_config("configs/nr_combined_upgrade.yaml")
        self.assertFalse(
            config.get("mixup", {}).get("enabled", False),
            "MixUp must be disabled for NR_COMBINED_UPGRADE",
        )

    def test_config_sample_weighting_includes_feat_distill(self):
        """Feature distillation weight is specified in config."""
        from common.utils import load_config

        config = load_config("configs/nr_combined_upgrade.yaml")
        sw = config.get("sample_weighting", {})
        self.assertIn("feature_distillation_weight", sw)
        self.assertGreater(sw["feature_distillation_weight"], 0)


# ──────────────────────────────────────────────────────────────────────
# 7. Pre-PEFT weight loading & checkpoint round-trip
# ──────────────────────────────────────────────────────────────────────


class TestPrePEFTWeightLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import clip
        cls.clip_model, cls.preprocess = clip.load("ViT-B/32", device="cpu")
        cls.clip_model.visual = cls.clip_model.visual.float()

    def _build_model(self):
        from experiments.baseline.model import CLIPLinearClassifier
        clip_copy = copy.deepcopy(self.clip_model)
        return CLIPLinearClassifier(clip_copy, num_classes=500, freeze_clip=False)

    def test_strict_load_before_peft_succeeds(self):
        """Parent state_dict loads with strict=True BEFORE PEFT (no LoRA keys)."""
        model = self._build_model()
        state_dict = model.state_dict()

        # Verify strict loading works
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)

    def test_strict_load_after_visual_lora_has_missing_keys(self):
        """After visual_lora, parent state_dict won't match (strict fails)."""
        from common.peft import apply_peft

        model = self._build_model()
        parent_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Apply LoRA → key names change (out_proj.weight → out_proj.base.weight)
        apply_peft(model, {
            "type": "visual_lora",
            "lora_last_n_blocks": 4,
            "lora_rank": 8,
            "lora_alpha": 8,
            "lora_adapt_qv": True,
            "lora_adapt_out": True,
        })

        # Strict load should fail because LoRA keys don't exist in parent
        with self.assertRaises(RuntimeError):
            model.load_state_dict(parent_state, strict=True)

        # Non-strict should succeed with missing (LoRA) and unexpected (base) keys
        missing, unexpected = model.load_state_dict(parent_state, strict=False)
        self.assertGreater(len(missing), 0, "Should have missing LoRA keys")
        self.assertGreater(len(unexpected), 0,
                           "Should have unexpected base keys (wrapped by LoRA)")

    def test_lora_ckpt_roundtrip(self):
        """LoRA parameters survive save → load cycle."""
        from common.peft import apply_peft

        model = self._build_classifier_model()
        peft_cfg = {
            "type": "visual_lora",
            "lora_last_n_blocks": 2,
            "lora_rank": 4,
            "lora_alpha": 4,
            "lora_adapt_qv": True,
            "lora_adapt_out": True,
        }
        apply_peft(model, peft_cfg)

        # Tweak a LoRA param to nonzero
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.data.add_(0.1)

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"model_state_dict": model.state_dict()}, f.name)

            # Load into fresh model
            model2 = self._build_classifier_model()
            apply_peft(model2, peft_cfg)
            ckpt = torch.load(f.name, map_location="cpu")
            missing, unexpected = model2.load_state_dict(
                ckpt["model_state_dict"], strict=False,
            )
            # LoRA keys should match → no missing (except maybe non-LoRA adapters)
            lora_missing = [k for k in missing if "lora_" in k]
            self.assertEqual(len(lora_missing), 0,
                             f"LoRA keys lost in round-trip: {lora_missing}")

        # Verify LoRA params match
        for n, p in model.named_parameters():
            if "lora_" in n:
                p2 = dict(model2.named_parameters())[n]
                self.assertTrue(torch.equal(p.data, p2.data),
                                f"LoRA param {n} mismatch after round-trip")

    def _build_classifier_model(self):
        from experiments.baseline.model import CLIPLinearClassifier
        clip_copy = copy.deepcopy(self.clip_model)
        return CLIPLinearClassifier(clip_copy, num_classes=500, freeze_clip=False)


# ──────────────────────────────────────────────────────────────────────
# 8. CPU dry-run (single batch)
# ──────────────────────────────────────────────────────────────────────


class TestCPUDryRun(unittest.TestCase):

    def test_cpu_single_batch_dry_run(self):
        """A single forward/backward pass completes on CPU without error."""
        from common.peft import apply_peft
        from common.feature_distillation import FeatureDistillation

        # Build minimal model
        import clip
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        clip_model.visual = clip_model.visual.float()

        from experiments.baseline.model import CLIPLinearClassifier
        clip_copy1 = copy.deepcopy(clip_model)
        # Student: freeze_clip=False so LoRA gradients flow through encode_image
        model = CLIPLinearClassifier(clip_copy1, num_classes=500, freeze_clip=False)

        # Load a dummy "parent" state (copy of initial weights, pre-LoRA)
        parent_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Apply visual_lora
        peft_cfg = {
            "type": "visual_lora",
            "lora_last_n_blocks": 2,
            "lora_rank": 4,
            "lora_alpha": 4,
            "lora_adapt_qv": True,
            "lora_adapt_out": True,
        }
        apply_peft(model, peft_cfg)

        # Build frozen parent for distillation — no LoRA, frozen CLIP
        clip_copy2 = copy.deepcopy(clip_model)
        parent_model = CLIPLinearClassifier(clip_copy2, num_classes=500, freeze_clip=True)
        parent_model.load_state_dict(parent_state, strict=True)
        for p in parent_model.parameters():
            p.requires_grad_(False)
        parent_model.eval()

        distill = FeatureDistillation(parent_model)

        # Single batch
        images = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])

        model.train()
        logits = model(images)
        task_loss = nn.functional.cross_entropy(logits, labels)

        # Feature distillation on both samples (simulating rejected)
        s_feat = model.encode_image(images)
        with torch.no_grad():
            p_feat = distill.get_parent_features(images)
        feat_loss = distill.compute_loss(s_feat, p_feat)

        total_loss = task_loss + 2.0 * feat_loss
        total_loss.backward()

        # Verify gradients flow
        head_grad = model.classifier.weight.grad
        self.assertIsNotNone(head_grad, "Classifier head got no gradient")
        self.assertGreater(head_grad.abs().sum().item(), 0,
                           "Classifier gradient is zero")

        # Verify parent model has NO gradients
        for n, p in parent_model.named_parameters():
            self.assertIsNone(p.grad, f"Parent param {n} should have no grad")

        # Verify LoRA params get gradients
        lora_with_grad = 0
        for n, p in model.named_parameters():
            if "lora_" in n and p.requires_grad:
                if p.grad is not None:
                    lora_with_grad += 1
        self.assertGreater(lora_with_grad, 0,
                           "No LoRA parameters received gradients")

        self.assertTrue(torch.isfinite(total_loss).item(),
                        "Total loss is not finite")


if __name__ == "__main__":
    unittest.main()
