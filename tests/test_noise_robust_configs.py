"""Validate Wave A configs share the standard block and differ only in expected fields."""

import yaml
import pytest

CONFIGS = [
    "configs/nr_ctrl_oof_zero_0001_fixed.yaml",
    "configs/nr_cl_classwise_drop.yaml",
    "configs/nr_cl_knn_drop.yaml",
    "configs/nr_consensus_relabel_v2_top100.yaml",
    "configs/nr_consensus_relabel_v2_top300.yaml",
]


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


class TestWaveAConfigs:
    def test_split_dir_consistent(self):
        for path in CONFIGS:
            c = _load(path)
            assert c["data"]["split_dir"] == "outputs/data/d3_strict/seed42", path

    def test_backbone_frozen(self):
        for path in CONFIGS:
            c = _load(path)
            assert c["model"]["clip_model_name"] == "ViT-B/32", path
            assert c["model"]["freeze_clip"] is True, path

    def test_standard_training_recipe(self):
        for path in CONFIGS:
            c = _load(path)
            t = c["train"]
            assert t["lr"] == 0.005, path
            assert t["epochs"] == 50, path
            assert t["weight_decay"] == 0.0001, path

    def test_gce_q05_with_mixup(self):
        for path in CONFIGS:
            c = _load(path)
            assert c["loss"]["name"] == "gce", path
            assert c["loss"]["q"] == 0.5, path
            assert c["mixup"]["enabled"] is True, path

    def test_missing_weight_policy_is_error(self):
        for path in CONFIGS:
            c = _load(path)
            sw = c.get("sample_weighting", {})
            assert sw.get("missing_weight_policy") == "error", path
            assert "missing_policy" not in sw, f"{path} uses deprecated missing_policy"

    def test_min_weight_zero(self):
        for path in CONFIGS:
            c = _load(path)
            sw = c.get("sample_weighting", {})
            assert sw.get("min_weight") == 0.0, path

    def test_experiment_ids_differ(self):
        ids = [_load(p)["experiment"]["id"] for p in CONFIGS]
        assert len(set(ids)) == len(ids), f"Duplicate experiment ids: {ids}"
