import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.oof.finalize_manifests import (
    build_canonical_manifest,
    finalize_manifests,
)
from common.manifest_loader import ManifestLoader


class TestOOFManifestFinalize(unittest.TestCase):
    def _quality(self):
        return pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d", "e", "f"],
                "image_path": [
                    "train_dedup/0000/a.jpg",
                    "train_dedup/0000/b.jpg",
                    "train_dedup/0000/c.jpg",
                    "train_dedup/0000/d.jpg",
                    "train_dedup/0001/e.jpg",
                    "train_dedup/0001/f.jpg",
                ],
                "original_label": [0, 0, 0, 0, 1, 1],
                "quality": [0.1, 0.2, 0.7, 0.9, 0.6, 0.8],
                "soft_weight": [0.37, 0.44, 0.79, 0.93, 0.72, 0.86],
                "discrete_weight": [0.3, 0.3, 0.6, 1.0, 0.3, 1.0],
                "oof_top1": [1, 0, 0, 0, 1, 1],
                "p_original_label": [0.1, 0.2, 0.7, 0.9, 0.6, 0.8],
                "p_top1": [0.8, 0.5, 0.7, 0.9, 0.6, 0.8],
                "prototype_margin": [-0.2, -0.1, 0.2, 0.4, 0.1, 0.3],
                "knn_agreement": [0.0, 0.2, 0.8, 1.0, 0.6, 0.9],
                "flip_consistency": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

    def test_canonical_manifest_loads_with_training_schema(self):
        manifest = build_canonical_manifest(
            self._quality(), "soft_weight", "oof_soft"
        )
        self.assertTrue(
            {
                "training_label",
                "sample_weight",
                "quality_score",
                "source",
            }.issubset(manifest.columns)
        )
        self.assertEqual(
            manifest["training_label"].tolist(),
            manifest["original_label"].tolist(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            manifest.to_csv(path, index=False)
            loaded = ManifestLoader(str(path)).load()
            self.assertEqual(len(loaded), 6)

    def test_finalize_stops_when_soft_low_weight_gate_triggers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_path = root / "sample_quality.csv"
            strict_path = root / "train.csv"
            self._quality().to_csv(quality_path, index=False)
            self._quality()[["image_path", "original_label"]].rename(
                columns={"original_label": "label"}
            ).to_csv(strict_path, index=False)

            audit = finalize_manifests(quality_path, strict_path, root)

            self.assertFalse(audit["overall_training_allowed"])
            self.assertEqual(
                audit["decision"], "stop_before_weight_training"
            )
            self.assertEqual(
                audit["soft"]["classes_with_over_30pct_weight_below_0_5"],
                [0],
            )
            self.assertFalse(audit["discrete"]["training_allowed"])
            self.assertTrue(
                (root / "oof_soft_weight_manifest.csv").exists()
            )
            on_disk = json.loads((root / "weight_audit.json").read_text())
            self.assertEqual(on_disk["sample_count"], 6)

    def test_duplicate_sample_id_fails_closed(self):
        quality = self._quality()
        quality.loc[1, "sample_id"] = quality.loc[0, "sample_id"]
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            build_canonical_manifest(quality, "soft_weight", "oof_soft")


if __name__ == "__main__":
    unittest.main()
