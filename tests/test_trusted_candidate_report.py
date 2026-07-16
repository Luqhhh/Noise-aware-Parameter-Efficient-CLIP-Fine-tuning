import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.trusted_validation.evaluate_candidate import (
    evaluate_candidate,
    stable_sample_key,
)


class TestTrustedCandidateReport(unittest.TestCase):
    def _signals(self):
        rows = []
        for label in (0, 1):
            for index in range(5):
                rows.append(
                    {
                        "image_path": f"/other/train/{label:04d}/{index}.jpg",
                        "noisy_label": label,
                        "knn_label_agreement": 0.9,
                        "prototype_supports_noisy_label": True,
                        "prototype_margin": 0.08,
                        "clip_flip_cosine": 0.95,
                        "cross_class_duplicate_conflict": False,
                    }
                )
        return pd.DataFrame(rows)

    def _predictions(self):
        rows = []
        for label in (0, 1):
            for index in range(5):
                rows.append(
                    {
                        "image_path": f"/local/repo/train/{label:04d}/{index}.jpg",
                        "true_label": label,
                        "pred_label": label if index < 4 else 1 - label,
                        "pred_conf": 0.9,
                    }
                )
        return pd.DataFrame(rows)

    def test_stable_key_ignores_machine_prefix(self):
        self.assertEqual(
            stable_sample_key("/a/b/train/0042/x.jpg"),
            "0042/x.jpg",
        )

    def test_report_has_full_coverage_and_finite_trusted_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_path = root / "signals.csv"
            prediction_path = root / "predictions.csv"
            parent_path = root / "parent.csv"
            output_dir = root / "report"
            self._signals().to_csv(signal_path, index=False)
            predictions = self._predictions()
            predictions.to_csv(prediction_path, index=False)
            parent = predictions.copy()
            parent["pred_label"] = parent["true_label"]
            parent.to_csv(parent_path, index=False)

            metrics = evaluate_candidate(
                "candidate",
                prediction_path,
                signal_path,
                output_dir,
                parent_path,
            )

            self.assertAlmostEqual(metrics["raw_micro"], 0.8)
            self.assertAlmostEqual(metrics["raw_macro"], 0.8)
            self.assertAlmostEqual(metrics["trusted_micro"], 0.8)
            self.assertEqual(metrics["trusted_represented_classes"], 2)
            self.assertAlmostEqual(metrics["prediction_change_vs_parent"], 0.2)
            self.assertTrue((output_dir / "protocol_audit.json").exists())
            self.assertTrue((output_dir / "per_class_delta.csv").exists())

    def test_coverage_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_path = root / "signals.csv"
            prediction_path = root / "predictions.csv"
            self._signals().to_csv(signal_path, index=False)
            self._predictions().iloc[:-1].to_csv(prediction_path, index=False)

            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                evaluate_candidate(
                    "candidate",
                    prediction_path,
                    signal_path,
                    root / "report",
                )


if __name__ == "__main__":
    unittest.main()
