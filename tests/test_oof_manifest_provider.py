import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from common.sample_weighting import OOFManifestProvider


class TestOOFManifestProvider(unittest.TestCase):
    def test_symlinked_dataset_path_matches_resolved_manifest_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "real_train" / "0001"
            target_dir.mkdir(parents=True)
            image_path = target_dir / "sample.jpg"
            image_path.touch()

            alias_root = root / "train_dedup"
            alias_root.symlink_to(root / "real_train", target_is_directory=True)
            alias_path = alias_root / "0001" / "sample.jpg"

            manifest_path = root / "manifest.csv"
            pd.DataFrame(
                {
                    "sample_id": ["sample"],
                    "image_path": [str(alias_path)],
                    "original_label": [1],
                    "training_label": [1],
                    "sample_weight": [0.3],
                    "quality_score": [0.2],
                }
            ).to_csv(manifest_path, index=False)

            provider = OOFManifestProvider(str(manifest_path))
            weights = provider.get_weights(
                [str(alias_path)],
                labels=torch.tensor([1]),
                epoch=0,
            )

            self.assertAlmostEqual(weights.item(), 0.3)


if __name__ == "__main__":
    unittest.main()
