import unittest
import warnings

import numpy as np
import pandas as pd
import torch

from analysis.oof.quality import add_quality_weights, build_sample_quality

warnings.filterwarnings("error", category=UserWarning)


class TestOOFQuality(unittest.TestCase):
    def test_build_sample_quality_derives_probability_and_margin_fields(self):
        assignments = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "image_path": ["a.jpg", "b.jpg", "c.jpg"],
                "label": [0, 1, 1],
                "fold": [0, 1, 2],
            }
        )
        logits = torch.tensor(
            [[3.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]
        )

        warnings.simplefilter("error", UserWarning)
        quality = build_sample_quality(
            assignments=assignments,
            logits=logits,
            prototype_own_similarity=np.array([0.8, 0.4, 0.7]),
            prototype_margin=np.array([0.3, -0.1, 0.2]),
            prototype_top1=np.array([0, 0, 1]),
            knn_agreement=np.array([1.0, 0.2, 0.8]),
            knn_top1=np.array([0, 0, 1]),
            flip_consistency=np.array([1.0, 0.0, 1.0]),
            clip_flip_cosine=np.array([0.99, 0.75, 0.96]),
            duplicate_conflict_flag=np.array([False, True, False]),
        )

        probs = logits.softmax(dim=1)
        self.assertEqual(quality["oof_top1"].tolist(), [0, 0, 1])
        self.assertAlmostEqual(quality.loc[1, "p_original_label"], probs[1, 1].item())
        self.assertAlmostEqual(
            quality.loc[0, "top1_margin"],
            (probs[0].topk(2).values[0] - probs[0].topk(2).values[1]).item(),
        )
        self.assertAlmostEqual(
            quality.loc[2, "oof_cross_entropy"],
            -np.log(probs[2, 1].item()),
        )
        self.assertEqual(quality["class_frequency"].tolist(), [1, 2, 2])
        self.assertEqual(quality["duplicate_conflict_flag"].tolist(), [False, True, False])

    def test_add_quality_weights_stays_in_protocol_ranges(self):
        frame = pd.DataFrame(
            {
                "original_label": [0, 0, 0, 1, 1, 1],
                "p_original_label": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8],
                "prototype_margin": [-0.2, 0.0, 0.3, -0.1, 0.1, 0.2],
                "knn_agreement": [0.0, 0.5, 1.0, 0.2, 0.6, 0.9],
                "flip_consistency": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            }
        )

        weighted = add_quality_weights(frame)

        self.assertTrue(weighted["quality"].between(0.0, 1.0).all())
        self.assertTrue(weighted["soft_weight"].between(0.3, 1.0).all())
        self.assertEqual(set(weighted["discrete_weight"]), {0.3, 0.6, 1.0})
        self.assertLess(weighted.loc[0, "quality"], weighted.loc[2, "quality"])


if __name__ == "__main__":
    unittest.main()
