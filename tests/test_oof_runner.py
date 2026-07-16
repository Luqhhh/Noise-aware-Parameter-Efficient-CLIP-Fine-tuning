import unittest

import torch

from analysis.oof.run_oof import (
    compute_reference_signals,
    infer_logits,
    train_linear_head,
)


class TestOOFRunner(unittest.TestCase):
    def test_linear_head_training_and_inference_are_reproducible(self):
        features = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 0.9],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 2])

        first, _ = train_linear_head(
            features, labels, num_classes=3, epochs=15, batch_size=3,
            lr=0.1, weight_decay=0.0, warmup_epochs=1, q=0.5,
            seed=42, device=torch.device("cpu"),
        )
        second, _ = train_linear_head(
            features, labels, num_classes=3, epochs=15, batch_size=3,
            lr=0.1, weight_decay=0.0, warmup_epochs=1, q=0.5,
            seed=42, device=torch.device("cpu"),
        )

        first_logits = infer_logits(first, features, 2, torch.device("cpu"))
        second_logits = infer_logits(second, features, 2, torch.device("cpu"))
        self.assertTrue(torch.equal(first_logits, second_logits))
        self.assertEqual(tuple(first_logits.shape), (6, 3))

    def test_reference_signals_use_only_training_bank(self):
        train_features = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        )
        train_labels = torch.tensor([0, 0, 1, 1])
        holdout_features = torch.tensor([[0.95, 0.05], [0.05, 0.95]])
        holdout_labels = torch.tensor([0, 1])

        signals = compute_reference_signals(
            train_features, train_labels, holdout_features, holdout_labels,
            num_classes=2, k_neighbors=2, query_batch_size=2,
            reference_chunk_size=2, device=torch.device("cpu"),
        )

        self.assertEqual(signals["prototype_top1"].tolist(), [0, 1])
        self.assertEqual(signals["knn_top1"].tolist(), [0, 1])
        self.assertEqual(signals["knn_agreement"].tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
