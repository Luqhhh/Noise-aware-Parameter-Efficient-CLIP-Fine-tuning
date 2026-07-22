import torch

from aegis_clip.fine_trust import (
    conflict_geometry_cap,
    cross_fitted_fine_scores,
)


def test_cross_fitted_fine_scores_follow_held_out_class_geometry() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [1.0, 0.0],
            [0.9, -0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.0, 1.0],
            [-0.1, 0.9],
        ]
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    folds = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    scores = cross_fitted_fine_scores(
        features, labels, folds, num_classes=2, power_iterations=5
    )
    assert scores.shape == (8,)
    assert torch.isfinite(scores).all()
    assert float(scores.min()) > 0.95


def test_conflict_geometry_cap_only_changes_same_alternative_conflicts() -> None:
    clean = torch.tensor([1.0, 0.8, 0.7, 0.6])
    labels = torch.tensor([0, 0, 0, 0])
    prototype = torch.tensor([0, 1, 1, 2])
    probe = torch.tensor([0, 1, 2, 2])
    fine = torch.tensor([0.2, 0.3, 0.1, 0.9])
    capped, conflict = conflict_geometry_cap(
        clean, labels, prototype, probe, fine
    )
    assert conflict.tolist() == [False, True, False, True]
    assert torch.allclose(capped, torch.tensor([1.0, 0.3, 0.7, 0.6]))
