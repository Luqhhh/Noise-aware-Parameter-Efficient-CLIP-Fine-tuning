import numpy as np
import pandas as pd

from scripts.prepare_l05_oof_hard_filter import make_folds, select_rejections


def test_group_folds_keep_identical_content_together_and_train_all_classes():
    frame = pd.DataFrame({
        'image_path': [f'train/{label}/{i:03d}.jpg' for label in range(3) for i in range(12)],
        'label': [label for label in range(3) for _ in range(12)],
        'content_group': [f'g-{label}-{i // 2}' for label in range(3) for i in range(12)],
    })
    folds = make_folds(frame, folds=3, seed=42, classes=3)
    assert len(folds) == len(frame)
    assert set(folds) == {0, 1, 2}
    assert pd.DataFrame({'group': frame.content_group, 'fold': folds}).groupby('group').fold.nunique().max() == 1


def test_consensus_and_per_class_cap_reject_only_strong_agreed_labels():
    frame = pd.DataFrame({
        'image_path': [f'train/{label}/{i:03d}.jpg' for label in range(3) for i in range(20)],
        'label': [label for label in range(3) for _ in range(20)],
        'content_group': [f'g-{label}-{i}' for label in range(3) for i in range(20)],
    })
    scores = pd.DataFrame({
        'centroid_top1': frame.label.to_numpy().copy(),
        'ridge_top1': frame.label.to_numpy().copy(),
        'centroid_margin': np.zeros(len(frame)),
        'ridge_margin': np.zeros(len(frame)),
    })
    for row, margin in ((0, 0.9), (1, 0.8), (20, 0.7), (40, 0.6)):
        scores.loc[row, 'centroid_top1'] = (frame.loc[row, 'label'] + 1) % 3
        scores.loc[row, 'ridge_top1'] = (frame.loc[row, 'label'] + 1) % 3
        scores.loc[row, ['centroid_margin', 'ridge_margin']] = margin
    scores.loc[40, 'ridge_top1'] = 1  # disagreement is ineligible
    rejected = select_rejections(
        frame, scores, max_drop_fraction=0.05, max_drop_per_class_fraction=0.05,
        minimum_class_support_for_drop=20, minimum_class_size_after_drop=4,
    )
    assert rejected.image_path.tolist() == ['train/0/000.jpg', 'train/1/000.jpg']
