import unittest

import pandas as pd

from analysis.oof.build_folds import assign_group_stratified_folds, audit_folds


def _toy_samples() -> pd.DataFrame:
    rows = []
    for label in range(3):
        for group_index in range(6):
            group = f"class-{label}-group-{group_index}"
            rows.append(
                {
                    "sample_id": f"{label}-{group_index}-a",
                    "image_path": f"train/{label:04d}/{group_index}-a.jpg",
                    "label": label,
                    "sha256": group,
                }
            )
            if group_index == 0:
                rows.append(
                    {
                        "sample_id": f"{label}-{group_index}-b",
                        "image_path": f"train/{label:04d}/{group_index}-b.jpg",
                        "label": label,
                        "sha256": group,
                    }
                )
    return pd.DataFrame(rows)


class TestOOFFolds(unittest.TestCase):
    def test_group_stratified_folds_keep_duplicate_groups_together(self):
        assignments = assign_group_stratified_folds(
            _toy_samples(), n_splits=3, seed=42
        )

        self.assertEqual(set(assignments["fold"]), {0, 1, 2})
        self.assertEqual(assignments.groupby("sha256")["fold"].nunique().max(), 1)
        self.assertTrue(assignments["sample_id"].is_unique)
        self.assertTrue(assignments["fold"].notna().all())

    def test_group_stratified_folds_are_deterministic(self):
        samples = _toy_samples()

        first = assign_group_stratified_folds(samples, n_splits=3, seed=2026)
        second = assign_group_stratified_folds(samples, n_splits=3, seed=2026)

        self.assertTrue(
            first[["sample_id", "fold"]].equals(second[["sample_id", "fold"]])
        )

    def test_fold_audit_rejects_original_validation_overlap(self):
        assignments = assign_group_stratified_folds(
            _toy_samples(), n_splits=3, seed=42
        )
        overlapping_path = assignments.iloc[0]["image_path"]

        with self.assertRaisesRegex(ValueError, "original validation"):
            audit_folds(
                assignments,
                original_val_paths={overlapping_path},
                n_splits=3,
            )

    def test_fold_audit_reports_complete_leak_free_assignment(self):
        assignments = assign_group_stratified_folds(
            _toy_samples(), n_splits=3, seed=42
        )

        audit = audit_folds(assignments, original_val_paths=set(), n_splits=3)

        self.assertEqual(audit["sample_count"], len(assignments))
        self.assertEqual(audit["duplicate_group_fold_leakage_count"], 0)
        self.assertEqual(audit["original_validation_overlap_count"], 0)
        self.assertEqual(sum(audit["fold_counts"].values()), len(assignments))


if __name__ == "__main__":
    unittest.main()
