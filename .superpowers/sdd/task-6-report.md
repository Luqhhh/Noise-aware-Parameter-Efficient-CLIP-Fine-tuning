# Task 6: Generate artifact_manifest.json

**Status:** Complete

**Commit hash:** `201b4cb`

**Test result:** `python3 -c "from experiments.baseline.train import main; print('import OK')"` passed.

**Changes made:**
1. Added import: `from common.artifact_manifest import build_artifact_manifest, write_artifact_manifest` at line 38 of `experiments/baseline/train.py`.
2. Added artifact manifest generation block after the `eval_results` save block (after `train_logger.info(f"Eval results saved to: {eval_path}")`) inside the `if mode in ("dev", "confirm"):` block. This calls `build_artifact_manifest()` with the `resolved` config, checkpoint paths, split CSVs, and extra metadata (best accuracies, best epoch, sample weighting type), then writes via `write_artifact_manifest()`.

**Concerns:** None.
