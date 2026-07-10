"""
B0 regression protocol: reproduce the original baseline with exact hyperparameters.

This verifies that infrastructure refactoring didn't change the baseline results.
B0 MUST use online encoding and the original training protocol.

The B0_FIXTURE dict defines the resolved hyperparameters. The save_b0_fixture()
helper writes it to disk for regression comparison.
"""

import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

B0_FIXTURE: Dict[str, object] = {
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "epochs": 20,
    "scheduler": "CosineAnnealingLR",
    "warmup_epochs": 1,
    "amp": True,
    "max_grad_norm": 1.0,
    "checkpoint_policy": "best_val",
    "split_seed": 42,
    "train_seed": 42,
}


def save_b0_fixture(output_dir: str) -> Path:
    """Save the resolved B0 config for regression comparison.

    Args:
        output_dir: Directory where the fixture JSON will be saved.

    Returns:
        Path to the saved fixture file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / "b0_regression_fixture.json"
    with open(fixture_path, "w") as f:
        json.dump(B0_FIXTURE, f, indent=2)
    logger.info(f"B0 regression fixture saved to {fixture_path}")
    return fixture_path
