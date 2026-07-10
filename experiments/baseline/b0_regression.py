"""
B0 Regression Fixture.

B0 is the original baseline protocol, verifying that infrastructure refactoring
did not change the baseline results. It uses the original complete training
protocol (lr=1e-3, wd=1e-4, epochs=20, batch_size=128, AdamW, CosineAnnealingLR,
warmup=1, AMP, max_grad_norm=1.0, split_seed=42, train_seed=42).

This fixture is saved as JSON for easy comparison across refactoring attempts.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

B0_FIXTURE: Dict[str, Any] = {
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


def save_b0_fixture(output_dir: str) -> str:
    """Save the B0 regression fixture as JSON.

    Args:
        output_dir: Directory to save the fixture file.

    Returns:
        Path to the saved fixture file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fixture_path = output_path / "b0_fixture.json"
    with open(fixture_path, "w") as f:
        json.dump(B0_FIXTURE, f, indent=2)

    logger.info(f"B0 regression fixture saved to {fixture_path}")
    return str(fixture_path)
