"""Fail-closed split lineage audit CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_clip.config import load_config
from aegis_clip.lineage import run_lineage_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    train_cfg = config["train"]
    source = train_cfg.get("init_checkpoint")
    if not source:
        raise ValueError("train.init_checkpoint is required for lineage audit")
    run_dir = (
        Path(config["output"]["root"])
        / config["project"]["experiment_id"]
        / f"seed{config['project'].get('seed', 42)}"
    )
    output = Path(args.output) if args.output else run_dir / "split_lineage_audit.json"
    audit = run_lineage_audit(
        config,
        child_train_csv=config["data"]["train_csv"],
        child_val_csv=config["data"]["val_csv"],
        checkpoint_path=source,
        output_path=output,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
