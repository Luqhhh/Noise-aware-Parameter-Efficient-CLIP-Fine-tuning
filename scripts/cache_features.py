#!/usr/bin/env python3
"""
Build the deterministic CLIP feature cache for a given stage.

Encodes the FULL training set once with frozen CLIP ViT-B/32 and saves
features to cache/{stage}/clip_vit_b32_openai/.

Usage:
    python scripts/cache_features.py --config configs/baseline.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from common.cache import FeatureCacheBuilder
from common.utils import load_config

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build CLIP feature cache")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    config = load_config(args.config)

    device_str = args.device or config.get("train", {}).get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    builder = FeatureCacheBuilder(config, device)
    cache_dir = builder.build()
    logger.info(f"Cache complete: {cache_dir}")


if __name__ == "__main__":
    main()
