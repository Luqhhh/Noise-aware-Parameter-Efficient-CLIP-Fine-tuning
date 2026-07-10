"""
Utility functions for the CLIP baseline project.

Provides: config loading, seed setting, path management, logging utilities,
and miscellaneous helpers used across the project.
"""

import json
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def set_train_seed(seed: int) -> None:
    """Set random seeds for training reproducibility.

    Unlike set_seed (used for split generation), this does NOT set
    cudnn.deterministic=True because it significantly slows training.
    It sets the core seeds that ensure DataLoader shuffles and model
    initialization are reproducible.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(log_dir: str, name: str = "train") -> logging.Logger:
    """Set up logging to both console and file.

    Args:
        log_dir: Directory to save log files.
        name: Logger name prefix.

    Returns:
        Configured logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def save_config_snapshot(config: Dict[str, Any], output_dir: str) -> None:
    """Save a copy of the config for reproducibility.

    Args:
        config: Configuration dictionary.
        output_dir: Directory to save the snapshot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = output_dir / f"config_snapshot_{timestamp}.yaml"

    with open(snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def count_parameters(model: torch.nn.Module) -> tuple:
    """Count total and trainable parameters of a model.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds: float) -> str:
    """Format a time duration in seconds to a human-readable string.

    Args:
        seconds: Time duration in seconds.

    Returns:
        Formatted string like "1h 23m 45s".
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return as Path.

    Args:
        path: Directory path.

    Returns:
        pathlib.Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
