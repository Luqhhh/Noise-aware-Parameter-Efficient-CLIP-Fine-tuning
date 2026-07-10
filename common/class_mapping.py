"""
Canonical class mapping — generated once from the full training directory,
stored at data/{stage}/metadata/, and reused by ALL stages (dev/confirm/final-fit).

Lifecycle:
  - not exist -> generate
  - exists and matches -> reuse
  - exists but inconsistent -> ValueError (needs --regenerate-class-mapping)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def validate_class_directory_names(class_names):
    """Validate that class directory names are 4-digit strings.

    Args:
        class_names: List of class directory names.

    Raises:
        ValueError: If any class name doesn't match the 4-digit format.
    """
    for name in class_names:
        if len(name) != 4 or not name.isdigit():
            raise ValueError(f"Invalid class directory name: {name!r}")
    logger.info(f"Validated {len(class_names)} class directory names.")


def generate_canonical_mapping(train_dir, expected_num_classes):
    """Generate class_to_idx and idx_to_class from a training directory.

    Args:
        train_dir: Path to the training data root (class subdirectories).
        expected_num_classes: Expected number of classes (500 preliminary, etc).

    Returns:
        Tuple of (class_to_idx: Dict[str,int], idx_to_class: Dict[str,str]).

    Raises:
        ValueError: If directory count != expected, or any class name is invalid.
    """
    train_dir = Path(train_dir)
    class_dirs = sorted(
        [p for p in train_dir.iterdir() if p.is_dir()],
        key=lambda x: x.name,
    )
    class_names = [d.name for d in class_dirs]

    validate_class_directory_names(class_names)

    if len(class_names) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, found {len(class_names)}"
        )

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {str(i): name for name, i in class_to_idx.items()}

    logger.info(
        f"Generated canonical mapping: {len(class_to_idx)} classes "
        f"from {train_dir}"
    )
    return class_to_idx, idx_to_class



def load_or_generate_mapping(
    metadata_dir, train_dir, expected_num_classes, regenerate=False
):
    """Load an existing canonical mapping or generate a new one.

    Lifecycle:
      - not exist -> generate and save
      - exists and matches train_dir -> load
      - exists but inconsistent -> ValueError (regenerate=True to force)

    Args:
        metadata_dir: Path to store/load mapping JSON files.
        train_dir: Path to training data root.
        expected_num_classes: Expected number of classes.
        regenerate: If True, overwrite existing mapping.

    Returns:
        Tuple of (class_to_idx: Dict[str,int], idx_to_class: Dict[str,str]).

    Raises:
        ValueError: If existing mapping is inconsistent with train_dir.
    """
    metadata_dir = Path(metadata_dir)
    class_to_idx_path = metadata_dir / "class_to_idx.json"
    idx_to_class_path = metadata_dir / "idx_to_class.json"

    if regenerate or not class_to_idx_path.exists():
        metadata_dir.mkdir(parents=True, exist_ok=True)
        class_to_idx, idx_to_class = generate_canonical_mapping(
            train_dir, expected_num_classes
        )
        with open(class_to_idx_path, "w") as f:
            json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
        with open(idx_to_class_path, "w") as f:
            json.dump(idx_to_class, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved canonical mapping to {metadata_dir}")
        return class_to_idx, idx_to_class

    # Load existing
    with open(class_to_idx_path, "r") as f:
        class_to_idx = json.load(f)
    with open(idx_to_class_path, "r") as f:
        idx_to_class = json.load(f)

    # Validate against current directory
    train_dir = Path(train_dir)
    current_dirs = sorted(
        [p.name for p in train_dir.iterdir() if p.is_dir()]
    )

    validate_class_directory_names(current_dirs)

    if len(current_dirs) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, found {len(current_dirs)}"
        )

    expected_class_to_idx = {name: i for i, name in enumerate(current_dirs)}
    if class_to_idx != expected_class_to_idx:
        raise ValueError(
            f"Cached class mapping is inconsistent with current training directory. "
            f"Re-run with --regenerate-class-mapping to overwrite."
        )

    logger.info(f"Loaded existing canonical mapping from {metadata_dir}")
    return class_to_idx, idx_to_class
