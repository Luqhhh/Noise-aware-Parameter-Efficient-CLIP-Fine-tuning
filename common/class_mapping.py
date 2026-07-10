"""
Canonical class mapping generation and management.

The canonical class mapping is generated once from the full training directory
and reused by ALL stages of the pipeline. This ensures consistency across
dev, confirm, final-fit, and inference.

Lifecycle:
    - Mapping does not exist -> generate and save
    - Mapping exists and matches current data -> reuse
    - Mapping exists but data has changed -> error (requires --regenerate-class-mapping)
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def validate_class_names(class_names: List[str]) -> None:
    """Validate that all class directory names are 4-digit numeric strings.

    Args:
        class_names: List of class directory names.

    Raises:
        ValueError: If any class name is not a 4-digit string.
    """
    for name in class_names:
        if len(name) != 4 or not name.isdigit():
            raise ValueError(
                f"Invalid class directory name: {name!r}. "
                f"Expected a 4-digit numeric string."
            )


def generate_class_mapping(
    train_dir: str,
    expected_num_classes: int,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Generate canonical class mapping by scanning the training directory.

    Args:
        train_dir: Path to the training data root (contains class subdirectories).
        expected_num_classes: Expected number of classes (e.g., 500 for preliminary).

    Returns:
        Tuple of (class_to_idx, idx_to_class).
        - class_to_idx: {class_name: int_index}
        - idx_to_class: {str_index: class_name}

    Raises:
        FileNotFoundError: If train_dir does not exist.
        ValueError: If class directory names are invalid or count mismatches.
    """
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    class_names = sorted(
        p.name for p in train_path.iterdir() if p.is_dir()
    )

    if not class_names:
        raise ValueError(f"No class directories found in {train_dir}")

    validate_class_names(class_names)

    if len(class_names) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, "
            f"found {len(class_names)} in {train_dir}"
        )

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {str(i): name for name, i in class_to_idx.items()}

    logger.info(
        f"Generated class mapping: {len(class_to_idx)} classes "
        f"from {train_dir}"
    )

    return class_to_idx, idx_to_class


def save_class_mapping(
    output_dir: str,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[str, str],
) -> None:
    """Save class mapping files to the output directory.

    Args:
        output_dir: Directory to save mapping files.
        class_to_idx: {class_name: int_index}
        idx_to_class: {str_index: class_name}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    c2i_path = output_path / "class_to_idx.json"
    i2c_path = output_path / "idx_to_class.json"

    with open(c2i_path, "w") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    with open(i2c_path, "w") as f:
        json.dump(idx_to_class, f, indent=2, ensure_ascii=False)

    logger.info(f"Class mapping saved to {output_path}")


def load_existing_mapping(
    mapping_path: str,
) -> Optional[Tuple[Dict[str, int], Dict[str, str]]]:
    """Load an existing class mapping from disk.

    Args:
        mapping_path: Path to class_to_idx.json.

    Returns:
        Tuple of (class_to_idx, idx_to_class), or None if file doesn't exist.
    """
    c2i_path = Path(mapping_path)
    if not c2i_path.exists():
        return None

    with open(c2i_path, "r") as f:
        class_to_idx = json.load(f)

    idx_to_class = {str(v): k for k, v in class_to_idx.items()}

    logger.info(f"Loaded existing class mapping from {mapping_path}")
    return class_to_idx, idx_to_class


def compute_mapping_hash(class_to_idx: Dict[str, int]) -> str:
    """Compute SHA256 hash of the class mapping for cache integrity checks.

    Args:
        class_to_idx: Class name to index mapping.

    Returns:
        Hex digest of the mapping serialized as sorted JSON.
    """
    serialized = json.dumps(class_to_idx, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_or_generate_class_mapping(
    train_dir: str,
    expected_num_classes: int,
    mapping_path: Optional[str] = None,
    regenerate: bool = False,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Load existing mapping or generate a new one.

    Args:
        train_dir: Path to the training data root.
        expected_num_classes: Expected number of classes.
        mapping_path: Path to an existing class_to_idx.json file.
                       If None, generates from train_dir.
        regenerate: If True, force regeneration even if mapping exists.

    Returns:
        Tuple of (class_to_idx, idx_to_class).

    Raises:
        ValueError: If existing mapping is inconsistent with current data.
    """
    if mapping_path and not regenerate:
        existing = load_existing_mapping(mapping_path)
        if existing is not None:
            class_to_idx, idx_to_class = existing
            if len(class_to_idx) != expected_num_classes:
                raise ValueError(
                    f"Existing mapping has {len(class_to_idx)} classes, "
                    f"but expected {expected_num_classes}. "
                    f"Use --regenerate-class-mapping to regenerate."
                )
            logger.info("Reusing existing class mapping")
            return class_to_idx, idx_to_class

    # Generate fresh mapping
    class_to_idx, idx_to_class = generate_class_mapping(
        train_dir, expected_num_classes
    )

    if mapping_path:
        save_dir = Path(mapping_path).parent
        save_class_mapping(save_dir, class_to_idx, idx_to_class)

    return class_to_idx, idx_to_class
