"""
Dataset classes for the CLIP baseline project.

Provides:
- TrainImageDataset: loads training images with labels, returns (image, label, image_path)
- TestImageDataset: loads test images without labels, returns (image, image_name, image_path)

Both classes handle corrupted images gracefully and support the sample_weight
interface for future noise-robust training extensions.
"""

import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Allow loading of truncated images to avoid crashes on corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_images_in_dir(
    directory: Path, extensions: Optional[set] = None
) -> List[Path]:
    """Find all image files in a directory with given extensions.

    Args:
        directory: Directory to scan.
        extensions: Set of allowed extensions (lowercase). Defaults to IMAGE_EXTENSIONS.

    Returns:
        Sorted list of Path objects for all matching images.
    """
    if extensions is None:
        extensions = IMAGE_EXTENSIONS

    images = []
    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(images)


def _safe_load_image(
    image_path: Path, transform: Optional[Callable] = None
) -> torch.Tensor:
    """Safely load and transform an image.

    On failure, returns a blank (all-zeros) RGB image of default size or prints
    a warning and re-raises for the caller to decide.

    Args:
        image_path: Path to the image file.
        transform: Optional torchvision transform to apply.

    Returns:
        Transformed image tensor, or None if the image could not be loaded.
    """
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None

    if transform is not None:
        try:
            image = transform(image)
        except Exception as e:
            logger.warning(f"Failed to transform image {image_path}: {e}")
            return None

    return image


class TrainImageDataset(Dataset):
    """Dataset for training/validation images organized in class-folders.

    Each class folder contains images belonging to that class. The class index
    is derived from the sorted order of class folder names.

    Args:
        data_root: Root directory containing class subdirectories.
        split_csv: Optional path to a CSV with columns [image_path, label, class_name].
                   If provided, only images listed in this CSV are used (for train/val splits).
        class_to_idx: Optional dict mapping class folder names to integer indices.
                      If None, built by sorting class folder names.
        transform: torchvision transform to apply to images.
        return_path: If True, __getitem__ also returns the image path (for sample_weight
                     interface and noise-robust extensions).
    """

    def __init__(
        self,
        data_root: str,
        split_csv: Optional[str] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
        return_path: bool = True,
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.return_path = return_path

        if split_csv is not None:
            # Load image list from split CSV
            self._load_from_csv(split_csv, class_to_idx)
        else:
            # Build from directory structure
            self._load_from_directory(class_to_idx)

        logger.info(f"TrainImageDataset: {len(self.samples)} samples, "
                     f"{len(self.class_to_idx)} classes")

    def _load_from_csv(
        self, split_csv: str, class_to_idx: Optional[Dict[str, int]] = None
    ) -> None:
        """Load dataset from a split CSV file.

        CSV format: image_path,label,class_name
        """
        import pandas as pd

        df = pd.read_csv(split_csv)
        self.samples = []
        self.labels = []

        for _, row in df.iterrows():
            img_path = Path(row["image_path"])
            # image_path in CSV may be relative; resolve relative to CWD or absolute
            if not img_path.is_absolute():
                img_path = Path.cwd() / img_path
            self.samples.append(img_path)
            self.labels.append(int(row["label"]))

        # Build or validate class_to_idx
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            unique_classes = sorted(set(row["class_name"] for _, row in df.iterrows()))
            self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def _load_from_directory(
        self, class_to_idx: Optional[Dict[str, int]] = None
    ) -> None:
        """Build dataset by scanning class subdirectories."""
        class_dirs = sorted(
            [d for d in self.data_root.iterdir() if d.is_dir()],
            key=lambda x: x.name,
        )

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {
                d.name: i for i, d in enumerate(class_dirs)
            }

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.samples = []
        self.labels = []

        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                logger.warning(f"Skipping unknown class directory: {class_name}")
                continue

            label = self.class_to_idx[class_name]
            images = _find_images_in_dir(class_dir)

            for img_path in images:
                self.samples.append(img_path)
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label, image_path_str).
            If loading fails, returns a blank image (all-zeros tensor) with the
            original label and path, so training can continue.
        """
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = _safe_load_image(img_path, self.transform)

        if image is None:
            # Return a blank image so training doesn't crash
            logger.warning(
                f"Returning blank image for {img_path} (index {idx})"
            )
            # Create a properly-sized blank tensor
            # CLIP transforms produce 3x224x224 tensors
            image = torch.zeros(3, 224, 224)

        if self.return_path:
            return image, label, str(img_path)
        else:
            return image, label

    def get_sample_weights(self, weight_dict: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Placeholder for sample-level weighting (for future noise-robust methods).

        Args:
            weight_dict: Optional dict mapping image_path to weight.

        Returns:
            Tensor of ones (uniform weighting) if weight_dict is None,
            otherwise weights from the dict.
        """
        if weight_dict is None:
            return torch.ones(len(self.samples), dtype=torch.float32)

        weights = []
        for img_path in self.samples:
            weights.append(weight_dict.get(str(img_path), 1.0))
        return torch.tensor(weights, dtype=torch.float32)


class TestImageDataset(Dataset):
    """Dataset for test images without labels.

    Scans a flat directory of test images and returns each image with its
    filename (for submission generation).

    Args:
        data_root: Directory containing test images.
        transform: torchvision transform to apply.
        extensions: Set of allowed file extensions.
    """

    def __init__(
        self,
        data_root: str,
        transform: Optional[Callable] = None,
        extensions: Optional[set] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform

        if extensions is None:
            extensions = IMAGE_EXTENSIONS

        self.images = _find_images_in_dir(self.data_root, extensions)

        if len(self.images) == 0:
            raise RuntimeError(
                f"No test images found in {self.data_root} "
                f"with extensions {extensions}"
            )

        logger.info(f"TestImageDataset: {len(self.images)} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a test sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, image_name, image_path_str).
        """
        img_path = self.images[idx]
        image_name = img_path.name  # e.g., "00012f3f8db94b50bdd15611c1bb699b.jpg"

        image = _safe_load_image(img_path, self.transform)

        if image is None:
            logger.warning(
                f"Returning blank image for {img_path} (index {idx})"
            )
            image = torch.zeros(3, 224, 224)

        return image, image_name, str(img_path)
