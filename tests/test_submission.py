"""Test submission format correctness."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_submission_format():
    """Each line of pred_results.csv must match 'image_name.jpg, 0001' format."""
    submission_path = Path("outputs/baseline/submissions/pred_results.csv")
    if not submission_path.exists():
        return

    with open(submission_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    pattern = re.compile(r"^.+\.(jpg|jpeg|png|bmp|webp), \d{4}$")
    for i, line in enumerate(lines, start=1):
        assert pattern.match(line), f"Line {i} does not match expected format: '{line}'"
