"""Build content-hash groups so duplicates never cross OOF folds."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from aegis_clip.config import load_config
from aegis_clip.data import resolve_image_path
from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    frames = [pd.read_csv(data["train_csv"]), pd.read_csv(data["val_csv"])]
    frame = pd.concat(frames, ignore_index=True)
    canonical = [canonical_sample_path(path) for path in frame["image_path"].astype(str)]
    if len(canonical) != len(set(canonical)):
        raise ValueError("Train and validation splits overlap or duplicate paths")
    root = Path(data["train_root"])
    absolute = [resolve_image_path(root, path) for path in frame["image_path"].astype(str)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        hashes = list(executor.map(sha256_file, absolute))
    output = Path(
        args.output
        or config.get("trust", {}).get("groups_path")
        or "artifacts/trust/content_groups.json"
    )
    atomic_json_dump(dict(zip(canonical, hashes)), output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "samples": len(canonical),
                "unique_groups": len(set(hashes)),
                "duplicate_samples": len(hashes) - len(set(hashes)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

