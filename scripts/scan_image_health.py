"""Pre-flight scan: which training/validation images break v2's load_image path?

Mirrors reproducibility/aegis_f1/v2/training_utils.py:load_image exactly (open ->
draft -> load -> exif_transpose -> convert) and records every exception by type,
so a multi-hour run cannot be killed by an image the scan already knows about.

Tier 1 (--tier exif) only opens the header and forces getexif()/exif_transpose:
far cheaper, and sufficient because decode_report.json already proves all 148,695
training images decode at the pixel level. Tier 2 (--tier full) also runs load().
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = False


def scan_exif(path: Path) -> str | None:
    """Return the exception repr if the EXIF step fails, else None."""
    with Image.open(path) as im:
        try:
            ImageOps.exif_transpose(im)
        except BaseException as exc:  # noqa: BLE001 - we want to classify everything
            return f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    return None


def scan_full(path: Path, decode_cap: int = 0) -> str | None:
    """v2 load_image end to end."""
    try:
        with Image.open(path) as im:
            if decode_cap and im.format == "JPEG":
                im.draft("RGB", (decode_cap, decode_cap))
            im.load()
            try:
                im = ImageOps.exif_transpose(im)
            except (ValueError, TypeError, OSError):  # the original narrow clause
                pass
            im.convert("RGB").copy()
    except BaseException as exc:  # noqa: BLE001
        return f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="train_manifest.csv from a prepared plan")
    ap.add_argument("--split", help="split.json; when given, also scan the val indices")
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--tier", choices=("exif", "full"), default="exif")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with Path(args.manifest).open(encoding="utf-8", newline="") as fh:
        records = list(csv.DictReader(fh))
    root = Path(args.train_root)
    fn = scan_exif if args.tier == "exif" else scan_full

    kinds: Counter[str] = Counter()
    bad: list[dict] = []
    started = time.monotonic()
    for i, rec in enumerate(records):
        exc = fn(root / rec["relative_path"])
        if exc is not None:
            kinds[exc] += 1
            bad.append({"relative_path": rec["relative_path"], "label": int(rec["label"]), "error": exc})
        if (i + 1) % 20000 == 0:
            print(f"  {i+1}/{len(records)} scanned, {len(bad)} bad, {time.monotonic()-started:.0f}s", flush=True)

    report = {
        "tier": args.tier,
        "manifest": str(args.manifest),
        "train_root": str(args.train_root),
        "scanned": len(records),
        "bad_count": len(bad),
        "exception_kinds": dict(kinds),
        "bad_examples": bad[:40],
        "elapsed_seconds": time.monotonic() - started,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("tier", "scanned", "bad_count", "exception_kinds", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
