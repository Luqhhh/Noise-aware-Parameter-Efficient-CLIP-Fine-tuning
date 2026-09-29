#!/usr/bin/env python3
"""One fixed pixel-only detector. No models, text prototypes, labels or audit IDs in selection."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy import ndimage


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def detect(image, cfg):
    """Positive means geometric text-page proxy, never confirmed open-set truth."""
    metrics = {'selected': False, 'reason': 'small'}
    if min(image.size) < cfg['min_side']: return metrics
    image = image.convert('RGB')
    image.thumbnail((cfg['max_side'], cfg['max_side']), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.int16)
    gray = rgb.mean(2)
    h, w = gray.shape
    white = float((rgb.min(2) >= cfg['white_level']).mean())
    colored = float(((rgb.max(2)-rgb.min(2)) > cfg['chroma_threshold']).mean())
    metrics.update(white_fraction=white, colored_fraction=colored, reason='background')
    if white < cfg['min_white_fraction'] or colored > cfg['max_colored_fraction']: return metrics
    ink = gray < cfg['ink_level']
    ink_pixels = int(ink.sum())
    ink_fraction = float(ink.mean())
    metrics.update(ink_fraction=ink_fraction, reason='ink_fraction')
    if not cfg['min_ink_fraction'] <= ink_fraction <= cfg['max_ink_fraction']: return metrics
    labels, count = ndimage.label(ink, np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel())
    glyph_pixels, glyph_count = 0, 0
    for i, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None: continue
        bh, bw = box[0].stop-box[0].start, box[1].stop-box[1].start
        if sizes[i] >= 2 and 2 <= bh <= max(3, int(h*cfg['glyph_max_height_fraction'])) and 1 <= bw <= max(3, int(w*cfg['glyph_max_width_fraction'])):
            glyph_pixels += int(sizes[i])
            glyph_count += 1
    residual = 1-glyph_pixels/ink_pixels
    metrics.update(glyphs=glyph_count, non_glyph_ink_fraction=residual, reason='components')
    if glyph_count < cfg['min_glyphs'] or residual > cfg['max_non_glyph_ink_fraction']: return metrics
    row_counts = ink.sum(1)
    line_rows = row_counts >= w*cfg['line_occupancy_fraction']
    line_labels, _ = ndimage.label(line_rows)
    bands = [b[0] for b in ndimage.find_objects(line_labels)]
    eligible = [b for b in bands if 2 <= b.stop-b.start <= max(3, int(h*cfg['max_line_height_fraction']))]
    admitted_rows = np.zeros(h, dtype=bool)
    for b in eligible: admitted_rows[b] = True
    centers = np.array([(b.start+b.stop)/2 for b in eligible])
    spacing = np.diff(centers)
    cv = float(spacing.std()/spacing.mean()) if len(spacing) else 999.
    on_lines = float(row_counts[admitted_rows].sum()/ink_pixels)
    metrics.update(lines=len(eligible), ink_on_lines_fraction=on_lines, line_spacing_cv=cv, reason='lines')
    if len(eligible) < cfg['min_lines'] or on_lines < cfg['min_ink_on_lines_fraction'] or cv > cfg['max_line_spacing_cv']: return metrics
    metrics.update(selected=True, reason='dense_text_geometry')
    return metrics


def inspect_one(job):
    row, root, cfg = job
    path = Path(root)/row['image_path']
    # Bind actual bytes to the current-stage manifest; any decode/hash error aborts.
    if sha(path) != row['file_sha256']: raise ValueError(f'Image changed: {path}')
    with Image.open(path) as image: result = detect(image, cfg)
    return dict(split=row['split'], image_path=row['image_path'], label=int(row['label']),
                content_group=row['content_group'], file_sha256=row['file_sha256'], **result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True, help='Parent of train/, not test images')
    p.add_argument('--rule', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.out.exists(): p.error('Refusing to overwrite output')
    cfg = json.loads(args.rule.read_text())
    rows, seen = [], set()
    for split, filename in [('train', 'train_dev.csv'), ('val', 'val_dev.csv')]:
        with (args.stage/filename).open() as f:
            for row in csv.DictReader(f):
                if row['image_path'] in seen or not row['image_path'].startswith('train/') or '..' in Path(row['image_path']).parts:
                    raise ValueError('Invalid or repeated current-stage image path')
                seen.add(row['image_path'])
                rows.append(dict(row, split=split))
    args.out.mkdir(parents=True)
    manifest = dict(rule=cfg, rule_sha256=sha(args.rule), script_sha256=sha(__file__),
                    splits={f:sha(args.stage/f) for f in ('train_dev.csv', 'val_dev.csv', 'class_to_idx.json')},
                    selection_uses_labels=False, selection_uses_predictions=False,
                    human_training_labels=False, rows=len(rows), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (args.out/'protocol.json').write_text(json.dumps(manifest, indent=2)+'\n')
    start = time.monotonic()
    counts = {s: {'rows':0, 'selected':0} for s in ('train','val')}
    with (args.out/'pixels.jsonl').open('x') as stream, ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(inspect_one, ((r,str(args.image_root),cfg) for r in rows), chunksize=64), 1):
            if time.monotonic()-start > cfg['scan_timeout_seconds']: raise TimeoutError('Fixed CPU scan budget exhausted; incomplete result cannot train')
            stream.write(json.dumps(result, separators=(',',':'))+'\n')
            counts[result['split']]['rows'] += 1
            counts[result['split']]['selected'] += int(result['selected'])
            if i % 10000 == 0: print(i, counts, round(time.monotonic()-start,1), flush=True)
    manifest.update(counts=counts, elapsed_seconds=time.monotonic()-start, pixels_sha256=sha(args.out/'pixels.jsonl'), complete=True)
    (args.out/'scan_summary.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(counts), flush=True)

if __name__ == '__main__': main()
