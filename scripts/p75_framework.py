#!/usr/bin/env python3
"""Materialize private, pinned focus runtimes for P75 candidates."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from p75_sam_overlay import apply as apply_sam

ROOT = Path(__file__).resolve().parents[1]
FOCUS = Path('/home/lux1/noise/worktrees/rematch750_f05_focus')
OUT = ROOT/'outputs/codex/p75_mechanisms_20260927'
COMMIT = 'f050ecb59e0a885da0b43cdc6c8c316953fb5e7d'
ARCHIVE_PATHS = (
    'reproducibility/aegis_f1',
    'scripts/cache_validation_tta_logits.py',
    'scripts/evaluate_l05_candidate.py',
    'scripts/build_l05_tta_prior_submission_final.py',
    'scripts/check_submission.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize(candidate: str) -> Path:
    if candidate not in ('patch', 'sam'):
        raise ValueError('Unknown P75 runtime')
    destination = OUT/f'framework_{candidate}'
    manifest = destination/'p75_source_manifest.json'
    if destination.exists():
        if not manifest.exists():
            raise FileExistsError('Partial private framework exists')
        recorded = json.loads(manifest.read_text())
        if recorded['commit'] != COMMIT or recorded['candidate'] != candidate:
            raise RuntimeError('Private framework identity changed')
        for name, expected in recorded['files'].items():
            if digest(destination/name) != expected:
                raise RuntimeError(f'Private framework file changed: {name}')
        return destination
    destination.mkdir(parents=True)
    archive = subprocess.check_output(['git', 'archive', COMMIT, *ARCHIVE_PATHS], cwd=FOCUS)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        members = stream.getmembers()
        if any(Path(member.name).is_absolute() or '..' in Path(member.name).parts
               for member in members):
            raise ValueError('Unsafe source archive member')
        stream.extractall(destination)
    if candidate == 'sam':
        apply_sam(destination/'reproducibility/aegis_f1/aegis_clip/trainer.py')
    files = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(destination))] = digest(path)
    manifest.write_text(json.dumps({'commit': COMMIT, 'candidate': candidate,
                                    'files': files}, indent=2, sort_keys=True)+'\n')
    return destination


if __name__ == '__main__':
    for name in ('patch', 'sam'):
        print(materialize(name))
