"""Fixed full-image perceptual fingerprint; decode failures never pass."""
from __future__ import annotations

import math
import time
import numpy as np
from PIL import Image
from scipy.fft import dctn

from .io import image_file
from v2.plan import require, sha
from v2.training_utils import load_image


def fingerprint(image):
    rgb = image.convert('RGB')
    gray = np.asarray(rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = dctn(gray, norm='ortho')[:8, :8].reshape(-1)
    bits = low > np.median(low[1:])
    bits[0] = False
    phash = sum(int(b) << i for i, b in enumerate(bits))
    thumb = np.asarray(rgb.resize((16, 16), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    return phash, thumb, rgb.width / rgb.height, float(gray.std() / 255)


def confirm(a, b, policy):
    hamming = (a[0] ^ b[0]).bit_count()
    rmse = float(np.sqrt(np.mean((a[1] - b[1]) ** 2)))
    aspect = abs(math.log(a[2] / b[2]))
    passed = hamming <= policy['max_phash_hamming'] and rmse <= policy['max_thumbnail_rmse']
    passed &= aspect <= policy['max_log_aspect_ratio'] and min(a[3], b[3]) >= policy['min_gray_std']
    return dict(phash_hamming=hamming, thumbnail_rmse=rmse, log_aspect_ratio=aspect,
                perceptual_pass=bool(passed), perceptual_error='')


def verify(rows, pairs, root, policy, progress=None, max_seconds=1800):
    started = time.monotonic()
    cache, errors = {}, {}
    # Decode only >=.94 non-exact candidate endpoints, including same-label controls.
    involved = sorted({int(p[k]) for p in pairs if not p['exact'] for k in ('i', 'j')})
    for number, i in enumerate(involved):
        require(time.monotonic() - started < max_seconds, 'A0 perceptual budget exhausted')
        path = image_file(root, rows[i])
        require(sha(path) == rows[i]['file_sha256'], f'Changed train pixels: {path}')
        try:
            cache[i] = fingerprint(load_image(path))
        except (OSError, ValueError, SyntaxError) as exc:
            errors[i] = type(exc).__name__
        if progress is not None and number % 1000 == 0:
            progress(number, len(involved))
    for pair in pairs:
        i, j = int(pair['i']), int(pair['j'])
        if pair['exact']:
            pair.update(phash_hamming='', thumbnail_rmse='', log_aspect_ratio='',
                        perceptual_pass=False, perceptual_error='exact_separate_accounting')
        elif i in errors or j in errors:
            pair.update(phash_hamming='', thumbnail_rmse='', log_aspect_ratio='',
                        perceptual_pass=False, perceptual_error=errors.get(i, '') + '/' + errors.get(j, ''))
        else:
            pair.update(confirm(cache[i], cache[j], policy))
    return pairs, dict(decoded_images=len(cache), failed_images=errors)
