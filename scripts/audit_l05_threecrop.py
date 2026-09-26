#!/usr/bin/env python3
"""Independently rebuild the fixed six-view fusion with NumPy float64."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_l05_nonlinear_head import preflight, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rows, ref, identity = preflight(cfg)
    out = Path(cfg['output'])
    result = json.loads((out/'result.json').read_text())
    saved = torch.load(out/'threecrop_validation.pt', map_location='cpu', weights_only=False)
    assert saved['identity'] == identity
    rebuilt = []
    seen = 0
    for name, digest in sorted(result['branch_cache_sha256'].items()):
        path = out/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        part = torch.load(path, map_location='cpu', weights_only=False)
        offset = part['identity']['offset']; n = len(part['branches'][0])
        assert offset == seen
        assert part['identity'] == {**identity, 'offset': offset,
                                    'paths': [r['image_path'] for r in rows['val'][offset:offset+n]]}
        six = [ref['original_logits'][offset:offset+n], ref['flip_logits'][offset:offset+n], *part['branches']]
        assert len(six) == 6
        probabilities = []
        for logits in six:
            z = logits.numpy().astype(np.float64)/cfg['temperature']
            z -= z.max(axis=1, keepdims=True)
            q = np.exp(z); q /= q.sum(axis=1, keepdims=True)
            probabilities.append(q)
        rebuilt.append(np.log(np.maximum(np.mean(probabilities, axis=0), 1e-12)))
        seen += n
    assert seen == len(rows['val'])
    x = np.concatenate(rebuilt)
    bias = saved['bias'].numpy().astype(np.float64)
    labels = saved['labels'].numpy()
    predicted = (x+cfg['prior_strength']*bias).argmax(1)
    expected = (saved['fused_logits']+cfg['prior_strength']*saved['bias']).argmax(1).numpy()
    error = float(np.max(np.abs(x-saved['fused_logits'].numpy())))
    counts = np.bincount(labels, minlength=x.shape[1])
    correct = np.bincount(labels[predicted == labels], minlength=x.shape[1])
    assert np.array_equal(predicted, expected) and error < 2e-5
    assert abs(float(np.mean(correct/counts))-result['metrics']['threecrop']['decode']['macro']) < 1e-12
    assert abs(float(np.mean(predicted == labels))-result['metrics']['threecrop']['decode']['micro']) < 1e-12
    parent_sha = hashlib.sha256(Path(cfg['parent_checkpoint']).read_bytes()).hexdigest()
    assert parent_sha == cfg['parent_sha256']
    report = {'independent_numpy_float64_fusion_max_abs': error,
              'prediction_agreement_count': int(np.sum(predicted == expected)), 'count': len(labels),
              'parent_sha256_after': parent_sha, 'all_15_chunk_hashes_and_paths_verified': True}
    write_json(out/'independent_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
