"""Check content-group exclusion and paired diagnostic accounting."""
import importlib.util
from pathlib import Path

import numpy as np
import torch
import pytest
from types import SimpleNamespace

path = Path(__file__).parents[1]/'scripts/probe_v1_feature_calibration.py'
spec = importlib.util.spec_from_file_location('feature_calibration_probe', path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_grouped_folds_keep_singleton_anchors_out_of_scoring_and_cover_fit_classes():
    rows = [dict(label='0', content_group='singleton')]
    rows += [dict(label=str(c), content_group=f'{c}:{g}') for c in (1,2) for g in range(5)]
    rows.append(dict(label='1', content_group='1:0'))
    folds, anchors = probe.grouped_two_folds(rows)
    assert anchors == [0] and folds[0] == -1
    for f in (0,1):
        fit = {r['content_group'] for r, flag in zip(rows, folds != f) if flag}
        held = {r['content_group'] for r, flag in zip(rows, folds == f) if flag}
        assert not fit & held
        assert {r['label'] for r, flag in zip(rows, folds != f) if flag} == {'0','1','2'}
    assert np.array_equal(folds, probe.grouped_two_folds(rows)[0])


def test_mixed_label_anchor_content_never_leaks_into_heldout():
    rows = [dict(label='0',content_group='mixed'),dict(label='1',content_group='mixed')]
    rows += [dict(label='1',content_group=f'one:{g}') for g in range(5)]
    folds, _ = probe.grouped_two_folds(rows)
    assert folds[0] == folds[1] == -1


def test_pairing_net_equals_difference_of_full_correct_counts():
    y=np.array([0,1,2,0]); a=np.array([0,0,2,1]); b=np.array([1,1,2,0])
    report=probe.paired(a,b,y)
    assert report == dict(changed=3,corrections=2,regressions=1,net_correct=1)
    m=probe.metric(b,y,3,np.array([0,1]))
    assert m['correct']==3 and m['macro']==5/6


def test_same_dimensionality_or_isometric_projection_preserves_cosine_neighbors():
    torch.manual_seed(42)
    z=torch.randn(12,4); p=torch.linalg.qr(torch.randn(4,4)).Q
    before=torch.nn.functional.normalize(z,dim=1)
    after=torch.nn.functional.normalize(z@p,dim=1)
    assert torch.allclose(before@before.T, after@after.T,atol=1e-6)
    # A rank-reducing projection can change the metric even when weights are frozen.
    reduced=torch.nn.functional.normalize(z@p[:,:2],dim=1)
    assert not torch.allclose(before@before.T,reduced@reduced.T,atol=1e-3)


def test_historical_code_bridge_rejects_changed_dataset_identity(tmp_path):
    import json
    p=tmp_path/'asset.pt'
    p.with_suffix('.sha256.json').write_text(json.dumps(dict(binding={
        'implementation_sha256':'old-code','dataset_manifest_sha256':'different-data'})))
    ctx=SimpleNamespace(binding={'implementation_sha256':'new-code','dataset_manifest_sha256':'current-data'})
    with pytest.raises(ValueError,match='identity changed'):
        probe.historical_artifact(p,ctx)
