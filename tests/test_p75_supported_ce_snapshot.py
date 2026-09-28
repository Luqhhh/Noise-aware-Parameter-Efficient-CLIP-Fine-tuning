"""CPU toy export checks raw per-view probabilities and strict image loading."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import p75_supported_ce as pipeline
import p75_supported_ce_snapshot as snapshot


def fixture_image(tmp_path):
    root = tmp_path/'train'
    root.mkdir()
    path = root/'a.png'
    pixels = np.tile(np.arange(384, dtype=np.uint16)[None, :, None], (384, 1, 3))
    Image.fromarray((pixels % 256).astype(np.uint8)).save(path)
    row = dict(image_path='train/a.png', label=0, content_group='audited-group',
               file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return root, path, row


def preprocess(image):
    return torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1)/255.


def test_snapshot_dataset_fails_on_changed_image_bytes(tmp_path):
    root, path, row = fixture_image(tmp_path)
    dataset = snapshot.SnapshotDataset([row], root, preprocess)
    tensor, index = dataset[0]
    assert tensor.shape == (3, 384, 384) and index == 0
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        dataset[0]


def test_raw_export_has_separate_views_exact_rows_and_no_fusion_or_prior(tmp_path, monkeypatch):
    root, path, row = fixture_image(tmp_path)
    assets = tmp_path/'assets'
    assets.mkdir()
    pipeline.write_rows(assets/'train_dev.csv', list(row), [row])
    output = tmp_path/'output'
    (output/'configs').mkdir(parents=True)
    original = dict(train=dict(device='cpu', amp=False), data=dict(train_root=str(root)))
    (output/'configs/l05_original.json').write_text(json.dumps(original))
    checkpoint_path = tmp_path/'best.pt'
    checkpoint_path.write_bytes(b'CPU toy model')
    fixed = tmp_path/'fixed.json'
    fixed.write_text(json.dumps(dict(assets=str(assets), control_checkpoint=str(checkpoint_path))))
    identity = dict(num_classes=2, control_sha256=pipeline.sha(checkpoint_path))
    monkeypatch.setattr(snapshot, 'OUT', output)
    monkeypatch.setattr(snapshot, 'FIXED', fixed)
    monkeypatch.setattr(snapshot, 'preflight', lambda: dict(identity=identity))
    class Toy(torch.nn.Module):
        def forward(self, *, images):
            pixel = images[:, 0, 0, 0]
            return torch.stack((pixel, -pixel), 1)
    checkpoint_module = ModuleType('aegis_clip.checkpoint')
    checkpoint_module.build_from_checkpoint = lambda *args, **kwargs: (Toy(), preprocess, {'epoch': 16})
    protocol = ModuleType('aegis_clip.rematch_protocol')
    protocol.validate_checkpoint = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'aegis_clip', ModuleType('aegis_clip'))
    monkeypatch.setitem(sys.modules, 'aegis_clip.checkpoint', checkpoint_module)
    monkeypatch.setitem(sys.modules, 'aegis_clip.rematch_protocol', protocol)
    manifest = snapshot.export_snapshot(batch_size=1, num_workers=0)
    first = np.load(output/'snapshot/center_probabilities.npy')
    second = np.load(output/'snapshot/flip_probabilities.npy')
    assert first.shape == second.shape == (1, 2)
    assert np.allclose(first, [[.5, .5]])
    expected = torch.tensor([[127/255, -127/255]]).softmax(1).numpy()
    assert np.allclose(second, expected)
    assert not np.allclose(first, second)
    assert manifest['temperature'] == 1.
    assert not manifest['prior_applied'] and not manifest['tta_fusion_applied']
    assert not manifest['validation_images_read'] and not manifest['test_images_read']
    pipeline.verify_files(output/'snapshot', manifest['files'])
    pipeline.assert_alignment([row], pipeline.read_rows(output/'snapshot/rows.csv'))
    with pytest.raises(FileExistsError):
        snapshot.export_snapshot(batch_size=1, num_workers=0)
