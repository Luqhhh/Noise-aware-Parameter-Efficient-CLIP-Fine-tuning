import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import CenterCrop
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_l05_threecrop import endpoint_boxes, fuse_views
from aegis_clip.tta import fuse_paired_logits

@pytest.mark.parametrize('shape',[(384,384),(513,384),(384,513),(1025,384)])
def test_endpoints_preserve_extent_and_center_can_match_native(shape):
    w,h=shape
    image=Image.fromarray(np.arange(h*w*3,dtype=np.uint8).reshape(h,w,3))
    a,b=endpoint_boxes(w,h,384)
    assert a[:2]==(0,0) and b[2:]==(w,h)
    for box in (a,b):
        assert image.crop(box).size==(384,384)
    left,top=round((w-384)/2),round((h-384)/2)
    assert np.array_equal(np.array(CenterCrop(384)(image)),np.array(image.crop((left,top,left+384,top+384))))
    if w==h:
        assert a==b


def test_square_views_reduce_to_existing_flip_protocol():
    g=torch.Generator().manual_seed(42)
    x,y=[torch.randn(7,11,generator=g) for _ in range(2)]
    actual=fuse_views([x,y,x,y,x,y],1.4)
    expected=fuse_paired_logits(x,y,mode='mean_probabilities',temperature=1.4)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(actual.exp().sum(1),torch.ones(7))


def test_fusion_rejects_invalid_views():
    x=torch.ones(3,4)
    with pytest.raises(ValueError):
        fuse_views([x]*5,1.4)
    with pytest.raises(ValueError):
        fuse_views([x]*5+[torch.full_like(x,float('nan'))],1.4)
    with pytest.raises(ValueError):
        endpoint_boxes(385,500,384)
