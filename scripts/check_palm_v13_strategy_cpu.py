#!/usr/bin/env python3
"""CPU-only official-model forward smoke; no training or GPU operations."""
from pathlib import Path
import argparse
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproducibility/aegis_f1'))
from palm_v13.model import LocalFTClassifier
from palm_v13.plan import json_read, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',default=str(ROOT/'configs/palm_v13_20260929/recipe.json'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(13)
    assert not torch.cuda.is_initialized()
    r=json_read(args.recipe)
    model=LocalFTClassifier(r).eval()
    with torch.no_grad():
        x=torch.randn(1,3,224,224)
        native=model.visual(x);adapted=model.embed(x)
        torch.testing.assert_close(native,adapted,rtol=1e-5,atol=1e-5)
        shapes={}
        for size in (384,448,576):
            larger=model(torch.randn(1,3,size,size))
            assert larger.shape==(1,750) and torch.isfinite(larger).all()
            shapes[str(size)]=list(larger.shape)
    assert not torch.cuda.is_initialized()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='cpu_forward_passed',gpu_started=False,training_started=False,
                native224_max_absolute_difference=float((native-adapted).abs().max()),
                resolution_logits_shapes=shapes,all_parameters_fp32=all(p.dtype==torch.float32 for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                official_sha256=sha(r['official_checkpoint']),local_metrics=None,platform_metrics=None)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
