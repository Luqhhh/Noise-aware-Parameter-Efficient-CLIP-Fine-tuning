#!/usr/bin/env python3
"""CPU official-model forward and optional synthetic backward; no optimizer or GPU."""
from pathlib import Path
import argparse
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproducibility/aegis_f1'))
from v2.preprojection import build_classifier
from v2.architecture import architecture_spec
from v2.plan import json_read, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',default=str(ROOT/'configs/v2/recipe.json'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--backward',action='store_true',help='Synthetic CPU backward only; zero optimizer updates')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(13)
    assert not torch.cuda.is_initialized()
    r=json_read(args.recipe)
    model=build_classifier(r).eval()
    with torch.no_grad():
        x=torch.randn(1,3,224,224)
        native=model.visual(x);adapted=model.embed(x)
        torch.testing.assert_close(native,adapted,rtol=1e-5,atol=1e-5)
        shapes={}
        for size in (384,448,576):
            larger=model(torch.randn(1,3,size,size))
            assert larger.shape==(1,r['num_classes']) and torch.isfinite(larger).all()
            shapes[str(size)]=list(larger.shape)
    gradient_report = None
    if args.backward:
        model.train()
        loss=torch.nn.functional.cross_entropy(model(torch.randn(1,3,384,384)),torch.tensor([0]))
        loss.backward()
        checked=0
        for name,parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
            checked+=1
        gradient_report=dict(synthetic_input=True,optimizer_steps=0,finite_parameter_gradients=checked,
                             loss=float(loss.detach()),gradient_checkpointing=model.gradient_checkpointing)
        model.zero_grad(set_to_none=True)
    assert not torch.cuda.is_initialized()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='cpu_forward_passed',gpu_started=False,training_started=False,
                architecture=architecture_spec(r),synthetic_backward=gradient_report,
                native224_max_absolute_difference=float((native-adapted).abs().max()),
                resolution_logits_shapes=shapes,all_parameters_fp32=all(p.dtype==torch.float32 for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                official_sha256=sha(r['official_checkpoint']),local_metrics=None,platform_metrics=None)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
