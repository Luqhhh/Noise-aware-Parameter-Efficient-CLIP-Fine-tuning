from pathlib import Path
import sys
import subprocess
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from p75_text_page_runtime import mask_classification
from p75_text_page_overlay import transform

@pytest.mark.parametrize('phase',['CE','GCE'])
@pytest.mark.parametrize('local_active',[False,True])
def test_global_local_fallback_mask_only_classification(phase,local_active):
    torch.manual_seed(3)
    g=torch.randn(4,3,requires_grad=True); l=torch.randn(4,3,requires_grad=True)
    anchor=torch.randn(4,requires_grad=True)
    y=torch.tensor([0,1,2,0]); selected=torch.tensor([True,False,True,False])
    def objective(enabled):
        def raw(x):
            py=x.softmax(1)[torch.arange(4),y]
            return -py.log() if phase=='CE' else 2*(1-py.sqrt())
        global_loss=mask_classification(raw(g),selected,enabled)
        loss=global_loss.mean()
        if local_active:
            local=mask_classification(raw(l),selected,enabled)
            gate=torch.tensor([True,True,False,False])
            loss=.75*loss+.25*torch.where(gate,local,global_loss).mean()
        return loss+2*anchor.square().mean()
    cg,ca=torch.autograd.grad(objective(False),(g,anchor),retain_graph=True)
    mg,ma=torch.autograd.grad(objective(True),(g,anchor),retain_graph=True)
    assert torch.equal(ca,ma)
    assert torch.equal(cg[~selected],mg[~selected])
    assert torch.count_nonzero(mg[selected])==0
    if local_active:
        lg=torch.autograd.grad(objective(True),l)[0]
        assert torch.count_nonzero(lg[selected])==0

def test_control_is_identity_and_invalid_shape_fails():
    x=torch.tensor([2.,3.],requires_grad=True); s=torch.tensor([True,False])
    assert mask_classification(x,s,False) is x
    with pytest.raises(ValueError): mask_classification(x,s[:,None],True)

def test_pinned_overlay_has_no_other_changes():
    source=subprocess.check_output(['git','show','f050ecb59e0a885da0b43cdc6c8c316953fb5e7d:reproducibility/aegis_f1/aegis_clip/trainer.py'],cwd=ROOT,text=True)
    updated=transform(source)
    assert updated.count('text_page_state.local_loss(')==1
    anchor=source[source.index('                distill_weight = _feature_anchor_weight('):source.index('                sam_update = None')]
    assert anchor in updated
    assert 'per_sample * effective_weights\n                ).sum() / effective_weights.sum().clamp_min(1.0e-8)' in updated
    with pytest.raises(RuntimeError): transform(updated)
