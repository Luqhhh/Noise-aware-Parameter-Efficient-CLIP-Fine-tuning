"""Paired overlap diagnostic for a preregistered fixed shared-head transform."""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from aegis_clip.balanced_inference import prediction_metrics
from aegis_clip.checkpoint import build_from_checkpoint, _atomic_torch_save
from aegis_clip.classifier_norm import mean_norm_scale, align_branch_logits
from aegis_clip.cli.cache_scope_parent import ValidationImages
from aegis_clip.data import TrustBundle, load_class_mapping
from aegis_clip.local_feature_adapter import load_local_feature_adapter
from aegis_clip.part_token_adapter import load_part_token_adapter
from aegis_clip.local_inference import adapted_dual_local_view_logits
from aegis_clip.localization import forward_with_last_block_attention, extract_attention_crops, fuse_global_multilocal_flip_probabilities
from aegis_clip.runtime import atomic_json_dump, set_seed, sha256_file, seed_worker


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--decision', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    decision = json.loads(Path(args.decision).read_text())
    if decision.get('approved') is not True or decision.get('selected_mechanism') != 'shared_classifier_mean_norm_alignment':
        raise ValueError('Registered norm-alignment decision required')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json_dump({'status':'running', 'decision_sha256':sha256_file(args.decision)}, output/'status.json')
    start = time.monotonic(); set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda': raise RuntimeError('Registered GPU budget requires CUDA')
    model, preprocess, checkpoint = build_from_checkpoint(args.checkpoint, device)
    if sha256_file(args.checkpoint) != 'f72b0104257f49d2667fe335553a861dd1dea947753feebdc7301b8890b48765':
        raise ValueError('Registered parent hash mismatch')
    model.eval(); model.requires_grad_(False)
    if not isinstance(model.classifier, torch.nn.Linear): raise ValueError('Shared linear classifier required')
    config = checkpoint['config']; data = config['data']
    mapping, _ = load_class_mapping(data['class_mapping'])
    if len(mapping) != 500: raise ValueError('Registered preliminary mapping required')
    o3 = load_local_feature_adapter(checkpoint, device).eval()
    pta = load_part_token_adapter(checkpoint, device).eval()
    o3.requires_grad_(False); pta.requires_grad_(False)
    pool = checkpoint['part_token_adapter']['spec']['part_pool_spec']
    trust = TrustBundle(config['trust']['bundle_path'])
    dataset = ValidationImages(Path(data['val_csv']), Path(data['train_root']), preprocess, trust)
    if len(dataset) != 10316: raise ValueError('Registered diagnostic split length mismatch')
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, timeout=120, pin_memory=True, worker_init_fn=seed_worker)
    weight, bias = model.classifier.weight, model.classifier.bias
    scale = mean_norm_scale(weight)
    # Test the affine shortcut against actual transformed-head arithmetic on
    # actual first-batch features, not only synthetic unit fixtures.
    numeric = {'max_abs_difference':0., 'argmax_disagreements':0, 'calls':0}
    def check_head(module, inputs, values):
        direct = F.linear(inputs[0], weight * scale[:,None], bias)
        derived = align_branch_logits(values, bias, scale)
        torch.testing.assert_close(direct, derived, rtol=1e-5, atol=1e-5)
        numeric['max_abs_difference'] = max(numeric['max_abs_difference'], float((direct-derived).abs().max()))
        numeric['argmax_disagreements'] += int((direct.argmax(1)!=derived.argmax(1)).sum())
        numeric['calls'] += 1
    hook = model.classifier.register_forward_hook(check_head)
    fields = {k:[] for k in ('label','clean_probability','pseudo_label','correction_alpha')}
    parent_preds, candidate_preds, paths = [], [], []
    spec = decision['fixed_hyperparameters_and_origin']
    for index, batch in enumerate(tqdm(loader, desc='norm alignment paired overlap diagnostic')):
        if time.monotonic()-start > 7200: raise RuntimeError('Registered two-hour budget exceeded')
        if batch['corrupt'].any():
            atomic_json_dump({'status':'failed','reason':'decode failure','paths':[p for p,b in zip(batch['path'],batch['corrupt']) if b]}, output/'status.json')
            raise RuntimeError('Decode failure; no zero substitute accepted')
        images = batch['images'].to(device, non_blocking=True)
        branches = []
        for view in (images, images.flip(3)):
            global_logits, attention = forward_with_last_block_attention(model, view)
            locals_ = []
            for crop in spec['crops']:
                local, _ = extract_attention_crops(view, attention, crop_size=crop, top_k=spec['top_k'])
                locals_.append(adapted_dual_local_view_logits(model,o3,pta,local,part_top_patches=int(pool['top_patches']),part_temperature=float(pool['temperature'])))
            branches.append((global_logits, locals_))
        if index == 0:
            hook.remove()
            if numeric['argmax_disagreements']: raise RuntimeError('Direct-head numerical argmax disagreement')
            atomic_json_dump(numeric, output/'numerical_check.json')
        def fused(transform):
            go,lo=branches[0];gf,lf=branches[1]
            return fuse_global_multilocal_flip_probabilities(transform(go),[transform(v) for v in lo],transform(gf),[transform(v) for v in lf],local_weight=spec['local_weight'],flip_weight=spec['flip_weight'],temperature=spec['local_temperature'],global_temperature=spec['global_temperature'],local_temperature=spec['local_temperature'],local_scale_weights=spec['scale_weights'])
        parent=fused(lambda x:x); candidate=fused(lambda x:align_branch_logits(x,bias,scale))
        if not torch.isfinite(parent).all() or not torch.isfinite(candidate).all(): raise RuntimeError('Nonfinite fused scores')
        parent_preds.append(parent.argmax(1).cpu()); candidate_preds.append(candidate.argmax(1).cpu());paths.extend(batch['path'])
        for name in fields: fields[name].append(batch[name].cpu())
        if index % 10 == 0: atomic_json_dump({'status':'running','completed_samples':len(paths),'elapsed_seconds':time.monotonic()-start},output/'status.json')
    fields={k:torch.cat(v) for k,v in fields.items()}; parent=torch.cat(parent_preds);candidate=torch.cat(candidate_preds)
    def metrics(pred): return prediction_metrics(pred, labels=fields['label'],clean_probability=fields['clean_probability'],pseudo_labels=fields['pseudo_label'],correction_alpha=fields['correction_alpha'],num_classes=500,clean_core_threshold=.7)
    clean=fields['clean_probability']>=.7
    def correct(pred,mask):return int((pred[mask]==fields['label'][mask]).sum())
    all_=torch.ones_like(clean)
    dr=correct(candidate,all_)-correct(parent,all_);dc=correct(candidate,clean)-correct(parent,clean)
    passed=dr>0 and dc>=0
    result={'status':'checks_passed' if passed else 'rejected','validation_scope':'overlap_diagnostic','independent_generalization_claim':False,'parent':metrics(parent),'candidate':metrics(candidate),'raw_correct_delta':dr,'clean_core_correct_delta':dc,'changed_predictions':int((parent!=candidate).sum()),'gate_passed':passed,'calibration':'none for both arms','elapsed_seconds':time.monotonic()-start,'max_cuda_memory_allocated':torch.cuda.max_memory_allocated(),'checkpoint_sha256':sha256_file(args.checkpoint),'class_mapping_sha256':sha256_file(data['class_mapping']),'validation_csv_sha256':sha256_file(data['val_csv']),'decision_sha256':sha256_file(args.decision),'numerical_check':numeric}
    _atomic_torch_save({'paths':paths,**fields,'parent_predictions':parent,'candidate_predictions':candidate},output/'paired_predictions.pt')
    if passed:
        artifact=copy.deepcopy(checkpoint)
        artifact['model_state_dict']['classifier.weight']=(weight*scale[:,None]).cpu()
        for key in ('optimizer_state_dict','scheduler_state_dict','scaler_state_dict','rng_state','data_generator_state','metrics','best_selector'):artifact.pop(key,None)
        artifact['classifier_norm_alignment']={'parent_sha256':result['checkpoint_sha256'],'decision_sha256':result['decision_sha256'],'inference_only':True,'scale':scale.cpu()}
        _atomic_torch_save(artifact,output/'candidate.pt')
    atomic_json_dump(result,output/'result.json');atomic_json_dump(result,output/'status.json')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
