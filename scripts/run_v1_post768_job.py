"""Two fixed post-768 jobs: a 512 decode control and a bounded frozen-head probe."""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import (
    StageContext, fingerprint, image_transform, load_artifact, load_recipe,
    loader_for, read_rows, save_artifact, seed_training, collect_logits,
)
from aegis_clip.v1_strategy import (
    CosineHead, build_classifier, load_trainable_state, target_probabilities,
)
from aegis_clip.v1_test_bias import infer_test_uniform
from probe_v1_feature_calibration import historical_artifact, metric, paired
from verify_v1_768_full_delivery import numpy_uniform_bias

ROOT = Path(__file__).resolve().parents[1]
DECODER = dict(scales=[448,512,576],flip=True,bias_source='test_uniform_experimental',
    experimental_test_bias=True,logit_reduction='sum',bias_iterations=200,bias_strength=1.)
HEAD_FIT = dict(epochs=20,batch_size=8192,lr=.001,minimum_lr=.0001,weight_decay=.01,
    dropout=.1,label_smoothing=.1,seed=42)


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != {'experiment_id','route','parent_config','parent_checkpoint','targets','output','desktop_prefix','head_fit'}:
        raise ValueError('Unexpected post-768 job fields')
    if cfg['route'] not in ('512_control','crt768_dev') or cfg['head_fit'] != HEAD_FIT:
        raise ValueError('Only the fixed control or bounded 20-epoch head pair is supported')
    return cfg


def inspect(cfg):
    ctx = StageContext(load_recipe(cfg['parent_config']))
    parent = historical_artifact(Path(cfg['parent_checkpoint']),ctx)
    population = 'full_train' if cfg['route']=='512_control' else 'train_dev'
    if (ctx.config['source']['partition'] != population or parent['classes'] != ctx.classes
            or parent['model_config'] != ctx.config['model'] or parent['selected_policy'] != 'swa_ema'
            or parent['swa_epochs'] != list(range(4,13))
            or parent['model_config'].get('feature_path','projection') != 'projection'
            or tuple(parent['selected_state']['head.weight'].shape) != (len(ctx.classes),512)):
        raise ValueError('The parent must be the correct current-stage 512 v1 SWA population')
    targets = None
    if cfg['route']=='crt768_dev':
        targets = historical_artifact(Path(cfg['targets']),ctx)
        if (parent['targets_sha256'] != sha256_file(cfg['targets'])
                or targets['image_paths'] != [r['image_path'] for r in ctx.train]
                or targets['class_names'] != ctx.classes
                or not torch.equal(targets['labels'],torch.tensor([int(r['label']) for r in ctx.train]))
                or set(targets['image_paths']) & {r['image_path'] for r in ctx.val}
                or not len(ctx.val)):
            raise ValueError('Frozen-head supervision must be isolated train_dev, bound to the parent')
    binding = dict(ctx.binding,recipe_sha256=fingerprint(cfg),
        parent_checkpoint_sha256=sha256_file(cfg['parent_checkpoint']),
        parent_binding_sha256=fingerprint(parent['binding']),
        post768_script_sha256=sha256_file(__file__),
        prior_alignment_sha256=sha256_file(ROOT/'reproducibility/aegis_f1/aegis_clip/prior_alignment.py'),
        test_bias_script_sha256=sha256_file(ROOT/'reproducibility/aegis_f1/aegis_clip/v1_test_bias.py'))
    if targets:
        binding['targets_sha256'] = sha256_file(cfg['targets'])
    return ctx,parent,targets,binding


@torch.no_grad()
def cpu_smoke(ctx,parent):
    """Use official weights and a real held-out image, without touching CUDA."""
    model=build_classifier(ctx.official,len(ctx.classes),parent['model_config'],'cpu').eval()
    load_trainable_state(model,parent['selected_state'])
    for parameter in model.parameters():parameter.requires_grad_(False)
    images=next(iter(loader_for(ctx,ctx.val[:1],image_transform(448))))[0]
    captured=[]
    hook=model.visual.ln_post.register_forward_hook(lambda module,args,value:captured.append(value))
    parent_logits=model(images);hook.remove()
    projection=model.visual.proj.detach().clone()
    model.visual.proj=None
    features=model.visual(images)
    torch.testing.assert_close(features,captured[0],rtol=0,atol=0)
    heads=[CosineHead(768,len(ctx.classes),dropout=.1) for _ in range(2)]
    state=dict(weight=parent['selected_state']['head.weight']@projection.t(),
        logit_scale=parent['selected_state']['head.logit_scale'])
    for head in heads:head.load_state_dict(state)
    predictor=IndependentHeads(model.visual,heads).eval()
    output=predictor(images)
    if not torch.isfinite(parent_logits).all() or not torch.isfinite(output).all():
        raise ValueError('Official real-image CPU forward is nonfinite')
    errors=[]
    for i,head in enumerate(heads):
        error=float((head(features)-output[:,i*len(ctx.classes):(i+1)*len(ctx.classes)]).abs().max())
        if error!=0:raise ValueError('Independent CPU heads differ')
        errors.append(error)
    return dict(device='cpu',real_images=1,feature_dimension=features.shape[1],
        independent_candidate_count=2,output_shape=list(output.shape),
        standalone_head_max_errors=errors,visual_requires_grad=False,cuda_used=False)


def equal_class_mass(weights,labels,classes):
    """Equalize the total reliable supervision mass, without changing labels."""
    mass = weights.new_zeros(classes).scatter_add_(0,labels,weights)
    if (not torch.isfinite(weights).all() or (weights<0).any() or (mass<=0).any()):
        raise ValueError('Every class needs finite positive reliable supervision')
    balanced = weights / mass[labels]
    balanced *= weights.sum()/balanced.sum()
    return balanced,mass


class IndependentHeads(nn.Module):
    """Share an identical frozen visual forward; keep each candidate separate."""
    def __init__(self,visual,heads):
        super().__init__()
        self.visual,self.heads = visual,nn.ModuleList(heads)
    def forward(self,images):
        features = self.visual(images.float())
        return torch.cat([head(features) for head in self.heads],dim=1)


def context_for(ctx,cfg,binding,output,model_config):
    clone = copy.copy(ctx)
    clone.config = copy.deepcopy(ctx.config)
    clone.config['project']['experiment_id'] = cfg['experiment_id']
    clone.config['model'] = copy.deepcopy(model_config)
    clone.config['decode'] = copy.deepcopy(DECODER)
    clone.config['output']['root'] = str(output)
    clone.binding = binding
    return clone


def verify_package(ctx,checkpoint,output):
    cache = load_artifact(output/'test_logits.pt',ctx.binding)
    calibration = load_artifact(output/'test_calibration.pt',ctx.binding)
    if (cache['checkpoint_sha256'] != sha256_file(checkpoint)
            or calibration['checkpoint_sha256'] != sha256_file(checkpoint)
            or calibration['test_logits_sha256'] != sha256_file(output/'test_logits.pt')):
        raise ValueError('Package lineage changed')
    matrix,bias = cache['logits'].numpy(),calibration['bias'].numpy()
    if matrix.shape != (ctx.manifest['test_samples'],len(ctx.classes)):
        raise ValueError('Wrong test matrix shape')
    predictions = {'submission_raw':matrix.argmax(1),'submission':(matrix+bias).argmax(1)}
    packages = {}
    for directory,prediction in predictions.items():
        package = output/directory
        expected = ''.join(f'{name}, {ctx.classes[index]}\r\n' for name,index in zip(cache['names'],prediction)).encode()
        if (package/'pred_results.csv').read_bytes() != expected:
            raise ValueError('Independent NumPy CSV replay differs')
        with zipfile.ZipFile(package/'submission.zip') as archive:
            if archive.namelist()!=['pred_results.csv'] or archive.read('pred_results.csv')!=expected:
                raise ValueError('ZIP differs from independently replayed CSV')
        if 'All checks passed' not in (package/'submission_check.log').read_text():
            raise ValueError('Submission checker did not pass')
        packages[directory] = dict(csv=str(package/'pred_results.csv'),zip=str(package/'submission.zip'),
            zip_sha256=sha256_file(package/'submission.zip'),samples=len(prediction),numpy_replay_matches=len(prediction))
    refitted = numpy_uniform_bias(matrix,200)
    error = float(np.max(np.abs(refitted-bias)))
    if error>1e-3:
        raise ValueError('Independent bias refitting exceeds numerical tolerance')
    return dict(packages=packages,checkpoint_sha256=sha256_file(checkpoint),
        independent_bias_max_error=error,independent_refit_prediction_matches=int(np.sum(
            (matrix.astype(np.float64)+refitted).argmax(1)==predictions['submission'])),
        raw_to_bias_changes=int(np.sum(predictions['submission']!=predictions['submission_raw'])),
        independent_test_accuracy=None,platform_score=None)


def desktop_copy(output,prefix,tag):
    paths = {}
    for directory,suffix in [('submission','test_bias'),('submission_raw','raw')]:
        destination = Path(f'{prefix}_{tag}_{suffix}_submission.zip')
        with (output/directory/'submission.zip').open('rb') as source,destination.open('xb') as target:
            shutil.copyfileobj(source,target)
        if sha256_file(destination) != sha256_file(output/directory/'submission.zip'):
            raise ValueError('Desktop package hash differs')
        paths[suffix]=str(destination)
    return paths


def control(ctx,parent,cfg,binding,output):
    run_ctx = context_for(ctx,cfg,binding,output,parent['model_config'])
    checkpoint = output/'selected.pt'
    # New decoder binding, with explicit original lineage. No source file or
    # model tensor is modified and no training/optimizer update is performed.
    save_artifact(checkpoint,dict(selected_state=parent['selected_state'],classes=ctx.classes,
        model_config=parent['model_config'],selected_policy=parent['selected_policy'],
        parent_checkpoint=cfg['parent_checkpoint'],parent_binding=parent['binding'],
        model_parameter_updates=False,training_started=False),binding)
    exported = load_artifact(checkpoint,binding)
    if any(not torch.equal(value,exported['selected_state'][name])
           for name,value in parent['selected_state'].items()):
        raise ValueError('Exported control changed a model tensor')
    infer_test_uniform(run_ctx,checkpoint,'cuda')
    result = verify_package(run_ctx,checkpoint,output)
    result.update(model_parameter_updates=False,training_started=False,
        desktop=desktop_copy(output,cfg['desktop_prefix'],'512'))
    return result


@torch.no_grad()
def extract_features(ctx,parent,output,binding):
    model = build_classifier(ctx.official,len(ctx.classes),parent['model_config'],'cuda').eval()
    load_trainable_state(model,parent['selected_state'])
    for parameter in model.parameters(): parameter.requires_grad_(False)
    projection = model.visual.proj.detach().cpu().clone()
    initial = dict(weight=parent['selected_state']['head.weight'] @ projection.t(),
        logit_scale=parent['selected_state']['head.logit_scale'].clone())
    captured=[]
    hook=model.visual.ln_post.register_forward_hook(lambda module,args,value:captured.append(value.detach()))
    features=torch.empty((len(ctx.full),768))
    val_indices={row['image_path']:i for i,row in enumerate(ctx.val)}
    val_logits=torch.empty((len(ctx.val),len(ctx.classes)))
    started=time.monotonic()
    loader=loader_for(ctx,ctx.full,image_transform(448))
    for batch,(images,indices) in enumerate(loader,1):
        captured.clear()
        logits=model(images.to('cuda'))
        features[indices]=captured[0].cpu()
        for position,index in enumerate(indices.tolist()):
            path=ctx.full[index]['image_path']
            if path in val_indices:val_logits[val_indices[path]]=logits[position].cpu()
        if batch==1 or batch%200==0 or batch==len(loader):
            progress=dict(stage='frozen_train_side_features',batch=batch,batches=len(loader),seconds=time.monotonic()-started)
            atomic_json_dump(progress,output/'progress.json');print(json.dumps(progress),flush=True)
    hook.remove();del model
    if not torch.isfinite(features).all() or not torch.isfinite(val_logits).all():
        raise ValueError('Nonfinite frozen training/validation features')
    cache=output/'frozen_features768.pt'
    save_artifact(cache,dict(features=features,image_paths=[row['image_path'] for row in ctx.full],
        parent_center_val_logits=val_logits,initial_head=initial,
        feature_path='ln_post_pre_projection',image_size=448,visual_parameter_updates=False),binding)
    return features,val_logits,initial,cache


def train_heads(ctx,targets,features,initial,cfg,output):
    fit=cfg['head_fit']
    index={row['image_path']:i for i,row in enumerate(ctx.full)}
    train_x=features[[index[row['image_path']] for row in ctx.train]].to('cuda')
    val_x=features[[index[row['image_path']] for row in ctx.val]].to('cuda')
    active=torch.where(targets['weights']>0)[0].to('cuda')
    labels=targets['targets'].to('cuda')
    reliability=targets['weights'].to('cuda')
    balanced,mass=equal_class_mass(reliability,labels,len(ctx.classes))
    probabilities=target_probabilities(labels,targets['labels'].to('cuda'),
        targets['original_alpha'].to('cuda'),len(ctx.classes),fit['label_smoothing'])
    heads,predictions,histories={}, {}, {}
    for arm,weights in [('unbalanced',reliability),('balanced',balanced)]:
        seed_training(fit['seed'])
        head=CosineHead(768,len(ctx.classes),dropout=fit['dropout']).to('cuda')
        head.load_state_dict(initial)
        optimizer=torch.optim.AdamW(head.parameters(),lr=fit['lr'],weight_decay=fit['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,fit['epochs'],eta_min=fit['minimum_lr'])
        generator=torch.Generator(device='cuda').manual_seed(fit['seed'])
        history=[]
        for epoch in range(1,fit['epochs']+1):
            head.train();losses=[]
            order=active[torch.randperm(len(active),device='cuda',generator=generator)]
            for indices in order.split(fit['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                per_row=-(probabilities[indices]*head(train_x[indices]).log_softmax(1)).sum(1)
                loss=(per_row*weights[indices]).sum()/weights[indices].sum()
                loss.backward();optimizer.step();losses.append(float(loss.detach()))
            scheduler.step()
            history.append(dict(epoch=epoch,mean_loss=float(np.mean(losses)),optimizer_steps=math.ceil(len(active)/fit['batch_size'])))
            print(json.dumps(dict(arm=arm,**history[-1])),flush=True)
        head.eval()
        with torch.no_grad():
            logits=torch.cat([head(values).cpu() for values in val_x.split(fit['batch_size'])])
        heads[arm]=head.cpu();predictions[arm]=logits.argmax(1).numpy();histories[arm]=history
        torch.save(head.state_dict(),output/f'head_{arm}.pt')
    del train_x,val_x,probabilities,balanced,reliability
    return heads,predictions,histories,mass.cpu()


@torch.no_grad()
def test_both(ctx,parent,cfg,binding,heads,output,cache_sha):
    model_config=dict(parent['model_config'],feature_path='pre_projection')
    contexts,checkpoints={},{}
    body_state={key:value for key,value in parent['selected_state'].items() if not key.startswith('head.')}
    for arm,head in heads.items():
        arm_output=output/arm;arm_output.mkdir()
        arm_binding=dict(binding,arm=arm,frozen_features_sha256=cache_sha)
        current=context_for(ctx,cfg,arm_binding,arm_output,model_config)
        state=dict(body_state,**{'head.'+key:value for key,value in head.state_dict().items()})
        checkpoint=arm_output/'selected.pt'
        save_artifact(checkpoint,dict(selected_state=state,classes=ctx.classes,model_config=model_config,
            selected_policy='frozen_visual_head_last20',parent_checkpoint=cfg['parent_checkpoint'],
            parent_binding=parent['binding'],head_epochs=20,visual_parameter_updates=False),arm_binding)
        contexts[arm],checkpoints[arm]=current,checkpoint
    model=build_classifier(ctx.official,len(ctx.classes),model_config,'cuda').eval()
    load_trainable_state(model,load_artifact(checkpoints['balanced'],contexts['balanced'].binding)['selected_state'])
    arms=list(heads)
    predictor=IndependentHeads(model.visual,[heads[arm].to('cuda') for arm in arms]).eval()
    # Prove each shared-forward block matches its standalone saved checkpoint.
    examples=next(iter(loader_for(ctx,ctx.val[:3],image_transform(448))))[0].to('cuda')
    probes=predictor(examples)
    cold_errors={}
    for i,arm in enumerate(arms):
        cold=build_classifier(ctx.official,len(ctx.classes),model_config,'cuda').eval()
        load_trainable_state(cold,load_artifact(checkpoints[arm],contexts[arm].binding)['selected_state'])
        error=float((cold(examples)-probes[:,i*len(ctx.classes):(i+1)*len(ctx.classes)]).abs().max())
        if error>2e-5:raise ValueError('Shared inference differs from standalone candidate')
        cold_errors[arm]=error;del cold
    base=Path(ctx.reference['data']['dataset_manifest']).parent
    rows=read_rows(base/'test_manifest.csv')
    names=[Path(row['image_path']).name for row in rows]
    logits=torch.zeros((len(rows),2*len(ctx.classes)))
    for scale in DECODER['scales']:
        loader=loader_for(ctx,rows,image_transform(448,scale=scale),root=ctx.test_root)
        logits+=2*collect_logits(predictor,loader,'cuda',2*len(ctx.classes),flip=True)
        progress=dict(stage='independent_test_inference',completed_scale=scale,samples=len(rows))
        atomic_json_dump(progress,output/'progress.json');print(json.dumps(progress),flush=True)
    del predictor,model
    results={}
    for i,arm in enumerate(arms):
        current=contexts[arm];arm_output=Path(current.config['output']['root'])
        save_artifact(arm_output/'test_logits.pt',dict(logits=logits[:,i*len(ctx.classes):(i+1)*len(ctx.classes)].contiguous(),
            names=names,class_names=ctx.classes,decoder=DECODER,checkpoint_sha256=sha256_file(checkpoints[arm]),
            selected_policy='frozen_visual_head_last20'),current.binding)
        infer_test_uniform(current,checkpoints[arm],'cuda')
        results[arm]=verify_package(current,checkpoints[arm],arm_output)
        results[arm]['desktop']=desktop_copy(arm_output,cfg['desktop_prefix'],arm)
    return results,cold_errors


def crt(ctx,parent,targets,cfg,binding,output):
    features,baseline,initial,cache=extract_features(ctx,parent,output,binding)
    heads,predictions,histories,mass=train_heads(ctx,targets,features,initial,cfg,output)
    labels=np.array([int(row['label']) for row in ctx.val])
    counts=np.bincount([int(row['label']) for row in ctx.train],minlength=len(ctx.classes))
    tail=np.lexsort((np.arange(len(counts)),counts))[:max(1,len(counts)//10)]
    before=baseline.argmax(1).numpy()
    metrics={arm:metric(pred,labels,len(ctx.classes),tail) for arm,pred in predictions.items()}
    comparisons={arm:paired(before,pred,labels) for arm,pred in predictions.items()}
    balancing_pair=paired(predictions['unbalanced'],predictions['balanced'],labels)
    tail_rows=np.isin(labels,tail)
    balancing_pair['tail75']=paired(predictions['unbalanced'][tail_rows],predictions['balanced'][tail_rows],labels[tail_rows])
    for arm in predictions:
        comparisons[arm]['tail75']=paired(before[tail_rows],predictions[arm][tail_rows],labels[tail_rows])
    # Independently replay every held-out prediction from the frozen features.
    index={row['image_path']:i for i,row in enumerate(ctx.full)}
    val_x=features[[index[row['image_path']] for row in ctx.val]].numpy()
    val_x=val_x/np.maximum(np.linalg.norm(val_x,axis=1,keepdims=True),1e-12)
    replay={}
    for arm,head in heads.items():
        weight=head.weight.detach().numpy();weight=weight/np.maximum(np.linalg.norm(weight,axis=1,keepdims=True),1e-12)
        replay[arm]=int(np.sum((val_x@weight.T).argmax(1)==predictions[arm]))
        if replay[arm]!=len(labels):raise ValueError('Independent held-out head replay differs')
    np.savez_compressed(output/'validation_predictions.npz',labels=labels,parent512=before,
        unbalanced768=predictions['unbalanced'],balanced768=predictions['balanced'],tail=tail,
        image_paths=np.array([row['image_path'] for row in ctx.val]))
    result=dict(parent_center_metric=metric(before,labels,len(ctx.classes),tail),metrics=metrics,
        parent_to_head=comparisons,unbalanced_to_balanced=balancing_pair,head_history=histories,
        independent_val_replay_matches=replay,validation_samples=len(labels),validation_independent=True,
        reliable_class_mass_min=float(mass.min()),reliable_class_mass_max=float(mass.max()),
        teacher_targets_unchanged=True,visual_parameter_updates=False,head_epochs=20,
        val_decoder='single 448 center, no bias; identical frozen features for both arms',
        clean_labels_known=False,automatic_full_training=False,platform_score=None)
    atomic_json_dump(result,output/'development_report.json')
    packages,cold_errors=test_both(ctx,parent,cfg,binding,heads,output,sha256_file(cache))
    result.update(packages=packages,standalone_cold_forward_max_errors=cold_errors)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--cpu-smoke',action='store_true')
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.preflight==args.execute:parser.error('Specify exactly one of --preflight and --execute')
    torch.set_num_threads(2)
    cfg=read_config(args.config)
    ctx,parent,targets,binding=inspect(cfg)
    output=Path(cfg['output']).resolve()
    if args.preflight:
        report=dict(experiment_id=cfg['experiment_id'],status='prepared_not_started',binding=binding,
            training_started=False,route=cfg['route'],parent_partition=ctx.config['source']['partition'],
            train_samples=len(ctx.train),val_samples=len(ctx.val),parent_checkpoint=cfg['parent_checkpoint'],
            output=str(output),platform_score=None)
        if targets:
            adjusted,mass=equal_class_mass(targets['weights'],targets['targets'],len(ctx.classes))
            report.update(active_training_rows=int((targets['weights']>0).sum()),
                reliable_class_mass_min=float(mass.min()),reliable_class_mass_max=float(mass.max()))
        if args.cpu_smoke:report['official_cpu_smoke']=cpu_smoke(ctx,parent)
        print(json.dumps(report,ensure_ascii=False,indent=2));return
    if not args.prepared or json.loads(args.prepared.read_text())['binding']!=binding:
        raise ValueError('Execution needs the unchanged verified preparation binding')
    if output.exists():raise FileExistsError('Fresh post-768 job needs an independent empty output')
    if not torch.cuda.is_available():raise RuntimeError('Local CUDA unavailable; no remote/NPU fallback')
    output.mkdir(parents=True)
    status=dict(experiment_id=cfg['experiment_id'],status='running',route=cfg['route'],binding=binding,
        training_started=cfg['route']=='crt768_dev',platform_score=None)
    atomic_json_dump(status,output/'status.json')
    started=time.monotonic()
    try:
        result=control(ctx,parent,cfg,binding,output) if cfg['route']=='512_control' else crt(ctx,parent,targets,cfg,binding,output)
        result.update(experiment_id=cfg['experiment_id'],status='completed_verified_delivery',
            binding=binding,elapsed_seconds=time.monotonic()-started,route=cfg['route'],
            training_started=cfg['route']=='crt768_dev',official_confirmation='user-relayed 2026-10-01')
        atomic_json_dump(result,output/'report.json')
        status.update(status='completed',elapsed_seconds=time.monotonic()-started,report=str(output/'report.json'))
        atomic_json_dump(status,output/'status.json')
    except BaseException as error:
        status.update(status='failed',error=str(error),elapsed_seconds=time.monotonic()-started)
        atomic_json_dump(status,output/'status.json');raise


if __name__=='__main__':main()
