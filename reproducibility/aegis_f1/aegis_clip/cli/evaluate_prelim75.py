"""Real ten-view overlap diagnostic under the fixed uncalibrated P protocol."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import torch
from aegis_clip.checkpoint import _atomic_torch_save
from aegis_clip.prelim75 import (
    OfficialImages, SCALES, adapted_dual_local_view_logits, extract_attention_crops,
    forward_features_with_last_block_attention, fused_logits, gpu_setup,
    load_composite, load_plan, loader, prediction_metrics, supervision,
)
from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


@torch.no_grad()
def evaluate(plan, name):
    if name not in ('P', 'H0', 'H1', 'V0', 'V1', 'C'):
        raise ValueError('Unknown candidate')
    root = Path(plan['output'])
    ckpt = Path(plan['parent']) if name == 'P' else root / name / 'candidate.pt'
    output = root / ('P_diagnostic' if name == 'P' else name + '/diagnostic')
    output.mkdir(parents=True, exist_ok=False)
    device = gpu_setup(); start = time.monotonic()
    data = supervision(plan)
    model, preprocess, checkpoint, o3, pta = load_composite(ckpt, device)
    model.requires_grad_(False); o3.requires_grad_(False); pta.requires_grad_(False)
    if name != 'P' and checkpoint.get('prelim75_training', {}).get('parent_sha256') != plan['parent_sha256']:
        raise ValueError('Candidate was not trained from the fixed P')
    if checkpoint.get('prelim75_training', {}).get('prior_in_inference', False):
        raise ValueError('Training prior must not enter inference')
    frame = pd.read_csv(plan['val_csv'])
    paths = [canonical_sample_path(p) for p in frame.image_path.astype(str)]
    if len(paths) != 10316 or len(set(paths)) != len(paths):
        raise ValueError('Diagnostic row identity mismatch')
    index = {p: i for i,p in enumerate(data['paths'])}
    idx = torch.tensor([index[p] for p in paths])
    labels = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
    if not torch.equal(labels, data['labels'][idx]):
        raise ValueError('Diagnostic labels differ from fixed fitting list')
    dataset = OfficialImages(paths, plan['train_root'], preprocess)
    stream = loader(dataset, 64, plan)
    predictions = []
    for step, batch in enumerate(stream):
        images = batch['images'].to(device, non_blocking=True)
        globals_ = []; locals_ = []
        for view in (images, images.flip(3)):
            global_logits, _, attention = forward_features_with_last_block_attention(model, view)
            globals_.append(global_logits)
            for scale in SCALES:
                crop, _ = extract_attention_crops(view, attention, crop_size=scale, top_k=5)
                locals_.append(adapted_dual_local_view_logits(model, o3, pta, crop))
        scores = fused_logits(torch.stack([*globals_, *locals_], dim=1))
        if scores.shape[1] != 500 or not torch.isfinite(scores).all():
            raise RuntimeError('Nonfinite/invalid diagnostic scores')
        predictions.append(scores.argmax(1).cpu())
        if step % 20 == 0:
            progress = {'status': 'running', 'candidate': name,
                        'completed_samples': sum(len(p) for p in predictions),
                        'elapsed_seconds': time.monotonic()-start}
            atomic_json_dump(progress, output/'status.json'); print(json.dumps(progress), flush=True)
    pred = torch.cat(predictions)
    fields = {'labels': labels, 'clean_probability': data['clean_probability'][idx],
              'pseudo_labels': data['pseudo_label'][idx], 'correction_alpha': data['correction_alpha'][idx]}
    metrics = prediction_metrics(pred, **fields, num_classes=500, clean_core_threshold=.7)
    result = {'status': 'complete', 'candidate': name, 'validation_scope': 'overlap_diagnostic',
              'independent_generalization_claim': False, 'metrics': metrics,
              'checkpoint_sha256': sha256_file(ckpt), 'val_csv_sha256': plan['val_csv_sha256'],
              'class_mapping_sha256': plan['class_mapping_sha256'],
              'prior_in_inference': False, 'real_candidate_attention_recomputed': True,
              'elapsed_seconds': time.monotonic()-start,
              'max_cuda_memory_allocated': torch.cuda.max_memory_allocated(), 'online_accuracy': None}
    if name != 'P':
        parent = json.loads((root/'P_diagnostic/result.json').read_text())
        if parent['checkpoint_sha256'] != plan['parent_sha256'] or parent['val_csv_sha256'] != plan['val_csv_sha256']:
            raise ValueError('Parent diagnostic binding mismatch')
        deltas = {k: 100*(metrics[k]-parent['metrics'][k]) for k in ('raw_micro', 'clean_core_micro')}
        result['delta_pp_vs_P'] = deltas
        result['engineering_stop'] = any(v < -2.0 for v in deltas.values())
    _atomic_torch_save({'paths': paths, 'prediction': pred, **fields}, output/'predictions.pt')
    atomic_json_dump(result, output/'result.json'); atomic_json_dump(result, output/'status.json')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidate', choices=['P', 'H0', 'H1', 'V0', 'V1', 'C'], required=True)
    args = parser.parse_args()
    evaluate(load_plan(args.config), args.candidate)

if __name__ == '__main__':
    main()
