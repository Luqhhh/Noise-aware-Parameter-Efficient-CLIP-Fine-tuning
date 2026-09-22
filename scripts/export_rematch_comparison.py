"""Export existing RM_FT/RM_LT checkpoints for paired local comparison."""
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.cli.cache_validation_logits import cache_validation_logits
from aegis_clip.config import load_config
from aegis_clip.data import TestImageDataset
from aegis_clip.rematch_assets import validate_checkpoint, validate_dataset
from aegis_clip.runtime import set_seed, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    for candidate in ('RM_FT', 'RM_LT'):
        set_seed(42, deterministic=True)
        config_path = ROOT / f'configs/rematch750_{candidate[3:].lower()}.yaml'
        config = load_config(config_path)
        run = ROOT / 'outputs/rematch750' / candidate / 'seed42'
        checkpoint_path = run / 'checkpoints/best.pt'
        out = run / 'comparison'
        out.mkdir(exist_ok=True)
        validate_dataset(config)
        validate_checkpoint(checkpoint_path, config)
        print(f'{candidate}: exporting val logits', flush=True)
        val_path = out / 'val_logits.pt'
        if not val_path.exists():
            cache_validation_logits(checkpoint_path, val_path,
                batch_size=128, num_workers=4, force_online_images=True)
        val = torch.load(val_path, map_location='cpu', weights_only=False)
        assert val['checkpoint_sha256'] == sha256_file(checkpoint_path)
        assert val['validation_csv_sha256'] == sha256_file(config['data']['val_csv'])
        split = rows(config['data']['val_csv'])
        canonical_paths = [r['image_path'] for r in split]
        assert all(a == b or 'train/' + a == b for a, b in zip(val['paths'], canonical_paths))
        assert len(val['paths']) == len(canonical_paths)
        val['paths'] = canonical_paths
        torch.save(val, val_path)
        labels = val['labels']
        assert labels.tolist() == [int(r['label']) for r in split]
        logits = val['logits']
        assert logits.shape == (14880, 750) and torch.isfinite(logits).all()
        pred = logits.argmax(1)
        prob = logits.softmax(1)
        correct = pred.eq(labels)
        support = torch.bincount(labels, minlength=750)
        counts = torch.bincount(labels[correct], minlength=750)
        recall = counts.double() / support
        historical = json.loads((run / 'checkpoints/best_evaluation.json').read_text())
        assert counts.tolist() == [r['correct'] for r in historical['per_class']]
        train_counts = [r['train_samples'] for r in historical['per_class']]
        tail = sorted(range(750), key=lambda i: (train_counts[i], i))[:75]
        worst = sorted(range(750), key=lambda i: (float(recall[i]), i))[:75]
        metrics = dict(micro=correct.double().mean().item(), macro=recall.mean().item(),
            bottom_10_percent_macro=recall[worst].mean().item(),
            bottom_10_percent_class_ids=worst,
            train_support_bottom_10_percent_macro=recall[tail].mean().item(),
            train_support_bottom_10_percent_class_ids=tail,
            correct=int(correct.sum()), samples=len(labels), best_epoch=8,
            historical_per_class_correct_exact_match=True)
        with (out / 'val_prediction_records.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'label', 'prediction', 'correct', 'confidence', 'true_label_probability'])
            for i, path in enumerate(val['paths']):
                writer.writerow([path, int(labels[i]), int(pred[i]), int(correct[i]),
                    float(prob[i, pred[i]]), float(prob[i, labels[i]])])
        print(f'{candidate}: val matches historical counts; exporting test logits', flush=True)
        device = torch.device('cuda')
        model, preprocess, checkpoint = build_from_checkpoint(checkpoint_path, device)
        model.eval()
        stage = Path(config['data']['dataset_manifest']).parent
        source_hashes = {Path(r['image_path']).name: r['file_sha256'] for r in rows(stage / 'test_manifest.csv')}
        dataset = TestImageDataset(config['data']['test_root'], preprocess, source_hashes=source_hashes)
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        parts, names = [], []
        for index, batch in enumerate(loader):
            with torch.autocast('cuda', enabled=config['train']['amp']):
                output = model(images=batch['images'].to(device))
            parts.append(output.float().cpu())
            names.extend(batch['name'])
            if index % 50 == 0:
                print(f'{candidate}: test {len(names)}/{len(dataset)}', flush=True)
        test_logits = torch.cat(parts)
        assert test_logits.shape == (37444, 750) and torch.isfinite(test_logits).all()
        mapping = json.loads(Path(config['data']['class_mapping']).read_text())
        inverse = {int(v): k for k, v in mapping.items()}
        with (run / 'submission/pred_results.csv').open() as f:
            submitted = {name: label.strip() for name, label in csv.reader(f)}
        assert len(submitted) == len(names)
        assert all(submitted[name] == inverse[p] for name, p in zip(names, test_logits.argmax(1).tolist()))
        torch.save(dict(logits=test_logits, names=names, checkpoint_sha256=sha256_file(checkpoint_path),
            class_mapping=mapping, inference='224 CLIP center crop, AMP, no TTA, no prior'), out / 'test_logits.pt')
        metrics['test_submission_predictions_exact_match'] = True
        bindings = [config_path, checkpoint_path, stage / 'train_dev.csv', stage / 'val_dev.csv',
            stage / 'class_to_idx.json', stage / 'dataset_manifest.json',
            out / 'val_logits.pt', out / 'test_logits.pt', out / 'val_prediction_records.csv']
        report = dict(candidate=candidate, metrics=metrics,
            bottom_10_percent_definition='Mean recall of the 75 lowest validation-recall classes, separately per model; ties by class ID',
            fixed_tail_definition='75 smallest train sample counts; ties by class ID',
            probability_conversion='torch.softmax(payload["logits"], dim=1); uncalibrated',
            sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in bindings})
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(candidate=candidate, **{k: v for k, v in metrics.items() if not isinstance(v, list)})), flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
