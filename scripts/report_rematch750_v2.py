"""Build the V2 comparison from completed execution audits and frozen baselines."""
import csv
import json
from pathlib import Path

from aegis_clip.runtime import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for candidate, score in [('RM_FT', 60.96570879179575), ('RM_LT', 60.885589146458706)]:
        path = ROOT / f'outputs/rematch750/{candidate}/seed42/checkpoints/best_evaluation.json'
        m = json.loads(path.read_text())
        tail = sorted(m['per_class'], key=lambda r: (r['train_samples'], r['label']))[:75]
        ids = sorted(r['label'] for r in tail)
        rest = [r for r in m['per_class'] if r['label'] not in ids]
        rows.append(dict(candidate=candidate, selected_epoch=8, raw_macro=m['raw_macro'], raw_micro=m['raw_micro'],
                         fixed_tail75_macro=sum(r['recall'] for r in tail)/75,
                         rest675_macro=sum(r['recall'] for r in rest)/675,
                         fixed_tail75_class_ids=ids, successful_optimizer_updates=33456,
                         training_seconds=None, train_cli_seconds=None, platform_score=score,
                         source=str(path.relative_to(ROOT)), source_sha256=sha256_file(path),
                         decision='retained_platform_baseline' if candidate == 'RM_FT' else 'closed_sampling_comparison'))
    baseline = rows[0]
    for name in ('npu1024', 'lora', 'npu32'):
        path = ROOT / f'results/rematch750_v2_{name}.json'
        m = json.loads(path.read_text())
        assert m['status'] == 'passed' and m['reload_metrics_exact']
        assert m['fixed_tail75_class_ids'] == baseline['fixed_tail75_class_ids']
        row = {k:m[k] for k in ('candidate','selected_epoch','raw_macro','raw_micro','fixed_tail75_macro',
                               'rest675_macro','successful_optimizer_updates','training_seconds','train_cli_seconds','platform_score')}
        delta = 100 * (m['raw_macro']-baseline['raw_macro'])
        row.update(source=str(path.relative_to(ROOT)),source_sha256=sha256_file(path),
                   macro_delta_vs_ft_pp=delta,
                   decision=('eligible_local_screen' if delta >= .20 else 'closed_accuracy_regression' if delta < -2
                             else 'below_promotion_gate'))
        row['submission_generated'] = False
        if name == 'npu32' and delta > -2:
            row['role'] = 'retained_efficiency_configuration_not_statistical_equivalence'
        rows.append(row)
    for row in rows:
        assert abs(.1*row['fixed_tail75_macro']+.9*row['rest675_macro']-row['raw_macro']) < 1e-6
    out = ROOT / 'results/rematch750_v2'
    out.mkdir(exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(dict(rows=rows, primary_metric='raw_macro',
        tiebreak_metric='raw_micro', promotion_gate_pp=.20,
        platform_note='Only FT/LT have measured platform scores; new runs are local validation only.'),indent=2)+'\n')
    columns=['candidate','selected_epoch','raw_macro','raw_micro','fixed_tail75_macro','rest675_macro',
             'successful_optimizer_updates','training_seconds','train_cli_seconds','platform_score','decision']
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    table='| candidate | selected_epoch | raw_macro | raw_micro | 固定尾部75类 macro | 其余675类 macro | 成功更新数 | 纯训练秒数 / CLI总秒数 | 平台分数 |\n'
    table+='|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        time='未记录' if r['training_seconds'] is None else f"{r['training_seconds']:.1f} / {r['train_cli_seconds']:.1f}"
        platform='未测量' if r['platform_score'] is None else f"{r['platform_score']:.7f}%"
        table+=f"| {r['candidate']} | {r['selected_epoch']} | {100*r['raw_macro']:.4f}% | {100*r['raw_micro']:.4f}% | {100*r['fixed_tail75_macro']:.4f}% | {100*r['rest675_macro']:.4f}% | {r['successful_optimizer_updates']} | {time} | {platform} |\n"
    (out/'comparison.md').write_text(table)
    print(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    curves = {}
    curves['GPU FT (reference)'] = [json.loads((ROOT / f'outputs/rematch750/RM_FT/seed42/logs/evaluation_epoch_{e}.json').read_text()) for e in (2, 4, 6, 8)]
    for name, label in [('npu1024','NPU FT batch 1024'),('npu32','NPU FT batch 32'),('lora','GPU LoRA batch 32')]:
        curves[label] = json.loads((ROOT / f'results/rematch750_v2_{name}.json').read_text())['curves']
    for ax, metric in zip(axes, ['raw_macro', 'raw_micro']):
        for label, points in curves.items():
            ax.plot([2,4,6,8], [100*p[metric] for p in points], marker='o', label=label,
                    linestyle='--' if label.startswith('GPU FT') else '-')
        ax.set(xlabel='Completed epochs', ylabel=metric+' (%)', xticks=[2,4,6,8])
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('REMATCH750_V2: independent validation, fixed eight-epoch recipes')
    fig.tight_layout()
    fig.savefig(out/'validation_curves.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
