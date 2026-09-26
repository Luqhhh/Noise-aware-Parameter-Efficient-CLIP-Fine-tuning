"""T0 过度离散统计量的零假设标定（rematch750）。

只读、CPU-only（不占用 GPU/NPU）。**不训练、不推理、不改判据、不派生候选、不占名额。**

## 为什么做

T0 审计（`audit_rematch750_class_coverage.py` §Q4）报告：

    k = 37444 / 14880 = 2.5164
    z_c = (Y_c/k - X_c) / sqrt(X_c * (1 + 1/k))
    sd(z) = 2.5128,  chi2/dof = 6.3175,  |z|>2 有 244 类（期望 33.7）, |z|>3 有 110 类

其中 X = RM-LP 在 val 上的逐类预测次数，Y = RM-LP 在 test 上的逐类预测次数。
当时的解读是「test 的逐类预测分布不只是 val 按 k 放大」，即 val 与 test 之间存在分布差。

**但这个解读缺一个对照。** 统计量在 H0（同分布）下的方差是 Var[z_c] ≈ 1 - p_c ≤ 1，
推导假定逐类计数服从多项分布。而图像不是独立同分布抽样 —— 近重复图会成簇，
逐类计数本身就有超出多项的方差。于是 sd(z) 会**在任何两个不相交样本之间**都大于 1，
不需要 val 与 test 有分布差。

本脚本就是判这一件事：**把同一统计量算在「同一总体的两个随机不相交子样本」上，
看 sd(z) 的自然基线是多少。**

## 设计

观察量 O1 = 已发布的 val↔test 统计量（RM-LP，n=14880 / 37444，k=2.5164）。

零假设标定（每项 1000 次重复，固定种子）：
  N1 `test_split_14880_22564` 把 test 的 37444 条预测随机置换后切成 14880 / 22564，k=1.5168。
     **与 O1 的 X 规模（14880）严格同尺度**，且两侧来自同一总体 ⇒ 任何超出 1 的离散
     都是统计量自身的基线，不是分布差。
  N2 `test_split_half`        test 18722 / 18722，k=1。k 对基线的影响。
  N3 `val_split_half`         val 14880 随机切 7440 / 7440，k=1。域内的对应基线。

O1 与 N1 的差别**只有一件事**：N1 的两半来自同一总体，O1 的两半来自 val 与 test。
其余（统计量、X 的样本量、逐类计数的稀疏程度）全部对齐。

## 预注册判据（跑之前写定，跑完不改）

对 sd(z) 与 chi2/dof 各算经验 p 值 = N1 中 >= O1 的比例。

- **p > 0.05** → O1 的离散度落在「同一总体两个子样本」的自然基线之内。
  T0 §Q4 **不构成 val↔test 分布差的证据**；「来源漂移」这一解读失去该条支持，
  11.90pp 的本地/平台差回到「机制未明」。
- **p <= 0.05** → O1 显著超出同总体基线。T0 §Q4 作为「存在分布差」的证据成立
  （但**机制仍不可辨识**：统计量比对的是预测分布，test 标签不可得，
  不能区分 a) test 的真实类别分布不同 与 b) 逐类漂移崩溃）。

两种情况都**不改任何判据、不派生候选、不占提交名额**。这只是对一条既有测量事实的标定。

## 附带（同一脚本、同样只读）

- 零结构对照：O1 中 X_c=0 的类有 9 个（7 个 Y_c 也为 0，2 个 Y_c>0）。
  N1 中 A_c=0 的类有多少、其中 B_c>0 的有多少 —— 同一尺度下的自然数量。
- 手臂对照：对 RM_FT 的 test CSV 重跑 N1，看过度离散是否与手臂有关
  （RM_FT 的 val 预测需 GPU 前向，本脚本不做，故 O1 只有 LP 一侧）。

用法：
    PYTHONPATH=reproducibility/aegis_f1 \\
    .venv/bin/python scripts/audit_rematch750_dispersion_null.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

N_CLASSES = 750
TEST_SAMPLES = 37444
VAL_SAMPLES = 14880
REPS = 1000
SEED = 42


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_pred(path: Path) -> list[int]:
    labs = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(', ')
            if len(parts) != 2:
                continue
            labs.append(int(parts[1]))
    return labs


def state_dict(ck):
    sd = ck
    for key in ('model', 'state_dict', 'model_state_dict', 'module'):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            sd = ck[key]
            break
    if not [k for k in sd if k.endswith('classifier.weight')]:
        sd = ck
    return sd


def dispersion(x: torch.Tensor, y: torch.Tensor) -> dict:
    """x, y 为两个长度 N_CLASSES 的计数张量；k = y.sum()/x.sum()。"""
    k = float(y.sum()) / float(x.sum())
    mask = x > 0
    z = ((y / k - x) / torch.sqrt(x * (1 + 1 / k)))[mask]
    return {
        'k': k,
        'defined_classes': int(mask.sum()),
        'mean_z': float(z.mean()),
        'sd_z': float(z.std()),
        'chi2_over_dof': float((z ** 2).mean()),
        'abs_z_gt_2': int((z.abs() > 2).sum()),
        'abs_z_gt_3': int((z.abs() > 3).sum()),
        'abs_z_gt_4': int((z.abs() > 4).sum()),
        'x0_y_pos': int(((x == 0) & (y > 0)).sum()),
        'x0_y0': int(((x == 0) & (y == 0)).sum()),
    }


def summarize(null_reps: list[dict], observed: dict) -> dict:
    out = {}
    for key in ('sd_z', 'chi2_over_dof', 'abs_z_gt_2', 'abs_z_gt_3', 'abs_z_gt_4',
                'defined_classes', 'x0_y_pos', 'x0_y0', 'mean_z'):
        vals = torch.tensor([r[key] for r in null_reps], dtype=torch.float64)
        obs = float(observed[key])
        # 经验 p 值：零分布中 >= 观察值的比例（单侧，越大越离散）
        p_ge = float((vals >= obs).float().mean())
        out[key] = {
            'observed': obs,
            'null_mean': float(vals.mean()),
            'null_sd': float(vals.std()),
            'null_p2_5': float(vals.quantile(0.025)),
            'null_median': float(vals.quantile(0.5)),
            'null_p97_5': float(vals.quantile(0.975)),
            'null_min': float(vals.min()),
            'null_max': float(vals.max()),
            'p_ge_observed': p_ge,
        }
    return out


def split_null(pool: torch.Tensor, n_a: int, reps: int, seed: int) -> list[dict]:
    """把 pool（逐样本预测标签）随机置换后切成 n_a / 其余，算统计量。"""
    total = pool.numel()
    n_b = total - n_a
    g = torch.Generator().manual_seed(seed)
    reps_out = []
    for _ in range(reps):
        perm = torch.randperm(total, generator=g)
        a = torch.bincount(pool[perm[:n_a]], minlength=N_CLASSES).float()
        b = torch.bincount(pool[perm[n_a:]], minlength=N_CLASSES).float()
        assert int(b.sum()) == n_b
        reps_out.append(dispersion(a, b))
    return reps_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='results/rematch750_dispersion_null_20260926.json')
    ap.add_argument('--reps', type=int, default=REPS)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    art = root / 'artifacts/stages/repechage/20260921'
    rep: dict = {
        'stage': 'measurement',
        'test_used_for_tuning': False,
        'changes_any_threshold': False,
        'derives_any_candidate': False,
        'uses_submission_slot': False,
        'reps': args.reps,
        'seed': SEED,
    }

    # ---------- 重建 RM-LP 的 val 预测（CPU，缓存特征 + 线性头） ----------
    feats = torch.as_tensor(torch.load(art / 'features/features.pt',
                                       map_location='cpu')).float()
    paths = json.loads((art / 'features/image_paths.json').read_text())
    index = {p: i for i, p in enumerate(paths)}
    vp, vlab = [], []
    import csv as _csv
    with (art / 'val_dev.csv').open(newline='', encoding='utf-8') as f:
        for r in _csv.DictReader(f):
            vp.append(r['image_path'].removeprefix('train/'))
            vlab.append(int(r['label']))
    vi = torch.tensor([index[p] for p in vp])
    vl = torch.tensor(vlab)
    ck = torch.load(root / 'outputs/rematch750/RM_LP/seed42/checkpoints/best.pt',
                    map_location='cpu', weights_only=False)
    sd = state_dict(ck)
    W = sd[[k for k in sd if k.endswith('classifier.weight')][0]].float()
    B = sd[[k for k in sd if k.endswith('classifier.bias')][0]].float()
    lp_val_pred = (feats[vi] @ W.t() + B).argmax(1)
    assert lp_val_pred.numel() == VAL_SAMPLES, lp_val_pred.numel()
    lp_val_eval = json.loads((root / 'outputs/rematch750/RM_LP/seed42/logs/'
                              'evaluation_epoch_20.json').read_text())
    rec_micro = float((lp_val_pred == vl).float().mean())
    print(f'[val-rebuild] rows={lp_val_pred.numel()} micro={rec_micro!r} '
          f'recorded={lp_val_eval["raw_micro"]!r} '
          f'diff_images={abs(rec_micro - lp_val_eval["raw_micro"]) * VAL_SAMPLES:.0f}')
    rep['val_prediction_source'] = {
        'method': 'frozen backbone + linear head on cached features; CPU only',
        'checkpoint': 'outputs/rematch750/RM_LP/seed42/checkpoints/best.pt',
        'rows': int(lp_val_pred.numel()),
        'reconstructed_micro': rec_micro,
        'recorded_micro': float(lp_val_eval['raw_micro']),
        'abs_micro_diff_images': abs(rec_micro - lp_val_eval['raw_micro']) * VAL_SAMPLES,
        'predicted_class_count': int(lp_val_pred.unique().numel()),
    }

    # ---------- 各手臂的 test 预测 ----------
    pools = {}
    for tag, rel in (('RM_LP', 'outputs/rematch750/RM_LP/seed42/submission/pred_results.csv'),
                     ('RM_FT', 'outputs/rematch750/RM_FT/seed42/submission/pred_results.csv')):
        p = root / rel
        labs = read_pred(p)
        assert len(labs) == TEST_SAMPLES, (tag, len(labs))
        pools[tag] = torch.tensor(labs)
        rep.setdefault('test_csv', {})[tag] = {
            'path': rel, 'sha256': sha256_file(p), 'rows': len(labs),
            'distinct_predicted_classes': len(Counter(labs)),
        }
        print(f'[{tag}] test rows={len(labs)} distinct={len(Counter(labs))} '
              f'sha={rep["test_csv"][tag]["sha256"][:12]}')

    x_lp_val = torch.bincount(lp_val_pred, minlength=N_CLASSES).float()

    # ---------- O1：已发布的 val↔test 统计量 ----------
    observed = {}
    for tag in ('RM_LP', 'RM_FT'):
        y_test = torch.bincount(pools[tag], minlength=N_CLASSES).float()
        if tag == 'RM_LP':
            observed['O1_RM_LP_val_vs_test'] = dispersion(x_lp_val, y_test)
        else:
            observed['O1b_RM_FT_test_only_no_val'] = None
    o1 = observed['O1_RM_LP_val_vs_test']
    print(f'[O1] val(LP) vs test(LP): k={o1["k"]:.4f} sd(z)={o1["sd_z"]:.4f} '
          f'chi2/dof={o1["chi2_over_dof"]:.4f} |z|>2={o1["abs_z_gt_2"]} '
          f'|z|>3={o1["abs_z_gt_3"]} defined={o1["defined_classes"]} '
          f'x0_y_pos={o1["x0_y_pos"]} x0_y0={o1["x0_y0"]}')
    rep['O1_val_vs_test'] = o1

    # ---------- N1：test 内部同尺度切分（决定性对照） ----------
    n1 = split_null(pools['RM_LP'], VAL_SAMPLES, args.reps, SEED)
    rep['N1_test_split_14880_22564'] = {
        'what': 'test 37444 随机切 14880 / 22564，两侧同总体，k=1.5168；X 规模与 O1 严格一致',
        'pool': 'RM_LP test pred_results.csv',
        'n_a': VAL_SAMPLES, 'n_b': TEST_SAMPLES - VAL_SAMPLES,
        'summary': summarize(n1, o1),
    }
    s = rep['N1_test_split_14880_22564']['summary']
    print(f'[N1] test-split same-population null (n={args.reps}): '
          f'sd(z) median={s["sd_z"]["null_median"]:.4f} '
          f'[{s["sd_z"]["null_p2_5"]:.4f}, {s["sd_z"]["null_p97_5"]:.4f}] '
          f'chi2/dof median={s["chi2_over_dof"]["null_median"]:.4f} '
          f'|z|>2 median={s["abs_z_gt_2"]["null_median"]:.0f} '
          f'p(sd_z>=O1)={s["sd_z"]["p_ge_observed"]:.4f} '
          f'p(chi2>=O1)={s["chi2_over_dof"]["p_ge_observed"]:.4f}')

    # ---------- N2：test 等半切分，k=1 ----------
    n2 = split_null(pools['RM_LP'], TEST_SAMPLES // 2, args.reps, SEED + 1)
    rep['N2_test_split_half'] = {
        'what': 'test 37444 随机切 18722 / 18722，k=1',
        'pool': 'RM_LP test pred_results.csv',
        'n_a': TEST_SAMPLES // 2, 'n_b': TEST_SAMPLES - TEST_SAMPLES // 2,
        'summary': summarize(n2, o1),
    }
    s2 = rep['N2_test_split_half']['summary']
    print(f'[N2] test half/half (k=1): sd(z) median={s2["sd_z"]["null_median"]:.4f} '
          f'[{s2["sd_z"]["null_p2_5"]:.4f}, {s2["sd_z"]["null_p97_5"]:.4f}] '
          f'p(sd_z>=O1)={s2["sd_z"]["p_ge_observed"]:.4f}')

    # ---------- N3：val 内部等半切分，域内 ----------
    n3 = split_null(lp_val_pred, VAL_SAMPLES // 2, args.reps, SEED + 2)
    rep['N3_val_split_half'] = {
        'what': 'val 14880 随机切 7440 / 7440，k=1，域内基线',
        'pool': 'RM-LP val predictions (CPU reconstruction)',
        'n_a': VAL_SAMPLES // 2, 'n_b': VAL_SAMPLES - VAL_SAMPLES // 2,
        'summary': summarize(n3, o1),
    }
    s3 = rep['N3_val_split_half']['summary']
    print(f'[N3] val half/half (k=1): sd(z) median={s3["sd_z"]["null_median"]:.4f} '
          f'[{s3["sd_z"]["null_p2_5"]:.4f}, {s3["sd_z"]["null_p97_5"]:.4f}] '
          f'p(sd_z>=O1)={s3["sd_z"]["p_ge_observed"]:.4f}')

    # ---------- N1-FT：同一对照跑在 RM_FT 的 test CSV 上 ----------
    n1ft = split_null(pools['RM_FT'], VAL_SAMPLES, args.reps, SEED + 3)
    rep['N1ft_test_split_RM_FT'] = {
        'what': 'N1 同设计，但用 RM_FT 的 test 预测（手臂对照）',
        'pool': 'RM_FT test pred_results.csv',
        'n_a': VAL_SAMPLES, 'n_b': TEST_SAMPLES - VAL_SAMPLES,
        'summary': summarize(n1ft, o1),
    }
    s1ft = rep['N1ft_test_split_RM_FT']['summary']
    print(f'[N1ft] RM_FT test-split null: sd(z) median={s1ft["sd_z"]["null_median"]:.4f} '
          f'[{s1ft["sd_z"]["null_p2_5"]:.4f}, {s1ft["sd_z"]["null_p97_5"]:.4f}] '
          f'|z|>2 median={s1ft["abs_z_gt_2"]["null_median"]:.0f}')

    # ---------- 预注册判据 ----------
    p_sd = s['sd_z']['p_ge_observed']
    p_chi = s['chi2_over_dof']['p_ge_observed']
    if p_sd > 0.05:
        verdict = ('NOT_EVIDENCE: O1 的 sd(z) 落在同总体切分基线内。'
                   'T0 §Q4 不构成 val↔test 分布差的证据；「来源漂移」解读失去该条支持，'
                   '本地/平台 11.90pp 差的机制回到未明。')
    else:
        verdict = ('EVIDENCE_OF_DIFFERENCE: O1 的 sd(z) 显著超出同总体切分基线。'
                   '但机制仍不可辨识 —— 统计量比对预测分布且 test 标签不可得，'
                   '不能区分「test 真实类别分布不同」与「逐类漂移崩溃」。')
    rep['preregistered_verdict'] = {
        'rule': 'p_ge(sd_z) > 0.05 → NOT_EVIDENCE; <= 0.05 → EVIDENCE_OF_DIFFERENCE',
        'p_ge_sd_z_N1': p_sd,
        'p_ge_chi2_N1': p_chi,
        'verdict': verdict.split(':')[0],
        'text': verdict,
    }
    print(f'\n[VERDICT] p_ge(sd_z|null N1)={p_sd:.4f}  p_ge(chi2|null N1)={p_chi:.4f}')
    print(f'          {verdict}')

    out = root / args.out
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
