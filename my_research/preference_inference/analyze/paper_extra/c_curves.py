"""
Paper extra C, learning stability (docs/paper_extra_log.md Sec.14.4; descriptive): training curves of the
sign-randomised, zero-pleasure and experienced-80% runs (+ no experience for reference), seeds 0-2, from the
training logs data/result/<exp>/<seed>/log/{train,test}/{self_vision,feature_prediction_self,feature_prediction_other}.log
(epoch, value). Works on partial logs (runs still training). Writes
data/result/paper_extra/C_signrand/curves/curves.json (last value, minimum, epochs with NaN / increase > 20% over
10 epochs in train self_vision) and curves.{pdf,png} (Type 42 via analyze/paper/paper_style).
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'paper'))
import paper_style as ps  # noqa

RES = os.path.join(_PI, 'data', 'result')
OUT = os.path.join(RES, 'paper_extra', 'C_signrand', 'curves')
RUNS = (('signrand', 'paper_extra/C_signrand/s1_signrand_mse', 'paper_extra/C_signrand/s1_signrand_l1_s{s}', '#8a3ffc', 'D', '符号無作為'),
        ('pzero', 'paper_extra/C_pzero/s1_pzero_mse', 'paper_extra/C_pzero/s1_pzero_l1_s{s}', '#1baf7a', 'v', '快・不快 零'),
        ('mixed80', 'v13_s1_mixed_mse', 'v13_s1_mixed_l1_s{s}', ps.C_80, ps.M_80, ps.COND_LABEL['80']),
        ('random', 'v12_s1_random_mse', 'v12_s1_random_l1_s{s}', ps.C_NONE, ps.M_NONE, ps.COND_LABEL['none']))
SERIES = (('train', 'self_vision'), ('test', 'self_vision'), ('train', 'feature_prediction_other'))


def read(exp, s, split, q):
    f = os.path.join(RES, exp, str(s), 'log', split, f'{q}.log')
    if not os.path.exists(f):
        return np.zeros((0, 2))
    a = np.loadtxt(f, delimiter=',', ndmin=2)
    return a if a.size else np.zeros((0, 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    ps.FIGS = OUT
    ps.setup()
    plt = ps.plt
    summ = {}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for i, (stage, title) in enumerate((('mse', 'MSE 期間'), ('l1', 'L1 期間'))):
        for j, (split, q) in enumerate(SERIES):
            ax = axes[i, j]
            for c, mse, l1, col, mk, lab in RUNS:
                for s in range(3):
                    a = read(mse if stage == 'mse' else l1.format(s=s), s, split, q)
                    if not len(a):
                        continue
                    ax.plot(a[:, 0], a[:, 1], color=col, lw=0.8, ls=('-', '--', ':')[s], label=lab if s == 0 else None)
                    v = a[:, 1]
                    jump = [int(a[k, 0]) for k in range(10, len(v)) if v[k] > 1.2 * v[k - 10]] if split == 'train' else []
                    summ[f'{c}_s{s}_{stage}_{split}_{q}'] = {'n': int(len(v)), 'last_epoch': int(a[-1, 0]), 'last': float(v[-1]),
                                                             'min': float(np.nanmin(v)), 'nan': int(np.isnan(v).sum()),
                                                             'jumps_20pct_over_10ep': jump[:20]}
            ax.set_title(f'{title}：{"学習" if split == "train" else "検証"} {"自分の視界" if q == "self_vision" else "他者の特徴予測"}', fontsize=8)
            ax.set_xlabel('エポック')
            if stage == 'mse' and split == 'train':
                ax.set_yscale('log')
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    fig.text(0.99, -0.03, '組1 実線、組2 破線、組3 点線。値は各エポックの損失（runner の合計）', ha='right', fontsize=7)
    fig.tight_layout()
    ps.save(fig, 'curves')
    with open(os.path.join(OUT, 'curves.json'), 'w') as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: (v['last_epoch'], round(v['last'], 3), v['nan'], v['jumps_20pct_over_10ep'][:3]) for k, v in summ.items()
                      if 'train_self' in k}, indent=0))


if __name__ == '__main__':
    main()
