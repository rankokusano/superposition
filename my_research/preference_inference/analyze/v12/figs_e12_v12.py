"""
v12 E1/E2 figure (docs/v12_experiment_log.md Sec.53, docs/v12_summary.md): self/other code alignment, Fig.5 margin
and layout read from Encoder-2 for v4, trial (P-a), trial v12 body, E1 and E2 (mse ep100/150/200). Encoder measures
from data/result/v12/e12/*.json (same script for every row); Fig.5 margins of v4 / trials from the log (Sec.8), of
E1 / E2 from data/result/v12/e12/fig5_*. CPU. Writes data/result/v12/report/figs/e12_alignment.png.
"""

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = os.path.join(_PI, 'data', 'result', 'v12', 'e12')
EPS = (100, 150, 200)


def enc(name):
    j = json.load(open(os.path.join(E, name + '.json')))
    return j['alignment']['cos_ov(A1)_sv(A2)'], j['v11_s1_random']['layout_from_ov_enc']


def fig5(t, e):
    j = json.load(open(glob.glob(os.path.join(E, f'fig5_{t}_mse{e}', '*.json'))[0]))
    return j['original']['margin_other_self_minus_other_other']


def main():
    refs = {'v4 (frozen in v12 main)': (enc('check_v4_ep200'), 0.091, 'k'),
            'trial (P-a), ep150, 1000 ep.': (enc('v12_trial_pa_random_mse_ep150'), 0.016, '#7f7f7f'),
            'trial v12 body, ep150, 1000 ep.': (enc('v12_trial_mixed_mse_ep150'), -0.025, '#bcbd22')}
    runs = {'E1: paper model, 3000 ep.': ('e1', 'v12_e1_paper_random_mse', '#1f77b4'),
            'E2: v12 model from scratch, 3000 ep.': ('e2', 'v12_e2_full_mixed_mse', '#d62728')}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, k, ttl in ((axes[0], 0, '(a) self/other code alignment cos'), (axes[1], 2, '(b) Fig.5 margin (original layout)'),
                       (axes[2], 1, '(c) layout read from Encoder-2 (common test; chance 0.5)')):
        for lab, ((al, lay), mg, col) in refs.items():
            v = (al, lay, mg)[k]
            ax.axhline(v, color=col, ls='--', lw=1.2, label=f'{lab}: {v:+.3f}' if k == 2 and lab.startswith('trial v12') else f'{lab}: {v:.3f}')
        for lab, (t, base, col) in runs.items():
            ys = []
            for e in EPS:
                al, lay = enc(f'{base}_ep{e}')
                ys.append((al, lay, fig5(t, e))[k])
            ax.plot(EPS, ys, 'o-', color=col, lw=2, label=lab)
        ax.set_xlabel('mse-stage epoch'); ax.set_title(ttl, fontsize=10); ax.set_xticks(EPS)
        ax.legend(fontsize=7, loc='upper center' if k == 1 else 'best', ncol=2 if k == 1 else 1)
    axes[0].set_ylim(0, 0.9); axes[1].set_ylim(-0.04, 0.11); axes[2].set_ylim(0.45, 1.12)
    fig.suptitle('E1 / E2 (post-hoc check): does self/other alignment grow with full-scale from-scratch training in the swapping environment?', fontsize=11)
    fig.tight_layout()
    out = os.path.join(_PI, 'data', 'result', 'v12', 'report', 'figs', 'e12_alignment.png')
    fig.savefig(out, dpi=120)
    print('saved', out)


if __name__ == '__main__':
    main()
