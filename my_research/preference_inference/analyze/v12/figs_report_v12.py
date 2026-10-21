"""
Figures for the v11/v12 review report (docs/v12_experiment_log.md Sec.42), CPU only, from saved checkpoints/outputs:
  q2_timeseries.png : generated Q2 (bins 0,1,6,7 and bins 2-5) and A-2's true pleasure P_A2 over time for three
                      v11_s2t_green episodes, stage 2 (formal) and stage 2b, pair s0, ep200, both conditions;
  t2t5_ckpts.png    : T2 (setting 1) and T5 (setting 2) per G checkpoint, every pair, formal and stage 2b;
  t4_dist.png       : (B4) per-episode T4 score (swap minus no-swap) per pair and condition, formal and stage 2b.
Writes data/result/v12/report/figs/.
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import torch  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v12 import RES, load_v12  # noqa
from eval_s2_v11 import load_set, true_pleasure  # noqa
import util  # noqa

OUT = os.path.join(RES, 'v12', 'report', 'figs')
CC = {'random': 'tab:gray', 'mixed': 'tab:red'}
EPS = [160, 170, 180, 190, 200]


@torch.no_grad()
def q2_series(exp, seed, d, n):
    m, _ = load_v12(exp, seed, 200, torch.device('cpu'))
    import h5py
    g = h5py.File(d['path'], 'r')[d['split']]
    x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][:n])))
    mo = torch.from_numpy(g['self_motion'][:n]).float()
    util.seed_all(0)
    m.init_state(n)
    q = []
    for t in range(100):
        pm = 0 if t == 0 else 1
        q.append(m({'self_vision': x[:, t], 'self_motion': mo[:, t]}, pm, pm)['q2_hat'].numpy())
    return np.stack(q, 1)


def fig_q2ts():
    torch.set_num_threads(8)
    d = load_set('green')
    P = true_pleasure('green', d)
    n = 3
    fams = [('v12_s2', 'formal stage 2'), ('v12_s2b', 'stage 2b (shaped)')]
    fig, axes = plt.subplots(len(fams) * 2, n, figsize=(5 * n, 10), sharex=True)
    for i, (pre, lab) in enumerate(fams):
        for j, c in enumerate(('random', 'mixed')):
            Q = q2_series(f'{pre}_green_{c}_s0', 0, d, n)
            for e in range(n):
                ax = axes[2 * i + j, e]
                for k, col in zip((0, 1, 6, 7), ('#1b9e77', '#66c2a5', '#d95f02', '#fc8d62')):
                    ax.plot(Q[e, :, k], color=col, lw=1.3, label=f'Q2 bin {k}')
                    ax.plot(P[e, :100, k], color=col, lw=1, ls=':', label=f'P_A2 bin {k}')
                if pre == 'v12_s2':
                    ax.plot(Q[e, :, 2:6].mean(1), color='k', lw=0.8, alpha=0.6, label='Q2 bins 2-5 (mean)')
                ax.axhline(0, color='0.7', lw=0.5); ax.set_ylim(-1.05, 1.05)
                lay = 'original' if d['layout'][e] == 0 else 'swapped'
                ax.set_title(f'{lab}, {c}, episode {e} ({lay})', fontsize=8)
                if e == 0:
                    ax.set_ylabel('value')
    axes[0, 0].legend(fontsize=6, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel('t')
    fig.suptitle('Generated Q2 (solid) vs A-2 true pleasure P_A2 (dotted), v11_s2t_green, pair s0, ep200', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'q2_timeseries.png'), dpi=110); plt.close(fig)


def fig_t2t5():
    E = {'formal stage 2': os.path.join(RES, 'v12', 'stage2', 'eval'), 'stage 2b': os.path.join(RES, 'v12', 'stage2b', 'eval')}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for i, (lab, root) in enumerate(E.items()):
        for j, (st, key, ttl) in enumerate((('green', 'sT2_t50', 'T2 s_T2 (setting 1)'), ('pref', 'dT5', 'T5 d_T5 (setting 2)'))):
            ax = axes[i, j]
            for s, mk in zip((0, 1, 2, 3), 'os^v'):
                ok = all(os.path.exists(os.path.join(root, f'{st}_{c}_s{s}_ep{e}_per.npz')) for c in ('random', 'mixed') for e in EPS)
                if not ok:
                    continue
                for c in ('random', 'mixed'):
                    y = [np.load(os.path.join(root, f'{st}_{c}_s{s}_ep{e}_per.npz'))[key].mean() for e in EPS]
                    ax.plot(EPS, y, marker=mk, color=CC[c], ls='-' if c == 'mixed' else '--', label=f'{c} s{s}')
            ax.axhline(0, color='k', lw=0.8)
            ax.set_title(f'{lab}: {ttl} (D6 = {0.60 if key == "sT2_t50" else 0.42:+.2f}, off scale)', fontsize=9)
            ax.set_xlabel('G checkpoint'); ax.legend(fontsize=6, ncol=4)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 't2t5_ckpts.png'), dpi=110); plt.close(fig)


def fig_t4():
    E = {'formal stage 2': os.path.join(RES, 'v12', 'stage2', 't4t6'), 'stage 2b': os.path.join(RES, 'v12', 'stage2b', 't4t6')}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (lab, root) in zip(axes, E.items()):
        data, labels, cols = [], [], []
        for s in (0, 1, 2, 3):
            for c in ('random', 'mixed'):
                f = os.path.join(root, f'green_{c}_s{s}_ep200_per.npz')
                if os.path.exists(f):
                    v = np.load(f)['gen_T4']
                    data.append(v[np.isfinite(v)]); labels.append(f'{c[:3]} s{s}'); cols.append(CC[c])
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
            for p, c in zip(bp['boxes'], cols):
                p.set_facecolor(c); p.set_alpha(0.5)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title(f'{lab}: T4 per episode (setting-1 models, ep200)\ncos(D, new green) - cos(D, old green), swap minus no-swap', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 't4_dist.png'), dpi=110); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_t2t5()
    fig_t4()
    fig_q2ts()
    print('saved', OUT)


if __name__ == '__main__':
    main()
