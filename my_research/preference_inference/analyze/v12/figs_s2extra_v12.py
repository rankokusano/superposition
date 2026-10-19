"""
v12 stage-2 figures not covered by the v11 scripts (instructions Sec.5.3; docs/v12_experiment_log.md Sec.21):
  q2_vs_pa2_<tag>.png : generated Q2 against A-2's true pleasure P_A2 per frame and bin (v11_s2t_green),
                        bins 0,1,6,7 (colours visible) and 2-5 (P_A2 = 0) separately, with Pearson r;
  t4_example_<tag>.png: predicted images t = 49..60 of one episode with and without the colour swap at
                        t = 50, with the A-2 position read from each image (v11 Sec.7.53 read-out),
                        plus the per-episode T4 score distribution if the t4t6 per-episode file exists.
Usage: python analyze/v12/figs_s2extra_v12.py --exp_fmt v12_s2_green_{}_s0 --seed 0 --epoch 200 --tag green_s0
       [--t4_per data/result/v12/stage2/t4t6/green_{}_s0_ep200_per.npz] [--episode 0]
"""

import argparse
import os
import sys

import h5py
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
from p2test_v11 import calibrate, read_a2  # noqa
import util  # noqa

CC = {'random': 'tab:gray', 'mixed': 'tab:red'}


@torch.no_grad()
def run(model, d, dev, n=None, keep_pred=False, batch=50):
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0] if n is None else n
    util.seed_all(0)
    Q, PR = [], []
    for b0 in range(0, N, batch):
        B = min(batch, N - b0)
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + B]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b0 + B]).float().to(dev)
        model.init_state(B)
        q, pr = [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            out = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm)
            q.append(out['q2_hat'].cpu().numpy())
            if keep_pred:
                pr.append(((out['self_vision'] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
        Q.append(np.stack(q, 1))
        if keep_pred:
            PR.append(np.stack(pr, 1))
    return np.concatenate(Q), (np.concatenate(PR) if keep_pred else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default='stage2/figs')
    ap.add_argument('--t4_per', default=None, help='t4t6 per-episode npz with {} for the condition')
    ap.add_argument('--episode', type=int, default=0)
    args = ap.parse_args()
    out = os.path.join(RES, 'v12', args.out); os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    dg, ds = load_set('green'), load_set('swap')
    P = true_pleasure('green', dg)[:, 1:100]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    preds = {}
    for j, c in enumerate(('random', 'mixed')):
        model, _ = load_v12(args.exp_fmt.format(c), args.seed, args.epoch, dev)
        Q, _ = run(model, dg, dev)
        Q = Q[:, 1:100]
        for i, (bins, nm) in enumerate((([0, 1, 6, 7], 'bins 0,1,6,7 (colours visible)'), ([2, 3, 4, 5], 'bins 2-5 (P_A2 = 0)'))):
            a, b = P[..., bins].reshape(-1), Q[..., bins].reshape(-1)
            sel = np.random.default_rng(0).choice(len(a), min(20000, len(a)), replace=False)
            ax = axes[i, j]
            if a.std() > 0:
                ax.scatter(a[sel], b[sel], s=2, alpha=0.2, color=CC[c])
                r = np.corrcoef(a, b)[0, 1]
                ax.set_title(f'{c}: {nm}\nr = {r:+.3f} (all frames x bins)', fontsize=9)
                ax.set_xlabel('A-2 true pleasure P_A2'); ax.set_ylabel('generated Q2')
            else:   # P_A2 is always 0 in bins 2-5: show where Q2 goes instead
                ax.hist(b, bins=60, color=CC[c], alpha=0.7)
                ax.set_title(f'{c}: {nm}\nQ2 distribution (P_A2 is always 0 here); std {b.std():.3f}', fontsize=9)
                ax.set_xlabel('generated Q2'); ax.set_ylabel('frames x bins')
        preds[c] = {}
        for nm, d in (('noswap', dg), ('swap', ds)):
            _, pr = run(model, d, dev, n=args.episode + 1, keep_pred=True)
            preds[c][nm] = pr[args.episode]
    fig.suptitle(f'Generated Q2 vs A-2 true pleasure ({args.exp_fmt}, ep{args.epoch}, v11_s2t_green, t = 1..99)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, f'q2_vs_pa2_{args.tag}.png'), dpi=110); plt.close(fig)
    g = h5py.File(dg['path'], 'r')['test']
    alpha, beta = calibrate(g['self_vision'][:, :100:5], dg['self_position'][:, :100:5], dg['other_position'][:, :100:5])
    ts = [49, 50, 51, 53, 55, 57, 59]
    e = args.episode
    rows = [('truth no swap', None), ('truth swap', None)] + [(f'{c} {nm}', (c, nm)) for c in ('random', 'mixed') for nm in ('noswap', 'swap')]
    has_t4 = args.t4_per is not None and os.path.exists(args.t4_per.format('mixed'))
    fig, axes = plt.subplots(len(rows) + (1 if has_t4 else 0), len(ts), figsize=(2.2 * len(ts), 1.0 * len(rows) + (2.5 if has_t4 else 0)), squeeze=False)
    svs = {'noswap': h5py.File(dg['path'], 'r')['test']['self_vision'][e], 'swap': h5py.File(ds['path'], 'r')['test']['self_vision'][e]}
    for i, (lab, key) in enumerate(rows):
        for j, t in enumerate(ts):
            ax = axes[i, j]
            if key is None:
                img = svs['swap' if 'swap' in lab and 'no' not in lab else 'noswap'][t + 1]
            else:
                img = preds[key[0]][key[1]][t]
                est = read_a2(img[None], ds['self_position'][e:e + 1, t + 1], alpha, beta)[0]
                rel = est - ds['self_position'][e, t + 1]
                if np.all(np.isfinite(rel)):
                    col = (np.degrees(np.arctan2(rel[1], rel[0])) + 180) / 5.625 - 0.5
                    ax.axvline(col, color='yellow', lw=0.8)
            ax.imshow(img, aspect='auto'); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f'pred of v_{t + 1}' if True else '', fontsize=7)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
    if has_t4:
        gs = axes[-1, 0].get_gridspec()
        for a in axes[-1]:
            a.remove()
        ax = fig.add_subplot(gs[-1, :])
        for c in ('random', 'mixed'):
            v = np.load(args.t4_per.format(c))['gen_T4']
            ax.hist(v[np.isfinite(v)], bins=30, alpha=0.5, color=CC[c], label=f'{c} (mean {np.nanmean(v):+.3f})')
        ax.axvline(0, color='k', lw=0.8); ax.legend(fontsize=7)
        ax.set_xlabel('T4 score per episode: cos(D, new green) - cos(D, old green), swap minus no-swap')
    fig.suptitle(f'T4 example: episode {e}, colours swapped at t = 50 (yellow line = A-2 direction read from the image)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, f't4_example_{args.tag}.png'), dpi=110); plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
