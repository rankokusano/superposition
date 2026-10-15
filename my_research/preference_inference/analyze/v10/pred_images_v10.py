"""
v10 prediction-image panels (docs/v10_instructions.md Sec.7.1) for one model
checkpoint, in both layouts, on the viz20 datasets (same episode numbers;
v10_swap_r2_viz20 episodes pair with r2_a1random_a2rl_viz20 episodes: same
A-1 path).

Rows (columns = timesteps t):
  self input (A-1 view at t) / self truth (t+1) / self prediction (t+1) / |error|
  other truth (A-2's real view at t) / other via Fig.5 path (Autoencoder:
  other-encoder on A-1's view at t -> decoder) / |error|

Inputs: test_v10.py outputs (saved.h5) of the shared-module model and of the
Fig.5 Autoencoder on each viz20 dataset. Writes PNG per (layout, episode),
one .npz with all arrays used, and a JSON of mean absolute errors.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import git_hash  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

TIMES = [1, 5, 10, 20, 50]


def load_pair(model_h5, ae_h5, data_h5, model_epoch, ae_epoch):
    with h5py.File(model_h5, 'r') as f:
        g = f[f'{model_epoch:05d}/eval/self_vision']
        sv_in, sv_truth, sv_pred = g['input'][()], g['truth'][()], g['prediction'][()]
    with h5py.File(ae_h5, 'r') as f:
        ov_rec = f[f'{ae_epoch:05d}/eval/other_vision/reconstruction'][()]
    with h5py.File(data_h5, 'r') as f:
        ov_true = f['train/other_vision'][()].astype(np.float32) / 255.0
    return dict(sv_in=sv_in, sv_truth=sv_truth, sv_pred=sv_pred, ov_rec=ov_rec, ov_true=ov_true)


def show(ax, img, title=None):
    ax.imshow(np.clip(img, 0, 1), aspect='auto', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--model_epoch', type=int, required=True)
    ap.add_argument('--ae_epoch', type=int, required=True)
    ap.add_argument('--episodes', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--layouts', nargs='+', default=['original', 'swap_rg'])
    ap.add_argument('--model_h5', nargs='+', required=True, help='one per layout')
    ap.add_argument('--ae_h5', nargs='+', required=True, help='one per layout')
    ap.add_argument('--data_h5', nargs='+', required=True, help='one per layout (viz20 data.h5)')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    arrays, summary = {}, {'label': args.label, 'model_epoch': args.model_epoch,
                           'ae_epoch': args.ae_epoch, 'git_commit': git_hash(), 'layouts': {}}
    rows = ['self input (t)', 'self truth (t+1)', 'self prediction (t+1)', '|self error|',
            'other truth: A-2 view (t)', 'other via Fig.5 path (t)', '|other error|']
    for lo, mh, ah, dh in zip(args.layouts, args.model_h5, args.ae_h5, args.data_h5):
        d = load_pair(mh, ah, dh, args.model_epoch, args.ae_epoch)
        se = np.abs(d['sv_pred'] - d['sv_truth'])
        oe = np.abs(d['ov_rec'] - d['ov_true'])
        summary['layouts'][lo] = {
            'self_pred_mean_abs_error_all20ep': float(se.mean()),
            'other_fig5_mean_abs_error_all20ep': float(oe.mean()),
            'model_h5': mh, 'ae_h5': ah, 'data_h5': dh,
        }
        for ep in args.episodes:
            fig, axes = plt.subplots(len(rows), len(TIMES), figsize=(3.2 * len(TIMES), 1.0 * len(rows) + 1.2),
                                     constrained_layout=True)
            for c, t in enumerate(TIMES):
                panels = [d['sv_in'][ep, t], d['sv_truth'][ep, t], d['sv_pred'][ep, t],
                          se[ep, t] / max(se[ep, t].max(), 1e-9),
                          d['ov_true'][ep, t], d['ov_rec'][ep, t], oe[ep, t] / max(oe[ep, t].max(), 1e-9)]
                for r, img in enumerate(panels):
                    show(axes[r, c], img, f't={t}' if r == 0 else None)
                    if c == 0:
                        axes[r, c].set_ylabel(rows[r], fontsize=8, rotation=0, ha='right', va='center')
            fig.suptitle(f'{args.label} | layout {lo} | viz20 episode {ep} | '
                         f'self |err| {se[ep].mean():.4f}, other(Fig.5) |err| {oe[ep].mean():.4f} '
                         f'(error rows normalised per panel)', fontsize=10)
            fig.savefig(os.path.join(args.out_dir, f'pred_images_{lo}_ep{ep}.png'), dpi=100)
            plt.close(fig)
            for k in ('sv_in', 'sv_truth', 'sv_pred', 'ov_true', 'ov_rec'):
                arrays[f'{lo}_ep{ep}_{k}'] = d[k][ep]
    np.savez_compressed(os.path.join(args.out_dir, 'pred_images_arrays.npz'), times=np.array(TIMES), **arrays)
    with open(os.path.join(args.out_dir, 'pred_images_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
