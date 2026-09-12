"""
Fig.4c-equivalent state map (2026-09-08, health check for v5 base stage).

PCA the process-1 hidden state h1 and process-2 hidden state h2 to 2D, colour
each point by A-1 position (for h1) and A-2 position (for h2). The original
paper's Fig.4c shows h1/h2 organised into a spatial ("social place cell")
layout; this checks whether v5's from-scratch Encoder still produces that.

Reads the harness saved.h5 (eval/state/{self,other}/hidden, eval/{self,other}_position/input).

Usage (from /work/my_research/preference_inference):
  python analyze/plot_pca_state_v4.py \
     --h5 data/result/v5_base_mse/0/test/r3_stay_canon_es0/save/saved.h5 --epoch 400 --label v5_r3stay
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'data/result/baseline_v4'


def pca2(x):
    x = x - x.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T, (s[:2] ** 2 / (s ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--n_ep', type=int, default=800)
    ap.add_argument('--stride', type=int, default=5)  # subsample timesteps
    args = ap.parse_args()

    g = f'{args.epoch:05d}/eval'
    with h5py.File(args.h5, 'r') as f:
        h1 = f[f'{g}/state/self/hidden'][:args.n_ep, ::args.stride]
        h2 = f[f'{g}/state/other/hidden'][:args.n_ep, ::args.stride]
        sp = f[f'{g}/self_position/input'][:args.n_ep, ::args.stride]
        op = f[f'{g}/other_position/input'][:args.n_ep, ::args.stride]

    h1 = h1.reshape(-1, h1.shape[-1])
    h2 = h2.reshape(-1, h2.shape[-1])
    sp = sp.reshape(-1, 2)
    op = op.reshape(-1, 2)

    p1, ev1 = pca2(h1)
    p2, ev2 = pca2(h2)

    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for col, (P, ev, pos, who) in enumerate([
        (p1, ev1, sp, 'h1 (process-1)  coloured by A-1 pos'),
        (p2, ev2, op, 'h2 (process-2)  coloured by A-2 pos'),
    ]):
        for row, (ci, cname) in enumerate([(0, 'pos-x'), (1, 'pos-y')]):
            a = ax[row, col]
            sc = a.scatter(P[:, 0], P[:, 1], c=pos[:, ci], s=3, cmap='coolwarm',
                           alpha=0.5, rasterized=True)
            a.set_title(f'{who}\nPC1,2 var={ev[0]:.2f},{ev[1]:.2f}  colour={cname}',
                        fontsize=10)
            a.set_xlabel('PC1'); a.set_ylabel('PC2')
            plt.colorbar(sc, ax=a)
    fig.suptitle(f'{args.label}  (epoch {args.epoch})', fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT, f'pca_state_{args.label}.png')
    fig.savefig(out, dpi=110)
    print('saved', out)

    # quick quant: linear R^2 of the 2 PCs vs position (how spatial is the map)
    from numpy.linalg import lstsq
    for P, pos, name in [(p1, sp, 'h1->A1pos'), (p2, op, 'h2->A2pos'),
                         (p2, sp, 'h2->A1pos(cross)')]:
        X = np.c_[P, np.ones(len(P))]
        r2s = []
        for ci in range(2):
            beta, *_ = lstsq(X, pos[:, ci], rcond=None)
            resid = pos[:, ci] - X @ beta
            r2s.append(1 - resid.var() / pos[:, ci].var())
        print(f'  {name}: PC-plane R^2 (x,y) = {r2s[0]:.3f}, {r2s[1]:.3f}')


if __name__ == '__main__':
    main()
