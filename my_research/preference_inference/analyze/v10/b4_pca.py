"""
v10 B4 PCA state maps (Sec.5.6 item 6, 5.30): h1 of the v10_swapmix_r2 test
split at ep400 for B3-ft s0, B3-frozen s0, B3-frozen s1, R3-A_400 s0. PCA fitted
on all frames of both layouts; PC1-PC2 per layout, coloured by A-1's x, y and
distance to red. Input: test_v10 output dirs (state + self_position, `test`).
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR, git_hash  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

RED = {0: np.array([-9.0, 9.0]), 1: np.array([-9.0, -9.0])}
NAME = {0: 'original', 1: 'swapped'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--run_dirs', nargs='+', required=True)
    ap.add_argument('--epoch', type=int, default=400)
    args = ap.parse_args()
    with h5py.File(os.path.join(PI_DIR, 'data', 'data', 'v10_swapmix_r2', 'data.h5'), 'r') as f:
        lay = f['test/layout'][()].astype(int)
    out = {'git_commit': git_hash(), 'epoch': args.epoch, 'models': {}}
    nrow = len(args.labels)
    fig, axes = plt.subplots(nrow, 6, figsize=(24, 3.9 * nrow), constrained_layout=True)
    rng = np.random.RandomState(0)
    for r, (lab, rd) in enumerate(zip(args.labels, args.run_dirs)):
        with h5py.File(os.path.join(rd, 'save', 'saved.h5'), 'r') as f:
            g = f[f'{args.epoch:05d}/test']
            h1 = g['state/self/hidden'][()]
            sp = g['self_position/input'][()]
        N, T, D = h1.shape
        pca = PCA(n_components=4).fit(h1.reshape(-1, D))
        pcs = pca.transform(h1.reshape(-1, D)).reshape(N, T, -1)
        dred = np.linalg.norm(sp - np.stack([RED[c] for c in lay])[:, None, :], axis=2)
        out['models'][lab] = {'explained_variance_ratio': pca.explained_variance_ratio_.tolist()}
        for c in (0, 1):
            idx = np.where(lay == c)[0]
            P = pcs[idx].reshape(-1, 4)
            S = sp[idx].reshape(-1, 2)
            Dr = dred[idx].ravel()
            sel = rng.choice(len(P), min(8000, len(P)), replace=False)
            for k, (val, ttl, cm) in enumerate(((S[:, 0], 'self x', 'coolwarm'), (S[:, 1], 'self y', 'coolwarm'),
                                                (Dr, 'distance to red', 'viridis'))):
                ax = axes[r, 3 * c + k]
                sc = ax.scatter(P[sel, 0], P[sel, 1], c=val[sel], s=2, cmap=cm)
                ax.set_title(f'{lab}, {NAME[c]} episodes: {ttl}', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(sc, ax=ax, shrink=0.75)
        axes[r, 0].set_ylabel(f'{lab}\nPC2 (PC1 x)', fontsize=9)
    fig.suptitle('v10 B4: PCA of h1 (test split, ep400; PCA fitted on both layouts; PC1 horizontal, PC2 vertical)')
    d = os.path.join(OUT_ROOT, 'b4')
    fig.savefig(os.path.join(d, 'b4_pca_state_maps.png'), dpi=100)
    json.dump(out, open(os.path.join(d, 'b4_pca.json'), 'w'), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
