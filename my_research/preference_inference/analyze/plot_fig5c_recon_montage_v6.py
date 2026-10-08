"""
Fig.5c-equivalent reconstruction montage for v6 (fig5_v6_base), in the same
format as data/result/baseline_v4/fig5c_recon_montage_{ep10,ep100}.png
(generated inline 2026-09-09; code recovered from that session and reproduced
here): same 4 (A-1 idx, A-2 idx) grid sample pairs, same columns
(A-1 true view / self_rec via Enc-1 / A-2 true view / other_rec via Enc-2),
same index convention (recon[oxi,oyi,sxi,syi] on the 21x21x21x21 grid).

Rows per sample: paper exp2 ep10 (read-only, protected asset) and v6. The
v4/v5 grid saved.h5 were deleted after their montage was made, so their rows
cannot be redrawn here; compare against the existing baseline_v4 montage,
which uses the identical sample positions.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/plot_fig5c_recon_montage_v6.py --epoch 10
    python analyze/plot_fig5c_recon_montage_v6.py --epoch 100
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRID = 'data/data/grid/data.h5'
PAPER = '/work/data/result/exp2/0/test/grid/save/saved.h5'
V6 = 'data/result/fig5_v6_base/0/test/grid_recon/save/saved.h5'
SAVE_DIR = 'data/result/v6_baseline'
SAMPLES = [(4, 10, 15, 5), (10, 4, 5, 15), (15, 15, 4, 4), (8, 12, 12, 8)]
COLS = ['A-1 true view', 'self_rec (Enc-1 path)', 'A-2 true view', 'other_rec (Enc-2 path)']
N = 21


def clip(a):
    a = np.asarray(a, dtype=float)
    if a.max() > 1.5:
        a = a / 255.
    return np.clip(a, 0, 1)


def recon_pair(h5, ep, sxi, syi, oxi, oyi):
    # original: arr.reshape(21,21,21,21,...)[oxi,oyi,sxi,syi] == arr[oxi*21+oyi, sxi*21+syi]
    i, j = oxi * N + oyi, sxi * N + syi
    with h5py.File(h5, 'r') as f:
        g = f[f'{ep:05d}/eval']
        return g['self_vision/reconstruction'][i, j], g['other_vision/reconstruction'][i, j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epoch', type=int, required=True, help='v6 Autoencoder epoch (paper stays ep10)')
    args = ap.parse_args()

    # original: true_img = st_na.reshape(21,21,21,21,...)[0,0]  -> data[0, sxi*21+syi]
    with h5py.File(GRID, 'r') as f:
        true_row = f['train/self_vision_no_agent'][0]  # (441,16,64,3)
    true_img = true_row.reshape(N, N, *true_row.shape[1:])

    conds = [('paper exp2', PAPER, 10), ('v6 (scratch Enc, value input)', V6, args.epoch)]
    nrows = len(SAMPLES) * len(conds)
    fig, ax = plt.subplots(nrows, 4, figsize=(13, 1.4 * nrows))
    r = 0
    for (sxi, syi, oxi, oyi) in SAMPLES:
        a1_true = clip(true_img[sxi, syi])
        a2_true = clip(true_img[oxi, oyi])
        for name, h5, ep in conds:
            srec, orec = recon_pair(h5, ep, sxi, syi, oxi, oyi)
            for c, p in enumerate([a1_true, clip(srec), a2_true, clip(orec)]):
                ax[r, c].imshow(p)
                ax[r, c].axis('off')
                if r == 0:
                    ax[r, c].set_title(COLS[c], fontsize=10)
            ax[r, 0].text(-0.4, 0.5, f'A1({sxi},{syi}) A2({oxi},{oyi})\n{name} ep{ep}',
                          transform=ax[r, 0].transAxes, fontsize=7, va='center', ha='right')
            r += 1
    fig.suptitle(f'Fig.5c-equivalent reconstruction (v6 ep{args.epoch} vs paper exp2 ep10); '
                 f'v4/v5 rows: see baseline_v4/fig5c_recon_montage_ep{args.epoch}.png (same samples)',
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    out = os.path.join(SAVE_DIR, f'fig5c_recon_montage_v6_ep{args.epoch}.png')
    fig.savefig(out, dpi=110)
    print('saved', out)


if __name__ == '__main__':
    main()
