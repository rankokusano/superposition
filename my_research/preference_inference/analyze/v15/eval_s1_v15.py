"""
v15 stage-1 measures (docs/v15_experiment_log.md Sec.8.4), CPU:
  M1 self/other code alignment cos_ov(A1)_sv(A2) (fig5_extra_v11.cosm) on v15_s1_p80_3000 test frames
     (t = 0, 5, ..., 95), and the v12 Sec.8 value on the v11 common test (eval_trial_v12.alignment) for reference;
  M3 each corner's state (TL, BL in {Red, Green, Gray}; BR in {Blue, Gray}; TR in {Cyan, Gray}) read linearly
     from Encoder-2 (and Encoder-1) of A-1's view (enc_colour_v12.acc_cv: scaler + logistic regression,
     5-fold grouped by episode), with the majority-class rate;
  M4 colour separation of the Fig.5 view D(E2(v_A1)) against A-2's true view and of the self reconstruction
     D(E1(v_A1)) against A-1's view: sep_RG (fig5_colour_check_v12.sep) and sep_BC = mean(B - G | true blue)
     - mean(B - G | true cyan); plus an example image (needs --ae).

    python analyze/v15/eval_s1_v15.py --tag v15_plain_mse200 --ckpt data/result/v15_s1_plain_mse/0/model/00200.pth \
        --ae data/result/v15_s1_plain_fig5_mse200/0/model/00100.pth
Writes data/result/v15/stage1/<tag>.json and <tag>_examples.png.
"""

import argparse
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import torch  # noqa

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
sys.path.insert(0, os.path.join(_PI_DIR, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI_DIR, 'analyze', 'v11'))
from fig5_extra_v11 import make_ae, enc, cosm  # noqa
from enc_colour_v12 import acc_cv  # noqa
from eval_trial_v12 import alignment  # noqa
from fig5_colour_check_v12 import rec, sep  # noqa

DATA = os.path.join(_PI_DIR, 'data', 'data', 'v15_s1_p80_half_3000', 'data.h5')  # log Sec.12 (gray in half of the episodes)
OUT = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'stage1')
TS = np.arange(0, 100, 5)
# colour code (0 Red, 1 Green, 2 Blue, 3 Cyan) -> corner, per layout code
COL_CORNER = {0: ('TL', 'BL', 'BR', 'TR'), 1: ('BL', 'TL', 'BR', 'TR')}
STATE_NAMES = {'TL': ('Red', 'Green', 'Gray'), 'BL': ('Red', 'Green', 'Gray'), 'BR': ('Blue', 'Gray'), 'TR': ('Cyan', 'Gray')}


def load_frames(path=DATA, split='test', n_max=None):
    g = h5py.File(path, 'r')[split]
    N = g['layout'].shape[0] if n_max is None else n_max
    sv = g['self_vision'][:N][:, TS].reshape(-1, 16, 64, 3)
    ov = g['other_vision'][:N][:, TS].reshape(-1, 16, 64, 3)
    av = g['available'][:N][:, TS].reshape(-1, 4)
    lay = np.repeat(g['layout'][:N][:, None], len(TS), 1).reshape(-1)
    grp = np.repeat(np.arange(N)[:, None], len(TS), 1).reshape(-1)
    states = {c: np.zeros(len(lay), int) for c in STATE_NAMES}
    for i in range(len(lay)):
        for c in range(4):
            corner = COL_CORNER[lay[i]][c]
            name = ('Red', 'Green', 'Blue', 'Cyan')[c] if av[i, c] else 'Gray'
            states[corner][i] = STATE_NAMES[corner].index(name)
    return sv, ov, lay, grp, states


def sep_bc(recon, true):
    t = true.astype(np.int16)
    blue = (t[..., 2] > 150) & (t[..., 0] < 80) & (t[..., 1] < 80)
    cyan = (t[..., 1] > 150) & (t[..., 2] > 150) & (t[..., 0] < 80)
    d = recon[..., 2] - recon[..., 1]
    return float(d[blue].mean() - d[cyan].mean())


def examples(m, sv, ov, lay, states, path, title):
    # one frame per layout with a gray left corner, and one with both left corners coloured
    picks = []
    for L in (0, 1):
        for want_gray in (True, False):
            ok = np.where((lay == L) & (((states['TL'] == 2) | (states['BL'] == 2)) == want_gray))[0]
            if len(ok):
                picks.append(ok[len(ok) // 2])
    rs, ro = rec(m, sv[picks]), rec(m, sv[picks], other=True)
    fig, ax = plt.subplots(len(picks), 4, figsize=(14, 1.3 * len(picks) + 0.6))
    for i, j in enumerate(picks):
        for k, (img, lab) in enumerate(((sv[j], "A-1 view (input)"), (rs[i], 'self rec D(E1)'),
                                        (ov[j], "A-2 true view"), (ro[i], 'Fig.5 view D(E2)'))):
            ax[i, k].imshow(img if img.dtype == np.uint8 else (img * 255).astype(np.uint8), interpolation='nearest')
            ax[i, k].set_xticks([]); ax[i, k].set_yticks([])
            if i == 0:
                ax[i, k].set_title(lab, fontsize=8)
        ax[i, 0].set_ylabel(f"lay {lay[j]}\n" + ' '.join(f"{c}:{ {'Red': 'R', 'Green': 'G', 'Gray': '-'}[STATE_NAMES[c][states[c][j]]]}" for c in ('TL', 'BL')),
                            fontsize=7)
    fig.suptitle(title + '  (images shown with channel 0 as red, as stored)', fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ae', default=None)
    ap.add_argument('--data', default=DATA, help='(script checks only) another v15 dataset')
    ap.add_argument('--split', default='test')
    ap.add_argument('--n_max', type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    sv, ov, lay, grp, states = load_frames(a.data, a.split, a.n_max)
    m, _ = make_ae(a.ckpt, a.ae)
    E1, E2 = m.self_vision_encoder_module, m.other_vision_encoder_module
    r = {'tag': a.tag, 'ckpt': a.ckpt, 'ae': a.ae, 'data': a.data, 'split': a.split, 'n_frames': int(len(sv))}
    r['M1_alignment_v15_test'] = cosm(enc(E2, sv), enc(E1, ov))
    r['M1_alignment_v11_common_test'] = alignment(m)
    S1, S2 = enc(E1, sv), enc(E2, sv)
    r['M3'] = {}
    for corner, y in states.items():
        r['M3'][corner] = {'enc2': acc_cv(S2, y, grp), 'enc1': acc_cv(S1, y, grp),
                           'majority': float(np.bincount(y).max() / len(y)),
                           'counts': {STATE_NAMES[corner][k]: int(v) for k, v in enumerate(np.bincount(y, minlength=len(STATE_NAMES[corner])))}}
    r['M3_enc2_min'] = min(v['enc2'] for v in r['M3'].values())
    if a.ae:
        ro, rs = rec(m, sv, other=True), rec(m, sv)
        r['M4'] = {'other_sepRG': sep(ro, ov)[0], 'other_sepBC': sep_bc(ro, ov),
                   'self_sepRG': sep(rs, sv)[0], 'self_sepBC': sep_bc(rs, sv),
                   'true_sepRG': sep(ov.astype(np.float32) / 255, ov)[0], 'true_sepBC': sep_bc(ov.astype(np.float32) / 255, ov)}
        examples(m, sv, ov, lay, states, os.path.join(OUT, f'{a.tag}_examples.png'), a.tag)
    json.dump(r, open(os.path.join(OUT, f'{a.tag}.json'), 'w'), indent=1)
    print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
