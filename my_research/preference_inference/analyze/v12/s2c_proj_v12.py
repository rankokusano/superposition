"""
v12 stage 2c post-hoc analysis (docs/v12_experiment_log.md Sec.51; NOT pre-registered, no training, judgements
unchanged). Kusano-san's reading of the reversed hidden-test T2: while A-2 is hidden, the experienced model may move
A-2 toward its OWN expected direction (A-1's liked colour, red) rather than A-2's true preference (green).
Check: run the stage-2c G models (setting 1, green-liker data; ep200) normally on the hidden test (network input =
self_vision_in, G generating), read A-2's position from the predicted images, and inside a hidden interval project
A-2's displacement (from its true position at the interval start) onto the directions of the red landmark (A-1's
liked colour) and the green landmark (A-2's liked colour); slope per step (k = 1..min(15, interval length)).
  * no-swap set (v12_hide_v11_s2t_green): the first hidden interval starting at t >= 20;
  * swap set (v12_hide_v11_s2t_green_swap): the episodes hidden at t = 50, window from t = 50 (directions to the NEW
    red / green after the swap).
The same projection of A-2's TRUE positions is given as the data reference. Read-outs: primary and B1 (background).
Writes data/result/v12/post/s2c_proj_s<s>.json.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
sys.path.insert(0, _HERE)
import eval_s2_v11 as E  # noqa
from eval_trial_v12 import RES, load_v12, summ  # noqa
from eval_g5_v12 import calibrate, calibrate_bg, read_a2, read_a2_bg  # noqa
from eval_m1_v11 import refs  # noqa
from eval_s2c_v12 import SETS_H  # noqa
import util  # noqa

KMAX = 15


@torch.no_grad()
def read_series(model, d, dev, ab, bgc, batch=50):
    """A-2 position read from the predicted image of every step (index t = position at t + 1)."""
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    util.seed_all(0)
    Es, EB = [], []
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision_in'][b0:b0 + batch]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b0 + batch]).float().to(dev)
        sp = d['self_position'][b0:b0 + batch]
        model.init_state(x.shape[0])
        e, eb = [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            out = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm)
            p01 = ((out['self_vision'] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            e.append(read_a2(p01, sp[:, t + 1], *ab)); eb.append(read_a2_bg(p01, sp[:, t + 1], *bgc))
        Es.append(np.stack(e, 1)); EB.append(np.stack(eb, 1))
    return np.concatenate(Es), np.concatenate(EB)   # (N, 100, 2): position at t + 1


def windows(H, kind):
    W = []
    for e in range(H.shape[0]):
        if kind == 'noswap':
            t = 20
            while t < 100 and not (H[e, t] and (t == 0 or not H[e, t - 1])):
                t += 1
            if t >= 95:
                continue
        else:
            if not H[e, 50]:
                continue
            t = 50
        L = 0
        while t + L < 100 and H[e, t + L] and L < KMAX:
            L += 1
        if L >= 5:
            W.append((e, t, L))
    return W


def slopes(pos_at, d, W):
    """pos_at(e, t) -> (2,) A-2 position at time t (NaN if unread). Slope per step of the projection on red / green."""
    out = {'Red': [], 'Green': []}
    for e, t0, L in W:
        p0 = d['other_position'][e, t0]
        lay = d['layout_t'][e, t0]
        for X in ('Red', 'Green'):
            u = refs([lay], X)[0, 0] - p0
            u = u / max(np.linalg.norm(u), 1e-9)
            ks = np.arange(1, L + 1)
            pr = np.array([np.dot(pos_at(e, t0 + k) - p0, u) for k in ks])
            ok = np.isfinite(pr)
            out[X].append(np.polyfit(ks[ok], pr[ok], 1)[0] if ok.sum() >= 5 else np.nan)
    return {X: np.array(v) for X, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epoch', type=int, default=200)
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: E.load_set(k) for k in ('green', 'swap')}
    gg = h5py.File(D['green']['path'], 'r')['test']
    ab = calibrate(gg['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    bgc = calibrate_bg(gg['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    res = {'seed': a.seed, 'epoch': a.epoch, 'KMAX': KMAX}
    per = {}
    for kind, k in (('noswap', 'green'), ('swap', 'swap')):
        d = D[k]
        H = h5py.File(d['path'], 'r')['test']['a2_hidden'][()]
        W = windows(H, kind)
        res[f'{kind}_n_windows'] = len(W)
        true = slopes(lambda e, t: d['other_position'][e, t], d, W)
        res[f'{kind}_true'] = {X: summ(v[np.isfinite(v)]) for X, v in true.items()}
        for c in ('random', 'mixed'):
            model, _ = load_v12(f'v12_s2c_green_{c}_s{a.seed}', a.seed, a.epoch, dev)
            est, estb = read_series(model, d, dev, ab, bgc)
            for ro, A in (('main', est), ('bg', estb)):
                sl = slopes(lambda e, t: A[e, t - 1], d, W)
                per[(kind, c, ro)] = sl
                res[f'{kind}_{c}_{ro}'] = {X: summ(v[np.isfinite(v)]) for X, v in sl.items()}
                dg = sl['Green'] - sl['Red']
                res[f'{kind}_{c}_{ro}']['green_minus_red'] = summ(dg[np.isfinite(dg)])
                res[f'{kind}_{c}_{ro}']['frac_read'] = float(np.isfinite(A[..., 0]).mean())
        for ro in ('main', 'bg'):
            pm, pr_ = per[(kind, 'mixed', ro)], per[(kind, 'random', ro)]
            res[f'{kind}_paired_{ro}'] = {}
            for X in ('Red', 'Green'):
                v = pm[X] - pr_[X]
                res[f'{kind}_paired_{ro}'][X] = summ(v[np.isfinite(v)])
            v = (pm['Green'] - pm['Red']) - (pr_['Green'] - pr_['Red'])
            res[f'{kind}_paired_{ro}']['green_minus_red'] = summ(v[np.isfinite(v)])
        print(kind, json.dumps({kk: {X: round(vv[X]['mean'], 3) for X in ('Red', 'Green')} for kk, vv in res.items()
                                if kk.startswith(kind) and isinstance(vv, dict) and 'Red' in vv}), flush=True)
    os.makedirs(os.path.join(RES, 'v12', 'post'), exist_ok=True)
    json.dump(res, open(os.path.join(RES, 'v12', 'post', f's2c_proj_s{a.seed}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
