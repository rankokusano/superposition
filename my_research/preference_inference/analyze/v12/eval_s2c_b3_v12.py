"""
v12 stage 2c priority secondary B3 on the HIDDEN test (docs/v12_experiment_log.md Sec.46): the G5a measure with the
GENERATED Q2 (analyze/v12/s2_post_v12.py, Sec.41): run normally (network input = self_vision_in) up to t = 20, take
G's Q2 at t = 20 and keep it for K = 15 steps (process-2 = [0, 0, Q2(20), 1]) vs keep generating; g_other = slope
toward the liked colour minus the mean of the other three, from A-2's position read off the predicted images
(primary read-out and the B1 background read-out; calibrated on the true views as in Sec.41).
Sets: v12_hide_v11_s2t_green (1: green-liker) and v12_hide_v11_s2_pref test (2: per-episode preference).
Main subset: episodes with A-2 HIDDEN at t = 20; reference: VISIBLE at t = 20, and all episodes.
Writes data/result/v12/<out>/b3/<setting>_s<s>.json (per condition / set / epoch / subset, and mixed - random paired).
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
from eval_g5_v12 import g5a_metric, calibrate, calibrate_bg, read_a2, read_a2_bg, T0  # noqa
from eval_s2c_v12 import SETS_H  # noqa  (also points E.SETS at the hidden sets)
import util  # noqa

K = 15


@torch.no_grad()
def g5a_generated_in(model, d, dev, ab, bgc, freeze, key='self_vision_in', batch=50):
    """s2_post_v12.g5a_generated with the input view from <key>."""
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    util.seed_all(0)
    Es, EB = [], []
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g[key][b0:b0 + batch]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b0 + batch]).float().to(dev)
        sp = d['self_position'][b0:b0 + batch]
        B = x.shape[0]
        model.init_state(B)
        e, eb, qhold = [], [], None
        for t in range(T0 + K):
            pm = 0 if t == 0 else 1
            ov = None
            if freeze and t > T0:
                ov = torch.cat([torch.zeros(B, 2, device=dev), qhold, torch.ones(B, 1, device=dev)], 1)
            out = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=ov)
            if t == T0:
                qhold = out['q2_hat'].detach()
            if t >= T0:
                p01 = ((out['self_vision'] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                e.append(read_a2(p01, sp[:, t + 1], *ab))
                eb.append(read_a2_bg(p01, sp[:, t + 1], *bgc))
        Es.append(np.stack(e, 1)); EB.append(np.stack(eb, 1))
    return np.concatenate(Es), np.concatenate(EB)


def summ_sub(v, m):
    v = v[m]
    return summ(v[np.isfinite(v)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='v12_s2c')
    ap.add_argument('--out', default='stage2c')
    ap.add_argument('--setting', required=True, help='green or pref (the G training data)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--input_key', default='self_vision_in')
    a = ap.parse_args()
    out = os.path.join(RES, 'v12', a.out, 'b3'); os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: E.load_set(k) for k in ('green', 'pref')}
    gg = h5py.File(D['green']['path'], 'r')['test']
    # read-out calibration on the TRUE views (A-2 visible), as Sec.41
    ab = calibrate(gg['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    bgc = calibrate_bg(gg['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    Hs = {k: h5py.File(D[k]['path'], 'r')[D[k]['split']]['a2_hidden'][:, T0] == 1 for k in D}
    subs = lambda k: {'hidden_t20': Hs[k], 'visible_t20': ~Hs[k], 'all': np.ones_like(Hs[k])}  # noqa
    res = {'prefix': a.prefix, 'setting': a.setting, 'seed': a.seed, 'input_key': a.input_key,
           'n_hidden_t20': {k: int(Hs[k].sum()) for k in D}}
    per = {}
    for c in ('random', 'mixed'):
        for e in a.epochs:
            model, _ = load_v12(f'{a.prefix}_{a.setting}_{c}_s{a.seed}', a.seed, e, dev)
            for k, d in D.items():
                for freeze in (True, False):
                    tag = 'freeze' if freeze else 'continue'
                    est, estb = g5a_generated_in(model, d, dev, ab, bgc, freeze, a.input_key)
                    gm, _, _ = g5a_metric(est, d)
                    gb, _, _ = g5a_metric(estb, d)
                    per[(c, e, k, tag)] = (gm, gb)
                    res.setdefault(f'{c}_ep{e}', {})[f'{k}_{tag}'] = {
                        sn: {'g_other': summ_sub(gm, sm), 'g_other_bg': summ_sub(gb, sm)} for sn, sm in subs(k).items()}
            print(c, e, json.dumps({kk: v['hidden_t20']['g_other']['mean'] for kk, v in res[f'{c}_ep{e}'].items()}), flush=True)
    for e in a.epochs:
        for k in D:
            for tag in ('freeze', 'continue'):
                dm = per[('mixed', e, k, tag)][0] - per[('random', e, k, tag)][0]
                db = per[('mixed', e, k, tag)][1] - per[('random', e, k, tag)][1]
                res.setdefault(f'paired_ep{e}', {})[f'{k}_{tag}'] = {
                    sn: {'g_other': summ_sub(dm, sm), 'g_other_bg': summ_sub(db, sm)} for sn, sm in subs(k).items()}
    json.dump(res, open(os.path.join(out, f'{a.setting}_s{a.seed}.json'), 'w'), indent=1)
    print('saved', os.path.join(out, f'{a.setting}_s{a.seed}.json'))


if __name__ == '__main__':
    main()
