"""
v11 M1(d) with red/green pixels excluded (docs/v11_experiment_log.md Sec.7.14).
Same frames / counterfactual views as eval_m1_v11.m1d. Pixel L1 is computed
only over pixels that are neither red nor green (Sec.5.0 thresholds) in the
true next view v_{t+1} nor in the counterfactual view v_cf^X; 'dil1' also
drops a 1-pixel 4-neighbourhood around them. Variant 'none' reproduces M1(d).
Also reports the colour criterion (Sec.7.13): swap episodes, S_Red - S_Green.
Reads only; writes data/result/v11/trial_m1d/m1d_masked_<tag>.json.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import DATA, LAYNAME, RES, boot_ci, load_model, run  # noqa
from eval_m1_v11 import COLORS, refs, summarize_ci  # noqa
from v11core.pleasure import color_mask  # noqa


def rg_mask(v):
    """v: (..., 16, 64, 3) uint8 -> bool (..., 16, 64) red or green."""
    return color_mask(v, 'Red') | color_mask(v, 'Green')


def dilate(m):
    d = m.copy()
    d[..., 1:, :] |= m[..., :-1, :]
    d[..., :-1, :] |= m[..., 1:, :]
    d[..., :, 1:] |= m[..., :, :-1]
    d[..., :, :-1] |= m[..., :, 1:]
    return d


def m1d_variants(d, src, cf, cf_valid, truth):
    pos = d['pos']
    m = pos[:, 1:101] - pos[:, :100]
    p = pos[:, :100]
    ok0 = (src[:, :100] != 3) & (np.linalg.norm(m, axis=-1) > 1e-6)
    pred = d['pred'].astype(np.float32)
    tr = truth.astype(np.float32) / 255.0
    rg_true = rg_mask(truth)
    out = {v: {} for v in ('none', 'rg', 'rg_dil1')}
    kept = {v: [] for v in ('rg', 'rg_dil1')}
    for k, X in enumerate(COLORS):
        u = refs(d['layout'], X) - p
        dist = np.linalg.norm(u, axis=-1)
        cosang = (m * u).sum(-1) / np.maximum(np.linalg.norm(m, axis=-1) * dist, 1e-9)
        sel = ok0 & cf_valid[:, :, k] & (cosang < np.cos(np.deg2rad(45)))
        cfx = cf[:, :, k]
        a_err = np.abs(pred - tr).mean(-1)                      # (N,100,16,64)
        a_cf = np.abs(pred - cfx.astype(np.float32) / 255.0).mean(-1)
        excl = rg_true | rg_mask(cfx)
        for v, keep in (('none', np.ones_like(excl)), ('rg', ~excl), ('rg_dil1', ~dilate(excl))):
            n = keep.sum((2, 3)).astype(np.float32)
            s = ((a_err - a_cf) * keep).sum((2, 3)) / np.maximum(n, 1)
            S = np.array([s[i][sel[i]].mean() if sel[i].any() else np.nan for i in range(len(p))])
            out[v][X] = S
            if v != 'none':
                kept[v].append(float((n[sel] / (16 * 64)).mean()))
    res = {}
    for v, S in out.items():
        c = S['Red'] - (S['Green'] + S['Blue'] + S['Cyan']) / 3
        swap = d['layout'] == 1
        orig = d['layout'] == 0
        res[v] = {'c_cf': c, 'S': S,
                  'colour_swap': (S['Red'] - S['Green'])[swap],
                  'colour_orig': (S['Red'] - S['Green'])[orig]}
    return res, {v: float(np.mean(x)) for v, x in kept.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = os.path.join(DATA, 'v11_s1_random', 'data.h5')
    g = h5py.File(rnd, 'r')['test']
    src = g['a1_source'][()]
    truth = g['self_vision'][:, 1:101]
    fcf = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_views.h5'), 'r')
    cf, cf_valid = fcf['cf_vision'][()], fcf['valid'][()]
    S, per = {}, {}
    for cond in ('random', 'mixed'):
        model, p_mask = load_model(args.exp_fmt.format(cond), args.seed, args.epoch, dev)
        d = run(model, p_mask, rnd, 'test', 0, dev, keep_pred='f16')
        res, kept = m1d_variants(d, src, cf, cf_valid, truth)
        per[cond] = res
        S[cond] = {'kept_pixel_frac': kept}
        for v, r in res.items():
            S[cond][v] = {'c_cf': summarize_ci(r['c_cf']),
                          'S': {X: summarize_ci(x) for X, x in r['S'].items()},
                          'colour_swap_Sred_minus_Sgreen': summarize_ci(r['colour_swap']),
                          'colour_orig_Sred_minus_Sgreen': summarize_ci(r['colour_orig'])}
        del d
    S['paired_mixed_minus_random'] = {v: summarize_ci(per['mixed'][v]['c_cf'] - per['random'][v]['c_cf'])
                                      for v in ('none', 'rg', 'rg_dil1')}
    tag = f'{args.exp_fmt.format("X")}_s{args.seed}_ep{args.epoch}'
    with open(os.path.join(RES, 'v11', 'trial_m1d', f'm1d_masked_{tag}.json'), 'w') as f:
        json.dump(S, f, indent=1)
    for v in ('none', 'rg', 'rg_dil1'):
        print(args.epoch, v, 'random', S['random'][v]['c_cf']['mean'], 'mixed', S['mixed'][v]['c_cf'],
              'diff', S['paired_mixed_minus_random'][v], 'colour_swap mixed', S['mixed'][v]['colour_swap_Sred_minus_Sgreen'])


if __name__ == '__main__':
    main()
