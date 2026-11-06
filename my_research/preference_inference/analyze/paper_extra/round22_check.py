"""
Paper round 22 Sec.1 checks (paper/check/paper_instructions_round22.md; docs/paper_extra_log.md Sec.46; exploratory,
outside the pre-registration). CPU, model independent (pleasure = model.compute_pleasure with scale 1.0 for every
condition = v11core.pleasure.pleasure_torch on the scaled view, liked Red / disliked Green).
  1. the stored round-21 results give Delta for both pathways directly: every Round21/eval/<cond>_s<s>_ep<e>.npz
     (p50 / none / sr50 / p80, pairs 0-5, ep160-200) has '<swap|noswap>_<V1|V0>_<self|other>_<hold|flip|zero>' of
     shape (600,); p80 (experience 80%) is therefore already evaluated with the round-21 flow.
  2. (GPU, round22_eval.py check) the sign-random 80% predictor runs with the round-21 flow.
  3. size of the pleasure put into the other pathway (A-2's view, frame T0-1) vs into the self pathway (A-1's view,
     frame T0-1), per range (8 ranges): mean |P| over episodes x ranges, max |P|, mean over nonzero ranges, for V1
     (T0 = 23) and V0 (T0 = 20), swap and noswap. Also (descriptive, fixes the Sec.2.3 inputs):
       X1 clipping fractions of 2x / 4x the other's pleasure clipped to [-1, 1];
       X2 factor = mean |P_self| / mean |P_other| (pooled over swap and noswap, per flow) and its clipping fraction;
       X3 range of the direction from A-2 to L_old / L_new (argmax cos(angle - BIN_ANGLES)) and how often the largest
       held pleasure of A-2's view falls in that range (noswap: L_old, swap: L_new; V1);
       Sec.2.4 reference positions: fraction of (episode, target) with A-2 moved d = -5..5 steps toward L inside the
       arena (|q| <= 10).
-> data/result/paper_extra/Round22/check/check.json
"""

import json
import os
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
from sim_v10 import layout as L  # noqa
from v11core.pleasure import pleasure_torch, BIN_ANGLES  # noqa
import util  # noqa

RES = os.path.join(_PI, 'data', 'result')
EV21 = os.path.join(RES, 'paper_extra', 'Round21', 'eval')
DATA = os.path.join(_PI, 'data', 'data', 'paper_r21_swap')
OUT = os.path.join(RES, 'paper_extra', 'Round22', 'check')
CONDS = ('p50', 'none', 'sr50', 'p80')
EPS = (160, 170, 180, 190, 200)
FLOWS = (('V1', 23), ('V0', 20))
DREF = np.arange(-5, 6)


def pleas(v):
    """v uint8 (N, H, W, 3) -> (N, 8) float64, exactly model.compute_pleasure on the scaled view (scale 1.0)."""
    x = torch.from_numpy(util.scale_vision(util.transpose_vision(v[:, None])))[:, 0]
    return pleasure_torch((x + 1) / 2, 'Red', 'Green').double().numpy()


def bin_of(vec):
    ang = np.arctan2(vec[:, 1], vec[:, 0])
    return np.argmax(np.cos(ang[:, None] - BIN_ANGLES[None]), 1)


def mag(P):
    a = np.abs(P)
    nz = a[a > 0]
    return {'mean_abs': float(a.mean()), 'max_abs': float(a.max()), 'mean_abs_nonzero': float(nz.mean()) if nz.size else None,
            'frac_nonzero': float((a > 0).mean()), 'mean_episode_max_abs': float(a.max(1).mean())}


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {}
    # 1
    bad, nf = [], 0
    for c in CONDS:
        for s in range(6):
            for e in EPS:
                z = np.load(os.path.join(EV21, f'{c}_s{s}_ep{e}.npz'))
                nf += 1
                for d in ('swap', 'noswap'):
                    for v, _ in FLOWS:
                        for p in ('self', 'other'):
                            for pv in ('hold', 'flip', 'zero'):
                                k = f'{d}_{v}_{p}_{pv}'
                                if k not in z.files or z[k].shape != (600,):
                                    bad.append(f'{c}_s{s}_ep{e}:{k}')
    R['check1'] = {'files': nf, 'missing_or_bad': bad, 'pass': nf == 120 and not bad}
    # 3
    D = {v: h5py.File(os.path.join(DATA, f'data_{v}.h5'), 'r')['test'] for v in ('swap', 'noswap')}
    lay = D['noswap']['layout'][()]
    lo = np.array([L.LANDMARKS[L.CODE_LAYOUT[int(c)]]['Red'] for c in lay], np.float64)
    ln = np.array([L.LANDMARKS[L.other_layout(L.CODE_LAYOUT[int(c)])]['Red'] for c in lay], np.float64)
    p2 = D['noswap']['other_position'][:, 0].astype(np.float64)
    c3, x = {}, {}
    for vname, T0 in FLOWS:
        Ps, Po = {}, {}
        for d in ('swap', 'noswap'):
            Ps[d] = pleas(D[d]['self_vision'][:, T0 - 1])
            Po[d] = pleas(D[d]['other_vision'][:, T0 - 1])
            c3[f'{vname}|{d}|self'] = mag(Ps[d])
            c3[f'{vname}|{d}|other'] = mag(Po[d])
            c3[f'{vname}|{d}|ratio_mean_abs_self_over_other'] = c3[f'{vname}|{d}|self']['mean_abs'] / c3[f'{vname}|{d}|other']['mean_abs']
        As = np.abs(np.concatenate([Ps['swap'], Ps['noswap']]))
        Ao = np.abs(np.concatenate([Po['swap'], Po['noswap']]))
        f2 = float(As.mean() / Ao.mean())
        x[f'{vname}|X2_factor'] = f2
        for name, k in (('X1x2', 2.0), ('X1x4', 4.0), ('X2', f2)):
            x[f'{vname}|{name}|clip_frac_all'] = float((k * Ao > 1).mean())
            x[f'{vname}|{name}|clip_frac_nonzero'] = float((k * Ao[Ao > 0] > 1).mean())
            x[f'{vname}|{name}|mean_abs_after_clip'] = float(np.minimum(k * Ao, 1).mean())
        x[f'{vname}|self_mean_abs_pooled'] = float(As.mean())
        x[f'{vname}|other_mean_abs_pooled'] = float(Ao.mean())
        if vname == 'V1':
            bo, bn = bin_of(lo - p2), bin_of(ln - p2)
            x['X3_bin_Lold_counts'] = np.bincount(bo, minlength=8).tolist()
            x['X3_bin_Lnew_counts'] = np.bincount(bn, minlength=8).tolist()
            x['X3_bins_equal_frac'] = float((bo == bn).mean())
            for d, b in (('noswap', bo), ('swap', bn)):
                P = Po[d]
                has = P.max(1) > 0
                x[f'X3_argmax_held_equals_red_bin|{d}'] = float((P.argmax(1)[has] == b[has]).mean())
                x[f'X3_argmax_held_within1_red_bin|{d}'] = float((np.abs(((P.argmax(1) - b + 4) % 8) - 4)[has] <= 1).mean())
                x[f'X3_frac_positive_held|{d}'] = float(has.mean())
    R['check3'] = c3
    R['inputs'] = x
    ok = {}
    for nm, tg in (('Lold', lo), ('Lnew', ln)):
        u = tg - p2
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        q = p2[:, None] + DREF[None, :, None] * u[:, None]
        ok[nm] = (np.abs(q) <= 10).all(2)
    R['ref_positions'] = {'d': DREF.tolist(),
                          'frac_all_valid_Lold': float(ok['Lold'].all(1).mean()),
                          'frac_all_valid_Lnew': float(ok['Lnew'].all(1).mean()),
                          'frac_valid_d0to5_both': float((ok['Lold'][:, 5:].all(1) & ok['Lnew'][:, 5:].all(1)).mean()),
                          'frac_valid_per_d_Lold': ok['Lold'].mean(0).round(4).tolist(),
                          'frac_valid_per_d_Lnew': ok['Lnew'].mean(0).round(4).tolist()}
    json.dump(R, open(os.path.join(OUT, 'check.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
