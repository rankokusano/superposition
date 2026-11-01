"""
Paper round 16 Sec.3, small computation E (docs/paper_extra_log.md Sec.40; exploratory, outside the pre-registration).
CPU, saved arrays only, no model is run.
  E1 stage-B Q_e (c+-, FP, frames 1-50 = steps 0..49, MSE-stage late 5; vdirb_analysis.load_q) per condition and pair:
     episode SD per point (1088 direction-present episodes, main regression sample; and all 1200), averaged over the
     5 points, and the SD of the 5-point-mean Q_e; b_pref (main form, per pair, Vdir_B/analysis/summary.json) / SD.
  E2 stage-I training data (train 3000; v11_s1_mixed = 50%, v13_s1_mixed80 = 80%; A-1 likes Red): every step with
     a1_source == 2 (rule turn that fell back to a random step) is re-classified from the saved own view and position
     with the rule code itself (v11core.rule_agent): liked colour not visible (color_direction None) or, if visible,
     no feasible step at the wall (slide_step None; world bounds +-10). Check: on rule steps (a1_source == 1) the same
     code must reproduce the saved motion.
-> data/result/paper_extra/Round16/E/summary.json
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _PI)
import vdirb_analysis as A  # noqa
from v11core.pleasure import color_direction  # noqa
from v11core.rule_agent import slide_step  # noqa

OUT = os.path.join(A.V.RES, 'paper_extra', 'Round16', 'E')
BOUNDS = ((-10.0, 10.0), (-10.0, 10.0))   # sim_v10/config/world/default.yml xmin..ymax


def e1():
    m = A.meta()
    p = m['p']['1-50']
    keep = (np.abs(p) >= 1.0) & (p != 0)
    S = json.load(open(os.path.join(A.V.RES, 'paper_extra', 'Vdir_B', 'analysis', 'summary.json')))
    out = {}
    for c in A.CONDS:
        Q = A.load_q('cpm', 'FP', c, m)['1-50']                         # (pairs, 5, 1200)
        b = np.array(S['coef'][f'cpm|FP|1-50|1.0|main|{c}']['per_pair']['pref'])
        sd88 = Q[:, :, keep].std(2, ddof=1).mean(1)
        sd12 = Q.std(2, ddof=1).mean(1)
        sdm = Q.mean(1)[:, keep].std(1, ddof=1)
        r = b / sd88
        out[c] = {'n_pairs': int(Q.shape[0]), 'sd_point_1088': sd88.tolist(), 'sd_point_1200': sd12.tolist(),
                  'sd_mean5_1088': sdm.tolist(), 'b_pref': b.tolist(), 'b_pref_over_sd': r.tolist(),
                  'summary': {k: {'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v))}
                              for k, v in (('sd_point_1088', sd88), ('sd_point_1200', sd12), ('sd_mean5_1088', sdm),
                                           ('b_pref', b), ('b_pref_over_sd', r))}}
    return out


def e2():
    out = {}
    for nm, ds in (('p50', 'v11_s1_mixed'), ('p80', 'v13_s1_mixed80')):
        f = h5py.File(os.path.join(A.V.RES, '..', 'data', ds, 'data.h5'), 'r')
        assert f.attrs['a1_liked'] == 'Red'
        g = f['train']
        src = g['a1_source'][()]
        pos = g['self_position'][()].astype(np.float64)
        mot = g['self_motion'][()].astype(np.float64)
        cnt = {'fallback': 0, 'not_visible': 0, 'wall_no_step': 0, 'unexplained': 0,
               'rule_steps_checked': 0, 'rule_steps_reproduced': 0, 'awake': int((src != 3).sum())}
        for n in range(src.shape[0]):
            ts = np.where((src[n] == 2) | (src[n] == 1))[0]
            if len(ts) == 0:
                continue
            V = g['self_vision'][n]                                      # (101, 16, 64, 3) uint8
            for t in ts:
                d = color_direction(V[t], 'Red')
                st = None if d is None else slide_step(pos[n, t], d, BOUNDS, 1.0)
                if src[n, t] == 2:
                    cnt['fallback'] += 1
                    if d is None:
                        cnt['not_visible'] += 1
                    elif st is None:
                        cnt['wall_no_step'] += 1
                    else:
                        cnt['unexplained'] += 1
                else:
                    cnt['rule_steps_checked'] += 1
                    if st is not None and np.abs(st - mot[n, t]).max() < 1e-4:
                        cnt['rule_steps_reproduced'] += 1
        cnt['frac_of_fallback'] = {k: cnt[k] / max(cnt['fallback'], 1) for k in ('not_visible', 'wall_no_step', 'unexplained')}
        cnt['frac_of_awake'] = {k: cnt[k] / cnt['awake'] for k in ('fallback', 'not_visible', 'wall_no_step')}
        out[nm] = cnt
        print(nm, json.dumps(cnt), flush=True)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {'note': 'exploratory, outside the pre-registration (Sec.40)', 'E1': e1()}
    print('E1 done', flush=True)
    R['E2'] = e2()
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    print(json.dumps({c: R['E1'][c]['summary'] for c in R['E1']}, indent=0), flush=True)


if __name__ == '__main__':
    main()
