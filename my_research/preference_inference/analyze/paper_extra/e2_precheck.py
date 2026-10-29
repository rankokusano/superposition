"""
Paper round 11 (docs/paper_extra_log.md Sec.35.1; instructions paper/check/paper_instructions_round11.md Sec.1): checks
before E2, CPU, nothing is run through a model.
  sets   : the stage-2 test sets green / pref (eval_s2_v11.SETS) and the A-2 p = 0.8 sets green_a2p80 / pref_a2p80
           (a_cross_eval): N, a2_liked counts, whether a2_liked is stored in the file.
  sign   : P as if A-2 liked Red and as if A-2 liked Green (eval_inverse_v14.pa2_as = pleasure_np(A-2 view, liked,
           disliked)) are exact negatives; A-2's true pleasure (eval_s2_v11.true_pleasure, the true-pleasure condition)
           equals the one of A-2's liked colour, so the opposite preference is exactly -P_A2.
  fp_l1  : the other-side feature-prediction error does not change in the L1 stage (decoder only): the saved
           eval_inverse_v14 FP arrays of the L1-stage points ep160..200 equal those of the MSE-stage ep200, per model.
  inv    : the saved eval_inverse_v14 runs per model family / pair / set / epoch (which E2 inputs already exist).
  models : checkpoints of none (v12_s1_random), 50% (v12_s1_mixed), 80% (v13_s1_mixed), signrand 80%
           (paper_extra/C_signrand/s1_signrand) for pairs 1-6, both stages.
-> data/result/paper_extra/E2/precheck.json
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_PI, 'analyze', 'v14'), os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)
import a_cross_eval  # noqa  (registers green_a2p80 / pref_a2p80 in eval_s2_v11.SETS)
from eval_s2_v11 import load_set, true_pleasure  # noqa
from eval_inverse_v14 import pa2_as  # noqa

RES = a_cross_eval.RES
OUT = os.path.join(RES, 'paper_extra', 'E2')
SETS = ('pref', 'green', 'pref_a2p80', 'green_a2p80')
INV = {'v14': os.path.join(RES, 'v14', 'inverse'), 'csr': os.path.join(RES, 'paper_extra', 'C_signrand', 'eval', 'inverse')}
FAMS = {'none_l1': ('v14', 'v12_random_{s}'), 'none_mse': ('v14', 'v12mse_random_{s}'),
        'p50_l1': ('v14', 'v12_mixed_{s}'), 'p50_mse': ('v14', 'v12mse_mixed_{s}'),
        'p80_l1': ('v14', 'v13_mixed_{s}'), 'p80_mse': ('v14', 'v13mse_mixed_{s}'),
        'sr80_l1': ('csr', 'csr_signrand_{s}'), 'sr80_mse': ('csr', 'csrmse_signrand_{s}')}
CKPT = {'none': ('v12_s1_random_mse/{s}', 'v12_s1_random_l1_s{s}/{s}'), 'p50': ('v12_s1_mixed_mse/{s}', 'v12_s1_mixed_l1_s{s}/{s}'),
        'p80': ('v13_s1_mixed_mse/{s}', 'v13_s1_mixed_l1_s{s}/{s}'),
        'sr80': ('paper_extra/C_signrand/s1_signrand_mse/{s}', 'paper_extra/C_signrand/s1_signrand_l1_s{s}/{s}')}
EPS = (160, 170, 180, 190, 200)


def main():
    os.makedirs(OUT, exist_ok=True)
    rec = {'sets': {}, 'sign': {}, 'fp_l1': {}, 'inv': {}, 'models': {}}
    for k in SETS:
        d = load_set(k)
        stored = 'a2_liked' in h5py.File(d['path'], 'r')[d['split']]
        lk = d['a2_liked']
        rec['sets'][k] = {'path': os.path.relpath(d['path'], _PI), 'split': d['split'], 'n': int(len(lk)),
                          'a2_liked_stored': bool(stored), 'n_liked_red': int((lk == 1).sum()), 'n_liked_green': int((lk == 0).sum())}
        pr, pg, pt = pa2_as(d, 'Red'), pa2_as(d, 'Green'), true_pleasure(k, d)
        own = np.where((lk == 1)[:, None, None], pr, pg)
        rec['sign'][k] = {'red_equals_minus_green': bool(np.array_equal(pr, -pg)),
                          'max_abs_red_plus_green': float(np.abs(pr + pg).max()),
                          'true_equals_liked': bool(np.array_equal(pt, own)),
                          'true_equals_minus_disliked': bool(np.array_equal(pt, -np.where((lk == 1)[:, None, None], pg, pr))),
                          'frac_steps_nonzero': float((np.abs(pt).sum(2) > 0).mean()), 'shape': list(pt.shape)}
    for fam in ('none', 'p50', 'p80', 'sr80'):
        for s in range(6):
            for k in ('pref', 'green'):
                src, mse = FAMS[f'{fam}_mse']
                f2 = os.path.join(INV[src], mse.format(s=s) + f'_ep200_{k}.npz')
                src1, l1 = FAMS[f'{fam}_l1']
                fs = [os.path.join(INV[src1], l1.format(s=s) + f'_ep{e}_{k}.npz') for e in EPS]
                if not (os.path.exists(f2) and all(map(os.path.exists, fs))):
                    continue
                z2 = np.load(f2)
                keys = [q for q in z2.files if q.startswith('FP_')]
                rec['fp_l1'][f'{fam}_s{s}_{k}'] = all(np.array_equal(np.load(f)[q], z2[q]) for f in fs for q in keys)
    for name, (src, pat) in FAMS.items():
        for s in range(6):
            for k in ('pref', 'green'):
                rec['inv'][f'{name}_s{s}_{k}'] = [e for e in EPS if os.path.exists(os.path.join(INV[src], pat.format(s=s) + f'_ep{e}_{k}.npz'))]
    for c, (m, l) in CKPT.items():
        for s in range(6):
            dm, dl = os.path.join(RES, m.format(s=s), 'model'), os.path.join(RES, l.format(s=s), 'model')
            rec['models'][f'{c}_s{s}'] = {'mse': len(os.listdir(dm)) if os.path.isdir(dm) else 0,
                                          'l1': len(os.listdir(dl)) if os.path.isdir(dl) else 0,
                                          'late5_mse': all(os.path.exists(os.path.join(dm, f'{e:05d}.pth')) for e in EPS),
                                          'late5_l1': all(os.path.exists(os.path.join(dl, f'{e:05d}.pth')) for e in EPS)}
    with open(os.path.join(OUT, 'precheck.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ('sets', 'sign')}, indent=1))
    print('fp_l1', rec['fp_l1'])
    print('missing inv', {k: v for k, v in rec['inv'].items() if len(v) < 5})
    print('models', {k: v for k, v in rec['models'].items() if not (v['late5_mse'] and v['late5_l1'])} or 'all late-5 present',
          {k: (v['mse'], v['l1']) for k, v in rec['models'].items()})


if __name__ == '__main__':
    main()
