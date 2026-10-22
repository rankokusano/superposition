"""
v13 descriptive subsets and the v13 - v12 comparison (docs/v13_experiment_log.md Sec.3.3 end, 3.4).
CPU only; reads the per-episode values saved by the evaluations:
    G3  : <root>/s1eval/s<s>/per_ep<e>.npz          (eval_trial_v12, keys '<cond>__freeze_K15_hold')
    G5  : <root>/g5/s<s>/g5_per_ep<e>.npz            (eval_g5_v13, keys '<cond>__G5a_<set>_K15_freeze', 'G5b')
for the v13 pairs (root data/result/v13/main) and the v12 reference (G3: data/result/v12/main/s1eval,
G5: data/result/v13/v12ref/g5, the v12 50% runs evaluated with eval_g5_v13).
  * subsets (Sec.3.3): G3 - episodes whose A-1 position at t = 20 is farther than 5 from red;
    G5a - episodes whose (static) A-2 is farther than 5 from its liked colour's reference (main) or
    from red (reference). Experienced, no experience and the paired difference, all and subset.
  * comparison (Sec.3.4): v13 experienced - v12 experienced, same seed, same test episodes paired;
    per pair the late-5 points with mean > 0 and CI excluding 0; 'stronger' if >= 3/5 in >= 2 of 3 pairs.
CIs: eval_trial_v11.boot_ci (episode bootstrap 1000, seed 0), as in the gates.
Writes data/result/v13/main/subsets/subsets_v13.json.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _PI)
from eval_trial_v11 import DATA, RES, boot_ci  # noqa
from eval_m1_v11 import refs  # noqa
from eval_s2_v11 import load_set  # noqa
import gate61_v11 as g11  # noqa

EPS = (160, 170, 180, 190, 200)
T0 = 20


def summ(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return {'mean': float('nan'), 'ci95': [float('nan')] * 2, 'n': int(len(v))}
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def pos_ok(s):
    return s['mean'] > 0 and s['ci95'][0] > 0


def three_way(per, key, mask):
    e, r = per[f'mixed__{key}'], per[f'random__{key}']
    return {'mixed': summ(e[mask]), 'random': summ(r[mask]), 'paired': summ((e - r)[mask])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--root', default=os.path.join(RES, 'v13', 'main'))
    ap.add_argument('--v12_g3', default=os.path.join(RES, 'v12', 'main', 's1eval'))
    ap.add_argument('--v12_g5', default=os.path.join(RES, 'v13', 'v12ref', 'g5'))
    args = ap.parse_args()

    g = h5py.File(os.path.join(DATA, 'v11_s1_random', 'data.h5'), 'r')['test']
    red = np.stack([g11.RED_REF[int(l)] for l in g['layout'][()]])
    far_g3 = np.linalg.norm(g['self_position'][:, T0] - red, axis=1) >= 5
    sets = {k: load_set(k) for k in ('green', 'pref')}
    far_g5 = {}
    for k, d in sets.items():
        p2 = d['other_position'][:, T0]
        liked = np.where(d['a2_liked'] == 1, 'Red', 'Green')
        ref_l = np.stack([refs(d['layout'][i:i + 1], liked[i])[0, 0] for i in range(len(liked))])
        far_g5[k] = {'far_liked': np.linalg.norm(p2 - ref_l, axis=1) >= 5,
                     'far_red': np.linalg.norm(p2 - refs(d['layout'], 'Red')[:, 0], axis=1) >= 5}

    out = {'n_far': {'G3_far_red_t20': int(far_g3.sum()),
                     **{f'G5a_{k}_{m}': int(v[m].sum()) for k, v in far_g5.items() for m in v}},
           'subsets': {}, 'v12_subsets': {}, 'compare': {}}
    allm = np.ones(len(far_g3), bool)
    for s in args.seeds:
        for e in EPS:
            key = f's{s}_ep{e}'
            for tag, g3f, g5f in (('subsets', os.path.join(args.root, 's1eval', f's{s}', f'per_ep{e}.npz'),
                                   os.path.join(args.root, 'g5', f's{s}', f'g5_per_ep{e}.npz')),
                                  ('v12_subsets', os.path.join(args.v12_g3, f's{s}', f'per_ep{e}.npz'), None)):
                r = {}
                if os.path.exists(g3f):
                    p = np.load(g3f)
                    r['G3_K15_hold'] = {'all': three_way(p, 'freeze_K15_hold', allm),
                                        'far_red_t20': three_way(p, 'freeze_K15_hold', far_g3)}
                if g5f and os.path.exists(g5f):
                    p = np.load(g5f)
                    for k in sets:
                        n = len(far_g5[k]['far_liked'])
                        r[f'G5a_{k}_K15_freeze'] = {'all': three_way(p, f'G5a_{k}_K15_freeze', np.ones(n, bool)),
                                                    **{m: three_way(p, f'G5a_{k}_K15_freeze', far_g5[k][m]) for m in far_g5[k]}}
                if r:
                    out[tag][key] = r
            # v12 reference G5a subsets: experienced from the v12ref run, no experience from the v13 run (same model)
            f13, f12 = os.path.join(args.root, 'g5', f's{s}', f'g5_per_ep{e}.npz'), os.path.join(args.v12_g5, f's{s}', f'g5_per_ep{e}.npz')
            if os.path.exists(f12) and os.path.exists(f13):
                p13, p12 = np.load(f13), np.load(f12)
                rr = {}
                for k in sets:
                    kk = f'G5a_{k}_K15_freeze'
                    pp = {f'mixed__{kk}': p12[f'mixed__{kk}'], f'random__{kk}': p13[f'random__{kk}']}
                    n = len(far_g5[k]['far_liked'])
                    rr[kk] = {'all': three_way(pp, kk, np.ones(n, bool)), **{m: three_way(pp, kk, far_g5[k][m]) for m in far_g5[k]}}
                out['v12_subsets'].setdefault(key, {}).update(rr)
            # comparison v13 - v12 (experienced), same seed and episodes
            c = {}
            a, b = os.path.join(args.root, 's1eval', f's{s}', f'per_ep{e}.npz'), os.path.join(args.v12_g3, f's{s}', f'per_ep{e}.npz')
            if os.path.exists(a) and os.path.exists(b):
                c['G3_K15_hold'] = summ(np.load(a)['mixed__freeze_K15_hold'] - np.load(b)['mixed__freeze_K15_hold'])
            if os.path.exists(f13) and os.path.exists(f12):
                p13, p12 = np.load(f13), np.load(f12)
                for kk in ('G5a_green_K15_freeze', 'G5a_pref_K15_freeze', 'G5b'):
                    c[kk] = summ(p13[f'mixed__{kk}'] - p12[f'mixed__{kk}'])
            if c:
                out['compare'][key] = c
    # verdict of the comparison (Sec.3.4)
    verdict = {}
    for kk in ('G3_K15_hold', 'G5a_green_K15_freeze', 'G5a_pref_K15_freeze', 'G5b'):
        per_pair = {}
        for s in args.seeds:
            pts = [out['compare'].get(f's{s}_ep{e}', {}).get(kk) for e in EPS]
            if all(p is not None for p in pts):
                per_pair[f's{s}'] = sum(pos_ok(p) for p in pts)
        if per_pair:
            verdict[kk] = {'points_per_pair': per_pair, 'pairs_ok': sum(v >= 3 for v in per_pair.values()),
                           'stronger': sum(v >= 3 for v in per_pair.values()) >= 2 and len(per_pair) >= 3}
    out['compare_verdict'] = verdict
    od = os.path.join(RES, 'v13', 'main', 'subsets')
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, 'subsets_v13.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({'n_far': out['n_far'], 'compare_verdict': verdict}, indent=1))


if __name__ == '__main__':
    main()
