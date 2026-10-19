"""
v11 stage-2 primary judgement (docs/v11_experiment_log.md Sec.7.42 B-C) from eval_s2_v11.py outputs.
Per pair s, setting, G checkpoint: the mixed value must be in the hypothesised direction (> 0)
and the paired difference (mixed - random; T2 per episode, T5 per matched pair, resampled by
green episode) must have a bootstrap (1000, seed 0) 95% CI excluding 0. Pair passes with >= 3/5
of the late-5; the setting passes when every pair passes (2 pairs) / >= 2 of 3.
Also tabulates the priority secondary values (mean over the late-5, paired by checkpoint).
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import RES  # noqa

EV = os.path.join(RES, 'v11', 'stage2', 'eval')
EPS = [160, 170, 180, 190, 200]


def boot(vm, vr, groups, n=1000, seed=0):
    d = vm - vr
    ug, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, d); c = np.bincount(inv)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        k = rng.integers(0, len(ug), len(ug))
        bs.append(s[k].sum() / c[k].sum())
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 2])
    args = ap.parse_args()
    out = {}
    for st, keys in (('green', ['sT2_t50', 'sT2_t50_52', 'sT2_pos', 'dT5']), ('pref', ['dT5', 'sT2_t50', 'sT2_t50_52'])):
        for key in keys:
            res = {}
            for s in args.seeds:
                rows = []
                for e in EPS:
                    m = np.load(os.path.join(EV, f'{st}_mixed_s{s}_ep{e}_per.npz'))
                    r = np.load(os.path.join(EV, f'{st}_random_s{s}_ep{e}_per.npz'))
                    grp = m['t5_green_ep'] if key == 'dT5' else np.arange(len(m[key]))
                    diff, ci = boot(m[key], r[key], grp)
                    ok = bool(m[key].mean() > 0 and ci[0] > 0)
                    rows.append({'ep': e, 'mixed': float(m[key].mean()), 'random': float(r[key].mean()),
                                 'diff': diff, 'ci': ci, 'pass': ok})
                res[f's{s}'] = {'per_ckpt': rows, 'n_pass': sum(x['pass'] for x in rows),
                                'pair_pass': sum(x['pass'] for x in rows) >= 3}
            n_ok = sum(v['pair_pass'] for v in res.values())
            need = len(args.seeds) if len(args.seeds) == 2 else 2
            res['setting_pass'] = n_ok >= need
            out[f'{st}:{key}'] = res
    # priority secondary: late-5 means
    sec = {}
    for st in ('green', 'pref'):
        for s in args.seeds:
            for c in ('random', 'mixed'):
                js = [json.load(open(os.path.join(EV, f'{st}_{c}_s{s}_ep{e}.json'))) for e in EPS]
                flat = {}
                for j in js:
                    for k1 in ('corr_Q2_PA2', 'motion_readout'):
                        for k2, v2 in j[k1].items():
                            for k3, v3 in v2.items():
                                flat.setdefault(f'{k1}/{k2}/{k3}', []).append(v3)
                    for k3, v3 in j['self_vision_L1_per_pixel'].items():
                        flat.setdefault(f'selfL1/{k3}', []).append(v3)
                    for k3, v3 in j['T5_by_t_bins'].items():
                        flat.setdefault(f'T5_t/{k3}', []).append(v3)
                sec[f'{st}_{c}_s{s}'] = {k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for k, v in flat.items()}
    out['secondary_late5_mean_min_max'] = sec
    out['d6_PA2'] = json.load(open(os.path.join(EV, 'd6_PA2.json')))
    with open(os.path.join(EV, 'aggregate_primary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        if ':' not in k:
            continue
        print(k, 'setting_pass' if v['setting_pass'] else 'setting_FAIL')
        for s in args.seeds:
            pr = v[f's{s}']
            print(f'  s{s} n_pass={pr["n_pass"]}/5 ',
                  ' '.join(f'ep{x["ep"]}:m{x["mixed"]:+.4f} r{x["random"]:+.4f} d{x["diff"]:+.4f}[{x["ci"][0]:+.4f},{x["ci"][1]:+.4f}]' for x in pr['per_ckpt']))


if __name__ == '__main__':
    main()
