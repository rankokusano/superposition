"""
v13 gate G5 (docs/v13_experiment_log.md Sec.3.3): the main() of analyze/v12/eval_g5_v12.py with two
additions; every measure is the imported v12 function (g5a_est, g5a_metric, g5b, calibrations), so the
JSON outputs have the same keys and values as the v12 script would write.
  * --conds (default random mixed): the v12 reference run (Sec.3.3 splits, Sec.3.4) needs only the
    50% experienced model;
  * the per-episode values are also saved (g5_per_ep<e>.npz, keys '<cond>__<measure>') for the
    descriptive subsets (Sec.3.3) and the v13 - v12 comparison (Sec.3.4).
Run through analyze/v13/v12run.py g5 (maps v13 model names to the reused v12 runs).
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v12'))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_g5_v12 import RES, calibrate, calibrate_bg, g5a_est, g5a_metric, g5b, load_set, summ, true_pleasure  # noqa
from eval_trial_v12 import load_v12  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--sub', required=True, help='output dir under data/result/v12 (use ../v13/...)')
    ap.add_argument('--conds', nargs='+', default=['random', 'mixed'])
    args = ap.parse_args()
    out = os.path.join(RES, 'v12', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in ('green', 'pref')}
    PA2 = {k: true_pleasure(k, D[k]) for k in D}
    g = h5py.File(D['green']['path'], 'r')['test']
    alpha, beta = calibrate(g['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    bgc = calibrate_bg(g['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    per = {}
    for c in args.conds:
        model, p_mask = load_v12(args.exp_fmt.format(c), args.seed, args.epoch, dev)
        r = {'exp': args.exp_fmt.format(c), 'epoch': args.epoch, 'G5a': {}}
        model.motion_masking = False     # stage-2 protocol for G5a
        for k in D:
            for K in (15, 20):
                for variant in ('freeze', 'continue', 'zero'):
                    if K == 20 and variant != 'freeze':
                        continue
                    est, est_bg = g5a_est(model, D[k], PA2[k], dev, alpha, beta, K, variant, bg=bgc)
                    gb, slb, Sb = g5a_metric(est_bg, D[k])
                    cb = g5a_metric.curves
                    r.setdefault('G5a_bg', {})[f'{k}_K{K}_{variant}'] = {'g_other': summ(gb), 's_liked': summ(slb),
                                                                       'curve_liked': cb[0], 'curve_others': cb[1],
                                                                       'S_mean_RGBC': np.nanmean(Sb, 0).round(4).tolist(),
                                                                       'frac_read': float(np.isfinite(est_bg[..., 0]).mean())}
                    per[(c, f'G5a_bg_{k}_K{K}_{variant}')] = gb
                    gg, sl, S = g5a_metric(est, D[k])
                    r['G5a'][f'{k}_K{K}_{variant}'] = {'g_other': summ(gg), 's_liked': summ(sl),
                                                       'curve_liked': g5a_metric.curves[0], 'curve_others': g5a_metric.curves[1],
                                                       'S_mean_RGBC': np.nanmean(S, 0).round(4).tolist(),
                                                       'frac_read': float(np.isfinite(est[..., 0]).mean())}
                    per[(c, f'G5a_{k}_K{K}_{variant}')] = gg
        model.motion_masking = True      # canonical for G5b (as the stage-1 process-2 test)
        r['G5b'], main_b, main_b_bg = g5b(model, p_mask, dev)
        per[(c, 'G5b')] = main_b
        per[(c, 'G5b_bg')] = main_b_bg
        r['calib_bg_G5a'] = list(bgc)
        with open(os.path.join(out, f'g5_{c}_ep{args.epoch}.json'), 'w') as f:
            json.dump(r, f, indent=1)
        print(c, json.dumps({'G5a_freeze_K15': {k: r['G5a'][f'{k}_K15_freeze']['g_other']['mean'] for k in D},
                             'G5b': r['G5b']['main_pm_own']['mean']}), flush=True)
    np.savez_compressed(os.path.join(out, f'g5_per_ep{args.epoch}.npz'), **{f'{c}__{k}': v for (c, k), v in per.items()})
    if len(args.conds) < 2:
        return
    keys = sorted({k for (_, k) in per})
    paired = {k: summ(per[('mixed', k)] - per[('random', k)]) for k in keys}
    with open(os.path.join(out, f'g5_paired_ep{args.epoch}.json'), 'w') as f:
        json.dump(paired, f, indent=1)
    print('paired', json.dumps({k: round(v['mean'], 4) for k, v in paired.items()}), flush=True)


if __name__ == '__main__':
    main()
