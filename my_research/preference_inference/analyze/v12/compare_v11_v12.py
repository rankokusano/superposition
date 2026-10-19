"""
v11 vs v12 comparison table and figure (instructions Sec.5.4; docs/v12_experiment_log.md Sec.21).
Reads the saved v11 outputs and the v12 outputs (main run by default; --v12 trial uses the (P-c) trial files
for a dry run). Missing v12 entries are shown as '-'. Writes data/result/v12/<out>/compare_v11_v12.{md,png}.
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v12 import RES  # noqa

V11 = os.path.join(RES, 'v11')
V12 = os.path.join(RES, 'v12')


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def v11_values():
    g = jl(os.path.join(V11, 'main', 'gate', 'gate_mse_late5.json'))['table']
    loc = jl(os.path.join(V11, 'main', 'localise', 'localise_ep200.json'))
    al = jl(os.path.join(V11, 'main', 'fig5_reason', 'fig5_extra.json'))['check2']
    f5 = [jl(f)['original']['margin_other_self_minus_other_other'] for f in glob.glob(os.path.join(V11, 'main', 'fig5', 'fig5_*.json'))]
    p2 = jl(os.path.join(V11, 'main', 'p2test', 'p2test_ep200.json'))
    s2 = jl(os.path.join(V11, 'stage2', 'eval', 'aggregate_primary.json'))
    sec = s2['secondary_late5_mean_min_max']
    return {
        'self M1(d) paired diff': g['c_cf_diff']['all_seed_mean'],
        'self accumulation g (experienced)': float(np.mean([loc[f'mixed_s{s}']['hold_K20']['g']['mean'] for s in (0, 1, 2)])),
        'self/other alignment cos': float(np.mean([al[f'v11_{c}_s{s}']['cos_before'] for c in ('random', 'mixed') for s in (0, 1, 2)])),
        'Fig.5 margin': float(np.mean(f5)) if f5 else None,
        'process-2 test d_spec (+a, experienced)': float(np.mean([p2[f'mixed_s{s}']['p']['d_spec']['mean'] for s in (0, 2)])),
        'stage 2: T2 s_T2 (setting 1, experienced)': float(np.mean([x['mixed'] for s in ('s0', 's2') for x in s2['green:sT2_t50'][s]['per_ckpt']])),
        'stage 2: T5 d_T5 (setting 2, experienced)': float(np.mean([x['mixed'] for s in ('s0', 's2') for x in s2['pref:dT5'][s]['per_ckpt']])),
        'stage 2: corr(Q2, P_A2) (experienced, set 1)': float(np.mean([sec[f'green_mixed_s{s}']['corr_Q2_PA2/green_set/all'][0] for s in (0, 2)])),
    }


def v12_values(which):
    if which == 'trial':
        T = os.path.join(V12, 'trial')
        e = jl(os.path.join(T, 'pc_eval_mixed_ep150.json')); p = jl(os.path.join(T, 'pc_paired_ep150.json'))
        g5 = jl(os.path.join(T, 'g5_pc', 'g5_paired_ep150.json'))
        f5 = jl(os.path.join(T, 'fig5_pc_mixed', 'fig5_v12_trial_fig5_pc_mixed_s0_ep100.json'))
        return {'self M1(d) paired diff': p['all_c_cf']['mean'], 'self accumulation g (experienced)': e['freeze']['K15_hold']['g']['mean'],
                'self/other alignment cos': e['alignment']['cos_ov(A1)_sv(A2)'], 'Fig.5 margin': f5['original']['margin_other_self_minus_other_other'],
                'process-2 test d_spec (+a, experienced)': e['p2test']['flag1']['p']['d_spec']['mean'],
                'G5a paired diff (set 1)': g5['G5a_green_K15_freeze']['mean'], 'G5b paired diff': g5['G5b']['mean']}
    out = {}
    rows = []
    for s in (0, 1, 2):
        for ep in (160, 170, 180, 190, 200):
            e = jl(os.path.join(V12, 'main', 's1eval', f's{s}', f'eval_mixed_ep{ep}.json'))
            p = jl(os.path.join(V12, 'main', 's1eval', f's{s}', f'paired_ep{ep}.json'))
            if e and p:
                rows.append((p['all_c_cf']['mean'], e['freeze']['K15_hold']['g']['mean'], e['alignment']['cos_ov(A1)_sv(A2)']))
    if rows:
        r = np.array(rows).mean(0)
        out.update({'self M1(d) paired diff': r[0], 'self accumulation g (experienced)': r[1], 'self/other alignment cos': r[2]})
    f5 = jl(os.path.join(V12, 'main', 'fig5', 'fig5_v12_fig5_s0_s0_ep100.json'))
    if f5:
        out['Fig.5 margin'] = f5['original']['margin_other_self_minus_other_other']
    g5 = [jl(f) for f in glob.glob(os.path.join(V12, 'main', 'g5', 's*', 'g5_paired_ep*.json'))]
    if g5:
        out['G5a paired diff (set 1)'] = float(np.mean([x['G5a_green_K15_freeze']['mean'] for x in g5]))
        out['G5b paired diff'] = float(np.mean([x['G5b']['mean'] for x in g5]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v12', default='main', choices=['main', 'trial'])
    ap.add_argument('--out', default='main/compare')
    args = ap.parse_args()
    a, b = v11_values(), v12_values(args.v12)
    keys = list(dict.fromkeys(list(a) + list(b)))
    f = lambda v: '-' if v is None else f'{v:+.4f}'  # noqa
    md = ['| metric | v11 | v12 (' + args.v12 + ') |', '|---|---|---|'] + [f'| {k} | {f(a.get(k))} | {f(b.get(k))} |' for k in keys]
    out = os.path.join(V12, args.out); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, f'compare_v11_v12_{args.v12}.md'), 'w').write('\n'.join(md) + '\n')
    both = [k for k in keys if a.get(k) is not None and b.get(k) is not None]
    fig, axes = plt.subplots(1, len(both), figsize=(2.6 * len(both), 3.4), squeeze=False)
    for ax, k in zip(axes[0], both):
        ax.bar([0, 1], [a[k], b[k]], color=['tab:gray', 'tab:red']); ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['v11', 'v12']); ax.set_title(k, fontsize=7)
    fig.suptitle(f'v11 vs v12 ({args.v12})', fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(out, f'compare_v11_v12_{args.v12}.png'), dpi=120)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
