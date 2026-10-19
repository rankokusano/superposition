"""
v12 stage 1: gate judgement (G1-G3) and figures from eval_trial_v12.py outputs
(docs/v12_experiment_log.md Sec.11.2, 13; instructions Sec.5.2).
Input layout: data/result/v12/<root>/s<seed>/<prefix>eval_{random,mixed}_ep<e>.json and
<prefix>paired_ep<e>.json (for the trial test, --root trial/figtest/s1eval --flat uses <root>/ directly).
Per checkpoint (as v11 Sec.7.13):
  G2 M1(c): mixed hidden c_pred < 0 and paired CI upper < 0; M1(d): mixed c_cf > 0 and paired CI lower > 0;
  colour: mixed hidden S_Red - S_Green on swap episodes, CI lower > 0;
  G3: mixed g (K=15, hold) CI lower > 0 and paired CI lower > 0;
  G1 (recorded with the registered thresholds, see Sec.11.2 / 14): tracking R^2 of both conditions.
A pair passes with >= --min_ckpt of the checkpoints; the gate passes with >= --min_seed pairs.
Figures (written to <root>/figs/): gate bars (hidden / visible), tracking, M1(d) per colour,
self-side accumulation (d* curves per colour; slopes vs v11).
"""

import argparse
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

COLORS = ('Red', 'Green', 'Blue', 'Cyan')
CC = {'random': 'tab:gray', 'mixed': 'tab:red'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='under data/result/v12')
    ap.add_argument('--prefix', default='')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--flat', action='store_true', help='files directly under root (single-seed trial test)')
    ap.add_argument('--min_ckpt', type=int, default=3)
    ap.add_argument('--min_seed', type=int, default=2)
    ap.add_argument('--g1_visible', type=float, default=0.80)
    ap.add_argument('--v11_localise', default='v11/main/localise/localise_ep200.json')
    args = ap.parse_args()
    root = os.path.join(RES, 'v12', args.root)
    figd = os.path.join(root, 'figs'); os.makedirs(figd, exist_ok=True)
    J = lambda s, f: json.load(open(os.path.join(root if args.flat else os.path.join(root, f's{s}'), f'{args.prefix}{f}')))  # noqa
    res = {'per_seed': {}}
    for s in args.seeds:
        rows = []
        for e in args.epochs:
            r = {c: J(s, f'eval_{c}_ep{e}.json') for c in ('random', 'mixed')}
            p = J(s, f'paired_ep{e}.json')
            m = r['mixed']
            rows.append({
                'ep': e,
                'M1c': bool(m['M1']['hidden']['M1c_c_pred']['mean'] < 0 and p['hidden_c_pred']['ci95'][1] < 0),
                'M1d': bool(m['M1']['hidden']['M1d_c_cf']['mean'] > 0 and p['hidden_c_cf']['ci95'][0] > 0),
                'colour': bool(m['M1']['hidden']['colour_swap_SRed_minus_SGreen']['ci95'][0] > 0),
                'G3': bool(m['freeze']['K15_hold']['g']['ci95'][0] > 0 and p['freeze_K15_hold']['ci95'][0] > 0),
                'G1_visible_both': bool(min(r[c]['tracking']['p_next']['visible'] for c in r) >= args.g1_visible),
                'values': {
                    'M1c_mixed': m['M1']['hidden']['M1c_c_pred']['mean'], 'M1c_paired': p['hidden_c_pred']['mean'],
                    'M1d_mixed': m['M1']['hidden']['M1d_c_cf']['mean'], 'M1d_paired': p['hidden_c_cf']['mean'],
                    'colour_mixed': m['M1']['hidden']['colour_swap_SRed_minus_SGreen']['mean'],
                    'G3_mixed': m['freeze']['K15_hold']['g']['mean'], 'G3_paired': p['freeze_K15_hold']['mean'],
                    'track': {c: {k: r[c]['tracking']['p_next'][k] for k in ('all', 'visible', 'hidden')} for c in r}}})
        seed_pass = {k: sum(x[k] for x in rows) >= args.min_ckpt for k in ('M1c', 'M1d', 'colour', 'G3', 'G1_visible_both')}
        res['per_seed'][str(s)] = {'rows': rows, 'seed_pass': seed_pass}
    res['gate'] = {k: sum(res['per_seed'][str(s)]['seed_pass'][k] for s in args.seeds) >= args.min_seed
                   for k in ('M1c', 'M1d', 'colour', 'G3', 'G1_visible_both')}
    res['gate']['G2'] = res['gate']['M1c'] and res['gate']['M1d'] and res['gate']['colour']
    json.dump(res, open(os.path.join(root, f'{args.prefix}gate_s1.json'), 'w'), indent=1)
    print(json.dumps(res['gate']))

    # ---- figures (late-checkpoint means per seed) ----
    # 1. gate bars: hidden / visible
    items = [('M1c_c_pred', 'M1(c) c_pred (<0)'), ('M1d_c_cf', 'M1(d) c_cf (>0)'), ('colour_swap_SRed_minus_SGreen', 'colour S_red-S_green, swap eps (>0)')]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for i, nm in enumerate(('hidden', 'visible')):
        for j, (k, lab) in enumerate(items):
            ax = axes[i, j]
            for si, s in enumerate(args.seeds):
                for ci_, c in enumerate(('random', 'mixed')):
                    v = [J(s, f'eval_{c}_ep{e}.json')['M1'][nm][k] for e in args.epochs]
                    mu = np.mean([x['mean'] for x in v]); lo = np.mean([x['ci95'][0] for x in v]); hi = np.mean([x['ci95'][1] for x in v])
                    ax.bar(si * 2.5 + ci_, mu, color=CC[c], yerr=[[mu - lo], [hi - mu]], capsize=3, label=c if si == 0 else None)
            ax.axhline(0, color='k', lw=0.8); ax.set_xticks([si * 2.5 + 0.5 for si in range(len(args.seeds))])
            ax.set_xticklabels([f's{s}' for s in args.seeds]); ax.set_title(f'{lab}\nmotion-{nm} steps', fontsize=9)
        ax = axes[i, 3]
        if nm == 'hidden':
            for si, s in enumerate(args.seeds):
                for ci_, c in enumerate(('random', 'mixed')):
                    v = [J(s, f'eval_{c}_ep{e}.json')['freeze']['K15_hold']['g'] for e in args.epochs]
                    mu = np.mean([x['mean'] for x in v]); lo = np.mean([x['ci95'][0] for x in v]); hi = np.mean([x['ci95'][1] for x in v])
                    ax.bar(si * 2.5 + ci_, mu, color=CC[c], yerr=[[mu - lo], [hi - mu]], capsize=3)
            ax.axhline(0, color='k', lw=0.8); ax.set_xticks([si * 2.5 + 0.5 for si in range(len(args.seeds))])
            ax.set_xticklabels([f's{s}' for s in args.seeds]); ax.set_title('G3 input freeze g (K=15, pleasure held, >0)', fontsize=9)
        else:
            ax.axis('off')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'v12 stage-1 gate ({args.root}, mean over epochs {args.epochs}; whiskers = mean 95% CI bounds)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(figd, f'{args.prefix}gate_s1.png'), dpi=120); plt.close(fig)
    # 2. tracking
    fig, ax = plt.subplots(figsize=(8, 3.8))
    keys = [('visible', 'visible'), ('hidden', 'hidden'), ('hidden_step1_5', 'hidden 1-5'), ('hidden_step6_10', 'hidden 6-10'), ('hidden_step11_15', 'hidden 11-15')]
    for si, s in enumerate(args.seeds):
        for c in ('random', 'mixed'):
            y = [np.mean([J(s, f'eval_{c}_ep{e}.json')['tracking']['p_next'][k] for e in args.epochs]) for k, _ in keys]
            ax.plot(range(len(keys)), y, marker='os^v'[si % 4], color=CC[c], label=f'{c} s{s}')
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys]); ax.set_ylabel('R^2  h1 -> A-1 position p_{t+1}')
    ax.set_title('Self-position tracking by motion visibility', fontsize=10); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figd, f'{args.prefix}tracking_s1.png'), dpi=120); plt.close(fig)
    # 3. M1(d) per colour (hidden steps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, nm in zip(axes, ('hidden', 'visible')):
        w = 0.13
        for si, s in enumerate(args.seeds):
            for ci_, c in enumerate(('random', 'mixed')):
                v = np.mean([[J(s, f'eval_{c}_ep{e}.json')['M1'][nm]['M1d_S'][X] for X in COLORS] for e in args.epochs], 0)
                ax.bar(np.arange(4) + (si * 2 + ci_ - len(args.seeds) + 0.5) * w, v, w, color=CC[c], alpha=0.5 + 0.5 * (si + 1) / len(args.seeds),
                       label=f'{c} s{s}')
        ax.axhline(0, color='k', lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(COLORS)
        ax.set_title(f'M1(d) S_X per colour, motion-{nm} steps', fontsize=9); ax.legend(fontsize=6, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(figd, f'{args.prefix}m1d_colours_s1.png'), dpi=120); plt.close(fig)
    # 4. self-side accumulation: d* curves per colour (K=15, hold), mean over seeds and epochs; slopes vs v11
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, c in zip(axes[:2], ('random', 'mixed')):
        cur = np.mean([[J(s, f'eval_{c}_ep{e}.json')['freeze']['K15_hold']['dstar_curves_RGBC'] for e in args.epochs] for s in args.seeds], (0, 1))
        for i, X in enumerate(COLORS):
            ax.plot(range(1, cur.shape[1] + 1), cur[i], color={'Red': 'red', 'Green': 'green', 'Blue': 'blue', 'Cyan': 'c'}[X], label=X, lw=2 if X == 'Red' else 1)
        ax.set_xlabel('step k after the freeze (motion & vision hidden, pleasure held)'); ax.set_ylabel('d* (steps toward X best matching the prediction)')
        ax.set_title(f'v12 {c}: input-freeze d* per colour', fontsize=9); ax.legend(fontsize=7)
    ax = axes[2]
    v11 = json.load(open(os.path.join(RES, args.v11_localise)))
    lab, vals, cols = [], [], []
    for c in ('random', 'mixed'):
        s11 = np.mean([[v11[f'{c}_s{s}']['hold_K20']['slope'][X]['mean'] for X in COLORS] for s in (0, 1, 2)], 0)
        s12 = np.mean([[J(s, f'eval_{c}_ep{e}.json')['freeze']['K15_hold']['slope'][X] for X in COLORS] for e in args.epochs for s in args.seeds], 0)
        for tag, sv in (('v11', s11), ('v12', s12)):
            lab.append(f'{tag} {c}'); vals.append(sv)
    vals = np.array(vals)
    for i, X in enumerate(COLORS):
        ax.bar(np.arange(4) + (i - 1.5) * 0.2, vals[:, i], 0.2, color={'Red': 'red', 'Green': 'green', 'Blue': 'blue', 'Cyan': 'c'}[X], label=X)
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=8); ax.axhline(0, color='k', lw=0.8)
    ax.set_title('d* slope per step (v11: K=20 hold, 3 seeds; v12: K=15 hold)', fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figd, f'{args.prefix}accumulation_s1.png'), dpi=120); plt.close(fig)
    print('figures in', figd)


if __name__ == '__main__':
    main()
