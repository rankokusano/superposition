"""
One table and one figure for v12 (50%, shared), v13 (80%, shared) and v13-NS (80%, not shared)
(docs/v13_experiment_log.md Sec.29; documents only, CPU). Pairs s0-s2 (the gate pairs; NS has no s3), late-5.
  self side : G2 = M1(d) on motion-hidden steps (mse stage), G3 g (mse stage)
  other side: G5b (l1 stage, main), sign-following part of G5a = (g(freeze) - g(flip)) / 2, sets (1) and (2)
  stage 2   : verdicts of T2 / T5 (from the logs; NS: not run)
No-experience partners: v12 / v13 -> v12_s1_random (same model), NS -> v13ns_s1_random.
Per pair: the late-5 mean of the experienced value, the mean CI bounds, and the number of late-5 points where the
paired difference (experienced - no experience) and the experienced value are both in the hypothesised direction
with CIs excluding 0 (the gate rule). All values from saved per-episode files (CI: eval_trial_v11.boot_ci).
Writes data/result/v13/compare_all/compare_v12_v13_ns.json, figs/compare_v12_v13_ns.png; prints markdown.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES, boot_ci  # noqa

EPS = (160, 170, 180, 190, 200)
SEEDS = (0, 1, 2)
OUT = os.path.join(RES, 'v13', 'compare_all')
MODELS = ('v12', 'v13', 'ns')
LABEL = {'v12': 'v12 (50%, shared)', 'v13': 'v13 (80%, shared)', 'ns': 'v13-NS (80%, not shared)'}
COLOR = {'v12': 'tab:orange', 'v13': 'tab:red', 'ns': 'tab:blue'}
P = lambda *a: os.path.join(RES, *a)  # noqa


def files(m, s, e):
    """(experienced, no-experience) per-episode arrays for each measure."""
    s1 = {'v12': P('v12', 'main', 's1eval', f's{s}', f'per_ep{e}.npz'), 'v13': P('v13', 'main', 's1eval', f's{s}', f'per_ep{e}.npz'),
          'ns': P('v13', 'ns', 's1eval', f's{s}', f'per_ep{e}.npz')}[m]
    g5x = {'v12': P('v13', 'v12ref', 'g5', f's{s}', f'g5_per_ep{e}.npz'), 'v13': P('v13', 'main', 'g5', f's{s}', f'g5_per_ep{e}.npz'),
           'ns': P('v13', 'ns', 'g5', f's{s}', f'g5_per_ep{e}.npz')}[m]
    g5r = P('v13', 'ns', 'g5', f's{s}', f'g5_per_ep{e}.npz') if m == 'ns' else P('v13', 'main', 'g5', f's{s}', f'g5_per_ep{e}.npz')
    dec = P('v13', 'post', 'g5a_decomp_ns' if m == 'ns' else 'g5a_decomp', f's{s}')
    dx = np.load(os.path.join(dec, f"decomp_{'mixed50' if m == 'v12' else 'mixed80'}_ep{e}.npz"))
    dr = np.load(os.path.join(dec, f'decomp_random_ep{e}.npz'))
    a, x, r = np.load(s1), np.load(g5x), np.load(g5r)
    out = {'G2_M1d_hidden': (a['mixed__hidden_c_cf'], a['random__hidden_c_cf']),
           'G3_g': (a['mixed__freeze_K15_hold'], a['random__freeze_K15_hold']),
           'G5b': (x['mixed__G5b'], r['random__G5b'])}
    for k in ('green', 'pref'):
        out[f'sign_{k}'] = ((x[f'mixed__G5a_{k}_K15_freeze'] - dx[f'{k}_flip_g']) / 2,
                            (r[f'random__G5a_{k}_K15_freeze'] - dr[f'{k}_flip_g']) / 2)
    return out


def summ(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def main():
    os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)
    res = {}
    for m in MODELS:
        for s in SEEDS:
            pts, ex, nx, pr = {}, {}, {}, {}
            for e in EPS:
                for q, (x, r) in files(m, s, e).items():
                    sx, sp, sr = summ(x), summ(x - r), summ(r)
                    ex.setdefault(q, []).append(sx); pr.setdefault(q, []).append(sp); nx.setdefault(q, []).append(sr)
                    pts[q] = pts.get(q, 0) + int(sx['ci95'][0] > 0 and sp['ci95'][0] > 0)
            res[f'{m}_s{s}'] = {q: {'exp_mean': float(np.mean([d['mean'] for d in ex[q]])),
                                    'exp_ci': [float(np.mean([d['ci95'][0] for d in ex[q]])), float(np.mean([d['ci95'][1] for d in ex[q]]))],
                                    'none_mean': float(np.mean([d['mean'] for d in nx[q]])),
                                    'paired_mean': float(np.mean([d['mean'] for d in pr[q]])), 'points': pts[q]} for q in ex}
    stage2 = {'v12': {'T2': '1/4 組（3 組で 1/3）', 'T5': '1/4 組（3 組で 1/3）'},
              'v13': {'T2': '2/4 組（3 組で 2/3）', 'T5': '1/4 組（3 組で 1/3）'}, 'ns': {'T2': '回していない', 'T5': '回していない'}}
    res['stage2'] = stage2
    json.dump(res, open(os.path.join(OUT, 'compare_v12_v13_ns.json'), 'w'), indent=1)

    names = {'G2_M1d_hidden': 'G2：M1(d)（運動が隠れた一歩）', 'G3_g': 'G3：自分側の積み上がり g', 'G5b': 'G5b（他者側の主の根拠）',
             'sign_green': 'G5a の符号に従う分 ①', 'sign_pref': 'G5a の符号に従う分 ②'}
    print('| 指標 | v12（50%、共有あり） | v13（80%、共有あり） | v13-NS（80%、共有なし） |')
    print('|---|---|---|---|')
    for q, nm in names.items():
        cells = []
        for m in MODELS:
            v = [res[f'{m}_s{s}'][q]['exp_mean'] for s in SEEDS]
            n = [res[f'{m}_s{s}'][q]['none_mean'] for s in SEEDS]
            p = [res[f'{m}_s{s}'][q]['points'] for s in SEEDS]
            ok = sum(x >= 3 for x in p)
            cells.append(f"{min(v):+.3f}〜{max(v):+.3f}（経験なし {min(n):+.3f}〜{max(n):+.3f}）、{'/'.join(map(str, p))} → {ok}/3 組")
        print(f'| {nm} | ' + ' | '.join(cells) + ' |')
    for t in ('T2', 'T5'):
        print(f"| 2 段目 {'① T2' if t == 'T2' else '② T5'} | " + ' | '.join(stage2[m][t] for m in MODELS) + ' |')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qs = [('G2_M1d_hidden', 'Self: G2 M1(d), hidden steps'), ('G3_g', 'Self: G3 accumulation g'),
          ('G5b', 'Other: G5b (main)'), ('sign_green', 'Other: G5a sign-following, set 1'), ('sign_pref', 'Other: G5a sign-following, set 2')]
    fig, axes = plt.subplots(1, len(qs) + 1, figsize=(3.6 * (len(qs) + 1), 3.8))
    for ax, (q, title) in zip(axes, qs):
        for j, m in enumerate(MODELS):
            for i, s in enumerate(SEEDS):
                d = res[f'{m}_s{s}'][q]
                x = i + (j - 1) * 0.26
                ax.bar(x, d['exp_mean'], width=0.24, color=COLOR[m], alpha=0.85, label=LABEL[m] if i == 0 else None,
                       edgecolor='k' if d['points'] >= 3 else 'none', linewidth=1.2)
                ax.errorbar(x, d['exp_mean'], yerr=[[d['exp_mean'] - d['exp_ci'][0]], [d['exp_ci'][1] - d['exp_mean']]], color='k', lw=0.8)
                ax.plot(x, d['none_mean'], 'o', mfc='white', mec='k', ms=4, label='no experience' if (i == 0 and j == 0) else None)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(range(len(SEEDS)))
        ax.set_xticklabels([f's{s}' for s in SEEDS])
        ax.set_title(title, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, frameon=False)
    ax = axes[-1]
    ax.axis('off')
    txt = ('Stage 2 (other-preference read-out)\n\n'
           'T2: v12 1/4 pairs, v13 2/4 pairs\n     (3 pairs: 1/3, 2/3)\n'
           'T5: v12 1/4, v13 1/4\n'
           'v13-NS: not run\n\n'
           'corr(Q2, true pleasure): negative\n  in v12 and v13 (position passing)\n\n'
           'Bars: experienced, late-5 mean;\n whiskers: mean CI; black edge:\n pair passes (>= 3/5 points);\n'
           ' circles: no-experience model')
    ax.text(0.0, 1.0, txt, va='top', fontsize=8.5, family='monospace')
    fig.suptitle('Self-side expectation grows with experience (v12 -> v13) in both shared and non-shared models; '
                 'the other-side use of pleasure appears only with sharing', fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, 'figs', 'compare_v12_v13_ns.png'), dpi=140)


if __name__ == '__main__':
    main()
