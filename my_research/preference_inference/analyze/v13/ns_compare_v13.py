"""
v13 vs v13-NS (docs/v13_experiment_log.md Sec.19.3, 28): pairs s0-s2 (NS was not run for s3), late-5.
Self side (mse stage): self-vision prediction error (L1 per pixel, motion-visible / hidden steps), G1 (visible-step
R^2), G2 (M1(c), M1(d), colour), G3 g. Other side (l1 stage): h2 -> A-2 position R^2 (mse ep200), G5a g_other,
G5b, sign-following part of G5a ((freeze - flip)/2, Sec.15-16). For each: experienced values of v13 and NS, the
paired difference (experienced - no experience) of each, and v13 - NS for the experienced models on the same
episodes (same seed -> same initial weights, batch order and hiding schedule, Sec.21).
Writes data/result/v13/ns/compare.json, prints markdown, draws data/result/v13/ns/figs/compare_v13_ns.png.
"""

import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES, boot_ci  # noqa

EPS = (160, 170, 180, 190, 200)
SEEDS = (0, 1, 2)
ROOT = {'v13': os.path.join(RES, 'v13', 'main'), 'ns': os.path.join(RES, 'v13', 'ns')}
DEC = {'v13': os.path.join(RES, 'v13', 'post', 'g5a_decomp'), 'ns': os.path.join(RES, 'v13', 'post', 'g5a_decomp_ns')}
DEC_KEY = {'v13': {'random': 'random', 'mixed': 'mixed80'}, 'ns': {'random': 'random', 'mixed': 'mixed80'}}


def summ(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def per_episode(m, s, e):
    """Per-episode arrays for model family m ('v13' or 'ns'), keys -> {'random': arr, 'mixed': arr}."""
    p1 = np.load(os.path.join(ROOT[m], 's1eval', f's{s}', f'per_ep{e}.npz'))
    p5 = np.load(os.path.join(ROOT[m], 'g5', f's{s}', f'g5_per_ep{e}.npz'))
    out = {}
    for c in ('random', 'mixed'):
        dd = np.load(os.path.join(DEC[m], f's{s}', f'decomp_{DEC_KEY[m][c]}_ep{e}.npz'))
        out.setdefault('M1c_hidden', {})[c] = p1[f'{c}__hidden_c_pred']
        out.setdefault('M1d_hidden', {})[c] = p1[f'{c}__hidden_c_cf']
        out.setdefault('G3', {})[c] = p1[f'{c}__freeze_K15_hold']
        out.setdefault('G5b', {})[c] = p5[f'{c}__G5b']
        for k in ('green', 'pref'):
            fr = p5[f'{c}__G5a_{k}_K15_freeze']
            out.setdefault(f'G5a_{k}', {})[c] = fr
            out.setdefault(f'signpart_{k}', {})[c] = (fr - dd[f'{k}_flip_g']) / 2
    return out


def scalars(m, s, e):
    """Non-per-episode values from the JSONs: self error, tracking, colour."""
    r = {}
    for c in ('random', 'mixed'):
        d = json.load(open(os.path.join(ROOT[m], 's1eval', f's{s}', f'eval_{c}_ep{e}.json')))
        r[c] = {'err_visible': d['err_mean_visible'], 'err_hidden': d['err_mean_hidden'],
                'G1_visible': d['tracking']['p_next']['visible'],
                'colour': d['M1']['hidden']['colour_swap_SRed_minus_SGreen']['mean']}
    return r


def h2(m, s):
    f = sorted(glob.glob(os.path.join(ROOT[m], f'h2_check_s{s}', '*.json')))[-1]
    d = json.load(open(f))
    return {c: d[f'{c}:{s}:200']['ridge_xy'] for c in ('random', 'mixed')}


def main():
    res = {}
    for s in SEEDS:
        for e in EPS:
            pv, pn = per_episode('v13', s, e), per_episode('ns', s, e)
            r = {}
            for q in pv:
                r[q] = {'v13_mixed': summ(pv[q]['mixed']), 'ns_mixed': summ(pn[q]['mixed']),
                        'v13_paired': summ(pv[q]['mixed'] - pv[q]['random']), 'ns_paired': summ(pn[q]['mixed'] - pn[q]['random']),
                        'v13_minus_ns': summ(pv[q]['mixed'] - pn[q]['mixed'])}
            r['scalars'] = {'v13': scalars('v13', s, e), 'ns': scalars('ns', s, e)}
            res[f's{s}_ep{e}'] = r
        res[f's{s}_h2'] = {'v13': h2('v13', s), 'ns': h2('ns', s)}
    gates = {m: {'s1': json.load(open(os.path.join(ROOT[m], 's1eval', 'gate_s1.json')))['gate'],
                 'g5': json.load(open(os.path.join(ROOT[m], 'g5', 'gate_g5.json')))['gate']} for m in ROOT}
    res['gates'] = gates
    os.makedirs(os.path.join(RES, 'v13', 'ns', 'figs'), exist_ok=True)
    json.dump(res, open(os.path.join(RES, 'v13', 'ns', 'compare.json'), 'w'), indent=1)

    def rng(s, q, k):
        v = [res[f's{s}_ep{e}'][q][k]['mean'] for e in EPS]
        return f'{min(v):+.3f}〜{max(v):+.3f}'

    def pts(s, q, k, sign=1):
        return sum((sign * res[f's{s}_ep{e}'][q][k]['mean'] > 0) and
                   (res[f's{s}_ep{e}'][q][k]['ci95'][0] > 0 if sign > 0 else res[f's{s}_ep{e}'][q][k]['ci95'][1] < 0) for e in EPS)

    def srng(s, m, c, k):
        v = [res[f's{s}_ep{e}']['scalars'][m][c][k] for e in EPS]
        return f'{min(v):.3f}〜{max(v):.3f}'
    print('関門：', json.dumps(gates, ensure_ascii=False))
    print('\n**自分側**（経験ありの値。late-5 の範囲）\n')
    print('| 組 | 予測誤差（見える／隠れた）v13 | 同 NS | G1 v13 / NS | 色の基準 v13 / NS |')
    print('|---|---|---|---|---|')
    for s in SEEDS:
        print(f"| s{s} | {srng(s, 'v13', 'mixed', 'err_visible')} / {srng(s, 'v13', 'mixed', 'err_hidden')} | "
              f"{srng(s, 'ns', 'mixed', 'err_visible')} / {srng(s, 'ns', 'mixed', 'err_hidden')} | "
              f"{srng(s, 'v13', 'mixed', 'G1_visible')} / {srng(s, 'ns', 'mixed', 'G1_visible')} | "
              f"{srng(s, 'v13', 'mixed', 'colour')} / {srng(s, 'ns', 'mixed', 'colour')} |")
    names = {'M1c_hidden': 'M1(c)（<0）', 'M1d_hidden': 'M1(d)', 'G3': 'G3 g', 'G5b': 'G5b', 'G5a_green': 'G5a ①',
             'G5a_pref': 'G5a ②', 'signpart_green': '符号に従う分 ①', 'signpart_pref': '符号に従う分 ②'}
    print('\n**指標ごと**（late-5 の範囲。［ ］は組の差が仮説の向きで CI が 0 を含まない点の数）\n')
    print('| 指標 | 組 | v13 経験あり ［組の差］ | NS 経験あり ［組の差］ | v13 − NS（経験ありどうし） |')
    print('|---|---|---|---|---|')
    for q, nm in names.items():
        sg = -1 if q == 'M1c_hidden' else 1
        for s in SEEDS:
            print(f"| {nm} | s{s} | {rng(s, q, 'v13_mixed')} ［{pts(s, q, 'v13_paired', sg)}/5］ | {rng(s, q, 'ns_mixed')} ［{pts(s, q, 'ns_paired', sg)}/5］ | "
                  f"{rng(s, q, 'v13_minus_ns')} |")
    print('\nh²→A-2 の位置 R²（x, y）：' + '；'.join(f"s{s} v13 {np.round(res[f's{s}_h2']['v13']['mixed'], 3).tolist()} NS 経験なし {np.round(res[f's{s}_h2']['ns']['random'], 3).tolist()} 経験あり {np.round(res[f's{s}_h2']['ns']['mixed'], 3).tolist()}" for s in SEEDS))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qs = ['G3', 'G5b', 'signpart_green', 'signpart_pref']
    fig, axes = plt.subplots(1, len(qs), figsize=(4 * len(qs), 3.6))
    for ax, q in zip(axes, qs):
        for j, (m, col) in enumerate((('v13', 'tab:red'), ('ns', 'tab:blue'))):
            for i, s in enumerate(SEEDS):
                for kind, off, hatch in (('mixed', -0.12, None), ('paired', 0.12, '//')):
                    v = [res[f's{s}_ep{e}'][q][f'{m}_{kind}']['mean'] for e in EPS]
                    lo = [res[f's{s}_ep{e}'][q][f'{m}_{kind}']['ci95'][0] for e in EPS]
                    hi = [res[f's{s}_ep{e}'][q][f'{m}_{kind}']['ci95'][1] for e in EPS]
                    x = i + (j - 0.5) * 0.5 + off * 0.5
                    ax.bar(x, np.mean(v), width=0.11, color=col, hatch=hatch, alpha=0.8 if kind == 'mixed' else 0.45,
                           label=f"{'v13 (shared)' if m == 'v13' else 'NS (not shared)'} {'experienced' if kind == 'mixed' else 'paired diff'}" if i == 0 else None)
                    ax.errorbar(x, np.mean(v), yerr=[[np.mean(v) - np.mean(lo)], [np.mean(hi) - np.mean(v)]], color='k', lw=0.8)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(range(len(SEEDS)))
        ax.set_xticklabels([f's{s}' for s in SEEDS])
        ax.set_title({'G3': 'G3 g (self side)', 'G5b': 'G5b (other side, main)', 'signpart_green': 'G5a sign-following part, set 1',
                      'signpart_pref': 'G5a sign-following part, set 2'}[q], fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle('v13 (shared superposition LSTM) vs v13-NS (no sharing): late-5 mean, whiskers = mean CI bounds', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'v13', 'ns', 'figs', 'compare_v13_ns.png'), dpi=130)


if __name__ == '__main__':
    main()
