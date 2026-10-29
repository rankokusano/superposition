"""
Paper figure 3 with pairs 1-6 (paper round 8; docs/paper_extra_log.md Sec.34.8, paper/check/signrand_s3s5_results.md).
Same panels as analyze/paper_extra/fig_c_signrand.py (pairs 1-3 version, kept unchanged):
  (a) d_P (mse late-5), (b) G5b (l1 late-5), (c)(d) sign-following part green / mixed (l1 late-5),
  (e) aux: G3 hold g (mse late-5), (f) aux: true-pleasure Delta (l1 ep200; filled green-liking, hollow mixed).
Conditions on the x axis: no experience, sign-randomised, zero pleasure (pairs 1-3 only), 80%.
Pairs drawn as in the central figure: pairs 1-3 grey, pairs 4-6 orange; one thin line per pair across conditions.
Values: no experience / sign-randomised / 80% for pairs 1-6 from C_signrand/s3s5/summary.json ('values');
zero pleasure (pairs 1-3) from C_signrand/summary.json. Pairs 1-3 of the first file are asserted equal to the second.
No new computation. -> paper/figs/c_signrand.{pdf,png}, paper/figs/c_signrand_values.json
"""

import json
import os

import numpy as np

import paper_style as ps

C = os.path.join(ps.PI, 'data', 'result', 'paper_extra', 'C_signrand')
CONDS = (('random', '経験\nなし'), ('signrand', '符号\n無作為'), ('pzero', '零入力'), ('mixed80', '経験\n80%'))
PANELS = (('dP', 'G3_dP', '(a) 自分側の快・不快への依存 $d_P$'), ('G5b', 'G5b', '(b) 他者側の符号応答 G5b'),
          ('signfollow_green', 'signfollow_green', '(c) 符号に従う分（緑好み）'),
          ('signfollow_pref', 'signfollow_pref', '(d) 符号に従う分（好み混合）'),
          ('G3_hold', 'G3_hold', '(e) 補助：自分側の積み上がり $g$（P を保つ）'),
          ('delta', 'delta', '(f) 補助：真値条件 Δ（真値 − 零）'))
AUX = ('G3_hold', 'delta')


def load():
    new = json.load(open(os.path.join(C, 's3s5', 'summary.json')))['values']
    old = json.load(open(os.path.join(C, 'summary.json')))
    V = {}
    for s in range(6):
        V[s] = {c: dict(new[f's{s}'][c]) for c in ('random', 'signrand', 'mixed80')}
        if s <= 2:
            for c in ('random', 'signrand', 'mixed80'):        # pairs 1-3 must equal the saved values
                o = old['signrand']['per_pair'][f's{s}']['values'][c]
                for qn, qo in (('dP', 'G3_dP'), ('G5b', 'G5b'), ('signfollow_green', 'signfollow_green'),
                               ('signfollow_pref', 'signfollow_pref'), ('G3_hold', 'G3_hold'),
                               ('delta_green', 'delta_green'), ('delta_pref', 'delta_pref')):
                    assert np.array_equal(np.asarray(V[s][c][qn], float), np.asarray(o[qo], float)), (s, c, qn)
            o = old['pzero']['per_pair'][f's{s}']['values']['pzero']
            V[s]['pzero'] = {'dP': o['G3_dP'], 'G5b': o['G5b'], 'signfollow_green': o['signfollow_green'],
                             'signfollow_pref': o['signfollow_pref'], 'G3_hold': o['G3_hold'],
                             'delta_green': o['delta_green'], 'delta_pref': o['delta_pref']}
    return V


def main():
    ps.setup()
    plt = ps.plt
    V = load()
    out = {}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for ax, (q, _, title) in zip(axes.flat, PANELS):
        aux = q in AUX
        alpha, ms = (0.5, 2.6) if aux else (1.0, 3.0)
        out[q] = {}
        for s in range(6):
            col = ps.C_MAIN if s <= 2 else ps.C_EXTRA
            mk = 'o' if s <= 2 else 's'
            dx = (s - 2.5) * 0.07
            keys = (('delta_green', col), ('delta_pref', 'white')) if q == 'delta' else ((q, col),)
            for k, mfc in keys:
                pts = []
                for i, (c, _) in enumerate(CONDS):
                    if c not in V[s]:
                        continue
                    v = np.asarray(V[s][c][k], float)
                    m = float(v.mean())
                    x = i + dx
                    if v.size > 1:
                        ax.errorbar(x, m, yerr=[[m - v.min()], [v.max() - m]], fmt=mk, color=col, mfc=mfc, ms=ms,
                                    mew=0.8, lw=0.6, capsize=1.0, alpha=alpha)
                    else:
                        ax.plot(x, m, mk, color=col, mfc=mfc, ms=ms, mew=0.8, alpha=alpha)
                    pts.append((x, m))
                    out[q].setdefault(ps.pair_label(s), {}).setdefault(c, {})[k] = m
                ax.plot(*zip(*pts), color=col, lw=0.6, alpha=0.7 * alpha, zorder=0)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([lab for _, lab in CONDS], fontsize=7)
        ax.set_xlim(-0.6, len(CONDS) - 0.4)
        ax.set_title(title, fontsize=8, color='0.45' if aux else 'k')
        if aux:
            for sp in ax.spines.values():
                sp.set_color('0.6')
            ax.tick_params(colors='0.45')
    axes[0, 1].plot([], [], 'o-', color=ps.C_MAIN, ms=3.0, lw=0.6, label='組1〜3')
    axes[0, 1].plot([], [], 's-', color=ps.C_EXTRA, ms=3.0, lw=0.6, label='組4〜6')
    axes[0, 1].legend(frameon=False, fontsize=6.5, loc='upper left')
    fig.text(0.99, -0.04, '点：最終5点の平均（棒は5点の範囲）、細い線は同じ組。零入力条件は組1〜3 のみ。(e)(f) は補助。'
             '(f) は L1 期間 ep200 の1点、塗り：緑好み、白抜き：好み混合。$d_P$ =（保つ − 反転）/ 2', ha='right', fontsize=6.5)
    fig.tight_layout()
    ps.save(fig, 'c_signrand')
    json.dump(out, open(os.path.join(ps.FIGS, 'c_signrand_values.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
