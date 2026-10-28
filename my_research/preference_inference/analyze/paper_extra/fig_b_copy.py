"""
Paper extra B figure (docs/paper_extra_log.md Sec.18.5, 24). Per model on the x axis, one point per pair (pairs 1-3,
markers o / s / ^), late-5 mean with the range of the 5 points as a bar (single-epoch quantities without bar):
  (a) G5b (l1 late-5), (b) sign-following part, green-liking data, (c) BA other-feature criterion (mse late-5, T = 50),
  (d) h2 -> A-2 position R^2 (smaller of x, y; 0.8 = readable threshold, Sec.3), (e) true-pleasure Delta at A-2 0.5
  (green-liking, l1 ep200), (f) Delta(0.8) - Delta(0.5) (green-liking).
Hollow markers in (a)(b): pair not readable (h2 R^2 < 0.8), not counted in B1 / B3.
From B_copy/summary.json (GPU) and B_extra/summary.json (Sec.24, CPU; outside the pre-registration). No new computation.
-> data/result/paper_extra/B_copy/figs/b_copy.{pdf,png}
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'paper'))
import paper_style as ps  # noqa

X = os.path.join(_PI, 'data', 'result', 'paper_extra')
ps.FIGS = os.path.join(X, 'B_copy', 'figs')
# key in B_copy desc, key in B_extra, label, colour
MODELS = (('NSrand', 'ns_random', '非共有\n経験なし', ps.C_NONE), ('pcols_NSrand', 'nscpp_random', '＋列を\n写す', ps.C_NONE),
          ('copy_NSrand', 'nscp_random', '＋全部\n写す', ps.C_NONE), ('NS', 'ns_mixed', '非共有\n経験80%', ps.C_NS),
          ('pcols', 'nscpp_mixed', '＋列を\n写す', ps.C_NS), ('copy', 'nscp_mixed', '＋全部\n写す', ps.C_NS),
          ('80', 'mixed80', '共有\n経験80%', ps.C_80))
PAIR_MK = ('o', 's', '^')
BA_EXTRA = {'NSrand': 'ns_random_gpu', 'pcols_NSrand': 'nscpp_random', 'copy_NSrand': 'nscp_random'}


def main():
    ps.setup()
    plt = ps.plt
    B = json.load(open(os.path.join(X, 'B_copy', 'summary.json')))['per_pair']
    E = json.load(open(os.path.join(X, 'B_extra', 'summary.json')))

    def val(q, m, bx, s):
        d, e = B[f's{s}']['desc'].get(m, {}), E[f's{s}']
        if q == 'G5b':
            return d.get('G5b_late5')
        if q == 'sf':
            return d.get('signfollow_green')
        if q == 'BA':
            return e['BA'][BA_EXTRA[m]]['BA_FP'] if m in BA_EXTRA else d.get('BA_FP_mselate5')
        if q == 'h2':
            r = B[f's{s}']['h2_A2pos_r2_xy'].get(m)
            return None if r is None else [min(r)]
        if q == 'delta':
            return [e['d6']['Delta'][f'{bx}_green']['mean']] if f'{bx}_green' in e['d6']['Delta'] else None
        if q == 'D':
            k = f'{bx}_green'
            return [e['d6']['D_08_minus_05'][k]['mean']] if k in e['d6']['D_08_minus_05'] else None

    panels = (('G5b', '(a) 他者側の符号応答 G5b'), ('sf', '(b) 符号に従う分（緑好み）'),
              ('BA', '(c) 照合の BA（他者特徴基準）'), ('h2', '(d) $h^2$ から A-2 の位置（$R^2$）'),
              ('delta', '(e) 真値条件 Δ（A-2 0.5、緑好み）'), ('D', '(f) Δ の変化 0.8 − 0.5（緑好み）'))
    fig, axes = plt.subplots(3, 2, figsize=(6.4, 6.6))
    for ax, (q, title) in zip(axes.flat, panels):
        for i, (m, bx, _, col) in enumerate(MODELS):
            for s in range(3):
                v = val(q, m, bx, s)
                if v is None:
                    continue
                v = np.asarray(v, float)
                rd = B[f's{s}']['readable'].get(m, True)
                hollow = q in ('G5b', 'sf') and not rd
                x = i + (s - 1) * 0.22
                ax.errorbar(x, v.mean(), yerr=[[v.mean() - v.min()], [v.max() - v.mean()]] if len(v) > 1 else None,
                            fmt=PAIR_MK[s], color=col, mfc='white' if hollow else col, ms=3.6, lw=0.7, capsize=1.2)
        ax.axhline({'BA': 0.5, 'h2': 0.8}.get(q, 0), color='0.6', lw=0.5, ls='--' if q in ('BA', 'h2') else '-')
        for xs in (2.5, 5.5):
            ax.axvline(xs, color='0.85', lw=0.5)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([lab for _, _, lab, _ in MODELS], fontsize=7)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_title(title, fontsize=8)
    for s in range(3):
        axes[0, 0].plot([], [], PAIR_MK[s], color='0.3', ms=3.6, label=ps.pair_label(s))
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc='lower left')
    fig.text(0.99, -0.045, '点：最終5点の平均（棒は5点の範囲）、(d)〜(f) は1点。白抜き：$h^2$ が読めない組（$R^2$ < 0.8）。\n'
             '経験なし系の写し・列だけは事前登録外。空き：測っていない', ha='right', fontsize=6.5)
    fig.tight_layout()
    ps.save(fig, 'b_copy')


if __name__ == '__main__':
    main()
