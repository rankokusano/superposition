"""
Paper figure 4 'projection_error' (experiment 2, true-value condition; documents only, CPU).
(a) other-side feature-prediction error (T3, other_fp_mse) with process-2 = zero / A-2's true pleasure P_A2 (D6) /
    the free-output module's generated Q2, per condition (no experience, 50%, 80%) and pair 1-4, for the green-liker
    set and the mixed-preference set. Source: data/result/{v12,v13}/stage2/t1t3/<set>_<cond>_s<s>_ep<e>.json
    (key <set>; the green set from the green-trained module, the mixed set from the pref-trained module). Zero and D6
    do not depend on the module (identical over ep160-200, checked here); 'gen' is the late-5 mean (range as bars).
    No experience = v12 random (the same model is reused for 80%; v13 files identical, checked).
(b) D6 split by A-2's step type (followed its preference / did not follow / sleep), pairs 1-3, from
    data/result/v13/post/d6_split/summary.json (v13 log Sec.24-25).
Writes paper/figs/projection_error.{pdf,png} and paper/figs/projection_values.json.
"""

import json
import os

import numpy as np

from paper_style import C_50, C_80, C_EXTRA, C_NONE, FIGS, RES, pair_label, plt, save, setup

EP = (160, 170, 180, 190, 200)
SETS = (('green', '緑好みデータ'), ('pref', '好み混合データ'))
CONDS = (('none', 'v12', 'random', '経験なし'), ('50', 'v12', 'mixed', '経験50%'), ('80', 'v13', 'mixed', '経験80%'))
INPUTS = (('zero', '零', '#7a7a7a', 'o'), ('d6', '真値（本当の快・不快）', '#d55181', 's'),
          ('gen', '他者用モジュールの生成', '#1baf7a', '^'))
PAIR_LS = {0: '-', 1: '-', 2: '-', 3: '--'}


def t3(v, st, cond, s):
    out = {}
    for e in EP:
        d = json.load(open(os.path.join(RES, v, 'stage2', 't1t3', f'{st}_{cond}_s{s}_ep{e}.json')))[st]
        for k in ('zero', 'd6', 'gen'):
            out.setdefault(k, []).append(d[k]['other_fp_mse'])
    assert np.ptp(out['zero']) == 0 and np.ptp(out['d6']) == 0
    return {'zero': out['zero'][0], 'd6': out['d6'][0], 'gen': float(np.mean(out['gen'])),
            'gen_min': min(out['gen']), 'gen_max': max(out['gen'])}


def main():
    setup()
    vals = {}
    for st, _ in SETS:
        for cn, v, cond, _ in CONDS:
            for s in range(4):
                vals[f'{st}|{cn}|{pair_label(s)}'] = t3(v, st, cond, s)
                if cn == 'none':
                    assert t3('v13', st, 'random', s) == vals[f'{st}|{cn}|{pair_label(s)}']
    sp = json.load(open(os.path.join(RES, 'v13', 'post', 'd6_split', 'summary.json')))
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 5.2), sharey='row')
    for col, (st, stname) in enumerate(SETS):
        ax = axs[0, col]
        for g, (cn, _, _, clab) in enumerate(CONDS):
            for s in range(4):
                r = vals[f'{st}|{cn}|{pair_label(s)}']
                xs = [g + (i - 1) * 0.26 + (s - 1.5) * 0.045 for i in range(3)]
                ax.plot(xs, [r['zero'], r['d6'], r['gen']], color='0.6' if s < 3 else C_EXTRA, lw=0.6, ls=PAIR_LS[s],
                        zorder=1, label=('組1〜3' if s == 0 else '組4（予備）' if s == 3 else None) if g == 0 else None)
                for i, (k, lab, c, mk) in enumerate(INPUTS):
                    yerr = [[r['gen'] - r['gen_min']], [r['gen_max'] - r['gen']]] if k == 'gen' else None
                    ax.errorbar(xs[i], r[k], yerr=yerr, color=c, marker=mk, ms=3.5, lw=0.8, capsize=1.2, zorder=2,
                                label=lab if (g == 0 and s == 0) else None)
        ax.set_xticks(range(3))
        ax.set_xticklabels([c[3] for c in CONDS])
        ax.set_title(f'(a) {stname}', loc='left', fontsize=9)
        if col == 0:
            ax.set_ylabel('他者側の特徴予測の誤差')
        # (b)
        ax = axs[1, col]
        steps = (('follow', '好みに従った一歩'), ('notfollow', '従わなかった一歩'), ('sleep', '休止'))
        for g, (cn, key, c, lab) in enumerate((('none', 'random_d6', C_NONE, '経験なし'), ('50', 'mixed50_d6', C_50, '経験50%'),
                                               ('80', 'mixed80_d6', C_80, '経験80%'))):
            for s in range(3):
                ys = [sp[f's{s}_{st}_{k}'][key]['mean'] for k, _ in steps]
                xs = np.arange(3) + (g - 1) * 0.22 + (s - 1) * 0.05
                ax.plot(xs, ys, color=c, marker=('o', 's', '^')[g], ms=3.2, lw=0.8,
                        label=lab if s == 0 else None)
                for k, _ in steps:
                    vals[f'split|{st}|{cn}|{pair_label(s)}|{k}'] = sp[f's{s}_{st}_{k}'][key]['mean']
        ax.set_xticks(range(3))
        ax.set_xticklabels([n for _, n in steps])
        ax.set_title(f'(b) {stname}：A-2 の一歩の種類ごと（組1〜3）', loc='left', fontsize=9)
        if col == 0:
            ax.set_ylabel('真値条件の誤差')
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(h[2:] + h[:2], l[2:] + l[:2], frameon=False, fontsize=6.8, loc='upper left', ncol=1)
    axs[1, 0].legend(frameon=False, fontsize=6.8, loc='center left', bbox_to_anchor=(0.0, 0.42))
    fig.tight_layout(h_pad=1.6)
    save(fig, 'projection_error')
    json.dump(vals, open(os.path.join(FIGS, 'projection_values.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
