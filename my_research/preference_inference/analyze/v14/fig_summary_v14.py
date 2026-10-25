"""
v14 summary figures (docs/v14_summary.md), from the saved judgement / check JSONs (no model runs). Output
data/result/v14/summary_figs/:
  fig1_ba_vs_T.png        v14-A BA vs T (pref test, self-vision L1, l1 late-5 mean, bars = range): v12 exp, v13 exp,
                          v12 no-exp, v13-NS exp; one panel per pair s0..s3; chance line
  fig2_follow_pref.png    share of choices following preference among episodes where preference and A-2's direction
                          disagree (n = 88, ep200, 95% CI): FP and self-vision L1; v13 (s0..s3), v12 exp, v12 no-exp, NS exp
  fig3_write_vs_choose.png  same models (v12 / v13 experienced): agreement with the correct candidate and share following
                          preference in disagreeing episodes, for "choose" (v14-A, L1 and FP) and "write out" (v14-B, u)
  fig4_ns_bias.png        share choosing c+ per cell (layout x A-2 liked), v13-NS exp / no-exp and v13 exp, L1 and FP
  fig5_u_timeseries.png   v14-B: mean u over time per cell (v12 exp / no-exp, v13 exp), and sign agreement per layout
  fig6_v14c_gates.png     v14-C gates by A-1's liked colour (all / Red / Green), experienced vs v12 no-exp vs no-exp version
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
R = os.path.join(_PI, 'data', 'result', 'v14')
OUT = os.path.join(R, 'summary_figs')
TS = (5, 10, 20, 30, 50, 100)
EP = ('160', '170', '180', '190', '200')
C = {'v13': '#2a78d6', 'v12': '#1baf7a', 'noexp': '#eb6834', 'ns': '#e87ba4', 'nsno': '#b0a9a0', 'v14c': '#eda100'}
LAB = {'v13': 'v13 experienced (80%)', 'v12': 'v12 experienced (50%)', 'noexp': 'no experience (v12)',
       'ns': 'v13-NS experienced (no sharing)', 'nsno': 'v13-NS no experience'}
CELLS = ('L0 Green\n(correct c-)', 'L0 Red\n(correct c+)', 'L1 Green\n(correct c+)', 'L1 Red\n(correct c-)')


def J(*p):
    return json.load(open(os.path.join(R, *p)))


def ba_curve(detail, pair, key):
    a = np.array([[detail[pair][f'pref_L1_T{T}']['points'][e][key] for e in EP] for T in TS])
    return a.mean(1), a.min(1), a.max(1)


def fig1():
    v13 = J('inverse', 'aggregate_v13_s0s1s2s3.json')['detail']
    v12 = J('inverse', 'aggregate_v12_s0s1s2s3.json')['detail']
    ns = J('inverse', 'aggregate_v13_vs_ns.json')['detail']
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.3), sharey=True)
    x = np.arange(len(TS))
    for s, ax in enumerate(axs):
        rows = [('v13', v13, f'v13:mixed:{s},v12:random:{s}', 'ba_exp'),
                ('v12', v12, f'v12:mixed:{s},v12:random:{s}', 'ba_exp'),
                ('noexp', v12, f'v12:mixed:{s},v12:random:{s}', 'ba_noexp')]
        if s < 3:
            rows.append(('ns', ns, f'v13:mixed:{s},v13ns:mixed:{s}', 'ba_noexp'))
        for k, d, pr, key in rows:
            m, lo, hi = ba_curve(d, pr, key)
            ax.errorbar(x, m, yerr=[m - lo, hi - m], color=C[k], lw=2, marker='o', ms=5, capsize=3, label=LAB[k])
        ax.axhline(0.5, color='0.3', lw=0.8)
        ax.axvline(x[TS.index(50)], color='0.88', lw=8, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels([str(T) for T in TS]); ax.set_xlabel('T (frames accumulated)')
        n13 = v13[f'v13:mixed:{s},v12:random:{s}']['pref_L1_T50']['n_pass']
        ax.set_title(f'pair s{s}{" (spare)" if s == 3 else ""}: v13 passing points {n13}/5' + ('' if s < 3 else '\n(no NS run for s3)'), fontsize=10)
    axs[0].set_ylabel('balanced accuracy (choose c+ / c-)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.suptitle('v14-A: inferring A-2\'s preference from A-1\'s view by the cumulative self-vision prediction error '
                 '(pref test; mean over l1 late-5, bars = range; grey band = registered T = 50)', fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, 'fig1_ba_vs_T.png'), dpi=120)


def follow(d, key):
    r = d[key]
    return r['disagree_follow_pref'], r['disagree_follow_pref_ci']


def fig2():
    v13 = J('inverse', 'fp_direction_v13l1.json'); v13.update(J('inverse', 'fp_direction_v13l1_s3.json'))
    v12 = J('inverse', 'fp_direction.json')
    ns = J('inverse', 'fp_direction_nsl1.json')
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for ax, q, title in zip(axs, ('FP', 'L1'), ('choosing by the other-side feature-prediction error (FP)',
                                                 'choosing by the self-vision error (L1, main of v14-A)')):
        for s in range(4):
            rows = [('v13', v13, f'v13_mixed_{s}|{q}|all|ep200'), ('v12', v12, f'mixed_s{s}|{q}|all|ep200'),
                    ('noexp', v12, f'random_s{s}|{q}|all|ep200')]
            if s < 3:
                rows.append(('ns', ns, f'v13ns_mixed_{s}|{q}|all|ep200'))
            for j, (k, d, key) in enumerate(rows):
                m, c = follow(d, key)
                xx = s + (j - 1.5) * 0.18
                ax.errorbar(xx, m, yerr=[[m - c[0]], [c[1] - m]], color=C[k], marker='o', ms=6, capsize=3, lw=1.5,
                            label=LAB[k] if s == 0 else None)
        ax.axhline(0.5, color='0.3', lw=0.8, ls='--')
        ax.set_xticks(range(4)); ax.set_xticklabels(['pair s0', 'pair s1', 'pair s2', 'pair s3 (spare)'])
        ax.set_title(title, fontsize=10)
    axs[0].set_ylabel('share following preference')
    axs[1].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Episodes where A-2\'s preference and its direction over t = 1-50 disagree (n = 88): '
                 'does the choice follow the preference? (ep200, 95% CI; > 0.5 = preference)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_follow_pref.png'), dpi=120)


def fig3():
    A = {'v12': J('inverse', 'aggregate_v12_s0s1s2.json')['detail'], 'v13': J('inverse', 'aggregate_v13_s0s1s2.json')['detail']}
    Bu = {'v12': J('stage2u', 'u', 'aggregate_u_s0s1s2.json')['detail'], 'v13': J('stage2u', 'u', 'aggregate_u_v13.json')['detail']}
    Ad = {'v12': J('inverse', 'fp_direction.json'), 'v13': J('inverse', 'fp_direction_v13l1.json')}
    Bd = {'v12': J('stage2u', 'u_direction.json'), 'v13': J('stage2u', 'u_direction_v13.json')}
    akey = {'v12': lambda s: f'mixed_s{s}', 'v13': lambda s: f'v13_mixed_{s}'}
    bkey = {'v12': lambda s: f'mixed_s{s}', 'v13': lambda s: f'mixed80_s{s}'}
    meth = (('choose (v14-A), self-vision L1', '#2a78d6', 'o'), ('choose (v14-A), FP', '#1baf7a', 's'),
            ('write out (v14-B), one-bit u', '#eb6834', 'D'))
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharey='row')
    for col, mdl in enumerate(('v12', 'v13')):
        for s in range(3):
            pr = f'{mdl}:mixed:{s},v12:random:{s}'
            agree = [(A[mdl][pr]['pref_L1_T50']['points']['200']['ba_exp'], A[mdl][pr]['pref_L1_T50']['points']['200']['ba_exp_ci']),
                     (A[mdl][pr]['pref_FP_T50']['points']['200']['ba_exp'], A[mdl][pr]['pref_FP_T50']['points']['200']['ba_exp_ci']),
                     (Bu[mdl][f's{s}']['points']['200']['u_ba_exp'], Bu[mdl][f's{s}']['points']['200']['u_ba_exp_ci'])]
            fol = [follow(Ad[mdl], f'{akey[mdl](s)}|L1|all|ep200'), follow(Ad[mdl], f'{akey[mdl](s)}|FP|all|ep200'),
                   follow(Bd[mdl], f'{bkey[mdl](s)}|ep200|all')]
            for row, vals in enumerate((agree, fol)):
                ax = axs[row, col]
                for j, ((m, c), (lab, cl, mk)) in enumerate(zip(vals, meth)):
                    xx = s + (j - 1) * 0.22
                    ax.errorbar(xx, m, yerr=[[m - c[0]], [c[1] - m]], color=cl, marker=mk, ms=7, capsize=3, lw=1.5,
                                label=lab if s == 0 else None)
        for row in range(2):
            ax = axs[row, col]
            ax.axhline(0.5, color='0.3', lw=0.8, ls='--')
            ax.set_xticks(range(3)); ax.set_xticklabels(['pair s0', 'pair s1', 'pair s2'])
        axs[0, col].set_title(f'{LAB[mdl]}', fontsize=10)
    axs[0, 0].set_ylabel('agreement with the correct candidate\n(balanced accuracy, ep200)')
    axs[1, 0].set_ylabel('share following preference where\npreference and direction disagree (n = 88)')
    axs[0, 1].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Choosing the candidate by the network\'s own expectation (v14-A) vs. writing it out with a module (v14-B): '
                 'same stage-1 models, 95% CI', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_write_vs_choose.png'), dpi=120)


def fig4():
    ns = J('inverse', 'fp_check_ns.json')
    v13 = J('inverse', 'fp_check_v13.json')
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    for row, q in enumerate(('L1', 'FP')):
        for s in range(3):
            ax = axs[row, s]
            rows = [('v13', v13[f'v13_mixed_{s}|{q}|T50|ep200']['exp']), ('ns', ns[f'v13ns_mixed_{s}|{q}|T50|ep200']['exp']),
                    ('nsno', ns[f'v13ns_mixed_{s}|{q}|T50|ep200']['noexp'])]
            w = 0.26
            for j, (k, r) in enumerate(rows):
                v = list(r['frac_choose_cplus_by_cell'].values())
                ax.bar(np.arange(4) + (j - 1) * w, v, w * 0.9, color=C[k], label=LAB[k] if (row == 0 and s == 0) else None)
            for i, corr in enumerate((0, 1, 1, 0)):
                ax.plot([i - 0.45, i + 0.45], [corr, corr], color='0.2', lw=1.2, ls=':')
            ax.axhline(0.5, color='0.5', lw=0.8)
            ax.set_xticks(range(4)); ax.set_xticklabels(CELLS, fontsize=8)
            ax.set_title(f'pair s{s}, choosing by {"self-vision L1" if q == "L1" else "FP"}', fontsize=10)
        axs[row, 0].set_ylabel('share choosing c+ (ep200)')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Which candidate is chosen, per cell (layout x A-2\'s liked colour); dotted = perfect choice (1 = c+, 0 = c-). '
                 'NS shifts to one candidate in every cell; v13 switches with the cell', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, 'fig4_ns_bias.png'), dpi=120)


def fig5():
    import numpy as np  # noqa
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.2), sharey='row')
    cell_cols = ('#2a78d6', '#eb6834', '#1baf7a', '#e87ba4')
    for col, (tag, title) in enumerate((('mixed', 'v12 experienced'), ('random', 'v12 no experience'), ('mixed80', 'v13 experienced'))):
        ax = axs[0, col]
        z = np.load(os.path.join(R, 'stage2u', 'u', f'u_{tag}_s0_ep200.npz'))
        u, cl = z['pref_u'], z['pref_cell']
        for k in range(4):
            ax.plot(np.arange(1, 100), u[cl == k, 1:100].mean(0), color=cell_cols[k], lw=1.8, label=CELLS[k].replace('\n', ' '))
        ax.axhline(0, color='0.3', lw=0.8)
        ax.set_title(f'{title} (pair s0, ep200): mean u per cell', fontsize=10)
        ax.set_xlabel('t')
        ax = axs[1, col]
        for s in range(3):
            j = J('stage2u', 'u', f'u_{tag}_s{s}_ep200.json')['pref']
            ax.bar(s - 0.18, j['sign_agree_original'], 0.34, color='#2a78d6', label='layout 0' if s == 0 else None)
            ax.bar(s + 0.18, j['sign_agree_swap_rg'], 0.34, color='#eda100', label='layout 1' if s == 0 else None)
        ax.axhline(0.5, color='0.3', lw=0.8, ls='--')
        ax.set_xticks(range(3)); ax.set_xticklabels(['s0', 's1', 's2'])
        ax.set_title('sign agreement of Q2 with P_A2 (ep200)', fontsize=10)
    axs[0, 0].set_ylabel('u'); axs[1, 0].set_ylabel('share of frames x bins')
    axs[0, 0].legend(frameon=False, fontsize=8); axs[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('v14-B one-bit module G: u over time and the sign agreement (pref test)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_u_timeseries.png'), dpi=120)


def fig6():
    gv = J('gates', 'aggregate_gates_v12noexp.json')['per_seed']
    gn = J('gates', 'aggregate_gates_noexpv.json')['per_seed']
    items = (('s1_{sub}', 'M1c', 'M1(c) c_pred (< 0)'), ('s1_{sub}', 'M1d', 'M1(d) c_cf (> 0)'),
             ('s1_{sub}', 'colour_swap', 'colour S_L - S_D (> 0)'), ('s1_{sub}', 'G3', 'G3 g, input freeze (> 0)'))
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5))
    subs = ('all', 'red', 'green')
    for i, (rk, vk, title) in enumerate(items):
        ax = axs[0, i]
        for si, s in enumerate(('s0', 's1', 's2')):
            for j, sub in enumerate(subs):
                rows = gv[s]['rows'][rk.format(sub=sub)]
                m = np.mean([r['values'][vk]['mean'] for r in rows])
                ax.bar(si * 4 + j, m, color=('0.45', '#e8564c', '#3aa655')[j], label=('all episodes', 'A-1 likes Red', 'A-1 likes Green')[j] if si == 0 else None)
                if vk in ('G3', 'M1c', 'M1d'):
                    key = {'G3': 'G3_partner'}.get(vk)
                    if key:
                        mp = np.mean([r['values'][key]['mean'] for r in rows])
                        ax.plot([si * 4 + j - 0.35, si * 4 + j + 0.35], [mp, mp], color='k', lw=1.5)
        ax.axhline(0, color='0.3', lw=0.8)
        ax.set_xticks([1, 5, 9]); ax.set_xticklabels(['s0', 's1', 's2'])
        ax.set_title(f'{title}\nv14-C experienced' + (' (black bar = v12 no-exp)' if vk == 'G3' else ''), fontsize=9)
    axs[0, 0].legend(frameon=False, fontsize=8)
    # G5a (A-1 Red / Green) and G5b
    for i, (a1, setk) in enumerate((('red', 'green'), ('red', 'pref'), ('green', 'pref'))):
        ax = axs[1, i]
        for si, s in enumerate(('s0', 's1', 's2')):
            rows = gv[s]['rows'][f'g5a_a1{a1}']
            e = np.mean([r['values'][setk]['exp']['mean'] for r in rows])
            p = np.mean([r['values'][setk]['partner']['mean'] for r in rows])
            ax.bar(si * 3, e, color=C['v14c'], label='v14-C experienced' if si == 0 else None)
            ax.bar(si * 3 + 1, p, color=C['noexp'], label='v12 no experience' if si == 0 else None)
            if s in gn:
                q = np.mean([r['values'][setk]['partner']['mean'] for r in gn[s]['rows'][f'g5a_a1{a1}']])
                ax.plot([si * 3 + 1.6], [q], marker='D', color='0.2', ms=6, label='v14-C no-exp version' if si == 0 else None)
        ax.axhline(0, color='0.3', lw=0.8)
        ax.set_xticks([0.5, 3.5, 6.5]); ax.set_xticklabels(['s0', 's1', 's2'])
        ax.set_title(f'G5a g_other, set {"1 (green-liker)" if setk == "green" else "2 (pref)"}, A-1 likes {a1.capitalize()}', fontsize=9)
    axs[1, 0].legend(frameon=False, fontsize=8)
    ax = axs[1, 3]
    for si, s in enumerate(('s0', 's1', 's2')):
        for j, sub in enumerate(subs):
            rows = gv[s]['rows'][f'g5b_{sub}']
            ax.bar(si * 4 + j, np.mean([r['values']['exp']['mean'] for r in rows]), color=('0.45', '#e8564c', '#3aa655')[j])
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_xticks([1, 5, 9]); ax.set_xticklabels(['s0', 's1', 's2'])
    ax.set_title('G5b (+a vs -a), v14-C experienced\n(all / A-1 Red / A-1 Green episodes)', fontsize=9)
    fig.suptitle('v14-C (A-1\'s liked colour varies per episode): stage-1 gates, mean over the late-5 checkpoints. '
                 'Self side (top) is symmetric for Red / Green; other side (bottom) only s2 is clearly positive', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig6_v14c_gates.png'), dpi=120)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print(f.__name__, 'ok')
