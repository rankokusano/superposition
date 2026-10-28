"""
v15 summary figures for (A)-(D) (docs/v15_summary.md Sec.10), from saved results only (no training, no model evaluation). CPU.
    fig1: one generated pair (A-1's input views before / after eating, X and Y sides, true next targets) with the choice of
          2(b) v15(i) no experience seed 0 (blue / cyan, stored per pair in hide_<name>_cpu.json)
    fig2: pair accuracy on the A-2-hidden test: stage-0 observers, models trained without hiding, 2(b) shared, NS
    fig3: readout of A-2's preference from G's LSTM state, 0..10 steps after eating
    fig4: m (toward blue, mean of X / Y sides) and d (blue_X - blue_Y) per seed: v15(i) rank X / rank Y experienced, no experience

    python analyze/v15/figs_summary_v15.py   # -> data/result/v15/summary_figs/
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15 import eval_s2_hide_v15 as H  # noqa
from analyze.v15 import eval_s2_v15 as E  # noqa
from analyze.v15.judge_AB_v15 import S2, boot, per_pair  # noqa

OUT = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'summary_figs')
EXP, NOEXP, RANKY, BASE = '#2a78d6', '#eb6834', '#1baf7a', '#9a9a94'
LM_COL = {0: '#e3242b', 1: '#1aa31a', 2: '#2050e0', 3: '#14b8c8'}  # project Red / Green / Blue / Cyan (stage-0 figures)
INK, GRID = '#3d3d3a', '#e6e5e0'
SEEDS = (0, 1, 2)


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8)


def stem(base, s):
    return base + ('' if s == 0 else f'_s{s}')


def hide(base, s):
    n = stem(base, s)
    return json.load(open(os.path.join(S2, f"hide_{n if s == 0 else n + f'_seed{s}'}_cpu.json")))


# ------------------------------------------------------------------ fig 1
def fig1():
    both, te, pair, dropped = H.hidden_pairs('v15_s2_genx_p80', 'v15_s2_geny_p80')
    H2 = len(te) // 2
    r = hide('v15_s2p2h_noexp_v15i', 0)
    blue, member, rp = np.array(r['_blue_ev']), np.array(r['_member']), np.array(r['_pair'])
    k = 0  # the first kept pair, not selected by outcome
    iX, iY = k, k + H2
    assert pair[iX] == pair[iY] == k
    t_e = te[iX]
    choice = {s: blue[(rp == k) & (member == m)][0] for s, m in (('X', 'first'), ('Y', 'second'))}
    frames = [t_e - 8, t_e - 3, t_e, t_e + 3, t_e + 8]
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(2, 5, left=0.04, right=0.60, top=0.80, bottom=0.18, hspace=0.55, wspace=0.08)
    for row, (i, lab) in enumerate(((iX, 'X side (no swap)'), (iY, 'Y side (swap)'))):
        for c, t in enumerate(frames):
            a = fig.add_subplot(gs[row, c])
            a.imshow(both['self_vision_in'][i, t], interpolation='nearest', aspect=1.6)
            a.set_xticks([]); a.set_yticks([])
            same = np.array_equal(both['self_vision_in'][iX, t], both['self_vision_in'][iY, t])
            if row == 0:
                a.set_title(f"t = t_e{t - t_e:+d}" if t != t_e else 't = t_e (eats)', fontsize=8, color=INK)
            a.set_xlabel('X = Y' if same else 'X ≠ Y', fontsize=7, color=INK if same else '#c0392b')
            if c == 0:
                a.set_ylabel(lab, fontsize=8, color=INK)
    sub = fig.add_axes([0.64, 0.10, 0.34, 0.74])
    lay = both['layout'][iX]
    sub.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], color=BASE, lw=1)
    for c in range(4):
        x, y = E.CUBE_C[lay][c]
        col = LM_COL[c] if both['available'][iX, 0, c] else BASE
        sub.add_patch(plt.Rectangle((x - 3, y - 3), 6, 6, color=col, alpha=0.85))
    tl = E.CUBE_C[lay][0]  # X: red top-left; Y: green there (drawn as a split cube)
    sub.add_patch(plt.Rectangle((tl[0] - 3, tl[1] - 3), 3, 6, color=LM_COL[1], alpha=0.85))
    pos = both['other_position'][iX]
    sub.plot(pos[:t_e + 1, 0], pos[:t_e + 1, 1], color=INK, lw=2.5, alpha=0.4, label='A-2 until eating (same for X and Y)')
    p0 = pos[t_e]
    sub.plot(*p0, '*', color='white', ms=15, mec=INK, mew=1.0)
    for side, dy, col in (('X', -1.2, EXP), ('Y', 1.2, NOEXP)):
        tgt = E.CUBE_C[lay][2 if choice[side] else 3]
        d = E.unit(np.array(tgt) - p0) * 7
        sub.annotate('', xy=(p0[0] + d[0], p0[1] + d[1] + dy), xytext=(p0[0], p0[1] + dy),
                     arrowprops=dict(arrowstyle='-|>', color=col, lw=2.2))
    for c, name in ((2, 'X: correct = blue'), (3, 'Y: correct = cyan')):
        x, y = E.CUBE_C[lay][c]
        sub.add_patch(plt.Circle((x, y), 4.6, fill=False, ec=INK, lw=1.2, ls='--'))
        sub.text(x - 5.2, y, name.replace(': ', ':\n'), ha='right', va='center', fontsize=8, color=INK)
    sub.text(tl[0] - 3, tl[1] + 3.8, 'top-left: red (X) / green (Y), eaten at t_e', ha='left', va='bottom', fontsize=7.5, color=INK)
    from matplotlib.lines import Line2D
    sub.legend(handles=[Line2D([], [], color=INK, lw=2.5, alpha=0.4, label='A-2 until eating (same in X and Y)'),
                        Line2D([], [], color=EXP, lw=2.2, label=f"choice on X side: {'blue' if choice['X'] else 'cyan'}"),
                        Line2D([], [], color=NOEXP, lw=2.2, label=f"choice on Y side: {'blue' if choice['Y'] else 'cyan'}")],
               fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=1)
    sub.set_xlim(-17, 18.6); sub.set_ylim(-17, 19); sub.set_aspect('equal'); sub.set_xticks([]); sub.set_yticks([])
    sub.set_title('top view; arrows: choice of 2(b) v15(i) no experience, seed 0', fontsize=8.5, pad=4)
    fig.text(0.32, 0.93, f"Generated pair {k} (first kept pair; A-2 eats at t_e = {t_e}). A-1's input views, A-2 hidden from t_e-5:\n"
             'X and Y sides differ only before eating (top-left cube red vs green)', ha='center', fontsize=9.5, color=INK)
    fig.savefig(os.path.join(OUT, 'fig1_generated_pair.png'), dpi=130, bbox_inches='tight'); plt.close(fig)
    return {'pair': k, 't_e': int(t_e), 'choice_blue': {s: float(v) for s, v in choice.items()}, 'n_pairs': H2, 'dropped': dropped}


# ------------------------------------------------------------------ fig 2
def acc(base, s):
    return hide(base, s)['gen']['acc']


def fig2():
    s0 = json.load(open(os.path.join(_PI_DIR, 'data', 'result', 'v15', 'stage0', 'stage0_summary.json')))['generated']['summary']
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x, ticks, labs, groups = 0, [], [], []
    for key, lab in (('acc_a', '(a) direction\nextrapolated'), ('acc_b0', '(b0) position\n+ direction'), ('acc_d', '(d) ideal\nobserver')):
        ax.plot(x, s0[key], 'D', color=BASE, ms=8, mec=INK, mew=0.6)
        ticks.append(x); labs.append(lab); x += 1
    groups.append((0, x - 1, 'stage-0 observers\n(963 pairs)'))
    x += 0.6
    g0 = x
    for enc in ('v4', 'v15i'):
        for c, col in (('exp', EXP), ('noexp', NOEXP)):
            v = acc(f'v15_s2p2_{c}_{enc}', 0)
            ax.errorbar(x, v['mean'], yerr=[[v['mean'] - v['ci95'][0]], [v['ci95'][1] - v['mean']]], fmt='o', color=col, ms=7, capsize=3, lw=1.2)
            ticks.append(x); labs.append(f"{enc.replace('v15i', 'v15(i)')}\n{'exp' if c == 'exp' else 'no exp'}"); x += 1
    groups.append((g0, x - 1, 'trained without hiding\n(seed 0)'))
    for title, base, encs, mk in (('2(b) shared', 'v15_s2p2h_{c}_{enc}', ('v4', 'v15i'), 'o'),
                                  ('2(b) NS (not shared)', 'v15_s2p2h_{c}_{enc}_ns', ('v15i',), 's')):
        x += 0.6
        g0 = x
        for enc in encs:
            for c, col in (('exp', EXP), ('noexp', NOEXP)):
                for j, s in enumerate(SEEDS):
                    v = acc(base.format(c=c, enc=enc), s)
                    xx = x + (j - 1) * 0.22
                    ax.errorbar(xx, v['mean'], yerr=[[v['mean'] - v['ci95'][0]], [v['ci95'][1] - v['mean']]], fmt=mk, color=col,
                                ms=6, capsize=2, lw=1.1, mfc=col if mk == 'o' else 'white', mew=1.4)
                ticks.append(x); labs.append(f"{enc.replace('v15i', 'v15(i)')}\n{'exp' if c == 'exp' else 'no exp'}"); x += 1
        groups.append((g0, x - 1, title + '\n(seeds 0, 1, 2)'))
    ax.axhline(0.5, color=INK, ls='--', lw=1)
    ax.text(x - 0.4, 0.505, 'chance 0.5', fontsize=7.5, color=INK, ha='right', va='bottom')
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylim(0.4, 1.06); ax.set_xlim(-0.6, x - 0.4)
    ax.set_ylabel('pair accuracy (generated pairs, A-2 hidden)', fontsize=9)
    for a, b, t in groups:
        ax.text((a + b) / 2, 1.075, t, ha='center', va='bottom', fontsize=8, color=INK, transform=ax.get_xaxis_transform() if False else ax.transData)
        ax.plot([a - 0.3, b + 0.3], [1.065, 1.065], color=INK, lw=0.8, clip_on=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=EXP, label='experienced (shared)'),
                       Line2D([], [], marker='o', ls='', color=NOEXP, label='no experience (shared)'),
                       Line2D([], [], marker='s', ls='', color=EXP, mfc='white', mew=1.4, label='experienced (NS)'),
                       Line2D([], [], marker='s', ls='', color=NOEXP, mfc='white', mew=1.4, label='no experience (NS)'),
                       Line2D([], [], marker='D', ls='', color=BASE, mec=INK, label='observer (no network)')],
              fontsize=7.5, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.93), ncol=1)
    style(ax)
    ax.set_title('Pair accuracy after eating (930 pairs; points = seeds, bars = 95% CI over pairs)', fontsize=10, pad=34)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_pair_accuracy.png'), dpi=130); plt.close(fig)


# ------------------------------------------------------------------ fig 3
def curve(base, s):
    g = hide(base, s)['gen']['memory_readout']['g']
    return np.array([[g[str(o)]['mean'], *g[str(o)]['ci95']] for o in range(11)])


def fig3():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    panels = (('2(b) shared, v15(i), seeds 0-2', 'v15_s2p2h_{c}_v15i', SEEDS),
              ('trained without hiding, v15(i), seed 0', 'v15_s2p2_{c}_v15i', (0,)),
              ('2(b) NS (not shared), v15(i), seeds 0-2', 'v15_s2p2h_{c}_v15i_ns', SEEDS))
    for a, (title, base, seeds) in zip(ax, panels):
        for c, col, lab in (('exp', EXP, 'experienced'), ('noexp', NOEXP, 'no experience')):
            for j, s in enumerate(seeds):
                v = curve(base.format(c=c), s)
                a.fill_between(range(11), v[:, 1], v[:, 2], color=col, alpha=0.10, lw=0)
                a.plot(range(11), v[:, 0], '-', color=col, lw=1.8, marker='o', ms=3.5, label=lab if j == 0 else None)
        a.axhline(0.5, color=INK, ls='--', lw=1)
        a.set_title(title, fontsize=9); a.set_xlabel('steps after eating (A-2 hidden)', fontsize=8.5)
        a.set_xticks(range(0, 11, 2)); style(a)
    ax[0].set_ylabel("readout of A-2's preference (X / Y)\nfrom G's LSTM state", fontsize=8.5)
    ax[0].set_ylim(0.35, 1.0)
    ax[0].legend(fontsize=8, frameon=False, loc='lower left')
    ax[2].text(10, 0.505, 'chance 0.5', fontsize=7.5, color=INK, ha='right', va='bottom')
    fig.suptitle('Keeping the preference after eating: readout accuracy (lines = seeds, bands = 95% CI)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_g_readout_curves.png'), dpi=130); plt.close(fig)


# ------------------------------------------------------------------ fig 4
def fig4():
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    out = {}
    for base, col, mk, lab in (('v15_s2p2h_exp_v15i', EXP, 'o', 'rank X experienced (own next target: blue)'),
                               ('v15_s2p2h_expY_v15i', RANKY, 's', 'rank Y experienced (own next target: cyan)'),
                               ('v15_s2p2h_noexp_v15i', NOEXP, '^', 'no experience')):
        for s in SEEDS:
            p = per_pair(stem(base, s), s)
            m, d = boot(p['m']), boot(p['d'])
            out[f'{base}_seed{s}'] = {'m': m, 'd': d}
            ax.errorbar(m['mean'], d['mean'], xerr=[[m['mean'] - m['ci95'][0]], [m['ci95'][1] - m['mean']]],
                        yerr=[[d['mean'] - d['ci95'][0]], [d['ci95'][1] - d['mean']]], fmt=mk, color=col, ms=8, capsize=2.5,
                        lw=1.1, mec='white', mew=0.8, label=lab if s == 0 else None)
            ax.annotate(f's{s}', (m['mean'], d['mean']), xytext=(6, 4), textcoords='offset points', fontsize=7.5, color=INK)
    ax.axvline(0.5, color=BASE, ls=':', lw=1)
    ax.set_xlabel('m = share predicted toward blue, mean of X and Y sides\n(lower = toward cyan)', fontsize=8.5)
    ax.set_ylabel('d = toward blue on X side − on Y side\n(distinction between X and Y)', fontsize=8.5)
    ax.set_xlim(0.30, 0.62); ax.set_ylim(0.15, 0.85)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(colors=INK, labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    ax.set_title('v15(i), 2(b): where the prediction leans (m) and how much X and Y differ (d)\n'
                 '930 pairs, points = seeds, bars = 95% CI over pairs', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_m_d_by_rank.png'), dpi=130); plt.close(fig)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {'fig1': fig1()}
    fig2(); fig3()
    R['fig4'] = fig4()
    json.dump(R, open(os.path.join(OUT, 'figs_summary_values.json'), 'w'), indent=1)
    print(json.dumps(R['fig1']))


if __name__ == '__main__':
    main()
