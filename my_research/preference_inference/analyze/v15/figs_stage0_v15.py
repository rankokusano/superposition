"""
v15 stage-0 figures (docs/v15_instructions_stage0.md Sec.7) from
data/result/v15/stage0/stage0_full.pkl and the datasets.

    python analyze/v15/figs_stage0_v15.py   # -> data/result/v15/stage0/figs/
"""

import os
import pickle
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analyze.v15 import stage0_v15 as S  # noqa

FIG = os.path.join(S.OUT, 'figs')
PRED = ('a', 'b0', 'b1', 'b2', 'd', 'ds', 'c')
PRED_LABEL = {'a': '(a) extrapolate', 'b0': '(b0) position+dir', 'b1': '(b1) +layout,gray0', 'b2': '(b2) +available',
              'd': '(d) observer', 'ds': '(d) soft', 'c': '(c) true rank'}
PRED_COL = {'a': '#2a78d6', 'b0': '#eb6834', 'b1': '#eda100', 'b2': '#e87ba4', 'd': '#1baf7a', 'ds': '#008300', 'c': '#9a9a94'}
LM_COL = {0: '#e3242b', 1: '#1aa31a', 2: '#2050e0', 3: '#14b8c8'}  # project Red / Green / Blue / Cyan
GRAY = '#9a9a94'
INK = '#3d3d3a'


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8)


def fig_eats(R):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    series = [('v15_s1_p80', 'stage 1 A-1, p=80%'), ('v15_s2_c2_p80', 'stage 2 (2) A-2, p=80%'),
              ('v15_s2_c2_p90', '(2) A-2, p=90%'), ('v15_s2_c2_p50', '(2) A-2, p=50%'), ('v15_s1_p50', 'A-1, p=50%')]
    cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
    w = 0.16
    for i, (name, lab) in enumerate(series):
        h = np.array(R[name]['eat']['n_eat_hist'], float)
        h = h / h.sum()
        ax[0].bar(np.arange(len(h)) + (i - 2) * w, h, w * 0.9, color=cols[i], label=lab)
        ft = np.array(R[name]['eat']['first_times'])
        ft = ft[ft >= 0]
        ax[1].hist(ft, bins=np.arange(0, 102, 5), histtype='step', lw=2, color=cols[i], label=lab,
                   weights=np.ones(len(ft)) / 1000)
    ax[0].set_xlabel('eats per episode'); ax[0].set_ylabel('fraction of episodes')
    ax[1].set_xlabel('first eat frame t_e'); ax[1].set_ylabel('fraction of all episodes')
    ax[0].legend(fontsize=7, frameon=False)
    for a in ax:
        style(a)
    fig.suptitle('5.1 Eating: count per episode and time of the first eat (1000 episodes each)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f1_eats.png'), dpi=120); plt.close(fig)


def fig_turn(R):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for j, k in enumerate(S.KS):
        for name, lab, c in (('v15_s1_p80', 'stage 1 A-1 p80', '#2a78d6'), ('v15_s2_c2_p80', '(2) A-2 p80', '#eb6834')):
            a = np.array(R[name]['turn'][k]['angles'])
            ax[j].hist(a, bins=np.arange(0, 181, 10), histtype='step', lw=2, color=c, density=True,
                       label=f"{lab}: >45deg {np.mean(a > 45):.2f} (n={len(a)})")
        ax[j].axvline(45, color=GRAY, ls='--', lw=1)
        ax[j].set_title(f'k = {k}', fontsize=9); ax[j].set_xlabel('angle between pre and post displacement (deg)')
        ax[j].legend(fontsize=7, frameon=False); style(ax[j])
    ax[0].set_ylabel('density')
    fig.suptitle('5.2 Direction change at the eat moment', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f2_turn.png'), dpi=120); plt.close(fig)


def fig_pred(R):
    sets = [('v15_s2_c2_p80', '(2) A-2 p80 [main]'), ('v15_s2_c2_p90', '(2) A-2 p90'), ('v15_s2_c2_p50', '(2) A-2 p50'),
            ('v15_s2_c1_p80', '(1) A-2 p80'), ('v15_s1_p80', 'stage 1 A-1 p80')]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharey='row')
    w = 0.12
    for j, k in enumerate(S.KS):
        for i, t in enumerate(PRED):
            acc = [R[n]['pred'][k].get(f'acc_{t}', np.nan) for n, _ in sets]
            ang = [R[n]['pred'][k].get(f'angerr_median_{t}', np.nan) for n, _ in sets]
            x = np.arange(len(sets)) + (i - 3) * w
            ax[0, j].bar(x, acc, w * 0.88, color=PRED_COL[t], label=PRED_LABEL[t])
            ax[1, j].bar(x, [np.nan if v is None else v for v in ang], w * 0.88, color=PRED_COL[t])
        for r in (0, 1):
            ax[r, j].set_xticks(np.arange(len(sets)))
            ax[r, j].set_xticklabels([l for _, l in sets], fontsize=7, rotation=15)
            style(ax[r, j])
        ax[0, j].set_title(f'k = {k}' + (' (main)' if k == S.K_MAIN else ''), fontsize=9)
    ax[0, 0].set_ylabel('next-target accuracy'); ax[1, 0].set_ylabel('median angle error (deg)')
    ax[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle('5.3 Predicting the move right after eating (evaluation episodes 700-999)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f3_predictors.png'), dpi=120); plt.close(fig)


def fig_rise(R):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name, lab, c in (('v15_s2_c2_p80', '(2) p80', '#eb6834'), ('v15_s2_c2_p90', '(2) p90', '#1baf7a'),
                         ('v15_s2_c2_p50', '(2) p50', '#eda100'), ('v15_s1_p80', 'stage 1 A-1 p80', '#2a78d6')):
        r = R[name]['pred']['rise']
        a = np.array(sorted(int(k) for k in r if r[k][1] >= 20))
        ax.plot(a, [r[k][0] for k in a], color=c, lw=2, label=lab)
    ax.axhline(0.5, color=GRAY, ls='--', lw=1)
    ax.set_xlim(0, 30); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel('awake steps observed before the first eat'); ax.set_ylabel('posterior of the true candidate')
    ax.legend(fontsize=8, frameon=False); style(ax)
    ax.set_title('5.3 Ideal observer (d): how fast the rank is identified', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f4_posterior_rise.png'), dpi=120); plt.close(fig)


def draw_arena(ax, layout, avail0):
    ax.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], color=GRAY, lw=1)
    for c in range(4):
        x, y = S.CUBE_C[layout][c]
        ax.add_patch(plt.Rectangle((x - 3, y - 3), 6, 6, color=LM_COL[c] if avail0[c] else GRAY, alpha=0.85))
    ax.set_xlim(-17, 17); ax.set_ylim(-17, 17); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


def traj(ax, pos, eat, color, lw=1.5, label=None, upto=None):
    upto = len(pos) if upto is None else upto
    ax.plot(pos[:upto, 0], pos[:upto, 1], color=color, lw=lw, label=label)
    ax.plot(*pos[0], 'o', color=color, ms=5)
    for t in np.where(eat >= 0)[0]:
        if t + 1 < upto:
            ax.plot(*pos[t + 1], '*', color=LM_COL[int(eat[t])], ms=14, mec='k', mew=0.8)


def fig_examples():
    d = S.load('v15_s2_c2_p80', 'a2')
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.8))
    shown = 0
    for n in range(S.N_FIT, len(d['eat'])):
        if (d['eat'][n] >= 0).sum() < 2 or shown >= 8:
            continue
        a = ax.flat[shown]
        draw_arena(a, d['layout'][n], np.eye(4)[d['gray0'][n]] == 0)
        traj(a, d['pos'][n], d['eat'][n], INK)
        cand = 'X (R>B>C>G)' if d['a2_cand'][n] == 0 else 'Y (G>C>B>R)'
        a.set_title(f'ep {n}: {cand}', fontsize=8)
        shown += 1
    fig.suptitle('Top view, (2) A-2 p80: start o, eat * (colour of the eaten cube); gray cube = gray from the start',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f5_trajectories.png'), dpi=120); plt.close(fig)


def fig_pairs(R, key, nx, ny, title, fname, labs):
    dx, dy = S.load(nx, 'a2'), S.load(ny, 'a2')
    pairs = [p for p in R[key]['pairs']][:4]
    fig, ax = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4.3))
    for a, p in zip(np.atleast_1d(ax), pairs):
        n, te = p['n'], p['te']
        draw_arena(a, dx['layout'][n], np.eye(4)[dx['gray0'][n]] == 0)
        a.plot(dx['pos'][n, :te + 1, 0], dx['pos'][n, :te + 1, 1], color=INK, lw=3, alpha=0.35, label='shared until eat')
        for dd, c, lab in ((dx, '#2a78d6', labs[0]), (dy, '#eb6834', labs[1])):
            a.plot(dd['pos'][n, te:, 0], dd['pos'][n, te:, 1], color=c, lw=1.6, label=lab)
            for t in np.where(dd['eat'][n] >= 0)[0]:
                a.plot(*dd['pos'][n, min(t + 1, 100)], '*', color=LM_COL[int(dd['eat'][n, t])], ms=13, mec='k', mew=0.8)
        a.set_title(f'key {n}, eat at t={te}', fontsize=8)
    np.atleast_1d(ax)[0].legend(fontsize=7, frameon=False, loc='lower center')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=120); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    R = pickle.load(open(os.path.join(S.OUT, 'stage0_full.pkl'), 'rb'))
    fig_eats(R); fig_turn(R); fig_pred(R); fig_rise(R); fig_examples()
    fig_pairs(R, 'generated', 'v15_s2_genx_p80', 'v15_s2_geny_p80',
              'Main generated pairs: X (no swap, green gray) vs Y (swap, red gray); cube colours drawn for X',
              'f6_generated_pairs.png', ('X after eat', 'Y after eat'))
    fig_pairs(R, 'control3_p80', 'v15_s2_c3x_p80', 'v15_s2_c3xp_p80',
              'Control (3): X (R>B>C>G) vs X\' (R>C>B>G), green gray, same key',
              'f7_control_pairs.png', ('X after eat', "X' after eat"))
    print('figures in', FIG)


if __name__ == '__main__':
    main()
