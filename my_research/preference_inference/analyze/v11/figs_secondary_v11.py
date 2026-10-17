"""
v11 secondary-index figures (instructions Sec.7.4 row 6.4; docs/v11_experiment_log.md
Sec.7.18 items 8-12): for one checkpoint per seed,
  (1) M1(b') arrows: mean read-out displacement after 10 manipulated steps for
      +a (liked) and -a (disliked) in each of the 8 bins, per condition and seed;
  (2) M2 bars: registered angle criterion in the swap layout (bias direction vs
      new red / vs red's old place) and the proposal b_newRed - b_oldPlace, next
      to v10 stage A's A-1 critic (110.9 / 26.5 deg, v10_results_log Sec.3.2);
  (3) M5 bars: CV R^2 (r_t, G_t, cos/sin red direction) and cross-layout R^2
      (r_t, G_t only; the red-direction cross-layout R^2 is not interpretable,
      Sec.7.19);
  (4) M6 bars: h1 -> self, h2 -> A-2 position (CV R^2) and layout read-out
      accuracy over t.
Inputs: eval_m1_v11.py and m5m6_v11.py JSONs. Writes PNGs under data/result/v11/<sub>/.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V11 = os.path.join(_PI_DIR, 'data', 'result', 'v11')
BIN_ANGLES = np.deg2rad(-157.5 + 45.0 * np.arange(8))
V10_CRITIC = (110.9, 26.5)
CONDS = (('random', 'tab:gray'), ('mixed', 'tab:red'))


def load(p):
    with open(os.path.join(V11, p)) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m1_fmt', required=True)
    ap.add_argument('--m5m6_fmt', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--sub', default='main/secondary')
    ap.add_argument('--label', default='mse_ep200')
    args = ap.parse_args()
    out = os.path.join(V11, args.sub)
    os.makedirs(out, exist_ok=True)
    M1 = {s: load(args.m1_fmt.format(seed=s, epoch=args.epoch)) for s in args.seeds}
    M56 = {s: load(args.m5m6_fmt.format(seed=s, epoch=args.epoch)) for s in args.seeds}

    # (1) M1(b') arrows
    fig, axes = plt.subplots(2, len(args.seeds), figsize=(4.2 * len(args.seeds), 8.4), squeeze=False)
    for j, s in enumerate(args.seeds):
        for i, (c, _) in enumerate(CONDS):
            a_ = axes[i, j]
            b = M1[s][c]['M1b_prime']
            vp, vm = np.array(b['mean_vec_plus_step10']), np.array(b['mean_vec_minus_step10'])
            lim = max(1.0, np.abs(np.r_[vp, vm]).max() * 1.2)
            for k in range(8):
                u = np.array([np.cos(BIN_ANGLES[k]), np.sin(BIN_ANGLES[k])]) * lim * 0.95
                a_.plot([0, u[0]], [0, u[1]], color='0.85', lw=0.8)
                a_.text(u[0], u[1], str(k), fontsize=7, color='0.5', ha='center', va='center')
                a_.arrow(0, 0, vp[k, 0], vp[k, 1], color='tab:red', width=lim * 0.008, length_includes_head=True)
                a_.arrow(0, 0, vm[k, 0], vm[k, 1], color='tab:green', width=lim * 0.008, length_includes_head=True)
            a_.set_xlim(-lim, lim); a_.set_ylim(-lim, lim); a_.set_aspect('equal')
            a_.set_title(f'{c} s{s}: +a (red) / -a (green) per bin\nmain (d+ - d-) = {b["main_mean8"]["mean"]:+.3f}', fontsize=8)
    fig.suptitle(f"M1(b') (secondary): read-out displacement after 10 manipulated steps (a = 0.28), grey = bin directions; {args.label}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, f'm1b_arrows_{args.label}.png'), dpi=110); plt.close(fig)

    # (2) M2 bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = 0
    for s in args.seeds:
        for c, col in CONDS:
            a = M1[s][c]['M2_registered_angle']['swap_rg']
            axes[0].bar([x, x + 0.4], [a['to_red_deg'], a['to_other_place_deg']], width=0.38, color=[col, 'white'], edgecolor=col)
            p = M1[s][c]['M2_proposal_swap_newRed_minus_oldPlace']
            axes[1].bar(x, p['mean'], width=0.6, color=col, yerr=[[p['mean'] - p['ci95'][0]], [p['ci95'][1] - p['mean']]], capsize=3)
            x += 1
    axes[0].bar([x + 0.5, x + 0.9], V10_CRITIC, width=0.38, color=['tab:blue', 'white'], edgecolor='tab:blue')
    axes[0].set_title('M2 (secondary): swap layout, angle of bias to new red (filled) / to red\'s old place (open)\n'
                      'per seed: random (grey), mixed (red); right: v10 stage-A A-1 critic', fontsize=8)
    axes[0].set_ylabel('deg'); axes[0].set_xticks([])
    axes[1].axhline(0, color='k', lw=0.5); axes[1].set_xticks([])
    axes[1].set_title('M2 proposal: b(new red) - b(old place), swap episodes', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f'm2_bars_{args.label}.png'), dpi=110); plt.close(fig)

    # (3) M5 bars, (4) M6 bars
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    t5 = ['reward_r_t', 'return_G_t_gamma0.9', 'cos_red_dir', 'sin_red_dir']
    w = 0.12
    for j, s in enumerate(args.seeds):
        for i, (c, col) in enumerate(CONDS):
            off = (j * 2 + i) * w
            m = M56[s][c]
            axes[0].bar(np.arange(4) + off, [m['M5_cv_r2'][k] for k in t5], width=w, color=col, alpha=0.5 + 0.2 * j)
            cr = [np.mean(list(m['M5_cross_layout_r2'][k].values())) for k in t5[:2]]
            axes[1].bar(np.arange(2) + off, cr, width=w, color=col, alpha=0.5 + 0.2 * j)
            axes[2].bar(np.arange(2) + off, [m['M6_h1_to_self_cv_r2'], m['M6_h2_to_other_cv_r2']], width=w, color=col, alpha=0.5 + 0.2 * j)
            acc = m['M6_layout_readout_acc']
            axes[3].plot([int(k[1:]) for k in acc], list(acc.values()), 'o-', color=col, alpha=0.5 + 0.2 * j, label=f'{c} s{s}')
    axes[0].set_xticks(np.arange(4) + 2.5 * w); axes[0].set_xticklabels(['r_t (near-trivial)', 'G_t', 'cos red dir', 'sin red dir'], fontsize=8)
    axes[0].set_title('M5: CV R^2 from h1', fontsize=9)
    axes[1].set_xticks(np.arange(2) + 2.5 * w); axes[1].set_xticklabels(['r_t', 'G_t'], fontsize=8)
    axes[1].set_title('M5: cross-layout R^2 (mean of both directions)', fontsize=9)
    axes[2].set_xticks(np.arange(2) + 2.5 * w); axes[2].set_xticklabels(['h1 -> A-1 pos', 'h2 -> A-2 pos'], fontsize=8)
    axes[2].set_title('M6: CV R^2', fontsize=9)
    axes[3].set_xscale('log'); axes[3].axhline(0.5, color='k', lw=0.5, ls='--')
    axes[3].set_title('M6: layout read-out accuracy from h1', fontsize=9); axes[3].set_xlabel('t'); axes[3].legend(fontsize=6)
    for a_ in axes[:3]:
        a_.set_ylim(min(0, a_.get_ylim()[0]), 1)
    fig.suptitle(f'M5 / M6 (secondary), common random test; grey = random, red = mixed, shade = seed; {args.label}', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, f'm5m6_bars_{args.label}.png'), dpi=110); plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
