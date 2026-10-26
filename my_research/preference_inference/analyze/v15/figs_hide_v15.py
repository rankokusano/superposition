"""
v15 Sec.28.2 (descriptive): X / Y split and preference-memory curves on the A-2-hidden test, from the CPU re-runs
data/result/v15/stage2/hide_<exp>_cpu.json of the 2(b) models (v15_s2p2h_*) and the models trained without hiding
(v15_s2p2_*). Paired differences (experienced - no experience) over the same pairs, per side (X members / Y members),
2000 bootstrap draws.

    python analyze/v15/figs_hide_v15.py   # -> data/result/v15/stage2/hide_xy_summary.json, figs/s2b_*.png
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S2 = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'stage2')
FIG = os.path.join(S2, 'figs')
MODELS = [(f'v15_s2p2{h}_{c}_{e}', h, c, e) for h in ('h', '') for e in ('v4', 'v15i') for c in ('exp', 'noexp')]
COL = {'exp': '#2a78d6', 'noexp': '#eb6834'}
INK = '#3d3d3a'


def load(name):
    r = json.load(open(os.path.join(S2, f'hide_{name}_cpu.json')))
    acc = np.array(r['_acc_ev'], float); pair = np.array(r['_pair']); member = np.array(r['_member']); blue = np.array(r['_blue_ev'], float)
    return r, acc, pair, member, blue


def boot(v, seed=0):
    v = v[~np.isnan(v)]
    rng = np.random.RandomState(seed)
    d = [v[rng.randint(len(v), size=len(v))].mean() for _ in range(2000)]
    return {'mean': float(v.mean()), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'n': int(len(v))}


def side(arr, pair, member, which):
    """per pair, the value of the X ('first') or Y ('second') member, ordered by pair id."""
    m = member == which
    order = np.argsort(pair[m])
    return arr[m][order]


def main():
    os.makedirs(FIG, exist_ok=True)
    D = {m[0]: load(m[0]) for m in MODELS}
    S = {}
    for name, h, c, e in MODELS:
        r, acc, pair, member, blue = D[name]
        S[name] = {'acc_all': boot(acc), 'acc_X': boot(side(acc, pair, member, 'first')), 'acc_Y': boot(side(acc, pair, member, 'second')),
                   'blue_X': boot(side(blue, pair, member, 'first')), 'blue_Y': boot(side(blue, pair, member, 'second')),
                   'g_curve': [r['gen']['memory_readout']['g'][str(o)] for o in range(11)],
                   'h2_curve': [r['gen']['memory_readout']['h2'][str(o)] for o in range(11)]}
    for h in ('h', ''):
        for e in ('v4', 'v15i'):
            _, ae, pe, me, _ = D[f'v15_s2p2{h}_exp_{e}']
            _, an, pn, mn, _ = D[f'v15_s2p2{h}_noexp_{e}']
            assert np.array_equal(np.sort(pe), np.sort(pn))
            S[f'diff{h}_{e}'] = {w: boot(side(ae, pe, me, s) - side(an, pn, mn, s)) for w, s in (('X', 'first'), ('Y', 'second'))}
    json.dump(S, open(os.path.join(S2, 'hide_xy_summary.json'), 'w'), indent=1)

    # figure 1: X / Y accuracy and toward-blue, exp vs noexp, 2(b) and no-hiding, both encoders
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    groups = [(h, e) for h in ('h', '') for e in ('v4', 'v15i')]
    glab = {('h', 'v4'): '2(b) v4', ('h', 'v15i'): '2(b) v15i', ('', 'v4'): 'no hiding v4', ('', 'v15i'): 'no hiding v15i'}
    for k, (key, title, hline) in enumerate((('acc', 'pair-member accuracy: X side (left) / Y side (right)', 0.5),
                                             ('blue', "predicted toward blue (A-1's own next): X / Y side", 0.5))):
        for j, g in enumerate(groups):
            for i, c in enumerate(('exp', 'noexp')):
                for t, w in enumerate(('X', 'Y')):
                    v = S[f'v15_s2p2{g[0]}_{c}_{g[1]}'][f'{key}_{w}']
                    x = j * 2.2 + t * 0.9 + (i - 0.5) * 0.36
                    ax[k].bar(x, v['mean'], 0.33, color=COL[c], alpha=1.0 if w == 'X' else 0.55,
                              label=(f'{c} ({w} side)' if j == 0 else None))
                    ax[k].errorbar(x, v['mean'], yerr=[[v['mean'] - v['ci95'][0]], [v['ci95'][1] - v['mean']]], color=INK, lw=1, capsize=2)
        ax[k].axhline(hline, color=INK, ls='--', lw=1)
        ax[k].set_xticks([j * 2.2 + 0.45 for j in range(4)]); ax[k].set_xticklabels([glab[g] for g in groups], fontsize=8)
        ax[k].set_title(title, fontsize=9); ax[k].set_ylim(0, 1.05)
        ax[k].spines[['top', 'right']].set_visible(False); ax[k].grid(axis='y', color='#e6e5e0', lw=0.8); ax[k].set_axisbelow(True)
    for j, g in enumerate(groups):
        for t, w in enumerate(('X', 'Y')):
            v = S[f'diff{g[0]}_{g[1]}'][w]
            x = j * 2.2 + t * 0.9
            ax[2].bar(x, v['mean'], 0.6, color='#1baf7a' if w == 'X' else '#8fd6bc')
            ax[2].errorbar(x, v['mean'], yerr=[[v['mean'] - v['ci95'][0]], [v['ci95'][1] - v['mean']]], color=INK, lw=1, capsize=3)
            ax[2].text(x, v['mean'] + (0.02 if v['mean'] >= 0 else -0.05), f"{w} {v['mean']:+.2f}", ha='center', fontsize=7)
    ax[2].axhline(0, color=INK, lw=1)
    ax[2].set_xticks([j * 2.2 + 0.45 for j in range(4)]); ax[2].set_xticklabels([glab[g] for g in groups], fontsize=8)
    ax[2].set_title('experienced - no experience, per side (paired, 95% CI)', fontsize=9)
    ax[2].spines[['top', 'right']].set_visible(False); ax[2].grid(axis='y', color='#e6e5e0', lw=0.8)
    ax[0].legend(fontsize=7, frameon=False, ncol=2, loc='upper left')
    fig.suptitle('A-2-hidden test on the generated pairs (930 pairs; X side: correct = blue, Y side: correct = cyan)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 's2b_xy_split.png'), dpi=120); plt.close(fig)

    # figure 2: preference readout curves (G and h2), 8 models
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    ls = {('h', 'v4'): '-', ('h', 'v15i'): '-', ('', 'v4'): ':', ('', 'v15i'): ':'}
    mk = {'v4': 'o', 'v15i': 's'}
    for j, key in enumerate(('g_curve', 'h2_curve')):
        for name, h, c, e in MODELS:
            cur = S[name][key]
            m = [v['mean'] for v in cur]
            lo = [v['ci95'][0] for v in cur]; hi = [v['ci95'][1] for v in cur]
            ax[j].plot(range(11), m, ls[(h, e)], color=COL[c], marker=mk[e], ms=4, lw=1.6,
                       label=f"{'2(b)' if h else 'no hiding'} {e} {c}")
            ax[j].fill_between(range(11), lo, hi, color=COL[c], alpha=0.08)
        ax[j].axhline(0.5, color=INK, ls='--', lw=1)
        ax[j].set_xlabel('steps after eating (A-2 hidden from 5 steps before eating)')
        ax[j].set_title("A-2's preference (X/Y) read from " + ("G's LSTM state" if key == 'g_curve' else 'h2'), fontsize=9)
        ax[j].spines[['top', 'right']].set_visible(False); ax[j].grid(axis='y', color='#e6e5e0', lw=0.8)
    ax[0].set_ylabel('readout accuracy (generated pairs)'); ax[1].legend(fontsize=7, frameon=False, loc='lower right', ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 's2b_memory_curves.png'), dpi=120); plt.close(fig)
    for name, *_ in MODELS:
        s = S[name]
        print(name, 'acc X/Y', round(s['acc_X']['mean'], 3), round(s['acc_Y']['mean'], 3), 'blue X/Y', round(s['blue_X']['mean'], 3), round(s['blue_Y']['mean'], 3),
              'G10', round(s['g_curve'][10]['mean'], 3))
    for k in [k for k in S if k.startswith('diff')]:
        print(k, {w: (round(v['mean'], 3), [round(x, 3) for x in v['ci95']]) for w, v in S[k].items()})


if __name__ == '__main__':
    main()
