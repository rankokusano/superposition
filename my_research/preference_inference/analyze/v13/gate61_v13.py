"""
v13 data check (docs/v13_instructions.md Sec.4.1; the v11 gate 6.1 items of
docs/v11_experiment_log.md Sec.5.1 applied to v13_s1_mixed80, side by side
with v11_s1_random and v11_s1_mixed (50%)). Reads only; writes under
data/result/v13/gate61/. Definitions (red reference, near < 5, source codes,
rule recomputation) are those of analyze/v11/gate61_v11.py, imported.

  1 reproducibility: check_mixed80_a == check_mixed80_b == first 3 train eps
  2 pairing with v11_s1_random (decides whether the v12 no-experience models
    can be reused): layout, initial A-1 position, A-2 positions (all t) and
    the sleep frames (a1_source == 3) equal per episode, train and test
  4 rule: time within 5 of red per layout, first time within 5, a1_source
    fractions (rule turns = sources 1 + 2 should be ~0.80 of awake steps),
    speed, sleep fraction, rule-direction recomputation from the saved view
  5 basic stats: episodes per layout, pleasure per bin
"""

import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
sys.path.insert(0, os.path.join(_PI_DIR, 'analyze', 'v11'))
import gate61_v11 as g11  # noqa
from sim_v10 import layout as L  # noqa
from v11core.pleasure import color_direction, pleasure_np  # noqa
from v11core.rule_agent import SRC_RULE, SRC_RULE_FALLBACK, SRC_SLEEP, slide_step  # noqa

D = os.path.join(_PI_DIR, 'data')
OUT = os.path.join(D, 'result', 'v13', 'gate61')
SETS = {'random': 'v11_s1_random', 'mixed50': 'v11_s1_mixed', 'mixed80': 'v13_s1_mixed80'}
P_RULE = {'random': 0.0, 'mixed50': 0.5, 'mixed80': 0.8}


def h5(name):
    return h5py.File(os.path.join(D, 'data', SETS[name], 'data.h5'), 'r')


def item1():
    a = h5py.File(os.path.join(OUT, 'check_mixed80_a.h5'), 'r')['train']
    b = h5py.File(os.path.join(OUT, 'check_mixed80_b.h5'), 'r')['train']
    full = h5('mixed80')['train']
    res = {'a_eq_b': all(np.array_equal(a[k][()], b[k][()]) for k in g11.KEYS),
           'a_eq_full_first3': all(np.array_equal(a[k][()], full[k][:3]) for k in g11.KEYS)}
    res['pass'] = res['a_eq_b'] and res['a_eq_full_first3']
    return res


def item2():
    res = {}
    fr = h5('random')
    for other in ('mixed80', 'mixed50'):
        fo = h5(other)
        r_o = {}
        for split in ('train', 'test'):
            r, o = fr[split], fo[split]
            r_o[split] = {
                'n_episodes': int(o['layout'].shape[0]),
                'layout_eq': bool(np.array_equal(r['layout'][()], o['layout'][()])),
                'init_self_pos_eq': bool(np.array_equal(r['self_position'][:, 0], o['self_position'][:, 0])),
                'other_pos_eq_all_t': bool(np.array_equal(r['other_position'][()], o['other_position'][()])),
                'sleep_frames_eq': bool(np.array_equal(r['a1_source'][()] == SRC_SLEEP,
                                                       o['a1_source'][()] == SRC_SLEEP)),
            }
        r_o['pass'] = all(v['layout_eq'] and v['init_self_pos_eq'] and v['other_pos_eq_all_t']
                          and v['sleep_frames_eq'] for v in r_o.values())
        res[f'random_vs_{other}'] = r_o
    with h5py.File(os.path.join(D, 'data', SETS['mixed80'], 'data.h5'), 'r') as f:
        res['mixed80_attrs'] = {k: (v.item() if hasattr(v, 'item') else str(v)) for k, v in f.attrs.items()}
    return res


def item4_5():
    res, occ, near_t, pls = {}, {}, {}, {}
    for c in SETS:
        g = h5(c)['train']
        pos = g['self_position'][()]
        lay = g['layout'][()]
        src = g['a1_source'][()]
        mot = g['self_motion'][()]
        red = np.stack([g11.RED_REF[int(l)] for l in lay])[:, None, :]
        near = np.linalg.norm(pos - red, axis=-1) < 5
        sp = np.linalg.norm(mot, axis=-1)
        awake = src != SRC_SLEEP
        fr_aw = np.bincount(src[awake].ravel(), minlength=4)[:3] / awake.sum()
        r = {
            'near_red5': {g11.LAYNAME[k]: float(near[lay == k].mean()) for k in (0, 1)},
            'near_red5_all': float(near.mean()),
            'near_red5_t50_100': float(near[:, 50:].mean()),
            'source_frac_of_awake': {'random': float(fr_aw[0]), 'rule': float(fr_aw[1]),
                                     'rule_fallback': float(fr_aw[2])},
            'rule_turn_frac_of_awake': float(fr_aw[1] + fr_aw[2]),
            'sleep_frac': float((~awake).mean()),
            'awake_speed_ne1_frac': float((np.abs(sp[awake] - 1) > 1e-4).mean()),
            'n_episodes_per_layout': {g11.LAYNAME[k]: int((lay == k).sum()) for k in (0, 1)},
        }
        start_far = ~near[:, 0]
        ft = np.where(near.any(1), near.argmax(1), 101)[start_far]
        r['first_within5_median'] = float(np.median(ft))
        r['never_within5_frac'] = float((~near.any(1))[start_far].mean())
        # mean distance to the red reference in the second half
        r['dist_red_t50_100_mean'] = float(np.linalg.norm(pos - red, axis=-1)[:, 50:].mean())
        if c != 'random':
            sv = g['self_vision']
            n_chk, n_ok, maxerr = 0, 0, 0.0
            for n in range(0, pos.shape[0], 10):
                idx = np.where(src[n] == SRC_RULE)[0]
                if len(idx) == 0:
                    continue
                v_ep = sv[n]
                for t in idx:
                    d = color_direction(v_ep[t], 'Red')
                    d = slide_step(pos[n, t].astype(np.float64), d, g11.BOUNDS, 1.0) if d is not None else None
                    n_chk += 1
                    if d is not None:
                        e = float(np.abs(d - mot[n, t]).max())
                        maxerr = max(maxerr, e)
                        n_ok += e < 1e-4
            r['rule_recompute'] = {'checked': n_chk, 'match_1e-4': n_ok, 'max_abs_err': maxerr}
        sv = g['self_vision']
        P = np.concatenate([pleasure_np(sv[i:i + 200]) for i in range(0, pos.shape[0], 200)])
        pls[c] = (P, lay)
        r['pleasure'] = {
            'per_bin_mean': {g11.LAYNAME[k]: P[lay == k].mean((0, 1)).round(4).tolist() for k in (0, 1)},
            'per_bin_frac_positive': {g11.LAYNAME[k]: (P[lay == k] > 0).mean((0, 1)).round(4).tolist() for k in (0, 1)},
            'liked_bins_mean_positive_part': float(np.clip(P, 0, None).sum(-1).mean()),
        }
        res[c] = r
        occ[c] = (pos, lay)
        near_t[c] = near.mean(0)
    res['pass_item4'] = {
        'near_red_80_gt_50_gt_random': all(
            res['mixed80']['near_red5'][k] > res['mixed50']['near_red5'][k] > res['random']['near_red5'][k]
            for k in ('original', 'swap_rg')),
        'rule_turn_frac_in_0.75_0.85': 0.75 <= res['mixed80']['rule_turn_frac_of_awake'] <= 0.85,
        'speed_ne1_le_1pct': all(res[c]['awake_speed_ne1_frac'] <= 0.01 for c in SETS),
        'sleep_frac_diff_lt_1pt': abs(res['mixed80']['sleep_frac'] - res['random']['sleep_frac']) < 0.01,
    }

    edges = np.linspace(-10, 10, 21)
    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    for i, c in enumerate(SETS):
        pos, lay = occ[c]
        for j in (0, 1):
            p = pos[lay == j].reshape(-1, 2)
            H, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=[edges, edges])
            ax = axes[i, j]
            im = ax.imshow((H / H.sum()).T, origin='lower', extent=[-10, 10, -10, 10],
                           cmap='viridis', vmin=0, vmax=0.02)
            ax.plot(*g11.RED_REF[j], 'o', color='red', mec='w', ms=10)
            ax.plot(*L.LANDMARKS[g11.LAYNAME[j]]['Green'], 'o', color='lime', mec='k', ms=10)
            ax.set_title(f'{SETS[c]} / {g11.LAYNAME[j]}  (near red<5: {res[c]["near_red5"][g11.LAYNAME[j]]:.3f})',
                         fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('v13 data check: A-1 occupancy (train, all frames; fraction per 1x1 cell, colour max 0.02)\n'
                 'rows: random / mixed 50% (v11) / mixed 80% (v13)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'gate61_v13_occupancy.png'), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for c in SETS:
        axes[0].plot(np.arange(101), near_t[c], label=f'{SETS[c]} (p_rule {P_RULE[c]})')
    axes[0].set_xlabel('t'); axes[0].set_ylabel('fraction of episodes within 5 of red')
    axes[0].set_title('time within 5 of red (train)'); axes[0].legend(fontsize=8)
    for j, c in enumerate(('mixed50', 'mixed80')):
        P, lay = pls[c]
        Pr, layr = pls['random']
        for k in (0, 1):
            axes[1 + j].plot(range(8), P[lay == k].reshape(-1, 8).mean(0), marker='o',
                             label=f'{g11.LAYNAME[k]} {c}')
            axes[1 + j].plot(range(8), Pr[layr == k].reshape(-1, 8).mean(0), ls='--', color='gray',
                             label=f'{g11.LAYNAME[k]} random' if k == 0 else None)
        axes[1 + j].axhline(0, color='k', lw=0.5)
        axes[1 + j].set_title(f'mean pleasure per bin ({SETS[c]}, train)')
        axes[1 + j].set_xlabel('bin k (centre -157.5+45k deg)'); axes[1 + j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'gate61_v13_time_pleasure.png'), dpi=130)
    plt.close(fig)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {'item1': item1(), 'item2': item2()}
    r = item4_5()
    out['item4_5'] = {c: r[c] for c in SETS}
    out['item4_pass'] = r['pass_item4']
    with open(os.path.join(OUT, 'gate61_v13.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
