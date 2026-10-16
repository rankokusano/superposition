"""
v11 gate 6.1 (docs/v11_experiment_log.md Sec.5.1): data checks and figures.
Reads only; writes under data/result/v11/gate61/.

  1 reproducibility: check_<cond>_a == check_<cond>_b == first 3 train eps
  2 pairing: layout, initial A-1 position, A-2 positions equal per episode
  3 pleasure vs view figure (grid data, both layouts) + numpy==torch on all
    441 grid points of both layouts
  4 rule: time within 5 of red, occupancy heatmaps, a1_source fractions,
    speed distribution, sleep fraction, rule-direction recomputation from
    the saved uint8 view
  5 basic stats (episode counts, A-2 position map, pleasure distribution)
  6 M1(b) manipulation size a = 95th pct of positive pleasure values
    (v11_s1_random train, all frames, all bins), rounded to 2 decimals
"""

import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from sim_v10 import layout as L  # noqa
from v11core.pleasure import (BIN_ANGLES, color_direction, pleasure_np,  # noqa
                              pleasure_torch)
from v11core.rule_agent import SRC_RULE, SRC_SLEEP, slide_step  # noqa

D = os.path.join(_PI_DIR, 'data')
OUT = os.path.join(D, 'result', 'v11', 'gate61')
CONDS = ('random', 'mixed')
KEYS = ('self_vision', 'other_vision', 'self_motion', 'self_position',
        'other_motion', 'other_position', 'layout', 'a1_source')
BOUNDS = ((-10., 10.), (-10., 10.))
RED_REF = {0: L.LANDMARKS[L.ORIGINAL]['Red'], 1: L.LANDMARKS[L.SWAP_RG]['Red']}
LAYNAME = {0: 'original', 1: 'swap_rg'}


def h5(cond):
    return h5py.File(os.path.join(D, 'data', f'v11_s1_{cond}', 'data.h5'), 'r')


def item1():
    res = {}
    for c in CONDS:
        a = h5py.File(os.path.join(OUT, f'check_{c}_a.h5'), 'r')['train']
        b = h5py.File(os.path.join(OUT, f'check_{c}_b.h5'), 'r')['train']
        full = h5(c)['train']
        res[c] = {
            'a_eq_b': all(np.array_equal(a[k][()], b[k][()]) for k in KEYS),
            'a_eq_full_first3': all(np.array_equal(a[k][()], full[k][:3]) for k in KEYS),
        }
    res['pass'] = all(v['a_eq_b'] and v['a_eq_full_first3'] for v in res.values())
    return res


def item2():
    res = {}
    fr, fm = h5('random'), h5('mixed')
    for split in ('train', 'test'):
        r, m = fr[split], fm[split]
        res[split] = {
            'n_episodes': int(r['layout'].shape[0]),
            'layout_eq': bool(np.array_equal(r['layout'][()], m['layout'][()])),
            'init_self_pos_eq': bool(np.array_equal(r['self_position'][:, 0], m['self_position'][:, 0])),
            'other_pos_eq_all_t': bool(np.array_equal(r['other_position'][()], m['other_position'][()])),
            'a2_static': bool(np.all(r['other_position'][()] == r['other_position'][:, :1])),
        }
    res['pass'] = all(v['layout_eq'] and v['init_self_pos_eq'] and v['other_pos_eq_all_t']
                      for v in res.values())
    return res


def item3():
    grids = {0: os.path.join(D, 'data', 'grid', 'data.h5'),
             1: os.path.join(D, 'data', 'v10_swap_grid', 'data.h5')}
    pts = [(0, 0), (-8, 8), (-8, -8), (8, 8), (8, -8), (0, 9), (-9, 0)]
    views, vals, eq = {}, {}, {}
    for code, path in grids.items():
        f = h5py.File(path, 'r')['train']
        sv = f['self_vision_no_agent'][0]  # (441, 16, 64, 3) A-1 at each grid point
        sp = f['self_position'][0]
        if sv.dtype != np.uint8:
            sv = np.round(sv * 255).astype(np.uint8)
        pn = pleasure_np(sv)
        x = torch.from_numpy(sv.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        pt = pleasure_torch(x).numpy()
        eq[LAYNAME[code]] = float(np.abs(pn - pt).max())
        idx = [int(np.where((sp[:, 0] == px) & (sp[:, 1] == py))[0][0]) for px, py in pts]
        views[code] = sv[idx]
        vals[code] = pn[idx]
    fig, axes = plt.subplots(len(pts), 4, figsize=(16, 2.0 * len(pts)),
                             gridspec_kw={'width_ratios': [3, 1.3, 3, 1.3]})
    for i, p in enumerate(pts):
        for j, code in enumerate((0, 1)):
            ax = axes[i, 2 * j]
            ax.imshow(views[code][i][:, :, ::-1], aspect='auto', interpolation='nearest')  # BGR -> RGB
            for k in range(1, 8):
                ax.axvline(8 * k - 0.5, color='w', lw=0.5)
            ax.set_xticks(8 * np.arange(8) + 3.5)
            ax.set_xticklabels([f'{np.rad2deg(a):.0f}' for a in BIN_ANGLES], fontsize=6)
            ax.set_yticks([])
            ax.set_title(f'{LAYNAME[code]}  A-1 at {p}', fontsize=8)
            ax = axes[i, 2 * j + 1]
            v = vals[code][i]
            ax.bar(np.arange(8), v, color=['tab:red' if x > 0 else 'tab:green' for x in v])
            ax.set_ylim(-1, 1)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(range(8))
            ax.set_xticklabels([str(k) for k in range(8)], fontsize=6)
            ax.set_title('P (bins 0-7)', fontsize=8)
    fig.suptitle('v11 gate 6.1-3: view (displayed as RGB; project Red = red) and 8-bin pleasure '
                 '(liked Red - disliked Green)/128. Bin k centre = -157.5+45k deg', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'gate61_pleasure_views.png'), dpi=130)
    plt.close(fig)
    np.savez(os.path.join(OUT, 'gate61_pleasure_views.npz'),
             pts=np.array(pts), views_original=views[0], views_swap=views[1],
             vals_original=vals[0], vals_swap=vals[1])
    return {'numpy_vs_torch_maxdiff': eq, 'pass_numeric': all(v == 0 for v in eq.values()),
            'points': pts,
            'values': {LAYNAME[c]: np.round(vals[c], 4).tolist() for c in vals}}


def item4_5_6():
    res = {}
    occ = {}
    speeds = {}
    pls_all = {}
    for c in CONDS:
        g = h5(c)['train']
        pos = g['self_position'][()]
        lay = g['layout'][()]
        src = g['a1_source'][()]
        mot = g['self_motion'][()]
        red = np.stack([RED_REF[int(l)] for l in lay])[:, None, :]
        near = np.linalg.norm(pos - red, axis=-1) < 5
        sp = np.linalg.norm(mot, axis=-1)
        awake = src != SRC_SLEEP
        r = {
            'near_red5': {LAYNAME[k]: float(near[lay == k].mean()) for k in (0, 1)},
            'near_red5_all': float(near.mean()),
            'source_counts': np.bincount(src.ravel(), minlength=4).tolist(),
            'rule_frac_of_awake': float((src == SRC_RULE).sum() / awake.sum()),
            'source_frac_of_awake': (np.bincount(src[awake].ravel(), minlength=4)[:3] / awake.sum()).tolist(),
            'sleep_frac': float((~awake).mean()),
            'awake_speed_ne1_frac': float((np.abs(sp[awake] - 1) > 1e-4).mean()),
            'sleep_speed_max': float(sp[~awake].max()) if (~awake).any() else 0.0,
        }
        # first time within 5 of red (episodes that start farther than 5)
        start_far = ~near[:, 0]
        ft = np.where(near.any(1), near.argmax(1), 101)[start_far]
        r['first_within5_median'] = float(np.median(ft))
        # rule direction recomputation from saved uint8 views (unclipped rule steps)
        sv = g['self_vision']
        n_chk, n_ok, maxerr = 0, 0, 0.0
        for n in range(0, pos.shape[0], 10):  # every 10th episode
            idx = np.where(src[n] == SRC_RULE)[0]
            if len(idx) == 0:
                continue
            v_ep = sv[n]
            for t in idx:
                d = color_direction(v_ep[t], 'Red')
                d = slide_step(pos[n, t].astype(np.float64), d, BOUNDS, 1.0) if d is not None else None
                n_chk += 1
                if d is not None:
                    e = float(np.abs(d - mot[n, t]).max())
                    maxerr = max(maxerr, e)
                    n_ok += e < 1e-4
        r['rule_recompute'] = {'checked': n_chk, 'match_1e-4': n_ok, 'max_abs_err': maxerr}
        res[c] = r
        occ[c] = (pos, lay)
        speeds[c] = (sp, awake)
        # pleasure distribution (chunked)
        P = np.concatenate([pleasure_np(sv[i:i + 200]) for i in range(0, pos.shape[0], 200)])
        pls_all[c] = P
        r['pleasure'] = {
            'abs_max': float(np.abs(P).max()),
            'p99_abs_nonzero': float(np.percentile(np.abs(P[P != 0]), 99)),
            'per_bin_mean': {LAYNAME[k]: P[lay == k].mean((0, 1)).round(4).tolist() for k in (0, 1)},
            'per_bin_frac_positive': {LAYNAME[k]: (P[lay == k] > 0).mean((0, 1)).round(4).tolist() for k in (0, 1)},
            'per_bin_frac_negative': {LAYNAME[k]: (P[lay == k] < 0).mean((0, 1)).round(4).tolist() for k in (0, 1)},
        }
        r['n_episodes_per_layout'] = {LAYNAME[k]: int((lay == k).sum()) for k in (0, 1)}

    P = pls_all['random']
    pos_vals = P[P > 0]
    a_raw = float(np.percentile(pos_vals, 95))
    res['M1b_a'] = {'p95_positive_random_train': a_raw, 'a': round(a_raw, 2),
                    'n_positive_values': int(pos_vals.size)}
    res['pass_item4'] = (
        all(res['mixed']['near_red5'][k] > res['random']['near_red5'][k] for k in ('original', 'swap_rg'))
        and 0.45 <= res['mixed']['rule_frac_of_awake'] <= 0.55
        and all(res[c]['awake_speed_ne1_frac'] <= 0.01 for c in CONDS)
        and abs(res['mixed']['sleep_frac'] - res['random']['sleep_frac']) < 0.01)

    # heatmaps: rows = condition, cols = layout
    edges = np.linspace(-10, 10, 21)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for i, c in enumerate(CONDS):
        pos, lay = occ[c]
        for j in (0, 1):
            p = pos[lay == j].reshape(-1, 2)
            H, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=[edges, edges])
            ax = axes[i, j]
            im = ax.imshow((H / H.sum()).T, origin='lower', extent=[-10, 10, -10, 10],
                           cmap='viridis', vmin=0, vmax=0.02)
            rr = RED_REF[j]
            gg = L.LANDMARKS[LAYNAME[j]]['Green']
            ax.plot(*rr, 'o', color='red', mec='w', ms=10)
            ax.plot(*gg, 'o', color='lime', mec='k', ms=10)
            ax.set_title(f'{c} / {LAYNAME[j]}  (near red<5: {res[c]["near_red5"][LAYNAME[j]]:.3f})', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('v11 gate 6.1-4: A-1 occupancy (train, all frames; fraction per 1x1 cell, colour max 0.02)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'gate61_occupancy.png'), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for c in CONDS:
        sp, awake = speeds[c]
        axes[0].hist(sp.ravel(), bins=np.linspace(0, 1.2, 61), histtype='step', label=c, log=True)
    axes[0].set_xlabel('|m_t| (all frames)'); axes[0].legend(); axes[0].set_title('step speed (log count)')
    for c in CONDS:
        g = h5(c)['train']
        mot = g['self_motion'][()]
        src = g['a1_source'][()]
        ang = np.degrees(np.arctan2(mot[..., 1], mot[..., 0]))[src != SRC_SLEEP]
        axes[1].hist(ang, bins=np.linspace(-180, 180, 73), histtype='step', label=c, density=True)
    axes[1].set_xlabel('step direction (deg, awake steps)'); axes[1].legend()
    axes[1].set_title('step direction (all layouts)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'gate61_speed_direction.png'), dpi=130)
    plt.close(fig)

    # A-2 positions + pleasure distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    op = h5('random')['train']['other_position'][:, 0]
    H, _, _ = np.histogram2d(op[:, 0], op[:, 1], bins=[edges, edges])
    im = axes[0].imshow(H.T, origin='lower', extent=[-10, 10, -10, 10], cmap='viridis')
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    axes[0].set_title('A-2 static positions (train, per episode)')
    for j, c in enumerate(CONDS):
        P = pls_all[c]
        lay = occ[c][1]
        for k in (0, 1):
            m = P[lay == k].reshape(-1, 8)
            axes[1 + j].plot(range(8), m.mean(0), marker='o', label=f'{LAYNAME[k]} mean')
            axes[1 + j].fill_between(range(8), np.percentile(m, 5, 0), np.percentile(m, 95, 0), alpha=0.2)
        axes[1 + j].axhline(0, color='k', lw=0.5)
        axes[1 + j].set_ylim(-1, 1)
        axes[1 + j].set_title(f'pleasure per bin ({c}, train; band 5-95%)')
        axes[1 + j].set_xlabel('bin k (centre -157.5+45k deg)')
        axes[1 + j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'gate61_stats.png'), dpi=130)
    plt.close(fig)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {'item1': item1(), 'item2': item2(), 'item3': item3()}
    r = item4_5_6()
    out['item4_5'] = {c: r[c] for c in CONDS}
    out['item4_pass'] = r['pass_item4']
    out['item6_M1b_a'] = r['M1b_a']
    with open(os.path.join(OUT, 'gate61.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
