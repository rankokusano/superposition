"""
v15 Sec.36 (exploratory E1-E3, methods and reading rules fixed before computing) from data/result/v15/e123/<name>.npz
(extract_e123_v15.py). CPU only.
    E1: PCA of self h1 (label data) with the other's h2 (generated pairs, 1-10 steps after eating) on the same axes
    E2: blue-vs-cyan readout fitted on h1, applied unchanged to h2; agreement with the correct side per step
    E3: correlation over the 8 directions of G's output with A-2's true-value pleasure and with the model's own-value pleasure

    python analyze/v15/e123_v15.py   # -> data/result/v15/e123/e123_summary.json, figs/
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
from sklearn.decomposition import PCA  # noqa
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn.model_selection import GroupKFold  # noqa
from sklearn.pipeline import make_pipeline  # noqa
from sklearn.preprocessing import StandardScaler  # noqa

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15.eval_s2_v15 import boot_mean  # noqa

D = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'e123')
FIG = os.path.join(D, 'figs')
S2 = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'stage2')
SEEDS = (0, 1, 2)
CONDS = (('exp', 'v15_s2p2h_exp_v15i', 'shared, experienced'), ('noexp', 'v15_s2p2h_noexp_v15i', 'shared, no experience'),
         ('expY', 'v15_s2p2h_expY_v15i', 'rank Y experienced'), ('ns_exp', 'v15_s2p2h_exp_v15i_ns', 'NS, experienced'),
         ('ns_noexp', 'v15_s2p2h_noexp_v15i_ns', 'NS, no experience'))
OWN_RANK = {'exp': 'X', 'noexp': 'X', 'expY': 'Y', 'ns_exp': 'X', 'ns_noexp': 'X'}
BR, TR = 2, 3  # corner indices (TL, BL, BR, TR); blue is always BR, cyan always TR
BLUE, CYAN, INK, GRID, MUTED = '#2050e0', '#14b8c8', '#3d3d3a', '#e6e5e0', '#9a9a94'
COND_COL = {'exp': '#2a78d6', 'noexp': '#eb6834', 'expY': '#1baf7a', 'ns_exp': '#6250d6', 'ns_noexp': '#e87ba4'}
STEPS = np.arange(1, 11)


def name(stem, s):
    n = stem + ('' if s == 0 else f'_s{s}')
    return n if s == 0 else f'{n}_seed{s}'


def load(n):
    return dict(np.load(os.path.join(D, f'{n}.npz')))


def style(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=7.5)


def readout(z, tag):
    """blue-vs-cyan readout on h1 (BR / TR target frames): CV accuracy (GroupKFold 5 by episode) and the fitted model."""
    m = np.isin(z[f'{tag}_corner'], (BR, TR))
    X, y, g = z[f'{tag}_h1'][m], (z[f'{tag}_corner'][m] == BR).astype(int), z[f'{tag}_episode'][m]
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))  # noqa: E731
    cv = []
    for tr, te in GroupKFold(5).split(X, y, g):
        cv.append((mk().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
    return float(np.mean(cv)), mk().fit(X, y), {'n_blue': int(y.sum()), 'n_cyan': int((1 - y).sum())}


def agreement(clf, z):
    """per pair-member and step 1..10: readout says blue == correct is blue (X side)."""
    h2 = z['h2'][:, 1:11]
    pb = clf.predict(h2.reshape(-1, h2.shape[-1])).reshape(h2.shape[:2])
    truth = (z['member'] == 0)[:, None]
    return (pb == truth).astype(float)


def summarise(A, pair):
    return {'per_step': [boot_mean(A[:, k], pair) for k in range(10)], 'mean_1_10': boot_mean(A.mean(1), pair)}


def cond_state(ms):
    """Sec.36.3: per condition from the seeds' 1-10 mean agreement."""
    above = sum(1 for m in ms if m['ci95'][0] > 0.5)
    near = sum(1 for m in ms if (m['ci95'][0] <= 0.5 <= m['ci95'][1]) or abs(m['mean'] - 0.5) < 0.05)
    return 'above' if above >= 2 else 'near 0.5' if near >= 2 else 'describe'


def corr_rows(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    na, nb = np.sqrt((a ** 2).sum(-1)), np.sqrt((b ** 2).sum(-1))
    ok = (na > 1e-9) & (nb > 1e-9)
    c = np.full(a.shape[:-1], np.nan)
    c[ok] = (a * b).sum(-1)[ok] / (na * nb)[ok]
    return c


def nanboot(v, units):
    keep = ~np.isnan(v)
    return boot_mean(v[keep], units[keep]) if keep.any() else None


def e3(z, own):
    q = z['q2'][:, 1:11]
    pA2 = np.where((z['member'] == 0)[:, None, None], z['p_X'][:, 1:11], z['p_Y'][:, 1:11])
    pself = z[f'p_{own}'][:, 1:11]
    cA, cS = corr_rows(q, pA2), corr_rows(q, pself)
    diff_side = 1 if own == 'X' else 0  # member index of the side where A-2's preference differs from the model's own
    out = {'n_undefined_A2': int(np.isnan(cA).sum()), 'n_undefined_self': int(np.isnan(cS).sum()), 'n_total': int(cA.size),
           'diff_side': 'Y' if diff_side == 1 else 'X'}
    for side, mi in (('X', 0), ('Y', 1)):
        m = z['member'] == mi
        out[side] = {'corr_A2_per_step': [nanboot(cA[m, k], z['pair'][m]) for k in range(10)],
                     'corr_self_per_step': [nanboot(cS[m, k], z['pair'][m]) for k in range(10)],
                     'corr_A2_mean': nanboot(np.nanmean(cA[m], 1), z['pair'][m]),
                     'corr_self_mean': nanboot(np.nanmean(cS[m], 1), z['pair'][m])}
    m = z['member'] == diff_side
    d = np.nanmean(cA[m] - cS[m], 1)
    out['diff_mean'] = nanboot(d, z['pair'][m])
    lo, hi = out['diff_mean']['ci95']
    out['reading'] = 'estimates the other' if lo > 0 else 'projects own values' if hi < 0 else 'undecided'
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    Z = {(c, s): load(name(stem, s)) for c, stem, _ in CONDS for s in SEEDS}
    ref = load('ref0_v15i')
    R = {'E2': {}, 'E3': {}, 'E1': {}}
    # ---------------- E2
    ref_own = {}
    for tag in ('goalcap', 'rankXY'):
        cv, clf, n = readout(ref, tag)
        A = agreement(clf, ref)
        ref_own[tag] = {'h1_cv_acc': cv, **n, **summarise(A, ref['pair'])}
    R['E2']['ref0_own_procedure'] = ref_own
    for c, stem, lab in CONDS:
        R['E2'][c] = {}
        for s in SEEDS:
            z = Z[(c, s)]
            h = json.load(open(os.path.join(S2, f'hide_{name(stem, s)}_cpu.json')))['gen']['memory_readout']['h2']
            row = {'h2_direct_per_step': [h[str(o)]['mean'] for o in STEPS],
                   'h2_direct_mean_1_10': float(np.mean([h[str(o)]['mean'] for o in STEPS]))}
            for tag in ('goalcap', 'rankXY'):
                cv, clf, n = readout(z, tag)
                A = agreement(clf, z)
                row[tag] = {'h1_cv_acc': cv, 'readable': cv >= 0.60, **n, **summarise(A, z['pair']),
                            'applied_to_ref0_h2': boot_mean(agreement(clf, ref).mean(1), ref['pair'])}
            R['E2'][c][s] = row
        for tag in ('goalcap', 'rankXY'):
            ms = [R['E2'][c][s][tag]['mean_1_10'] for s in SEEDS if R['E2'][c][s][tag]['readable']]
            R['E2'][c][f'{tag}_state'] = cond_state(ms) if len(ms) >= 2 else 'not readable (fewer than 2 seeds)'
    for tag in ('goalcap', 'rankXY'):
        sh = [R['E2'][c][f'{tag}_state'] for c in ('exp', 'noexp')]
        ns = [R['E2'][c][f'{tag}_state'] for c in ('ns_exp', 'ns_noexp')]
        R['E2'][f'{tag}_rule'] = ('same form' if all(x == 'above' for x in sh) and all(x != 'above' for x in ns) else
                                  'kept in another form' if all(x == 'near 0.5' for x in sh) else 'describe')
    # ---------------- E3
    for c, _, _ in CONDS:
        R['E3'][c] = {s: e3(Z[(c, s)], OWN_RANK[c]) for s in SEEDS}
        rd = [R['E3'][c][s]['reading'] for s in SEEDS]
        top = max(set(rd), key=rd.count)
        R['E3'][c]['condition_reading'] = top if rd.count(top) >= 2 else 'mixed'
    R['E3']['ref0'] = e3(ref, 'X')
    # ---------------- E1 (PCA variance only; figure below)
    for tag in ('goalcap', 'rankXY'):
        R['E1'][tag] = {}
        for c, _, _ in CONDS:
            for s in SEEDS:
                p = PCA(2).fit(Z[(c, s)][f'{tag}_h1'])
                R['E1'][tag][f'{c}_seed{s}'] = [float(v) for v in p.explained_variance_ratio_]
    json.dump(R, open(os.path.join(D, 'e123_summary.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    figs(Z, ref, R)
    report(R)


def figs(Z, ref, R):
    rng = np.random.RandomState(0)
    pick = lambda idx, k=600: idx if len(idx) <= k else rng.choice(idx, k, replace=False)  # noqa: E731
    from matplotlib.lines import Line2D
    # E1
    for tag, title in (('goalcap', 'label data: no-experience test (main; targets in all four corners)'),
                       ('rankXY', 'label data: rank X + rank Y tests (secondary; label = which colour was eaten)')):
        fig, ax = plt.subplots(3, 5, figsize=(16, 9.6))
        for j, (c, _, lab) in enumerate(CONDS):
            for s in SEEDS:
                a = ax[s, j]; z = Z[(c, s)]
                p = PCA(2).fit(z[f'{tag}_h1'])
                P1 = p.transform(z[f'{tag}_h1']); cor = z[f'{tag}_corner']
                for k, col in ((0, MUTED), (1, MUTED), (BR, BLUE), (TR, CYAN)):
                    idx = pick(np.where(cor == k)[0])
                    if len(idx):
                        a.scatter(P1[idx, 0], P1[idx, 1], s=4, color=col, alpha=0.12, lw=0)
                h2 = z['h2'][:, 1:11]
                P2 = p.transform(h2.reshape(-1, h2.shape[-1])).reshape(h2.shape[0], 10, 2)
                for mi, col in ((0, BLUE), (1, CYAN)):
                    rows = np.where(z['member'] == mi)[0]
                    pts = P2[rows].reshape(-1, 2)
                    idx = pick(np.arange(len(pts)))
                    a.scatter(pts[idx, 0], pts[idx, 1], s=7, color=col, alpha=0.75, lw=0.3, edgecolor='k')
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_color(GRID)
                if s == 0:
                    a.set_title(lab, fontsize=9)
                if j == 0:
                    a.set_ylabel(f'seed {s}\nPC2', fontsize=8.5)
                if s == 2:
                    a.set_xlabel('PC1 (axes from self h1)', fontsize=8)
        fig.legend(handles=[Line2D([], [], marker='o', ls='', color=BLUE, alpha=0.3, label="self h1: A-1's next target BR (blue)"),
                            Line2D([], [], marker='o', ls='', color=CYAN, alpha=0.3, label="self h1: next target TR (cyan)"),
                            Line2D([], [], marker='o', ls='', color=MUTED, alpha=0.3, label='self h1: next target TL / BL'),
                            Line2D([], [], marker='o', ls='', color=BLUE, mec='k', label='other h2: X side (correct blue)'),
                            Line2D([], [], marker='o', ls='', color=CYAN, mec='k', label='other h2: Y side (correct cyan)')],
                   loc='lower center', ncol=5, fontsize=8, frameon=False)
        fig.suptitle(f'E1: self h1 (light) and the other\'s h2 1-10 steps after eating (dark) on the PCA axes of self h1. {title}', fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(os.path.join(FIG, f'e1_pca_{tag}.png'), dpi=110); plt.close(fig)
    # E2
    fig, ax = plt.subplots(2, 5, figsize=(16, 6.6), sharey=True)
    for r, tag in enumerate(('goalcap', 'rankXY')):
        for j, (c, _, lab) in enumerate(CONDS):
            a = ax[r, j]
            for s in SEEDS:
                row = R['E2'][c][s]
                ps = row[tag]['per_step']
                m = [v['mean'] for v in ps]; lo = [v['ci95'][0] for v in ps]; hi = [v['ci95'][1] for v in ps]
                a.fill_between(STEPS, lo, hi, color=COND_COL[c], alpha=0.12, lw=0)
                a.plot(STEPS, m, '-o', color=COND_COL[c], ms=3, lw=1.6, label='h1 readout applied to h2' if s == 0 else None)
                a.plot(STEPS, row['h2_direct_per_step'], ':', color=COND_COL[c], lw=1.3, label='readout fitted on h2 (reference)' if s == 0 else None)
            rp = R['E2']['ref0_own_procedure'][tag]['per_step']
            a.plot(STEPS, [v['mean'] for v in rp], '--', color=MUTED, lw=1.2, label='before training' if j == 0 else None)
            a.axhline(0.5, color=INK, ls='--', lw=0.9)
            a.set_ylim(0.2, 1.0); a.set_xticks(range(1, 11, 3)); style(a)
            if r == 0:
                a.set_title(lab, fontsize=9)
            if r == 1:
                a.set_xlabel('steps after eating', fontsize=8)
            if j == 0:
                a.set_ylabel(('no-experience labels (main)' if tag == 'goalcap' else 'rank X + Y labels') + '\nagreement with correct side', fontsize=8)
    ax[0, 0].legend(fontsize=7, frameon=False, loc='lower left')
    fig.suptitle('E2: blue-vs-cyan readout fitted on self h1, applied unchanged to the other\'s h2 (lines = seeds, bands = 95% CI over pairs)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'e2_transfer.png'), dpi=120); plt.close(fig)
    # E3 curves on the side where A-2's preference differs from the model's own
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
    for j, (c, _, lab) in enumerate(CONDS):
        a = ax[j]
        side = R['E3'][c][0]['diff_side']
        for s in SEEDS:
            e = R['E3'][c][s][side]
            a.plot(STEPS, [v['mean'] for v in e['corr_A2_per_step']], '-o', color=COND_COL[c], ms=3, lw=1.6, label="corr(G output, A-2's true values)" if s == 0 else None)
            a.plot(STEPS, [v['mean'] for v in e['corr_self_per_step']], '--s', color=MUTED, ms=3, lw=1.3, label='corr(G output, own values)' if s == 0 else None)
        a.axhline(0, color=INK, lw=0.8)
        a.set_title(f'{lab}\n({side} side: A-2 differs from own)', fontsize=8.5); a.set_xticks(range(1, 11, 3)); style(a)
        a.set_xlabel('steps after eating', fontsize=8)
    ax[0].set_ylabel('correlation over 8 directions', fontsize=8.5)
    ax[0].legend(fontsize=7, frameon=False, loc='lower left')
    fig.suptitle('E3: whose values does G write? (lines = seeds; side where A-2\'s preference differs from the model\'s own)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'e3_corr_curves.png'), dpi=120); plt.close(fig)
    # E3 bars for pair 0, seed-0 models
    fig, ax = plt.subplots(5, 2, figsize=(10, 11), sharex=True)
    zstd = lambda v: (v - v.mean()) / (v.std() + 1e-12)  # noqa: E731
    for j, (c, _, lab) in enumerate(CONDS):
        z = Z[(c, 0)]
        for mi, side in ((0, 'X'), (1, 'Y')):
            a = ax[j, mi]
            i = np.where((z['pair'] == 0) & (z['member'] == mi))[0][0]
            q = z['q2'][i, 1:11].mean(0); pA2 = z[f'p_{side}'][i, 1:11].mean(0); ps = z[f'p_{OWN_RANK[c]}'][i, 1:11].mean(0)
            for k, (v, col, l) in enumerate(((q, COND_COL[c], 'G output P̂'), (pA2, INK, 'A-2 true values P_A2'),
                                             (ps, MUTED, 'own values P_self'))):
                a.bar(np.arange(8) + (k - 1) * 0.27, zstd(v), 0.25, color=col, label=l)
            a.axhline(0, color=INK, lw=0.7); style(a)
            a.set_title(f'{lab}, seed 0, {side} side' + (' (A-2 = own)' if side == OWN_RANK[c] else ' (A-2 differs)'), fontsize=8.5)
            if j == 4:
                a.set_xlabel("direction bin of A-2's view (0-7)", fontsize=8)
            if mi == 0:
                a.set_ylabel('standardized\nover 8 bins', fontsize=8)
            if j == 0 and mi == 0:
                a.legend(fontsize=7, frameon=False, loc='upper left')
    fig.suptitle('E3: pair 0 (first kept pair, not selected), mean over 1-10 steps after eating; each vector standardized over the 8 bins', fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'e3_pair0_bars.png'), dpi=120); plt.close(fig)


def report(R):
    f = lambda b: f"{b['mean']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"  # noqa: E731
    for tag in ('goalcap', 'rankXY'):
        print(f'== E2 {tag}: rule -> {R["E2"][f"{tag}_rule"]}; ref0 own: cv {R["E2"]["ref0_own_procedure"][tag]["h1_cv_acc"]:.3f}, '
              f'transfer {f(R["E2"]["ref0_own_procedure"][tag]["mean_1_10"])}')
        for c, _, _ in CONDS:
            print(' ', c, R['E2'][c][f'{tag}_state'], ' | '.join(
                f"s{s}: cv {R['E2'][c][s][tag]['h1_cv_acc']:.3f} transfer {f(R['E2'][c][s][tag]['mean_1_10'])} "
                f"h2direct {R['E2'][c][s]['h2_direct_mean_1_10']:.3f} onref0 {R['E2'][c][s][tag]['applied_to_ref0_h2']['mean']:.3f}" for s in SEEDS))
    print('== E3')
    for c, _, _ in CONDS:
        e = R['E3'][c]
        print(' ', c, e['condition_reading'], 'side', e[0]['diff_side'], ' | '.join(
            f"s{s}: diff {f(e[s]['diff_mean'])} A2 {e[s][e[s]['diff_side']]['corr_A2_mean']['mean']:.3f} self {e[s][e[s]['diff_side']]['corr_self_mean']['mean']:.3f} "
            f"undef {e[s]['n_undefined_A2']}/{e[s]['n_undefined_self']}/{e[s]['n_total']}" for s in SEEDS))
        print('    same side:', ' | '.join(f"s{s}: A2 {e[s]['X' if e[s]['diff_side'] == 'Y' else 'Y']['corr_A2_mean']['mean']:.3f}" for s in SEEDS))
    e = R['E3']['ref0']
    print('  ref0', e['reading'], f(e['diff_mean']))


if __name__ == '__main__':
    main()
