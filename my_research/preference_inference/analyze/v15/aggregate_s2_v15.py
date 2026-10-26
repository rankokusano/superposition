"""
v15 stage-2 judgement and figures (pre-registered in docs/v15_experiment_log.md Sec.16, 16.7, 19, 22).
Reads data/result/v15/stage2/eval_<tag>.{json,npz} for exp/noexp x v4/v15i and the before-training references
eval_exp_<enc>_ref0. Paired differences exp - noexp use the same units (the same test episodes / pairs; the
event sets do not depend on the model): 2000 bootstrap draws over units (within cells for the equal-weight means).

    python analyze/v15/aggregate_s2_v15.py    # -> data/result/v15/stage2/summary.json and figs/
"""

import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15 import stage0_v15 as S0  # noqa

S2 = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'stage2')
RES = os.path.join(_PI_DIR, 'data', 'result')
FIG = os.path.join(S2, 'figs')
ENCS = ('v4', 'v15i')
ENC_LAB = {'v4': 'v4 encoder (main)', 'v15i': 'v15 (i) encoder'}
COND_COL = {'exp': '#2a78d6', 'noexp': '#eb6834', 'ref0': '#9a9a94'}
COND_LAB = {'exp': 'experienced', 'noexp': 'no experience (goal, no preference)', 'ref0': 'before training'}
BOOT = 2000
INK = '#3d3d3a'


def load(tag):
    return json.load(open(os.path.join(S2, f'eval_{tag}.json'))), np.load(os.path.join(S2, f'eval_{tag}.npz'))


def ci(draws):
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def ew_paired(a, b, cells, seed=0):
    """equal-weight mean over cells of (a - b) (b may be None -> mean of a); bootstrap within cells."""
    rng = np.random.RandomState(seed)
    names = sorted(set(cells))
    ok = ~np.isnan(a) & (np.ones_like(a, bool) if b is None else ~np.isnan(b))
    diff = a if b is None else a - b
    est = np.mean([diff[ok & (cells == c)].mean() for c in names])
    draws = []
    for _ in range(BOOT):
        m = []
        for c in names:
            v = diff[ok & (cells == c)]
            m.append(v[rng.randint(len(v), size=len(v))].mean())
        draws.append(np.mean(m))
    return {'mean': float(est), 'ci95': ci(draws)}


def pair_mean(acc, pair):
    """per-pair mean of the two members -> (pairs,)"""
    ok = ~np.isnan(acc)
    ps = np.unique(pair[ok])
    return np.array([acc[ok & (pair == p)].mean() for p in ps])


def boot1(v, seed=0):
    rng = np.random.RandomState(seed)
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': ci([v[rng.randint(len(v), size=len(v))].mean() for _ in range(BOOT)])}


def judge():
    S = {}
    for enc in ENCS:
        e, ze = load(f'exp_{enc}')
        n, zn = load(f'noexp_{enc}')
        r, zr = load(f'exp_{enc}_ref0')
        out = {}
        # prerequisite
        out['P1a'] = {k: d['P1a_r2_self_visible'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        out['P1b'] = {k: d['P1b_r2_other'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        # P1-c: equal-weight over eaten corners
        cells = ze['P1c_corner_ev']
        out['P1c_EW'] = {k: ew_paired(z['P1c_diff_ev'], None, cells) for k, z in (('exp', ze), ('noexp', zn), ('ref0', zr))}
        out['P1c_EW']['exp_minus_noexp'] = ew_paired(ze['P1c_diff_ev'], zn['P1c_diff_ev'], cells)
        out['P1c_EW']['pass_single'] = out['P1c_EW']['exp']['ci95'][0] > out['P1c_EW']['ref0']['mean']
        out['P1c_EW']['pass_main'] = out['P1c_EW']['exp_minus_noexp']['ci95'][0] > 0
        # P2-a: equal-weight over corner x candidate
        cells = ze['P2a_cell_ev']
        out['P2a_EW'] = {k: ew_paired(z['P2a_diff_ev'], None, cells) for k, z in (('exp', ze), ('noexp', zn), ('ref0', zr))}
        out['P2a_EW']['exp_minus_noexp'] = ew_paired(ze['P2a_diff_ev'], zn['P2a_diff_ev'], cells)
        out['P2a_EW']['pass_single'] = out['P2a_EW']['exp']['ci95'][0] > out['P2a_EW']['ref0']['mean']
        out['P2a_EW']['pass_main'] = out['P2a_EW']['exp_minus_noexp']['ci95'][0] > 0
        out['P2a_toward_blue'] = {k: d['P2a_toward_blue'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        # P2-b (post main, pre descriptive)
        one = np.zeros(len(ze['P2b_post_ev']), int)
        out['P2b_post'] = {k: boot1(z['P2b_post_ev'].astype(float)) for k, z in (('exp', ze), ('noexp', zn), ('ref0', zr))}
        out['P2b_post']['exp_minus_noexp'] = ew_paired(ze['P2b_post_ev'].astype(float), zn['P2b_post_ev'].astype(float), one)
        out['P2b_post']['pass_single'] = out['P2b_post']['exp']['ci95'][0] > 0.5
        out['P2b_post']['pass_main'] = out['P2b_post']['exp_minus_noexp']['ci95'][0] > 0
        out['P2b_pre'] = {k: boot1(z['P2b_pre_ev'].astype(float)) for k, z in (('exp', ze), ('noexp', zn), ('ref0', zr))}
        out['ab_baselines'] = e['baselines_ab_at_A2_moments']
        # P2-c and P2-c' (pair means)
        for key, accname in (('P2c', 'gen_acc_ev'), ("P2c'", 'gen_choose_acc_ev')):
            pm = {k: pair_mean(z[accname].astype(float), z['gen_pair']) for k, z in (('exp', ze), ('noexp', zn), ('ref0', zr))}
            d = {k: boot1(v) for k, v in pm.items()}
            d['exp_minus_noexp'] = boot1(pm['exp'] - pm['noexp'])
            thr = 0.5 if key == 'P2c' else d['ref0']['mean']
            d['pass_single'] = d['exp']['ci95'][0] > thr
            d['pass_main'] = d['exp_minus_noexp']['ci95'][0] > 0
            d['single_threshold'] = thr
            out[key] = d
        out['P2c_by_member'] = {k: d['P2c_gen']['acc_by_member'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        out['P2c_toward_blue'] = {k: d['P2c_gen']['toward_blue'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        out["P2c'_by_cand"] = {k: d["P2c'_gen"]['acc_by_cand'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        out["P2c'_chose_X"] = {k: d["P2c'_gen"]['chose_X'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        out['memory'] = {k: d['memory_readout_gen'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        out['ctrl3'] = {k: d['ctrl3_pipeline_check'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        out['natural'] = {k: d['P2c_natural_all_moments']['acc'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        out['q2'] = {k: d['q2_stats'] for k, d in (('exp', e), ('noexp', n), ('ref0', r))}
        S[enc] = out
    return S


# ------------------------------------------------------------------ figures
def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8)


def bars(ax, S, getter, title, ylabel, hline=None, conds=('exp', 'noexp', 'ref0')):
    w = 0.26
    for j, enc in enumerate(ENCS):
        for i, c in enumerate(conds):
            v = getter(S[enc], c)
            if v is None:
                continue
            m, lo, hi = (v['mean'], *v['ci95']) if isinstance(v, dict) else (v, v, v)
            x = j + (i - (len(conds) - 1) / 2) * w
            ax.bar(x, m, w * 0.9, color=COND_COL[c], label=COND_LAB[c] if j == 0 else None)
            if isinstance(v, dict):
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1, capsize=3)
            ax.text(x, m + (0.01 if m >= 0 else -0.04), f'{m:.2f}', ha='center', fontsize=7)
    if hline is not None:
        ax.axhline(hline, color=INK, ls='--', lw=1)
    ax.set_xticks(range(len(ENCS))); ax.set_xticklabels([ENC_LAB[e] for e in ENCS], fontsize=8)
    ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=8); style(ax)


def read_log(path):
    if not os.path.exists(path):
        return np.array([]), np.array([])
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    return a[:, 0], a[:, 1]


def fig_curves():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ls = {'v4': '-', 'v15i': '--'}
    for j, (ph, lab) in enumerate((('p1', 'phase 1 (mse, whole network except encoders)'), ('p2', 'phase 2 (l1, G only)'))):
        for enc in ENCS:
            for cond in ('exp', 'noexp'):
                exp = f'v15_s2p1_{cond}_{enc}_mse' if ph == 'p1' else f'v15_s2p2_{cond}_{enc}'
                e, v = read_log(os.path.join(RES, exp, '0', 'log', 'test', 'self_vision.log'))
                ax[j].plot(e, v, ls[enc], color=COND_COL[cond], lw=1.6, marker='o', ms=3, label=f'{cond} / {enc} (test)')
        ax[j].set_xlabel('epoch'); ax[j].set_ylabel('view loss (test, sum over steps)'); ax[j].set_title(lab, fontsize=9); style(ax[j])
    ax[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Stage 2 trial learning curves (seed 0; test every 10 epochs)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 's2_learning_curves.png'), dpi=120); plt.close(fig)


def fig_tracking(S):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bars(ax[0], S, lambda s, c: s['P1a'][c], 'P1-a: A-1 position from h1 (motion-visible frames)', 'R^2', hline=0.80)
    bars(ax[1], S, lambda s, c: s['P1b'][c], 'P1-b: A-2 position from h2 (static A-2)', 'R^2')
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(os.path.join(FIG, 's2_p1_tracking.png'), dpi=120); plt.close(fig)


def fig_posteat(S):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    bars(ax[0], S, lambda s, c: s['P1c_EW'][c], 'P1-c: cos(next=blue) - cos(cyan), equal weight TL/BL', 'contrast')
    bars(ax[1], S, lambda s, c: s['P1c_EW']['exp_minus_noexp'] if c == 'exp' else None,
         'P1-c: experienced - no experience (paired)', 'difference', hline=0, conds=('exp',))
    for j, enc in enumerate(ENCS):
        for i, c in enumerate(('exp', 'noexp', 'ref0')):
            e, z = load(f'{c}_{enc}' if c != 'ref0' else f'exp_{enc}_ref0')
            cell = e['P1c_diff_by_corner']['cells']
            for k, corner in enumerate(('TL', 'BL')):
                v = cell[corner]
                x = j * 2.4 + k * 1.0 + (i - 1) * 0.28
                ax[2].bar(x, v['mean'], 0.25, color=COND_COL[c], label=COND_LAB[c] if (j == 0 and k == 0) else None)
                ax[2].errorbar(x, v['mean'], yerr=[[v['mean'] - v['ci95'][0]], [v['ci95'][1] - v['mean']]], color=INK, lw=1, capsize=2)
    ax[2].set_xticks([0, 1, 2.4, 3.4]); ax[2].set_xticklabels(['v4 TL', 'v4 BL', 'v15i TL', 'v15i BL'], fontsize=8)
    ax[2].axhline(0, color=INK, lw=0.8); ax[2].set_title('P1-c contrast by eaten corner', fontsize=9); style(ax[2])
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Phase 1: after A-1 eats, does its predicted move head to the next liked object (blue)?', fontsize=10)
    fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(os.path.join(FIG, 's2_p1_posteat.png'), dpi=120); plt.close(fig)


def fig_p2(S):
    fig, ax = plt.subplots(1, 4, figsize=(19, 3.9))
    bars(ax[0], S, lambda s, c: s['P2a_EW'][c], "P2-a: next - other corner, equal weight (corner x X/Y)", 'contrast', hline=0)
    for j, enc in enumerate(ENCS):
        for i, c in enumerate(('exp', 'noexp', 'ref0')):
            for k, g in enumerate(('X', 'Y')):
                v = S[enc]['P2a_toward_blue'][c][g]
                x = j * 2.4 + k * 1.0 + (i - 1) * 0.28
                ax[1].bar(x, v['mean'], 0.25, color=COND_COL[c])
                ax[1].errorbar(x, v['mean'], yerr=[[v['mean'] - v['ci95'][0]], [v['ci95'][1] - v['mean']]], color=INK, lw=1, capsize=2)
    ax[1].axhline(0.5, color=INK, ls='--', lw=1)
    ax[1].set_xticks([0, 1, 2.4, 3.4]); ax[1].set_xticklabels(['v4: A-2 X', 'v4: A-2 Y', 'v15i: X', 'v15i: Y'], fontsize=8)
    ax[1].set_title("P2-a projection: predicted toward blue (A-1's own next)", fontsize=9); style(ax[1])
    bars(ax[2], S, lambda s, c: s['P2b_post'][c], 'P2-b: choose X/Y by post-eat error', 'accuracy', hline=0.5)
    ab = S['v4']['ab_baselines']
    ax[2].axhline(ab['a']['mean'], color='#1baf7a', ls=':', lw=1.2); ax[2].axhline(ab['b0']['mean'], color='#e87ba4', ls=':', lw=1.2)
    ax[2].text(1.45, ab['a']['mean'], '(a)', fontsize=7); ax[2].text(1.45, ab['b0']['mean'], '(b0)', fontsize=7)
    bars(ax[3], S, lambda s, c: s['P2b_pre'][c], 'P2-b (descriptive): choose by pre-eat error', 'accuracy', hline=0.5)
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(os.path.join(FIG, 's2_p2_g5_inverse.png'), dpi=120); plt.close(fig)


def fig_main(S):
    fig, ax = plt.subplots(1, 4, figsize=(19, 3.9))
    bars(ax[0], S, lambda s, c: s['P2c'][c], 'P2-c (main, write-out by G): pair accuracy', 'accuracy', hline=0.5)
    bars(ax[1], S, lambda s, c: s["P2c'"][c], "P2-c' (choose by pre-eat error): pair accuracy", 'accuracy', hline=0.5)
    for j, enc in enumerate(ENCS):
        for i, c in enumerate(('exp', 'noexp', 'ref0')):
            for k, g in enumerate(('first', 'second')):
                v = S[enc]['P2c_by_member'][c][g]
                x = j * 2.4 + k * 1.0 + (i - 1) * 0.28
                ax[2].bar(x, v['mean'], 0.25, color=COND_COL[c])
                ax[2].errorbar(x, v['mean'], yerr=[[v['mean'] - v['ci95'][0]], [v['ci95'][1] - v['mean']]], color=INK, lw=1, capsize=2)
    ax[2].axhline(0.5, color=INK, ls='--', lw=1)
    ax[2].set_xticks([0, 1, 2.4, 3.4]); ax[2].set_xticklabels(['v4: X (->blue)', 'v4: Y (->cyan)', 'v15i: X', 'v15i: Y'], fontsize=8)
    ax[2].set_title('P2-c by member (projection shows as X high / Y low)', fontsize=9); style(ax[2])
    bars(ax[3], S, lambda s, c: s['natural'][c], 'P2-c protocol on all stage-2 test moments (nearest corner)', 'accuracy')
    ab = S['v4']['ab_baselines']
    ax[3].axhline(ab['a']['mean'], color='#1baf7a', ls=':', lw=1.2); ax[3].axhline(ab['b0']['mean'], color='#e87ba4', ls=':', lw=1.2)
    ax[3].text(1.45, ab['a']['mean'], '(a)', fontsize=7); ax[3].text(1.45, ab['b0']['mean'], '(b0)', fontsize=7)
    fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Main test on the generated pairs (identical views after eating; (a), (b0) = 0.5 by construction)', fontsize=10)
    fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(os.path.join(FIG, 's2_main_pairs.png'), dpi=120); plt.close(fig)


def fig_memory(S):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    offs = ['0', '1', '3', '5', '10']
    for j, key in enumerate(('h2', 'g')):
        for enc, ls in (('v4', '-'), ('v15i', '--')):
            for c in ('exp', 'noexp', 'ref0'):
                m = S[enc]['memory'][c][key]
                ax[j].plot([int(o) for o in offs], [m[o]['mean'] for o in offs], ls, color=COND_COL[c], marker='o', ms=3,
                           label=f'{COND_LAB[c]} / {enc}')
        ax[j].axhline(0.5, color=INK, ls=':', lw=1)
        ax[j].set_xlabel('steps after eating (input frozen at the eat frame)')
        src = 'h2' if key == 'h2' else "G's LSTM state"
        ax[j].set_title("A-2's candidate (X/Y) read from " + src, fontsize=9); style(ax[j])
    ax[0].set_ylabel('readout accuracy (generated pairs)'); ax[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 's2_memory_readout.png'), dpi=120); plt.close(fig)


def fig_examples():
    e, ze = load('exp_v4')
    keys, te, d = ze['gen_keys'], ze['gen_te'], ze['gen_d']
    H = len(keys)
    dx = h5py.File(os.path.join(_PI_DIR, 'data', 'data', 'v15_s2_genx_p80', 'data.h5'), 'r')['train']
    dy = h5py.File(os.path.join(_PI_DIR, 'data', 'data', 'v15_s2_geny_p80', 'data.h5'), 'r')['train']
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
    for a, i in zip(axs, range(4)):
        n, t_e = int(keys[i]), int(te[i])
        for c, pos in enumerate(S0.CUBE_C[0]):
            col = {0: '#e3242b', 1: '#9a9a94', 2: '#2050e0', 3: '#14b8c8'}[c]  # X episode: green gray
            a.add_patch(plt.Rectangle((pos[0] - 3, pos[1] - 3), 6, 6, color=col, alpha=0.8))
        a.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], color='#9a9a94', lw=1)
        px, py = dx['other_position'][n].astype(float), dy['other_position'][n].astype(float)
        a.plot(px[:t_e + 1, 0], px[:t_e + 1, 1], color=INK, lw=3, alpha=0.3, label='A-2 until eating (shared)')
        a.plot(px[t_e:t_e + 11, 0], px[t_e:t_e + 11, 1], color='#2a78d6', lw=1.5, label='true after (X)')
        a.plot(py[t_e:t_e + 11, 0], py[t_e:t_e + 11, 1], color='#eb6834', lw=1.5, label='true after (Y)')
        p0 = px[t_e]
        for k, colr, off in ((i, '#2a78d6', np.array([0., 0.])), (i + H, '#eb6834', np.array([0.6, -0.6]))):
            a.annotate('', xy=p0 + off + d[k], xytext=p0 + off, arrowprops=dict(arrowstyle='->', color=colr, lw=2.0))
        a.set_xlim(-17, 17); a.set_ylim(-17, 17); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
        a.set_title(f'key {n}, eat at t={t_e}; |pred| X {np.linalg.norm(d[i]):.2f}, Y {np.linalg.norm(d[i + H]):.2f}', fontsize=8)
    axs[0].legend(fontsize=7, frameon=False, loc='lower center')
    fig.suptitle('Generated pairs, experienced v4 model: true paths after eating (solid) and predicted 10-step displacement '
                 'from h2 (arrows; Y drawn slightly offset); cube colours of the X episode (Y: top-left green, bottom-left red gray)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 's2_main_pair_examples.png'), dpi=120); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    S = judge()
    json.dump(S, open(os.path.join(S2, 'summary.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    fig_curves(); fig_tracking(S); fig_posteat(S); fig_p2(S); fig_main(S); fig_memory(S); fig_examples()
    for enc in ENCS:
        s = S[enc]
        print(enc, {k: (s[k]['pass_single'], s[k]['pass_main']) for k in ('P1c_EW', 'P2a_EW', 'P2b_post', 'P2c', "P2c'")})
    print('saved', S2)


if __name__ == '__main__':
    main()
