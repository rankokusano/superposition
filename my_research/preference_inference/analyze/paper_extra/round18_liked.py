"""
Paper round 18 (docs/papaer_instructions_round18.md; docs/paper_extra_log.md Sec.42; exploratory, outside the
pre-registration; procedure committed before computing). Experiment 2 split by A-2's liked colour (red bias check).
No training, no new evaluation: saved per-episode arrays only, CPU (nice 10, OMP_NUM_THREADS=2).
Stage B (vdirb_analysis: Q_e = mean_{frames 1-50} L(B pleasant) - L(A pleasant), MSE-stage late 5 points, tau 1.0,
direction-present episodes (1088)); conditions none / p50 / sr50, pairs 1-6, comparisons p50 - none, p50 - sr50.
  2.1  b_red of the stage-B main regression (Q = a + b_pref s_pref + b_dir s_dir + b_red s_red + c2 y0_A2 + c1 y0_A1).
  2.2  per liked colour (red / green, within the 1088): Q = a + b_pref s_pref + b_dir s_dir + c2 y0_A2 + c1 y0_A1, and
       the same without s_dir (same episodes); reference: without s_dir on all episodes of that colour (no CI boot).
       Pair value = mean over the 5 points of the per-point OLS coefficient; CI: 6-pair t and the stage-B two-stage
       bootstrap (vdirb_analysis.coef_block / two_stage, 1000, default_rng(0); the colour mask enters as weights, so the
       number of episodes of a colour varies over resamples). Check: full b_pref vs (b_red_liked + b_green_liked)/2,
       full b_red vs (b_red_liked - b_green_liked)/2.
  2.3  liked x layout cells (all 1200): n, n with direction, mean Q_e, share of episodes whose smaller-error candidate
       (sign Q_e; 0.5 for ties) is on the liked side (all episodes) and on the direction side (direction present);
       disagreeing episodes (direction present, s_dir != s_pref) by liked colour: count and liked-side share. 6-pair t.
  2.4  the same 2.1 / 2.2 for the stage-B +-P^A2 series (oracle: A pleasant = red-liked P if red is on side A);
       Experiment 2 as first judged (Sec.35.7, E2/runs, data 'pref' and 'pref_a2p80', FP and L1, frames 1-50 / 1-100):
       U_pref = L(opp) - L(true) by liked colour, conditions none / p50 / p80 / sr80 / sr50, comparisons p50 - none,
       p50 - sr50, p80 - none, p80 - sr80; 6-pair t for all; the two-stage bootstrap (pairs + per-pair episode resample,
       1000, default_rng(0)) for 'pref' FP 1-50.
  --check  Sec.1: inputs present; stage-B main regression rebuilt vs the saved summary; split possible; E2 liked split
           rebuilt vs the saved E2 summary.
-> data/result/paper_extra/Round18/{check.json, summary.json, tables.md, fig_round18_bpref.{pdf,png}}
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vdirb_analysis as A  # noqa
import e2_aggregate as E2A  # noqa
V = A.V
OUT = os.path.join(V.RES, 'paper_extra', 'Round18')
SB = os.path.join(V.RES, 'paper_extra', 'Vdir_B', 'analysis', 'summary.json')
CONDS = ('none', 'p50', 'sr50')
DIFFS = (('p50', 'none'), ('p50', 'sr50'))
LIKED = (('red', 1), ('green', 0))
W, TAU = '1-50', 1.0
E2C = ('none', 'p50', 'p80', 'sr80', 'sr50')
E2D = (('p50', 'none'), ('p50', 'sr50'), ('p80', 'none'), ('p80', 'sr80'))
E2SETS = ('pref', 'pref_a2p80')
NAME = {'none': '経験なし', 'p50': '経験50%', 'p80': '経験80%', 'sr80': '符号無作為（経験80%）', 'sr50': '符号無作為（経験50%）'}


def design_liked(m, keep1, v, adj, alleps=False):
    c = {'pref': m['s_pref']}
    if adj:
        c['dir'] = np.sign(m['p'][W])
    c.update(a2y0=m['a2y0'], a1y0=m['a1y0'])
    names = ['a'] + list(c)
    X = np.column_stack([np.ones(len(m['s_pref']))] + [c[k] for k in names[1:]])
    keep = (m['liked'] == v) & (np.ones_like(keep1) if alleps else keep1)
    return X, names, keep


def load_stageB(m, series):
    return {c: A.load_q(series, 'FP', c, m)[W] for c in CONDS}


def blocks(m, keep1, Q, boot=True):
    R = {'full': {}, 'liked': {}}
    X, names, keep = A.design(m, 'main', W, TAU)
    assert np.array_equal(keep, keep1)
    for c in CONDS:
        R['full'][c] = A.coef_block(X, names, keep, Q[c], boot=boot)
    for a, b in DIFFS:
        R['full'][f'{a}-{b}'] = A.coef_block(X, names, keep, Q[a], Q[b], boot=boot)
    for nm, v in LIKED:
        for form, adj, alle in (('adj', True, False), ('noadj', False, False), ('noadj_all', False, True)):
            Xl, nl, kl = design_liked(m, keep1, v, adj, alle)
            bt = boot and not alle
            for c in CONDS:
                R['liked'][f'{nm}|{form}|{c}'] = A.coef_block(Xl, nl, kl, Q[c], boot=bt)
            for a, b in DIFFS:
                R['liked'][f'{nm}|{form}|{a}-{b}'] = A.coef_block(Xl, nl, kl, Q[a], Q[b], boot=bt)
    # approximation check: full b_pref ~ (b_R + b_G)/2, b_red ~ (b_R - b_G)/2 (per pair, adjusted form)
    R['approx'] = {}
    for c in list(CONDS) + [f'{a}-{b}' for a, b in DIFFS]:
        f = R['full'][c]['per_pair']
        bR, bG = np.array(R['liked'][f'red|adj|{c}']['per_pair']['pref']), np.array(R['liked'][f'green|adj|{c}']['per_pair']['pref'])
        R['approx'][c] = {'b_pref_full': float(np.mean(f['pref'])), 'half_sum': float(np.mean((bR + bG) / 2)),
                          'b_red_full': float(np.mean(f['red'])), 'half_diff': float(np.mean((bR - bG) / 2)),
                          'maxabs_pair_pref': float(np.max(np.abs(np.array(f['pref']) - (bR + bG) / 2))),
                          'maxabs_pair_red': float(np.max(np.abs(np.array(f['red']) - (bR - bG) / 2)))}
    return R


def cells(m, keep1, Q):
    sd = np.sign(m['p'][W])
    R = {'cells': {}, 'disagree': {}}
    pv = {}
    for c in CONDS:
        q = Q[c]                                                               # (6, 5, n)
        hp = np.where(q == 0, 0.5, (np.sign(q) == m['s_pref']).astype(float))
        hd = np.where(q == 0, 0.5, (np.sign(q) == sd).astype(float))
        v = {}
        for nm, lv in LIKED:
            for L in (0, 1):
                mk = (m['liked'] == lv) & (m['layout'] == L)
                md = mk & keep1
                k = f'{nm}|lay{L}'
                v[f'{k}|meanQ'] = q[..., mk].mean(-1).mean(1)
                v[f'{k}|acc_pref'] = hp[..., mk].mean(-1).mean(1)
                v[f'{k}|acc_dir'] = hd[..., md].mean(-1).mean(1)
                R['cells'].setdefault(k, {'n': int(mk.sum()), 'n_dir': int(md.sum())})
            ml = m['liked'] == lv
            v[f'{nm}|BA_pref'] = np.mean([hp[..., ml & (m['layout'] == L)].mean(-1) for L in (0, 1)], 0).mean(1)
            dis = keep1 & (sd != m['s_pref']) & ml
            R['disagree'].setdefault(nm, {'n': int(dis.sum()), 'n_dir': int((keep1 & ml).sum())})
            v[f'{nm}|acc_disagree'] = hp[..., dis].mean(-1).mean(1)
        pv[c] = v
    R['values'] = {}
    for c in CONDS:
        R['values'][c] = {k: {'per_pair': x.tolist(), 't': A.tsum(x)} for k, x in pv[c].items()}
    for a, b in DIFFS:
        R['values'][f'{a}-{b}'] = {k: {'per_pair': (pv[a][k] - pv[b][k]).tolist(), 't': A.tsum(pv[a][k] - pv[b][k])}
                                  for k in pv[a]}
    return R


# ------------------------------------------------------------------ Experiment 2 (Sec.35.7)
def e2_u(f, k, q, w):
    """-> (6, n) per-episode U_pref averaged over the 5 points (mask is the same at every point)."""
    return np.array([np.mean([E2A.U(f, s, e, k, q, w) for e in E2A.EPS], 0) for s in E2A.SEEDS])


def e2_boot(ua, mask, ub=None):
    rng = np.random.default_rng(0)
    n = ua.shape[1]
    pi = rng.integers(0, 6, (A.B, 6))
    ei = rng.integers(0, n, (A.B, 6, n))
    wv = mask[ei].astype(float)
    def val(u):
        return ((u[pi[:, :, None], ei] * wv).sum(-1) / wv.sum(-1)).mean(1)
    d = val(ua) - (val(ub) if ub is not None else 0)
    return A.pct(d)


def e2_split():
    R = {}
    for k in E2SETS:
        lk = E2A.load('none', 0, 160, k, 'FP')['a2_liked']
        for q in E2A.ERRS:
            for w in E2A.WINS:
                cell = f'{k}|{q}|{w}'
                U = {f: e2_u(f, k, q, w) for f in E2C}
                blk = {'n': {nm: int((lk == v).sum()) for nm, v in LIKED}}
                main = cell == 'pref|FP|1-50'
                for nm, v in LIKED:
                    mk = lk == v
                    for f in E2C:
                        x = U[f][:, mk].mean(1)
                        blk[f'{nm}|{f}'] = {'per_pair': x.tolist(), 't': A.tsum(x)}
                        if main:
                            blk[f'{nm}|{f}']['boot2'] = e2_boot(U[f], mk)
                    for a, b in E2D:
                        x = U[a][:, mk].mean(1) - U[b][:, mk].mean(1)
                        blk[f'{nm}|{a}-{b}'] = {'per_pair': x.tolist(), 't': A.tsum(x)}
                        if main:
                            blk[f'{nm}|{a}-{b}']['boot2'] = e2_boot(U[a], mk, U[b])
                R[cell] = blk
    return R


# ------------------------------------------------------------------ Sec.1 check
def check():
    m = A.meta()
    keep1 = (np.abs(m['p'][W]) >= TAU) & (m['p'][W] != 0)
    R = {'meta_keys': sorted(k for k in m if k not in ('p', 'mag')), 'liked_values': np.unique(m['liked']).tolist(),
         'n': int(len(m['liked'])), 'n_dir': int(keep1.sum()),
         'cells': {f'liked{v}|lay{L}': {'n': int(((m['liked'] == v) & (m['layout'] == L)).sum()),
                                        'n_dir': int(((m['liked'] == v) & (m['layout'] == L) & keep1).sum())}
                   for v in (0, 1) for L in (0, 1)},
         'red_side_check': bool(np.array_equal(m['s_pref'], np.where(m['liked'] == 1, m['s_red'], -m['s_red'])))}
    S = json.load(open(SB))
    R['stageB'] = {}
    for series in ('cpm', 'oracle'):
        Q = {}
        for c in CONDS:
            q = A.load_q(series, 'FP', c, m)
            R['stageB'][f'{series}|{c}|present'] = q is not None
            if q is not None:
                Q[c] = q[W]
        X, names, keep = A.design(m, 'main', W, TAU)
        for c in list(CONDS) + [f'{a}-{b}' for a, b in DIFFS]:
            if '-' in c:
                a, b = c.split('-')
                blk = A.coef_block(X, names, keep, Q[a], Q[b], boot=False)
                sv = S['diff'][f'{series}|FP|{W}|{TAU}|main|{c}']
            else:
                blk = A.coef_block(X, names, keep, Q[c], boot=False)
                sv = S['coef'][f'{series}|FP|{W}|{TAU}|main|{c}']
            R['stageB'][f'{series}|{c}|maxabs_per_pair'] = max(
                float(np.max(np.abs(np.array(blk['per_pair'][k]) - np.array(sv['per_pair'][k])))) for k in names)
            R['stageB'][f'{series}|{c}|b_pref'] = blk['t']['pref']['mean']
            R['stageB'][f'{series}|{c}|b_pref_saved'] = sv['t']['pref']['mean']
    # bootstrap reproduction on the primary cell
    blk = A.coef_block(X, names, keep, load_stageB(m, 'cpm')['p50'], boot=True)
    sv = S['coef'][f'cpm|FP|{W}|{TAU}|main|p50']
    R['stageB']['cpm|p50|boot2_equal'] = all(np.allclose(blk['boot2'][k], sv['boot2'][k], rtol=0, atol=1e-12) for k in names)
    R['stageB']['cpm|p50|boot2_pref'] = blk['boot2']['pref']
    # E2
    E = json.load(open(os.path.join(E2A.D, 'summary.json')))
    R['e2'] = {'files_missing': []}
    for f in E2C:
        for s in E2A.SEEDS:
            for e in E2A.EPS:
                for k in E2SETS:
                    for st in ('mse', 'l1'):
                        p = os.path.join(E2A.RUNS, f'{f}_{st}_s{s}_ep{e}_{k}.npz')
                        if not os.path.exists(p):
                            R['e2']['files_missing'].append(os.path.basename(p))
    mx = 0.0
    for cell, v in E['liked_split'].items():
        k, q, w = cell.split('|')
        lk = E2A.load('none', 0, 160, k, q)['a2_liked']
        for f in v:
            for nm, lv in LIKED:
                x = e2_u(f, k, q, w)[:, lk == lv].mean(1)
                mx = max(mx, float(np.max(np.abs(x - np.array(v[f][nm])))))
    R['e2']['liked_split_maxabs'] = mx
    R['e2']['a2_liked_values'] = np.unique(E2A.load('none', 0, 160, 'pref', 'FP')['a2_liked']).tolist()
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, 'check.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(R, ensure_ascii=False, indent=1), flush=True)


# ------------------------------------------------------------------ run
def run():
    m = A.meta()
    keep1 = (np.abs(m['p'][W]) >= TAU) & (m['p'][W] != 0)
    R = {'note': 'exploratory, outside the pre-registration (Sec.42)', 'window': W, 'tau': TAU}
    for series in ('cpm', 'oracle'):
        Q = load_stageB(m, series)
        R[series] = blocks(m, keep1, Q)
        R[series]['cells'] = cells(m, keep1, Q)
        print('done', series, flush=True)
    R['e2'] = e2_split()
    print('done e2', flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), ensure_ascii=False, indent=0)
    tables(R)
    fig(R)


def ci(t, b=None, d=4):
    s = f'{t["mean"]:+.{d}f} [{t["ci95"][0]:+.{d}f}, {t["ci95"][1]:+.{d}f}]'
    return s + (f'・[{b[0]:+.{d}f}, {b[1]:+.{d}f}]' if b else '')


def tables(R):
    cols = list(CONDS) + [f'{a}-{b}' for a, b in DIFFS]
    hdr = '| | ' + ' | '.join(NAME.get(c, c) if '-' not in c else ' − '.join(NAME[x] for x in c.split('-')) for c in cols) + ' |'
    sep = '|' + '---|' * (len(cols) + 1)
    L = ['# 第18弾（探索・事前登録外）：集計の表', '',
         '値は組の平均 [6組の t の 95% CI]・[2段のブートストラップ 95%]。段階B：FP、MSE 期間の最終5点、対象フレーム 1〜50、τ = 1.0。', '']
    for series, sname in (('cpm', 'c±（主）'), ('oracle', '±P^A2')):
        S = R[series]
        L += [f'## {sname}：2.1 主の回帰（全 1088）の係数', '', hdr, sep]
        for k in ('pref', 'dir', 'red'):
            L.append(f'| b_{k} | ' + ' | '.join(ci(S['full'][c]['t'][k], S['full'][c].get('boot2', {}).get(k)) for c in cols) + ' |')
        L += ['', f'## {sname}：2.2 好み別の b_pref（向きを調整する形・しない形）', '', hdr, sep]
        for nm, _ in LIKED:
            for form, fn in (('adj', '調整'), ('noadj', '調整なし'), ('noadj_all', '調整なし・全エピソード（参考、t のみ）')):
                k0 = f'{nm}|{form}|'
                row = [ci(S['liked'][k0 + c]['t']['pref'], S['liked'][k0 + c].get('boot2', {}).get('pref')) for c in cols]
                L.append(f'| {"赤好き" if nm == "red" else "緑好き"} {fn}（n = {S["liked"][k0 + "p50"]["n_used"]}） | ' + ' | '.join(row) + ' |')
            k0 = f'{nm}|adj|'
            L.append(f'| {"赤好き" if nm == "red" else "緑好き"} 調整の b_dir | ' +
                     ' | '.join(ci(S['liked'][k0 + c]['t']['dir'], S['liked'][k0 + c].get('boot2', {}).get('dir')) for c in cols) + ' |')
        L += ['', '近似の確認（調整する形、組の平均）：', '', '| | b_pref（全体） | (b赤+b緑)/2 | 組ごとの差の最大 | b_red（全体） | (b赤−b緑)/2 | 組ごとの差の最大 |',
              '|---|---|---|---|---|---|---|']
        for c in cols:
            a = S['approx'][c]
            L.append(f'| {c} | {a["b_pref_full"]:+.4f} | {a["half_sum"]:+.4f} | {a["maxabs_pair_pref"]:.4f} | {a["b_red_full"]:+.4f} | '
                     f'{a["half_diff"]:+.4f} | {a["maxabs_pair_red"]:.4f} |')
        C = S['cells']
        L += ['', f'## {sname}：2.3 好み × 配置のセル（6組の t の 95% CI）', '',
              '| セル | n（向きあり） | 量 | ' + ' | '.join(NAME.get(c, c) if '-' not in c else c for c in cols) + ' |', '|' + '---|' * (len(cols) + 3)]
        for cell, nn in C['cells'].items():
            for q, qn in (('meanQ', 'Q_e の平均'), ('acc_pref', '好みの側'), ('acc_dir', '向きの側')):
                L.append(f'| {cell} | {nn["n"]}（{nn["n_dir"]}） | {qn} | ' + ' | '.join(ci(C['values'][c][f'{cell}|{q}']['t'], d=3 if q != 'meanQ' else 4) for c in cols) + ' |')
        for nm, _ in LIKED:
            L.append(f'| {nm} | | BA（好み、2配置の平均） | ' + ' | '.join(ci(C['values'][c][f'{nm}|BA_pref']['t'], d=3) for c in cols) + ' |')
            L.append(f'| {nm} | 食い違い {C["disagree"][nm]["n"]} / 向きあり {C["disagree"][nm]["n_dir"]} | 食い違う場面で好みの側 | '
                     + ' | '.join(ci(C['values'][c][f'{nm}|acc_disagree']['t'], d=3) for c in cols) + ' |')
        L.append('')
    L += ['## 2.4 当初の判定（E2、§35.7）の好み別の U_pref（6組の平均 [t の 95% CI]。好み混合・FP・1〜50 は 2段のブートストラップも）', '']
    e2c = list(E2C) + [f'{a}-{b}' for a, b in E2D]
    L += ['| データ・誤差・区間 | 好み（n） | ' + ' | '.join(e2c) + ' |', '|' + '---|' * (len(e2c) + 2)]
    for cell, blk in R['e2'].items():
        for nm, _ in LIKED:
            L.append(f'| {cell} | {nm}（{blk["n"][nm]}） | ' + ' | '.join(ci(blk[f'{nm}|{c}']['t'], blk[f'{nm}|{c}'].get('boot2')) for c in e2c) + ' |')
    txt = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'tables.md'), 'w').write(txt)
    print(txt)


def fig(R):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    panels = (('cpm', 'adj', 'c± (main), direction-adjusted'), ('cpm', 'noadj', 'c± (main), not adjusted'),
              ('oracle', 'adj', '±P^A2, direction-adjusted'))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    col = {'red': '#c0392b', 'green': '#27ae60'}
    for ax, (series, form, title) in zip(axs, panels):
        for i, c in enumerate(CONDS):
            for j, (nm, _) in enumerate(LIKED):
                b = R[series]['liked'][f'{nm}|{form}|{c}']
                x0 = i + (j - 0.5) * 0.3
                pp = b['per_pair']['pref']
                ax.scatter(np.full(len(pp), x0) + np.linspace(-0.05, 0.05, len(pp)), pp, s=14, color=col[nm], alpha=0.55)
                t = b['t']['pref']
                ax.errorbar(x0 + 0.09, t['mean'], yerr=[[t['mean'] - t['ci95'][0]], [t['ci95'][1] - t['mean']]],
                            fmt='o', color=col[nm], ms=6, capsize=3, label=f'A-2 likes {nm}' if i == 0 else None)
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels(['no experience', '50%', 'sign-random (50%)'])
        ax.set_title(title, fontsize=10)
    axs[0].set_ylabel('b_pref within liked colour\n(FP, frames 1-50; pairs and mean, t 95% CI)')
    axs[0].legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round18_bpref.{ext}'), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    check() if a.check else run()


if __name__ == '__main__':
    main()
