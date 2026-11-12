"""
Paper round 33 aggregation (docs/paper_extra_log.md Sec.57.2; quantities and reading rules fixed before computing).
CPU. Reads Round33/{flows,real,pos,geo}/ -> Round33/analysis/{summary.json, tables.md, fig_round33_*.{png,pdf}}.
Correlations pool episodes x time x ranges (per-episode sufficient statistics); value = mean over the 6 pairs;
CIs = 6-pair t 95% CI and two-stage bootstrap (pairs -> episodes, 1000) (round30_aggregate.stat). Pair-independent
rows (real-image geometry) have the episode bootstrap only.
"""

import glob
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round30_aggregate as A30  # noqa
import round30 as R30m  # noqa
import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

_PI = os.path.dirname(os.path.dirname(_HERE))
R33 = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round33')
OUT = os.path.join(R33, 'analysis')
DATA = os.path.join(_PI, 'data', 'data')
PAIRS = tuple(range(6))
CONDS = ('p50', 'none', 'sr50', 'p80')
S1DATA = {'p50': 'v11_s1_mixed', 'sr50': 'v11_s1_mixed', 'none': 'v11_s1_random', 'p80': 'v13_s1_mixed80'}
BS = {'c0167': [0, 1, 6, 7], 'all': list(range(8))}
C1 = [(v, k) for v in ('i', 'ii') for k in ('lin', 'mlp')]
COMPARE = {'r30_geometry_pred_b': 0.800, 'r30_geometry_pred_c': 0.655}


def sf(pred, targ, bins):
    return R30m.suff(np.asarray(pred, np.float32), np.asarray(targ, np.float32), bins)


def st(preds, targ, bins=BS['c0167']):
    """preds: list over pairs of (N, T, 8) (or a single array) -> round30_aggregate.stat dict."""
    preds = preds if isinstance(preds, list) else [preds]
    return A30.stat([sf(p, targ, bins) for p in preds])


def pt(preds, targ, bins):
    preds = preds if isinstance(preds, list) else [preds]
    return float(np.nanmean([A30.r_r2(sf(p, targ, bins).sum(0))[0] for p in preds]))


def per_step(preds, targ, bins=BS['c0167']):
    """(c) time course: per frozen step k, correlation pooled over episodes x ranges, mean over pairs."""
    preds = preds if isinstance(preds, list) else [preds]
    K = targ.shape[1]
    return [float(np.nanmean([A30.r_r2(sf(p[:, k:k + 1], targ[:, k:k + 1], bins).sum(0))[0] for p in preds])) for k in range(K)]


def missing(preds):
    preds = preds if isinstance(preds, list) else [preds]
    return float(np.mean([np.isnan(np.asarray(p, np.float32)).any(-1).mean() for p in preds]))


def main():
    os.makedirs(OUT, exist_ok=True)
    real = np.load(os.path.join(R33, 'real', 'real.npz'))
    T = {k: real[k] for k in ('truth_b', 'truth_b2', 'truth_c', 'truth_c_i', 'truth_c_old', 'truth_c_old_i', 'own_pleasure_b')}
    geo = {}
    for f in sorted(glob.glob(os.path.join(R33, 'geo', 'part*.npz'))):
        z = np.load(f)
        geo.update({k: z[k] for k in z.files})
    res = {'compare': COMPARE, 'rerender_mismatch_b': int(geo.get('rerender_mismatch_b', -1)), 'rows': {}, 'steps': {}}
    R = res['rows']
    for cond in CONDS:
        F = [np.load(os.path.join(R33, 'flows', f'{cond}_s{s}.npz')) for s in PAIRS]
        Pz = np.load(os.path.join(R33, 'pos', f'{cond}.npz'))
        own = R30m.pleasure_all(h5py.File(os.path.join(DATA, S1DATA[cond], 'data.h5'), 'r')['test']['self_vision'][()])
        res[f'{cond}|own_layout'] = {k: float(Pz[k]) for k in Pz.files if k.startswith('own_')}

        def add(key, preds, targ, desc=''):
            R[key] = {'c0167': st(preds, targ), 'all': pt(preds, targ, BS['all']),
                      'per_range': [pt(preds, targ, [b]) for b in range(8)], 'missing': missing(preds), 'desc': desc}

        # case 1
        for v, kd in C1:
            g = lambda sc: [f[f'{v}|{kd}|{sc}'] for f in F]  # noqa
            add(f'{cond}|case1_{v}_{kd}|self', g('self'), own if v == 'i' else own[:, 1:])
            add(f'{cond}|case1_{v}_{kd}|b', g('b'), T['truth_b'])
            add(f'{cond}|case1_{v}_{kd}|b_own', g('b_own'), T['own_pleasure_b'], 'R(h1) on the (b) data vs own pleasure')
            add(f'{cond}|case1_{v}_{kd}|b2', g('b2'), T['truth_b2'])
            tc, to = (T['truth_c_i'], T['truth_c_old_i']) if v == 'i' else (T['truth_c'], T['truth_c_old'])
            add(f'{cond}|case1_{v}_{kd}|c', g('c'), tc)
            add(f'{cond}|case1_{v}_{kd}|c_old', g('c'), to, 'vs the no-swap counterpart')
            res['steps'][f'{cond}|case1_{v}_{kd}'] = per_step(g('c'), tc)
        # diag
        for v in ('i', 'ii'):
            add(f'{cond}|diag_{v}|b', [f[f'{v}|diag|b_test'] for f in F], T['truth_b'][150:])
            tc = T['truth_c_i'] if v == 'i' else T['truth_c']
            add(f'{cond}|diag_{v}|c', [f[f'{v}|diag|c'] for f in F], tc)
            res['steps'][f'{cond}|diag_{v}'] = per_step([f[f'{v}|diag|c'] for f in F], tc)
        # case 2
        g2 = lambda k: [Pz[f's{s}|{k}'] for s in PAIRS]  # noqa
        add(f'{cond}|case2|self', g2('self'), own)
        for sc, targ in (('b', T['truth_b']), ('b2', T['truth_b2']), ('c', T['truth_c'])):
            add(f'{cond}|case2_true|{sc}', g2(f'{sc}_true'), targ)
            add(f'{cond}|case2_found|{sc}', g2(f'{sc}_found'), targ)
        add(f'{cond}|case2_found_pred|b', g2('b_found_pred'), T['truth_b'], 'found in the predicted images')
        add(f'{cond}|case2_found_real|c', g2('c_found_real'), T['truth_c'], 'found in the real images')
        add(f'{cond}|case2_found|c_old', g2('c_found'), T['truth_c_old'], 'vs the no-swap counterpart')
        res['steps'][f'{cond}|case2_found'] = per_step(g2('c_found'), T['truth_c'])
        res['steps'][f'{cond}|case2_true'] = per_step(g2('c_true'), T['truth_c'])
        # case 3
        gp = [geo[f'pred_c_{cond}_s{s}'] for s in PAIRS]
        add(f'{cond}|case3_pred|c', gp, T['truth_c'])
        add(f'{cond}|case3_pred|c_old', gp, T['truth_c_old'], 'vs the no-swap counterpart')
        res['steps'][f'{cond}|case3_pred'] = per_step(gp, T['truth_c'])
        if cond == 'p50':
            add('p50|case3_pred|b', [geo[f'pred_b_p50_s{s}'] for s in PAIRS], T['truth_b'], 'predicted images (round 30)')
        print('done', cond, flush=True)
    add_real = lambda key, arr, targ: R.__setitem__(key, {'c0167': st(arr, targ), 'all': pt(arr, targ, BS['all']),  # noqa
                                                         'per_range': [pt(arr, targ, [b]) for b in range(8)],
                                                         'missing': missing(arr), 'desc': 'real images, pair-independent'})
    add_real('all|case3_real|b', geo['real_b'], T['truth_b'])
    add_real('all|case3_real|b2', geo['real_b2'], T['truth_b2'])
    add_real('all|case3_real|c', geo['real_c'], T['truth_c'])
    res['steps']['all|case3_real'] = per_step(geo['real_c'], T['truth_c'])

    # ---------------- reading table (p50, (b), ranges 0, 1, 6, 7)
    m = lambda k: R[k]['c0167']['r']['mean']  # noqa
    c1 = {f'{v}_{kd}': m(f'p50|case1_{v}_{kd}|b') for v, kd in C1}
    dg = {v: m(f'p50|diag_{v}|b') for v in ('i', 'ii')}
    c2 = m('p50|case2_found|b')
    best1, bestd = max(c1.values()), max(dg.values())
    row1 = 1 if best1 >= 0.7 else (2 if bestd >= 0.7 else 3)
    row2 = None if row1 == 1 else (4 if c2 >= 0.7 else 5)
    res['table'] = {'case1': c1, 'case1_best': best1, 'diag': dg, 'diag_best': bestd, 'case2_found': c2,
                    'case2_true': m('p50|case2_true|b'), 'case3_real': m('all|case3_real|b'),
                    'row_case1': row1, 'row_case2': row2}
    json.dump(res, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=float)
    tables(res)
    figs(res)
    print(json.dumps(res['table']))


def tables(res):
    R = res['rows']
    L = ['# 第33弾 集計の表（`round33_aggregate.py`）', '',
         '値：6 組の平均 [6 組の t の 95% CI] [2 段のブートストラップ]。範囲 0・1・6・7。実際の画像の幾何は組によらない（回のブートストラップだけ）。', '',
         '| 条件 | 案 | 場面 | 0・1・6・7 | 全範囲 | 範囲ごと（0〜7） | 欠け | 注 |', '|---|---|---|---|---|---|---|---|']
    for k, v in R.items():
        c, a, sc = k.split('|')
        pr = '／'.join('—' if not np.isfinite(x) else f'{x:+.2f}' for x in v['per_range'])
        L.append(f"| {c} | {a} | {sc} | {A30.fm(v['c0167']['r'])} | {v['all']:+.3f} | {pr} | {v['missing']:.3f} | {v['desc']} |")
    L += ['', '## (c) 止めている間の時点ごとの相関（0・1・6・7、組の平均）', '', '| 条件・案 | k = 1〜15 |', '|---|---|']
    for k, v in res['steps'].items():
        L.append(f"| {k} | {' '.join(f'{x:+.2f}' for x in v)} |")
    L += ['', '## 自分の視界からの配置の読み取り（M の学び）', '']
    for c in CONDS:
        L.append(f"- {c}: {json.dumps(res[f'{c}|own_layout'])}")
    L += ['', f"- 幾何の描き直しの確かめ（不一致の数）：{res['rerender_mismatch_b']}", '', '## 読み方の表', '',
          f"- {json.dumps(res['table'], ensure_ascii=False, default=float)}"]
    open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(L) + '\n')


def figs(res):
    R = res['rows']
    schemes = [('case1_i_lin', '1(i) lin'), ('case1_i_mlp', '1(i) net'), ('case1_ii_lin', '1(ii) lin'), ('case1_ii_mlp', '1(ii) net'),
               ('diag_i', 'diag(i)'), ('diag_ii', 'diag(ii)'), ('case2_true', '2 true pos'), ('case2_found', '2 found pos'),
               ('case3', '3 geometry')]
    fig, ax = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
    for j, c in enumerate(CONDS):
        for i, sc in enumerate(('b', 'c')):
            a = ax[i, j]
            for x, (key, lab) in enumerate(schemes):
                if key == 'case3':
                    k = 'all|case3_real|b' if sc == 'b' else f'{c}|case3_pred|c'
                else:
                    k = f'{c}|{key}|{sc}'
                if k not in R:
                    continue
                r = R[k]['c0167']['r']
                pp = [p for p in r['per_pair'] if np.isfinite(p)] if 'per_pair' in r else []
                a.scatter([x] * len(pp), pp, s=12, color='#888888', zorder=2)
                a.scatter([x], [r['mean']], s=60, marker='_', color='#d62728', linewidths=3, zorder=3)
            a.axhline(0.7, color='k', lw=0.6, ls='--')
            a.axhline(0, color='k', lw=0.4)
            a.set_xticks(range(len(schemes)))
            a.set_xticklabels([s[1] for s in schemes], rotation=60, fontsize=7)
            a.set_title(f'{c}  ({sc}) {"observing" if sc == "b" else "swap, frozen 15 steps"}', fontsize=9)
    ax[0, 0].set_ylabel('r with A-2 red - green (ranges 0,1,6,7)')
    ax[1, 0].set_ylabel('r with A-2 red - green (ranges 0,1,6,7)')
    fig.suptitle('Round 33: case 1 (internal-state sharing), diagnostic, case 2 (position map), case 3 (geometry; (b) real images, (c) predicted images). Grey = pairs, red = mean', fontsize=9)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_round33_r.{e}'), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8), sharey=True)
    for j, c in enumerate(CONDS):
        a = ax[j]
        for key, lab, col in (('case1_i_lin', '1(i) lin', '#1f77b4'), ('case1_ii_lin', '1(ii) lin', '#aec7e8'),
                              ('case1_ii_mlp', '1(ii) net', '#9edae5'), ('diag_ii', 'diag(ii)', '#7f7f7f'),
                              ('case2_true', '2 true pos', '#2ca02c'), ('case2_found', '2 found pos', '#98df8a'),
                              ('case3_pred', '3 geometry (pred)', '#d62728')):
            k = f'{c}|{key}'
            if k in res['steps']:
                a.plot(range(1, 16), res['steps'][k], label=lab, color=col)
        a.plot(range(1, 16), res['steps']['all|case3_real'], label='3 geometry (real)', color='#ff9896', ls='--')
        a.axhline(0, color='k', lw=0.4)
        a.set_title(f'{c}: (c) frozen steps', fontsize=9)
        a.set_xlabel('frozen step k')
    ax[0].set_ylabel('r (ranges 0,1,6,7)')
    ax[-1].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_round33_steps.{e}'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
