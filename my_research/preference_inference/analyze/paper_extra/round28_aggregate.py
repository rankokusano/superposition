"""
Paper round 28 aggregation (docs/paper_extra_log.md Sec.52.2; definitions and reading rules fixed there before
computing). Reads round 27 runs (pairs 1-3, mixed) and round 28 runs (pairs 4-6 mixed; p50 pairs 1-6 not mixed).
  s1  Sec.1: D_T,late (epochs 40-50) per condition x pair (all / red / green; episode bootstrap); 6 pairs, pairs 1-3,
      pairs 4-6 (t CI over the pairs + two-stage bootstrap, round21_aggregate.stat); p50 - none (main), p50 - sr50;
      reading; C_self below half; S_WT (B) by component at epochs 0 / 50 (6 pairs, descriptive).
      -> Round28/analysis/s1_{summary.json,tables.md}, fig_round28_s1.{pdf,png}
  s2  Sec.2: p50 no-mix vs mix: Delta D = D_T,late(no mix) - D_T,late(mix) per pair (paired per episode), Delta C =
      C_self ep50 (no mix - mix); reading; per-run Spearman of C_self(e) vs D_T(e) over the 11 evaluations.
      -> Round28/analysis/s2_{summary.json,tables.md}, fig_round28_s2.{pdf,png}
Usage: python analyze/paper_extra/round28_aggregate.py s1|s2|all
"""

import glob
import json
import os
import sys

import h5py
import numpy as np
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round27_aggregate as G  # noqa  (cls, fmt, comps, diffs_from, d_of, boot_ep, mv)
import round21_aggregate as A  # noqa  (stat)
import vdir_stageA as VA  # noqa  (tsum)

R27, R28 = G.R27, os.path.join(G.RES, 'paper_extra', 'Round28')
AN = os.path.join(R28, 'analysis')
CONDS = ('p50', 'sr50', 'none')
CN = {'p50': '経験50%', 'sr50': '符号無作為（経験50%）', 'none': '経験なし', 'p50-none': '経験50% − 経験なし',
      'p50-sr50': '経験50% − 符号無作為', 'nomix': '経験50%・混ぜない', 'mix': '経験50%・混ぜる',
      'nomix-mix': '混ぜない − 混ぜる'}
PAIRS = tuple(range(6))
SETS = {'6': PAIRS, '1-3': (0, 1, 2), '4-6': (3, 4, 5)}
LIKES, LN = G.LIKES, G.LN
LV = {'all': None, 'red': 1, 'green': 0}
LATE = (40, 45, 50)


def load_run(p):
    ev = {int(os.path.basename(e)[2:5]): np.load(e) for e in sorted(glob.glob(os.path.join(p, 'eval', 'ep*.npz')))}
    sg = {int(os.path.basename(e)[6:9]): np.load(e) for e in sorted(glob.glob(os.path.join(p, 'eval', 'sig_ep*.npz')))}
    return {'ev': ev, 'sig': sg, 'log': json.load(open(os.path.join(p, 'log.json'))), 'path': p,
            'done': os.path.exists(os.path.join(p, 'done.json'))}


def runs_mix():
    out = {}
    for c in CONDS:
        for s in PAIRS:
            p = os.path.join(R27, 'runs', f's2_{c}_s{s}') if s < 3 else os.path.join(R28, 'runs', f'mix_{c}_s{s}')
            if os.path.exists(os.path.join(p, 'log.json')):
                out[(c, s)] = load_run(p)
    return out


def runs_nomix():
    out = {}
    for s in PAIRS:
        p = os.path.join(R28, 'runs', f'nomix_p50_s{s}')
        if os.path.exists(os.path.join(p, 'log.json')):
            out[s] = load_run(p)
    return out


def late_of(r, lk):
    if not all(e in r['ev'] for e in LATE):
        return None
    with np.errstate(all='ignore'):
        return np.nanmean(np.stack([G.d_of(r['ev'][e], LV[lk]) for e in LATE]), 0)


def cself(r, e):
    return r['ev'][e]['c_self'].astype(np.float64)


def f3(st):
    return f"{st['mean']:+.4f} [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}]"


def tci(x):
    t = VA.tsum(list(map(float, x)))
    return {'mean': float(np.mean(x)), 'ci95': t['ci95']}


# ---------------------------------------------------------------- Sec.1
def s1():
    os.makedirs(AN, exist_ok=True)
    R = runs_mix()
    late = {(c, s, lk): late_of(r, lk) for (c, s), r in R.items() for lk in LIKES}
    late = {k: v for k, v in late.items() if v is not None}
    per = {}
    for (c, s, lk), v in late.items():
        per[f'{c}|{s}|{lk}'] = G.boot_ep(v)
        per[f'{c}|{s}|{lk}|ep0'] = G.boot_ep(G.d_of(R[(c, s)]['ev'][0], LV[lk]))
    for o in ('none', 'sr50'):
        for s in PAIRS:
            if ('p50', s, 'all') in late and (o, s, 'all') in late:
                per[f'p50-{o}|{s}|all'] = G.boot_ep(late[('p50', s, 'all')] - late[(o, s, 'all')])
    multi = {}
    for sn, ps in SETS.items():
        for c in CONDS:
            for lk in LIKES:
                if all((c, s, lk) in late for s in ps):
                    multi[f'{sn}|{c}|{lk}'] = A.stat(np.stack([late[(c, s, lk)] for s in ps]))
        for o in ('none', 'sr50'):
            for lk in LIKES:
                if all(('p50', s, lk) in late and (o, s, lk) in late for s in ps):
                    multi[f'{sn}|p50-{o}|{lk}'] = A.stat(np.stack([late[('p50', s, lk)] - late[(o, s, lk)] for s in ps]))
    # reading (6 pairs, all)
    rd = {}
    if '6|p50-none|all' in multi and '6|p50-sr50|all' in multi:
        a, b = G.cls(multi['6|p50-none|all']), G.cls(multi['6|p50-sr50|all'])
        rd = {'p50-none': a, 'p50-sr50': b}
        if '食い違う' in (a, b):
            rd['row'] = None
            rd['note'] = '食い違う（判定しない）'
        elif a == '<0' and b == '<0':
            rd['row'] = 1
        elif a == '<0' and b == '≈0':
            rd['row'] = 2
        elif a == '≈0' and b == '≈0':
            rd['row'] = 3
        elif a == '>0':
            rd['row'] = 4
        else:
            rd['row'] = None
            rd['note'] = '表に無い組み合わせ（そのまま書く）'
        rd['cls_conditions'] = {c: G.cls(multi[f'6|{c}|all']) for c in CONDS if f'6|{c}|all' in multi}
    half = {}
    mon = {}
    for (c, s), r in R.items():
        mon[f'{c}|{s}'] = {str(e): {'d_t': float(np.nanmean(G.d_of(z))), 'c_self': float(np.nanmean(z['c_self'])),
                                    'l1_T_V': float(z['l1_T_V']), 'l1_T_B': float(z['l1_T_B'])} for e, z in r['ev'].items()}
        if 0 in r['ev'] and 50 in r['ev']:
            c0, c5 = float(np.nanmean(cself(r, 0))), float(np.nanmean(cself(r, 50)))
            half[f'{c}|{s}'] = {'c_self_0': c0, 'c_self_50': c5, 'below_half': bool(c5 < 0.5 * c0)}
    # S_WT (B) at epochs 0 / 50, 6 pairs, r25 train swap episodes (descriptive)
    g = h5py.File(G.D25, 'r')
    swm, liked = g['train']['swap_frame'][()] >= 0, g['train']['a2_liked'][()].astype(np.int64)
    SW = {}
    for c in CONDS:
        arr = {}
        for e in (0, 50):
            if all((c, s) in R and e in R[(c, s)]['sig'] for s in PAIRS):
                L = {i: np.stack([G.comps(R[(c, s)]['sig'][e][f'tr_B_{i}'].astype(np.float64))[swm] for s in PAIRS])
                     for i in ('T', 'W', 'Z')}
                arr[e] = G.diffs_from(L, liked[swm])
        for e, dd in arr.items():
            for comp in G.COMP:
                for lk in LIKES:
                    SW[f'{c}|ep{e}|{comp}|{lk}'] = A.stat(dd[('S_WT', comp, lk)])
        if 0 in arr and 50 in arr:
            for comp in G.COMP:
                for lk in LIKES:
                    SW[f'{c}|ep50-ep0|{comp}|{lk}'] = A.stat(arr[50][('S_WT', comp, lk)] - arr[0][('S_WT', comp, lk)])
    json.dump({'per_pair': per, 'multi': multi, 'reading': rd, 'c_self_half': half, 'monitor': mon, 'swt': SW,
               'runs': {f'{c}|{s}': {'path': r['path'], 'done': r['done'], 'last_epoch': max(r['ev']),
                                     'epoch_sec_median': float(np.median([x['sec'] for x in r['log']['epochs']]))
                                     if r['log']['epochs'] else None} for (c, s), r in R.items()}},
              open(os.path.join(AN, 's1_summary.json'), 'w'), indent=1, ensure_ascii=False)
    # tables
    L = ['# 第28弾 §1 組4〜6 を足した 6 組（T、B、足した層 + 共有の LSTM、段階Iのデータを混ぜる）', '',
         '値：平均 [組の t の 95% CI] [2段のブートストラップの 95% CI]（組ごとはエピソードのブートストラップ）。'
         '$D_{T,late}$ = 40・45・50 エポックの平均。組1〜3 は第27弾の回。', '',
         '## 読み方の表の当てはめ（6 組、全部）', '', '```', json.dumps(rd, ensure_ascii=False, indent=1), '```', '']
    for sn in ('6', '1-3', '4-6'):
        L += [f'## $D_{{T,late}}$（組 {sn}）', '', '| 条件・差 | 全部 | 赤好き | 緑好き | 判定（全部） |', '|---|---|---|---|---|']
        for c in list(CONDS) + ['p50-none', 'p50-sr50']:
            if f'{sn}|{c}|all' in multi:
                row = [G.fmt(multi[f'{sn}|{c}|{lk}']) if f'{sn}|{c}|{lk}' in multi else '—' for lk in LIKES]
                L.append(f'| {CN[c]} | ' + ' | '.join(row) + f" | {G.cls(multi[f'{sn}|{c}|all'])} |")
        L.append('')
    L += ['## 組ごとの $D_T$（全部：0 エポック目と late、late の赤好き・緑好き）', '',
          '| 組 | 条件・差 | 0 エポック目 | late 全部 | late 赤好き | late 緑好き | 判定（late 全部） |', '|---|---|---|---|---|---|---|']
    for s in PAIRS:
        for c in CONDS:
            if f'{c}|{s}|all' in per:
                L.append(f"| {s + 1} | {CN[c]} | {f3(per[f'{c}|{s}|all|ep0'])} | {f3(per[f'{c}|{s}|all'])} | "
                         f"{f3(per[f'{c}|{s}|red'])} | {f3(per[f'{c}|{s}|green'])} | {G.mv(per[f'{c}|{s}|all'])} |")
        for o in ('none', 'sr50'):
            k = f'p50-{o}|{s}|all'
            if k in per:
                L.append(f"| {s + 1} | {CN['p50-' + o]} | — | {f3(per[k])} | — | — | {G.mv(per[k])} |")
    L += ['', '## $C_{self}$：50 エポック目が 0 エポック目の半分を下回った回', '',
          '| 条件 | 組 | 0 エポック目 | 50 エポック目 | 半分を下回る |', '|---|---|---|---|---|']
    for k, v in sorted(half.items()):
        c, s = k.split('|')
        L.append(f"| {CN[c]} | {int(s) + 1} | {v['c_self_0']:+.3f} | {v['c_self_50']:+.3f} | {'はい' if v['below_half'] else 'いいえ'} |")
    eps = sorted({int(e) for d in mon.values() for e in d})
    L += ['', '## $D_T(e)$（全部）・$C_{self}$・A-1 の視界の L1（V／B）', '',
          '| 回 | 量 | ' + ' | '.join(f'ep{e}' for e in eps) + ' |', '|---|---|' + '---|' * len(eps)]
    for (c, s) in sorted(R):
        mm = mon[f'{c}|{s}']
        for q, nm, fm in (('d_t', '$D_T$', '{:+.3f}'), ('c_self', '$C_{self}$', '{:+.3f}'), ('l1_T_V', 'L1 V', '{:.4f}'),
                          ('l1_T_B', 'L1 B', '{:.4f}')):
            L.append(f'| {c} 組{s + 1} | {nm} | ' + ' | '.join(fm.format(mm[str(e)][q]) if str(e) in mm else '' for e in eps) + ' |')
    L += ['', '## 0・50 エポック目の $S_{\\mathrm{WT}}$（B、第25弾の学習用データ・入れ替えあり、6 組、記述）', '',
          '| 条件 | 成分 | 好み | 0 エポック目 | 50 エポック目 | 50 − 0 | 判定（0／50／差） |', '|---|---|---|---|---|---|---|']
    for c in CONDS:
        for comp in G.COMP:
            for lk in LIKES:
                k0, k5, kd = f'{c}|ep0|{comp}|{lk}', f'{c}|ep50|{comp}|{lk}', f'{c}|ep50-ep0|{comp}|{lk}'
                if k0 in SW and k5 in SW:
                    L.append(f'| {CN[c]} | {G.CMN[comp]} | {LN[lk]} | {G.fmt(SW[k0])} | {G.fmt(SW[k5])} | {G.fmt(SW[kd])} | '
                             f'{G.cls(SW[k0])}／{G.cls(SW[k5])}／{G.cls(SW[kd])} |')
    open(os.path.join(AN, 's1_tables.md'), 'w').write('\n'.join(L) + '\n')
    fig_s1(R, late, multi)
    print(json.dumps(rd, ensure_ascii=False, indent=1))


def fig_s1(R, late, multi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'p50': 'tab:red', 'sr50': 'tab:blue', 'none': 'tab:gray'}
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for j, c in enumerate(CONDS):
        a = ax[j]
        allc = []
        for s in PAIRS:
            if (c, s) not in R:
                continue
            ev = R[(c, s)]['ev']
            es = sorted(ev)
            y = [float(np.nanmean(G.d_of(ev[e]))) for e in es]
            a.plot(es, y, '-', color=col[c], alpha=0.35, lw=1, marker='.', label=f'pair {s + 1}' if j == 0 else None)
            a.annotate(str(s + 1), (es[-1], y[-1]), fontsize=7, color=col[c])
            allc.append(dict(zip(es, y)))
        es = sorted(set.intersection(*[set(d) for d in allc])) if allc else []
        if es:
            a.plot(es, [np.mean([d[e] for d in allc]) for e in es], '-', color=col[c], lw=3, label='mean of pairs')
        a.axhline(0, color='k', lw=0.6)
        a.set_title(f'{c}: D_T(e), all (thin = pairs, thick = mean)', fontsize=9)
        a.set_xlabel('epoch')
    a = ax[3]
    items = list(CONDS) + ['p50-none', 'p50-sr50']
    for i, c in enumerate(items):
        for s in PAIRS:
            if c in CONDS:
                v = late.get((c, s, 'all'))
            else:
                o = c.split('-')[1]
                v = late[('p50', s, 'all')] - late[(o, s, 'all')] if ('p50', s, 'all') in late and (o, s, 'all') in late else None
            if v is not None:
                a.plot(i + (s - 2.5) * 0.06, np.nanmean(v), 'o', color='tab:orange' if s >= 3 else 'k', ms=4)
        k = f'6|{c}|all'
        if k in multi:
            st = multi[k]
            a.errorbar(i + 0.3, st['mean'], yerr=[[st['mean'] - st['t']['ci95'][0]], [st['t']['ci95'][1] - st['mean']]],
                       fmt='s', color='tab:green', capsize=3)
    a.axhline(0, color='k', lw=0.6)
    a.set_xticks(range(len(items)))
    a.set_xticklabels(items, fontsize=8)
    a.set_title('D_T,late per pair (black 1-3, orange 4-6); green = 6-pair mean +- t CI', fontsize=9)
    fig.suptitle('round 28 Sec.1: T, flow B, adapter + shared LSTM (1/10), half stage-I data; 6 pairs')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(AN, f'fig_round28_s1.{ext}'), dpi=110)


# ---------------------------------------------------------------- Sec.2
def s2():
    os.makedirs(AN, exist_ok=True)
    RM, RN = runs_mix(), runs_nomix()
    pairs = [s for s in PAIRS if s in RN and ('p50', s) in RM]
    per, rho = {}, {}
    dD, dC, lN, lM, cN, cM = {}, {}, {}, {}, {}, {}
    for s in pairs:
        n, m = RN[s], RM[('p50', s)]
        ln, lm = late_of(n, 'all'), late_of(m, 'all')
        if ln is not None and lm is not None:
            lN[s], lM[s], dD[s] = ln, lm, ln - lm
            per[f'nomix|{s}|late'], per[f'mix|{s}|late'], per[f'nomix-mix|{s}|late'] = map(G.boot_ep, (ln, lm, ln - lm))
            for lk in ('red', 'green'):
                a, b = late_of(n, lk), late_of(m, lk)
                per[f'nomix-mix|{s}|late|{lk}'] = G.boot_ep(a - b)
        if 50 in n['ev'] and 50 in m['ev']:
            cN[s], cM[s] = cself(n, 50), cself(m, 50)
            dC[s] = cN[s] - cM[s]
            per[f'nomix|{s}|cself50'], per[f'mix|{s}|cself50'], per[f'nomix-mix|{s}|cself50'] = map(G.boot_ep, (cN[s], cM[s], dC[s]))
            per[f'nomix|{s}|cself0'] = G.boot_ep(cself(n, 0))
        for tag, r in (('nomix', n), ('mix', m)):
            es = sorted(r['ev'])
            x = [float(np.nanmean(r['ev'][e]['c_self'])) for e in es]
            y = [float(np.nanmean(G.d_of(r['ev'][e]))) for e in es]
            rho[f'{tag}|{s}'] = {'rho': float(spearmanr(x, y).correlation) if len(es) >= 3 else None, 'n_eval': len(es),
                                 'c_self': x, 'd_t': y, 'epochs': es}
    multi = {}
    full = [s for s in PAIRS if s in dD]
    if full:
        multi['late|nomix'] = A.stat(np.stack([lN[s] for s in full]))
        multi['late|mix'] = A.stat(np.stack([lM[s] for s in full]))
        multi['late|nomix-mix'] = A.stat(np.stack([dD[s] for s in full]))
    fullc = [s for s in PAIRS if s in dC]
    if fullc:
        multi['cself50|nomix'] = A.stat(np.stack([cN[s] for s in fullc]))
        multi['cself50|mix'] = A.stat(np.stack([cM[s] for s in fullc]))
        multi['cself50|nomix-mix'] = A.stat(np.stack([dC[s] for s in fullc]))
    rsum = {}
    for tag in ('nomix', 'mix'):
        v = [rho[f'{tag}|{s}']['rho'] for s in pairs if rho.get(f'{tag}|{s}', {}).get('rho') is not None]
        if len(v) >= 2:
            rsum[tag] = {**tci(v), 'per_pair': v}
    rd = {}
    if len(full) == 6 and len(fullc) == 6:
        cc, cd = G.cls(multi['cself50|nomix-mix']), G.cls(multi['late|nomix-mix'])
        rd = {'delta_c_self50': cc, 'delta_d_t_late': cd}
        if cc == '食い違う':
            rd['row'], rd['note'] = None, 'ΔC が食い違う（判定しない）'
        elif cc in ('≈0', '>0'):
            rd['row'] = 3
        elif cd == '>0':
            rd['row'] = 1
        elif cd == '≈0':
            rd['row'] = 2
        elif cd == '食い違う':
            rd['row'], rd['note'] = None, 'ΔC <0、ΔD が食い違う（判定しない）'
        else:
            rd['row'], rd['note'] = None, 'ΔC <0 かつ ΔD <0：表に無い形（そのまま書く）'
    json.dump({'per_pair': per, 'multi': multi, 'reading': rd, 'spearman': rho, 'spearman_summary': rsum,
               'pairs': pairs,
               'runs': {f'nomix|{s}': {'done': RN[s]['done'], 'last_epoch': max(RN[s]['ev']),
                                       'epoch_sec_median': float(np.median([x['sec'] for x in RN[s]['log']['epochs']]))
                                       if RN[s]['log']['epochs'] else None} for s in RN}},
              open(os.path.join(AN, 's2_summary.json'), 'w'), indent=1, ensure_ascii=False)
    L = ['# 第28弾 §2 自己の規則との競合（経験50%、段階Iのデータを混ぜない vs 混ぜる）', '',
         '値：平均 [6 組の t の 95% CI] [2段のブートストラップの 95% CI]（組ごとはエピソードのブートストラップ）。'
         '混ぜる形は、組1〜3 が第27弾、組4〜6 が第28弾 §1。', '',
         '## 読み方の表の当てはめ', '', '```', json.dumps(rd, ensure_ascii=False, indent=1), '```', '',
         '## 6 組', '', '| 量 | 混ぜない | 混ぜる | 混ぜない − 混ぜる | 判定（差） |', '|---|---|---|---|---|']
    for q, nm in (('late', '$D_{T,late}$（全部）'), ('cself50', '$C_{self}$（50 エポック目）')):
        if f'{q}|nomix-mix' in multi:
            L.append(f"| {nm} | {G.fmt(multi[f'{q}|nomix'])} | {G.fmt(multi[f'{q}|mix'])} | "
                     f"{G.fmt(multi[f'{q}|nomix-mix'])} | {G.cls(multi[f'{q}|nomix-mix'])} |")
    L += ['', '## 組ごと', '',
          '| 組 | $D_{T,late}$ 混ぜない | 混ぜる | 差 | 差の判定 | 差 赤好き | 差 緑好き | $C_{self}$ 0 エポック目 | 50 混ぜない | 50 混ぜる | 差 | 順位相関 混ぜない | 混ぜる |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for s in pairs:
        g_ = lambda k: f3(per[k]) if k in per else '—'  # noqa
        rr = lambda t: (f"{rho[f'{t}|{s}']['rho']:+.3f}" if rho.get(f'{t}|{s}', {}).get('rho') is not None else '—')  # noqa
        L.append(f"| {s + 1} | {g_(f'nomix|{s}|late')} | {g_(f'mix|{s}|late')} | {g_(f'nomix-mix|{s}|late')} | "
                 f"{G.mv(per[f'nomix-mix|{s}|late']) if f'nomix-mix|{s}|late' in per else '—'} | "
                 f"{g_(f'nomix-mix|{s}|late|red')} | {g_(f'nomix-mix|{s}|late|green')} | {g_(f'nomix|{s}|cself0')} | "
                 f"{g_(f'nomix|{s}|cself50')} | {g_(f'mix|{s}|cself50')} | {g_(f'nomix-mix|{s}|cself50')} | {rr('nomix')} | {rr('mix')} |")
    L += ['', '## 順位相関（$C_{self}(e)$ と $D_T(e)$、11 回の評価、記述）：組の平均 [t CI]', '']
    for tag, v in rsum.items():
        L.append(f"- {CN[tag]}：{v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]（組ごと "
                 + ', '.join(f'{x:+.2f}' for x in v['per_pair']) + '）')
    eps = sorted({e for k, v in rho.items() for e in v['epochs']})
    L += ['', '## 曲線（全部）', '', '| 回 | 量 | ' + ' | '.join(f'ep{e}' for e in eps) + ' |', '|---|---|' + '---|' * len(eps)]
    for s in pairs:
        for tag in ('nomix', 'mix'):
            r = rho.get(f'{tag}|{s}')
            if r:
                dd = dict(zip(r['epochs'], r['d_t']))
                cc_ = dict(zip(r['epochs'], r['c_self']))
                L.append(f'| {CN[tag]} 組{s + 1} | $D_T$ | ' + ' | '.join(f'{dd[e]:+.3f}' if e in dd else '' for e in eps) + ' |')
                L.append(f'| {CN[tag]} 組{s + 1} | $C_{{self}}$ | ' + ' | '.join(f'{cc_[e]:+.3f}' if e in cc_ else '' for e in eps) + ' |')
    for s in pairs:
        mm = RN[s]['ev']
        L.append(f'| {CN["nomix"]} 組{s + 1} | L1 V | ' + ' | '.join(f"{float(mm[e]['l1_T_V']):.4f}" if e in mm else '' for e in eps) + ' |')
    open(os.path.join(AN, 's2_tables.md'), 'w').write('\n'.join(L) + '\n')
    fig_s2(rho, pairs)
    print(json.dumps(rd, ensure_ascii=False, indent=1))


def fig_s2(rho, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm = plt.get_cmap('tab10')
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for s in pairs:
        for tag, ls in (('nomix', '-'), ('mix', '--')):
            r = rho.get(f'{tag}|{s}')
            if not r:
                continue
            lab = f'pair {s + 1} {tag}'
            ax[0].plot(r['epochs'], r['d_t'], ls, color=cm(s), marker='.', label=lab)
            ax[1].plot(r['epochs'], r['c_self'], ls, color=cm(s), marker='.', label=lab)
            ax[2].scatter(r['c_self'], r['d_t'], color=cm(s), marker='o' if tag == 'nomix' else 'x', s=18,
                          label=lab)
    for a, t in zip(ax, ('D_T(e), all (solid = no mix, dashed = mix)', 'C_self(e) (solid = no mix, dashed = mix)',
                         'C_self vs D_T per evaluation (o = no mix, x = mix)')):
        a.axhline(0, color='k', lw=0.6)
        a.set_title(t, fontsize=9)
    ax[0].set_xlabel('epoch')
    ax[1].set_xlabel('epoch')
    ax[2].set_xlabel('C_self')
    ax[2].set_ylabel('D_T')
    ax[1].legend(fontsize=6, ncol=2)
    fig.suptitle('round 28 Sec.2: p50, stage-I data not mixed vs mixed (T, flow B, adapter + shared LSTM 1/10)')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(AN, f'fig_round28_s2.{ext}'), dpi=110)


if __name__ == '__main__':
    md = sys.argv[1]
    if md in ('s1', 'all'):
        s1()
    if md in ('s2', 'all'):
        s2()
