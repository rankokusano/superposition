"""
Paper round 11 (docs/paper_extra_log.md Sec.35.12; outside the pre-registration, procedure committed before computing
in e8733b5): per-episode and per-frame analysis of the saved E2 arrays, and the 50% matching BA. CPU only, saved
arrays only (no model is run).
  R : 50% matching BA (fixed-shape candidates c+ / c-, v14 definitions): self-vision L1 and the L1-stage FP copied
      from v14/inverse/aggregate_v12_s0to5.json; mse late-5 FP computed from v14/inverse/v12mse_mixed_<s> (pairs
      1-3, 5, 6; pair 4 not saved) with aggregate_inverse_v14.correct / ba (T = 50).
  A : per episode and point U = mean_W FP_opp - mean_W FP_true; choice correct if U > 0 (0.5 on a tie); BA = mean of
      the layout x A-2-liked cell accuracies; positive share; subsets (liked red / green, layout 0 / 1, preference
      vs direction agree / disagree (88)); per point -> mean of 5 points -> 6-pair t (across_pairs_v13.test).
      Episode-value (5-point mean) percentiles per pair.
  B : d_e = L(green form) - L(red form); bias = mean over all episodes; discrimination = (mean over red-liked -
      mean over green-liked) / 2; per point -> 5-point mean -> 6-pair t.
  C : per-frame U_f = FP_opp[t] - FP_true[t] (frame f = t + 1), cumulative-window U(1..T), red / green split, bias /
      discrimination per frame; 6-pair means and t CI.
  checks: U pair values = E2 summary 'alone'; correct_sign identical across the v14 files used; 88 disagreeing
      episodes; this BA = v14 reference BA (ref_red / ref_green, correct_ref) for 50% pairs 1-3, 5, 6.
Main: preference-mixed set, FP, frames 1-50; secondary 1-100.
-> data/result/paper_extra/E2/episode/summary.json, tables.md, frames.npz
"""

import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_PI, 'analyze', 'v14'), os.path.join(_PI, 'analyze', 'v13'), os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)
from across_pairs_v13 import test as ttest  # noqa
from aggregate_inverse_v14 import ba, cells, correct, correct_ref  # noqa
from eval_s2_v11 import load_set  # noqa
from v11core.pleasure import BIN_UNIT  # noqa

RES = os.path.join(_PI, 'data', 'result')
E2 = os.path.join(RES, 'paper_extra', 'E2')
RUNS = os.path.join(E2, 'runs')
INV = os.path.join(RES, 'v14', 'inverse')
OUT = os.path.join(E2, 'episode')
FAMS = ('none', 'p50', 'p80', 'sr80')
NAME = {'none': '経験なし', 'p50': '経験50%', 'p80': '経験80%', 'sr80': '符号無作為（経験80%）'}
SEEDS = range(6)
EPS = (160, 170, 180, 190, 200)
WINS = {'1-50': 50, '1-100': 100}


def fp(f, s, e):
    z = np.load(os.path.join(RUNS, f'{f}_mse_s{s}_ep{e}_pref.npz'))
    return {k: z[k] for k in z.files}


def answers():
    """correct_sign (identical over the v14 files) and the direction answer (fp_direction_v14 definition)."""
    fs = sorted(glob.glob(os.path.join(INV, '*_ep*_pref.npz')))
    cs = np.load(fs[0])['correct_sign']
    same = all(np.array_equal(np.load(f)['correct_sign'], cs) for f in fs)
    d = load_set('pref')
    disp = d['other_position'][:, 51] - d['other_position'][:, 1]
    u01, u67 = BIN_UNIT[[0, 1]].mean(0), BIN_UNIT[[6, 7]].mean(0)
    u01, u67 = u01 / np.linalg.norm(u01), u67 / np.linalg.norm(u67)
    dirn = np.where(disp @ u67 > disp @ u01, 1, -1)
    return cs, dirn, same, len(fs), d['layout'], d['a2_liked']


def ucorr(U):
    return np.where(U > 0, 1.0, np.where(U < 0, 0.0, 0.5))


def acc_ba(c, cl, idx):
    return {'acc': float(c[idx].mean()), 'ba': ba(c, cl, idx)}


def per_point(z, n, groups):
    U = z['FP_opp'][:, :n].mean(1) - z['FP_true'][:, :n].mean(1)
    c, cl = ucorr(U), cells(z)
    r = {'U_mean': float(U.mean()), 'pos_share': float((U > 0).mean()), 'tie_share': float((U == 0).mean()),
         'BA': ba(c, cl)}
    for g, idx in groups.items():
        r[g] = acc_ba(c, cl, idx)
    red = z['a2_liked'] == 1
    de = np.where(red, U, -U)
    r['bias'] = float(de.mean())
    r['disc'] = float((de[red].mean() - de[~red].mean()) / 2)
    return r, U


def tblock(vals):
    return {'per_pair': vals, 't': ttest(vals)}


def main():
    os.makedirs(OUT, exist_ok=True)
    cs, dirn, cs_same, n_files, layout, liked = answers()
    agree = cs == dirn
    groups = {'liked_red': np.where(liked == 1)[0], 'liked_green': np.where(liked == 0)[0],
              'layout0': np.where(layout == 0)[0], 'layout1': np.where(layout == 1)[0],
              'pref_dir_agree': np.where(agree)[0], 'pref_dir_disagree': np.where(~agree)[0]}
    R = {'checks': {'correct_sign_identical': cs_same, 'correct_sign_files': n_files,
                    'n_disagree': int((~agree).sum()), 'n_red_liked': int((liked == 1).sum())},
         'A': {}, 'dist': {}}
    S2 = json.load(open(os.path.join(E2, 'summary.json')))
    frames = {}
    chk_u = []
    for w, n in WINS.items():
        for f in FAMS:
            pts = {s: [] for s in SEEDS}
            epv = {}
            for s in SEEDS:
                Us = []
                for e in EPS:
                    z = fp(f, s, e)
                    assert np.array_equal(z['layout'], layout) and np.array_equal(z['a2_liked'], liked)
                    r, U = per_point(z, n, groups)
                    pts[s].append(r)
                    Us.append(U)
                epv[s] = np.mean(Us, 0)
                uv = float(np.mean([p['U_mean'] for p in pts[s]]))
                chk_u.append(abs(uv - S2['alone'][f'pref|FP|{w}'][f]['pref'][s]))
            keys = ['U_mean', 'pos_share', 'tie_share', 'BA', 'bias', 'disc'] + \
                   [f'{g}|{m}' for g in groups for m in ('acc', 'ba')]
            blk = {}
            for k in keys:
                if '|' in k:
                    g, m = k.split('|')
                    v = [float(np.mean([p[g][m] for p in pts[s]])) for s in SEEDS]
                else:
                    v = [float(np.mean([p[k] for p in pts[s]])) for s in SEEDS]
                blk[k] = tblock(v)
            R['A'][f'{f}|{w}'] = blk
            R['dist'][f'{f}|{w}'] = {f's{s}': {'pct': [float(x) for x in np.percentile(epv[s], [5, 25, 50, 75, 95])],
                                                'pos_share_epmean': float((epv[s] > 0).mean())} for s in SEEDS}
    R['checks']['U_vs_E2_summary_max_abs'] = float(max(chk_u))
    # C: per frame
    for f in FAMS:
        Uf = np.zeros((6, 100)); red_f = np.zeros((6, 100)); green_f = np.zeros((6, 100))
        bias_f = np.zeros((6, 100)); disc_f = np.zeros((6, 100)); cum = np.zeros((6, 100))
        red = liked == 1
        for s in SEEDS:
            acc = {k: [] for k in ('U', 'r', 'g', 'b', 'd', 'c')}
            for e in EPS:
                z = fp(f, s, e)
                D = z['FP_opp'] - z['FP_true']                 # (N, 100)
                acc['U'].append(D.mean(0))
                acc['r'].append(D[red].mean(0)); acc['g'].append(D[~red].mean(0))
                de = np.where(red[:, None], D, -D)
                acc['b'].append(de.mean(0)); acc['d'].append((de[red].mean(0) - de[~red].mean(0)) / 2)
                cU = np.cumsum(z['FP_opp'], 1) / np.arange(1, 101) - np.cumsum(z['FP_true'], 1) / np.arange(1, 101)
                acc['c'].append(cU.mean(0))
            Uf[s], red_f[s], green_f[s] = np.mean(acc['U'], 0), np.mean(acc['r'], 0), np.mean(acc['g'], 0)
            bias_f[s], disc_f[s], cum[s] = np.mean(acc['b'], 0), np.mean(acc['d'], 0), np.mean(acc['c'], 0)
        frames[f] = {'U': Uf, 'red': red_f, 'green': green_f, 'bias': bias_f, 'disc': disc_f, 'cum': cum}
    np.savez_compressed(os.path.join(OUT, 'frames.npz'), **{f'{f}__{k}': v for f, d in frames.items() for k, v in d.items()})
    C = {}
    for f in FAMS:
        C[f] = {}
        for k in ('U', 'red', 'green', 'bias', 'disc', 'cum'):
            arr = frames[f][k]
            C[f][k] = {'mean': arr.mean(0).tolist(),
                       'ci_lo': [ttest(list(arr[:, t]))['ci95'][0] for t in range(100)],
                       'ci_hi': [ttest(list(arr[:, t]))['ci95'][1] for t in range(100)]}
        for a, b in ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)):
            C[f][f'bin{a}-{b}'] = {k: ttest(list(frames[f][k][:, a - 1:b].mean(1))) for k in ('U', 'red', 'green', 'bias', 'disc')}
        for T in (50, 100):
            C[f][f'cum_check_T{T}'] = float(np.max(np.abs(frames[f]['cum'][:, T - 1] - np.array(S2['alone'][f'pref|FP|1-{T}'][f]['pref']))))
    R['C'] = C
    # R: 50% matching BA
    agg = json.load(open(os.path.join(INV, 'aggregate_v12_s0to5.json')))['detail']
    Rm = {'L1_l1late5': {}, 'FP_l1stage': {}, 'FP_mselate5': {}, 'none_L1_l1late5': {}, 'none_FP_l1stage': {}}
    for s in SEEDS:
        dt = agg[f'v12:mixed:{s},v12:random:{s}']
        for q, k in (('L1', 'pref_L1_T50'), ('FP', 'pref_FP_T50')):
            p = dt[k]['points']
            nm = 'L1_l1late5' if q == 'L1' else 'FP_l1stage'
            Rm[nm][f's{s}'] = [p[str(e)]['ba_exp'] for e in EPS]
            Rm[f'none_{nm}'][f's{s}'] = [p[str(e)]['ba_noexp'] for e in EPS]
    ref_chk = {}
    for s in SEEDS:
        fs = [os.path.join(INV, f'v12mse_mixed_{s}_ep{e}_pref.npz') for e in EPS]
        if not all(os.path.exists(x) for x in fs):
            Rm['FP_mselate5'][f's{s}'] = None
            continue
        zs = [np.load(x) for x in fs]
        Rm['FP_mselate5'][f's{s}'] = [ba(correct(z, 'FP', 50), cells(z)) for z in zs]
        mine = [p['BA'] for p in [per_point(fp('p50', s, e), 50, groups)[0] for e in EPS]]
        theirs = [ba(correct_ref(z, 'FP', 50), cells(z)) for z in zs]
        ref_chk[f's{s}'] = {'equal': mine == theirs, 'max_abs': float(np.max(np.abs(np.array(mine) - np.array(theirs))))}
    R['checks']['BA_equals_v14_ref'] = ref_chk
    for k, v in list(Rm.items()):
        vals = [float(np.mean(x)) for x in v.values() if x is not None]
        Rm[k] = {'per_pair_points': v, 'per_pair_mean': {s: (float(np.mean(x)) if x is not None else None) for s, x in v.items()},
                 't': ttest(vals) if len(vals) > 1 else None, 'n_pairs': len(vals)}
    R['R'] = Rm
    with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)
    T = tables(R)
    with open(os.path.join(OUT, 'tables.md'), 'w') as fh:
        fh.write(T)
    print(T)


def f4(t):
    return f"{t['mean']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]"


def f3(t):
    return f"{t['mean']:.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]"


def tables(R):
    L = ['# E2 のエピソードごと・フレームごとの分析と経験50% の照合の BA（作業記録 §35.12、事前登録外）', '']
    L += ['## 表R 経験50% の照合の BA（固定の形の候補 c+・c−、好み混合、T = 50。組ごとの最終5点の平均）', '',
          '| 基準 | 組1 | 組2 | 組3 | 組4 | 組5 | 組6 | 平均 [95% CI] |', '|---|---|---|---|---|---|---|---|']
    lab = {'L1_l1late5': '自己視界（L1 期間の最終5点）', 'FP_l1stage': '他者特徴（L1 期間の点 = MSE 期間 ep200）',
           'FP_mselate5': '他者特徴（MSE 期間の最終5点）', 'none_L1_l1late5': '経験なし：自己視界', 'none_FP_l1stage': '経験なし：他者特徴（ep200）'}
    for k, v in R['R'].items():
        cells_ = [(f"{x:.3f}" if x is not None else '不明（保存なし）') for x in v['per_pair_mean'].values()]
        tt = f"{f3(v['t'])}（{v['n_pairs']} 組）" if v['t'] else '—'
        L.append(f"| {lab[k]} | " + ' | '.join(cells_) + f" | {tt} |")
    for w in WINS:
        L += ['', f'## 表A エピソードごとの二択（好み混合・FP・フレーム {w}。U_pref > 0 を「正しい好みを選んだ」とする。6 組の平均 [95% CI]）', '',
              '| 量 | ' + ' | '.join(NAME[f] for f in FAMS) + ' |', '|---|' + '---|' * len(FAMS)]
        rows = [('U_mean', 'U_pref の平均', f4), ('pos_share', 'U_pref > 0 の割合', f3), ('BA', 'BA（4 セル）', f3),
                ('liked_red|acc', 'A-2 赤好き：正解率', f3), ('liked_red|ba', 'A-2 赤好き：BA（配置 2 セル）', f3),
                ('liked_green|acc', 'A-2 緑好き：正解率', f3), ('liked_green|ba', 'A-2 緑好き：BA（配置 2 セル）', f3),
                ('layout0|ba', '配置0：BA（好み 2 セル）', f3), ('layout1|ba', '配置1：BA（好み 2 セル）', f3),
                ('pref_dir_agree|acc', '好みと方角が一致（212）：正解率', f3), ('pref_dir_agree|ba', '一致：BA（あるセル）', f3),
                ('pref_dir_disagree|acc', '食い違う（88）：正解率', f3), ('pref_dir_disagree|ba', '食い違う：BA（あるセル）', f3),
                ('bias', '偏り（全エピソードの d_e の平均）', f4), ('disc', '見分ける成分（(赤 − 緑)/2）', f4)]
        for k, nm, fm in rows:
            L.append(f'| {nm} | ' + ' | '.join(fm(R['A'][f'{f}|{w}'][k]['t']) for f in FAMS) + ' |')
        L += ['', f'### 組ごとの値（フレーム {w}）', '', '| 条件 | 量 | 組1 | 組2 | 組3 | 組4 | 組5 | 組6 |', '|---|---|---|---|---|---|---|---|']
        for f in FAMS:
            for k, nm in (('BA', 'BA'), ('pos_share', '正の割合'), ('pref_dir_disagree|acc', '食い違う 88 の正解率'),
                          ('bias', '偏り'), ('disc', '見分ける成分')):
                v = R['A'][f'{f}|{w}'][k]['per_pair']
                L.append(f'| {NAME[f]} | {nm} | ' + ' | '.join(f'{x:+.4f}' if k in ('bias', 'disc') else f'{x:.3f}' for x in v) + ' |')
        L += ['', f'### エピソードの値（5 点の平均）の分布（フレーム {w}。パーセンタイル 5 / 25 / 50 / 75 / 95）', '',
              '| 条件 | 組 | 5% | 25% | 50% | 75% | 95% | 正の割合 |', '|---|---|---|---|---|---|---|---|']
        for f in FAMS:
            for s in SEEDS:
                d = R['dist'][f'{f}|{w}'][f's{s}']
                L.append(f'| {NAME[f]} | {s + 1} | ' + ' | '.join(f'{x:+.4f}' for x in d['pct']) + f" | {d['pos_share_epmean']:.3f} |")
    L += ['', '## 表C フレームごとの U_pref（好み混合・FP。10 フレームごとの平均、6 組の平均 [95% CI]）', '',
          '| フレーム | ' + ' | '.join(NAME[f] for f in FAMS) + ' |', '|---|' + '---|' * len(FAMS)]
    for b in [k for k in R['C']['none'] if k.startswith('bin')]:
        L.append(f'| {b[3:]} | ' + ' | '.join(f4(R['C'][f][b]['U']) for f in FAMS) + ' |')
    for q, nm in (('red', 'A-2 赤好き'), ('green', 'A-2 緑好き'), ('bias', '偏り'), ('disc', '見分ける成分')):
        L += ['', f'### {nm}（10 フレームごと）', '', '| フレーム | ' + ' | '.join(NAME[f] for f in FAMS) + ' |', '|---|' + '---|' * len(FAMS)]
        for b in [k for k in R['C']['none'] if k.startswith('bin')]:
            L.append(f'| {b[3:]} | ' + ' | '.join(f4(R['C'][f][b][q]) for f in FAMS) + ' |')
    L += ['', '## 確認', '', '```', json.dumps(R['checks'], ensure_ascii=False),
          json.dumps({f: {k: v for k, v in R['C'][f].items() if k.startswith('cum_check')} for f in FAMS}), '```', '']
    return '\n'.join(L)


if __name__ == '__main__':
    main()
