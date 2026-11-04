"""
Paper round 19 Sec.2 (paper/check/paper_instructions_round19.md; docs/paper_extra_log.md Sec.43; exploratory, outside the
pre-registration; procedure committed before computing). Saved arrays only, CPU (nice 10, OMP_NUM_THREADS=2), no GPU,
no training, no new evaluation.
  --s21  Imposition cue (candidate B): stage-B c+- (Vdir_B/eval/inverse, FP, MSE-stage late 5 points), conditions
         none / p50 / sr50, pairs 1-6. Episodes: the 112 with no direction (|p_1-50| < 1.0, the ones dropped from the
         stage-B main regression); reference: all 1200. Q_e(W) = mean_{frames W} L(c- pleasant) - L(c+ pleasant) for
         W = frames 1-10, 1-25, 1-50 (steps a-1..b-1). Chosen side = sign(Q_e) (+1 = side A, tie 0.5); red-side share =
         share of episodes with chosen side = s_red. By A-2's liked colour (red / green) and their mean ("bias"); pair
         value = 5-point mean; 6-pair t 95% CI (df 5); comparisons p50 - none, p50 - sr50.
  --s22  Red / green from the other's viewpoint in the reference (frozen v4) Encoder-2 output: the saved stage-C (O)
         (StageC/eval, other_vision_encoder_module output before masking, identical for all v12 models; per episode the
         mean over steps 1-50, the mean over steps 51-99, step 99) on the stage-B set (1200). Read-outs with the v12
         Sec.19 functions: layout by logistic regression (enc_colour_v12.acc_cv: standardised, C = 1) and the unit
         vector from A-2 to the red / green corner reference point (vdir_stageA.LANDMARKS of the layout; averaged over
         the same steps as (O)) by Ridge(1) (fig5_extra_v11.cv_r2, pooled cos/sin R^2); episode GroupKFold(5).
         References: A-2 position (same-step mean) poly-2, A-2 position poly-2 x layout, (O) + A-2 position poly-2.
-> data/result/paper_extra/Round19/{s21.json, s22.json, tables.md}
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import vdirb_analysis as A  # noqa
V = A.V
for p in (os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11')):
    sys.path.insert(0, p)
OUT = os.path.join(V.RES, 'paper_extra', 'Round19')
CONDS = ('none', 'p50', 'sr50')
DIFFS = (('p50', 'none'), ('p50', 'sr50'))
WINS = {'1-10': (1, 10), '1-25': (1, 25), '1-50': (1, 50)}
LIKED = (('red', 1), ('green', 0))
SC_DIR = os.path.join(V.RES, 'paper_extra', 'StageC', 'eval')


def load_L(cond, s, e):
    pre = A.TAGS[cond][0]
    z = np.load(os.path.join(A.EV, f'{pre}_{s}_ep{e}_vdirb.npz'))
    return z['FP_cp'].astype(np.float64), z['FP_cm'].astype(np.float64), z


def s21():
    m = A.meta()
    keep1 = (np.abs(m['p']['1-50']) >= 1.0) & (m['p']['1-50'] != 0)
    nodir = ~keep1
    masks = {'nodir': nodir, 'all': np.ones_like(nodir)}
    R = {'n': {mk: {nm: int((mv & (m['liked'] == v)).sum()) for nm, v in LIKED} for mk, mv in masks.items()},
         'n_nodir_by_layout': {f'{nm}|lay{L}': int((nodir & (m['liked'] == v) & (m['layout'] == L)).sum())
                               for nm, v in LIKED for L in (0, 1)}}
    pv = {c: {} for c in CONDS}
    for c in CONDS:
        vals = {}
        for s in A.PAIRS[c]:
            per = {}
            for e in V.EPS:
                LA, LB, z = load_L(c, s, e)
                assert np.array_equal(z['correct_sign'], m['s_pref']) and np.array_equal(z['layout'], m['layout'])
                for w, (a, b) in WINS.items():
                    q = LB[:, a - 1:b].mean(1) - LA[:, a - 1:b].mean(1)
                    hit = np.where(q == 0, 0.5, (np.sign(q) == m['s_red']).astype(float))
                    for mk, mv in masks.items():
                        for nm, v in LIKED:
                            per.setdefault(f'{mk}|{w}|{nm}', []).append(hit[mv & (m['liked'] == v)].mean())
            for k, x in per.items():
                vals.setdefault(k, []).append(float(np.mean(x)))
        for k, x in vals.items():
            pv[c][k] = np.array(x)
        for mk in masks:
            for w in WINS:
                pv[c][f'{mk}|{w}|bias'] = (pv[c][f'{mk}|{w}|red'] + pv[c][f'{mk}|{w}|green']) / 2
    R['values'] = {}
    for c in CONDS:
        R['values'][c] = {k: {'per_pair': x.tolist(), 't': A.tsum(x)} for k, x in pv[c].items()}
    for a, b in DIFFS:
        R['values'][f'{a}-{b}'] = {k: {'per_pair': (pv[a][k] - pv[b][k]).tolist(), 't': A.tsum(pv[a][k] - pv[b][k])}
                                  for k in pv[a]}
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, 's21.json'), 'w'), ensure_ascii=False, indent=1)
    return R


def s22():
    from sklearn.preprocessing import PolynomialFeatures
    from enc_colour_v12 import acc_cv  # noqa
    from fig5_extra_v11 import cv_r2  # noqa
    m = A.meta()
    import h5py
    g = h5py.File(V.load_set('vdirb')['path'], 'r')['test']
    P2 = g['other_position'][()].astype(np.float64)                   # (N, 101, 2)
    N = len(P2)
    grp = np.arange(N)
    f0 = os.path.join(SC_DIR, 'v12_s2_pref_mixed_s0_ep200.npz')
    O0 = np.load(f0)
    R = {'source': os.path.basename(f0), 'identical_across_models': {}}
    for other in ('v12_s2_pref_random_s3_ep160.npz', 'v12_s2_pref_mixed_s2_ep0.npz', 'v12_s2_green_mixed_s1_ep180.npz'):
        z = np.load(os.path.join(SC_DIR, other))
        R['identical_across_models'][other] = all(np.array_equal(O0[k], z[k]) for k in ('O_m1_50', 'O_m51_99', 'O_last'))
    steps = {'m1_50': np.arange(1, 51), 'm51_99': np.arange(51, 100), 'last': np.array([99])}
    lay = m['layout'].astype(int)
    R['readout'] = {}
    for sk, st in steps.items():
        O = O0[f'O_{sk}'].astype(np.float64)
        pos = P2[:, st].mean(1)
        poly = PolynomialFeatures(2, include_bias=False).fit_transform(pos)
        r = {'layout_from_O': acc_cv(O, lay, grp), 'layout_from_A2pos_poly2': acc_cv(poly, lay, grp)}
        for c in ('Red', 'Green'):
            ref = np.array([V.LANDMARKS[V.CODE_LAYOUT[int(L)]][c] for L in lay])[:, None]
            v = ref - P2[:, st]
            u = (v / np.linalg.norm(v, axis=-1, keepdims=True)).mean(1)
            r[f'{c}_dir_from_O'] = cv_r2(O, u, grp)
            r[f'{c}_dir_from_A2pos_poly2'] = cv_r2(poly, u, grp)
            r[f'{c}_dir_from_A2pos_x_layout'] = cv_r2(np.concatenate([poly, poly * lay[:, None]], 1), u, grp)
            r[f'{c}_dir_from_O_plus_A2pos'] = cv_r2(np.concatenate([O, poly], 1), u, grp)
        R['readout'][sk] = r
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, 's22.json'), 'w'), ensure_ascii=False, indent=1)
    return R


def ci(t, d=3):
    return f'{t["mean"]:.{d}f} [{t["ci95"][0]:.{d}f}, {t["ci95"][1]:.{d}f}]'


def tables(R1, R2):
    cols = list(CONDS) + [f'{a}-{b}' for a, b in DIFFS]
    L = ['# 第19弾 §2（探索・事前登録外）', '', '## §2.1 誤差の小さい候補が赤の側だった割合（6組の平均 [t の 95% CI]）', '',
         f'エピソード数：向きなし {R1["n"]["nodir"]}、全部 {R1["n"]["all"]}。向きなしの配置別 {R1["n_nodir_by_layout"]}', '',
         '| エピソード | 区間 | A-2 の好き | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 3)]
    for mk in ('nodir', 'all'):
        for w in WINS:
            for nm in ('red', 'green', 'bias'):
                k = f'{mk}|{w}|{nm}'
                L.append(f'| {mk} | {w} | {nm} | ' + ' | '.join(ci(R1['values'][c][k]['t']) for c in cols) + ' |')
    L += ['', '## §2.2 保存した (O)（凍結した v4 の Encoder-2 の出力）からの読み出し（段階Bのデータ 1200、エピソードの GroupKFold(5)）', '',
          f'(O) がモデルによらず同じ：{R2["identical_across_models"]}', '', '| 量 | ステップ 1〜50 の平均 | 51〜99 の平均 | ステップ 99 |', '|---|---|---|---|']
    for k in R2['readout']['m1_50']:
        L.append(f'| {k} | ' + ' | '.join(f'{R2["readout"][s][k]:.3f}' for s in ('m1_50', 'm51_99', 'last')) + ' |')
    txt = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'tables.md'), 'w').write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s21', action='store_true')
    ap.add_argument('--s22', action='store_true')
    a = ap.parse_args()
    R1 = s21() if a.s21 else json.load(open(os.path.join(OUT, 's21.json')))
    R2 = s22() if a.s22 else json.load(open(os.path.join(OUT, 's22.json')))
    tables(R1, R2)


if __name__ == '__main__':
    main()
