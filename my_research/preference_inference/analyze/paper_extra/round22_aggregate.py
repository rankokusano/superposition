"""
Paper round 22 Sec.2 aggregation (docs/paper_extra_log.md Sec.46.2; exploratory, outside the pre-registration; written
and committed before any evaluation output was read). CPU.
Inputs: Round21/eval/<cond>_s<s>_ep<e>.npz (p50 / none / sr50 / p80, round 21's stored results, read as they are),
Round22/eval21/sr80_*.npz (round 21's flow for the sign-random 80% predictor), Round22/eval/<cond>_*.npz
(round22_eval.py: other runs, read-outs img and feat), Round22/check/check.json (input sizes).
Per episode and checkpoint (img = round 21's A-2-position read-out, feat = Sec.2.4 feature read-out, V1 only):
  Delta(d, V, input) = s_Lold - s_Lnew;  C(input) = Delta(noswap) - Delta(swap) (same episode / checkpoint);
  C(X3) = Delta(noswap, b_old) - Delta(swap, b_new)  (+0.28 in the range toward red's current place, as hold);
  HmF(d, base) = Delta(d, <base>hold) - Delta(d, <base>flip), base in orig / x2 / x4 / m;
  Rdir(d, V) = Delta(d, V, b_old) - Delta(d, V, b_new)  (response to the direction of the X3 input);
  shift_L = (est_10(b_L) - est_10(zero)) . unit(L - p2)  (img; last step, as G5b) and fdstar_L,10(b_L) -
            fdstar_L,10(zero) (feat), L in old / new.
Self pathway (C_self, HmF self) from round 21's stored results (colour-agnostic d* read-out), sr80 from eval21.
Episode value = mean over finite checkpoints; pair value = mean over finite episodes; pair t 95% CI (df 5) and
two-stage bootstrap (1000, default_rng(0)) as round21_aggregate. Differences per episode: p80 - sr80, p80 - none
(Sec.2.2), and p50 - none, p50 - sr50 (descriptive).
Reading (Sec.46.2, fixed): main read-out img, conditions p50 and p80; the same rules with feat as a supplement.
-> Round22/analysis/{summary.json, tables.md, fig_round22_C.{pdf,png}}
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round21_aggregate as A  # noqa  (load_ep, stat, excl, SIGN, fmt)
from round22_eval import targets_bins  # noqa

RES = A.RES
EV21 = os.path.join(RES, 'paper_extra', 'Round21', 'eval')
R22 = os.environ.get('ROUND22_DIR', os.path.join(RES, 'paper_extra', 'Round22'))  # override only for the synthetic test
OUT = os.path.join(R22, 'analysis')
CONDS = ('p50', 'none', 'sr50', 'p80', 'sr80')
XC = ('p50', 'p80')
CNAME = {'p50': '経験50%', 'none': '経験なし', 'sr50': '符号無作為（経験50%）', 'p80': '経験80%',
         'sr80': '符号無作為（経験80%）'}
DIFFS = (('p80', 'sr80'), ('p80', 'none'), ('p50', 'none'), ('p50', 'sr50'))
PAIRS = tuple(range(6))
EPS = (160, 170, 180, 190, 200)
ORIG = ('hold', 'flip', 'zero')
XIN = ('x2_hold', 'x2_flip', 'x4_hold', 'x4_flip', 'm_hold', 'm_flip', 'b_old', 'b_new')
BASES = (('orig', 'hold', 'flip'), ('x2', 'x2_hold', 'x2_flip'), ('x4', 'x4_hold', 'x4_flip'), ('m', 'm_hold', 'm_flip'))
CIN = (('hold', '元の入力（保つ）'), ('x2_hold', 'X1 2倍（保つ）'), ('x4_hold', 'X1 4倍（保つ）'),
       ('m_hold', 'X2 自己とそろえる（保つ）'), ('X3', 'X3 +0.28（赤の今の方角）'))
RO = ('img', 'feat')


def load_r22(c, s):
    lo, ln, p2, _, _ = targets_bins()
    u = {}
    for nm, tg in (('old', lo), ('new', ln)):
        v = tg - p2
        u[nm] = v / np.linalg.norm(v, axis=1, keepdims=True)
    inputs = ORIG + (XIN if c in XC else ())
    acc, frac = {}, {}
    for e in EPS:
        z = np.load(os.path.join(R22, 'eval', f'{c}_s{s}_ep{e}.npz'))
        for r in RO:
            D = {(d, inp): z[f'{d}_V1_{inp}_{r}'].astype(np.float64) for d in ('swap', 'noswap') for inp in inputs}
            for (d, inp), x in D.items():
                acc.setdefault(f'Delta|{d}|V1|{inp}|{r}', []).append(x)
                frac.setdefault(f'read|{d}|V1|{inp}|{r}', []).append(np.isfinite(x).mean())
            for inp in inputs:
                if inp not in ('b_old', 'b_new'):
                    acc.setdefault(f'C|V1|{inp}|{r}', []).append(D[('noswap', inp)] - D[('swap', inp)])
            for base, h, f in BASES:
                if h in inputs:
                    for d in ('swap', 'noswap'):
                        acc.setdefault(f'HmF|{d}|V1|{base}|{r}', []).append(D[(d, h)] - D[(d, f)])
            if c in XC:
                acc.setdefault(f'C|V1|X3|{r}', []).append(D[('noswap', 'b_old')] - D[('swap', 'b_new')])
                for d in ('swap', 'noswap'):
                    acc.setdefault(f'Rdir|{d}|V1|{r}', []).append(D[(d, 'b_old')] - D[(d, 'b_new')])
                    for L in ('old', 'new'):
                        if r == 'img':
                            a, b = z[f'{d}_V1_b_{L}_est'][:, -1].astype(np.float64), z[f'{d}_V1_zero_est'][:, -1].astype(np.float64)
                            acc.setdefault(f'shift|{d}|V1|{L}|img', []).append(((a - b) * u[L]).sum(-1))
                        else:
                            i = 0 if L == 'old' else 1
                            a = z[f'{d}_V1_b_{L}_fdstar'][:, i, -1].astype(np.float64)
                            b = z[f'{d}_V1_zero_fdstar'][:, i, -1].astype(np.float64)
                            acc.setdefault(f'shift|{d}|V1|{L}|feat', []).append(a - b)
        if c in XC:
            x = {inp: z[f'noswap_V0_{inp}_img'].astype(np.float64) for inp in ('b_old', 'b_new', 'zero')}
            acc.setdefault('Rdir|noswap|V0|img', []).append(x['b_old'] - x['b_new'])
            for inp in x:
                frac.setdefault(f'read|noswap|V0|{inp}|img', []).append(np.isfinite(x[inp]).mean())
            for L in ('old', 'new'):
                a, b = z[f'noswap_V0_b_{L}_est'][:, -1].astype(np.float64), z['noswap_V0_zero_est'][:, -1].astype(np.float64)
                acc.setdefault(f'shift|noswap|V0|{L}|img', []).append(((a - b) * u[L]).sum(-1))
        rp = [float(z[k]) for k in z.files if k.startswith('repro_') and not k.startswith('repro_nan')]
        acc.setdefault('repro_max', []).append(np.full(600, max(rp) if rp else 0.0))
    with np.errstate(all='ignore'):
        ev = {k: np.nanmean(np.stack(v), 0) for k, v in acc.items()}
    return ev, {k: float(np.mean(v)) for k, v in frac.items()}


def load_self(c, s):
    A.EV = os.path.join(R22, 'eval21') if c == 'sr80' else EV21
    ev, fr, _ = A.load_ep(c, s)
    keep = {}
    for k, v in ev.items():
        if '|V1|' in k and ('|self|' in k or '|other|' in k or k.endswith('|self') or k.endswith('|other')):
            keep['r21|' + k] = v
    return keep, {'r21|' + k: v for k, v in fr.items()}


def reading(S, r):
    """Sec.46.2 rules with read-out r (img = main, feat = supplement)."""
    def ex(q):
        return A.excl(S[q])
    out = {}
    pos, hf, zero_all = {}, {}, {}
    for c in XC:
        hs = ex(f'{c}|r21|HmF|noswap|V1|self')
        ho = ex(f'{c}|HmF|noswap|V1|orig|{r}')
        hf[c] = {'self': A.SIGN[hs], 'other': A.SIGN[ho], 'same_direction_nonzero': hs in (1, -1) and ho == hs}
        pos[c] = {inp: ex(f'{c}|C|V1|{inp}|{r}') == 1 for inp, _ in CIN}
        zero_all[c] = all(ex(f'{c}|HmF|noswap|V1|{b}|{r}') == 0 for b, _, _ in BASES) and \
            all(ex(f'{c}|C|V1|{inp}|{r}') == 0 for inp, _ in CIN)
    out['hold_minus_flip_noswap'] = hf
    out['C_other_sign'] = {c: {inp: A.SIGN[ex(f'{c}|C|V1|{inp}|{r}')] for inp, _ in CIN} for c in XC}
    rows = {}
    for c in XC:
        rows[f'row1_{c}'] = hf[c]['same_direction_nonzero'] and any(pos[c].values())
    rows['row2_p80_only'] = any(pos['p80'].values()) and not any(pos['p50'].values())
    rows['row3_all_zero'] = all(zero_all.values())
    for c in XC:
        p = pos[c]
        rows[f'row4_{c}'] = (p['x2_hold'] or p['x4_hold']) and not (p['hold'] or p['m_hold'] or p['X3'])
    out['rows'] = rows
    out['no_row'] = not any(rows.values())
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    EVV, FR = {}, {}
    for c in CONDS:
        for s in PAIRS:
            a, fa = load_r22(c, s)
            b, fb = load_self(c, s)
            a.update(b)
            fa.update(fb)
            EVV[(c, s)], FR[(c, s)] = a, fa
    S = {}
    for c in CONDS:
        for k in EVV[(c, 0)]:
            S[f'{c}|{k}'] = A.stat(np.stack([EVV[(c, s)][k] for s in PAIRS]))
    for a, b in DIFFS:
        for k in EVV[(b, 0)]:
            if k in EVV[(a, 0)] and k != 'repro_max':
                S[f'{a}-{b}|{k}'] = A.stat(np.stack([EVV[(a, s)][k] - EVV[(b, s)][k] for s in PAIRS]))
    FRm = {c: {k: float(np.mean([FR[(c, s)][k] for s in PAIRS])) for k in FR[(c, 0)]} for c in CONDS}
    RD = {r: reading(S, r) for r in RO}
    chk = json.load(open(os.path.join(R22, 'check', 'check.json')))
    json.dump({'stats': S, 'readable': FRm, 'reading': RD, 'check': chk}, open(os.path.join(OUT, 'summary.json'), 'w'),
              indent=1, ensure_ascii=False)
    f = A.fmt
    Lt = ['# 第22弾 §2（探索・事前登録外）', '',
          '値：組の値の平均 [組の t の 95% CI] [2段のブートストラップの 95% CI]、6 組。Δ = $L_{old}$ への傾き − $L_{new}$ への傾き。'
          '$C$ = Δ(入れ替えなし) − Δ(入れ替えあり)。img = 第21弾の読み出し（予測画像の A-2 の位置）、feat = 特徴の読み出し（§2.4）。'
          '自己は第21弾の保存した結果（色を問わない読み出し）。V1。', '',
          f"- 再現：元の入力の img の Δ と第21弾の保存値の差の最大 = "
          f"{max(S[f'{c}|repro_max']['mean'] for c in ('p50', 'none', 'sr50', 'p80')):.3g}", '']
    Lt += ['## 2.1・2.2 「保つ − 反転」の Δ（元の入力）', '',
           '| 条件 | 入れ替え | 自己 | 他者 img | 他者 feat |', '|---|---|---|---|---|']
    for c in CONDS:
        for d in ('noswap', 'swap'):
            Lt.append(f'| {CNAME[c]} | {d} | ' + ' | '.join([f(S[f'{c}|r21|HmF|{d}|V1|self'])] +
                                                           [f(S[f'{c}|HmF|{d}|V1|orig|{r}']) for r in RO]) + ' |')
    Lt += ['', '## 2.2 $C$（元の入力・保つ）と差', '', '| 条件・差 | $C_{self}$ | $C_{other}$ img | $C_{other}$ feat | 他者の保つ − 反転（noswap） img |',
           '|---|---|---|---|---|']
    for c in CONDS + tuple(f'{a}-{b}' for a, b in DIFFS):
        nm = CNAME.get(c, c)
        Lt.append(f'| {nm} | ' + ' | '.join([f(S[f'{c}|r21|C|V1|self|hold']), f(S[f'{c}|C|V1|hold|img']),
                                             f(S[f'{c}|C|V1|hold|feat']), f(S[f'{c}|HmF|noswap|V1|orig|img'])]) + ' |')
    Lt += ['', '- 経験80%・符号無作為（経験80%）は、自己位置の追跡の前提を満たさない。', '']
    for r in RO:
        Lt += [f'## 2.3 入力ごとの $C_{{other}}$（{r}）', '', '| 入力 | ' + ' | '.join(CNAME[c] for c in XC) + ' |',
               '|---|' + '---|' * len(XC)]
        for inp, nm in CIN:
            Lt.append(f'| {nm} | ' + ' | '.join(f(S[f'{c}|C|V1|{inp}|{r}']) for c in XC) + ' |')
        for inp, nm in (('flip', '元の入力（反転）'), ('x2_flip', 'X1 2倍（反転）'), ('x4_flip', 'X1 4倍（反転）'), ('m_flip', 'X2（反転）'),
                        ('zero', '零')):
            Lt.append(f'| {nm} | ' + ' | '.join(f(S[f'{c}|C|V1|{inp}|{r}']) for c in XC) + ' |')
        Lt += ['', f'### 「保つ − 反転」の Δ（{r}）', '', '| 入力 | 入れ替え | ' + ' | '.join(CNAME[c] for c in XC) + ' |',
               '|---|---|' + '---|' * len(XC)]
        for b, _, _ in BASES:
            for d in ('noswap', 'swap'):
                Lt.append(f'| {b} | {d} | ' + ' | '.join(f(S[f'{c}|HmF|{d}|V1|{b}|{r}']) for c in XC) + ' |')
        Lt += ['', f'### X3：入力の方角への応答 Rdir = Δ(b_old) − Δ(b_new)、ずれ（{r}）', '',
               '| 量 | ' + ' | '.join(CNAME[c] for c in XC) + ' |', '|---|' + '---|' * len(XC)]
        qs = [f'Rdir|{d}|V1|{r}' for d in ('noswap', 'swap')] + [f'shift|{d}|V1|{L}|{r}' for d in ('noswap', 'swap') for L in ('old', 'new')]
        if r == 'img':
            qs += ['Rdir|noswap|V0|img', 'shift|noswap|V0|old|img', 'shift|noswap|V0|new|img']
        for q in qs:
            Lt.append(f'| {q} | ' + ' | '.join(f(S[f'{c}|{q}']) for c in XC) + ' |')
        Lt.append('')
    x = chk['inputs']
    Lt += ['## 入力の大きさ（§1 確認3、V1、範囲ごとの値）', '',
           '| 量 | 自己 | 他者 |', '|---|---|---|']
    for k, nm in (('mean_abs', '平均 |P|'), ('max_abs', '最大 |P|'), ('mean_abs_nonzero', '0 でない範囲の平均 |P|'),
                  ('frac_nonzero', '0 でない範囲の割合'), ('mean_episode_max_abs', 'エピソードごとの最大 |P| の平均')):
        Lt.append(f"| {nm} | {chk['check3']['V1|noswap|self'][k]:.4f} | {chk['check3']['V1|noswap|other'][k]:.4f} |")
    Lt += ['', '| 入力 | 倍率 | 切り詰めの割合（全範囲） | 切り詰めの割合（0 でない範囲） | 切り詰めの後の平均 |P| |', '|---|---|---|---|---|']
    for nm, k in (('X1x2', '2'), ('X1x4', '4'), ('X2', f"{x['V1|X2_factor']:.4f}")):
        Lt.append(f"| {nm} | {k} | {x[f'V1|{nm}|clip_frac_all']:.4f} | {x[f'V1|{nm}|clip_frac_nonzero']:.4f} | "
                  f"{x[f'V1|{nm}|mean_abs_after_clip']:.4f} |")
    Lt += ['', '## 読めた割合（6 組・5 点の平均）', '', '| 条件 | 入力 | ' + ' | '.join(f'{d} {r}' for d in ('swap', 'noswap') for r in RO) + ' |',
           '|---|---|' + '---|' * 4]
    for c in CONDS:
        for inp in ORIG + (XIN if c in XC else ()):
            Lt.append(f'| {CNAME[c]} | {inp} | ' + ' | '.join(f"{FRm[c][f'read|{d}|V1|{inp}|{r}']:.3f}" for d in ('swap', 'noswap') for r in RO) + ' |')
    Lt += ['', '## 読み方の表（主 img、補足 feat）', '', '```', json.dumps(RD, indent=1, ensure_ascii=False), '```', '']
    open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(Lt))
    print('\n'.join(Lt))
    figure(S)


def figure(S):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    rows_self = [(f'{c} hold', f'{c}|r21|C|V1|self|hold') for c in CONDS] + \
                [(f'{a}-{b} hold', f'{a}-{b}|r21|C|V1|self|hold') for a, b in DIFFS]
    rows_other = [(f'{c} {inp}', f'{c}|C|V1|{inp}|{{}}') for c in XC for inp, _ in CIN] + \
                 [(f'{c} hold', f'{c}|C|V1|hold|{{}}') for c in ('none', 'sr50', 'sr80')] + \
                 [(f'{a}-{b} hold', f'{a}-{b}|C|V1|hold|{{}}') for a, b in DIFFS]
    fig, axs = plt.subplots(1, 3, figsize=(16, 0.3 * len(rows_other) + 1.5))
    panels = ((axs[0], rows_self, None, 'C_self (round-21 read-out)'), (axs[1], rows_other, 'img', 'C_other img'),
              (axs[2], rows_other, 'feat', 'C_other feat'))
    for ax, rows, r, title in panels:
        for i, (nm, q) in enumerate(rows):
            st = S[q.format(r) if r else q]
            y = len(rows) - i
            ax.plot(st['per_pair'], [y] * len(st['per_pair']), 'o', color='0.6', ms=3)
            ax.errorbar(st['mean'], y, xerr=[[st['mean'] - st['t']['ci95'][0]], [st['t']['ci95'][1] - st['mean']]],
                        fmt='D', color='C3' if nm.startswith(('p50 ', 'p80 ')) else 'k', ms=4, capsize=2)
        ax.axvline(0, color='0.3', lw=0.8)
        ax.set_yticks(range(len(rows), 0, -1))
        ax.set_yticklabels([r_[0] for r_ in rows], fontsize=7)
        ax.set_title(f'{title} (pairs grey, mean with pair t 95% CI)', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round22_C.{ext}'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
