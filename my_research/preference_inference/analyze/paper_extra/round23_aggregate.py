"""
Paper round 23 Sec.2 aggregation (docs/paper_extra_log.md Sec.47.2; exploratory, outside the pre-registration; written
and committed before any evaluation output was read). CPU.
Sec.2.1 (positive control, round 20 data p_rule 0.5, steps 53..79): per episode Delta = s_X - s_Y; the read-out says
"X" if Delta > 0, "Y" if Delta < 0 (0 counts 0.5); truth X for O and S, Y for N and R. Episode correctness = mean over
the finite checkpoints; balanced accuracy BA(N/O) = (mean correct of N + mean correct of O) / 2 over finite episodes,
BA(S/R) likewise. Read-outs: true features / true image (model independent, Round23/ref/true_r20.npz; CI = episode
bootstrap 1000, default_rng(0)), predicted features / predicted image per condition x input (pair t 95% CI and
two-stage bootstrap 1000, default_rng(0): pairs, then episodes with the same index for both variants).
Sec.2.2 (round 21 data, freeze flow): M(c) = slope along class c's direction with +0.28 in range c minus the slope
along the same direction with the zero input (same episode / checkpoint), read-outs feat (main) and img; episode value
= mean over finite checkpoints; pair t CI and two-stage bootstrap (round21_aggregate.stat); differences p50 - none,
p50 - sr50, p80 - none per episode. Raw slopes are also given.
Reading (Sec.47.2, fixed) in reading().
-> Round23/analysis/{summary.json, tables.md, fig_round23_BA.{pdf,png}, fig_round23_M.{pdf,png}}
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round21_aggregate as A  # noqa  (stat, excl, SIGN, fmt)
import vdir_stageA as VA  # noqa  (tsum)

RES = A.RES
R23 = os.environ.get('ROUND23_DIR', os.path.join(RES, 'paper_extra', 'Round23'))  # override only for the synthetic test
OUT = os.path.join(R23, 'analysis')
EPS = (160, 170, 180, 190, 200)
VARS = ('N', 'O', 'S', 'R')
TRUTH_X = {'N': False, 'O': True, 'S': True, 'R': False}
PAIRTYPES = (('NO', 'N', 'O'), ('SR', 'S', 'R'))
R20C = (('p50', 'Z', 6), ('p80', 'Z', 6), ('none', 'Z', 6), ('p50', 'T', 6), ('p80', 'T', 6), ('none', 'T', 6),
        ('Gp50', 'G', 4), ('Gnone', 'G', 4))
DIRC = ('p50', 'p80', 'none', 'sr50')
DIFFS = (('p50', 'none'), ('p50', 'sr50'), ('p80', 'none'))
CLASSES = ('E', 'B', 'R')
CNAME = {'p50': '経験50%', 'p80': '経験80%', 'none': '経験なし', 'sr50': '符号無作為（経験50%）', 'Gp50': '生成器（経験50%）',
         'Gnone': '生成器（経験なし）'}
B = 1000


def correct(delta, truth_x):
    c = np.where(delta > 0, 1.0, np.where(delta < 0, 0.0, 0.5))
    if not truth_x:
        c = 1.0 - c
    return np.where(np.isfinite(delta), c, np.nan)


def ba(c1, c2):
    with np.errstate(all='ignore'):
        return 0.5 * (np.nanmean(c1, -1) + np.nanmean(c2, -1))


def ba_episode_boot(c1, c2):
    rng = np.random.default_rng(0)
    N = c1.shape[0]
    ei = rng.integers(0, N, (B, N))
    v = ba(c1[ei], c2[ei])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def ba_two_stage(C1, C2):
    rng = np.random.default_rng(0)
    npairs, N = C1.shape
    pi = rng.integers(0, npairs, (B, npairs))
    ei = rng.integers(0, N, (B, npairs, N))
    a, b = C1[pi[:, :, None], ei], C2[pi[:, :, None], ei]
    with np.errstate(all='ignore'):
        v = np.nanmean(ba(a, b), 1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def ba_stat(C1, C2):
    pm = ba(C1, C2)
    return {'mean': float(pm.mean()), 't': VA.tsum(pm.tolist()), 'boot': ba_two_stage(C1, C2), 'per_pair': pm.tolist(),
            'readable': float(np.isfinite(np.concatenate([C1, C2], 1)).mean())}


def load_r20(c, inp, s):
    acc = {}
    for e in EPS:
        z = np.load(os.path.join(R23, 'eval_r20', f'{c}_s{s}_ep{e}.npz'))
        for r in ('feat', 'img'):
            for v in VARS:
                d = z[f'{inp}_{v}_{r}'].astype(np.float64)
                acc.setdefault((r, v), []).append(correct(d, TRUTH_X[v]))
                acc.setdefault((r, v, 'delta'), []).append(d)
    with np.errstate(all='ignore'):
        return {k: np.nanmean(np.stack(x), 0) for k, x in acc.items()}


def load_dir(c, s):
    acc = {}
    for e in EPS:
        z = np.load(os.path.join(R23, 'eval_dir', f'{c}_s{s}_ep{e}.npz'))
        for r in ('feat', 'img'):
            for k in CLASSES:
                a, b = z[f'{k}_{k}_{r}'].astype(np.float64), z[f'zero_{k}_{r}'].astype(np.float64)
                acc.setdefault(f'M|{k}|{r}', []).append(a - b)
                acc.setdefault(f'raw|{k}|{r}', []).append(a)
                acc.setdefault(f'rawzero|{k}|{r}', []).append(b)
    with np.errstate(all='ignore'):
        return {k: np.nanmean(np.stack(x), 0) for k, x in acc.items()}


def reading(S21, TRUE, S22):
    tb = {pt: TRUE[f'feat|{pt}']['mean'] for pt, _, _ in PAIRTYPES}
    true_high = all(v >= 0.8 for v in tb.values())
    true_low = any(v < 0.65 for v in tb.values())
    pred = {}
    for c in ('p50', 'p80'):
        for pt, _, _ in PAIRTYPES:
            st = S21[f'{c}|Z|feat|{pt}']
            pred[f'{c}|{pt}'] = bool(st['t']['ci95'][0] > 0.5 and st['boot'][0] > 0.5)
    pc = {'true_feat_BA': tb, 'true_high_ge_0.8': true_high, 'true_low_lt_0.65': true_low,
          'pred_feat_BA_clearly_above_0.5': pred,
          'row1_reads_movement': true_high and all(pred.values()), 'row2_not_movement': true_low}
    pc['no_row'] = not (pc['row1_reads_movement'] or pc['row2_not_movement'])

    def ex(q):
        return A.excl(S22[q])
    sg = {c: {k: A.SIGN[ex(f'{c}|M|{k}|feat')] for k in CLASSES} for c in DIRC}
    sgd = {f'{a}-{b}': {k: A.SIGN[ex(f'{a}-{b}|M|{k}|feat')] for k in CLASSES} for a, b in DIFFS}
    rows = {'rowA_E_pos_50_80_above_none': ex('p50|M|E|feat') == 1 and ex('p80|M|E|feat') == 1 and
            ex('p50-none|M|E|feat') == 1 and ex('p80-none|M|E|feat') == 1}
    for c in ('p50', 'p80'):
        e, b, r = (ex(f'{c}|M|{k}|feat') for k in CLASSES)
        rows[f'rowB_R_only_{c}'] = r == 1 and e == 0 and b == 0
        rows[f'rowC_R_and_B_not_E_{c}'] = r == 1 and b == 1 and e == 0
        rows[f'rowD_all_zero_{c}'] = e == 0 and b == 0 and r == 0
    dm = {'M_feat_sign': sg, 'diff_sign': sgd, 'rows': rows, 'no_row': not any(rows.values()),
          'read_as_movement_allowed': pc['row1_reads_movement']}
    dm['M_img_sign'] = {c: {k: A.SIGN[ex(f'{c}|M|{k}|img')] for k in CLASSES} for c in DIRC}
    return {'positive_control': pc, 'appearance_vs_movement': dm}


def main():
    os.makedirs(OUT, exist_ok=True)
    # Sec.2.1
    t = np.load(os.path.join(R23, 'ref', 'true_r20.npz'))
    TRUE = {}
    for r in ('feat', 'img'):
        cc = {v: correct(t[f'true_{v}_{r}'].astype(np.float64), TRUTH_X[v]) for v in VARS}
        for pt, a, b in PAIRTYPES:
            TRUE[f'{r}|{pt}'] = {'mean': float(ba(cc[a], cc[b])), 'boot_episode': ba_episode_boot(cc[a], cc[b]),
                                 'readable': float(np.isfinite(np.concatenate([cc[a], cc[b]])).mean())}
        for v in VARS:
            TRUE[f'{r}|delta_mean|{v}'] = float(np.nanmean(t[f'true_{v}_{r}']))
    S21 = {}
    for c, inp, npairs in R20C:
        EV = [load_r20(c, inp, s) for s in range(npairs)]
        for r in ('feat', 'img'):
            for pt, a, b in PAIRTYPES:
                S21[f'{c}|{inp}|{r}|{pt}'] = ba_stat(np.stack([E[(r, a)] for E in EV]), np.stack([E[(r, b)] for E in EV]))
            for v in VARS:
                S21[f'{c}|{inp}|{r}|delta|{v}'] = A.stat(np.stack([E[(r, v, 'delta')] for E in EV]))
    # Sec.2.2
    EVD = {(c, s): load_dir(c, s) for c in DIRC for s in range(6)}
    S22 = {}
    for c in DIRC:
        for k in EVD[(c, 0)]:
            S22[f'{c}|{k}'] = A.stat(np.stack([EVD[(c, s)][k] for s in range(6)]))
    for a, b in DIFFS:
        for k in EVD[(a, 0)]:
            S22[f'{a}-{b}|{k}'] = A.stat(np.stack([EVD[(a, s)][k] - EVD[(b, s)][k] for s in range(6)]))
    RD = reading(S21, TRUE, S22)
    r22 = json.load(open(os.path.join(RES, 'paper_extra', 'Round22', 'analysis', 'summary.json')))['stats']
    R22X3 = {f'{c}|{q}': r22[f'{c}|{q}'] for c in ('p50', 'p80') for q in ('Rdir|noswap|V1|img', 'Rdir|noswap|V1|feat')}
    json.dump({'true': TRUE, 'r20': S21, 'dir': S22, 'reading': RD, 'round22_X3': R22X3},
              open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
    f = A.fmt

    def fb(st):
        return f"{st['mean']:.3f} [{st['t']['ci95'][0]:.3f}, {st['t']['ci95'][1]:.3f}] [{st['boot'][0]:.3f}, {st['boot'][1]:.3f}]"
    Lt = ['# 第23弾 §2（探索・事前登録外）', '',
          '## 2.1 陽性対照：正答率（BA）', '',
          '第20弾のデータ `p_rule` 0.5、ステップ 53〜79。Δ = $s_X - s_Y$、Δ > 0 なら「X（古い場所）へ」。正解は O・S が X、N・R が Y。'
          '本当の特徴・本当の画像はモデルによらない（CI はエピソードのブートストラップ）。予測は「平均 [組の t の 95% CI] [2段のブートストラップ]」。', '',
          '| 読み出し | N/O | S/R | 読めた割合 |', '|---|---|---|---|']
    for r, nm in (('feat', '本当の特徴'), ('img', '本当の画像')):
        Lt.append(f'| {nm} | ' + ' | '.join(f"{TRUE[f'{r}|{pt}']['mean']:.3f} [{TRUE[f'{r}|{pt}']['boot_episode'][0]:.3f}, "
                                            f"{TRUE[f'{r}|{pt}']['boot_episode'][1]:.3f}]" for pt, _, _ in PAIRTYPES) +
                  f" | {TRUE[f'{r}|NO']['readable']:.3f} |")
    Lt += ['', '| 条件 | 入力 | 読み出し | N/O | S/R | 読めた割合 |', '|---|---|---|---|---|---|']
    for c, inp, _ in R20C:
        for r, nm in (('feat', '予測した特徴'), ('img', '予測画像の位置')):
            Lt.append(f'| {CNAME[c]} | {inp} | {nm} | ' + ' | '.join(fb(S21[f'{c}|{inp}|{r}|{pt}']) for pt, _, _ in PAIRTYPES) +
                      f" | {S21[f'{c}|{inp}|{r}|NO']['readable']:.3f} |")
    Lt += ['', '### Δ の平均（記述）', '', '| 読み出し | ' + ' | '.join(VARS) + ' |', '|---|' + '---|' * 4]
    for r in ('feat', 'img'):
        Lt.append(f'| 本当 {r} | ' + ' | '.join(f"{TRUE[f'{r}|delta_mean|{v}']:+.4f}" for v in VARS) + ' |')
    for c, inp, _ in R20C:
        for r in ('feat', 'img'):
            Lt.append(f'| {CNAME[c]} {inp} {r} | ' + ' | '.join(f"{S21[f'{c}|{inp}|{r}|delta|{v}']['mean']:+.4f}" for v in VARS) + ' |')
    Lt += ['', '## 2.2 見え方と動き：$M$（+0.28 を入れた範囲の方角への傾き − 零の入力の同じ方角への傾き）', '',
           '値：平均 [組の t の 95% CI] [2段のブートストラップ]、6 組。単位は 1 歩あたりの $d$（feat）・位置（img）。', '']
    for r in ('feat', 'img'):
        Lt += [f'### {r}', '', '| 条件・差 | E（箱のない方角） | B（青・シアン） | R（赤） |', '|---|---|---|---|']
        for c in DIRC + tuple(f'{a}-{b}' for a, b in DIFFS):
            Lt.append(f'| {CNAME.get(c, c)} | ' + ' | '.join(f(S22[f'{c}|M|{k}|{r}']) for k in CLASSES) + ' |')
        Lt += ['', f'#### 生の傾き（{r}、記述）', '', '| 条件 | 入力 E | 入力 B | 入力 R | 零（E の方角） | 零（B の方角） | 零（R の方角） |',
               '|---|---|---|---|---|---|---|']
        for c in DIRC:
            Lt.append(f'| {CNAME[c]} | ' + ' | '.join(f"{S22[f'{c}|raw|{k}|{r}']['mean']:+.4f}" for k in CLASSES) + ' | ' +
                      ' | '.join(f"{S22[f'{c}|rawzero|{k}|{r}']['mean']:+.4f}" for k in CLASSES) + ' |')
        Lt.append('')
    Lt += ['### 読めた割合（$M$ が有限の割合、条件 × 範囲）', '', '| 条件 | ' + ' | '.join(f'{k} {r}' for r in ('feat', 'img') for k in CLASSES) + ' |',
           '|---|' + '---|' * 6]
    for c in DIRC:
        Lt.append(f'| {CNAME[c]} | ' + ' | '.join(
            f"{np.mean([np.isfinite(EVD[(c, s)][f'M|{k}|{r}']).mean() for s in range(6)]):.3f}" for r in ('feat', 'img') for k in CLASSES) + ' |')
    Lt += ['', '### 第22弾の X3 の応答（記述、比べるため）', '']
    for k, st in R22X3.items():
        Lt.append(f'- {k}: {f(st)}')
    Lt += ['', '## 読み方の表', '', '```', json.dumps(RD, indent=1, ensure_ascii=False), '```', '']
    open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(Lt))
    print('\n'.join(Lt))
    figures(TRUE, S21, S22)


def figures(TRUE, S21, S22):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (pt, _, _) in zip(axs, PAIRTYPES):
        rows = [(f'true {r}', None, r) for r in ('feat', 'img')] + \
               [(f'{c} {inp} {r}', f'{c}|{inp}|{r}|{pt}', r) for c, inp, _ in R20C for r in ('feat', 'img')]
        for i, (nm, q, r) in enumerate(rows):
            y = len(rows) - i
            col = 'C0' if r == 'feat' else 'C1'
            if q is None:
                st = TRUE[f'{r}|{pt}']
                lo, hi = st['boot_episode']
                ax.errorbar(st['mean'], y, xerr=[[st['mean'] - lo], [hi - st['mean']]], fmt='s', color=col, ms=5, capsize=2)
            else:
                st = S21[q]
                ax.plot(st['per_pair'], [y] * len(st['per_pair']), 'o', color='0.6', ms=3)
                ax.errorbar(st['mean'], y, xerr=[[st['mean'] - st['t']['ci95'][0]], [st['t']['ci95'][1] - st['mean']]],
                            fmt='D', color=col, ms=4, capsize=2)
        ax.axvline(0.5, color='0.3', lw=0.8)
        ax.axvline(0.8, color='0.7', lw=0.8, ls=':')
        ax.set_yticks(range(len(rows), 0, -1))
        ax.set_yticklabels([r_[0] for r_ in rows], fontsize=7)
        ax.set_xlabel('balanced accuracy')
        ax.set_title(f'{pt}: true (episode bootstrap CI), predicted (pairs grey, pair t CI)', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round23_BA.{ext}'), dpi=130)
    plt.close(fig)
    rows = [(f'{c} {k}', f'{c}|M|{k}|{{}}') for c in DIRC for k in CLASSES] + \
           [(f'{a}-{b} {k}', f'{a}-{b}|M|{k}|{{}}') for a, b in DIFFS for k in CLASSES]
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.3 * len(rows) + 1.4), sharey=True)
    for ax, r in zip(axs, ('feat', 'img')):
        for i, (nm, q) in enumerate(rows):
            st = S22[q.format(r)]
            y = len(rows) - i
            ax.plot(st['per_pair'], [y] * len(st['per_pair']), 'o', color='0.6', ms=3)
            col = {'E': 'C2', 'B': 'C0', 'R': 'C3'}[nm.split()[-1]]
            ax.errorbar(st['mean'], y, xerr=[[st['mean'] - st['t']['ci95'][0]], [st['t']['ci95'][1] - st['mean']]],
                        fmt='D', color=col, ms=4, capsize=2)
        ax.axvline(0, color='0.3', lw=0.8)
        ax.set_yticks(range(len(rows), 0, -1))
        ax.set_yticklabels([r_[0] for r_ in rows], fontsize=7)
        ax.set_title(f'M_{r} (E green, B blue, R red; pairs grey, pair t 95% CI)', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round23_M.{ext}'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
