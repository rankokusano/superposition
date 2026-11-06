"""
Paper round 24 Sec.1.5 and Sec.2 (paper/check/paper_instrctions_round24.md; docs/paper_extra_log.md Sec.48.2;
exploratory, descriptive, no training). GPU (GPU1-3).
Data: round 20 paper_r20_swap_p80 (A-2 moves toward its liked colour before the swap, p_rule 0.8, T_s = 50; A-2 likes
Red or Green, 300 each). Variants: S (no swap) and O (swap; A-2's path equal to S) = main contrast (as round 21: same
path, colours differ); R (no swap) and N (swap; A-2's path equal to R) = descriptive.
Models: the stage-II models (stage-I predictor + G, final weights ep200): p50 v12_s2_pref_mixed_s<s>, none
v12_s2_pref_random_s<s>, p80 v13_s2_pref_mixed_s<s> (descriptive), s = 0-3 (the only trained generators; there is no
generator for the sign-random 50% predictor and for pairs 5-6).
Flow (round 21's freeze with G): eval_trial_v12.load_v12, model.eval(), util.seed_all(0) and init_state per batch of 50,
frames t = 0..T0-1 shown (mask 0; process-2 = G's own output), T0 = 53 (frames 50-52 after the swap seen, as round 21
saw 20-22); then K = 10 steps with vision masked (1.0), motion hidden, p1_override = 0 and p2_override = [0, 0, P2, 1]:
  G  P2 = G's output q2_hat at step T0-1, held (G has seen A-2's movement before the swap and frames 50-52);
  V  P2 = model.compute_pleasure(A-2's view at frame T0-1) (round 21's input: "A-2 likes Red");
  Z  P2 = 0.
Read-out (round 21 / 23 'img'): A-2 position est_k = read_a2(pred_k, A-1 at frame T0+k-1), common-test calibration;
s_L = OLS slope over k of (est_k - a2) . unit(L - a2), a2 = A-2's position at frame T0-1, >= 5 finite points;
L_old = A-2's liked colour in the pre-swap layout, L_new = in the swapped layout; Delta = s_Lold - s_Lnew.
C_SO = Delta(S) - Delta(O) (main), C_RN = Delta(R) - Delta(N), per episode; by liked colour and pooled.
-> Round24/eval/<cond>_s<s>.npz ; mode `aggregate` -> Round24/analysis/{eval_summary.json, eval_tables.md,
   fig_round24_C.{pdf,png}}
Usage: python analyze/paper_extra/round24_eval.py run --jobs p50:0 ... | aggregate
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round21_eval as R21  # noqa
import round21_aggregate as A  # noqa  (stat, excl, SIGN, fmt)
from p2test_v11 import read_a2  # noqa
from sim_v10 import layout as L  # noqa
import util  # noqa

T = R21.T
RES = R21.RES
OUT = os.path.join(RES, 'paper_extra', 'Round24')
DATA = os.path.join(R21._PI, 'data', 'data', 'paper_r20_swap_p80')
EXP = {'p50': 'v12_s2_pref_mixed_s{s}', 'none': 'v12_s2_pref_random_s{s}', 'p80': 'v13_s2_pref_mixed_s{s}'}
CONDS = ('p50', 'none', 'p80')
PAIRS = (0, 1, 2, 3)
VARS = ('S', 'O', 'R', 'N')
INPUTS = ('G', 'V', 'Z')
T0, K = 53, 10


def load(v):
    g = h5py.File(os.path.join(DATA, f'data_{v}.h5'), 'r')['test']
    return {k: g[k][()] for k in ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_position',
                                  'layout', 'a2_liked')}


def targets(lay, lk):
    col = np.where(lk == 1, 'Red', 'Green')
    lo = np.array([L.LANDMARKS[L.CODE_LAYOUT[int(a)]][c] for a, c in zip(lay, col)], np.float64)
    ln = np.array([L.LANDMARKS[L.other_layout(L.CODE_LAYOUT[int(a)])][c] for a, c in zip(lay, col)], np.float64)
    return lo, ln


@torch.no_grad()
def flow(model, D, dev, batch=50):
    """-> {inp: preds (N, K, 16, 64, 3)}, q2 at step T0-1 (N, 8)."""
    N = D['self_vision'].shape[0]
    out = {i: [] for i in INPUTS}
    Q2 = []
    for b0 in range(0, N, batch):
        x = R21.scaled(D['self_vision'][b0:b0 + batch, :T0 + K], dev)
        m = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(dev)
        ov = R21.scaled(D['other_vision'][b0:b0 + batch, T0 - 1:T0], dev)[:, 0]
        B = x.shape[0]
        for inp in INPUTS:
            util.seed_all(0)
            model.init_state(B)
            for t in range(T0):
                o = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0)
            q2 = o['q2_hat']
            if inp == 'G':
                Q2.append(q2.cpu().numpy())
            Q = torch.zeros(B, 11, device=dev)
            Q[:, 10] = 1.0
            if inp == 'G':
                Q[:, 2:10] = q2
            elif inp == 'V':
                Q[:, 2:10] = model.compute_pleasure(ov)
            P1 = torch.zeros(B, 8, device=dev)
            hid = torch.ones(B, dtype=torch.bool, device=dev)
            pr = []
            for k in range(K):
                o = model({'self_vision': x[:, T0 + k], 'self_motion': m[:, T0 + k]}, 1.0, 1.0, p1_override=P1,
                          p2_override=Q, hidden_override=hid)
                pr.append(((o['self_vision'] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
            out[inp].append(np.stack(pr, 1))
    return {i: np.concatenate(v) for i, v in out.items()}, np.concatenate(Q2)


def run(jobs):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    ab = R21.calib_common()
    DD = {v: load(v) for v in VARS}
    lo, ln = targets(DD['S']['layout'], DD['S']['a2_liked'])
    os.makedirs(os.path.join(OUT, 'eval'), exist_ok=True)
    for j in jobs:
        c, s = j.split(':')
        s = int(s)
        f = os.path.join(OUT, 'eval', f'{c}_s{s}.npz')
        if os.path.exists(f):
            continue
        t0 = time.time()
        model, _ = T.load_v12(EXP[c].format(s=s), s, 200, dev)
        model.eval()
        sav = {}
        for v in VARS:
            D = DD[v]
            a2 = D['other_position'][:, T0 - 1].astype(np.float64)
            sp = D['self_position'][:, T0 - 1:T0 - 1 + K].astype(np.float64)
            preds, q2 = flow(model, D, dev)
            sav[f'{v}_q2'] = q2.astype(np.float32)
            for inp, pr in preds.items():
                est = np.stack([read_a2(pr[:, i], sp[:, i], *ab) for i in range(K)], 1)
                S = R21.other_slopes(est, a2, lo, ln)
                sav[f'{v}_{inp}_delta'] = (S[:, 0] - S[:, 1]).astype(np.float32)
                sav[f'{v}_{inp}_est'] = est.astype(np.float32)
        np.savez_compressed(f + '.tmp.npz', **sav)
        os.replace(f + '.tmp.npz', f)
        print(f'{c}:{s} {time.time() - t0:.0f}s', flush=True)


def aggregate():
    lk = load('S')['a2_liked']
    S, FR, Q = {}, {}, {}
    for c in CONDS:
        Z = [np.load(os.path.join(OUT, 'eval', f'{c}_s{s}.npz')) for s in PAIRS]
        for inp in INPUTS:
            d = {v: np.stack([z[f'{v}_{inp}_delta'].astype(np.float64) for z in Z]) for v in VARS}
            for nm, a, b in (('C_SO', 'S', 'O'), ('C_RN', 'R', 'N')):
                X = d[a] - d[b]
                S[f'{c}|{inp}|{nm}|all'] = A.stat(X)
                for lv, ln_ in ((1, 'red'), (0, 'green')):
                    S[f'{c}|{inp}|{nm}|{ln_}'] = A.stat(X[:, lk == lv])
            for v in VARS:
                S[f'{c}|{inp}|Delta|{v}'] = A.stat(d[v])
                FR[f'{c}|{inp}|{v}'] = float(np.isfinite(d[v]).mean())
                FR[f'{c}|{inp}|est|{v}'] = float(np.mean([np.isfinite(z[f'{v}_{inp}_est'][..., 0]).mean() for z in Z]))
        q = np.stack([z['S_q2'] for z in Z])                   # (4, N, 8)
        Q[c] = {'red': q[:, lk == 1].mean((0, 1)).round(4).tolist(), 'green': q[:, lk == 0].mean((0, 1)).round(4).tolist(),
                'mean_abs': float(np.abs(q).mean())}
    r21 = json.load(open(os.path.join(RES, 'paper_extra', 'Round21', 'analysis', 'summary.json')))['stats']
    R21C = {f'{c}|{pv}': r21[f'{c}|C|V1|other|{pv}'] for c in ('p50', 'none', 'p80') for pv in ('hold', 'zero')}
    json.dump({'stats': S, 'readable': FR, 'q2_mean_S_frame52': Q, 'round21_C_other': R21C},
              open(os.path.join(OUT, 'analysis', 'eval_summary.json'), 'w'), indent=1, ensure_ascii=False)
    f = A.fmt
    CN = {'p50': '経験50%', 'none': '経験なし', 'p80': '経験80%（記述）'}
    IN = {'G': '生成器の出力（保つ）', 'V': '第21弾の入力（A-2 の視界、赤が好き）', 'Z': '零'}
    Lt = ['# 第24弾 §2 出発点の値（記述）', '', '第20弾のデータ（`p_rule` 0.8、$T_s=50$）、入れ替えの後のフレーム 50〜52 を見せて 53 から 10 歩止める。'
          '値：平均 [組の t の 95% CI（4 組、自由度 3）] [2段のブートストラップ]。$C_{SO}$ = Δ(S) − Δ(O)（主）、$C_{RN}$ = Δ(R) − Δ(N)（記述）。', '']
    for nm in ('C_SO', 'C_RN'):
        Lt += [f'## ${nm}$', '', '| 条件 | 入力 | 全部 | 赤好き | 緑好き |', '|---|---|---|---|---|']
        for c in CONDS:
            for inp in INPUTS:
                Lt.append(f'| {CN[c]} | {IN[inp]} | ' + ' | '.join(f(S[f'{c}|{inp}|{nm}|{g}']) for g in ('all', 'red', 'green')) + ' |')
        Lt.append('')
    Lt += ['## Δ（記述）', '', '| 条件 | 入力 | ' + ' | '.join(VARS) + ' |', '|---|---|' + '---|' * 4]
    for c in CONDS:
        for inp in INPUTS:
            Lt.append(f'| {CN[c]} | {inp} | ' + ' | '.join(f"{S[f'{c}|{inp}|Delta|{v}']['mean']:+.4f}" for v in VARS) + ' |')
    Lt += ['', '## 読めた割合（Δ が有限／A-2 の位置が読めた）', '', '| 条件 | 入力 | ' + ' | '.join(VARS) + ' |', '|---|---|' + '---|' * 4]
    for c in CONDS:
        for inp in INPUTS:
            Lt.append(f'| {CN[c]} | {inp} | ' + ' | '.join(f"{FR[f'{c}|{inp}|{v}']:.3f}／{FR[f'{c}|{inp}|est|{v}']:.3f}" for v in VARS) + ' |')
    Lt += ['', '## 生成器の出力（S、ステップ 52、範囲 0〜7 の平均）', '']
    for c in CONDS:
        Lt.append(f"- {CN[c]}：赤好き {Q[c]['red']}、緑好き {Q[c]['green']}、平均 |q2| {Q[c]['mean_abs']:.4f}")
    Lt += ['', '## 第21弾の $C_{other}$（静止した A-2、V1、6 組）', '']
    for k, st in R21C.items():
        Lt.append(f'- {k}: {f(st)}')
    open(os.path.join(OUT, 'analysis', 'eval_tables.md'), 'w').write('\n'.join(Lt))
    print('\n'.join(Lt))
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    rows = [(f'{c} {inp} {g}', f'{c}|{inp}|C_SO|{g}') for c in CONDS for inp in INPUTS for g in ('all', 'red', 'green')]
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(rows) + 1.2))
    for i, (nm, q) in enumerate(rows):
        st = S[q]
        y = len(rows) - i
        ax.plot(st['per_pair'], [y] * len(st['per_pair']), 'o', color='0.6', ms=3)
        col = {'G': 'C3', 'V': 'C0', 'Z': 'k'}[nm.split()[1]]
        ax.errorbar(st['mean'], y, xerr=[[st['mean'] - st['t']['ci95'][0]], [st['t']['ci95'][1] - st['mean']]],
                    fmt='D', color=col, ms=4, capsize=2)
    ax.axvline(0, color='0.3', lw=0.8)
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_title('C_SO = Delta(S) - Delta(O) (G red, V blue, Z black; pairs grey, pair t 95% CI)', fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, 'analysis', f'fig_round24_C.{ext}'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('run', 'aggregate'))
    ap.add_argument('--jobs', nargs='*', default=[])
    a = ap.parse_args()
    if a.mode == 'run':
        run(a.jobs)
    else:
        aggregate()
