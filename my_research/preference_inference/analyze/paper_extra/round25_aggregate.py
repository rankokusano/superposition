"""
Paper round 25 aggregation (docs/paper_extra_log.md Sec.49.2; trial on pair 1; written and committed before any
training output was read). CPU. Input: Round25/runs/<run>/{eval/ep<e>.npz, log.json} (round25_train.py).
Per evaluation epoch e and episode: C_G = Delta(S, G) - Delta(O, G), C_Z = Delta(S, Z) - Delta(O, Z),
D(e) = C_G - C_Z; 'red' / 'green' = A-2's liked colour (300 episodes each), 'all' = their mean (= the mean over the
600 episodes).
  D_early = stage-1 mean of D over epochs 5, 10, 15, 20, 25 (per episode, then the mean); D_end1 = stage 1 epoch 100;
  D_end2 = stage 2 epoch 50. Differences p50 - sr50 (main), p50 - none: per episode (same episodes).
With one pair the CI is the episode bootstrap (1000, default_rng(0)); the pair t CI and the two-stage bootstrap of
Sec.49.2 are for the full run (pairs 1-6).
Rules of Sec.2.3 (trial only) and the Sec.2.2 table applied as a description: see rules().
-> Round25/analysis/{summary.json, tables.md, fig_round25_D.{pdf,png}, fig_round25_self.{pdf,png},
   fig_round25_q2.{pdf,png}}
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
R25 = os.environ.get('ROUND25_DIR', os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round25'))  # override only for tests
OUT = os.path.join(R25, 'analysis')
CONDS = ('p50', 'sr50', 'none')
DIFFS = (('p50', 'sr50'), ('p50', 'none'))
RUNS1 = [f'stage1_{c}_{v}' for c in CONDS for v in ('M', 'N')]
RUNS2 = [f'stage2_{c}_r{r}' for c in CONDS for r in ('0.1', '0.01')]
EARLY = (5, 10, 15, 20, 25)
GROUPS = ('all', 'red', 'green')
B = 1000


def boot(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(0)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return {'mean': float(x.mean()), 'ci': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], 'n': int(len(x))}


def sign(st):
    lo, hi = st['ci']
    return '>0' if lo > 0 else ('<0' if hi < 0 else '≈0')


def load_run(run):
    d = os.path.join(R25, 'runs', run)
    ev = {}
    for f in sorted(os.listdir(os.path.join(d, 'eval'))):
        z = np.load(os.path.join(d, 'eval', f))
        e = int(f[2:5])
        cg = z['S_G_delta'].astype(float) - z['O_G_delta'].astype(float)
        cz = z['S_Z_delta'].astype(float) - z['O_Z_delta'].astype(float)
        r = {'C_G': cg, 'C_Z': cz, 'D': cg - cz, 'liked': z['a2_liked'],
             'readable': {k: float(z[k]) for k in z.files if k.endswith('_readable')},
             'q2_raw': z['q2_abs_raw'].mean(0), 'q2_in': z['q2_abs_in'].mean(0)}
        if 'self_l1' in z.files:
            r['self_l1'] = float(z['self_l1'])
            r['c_self'] = z['c_self'].astype(float)
        ev[e] = r
    log = json.load(open(os.path.join(d, 'log.json')))
    return ev, log


def grp(x, liked, g):
    if g == 'all':
        return x
    return x[liked == (1 if g == 'red' else 0)]


def summarise(R):
    S = {}
    for run, (ev, _) in R.items():
        for e, r in ev.items():
            for q in ('D', 'C_G', 'C_Z'):
                for g in GROUPS:
                    S[f'{run}|{e}|{q}|{g}'] = boot(grp(r[q], r['liked'], g))
            if 'c_self' in r:
                S[f'{run}|{e}|c_self'] = boot(r['c_self'])
                S[f'{run}|{e}|self_l1'] = r['self_l1']
    key = {}
    for c in CONDS:
        for v in ('M', 'N'):
            ev = R[f'stage1_{c}_{v}'][0]
            key[(c, v, 'early')] = np.nanmean(np.stack([ev[e]['D'] for e in EARLY]), 0)
            key[(c, v, 'end1')] = ev[100]['D']
            key[(c, v, 'liked')] = ev[100]['liked']
        for r in ('0.1', '0.01'):
            ev = R[f'stage2_{c}_r{r}'][0]
            key[(c, r, 'end2')] = ev[50]['D']
            key[(c, r, 'liked')] = ev[50]['liked']
    for (c, v, q), x in list(key.items()):
        if q == 'liked':
            continue
        lk = key[(c, v, 'liked')]
        for g in GROUPS:
            S[f'{c}|{v}|D_{q}|{g}'] = boot(grp(x, lk, g))
        for a, b in DIFFS:
            if c == a:
                for g in GROUPS:
                    S[f'{a}-{b}|{v}|D_{q}|{g}'] = boot(grp(x - key[(b, v, q)], lk, g))
    return S


def moves(S, c, v, qs):
    return any(sign(S[f'{c}|{v}|D_{q}|all']) != '≈0' for q in qs)


def rules(S, R):
    out = {}
    lr = {}
    for r in ('0.1', '0.01'):
        ok, detail = True, {}
        for c in CONDS:
            ev = R[f'stage2_{c}_r{r}'][0]
            l0, cs0 = ev[0]['self_l1'], S[f'stage2_{c}_r{r}|0|c_self']
            eps = sorted(e for e in ev if e > 0)
            l1_ok = all(ev[e]['self_l1'] <= 1.05 * l0 for e in eps)
            need_cs = cs0['ci'][0] > 0
            cs_ok = (not need_cs) or all(S[f'stage2_{c}_r{r}|{e}|c_self']['mean'] >= 0.5 * cs0['mean'] for e in eps)
            detail[c] = {'l1_start': l0, 'l1_max_ratio': max(ev[e]['self_l1'] / l0 for e in eps),
                         'c_self_start': cs0['mean'], 'c_self_start_ci': cs0['ci'], 'c_self_checked': need_cs,
                         'c_self_min_ratio': min(S[f'stage2_{c}_r{r}|{e}|c_self']['mean'] / cs0['mean'] for e in eps) if cs0['mean'] else None,
                         'l1_ok': l1_ok, 'c_self_ok': cs_ok}
            ok = ok and l1_ok and cs_ok
        lr[r] = {'pass': ok, 'conditions': detail}
    out['stage2_lr'] = lr
    out['stage2_lr_choice'] = '0.1' if lr['0.1']['pass'] else ('0.01' if lr['0.01']['pass'] else 'STOP')
    mv_M = {c: moves(S, c, 'M', ('early', 'end1')) or any(moves(S, c, r, ('end2',)) for r in ('0.1', '0.01')) for c in CONDS}
    mv_N = {c: moves(S, c, 'N', ('early', 'end1')) for c in CONDS}
    out['D_moves_M'] = mv_M
    out['D_moves_N'] = mv_N
    out['range_rule_stop'] = (not any(mv_M.values())) and any(mv_N.values())
    out['p_rule_rule_stop'] = not (any(mv_M.values()) or any(mv_N.values()))
    out['settings'] = {'stage2_lstm_lr_ratio': out['stage2_lr_choice'], 'ranges_2_5': 'M (masked)'}
    out['trial_reusable_as_pair1'] = out['stage2_lr_choice'] in ('0.1', '0.01') and not out['range_rule_stop'] and not out['p_rule_rule_stop']
    # Sec.2.2 as a description (one pair; main = M, stage 2 = the chosen ratio)
    rr = out['stage2_lr_choice'] if out['stage2_lr_choice'] != 'STOP' else '0.1'
    sg = lambda k: sign(S[k])  # noqa
    t = {'row1_early_p50_gt_sr50_and_none': sg('p50-sr50|M|D_early|all') == '>0' and sg('p50-none|M|D_early|all') == '>0',
         'row2_end1_only_p50_pos_and_end2_no_diff': sg('p50|M|D_end1|all') == '>0' and sg('sr50|M|D_end1|all') != '>0'
         and sg('none|M|D_end1|all') != '>0' and sg(f'p50-sr50|{rr}|D_end2|all') == '≈0' and sg(f'p50-none|{rr}|D_end2|all') == '≈0',
         'row3_all_end2_pos_and_no_early_diff': all(sg(f'{c}|{rr}|D_end2|all') == '>0' for c in CONDS)
         and sg('p50-sr50|M|D_early|all') == '≈0' and sg('p50-none|M|D_early|all') == '≈0',
         'row4_all_end2_zero': all(sg(f'{c}|{rr}|D_end2|all') == '≈0' for c in CONDS)}
    out['table_2_2_description'] = t
    out['green_p50_initial_D'] = S['stage1_p50_M|0|D|green']
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {run: load_run(run) for run in RUNS1 + RUNS2}
    S = summarise(R)
    RU = rules(S, R)
    q2 = {run: {e: {'raw': R[run][0][e]['q2_raw'].round(4).tolist(), 'in': R[run][0][e]['q2_in'].round(4).tolist()}
                for e in (0, max(R[run][0]))} for run in R}
    rd = {run: {e: r['readable'] for e, r in R[run][0].items()} for run in R}
    times = {run: {'epoch_sec_median': float(np.median([x['sec'] for x in R[run][1]['epochs']])),
                   'eval_sec_median': float(np.median([x['eval_sec'] for x in R[run][1]['epochs'] if 'eval_sec' in x])),
                   'g_init_sha': R[run][1]['g_init_sha']} for run in R}
    json.dump({'stats': S, 'rules': RU, 'q2': q2, 'readable': rd, 'times': times}, open(os.path.join(OUT, 'summary.json'), 'w'),
              indent=1, ensure_ascii=False)
    fm = lambda st: f"{st['mean']:+.4f} [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}]"  # noqa
    CN = {'p50': '経験50%', 'sr50': '符号無作為（経験50%）', 'none': '経験なし'}
    L = ['# 第25弾 試し（組1）', '', '値：平均 [エピソードのブートストラップの 95% CI]（1 組なので組の CI はない）。$D = C_{other}$(生成器) − $C_{other}$(零)、$C$ = Δ(S) − Δ(O)。', '',
         '## 主な量', '', '| 量 | 段・形 | 条件・差 | 全部 | 赤好き | 緑好き |', '|---|---|---|---|---|---|']
    for q, vs in (('D_early', ('M', 'N')), ('D_end1', ('M', 'N')), ('D_end2', ('0.1', '0.01'))):
        for v in vs:
            for c in CONDS + tuple(f'{a}-{b}' for a, b in DIFFS):
                L.append(f'| {q} | {v} | {CN.get(c, c)} | ' + ' | '.join(fm(S[f'{c}|{v}|{q}|{g}']) for g in GROUPS) + ' |')
    L += ['', '## $D(e)$（全部）', '', '| 走らせたもの | ' + ' | '.join(f'ep{e}' for e in sorted(R[RUNS1[0]][0])) + ' |',
          '|---|' + '---|' * len(R[RUNS1[0]][0])]
    for run in RUNS1:
        L.append(f'| {run} | ' + ' | '.join(f"{S[f'{run}|{e}|D|all']['mean']:+.3f}" for e in sorted(R[run][0])) + ' |')
    L += ['', '| 走らせたもの | ' + ' | '.join(f'ep{e}' for e in sorted(R[RUNS2[0]][0])) + ' |', '|---|' + '---|' * len(R[RUNS2[0]][0])]
    for run in RUNS2:
        L.append(f'| {run} | ' + ' | '.join(f"{S[f'{run}|{e}|D|all']['mean']:+.3f}" for e in sorted(R[run][0])) + ' |')
    L += ['', '## 自己の見張り（A-1 の視界の L1、$C_{self}$）', '', '| 走らせたもの | エポック | L1 | $C_{self}$ |', '|---|---|---|---|']
    for run in RUNS1 + RUNS2:
        for e in sorted(R[run][0]):
            if f'{run}|{e}|c_self' in S:
                L.append(f"| {run} | {e} | {S[f'{run}|{e}|self_l1']:.5f} | {fm(S[f'{run}|{e}|c_self'])} |")
    L += ['', '## 生成器の出力の大きさ（範囲 0〜7、ステップ 52、S。raw = G の出力、in = 足した層とマスクの後）', '']
    for run in RUNS1:
        for e, v in q2[run].items():
            L.append(f"- {run} ep{e}: raw {v['raw']} / in {v['in']}")
    L += ['', '## 規則（§2.3）と §2.2 の当てはめ（記述）', '', '```', json.dumps(RU, indent=1, ensure_ascii=False, default=str), '```', '',
          '## 時間', '', '```', json.dumps(times, indent=1), '```']
    open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(L))
    print('\n'.join(L))
    figures(R, S)


def figures(R, S):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    col = {'p50': 'C3', 'sr50': 'C0', 'none': 'C7'}
    fig, axs = plt.subplots(3, 4, figsize=(17, 9.5), sharey='row')
    panels = [('stage1', 'M'), ('stage1', 'N'), ('stage2', '0.1'), ('stage2', '0.01')]
    for j, (st, v) in enumerate(panels):
        for i, g in enumerate(GROUPS):
            ax = axs[i, j]
            for c in CONDS:
                run = f'stage1_{c}_{v}' if st == 'stage1' else f'stage2_{c}_r{v}'
                es = sorted(R[run][0])
                m = [S[f'{run}|{e}|D|{g}']['mean'] for e in es]
                lo = [S[f'{run}|{e}|D|{g}']['ci'][0] for e in es]
                hi = [S[f'{run}|{e}|D|{g}']['ci'][1] for e in es]
                ax.plot(es, m, '-o', color=col[c], ms=3, label=c)
                ax.fill_between(es, lo, hi, color=col[c], alpha=0.15)
            ax.axhline(0, color='0.3', lw=0.8)
            if st == 'stage1':
                ax.axvspan(5, 25, color='0.9', zorder=0)
            ax.set_title(f'{st} {"variant " + v if st == "stage1" else "LSTM lr x" + v}, D ({g})', fontsize=9)
            ax.set_xlabel('epoch')
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round25_D.{ext}'), dpi=120)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for c in CONDS:
        for r, ls in (('0.1', '-'), ('0.01', ':')):
            run = f'stage2_{c}_r{r}'
            es = sorted(R[run][0])
            axs[0].plot(es, [S[f'{run}|{e}|self_l1'] for e in es], ls, marker='o', ms=3, color=col[c], label=f'{c} x{r}')
            axs[1].plot(es, [S[f'{run}|{e}|c_self']['mean'] for e in es], ls, marker='o', ms=3, color=col[c], label=f'{c} x{r}')
    axs[0].set_title('stage 2: L1 of the predicted A-1 view (test episodes)', fontsize=9)
    axs[1].set_title('stage 2: C_self (round 21, hold)', fontsize=9)
    for ax in axs:
        ax.set_xlabel('stage-2 epoch')
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round25_self.{ext}'), dpi=120)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.4), sharey=True)
    for ax, c in zip(axs, CONDS):
        for k, (v, hatch) in enumerate((('M', ''), ('N', '//'))):
            run = f'stage1_{c}_{v}'
            e = max(R[run][0])
            ax.bar(np.arange(8) + (k - 0.5) * 0.4, R[run][0][e]['q2_in'], 0.4, label=f'{v} ep{e} (after A and mask)',
                   hatch=hatch, color=col[c], alpha=0.8 if v == 'M' else 0.4)
        ax.set_xticks(range(8))
        ax.set_xlabel('range')
        ax.set_title(f'{c}: mean |pleasure into the LSTM| at step 52', fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round25_q2.{ext}'), dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()
