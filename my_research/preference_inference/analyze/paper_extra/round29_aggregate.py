"""
Paper round 29 aggregation (docs/paper_extra_log.md Sec.53.2; quantities and reading rules as pre-registered there).
CPU only. Reads Round29/{s1,s2,s3,s4}/ and writes Round29/analysis/{s<k>_summary.json, s<k>_tables.md, fig_round29_s<k>.*}.
Usage: python analyze/paper_extra/round29_aggregate.py s1|s2|s3|s4|all
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round21_aggregate as A  # noqa  (stat = mean, pair t CI, two-stage bootstrap)
import vdir_stageA as VA  # noqa
import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

_PI = os.path.dirname(os.path.dirname(_HERE))
R29 = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round29')
OUT = os.path.join(R29, 'analysis')
CONDS1 = ('p50', 'none', 'sr50', 'p80')
CN = {'p50': '経験50%', 'none': '経験なし', 'sr50': '符号無作為（経験50%）', 'p80': '経験80%（記述）'}
COL = {'p50': 'C3', 'none': 'C7', 'sr50': 'C0', 'p80': 'C1'}
PAIRS = tuple(range(6))
EPS = (160, 170, 180, 190, 200)
NB = 8
DEG = [-157.5 + 45 * b for b in range(NB)]


def cls(st):
    t, b = st['t']['ci95'], st['boot']
    if t is None or b is None:
        return '—'
    if t[0] > 0 and b[0] > 0:
        return '>0'
    if t[1] < 0 and b[1] < 0:
        return '<0'
    if t[0] <= 0 <= t[1] and b[0] <= 0 <= b[1]:
        return '≈0'
    return '食い違う'


def tsum(v):
    v = [x for x in v if np.isfinite(x)]
    if len(v) >= 2 and np.std(v) == 0:
        return {'mean': float(v[0]), 'ci95': [float(v[0]), float(v[0])], 'n': len(v), 'sd': 0.0}
    return VA.tsum(v)


def stat(X):
    st = A.stat(X)
    st['t'] = tsum(st['per_pair'])
    return st


def fmt(st, d=3):
    t, b = st['t']['ci95'], st['boot']
    return f"{st['mean']:+.{d}f} [{t[0]:+.{d}f}, {t[1]:+.{d}f}] [{b[0]:+.{d}f}, {b[1]:+.{d}f}]"


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=110)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    plt.close(fig)


# ================================================================== Sec.1
def s1():
    X, curves = {}, {}
    for c in CONDS1:
        per = {}
        for s in PAIRS:
            acc = {}
            for e in EPS:
                z = np.load(os.path.join(R29, 's1', f'{c}_s{s}_ep{e}.npz'))
                mi = np.stack([z[f'd{b}_img'][:, b] - z['zero_img'][:, b] for b in range(NB)], 1)      # (N, 8)
                mf = np.stack([z[f'd{b}_feat'][:, b] - z['zero_feat'][:, b] for b in range(NB)], 1)
                gi = np.stack([z[f'd{b}_img'][:, b] for b in range(NB)], 1)
                with np.errstate(all='ignore'):
                    q = {'img': np.nanmean(mi, 1), 'feat': np.nanmean(mf, 1), 'gain': np.nanmean(gi, 1)}
                for b in range(NB):
                    q[f'img_b{b}'], q[f'feat_b{b}'] = mi[:, b], mf[:, b]
                for k, v in q.items():
                    acc.setdefault(k, []).append(v)
                if c == 'p50' and e == 200:
                    # displacement along u_b relative to zero input, per k (k <= KE)
                    p2 = None
                    KE = z['KE']
                    est0 = z['zero_est']
                    dk = np.full((est0.shape[0], NB, est0.shape[1]), np.nan)
                    for b in range(NB):
                        u = np.array([np.cos(np.deg2rad(DEG[b])), np.sin(np.deg2rad(DEG[b]))])
                        dy = ((z[f'd{b}_est'] - est0) * u).sum(-1)
                        ok = np.arange(1, est0.shape[1] + 1)[None] <= KE[:, b:b + 1]
                        dk[:, b] = np.where(ok, dy, np.nan)
                    with np.errstate(all='ignore'):
                        curves.setdefault('p50', []).append(np.nanmean(dk, (0, 1)))
            for k, v in acc.items():
                with np.errstate(all='ignore'):
                    per.setdefault(k, []).append(np.nanmean(np.stack(v), 0))
        X[c] = {k: np.stack(v) for k, v in per.items()}
    S = {}
    for c in CONDS1:
        for k, v in X[c].items():
            S[f'{c}|{k}'] = stat(v)
    for o in ('none', 'sr50'):
        for k in ('img', 'feat'):
            S[f'p50-{o}|{k}'] = stat(X['p50'][k] - X[o][k])
    ci, cf = cls(S['p50|img']), cls(S['p50|feat'])
    row = 1 if ci == '>0' else (2 if ci == '≈0' and cf == '>0' else (3 if ci == '≈0' and cf == '≈0' else None))
    rd = {'M_img': ci, 'M_feat': cf, 'row': row,
          'by_condition': {c: {'M_img': cls(S[f'{c}|img']), 'M_feat': cls(S[f'{c}|feat'])} for c in CONDS1},
          'directions_img': {c: [cls(S[f'{c}|img_b{b}']) for b in range(NB)] for c in CONDS1},
          'directions_feat': {c: [cls(S[f'{c}|feat_b{b}']) for b in range(NB)] for c in CONDS1}}
    out = {'reading': rd, 'stats': S, 'p50_disp_curve_ep200_pairs': [v.tolist() for v in curves.get('p50', [])]}
    json.dump(out, open(os.path.join(OUT, 's1_summary.json'), 'w'), indent=1, ensure_ascii=False)
    L = ['# 第29弾 §1 過程2の運動の欄に動きを入れたときの予測（学習なし）', '',
         '値：平均 [組の t の 95% CI] [2段のブートストラップの 95% CI]、6 組。$M$ = 方向 $u_b$ の動きを入れたとき − 零の入力（1 歩あたり。入れたとおりに動けば img は 1）。', '',
         '## 読み方の表の当てはめ', '', '```', json.dumps(rd, ensure_ascii=False, indent=1), '```', '',
         '## 方向の平均', '', '| 条件・差 | $M_{img}$ | 判定 | $M_{feat}$ | 判定 | 入力での傾きそのもの（img） |', '|---|---|---|---|---|---|']
    for c in CONDS1:
        L.append(f"| {CN[c]} | {fmt(S[f'{c}|img'])} | {cls(S[f'{c}|img'])} | {fmt(S[f'{c}|feat'])} | {cls(S[f'{c}|feat'])} | {fmt(S[f'{c}|gain'])} |")
    for o in ('none', 'sr50'):
        L.append(f"| 経験50% − {CN[o]} | {fmt(S[f'p50-{o}|img'])} | {cls(S[f'p50-{o}|img'])} | {fmt(S[f'p50-{o}|feat'])} | {cls(S[f'p50-{o}|feat'])} | — |")
    L += ['', '## 方向ごと（範囲 b の向き）', '', '| 条件 | 量 | ' + ' | '.join(f'b{b}（{DEG[b]:+.1f}°）' for b in range(NB)) + ' |',
          '|---|---|' + '---|' * NB]
    for c in CONDS1:
        for q in ('img', 'feat'):
            L.append(f'| {CN[c]} | {q} | ' + ' | '.join(f"{S[f'{c}|{q}_b{b}']['mean']:+.3f} {cls(S[f'{c}|{q}_b{b}'])}" for b in range(NB)) + ' |')
    L += ['', '- 方向ごとの CI の全部は `s1_summary.json`。',
          '- 「反対の方向」の入力は方向 b+4 の入力そのもの。$u_b$ の向きに測ると、作りから −M(b+4) になる。']
    if curves.get('p50'):
        cm = np.nanmean(np.stack(curves['p50']), 0)
        L += ['', '## 経験50%・ep200：零の入力との差のずれ（$u_b$ の向き、k 歩目、組と方向の平均。理想は k）', '',
              '| k | ' + ' | '.join(str(k) for k in range(1, len(cm) + 1)) + ' |', '|---|' + '---|' * len(cm),
              '| ずれ | ' + ' | '.join(f'{v:+.2f}' for v in cm) + ' |']
    open(os.path.join(OUT, 's1_tables.md'), 'w').write('\n'.join(L) + '\n')
    # figure
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    for ai, q in enumerate(('img', 'feat')):
        ax = axs[ai]
        for ci_, c in enumerate(CONDS1):
            xs = np.arange(NB + 1) + (ci_ - 1.5) * 0.18
            for j, key in enumerate([f'{q}_b{b}' for b in range(NB)] + [q]):
                st = S[f'{c}|{key}']
                ax.scatter([xs[j]] * len(st['per_pair']), st['per_pair'], s=8, color=COL[c], alpha=0.5)
                t = st['t']['ci95']
                ax.errorbar(xs[j], st['mean'], yerr=[[st['mean'] - t[0]], [t[1] - st['mean']]], fmt='s', ms=4,
                            color=COL[c], label=c if j == 0 else None)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(range(NB + 1))
        ax.set_xticklabels([f'b{b}' for b in range(NB)] + ['mean'])
        ax.set_title(f'M_{q} per direction (points = pairs, square = mean +- t CI)', fontsize=9)
        ax.legend(fontsize=7)
    ax = axs[2]
    if curves.get('p50'):
        for v in curves['p50']:
            ax.plot(range(1, len(v) + 1), v, color='C3', alpha=0.4, lw=0.8)
        ax.plot(range(1, len(cm) + 1), cm, color='C3', lw=2, label='p50 mean (ep200)')
        ax.plot([1, len(cm)], [1, len(cm)], 'k--', lw=0.8, label='ideal (k steps)')
    ax.set_xlabel('frozen step k')
    ax.set_title('p50: displacement along u_b vs zero input (img)', fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle('round 29 Sec.1: motion into process-2 motion slot (15 frozen steps)')
    fig.tight_layout()
    save_fig(fig, 'fig_round29_s1')
    print(json.dumps(rd, ensure_ascii=False))


# ================================================================== Sec.2
def s2():
    S, R = {}, {}
    run = json.load(open(os.path.join(R29, 's2', 's2_run.json')))
    for c in ('p50', 'p80', 'none', 'sr50'):
        z = np.load(os.path.join(R29, 's2', f'{c}.npz'))
        tile = lambda v: np.tile(v[None], (len(PAIRS), 1))  # noqa
        for lv in ('chance', 'freq', 'best'):
            S[f'{c}|{lv}'] = stat(tile(z[lv]))
        for kind in ('mlp', 'lin'):
            acc = np.stack([z[f'{kind}_s{s}_acc'] for s in PAIRS])
            S[f'{c}|{kind}|acc'] = stat(acc)
            S[f'{c}|{kind}|acc-freq'] = stat(acc - tile(z['freq']))
            S[f'{c}|{kind}|acc-chance'] = stat(acc - tile(z['chance']))
            S[f'{c}|{kind}|best-acc'] = stat(tile(z['best']) - acc)
            R[f'{c}|{kind}'] = np.mean(np.stack([z[f'{kind}_s{s}_react'] for s in PAIRS]), 0)
    def row_of(c, kind):
        a, f, b = S[f'{c}|{kind}|acc']['mean'], S[f'{c}|freq']['mean'], S[f'{c}|best']['mean']
        d = cls(S[f'{c}|{kind}|acc-freq'])
        ratio = (b - a) / (b - f) if b - f > 0 else None
        if d == '>0':
            r = 1 if ratio is not None and ratio <= 0.2 else 2
        elif d == '≈0':
            r = 3
        else:
            r = None
        return {'acc_minus_freq': d, 'gap_ratio': ratio, 'row': r}
    rd = {f'{c}|{k}': row_of(c, k) for c in ('p50', 'p80', 'none', 'sr50') for k in ('mlp', 'lin')}
    rd['main'] = rd['p50|mlp']
    out = {'reading': rd, 'stats': S, 'react': {k: v.tolist() for k, v in R.items()}, 'run': run}
    json.dump(out, open(os.path.join(OUT, 's2_summary.json'), 'w'), indent=1, ensure_ascii=False)
    L = ['# 第29弾 §2 快・不快 → 次の自分の動き（8 つの扇の正答率、検査用 300 エピソード）', '',
         '値：平均 [種の t の 95% CI] [2段のブートストラップの 95% CI]（種 → エピソード）。水準は種によらない。', '',
         '## 読み方の表の当てはめ', '', '```', json.dumps(rd, ensure_ascii=False, indent=1), '```', '',
         '| 条件 | 模型 | 正答率 | 偶然 1/8 | 頻度だけ | 最良（規則の神託） | 正答率 − 頻度 | 判定 | （最良 − 正答率）/（最良 − 頻度） |',
         '|---|---|---|---|---|---|---|---|---|']
    for c in ('p50', 'p80', 'none', 'sr50'):
        for k in ('mlp', 'lin'):
            g = rd[f'{c}|{k}']['gap_ratio']
            L.append(f"| {CN.get(c, c)} | {k} | {fmt(S[f'{c}|{k}|acc'])} | {S[f'{c}|chance']['mean']:.3f} | {S[f'{c}|freq']['mean']:.3f} | "
                     f"{S[f'{c}|best']['mean']:.3f} | {fmt(S[f'{c}|{k}|acc-freq'])} | {cls(S[f'{c}|{k}|acc-freq'])} | "
                     f"{'—' if g is None else f'{g:.3f}'} |")
    L += ['', '## 反応（範囲 b に +0.25 を入れたとき、扇 b へ動く確率。小さなネットワーク、6 種の平均）', '',
          '| 条件 | ' + ' | '.join(f'b{b}' for b in range(NB)) + ' |', '|---|' + '---|' * NB]
    for c in ('p50', 'p80', 'none', 'sr50'):
        r = R[f'{c}|mlp']
        L.append(f'| {CN.get(c, c)} | ' + ' | '.join(f'{r[b, b]:.3f}' for b in range(NB)) + ' |')
    L += ['', '- 範囲 b に −0.25 のときの、扇 b へ動く確率：']
    for c in ('p50', 'p80', 'none', 'sr50'):
        r = R[f'{c}|mlp']
        L.append(f'  - {CN.get(c, c)}：' + '、'.join(f'{r[NB + b, b]:.3f}' for b in range(NB)))
    L += ['', f"- 学習用の扇の頻度（経験50%）：{', '.join(f'{v:.3f}' for v in run['p50']['freq'])}。",
          f"- 規則の神託の確かめ（検査用の規則の歩で神託の 1 歩が作れた割合）：" +
          '、'.join(f"{c} {run[c]['oracle_check']['rule_ok_on_rule_steps']}" for c in ('p50', 'p80', 'none', 'sr50'))]
    open(os.path.join(OUT, 's2_tables.md'), 'w').write('\n'.join(L) + '\n')
    fig, axs = plt.subplots(1, 5, figsize=(19, 3.8))
    for i, c in enumerate(('p50', 'p80', 'none', 'sr50')):
        ax = axs[i]
        im = ax.imshow(R[f'{c}|mlp'][:NB], vmin=0, vmax=1, cmap='viridis')
        ax.set_xlabel('sector of the next step')
        ax.set_ylabel('+0.25 in range b')
        ax.set_title(f'{c}: P(sector | +0.25 in b), MLP', fontsize=9)
    fig.colorbar(im, ax=axs[3], fraction=0.046)
    ax = axs[4]
    for i, c in enumerate(('p50', 'p80', 'none', 'sr50')):
        for j, k in enumerate(('mlp', 'lin')):
            st = S[f'{c}|{k}|acc']
            x = i + (j - 0.5) * 0.3
            ax.scatter([x] * 6, st['per_pair'], s=8, color='C3' if k == 'mlp' else 'C0')
        ax.hlines([S[f'{c}|freq']['mean']], i - 0.4, i + 0.4, color='k', lw=1)
        ax.hlines([S[f'{c}|best']['mean']], i - 0.4, i + 0.4, color='g', lw=1, linestyles='--')
    ax.axhline(1 / 8, color='gray', lw=0.6, ls=':')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['p50', 'p80', 'none', 'sr50'])
    ax.set_title('accuracy (red MLP, blue linear); black = frequency, green dashed = rule oracle', fontsize=8)
    fig.tight_layout()
    save_fig(fig, 'fig_round29_s2')
    print(json.dumps(rd['main'], ensure_ascii=False))


# ================================================================== Sec.3
ROWS3 = ('own', 'other_real', 'enc2_real', 'pred_show', 'pred_freeze')
RN3 = {'own': '自分の視界', 'other_real': '実際の他者の視点（Encoder-1(A-2 の視界)）', 'enc2_real': 'Encoder-2(A-1 の視界)（記述）',
       'pred_show': '予測した他者の視点（見せる、主）', 'pred_freeze': '予測した他者の視点（止める、記述）'}
BINSETS = ['c0167', 'c2345'] + [f'b{b}' for b in range(NB)]


def r_r2(st):
    n, sx, sy, sxx, syy, sxy, sse = st
    vx, vy = n * sxx - sx * sx, n * syy - sy * sy
    r = (n * sxy - sx * sy) / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan
    sst = syy - sy * sy / n
    return r, (1 - sse / sst) if sst > 0 else np.nan


def stat3(Ps):
    """Ps list over pairs of (N, 7) per-episode sufficient statistics -> r / R2 with pair t CI and two-stage bootstrap."""
    vals = np.array([r_r2(P.sum(0)) for P in Ps])                    # (pairs, 2)
    rng = np.random.default_rng(0)
    np_, N = len(Ps), Ps[0].shape[0]
    bs = np.empty((1000, 2))
    for i in range(1000):
        tot = np.zeros(7)
        for p in rng.integers(0, np_, np_):
            tot += Ps[p][rng.integers(0, N, N)].sum(0)
        bs[i] = r_r2(tot)
    out = {}
    for j, nm in enumerate(('r', 'R2')):
        v = vals[:, j]
        out[nm] = {'mean': float(np.nanmean(v)), 't': tsum(v.tolist()), 'per_pair': v.tolist(),
                   'boot': [float(np.nanpercentile(bs[:, j], 2.5)), float(np.nanpercentile(bs[:, j], 97.5))]
                   if np.isfinite(bs[:, j]).any() else None}
    return out


def s3():
    S = {}
    for c in ('p50', 'none', 'sr50'):
        Z = [np.load(os.path.join(R29, 's3', f'{c}_s{s}.npz')) for s in PAIRS]
        for rn in ROWS3:
            for bn in BINSETS:
                S[f'{c}|{rn}|{bn}'] = stat3([z[f'{rn}|{bn}'] for z in Z])
    def fm(st, d=3):
        if st['boot'] is None or st['t']['ci95'] is None or not np.isfinite(st['mean']):
            return '—'
        return fmt(st, d)
    real, pred = S['p50|other_real|c0167']['r'], S['p50|pred_show|c0167']['r']
    hi = real['mean'] >= 0.7
    pp = cls(pred) == '>0'
    row = 1 if hi and pp else (2 if hi and not pp else (3 if not hi and not pp else None))
    rd = {'other_real_r': real['mean'], 'other_real_high': bool(hi), 'pred_show_r': pred['mean'], 'pred_show_cls': cls(pred),
          'row': row}
    out = {'reading': rd, 'stats': S}
    json.dump(out, open(os.path.join(OUT, 's3_summary.json'), 'w'), indent=1, ensure_ascii=False)
    L = ['# 第29弾 §3 自分の経験だけで学んだ読み出し（Encoder-1 → 赤 − 緑）を他者の視点に当てはめる', '',
         '値：平均 [組の t の 95% CI] [2段のブートストラップの 95% CI]。検査用 300 エピソードの全時点をまとめた相関と $R^2$。', '',
         '## 読み方の表の当てはめ（経験50%、範囲 0・1・6・7）', '', '```', json.dumps(rd, ensure_ascii=False, indent=1), '```', '']
    for bn, title in (('c0167', '範囲 0・1・6・7'), ('c2345', '範囲 2〜5')):
        L += [f'## {title}', '', '| 条件 | 当てはめた先 | 相関 r | 判定 | $R^2$ |', '|---|---|---|---|---|']
        for c in ('p50', 'none', 'sr50'):
            for rn in ROWS3:
                st = S[f'{c}|{rn}|{bn}']
                L.append(f"| {CN[c]} | {RN3[rn]} | {fm(st['r'])} | {cls(st['r']) if np.isfinite(st['r']['mean']) else '—'} | {fm(st['R2'])} |")
        L.append('')
    L += ['## 範囲ごとの相関（経験50%）', '', '| 当てはめた先 | ' + ' | '.join(f'b{b}' for b in range(NB)) + ' |', '|---|' + '---|' * NB]
    for rn in ROWS3:
        L.append(f'| {RN3[rn]} | ' + ' | '.join(
            ('—' if not np.isfinite(S[f'p50|{rn}|b{b}']['r']['mean']) else f"{S[f'p50|{rn}|b{b}']['r']['mean']:+.3f}") for b in range(NB)) + ' |')
    open(os.path.join(OUT, 's3_tables.md'), 'w').write('\n'.join(L) + '\n')
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for ai, bn in enumerate(('c0167', 'c2345')):
        ax = axs[ai]
        for ci_, c in enumerate(('p50', 'none', 'sr50')):
            for j, rn in enumerate(ROWS3):
                st = S[f'{c}|{rn}|{bn}']['r']
                if not np.isfinite(st['mean']):
                    continue
                x = j + (ci_ - 1) * 0.22
                ax.scatter([x] * 6, st['per_pair'], s=8, color=COL[c], alpha=0.6)
                ax.plot(x, st['mean'], 's', color=COL[c], label=c if j == 0 else None)
        ax.axhline(0, color='k', lw=0.6)
        ax.axhline(0.7, color='g', lw=0.6, ls='--')
        ax.set_xticks(range(len(ROWS3)))
        ax.set_xticklabels(['own', 'other real', 'Enc-2 real', 'pred (show)', 'pred (freeze)'], fontsize=8)
        ax.set_title(f'correlation of the own-experience read-out, ranges {bn}', fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, 'fig_round29_s3')
    print(json.dumps(rd, ensure_ascii=False))


# ================================================================== Sec.4
NLIST = {'p05': [1, 2, 5, 10, 20, 50, 100, 101], 'p08': [1, 2, 5, 10, 20, 50]}


def s4():
    out = {'run': json.load(open(os.path.join(R29, 's4', 's4_run.json')))}
    L = ['# 第29弾 §4 理想の観察者が好みを見分けるまでの歩数', '',
         '正答率：はじめから n 歩の対数尤度比の和の符号（0 は 0.5）。CI はエピソードのブートストラップ（1000 回）。', '']
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.4))
    for name in ('p05', 'p08'):
        z = np.load(os.path.join(R29, 's4', f'{name}.npz'))
        nst = z['llr72'].shape[1]
        out[name] = {}
        for nb in (72, 8):
            llr = z[f'llr{nb}']
            cum = np.cumsum(llr, 1)
            sc = (cum > 0) + 0.5 * (cum == 0)                         # (N, nst)
            acc = sc.mean(0)
            rng = np.random.default_rng(0)
            Nn = sc.shape[0]
            bi = rng.integers(0, Nn, (1000, Nn))
            bacc = np.stack([sc[i].mean(0) for i in bi])               # (1000, nst)
            first = lambda a, th: int(np.argmax(a >= th) + 1) if (a >= th).any() else None  # noqa
            post = 1 / (1 + np.exp(-cum))
            ep = {}
            for th in (0.9, 0.95):
                hit = post >= th
                fn = np.where(hit.any(1), hit.argmax(1) + 1, np.nan)
                ep[str(th)] = {'median': float(np.nanmedian(fn)) if np.isfinite(fn).any() else None,
                               'q25': float(np.nanpercentile(fn, 25)) if np.isfinite(fn).any() else None,
                               'q75': float(np.nanpercentile(fn, 75)) if np.isfinite(fn).any() else None,
                               'never_frac': float(np.isnan(fn).mean())}
            nb95 = [first(b, 0.95) for b in bacc]
            nb90 = [first(b, 0.9) for b in bacc]
            ok = z['info_ok']
            arr = z['arrived']
            bits = llr / np.log(2)
            info = {'all': float(bits[ok].mean()), 'before_arrival': float(bits[ok & ~arr].mean()) if (ok & ~arr).any() else None,
                    'after_arrival': float(bits[ok & arr].mean()) if (ok & arr).any() else None,
                    'frac_steps_after_arrival': float((ok & arr).sum() / ok.sum()),
                    'frac_episodes_arrived': float(arr[:, -1].mean())}
            out[name][str(nb)] = {
                'acc_at_n': {str(n): {'acc': float(acc[n - 1]), 'ci': [float(np.percentile(bacc[:, n - 1], 2.5)),
                                                                     float(np.percentile(bacc[:, n - 1], 97.5))]}
                             for n in NLIST[name] if n <= nst},
                'n90': first(acc, 0.9), 'n95': first(acc, 0.95),
                'n95_boot': [float(np.percentile([v if v is not None else np.inf for v in nb95], q)) for q in (2.5, 97.5)],
                'n90_boot': [float(np.percentile([v if v is not None else np.inf for v in nb90], q)) for q in (2.5, 97.5)],
                'per_episode_posterior': ep, 'info_bits_per_step': info, 'n_steps': nst, 'n_episodes': int(Nn)}
            ax.plot(np.arange(1, nst + 1), acc, ls='-' if nb == 72 else '--', color='C3' if name == 'p05' else 'C0',
                    label=f"p_rule {'0.5' if name == 'p05' else '0.8'}, {360 // nb} deg bins")
    r = out['p05']['72']
    out['reading'] = {'n95': r['n95'], 'half_length': 50.5, 'row': 1 if r['n95'] is not None and r['n95'] <= 50.5 else 2}
    ax.axhline(0.95, color='k', lw=0.6, ls=':')
    ax.axhline(0.9, color='gray', lw=0.6, ls=':')
    ax.axvline(50.5, color='g', lw=0.6, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('observed steps n')
    ax.set_ylabel('proportion correct')
    ax.set_ylim(0.45, 1.01)
    ax.legend(fontsize=7)
    ax.set_title('round 29 Sec.4: ideal observer of A-2 preference', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'fig_round29_s4')
    json.dump(out, open(os.path.join(OUT, 's4_summary.json'), 'w'), indent=1, ensure_ascii=False)
    L += ['## 読み方の表の当てはめ（`p_rule` 0.5、5° の区分）', '', '```', json.dumps(out['reading'], ensure_ascii=False), '```', '']
    for name, title in (('p05', '`p_rule` 0.5（v11_s2_pref、3300 エピソード、101 歩）'), ('p08', '`p_rule` 0.8（第20弾 S、600 エピソード、歩 0〜49）')):
        L += [f'## {title}', '']
        for nb in ('72', '8'):
            r = out[name][nb]
            L += [f'### 区分 {360 // int(nb)}°', '', '| n | ' + ' | '.join(r['acc_at_n']) + ' |', '|---|' + '---|' * len(r['acc_at_n']),
                  '| 正答率 | ' + ' | '.join(f"{v['acc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" for v in r['acc_at_n'].values()) + ' |', '',
                  f"- 90% に届く n：{r['n90']}（ブートストラップ {r['n90_boot']}）。95%：{r['n95']}（{r['n95_boot']}）。",
                  f"- エピソードごとの事後確率 0.9／0.95 に届く歩数：{json.dumps(r['per_episode_posterior'])}。",
                  f"- 1 歩あたりの情報量（ビット）：{json.dumps(r['info_bits_per_step'])}。", '']
    L += [f"- 規則の歩の 1 歩が本当の好みの規則の 1 歩と一致する割合：p05 {out['run']['p05']['rule_step_equals_true_rule_frac']:.4f}、"
          f"p08 {out['run']['p08']['rule_step_equals_true_rule_frac']:.4f}。"]
    open(os.path.join(OUT, 's4_tables.md'), 'w').write('\n'.join(L) + '\n')
    print(json.dumps(out['reading']))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    w = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for k, f in (('s1', s1), ('s2', s2), ('s3', s3), ('s4', s4)):
        if w in (k, 'all'):
            f()
