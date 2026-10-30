"""
Paper round 12, V-dir stage A (docs/paper_extra_log.md Sec.36; exploratory): markdown tables from Vdir_A/summary.json
(vdir_stageA.py) only; no computation beyond formatting.
-> data/result/paper_extra/Vdir_A/tables.md
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
D = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Vdir_A')
S = json.load(open(os.path.join(D, 'summary.json')))
CONDS = ('none', 'p50', 'p80', 'sr80', 'sr50', 'sw')
NAME = {'none': '経験なし', 'p50': '経験50%', 'p80': '経験80%', 'sr80': '符号無作為（80%）', 'sr50': '符号無作為（50%）',
        'sw': '好み切替（組1〜3）'}
WINS = ('1-25', '26-50', '51-100', '1-50', '1-100', 'past')
WN = {'1-25': '1〜25', '26-50': '26〜50', '51-100': '51〜100', '1-50': '1〜50', '1-100': '1〜100', 'past': '過去の向き'}
TAUS = ('0.0', '0.5', '1.0', '2.0')
SETS = ('pref', 'green', 'pref_a2p80', 'green_a2p80')
SN = {'oracle': 'oracle', 'cpm': 'c±'}


def f(x, n=4):
    return '—' if x is None else f'{x:+.{n}f}'


def ci(t, n=4):
    if t is None:
        return '—'
    if t.get('mean') is None:
        return '推定不可'
    if not t.get('ci95'):
        return f"{f(t['mean'], n)}（n={t.get('n')}）"
    return f"{f(t['mean'], n)} [{f(t['ci95'][0], n)}, {f(t['ci95'][1], n)}]"


def pc(p, n=4):
    return '—' if not p else f'[{f(p[0], n)}, {f(p[1], n)}]'


def nexcl(cis):
    """number of pairs whose episode-bootstrap CI excludes 0 (above / below)."""
    up = sum(1 for c in cis if c and c[0] > 0)
    dn = sum(1 for c in cis if c and c[1] < 0)
    return f'+{up}/−{dn}'


def coef_table(L, series, err, st, tau, wins=WINS, model='main'):
    L.append(f'\n#### {SN[series]} 系列・{err}・{st}・τ = {tau}・{model}\n')
    L.append('| 条件 | 区間 | n | $b_{pref}$ t CI | $b_{dir}$ t CI | $b_{red}$ t CI | $b_{dir}-b_{pref}$ t CI | '
             '$b_{dir}$ 2段 | $b_{dir}-b_{pref}$ 2段 | $b_{dir}$ 組内ブートで 0 を外す組 | $b_{pref}$ 同 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for cd in CONDS:
        for w in wins:
            b = S['coef'].get(f'{series}|{cd}|{err}|{st}|{w}|{tau}|{model}')
            if b is None:
                L.append(f'| {NAME[cd]} | {WN[w]} | 未保存 | | | | | | | | |')
                continue
            t = b['t']
            b2 = b.get('boot2', {})
            bp = b.get('boot_pair', {})
            L.append(f"| {NAME[cd]} | {WN[w]} | {b['n_used']} | {ci(t.get('pref'))} | {ci(t['dir'])} | {ci(t['red'])} | "
                     f"{ci(t.get('dir_minus_pref'))} | {pc(b2.get('dir'))} | {pc(b2.get('dir_minus_pref'))} | "
                     f"{nexcl(bp.get('dir', [])) if bp else '—'} | {nexcl(bp.get('pref', [])) if bp.get('pref') else '—'} |")


def main():
    L = ['# V-dir 段階A 表（探索的（事前登録外）。`analyze/paper_extra/vdir_stageA.py`、§36）\n']
    a = S['axis']
    L.append(f"- $u_A$ = ({a['u_A'][0]:+.4f}, {a['u_A'][1]:+.4f})、$u_B$ = ({a['u_B'][0]:+.4f}, {a['u_B'][1]:+.4f})、"
             f"$\\hat n$ = ({a['n_hat'][0]:+.4f}, {a['n_hat'][1]:+.4f})")
    L.append(f"- 赤の側（配置ごと）：{a['red_dir_by_layout']['pref']}")
    L.append('- 確認：' + json.dumps(S['checks'], ensure_ascii=False))
    L.append(f"- 未保存（系列|条件|誤差|データ|組）：{len(S['missing'])} 件：" + '、'.join(S['missing']))
    # counts
    L.append('\n## 件数（好み × 配置 × 向き A/B/向きなし）\n')
    for st in SETS:
        L.append(f'\n#### {st}\n')
        L.append('| 区間 | τ | 赤好き・配置0 | 赤好き・配置1 | 緑好き・配置0 | 緑好き・配置1 | 向きあり | 向きなし | 好みと向きの一致率 | 赤の側と向きの一致率 |')
        L.append('|---|---|---|---|---|---|---|---|---|---|')
        for w in WINS:
            for tau in TAUS:
                c = S['counts'][f'{st}|{w}|{tau}']
                cc = ' | '.join(f"{c['cells'][k]['A']}/{c['cells'][k]['B']}/{c['cells'][k]['none']}"
                                for k in ('red|L0', 'red|L1', 'green|L0', 'green|L1'))
                L.append(f"| {WN[w]} | {tau} | {cc} | {c['n_dir']} | {c['n_none']} | {f(c['pref_dir_agree'], 3)} | "
                         f"{f(c['red_dir_agree'], 3)} |")
    # correlations
    L.append('\n## 相関（Pearson、300 エピソード、τ = 0 の向き）\n')
    for st in ('pref', 'pref_a2p80'):
        for w in ('1-50', '1-25', '26-50', '51-100', '1-100'):
            c = S['corr'][f'{st}|{w}']
            L.append(f'\n#### {st}・{WN[w]}\n')
            L.append('| | ' + ' | '.join(c['vars']) + ' |')
            L.append('|---' * (len(c['vars']) + 1) + '|')
            for v, row in zip(c['vars'], c['r']):
                L.append(f'| {v} | ' + ' | '.join('—' if x is None else f'{x:+.2f}' for x in row) + ' |')
    # VIF
    L.append('\n## VIF\n')
    L.append('| データ | 区間 | τ | 形 | VIF |')
    L.append('|---|---|---|---|---|')
    for st in SETS:
        for w in WINS:
            for tau in TAUS:
                for model in ('main', 'cov', 'inter', 'cont'):
                    v = S['vif'].get(f'{st}|{w}|{tau}|{model}')
                    if v is None:
                        continue
                    L.append(f'| {st} | {WN[w]} | {tau} | {model} | ' + '、'.join(f'{k} {x:.2f}' for k, x in v.items()) + ' |')
    # coefficients
    L.append('\n## 係数（主の形）。t CI＝学習の変動（組の値の t、同じ 300 エピソード）；2段＝組とエピソードの2段ブートストラップ（両方、参考）；'
             '組内ブート＝組ごとのエピソードのブートストラップ（評価エピソードの変動）の CI が 0 を上に外す組 / 下に外す組\n')
    for series in ('oracle', 'cpm'):
        for err in ('FP', 'L1'):
            for st in SETS:
                for tau in TAUS:
                    coef_table(L, series, err, st, tau)
    # secondary models
    L.append('\n## 副の形（t CI だけ）\n')
    for series in ('oracle', 'cpm'):
        for err in ('FP', 'L1'):
            for model, tau in (('cov', '1.0'), ('inter', '1.0'), ('cont', '0.0')):
                L.append(f'\n#### {SN[series]}・{err}・pref・{model}・τ = {tau}\n')
                keys = None
                for cd in CONDS:
                    for w in WINS:
                        b = S['coef'].get(f'{series}|{cd}|{err}|pref|{w}|{tau}|{model}')
                        if b is None:
                            continue
                        if keys is None:
                            keys = [k for k in b['names'] if k != 'a'] + (['dir_minus_pref'] if 'pref' in b['names'] else [])
                            L.append('| 条件 | 区間 | ' + ' | '.join(keys) + ' |')
                            L.append('|---|---' + '|---' * len(keys) + '|')
                        L.append(f'| {NAME[cd]} | {WN[w]} | ' + ' | '.join(ci(b['t'].get(k)) for k in keys) + ' |')
    # stability
    L.append('\n## 係数の安定性（pref、6 組の平均。区間 × 形 × τ）\n')
    for series in ('oracle', 'cpm'):
        for err in ('FP', 'L1'):
            for term in ('dir', 'pref'):
                L.append(f'\n#### {SN[series]}・{err}・$b_{{{term}}}$\n')
                cols = [(m, t) for m in ('main', 'cov', 'inter') for t in TAUS] + [('cont', '0.0')]
                L.append('| 条件 | 区間 | ' + ' | '.join(f'{m} τ{t}' for m, t in cols) + ' |')
                L.append('|---|---' + '|---' * len(cols) + '|')
                for cd in CONDS:
                    for w in WINS:
                        row = []
                        for m, t in cols:
                            b = S['coef'].get(f'{series}|{cd}|{err}|pref|{w}|{t}|{m}')
                            row.append('—' if b is None else f(b['t'][term]['mean']))
                        if all(x == '—' for x in row):
                            continue
                        L.append(f'| {NAME[cd]} | {WN[w]} | ' + ' | '.join(row) + ' |')
    # experience dependence
    L.append('\n## 自己の価値経験への依存（差、同じ組の番号で対。t CI と 2段ブートストラップ）\n')
    for series in ('oracle', 'cpm'):
        for err in ('FP', 'L1'):
            for st in SETS:
                for tau in ('1.0', '0.0', '2.0'):
                    L.append(f'\n#### {SN[series]}・{err}・{st}・τ = {tau}\n')
                    L.append('| 差 | 区間 | 組 | Δ$b_{dir}$ t CI | 2段 | Δ$b_{pref}$ t CI | 2段 | Δ($b_{dir}-b_{pref}$) t CI | 2段 |')
                    L.append('|---|---|---|---|---|---|---|---|---|')
                    for comp in ('p50-none', 'p50-sr50', 'p80-none', 'p80-sr80'):
                        for w in WINS:
                            d = S['diff'].get(f'{series}|{err}|{st}|{w}|{tau}|{comp}')
                            if d == '未保存' or d is None:
                                L.append(f'| {comp} | {WN[w]} | 未保存 | | | | | | |')
                                continue
                            t, b2 = d['t'], d['boot2']
                            L.append(f"| {comp} | {WN[w]} | {len(d['pairs'])} | {ci(t['dir'])} | {pc(b2['dir'])} | "
                                     f"{ci(t.get('pref'))} | {pc(b2.get('pref'))} | {ci(t.get('dir_minus_pref'))} | "
                                     f"{pc(b2.get('dir_minus_pref'))} |")
    # cells
    L.append('\n## セルごとの $Q_e$ と「誤差の小さい方の側」の割合（pref、1〜50）\n')
    for series in ('oracle', 'cpm'):
        for err in ('FP', 'L1'):
            for tau in ('1.0', '0.0'):
                L.append(f'\n#### {SN[series]}・{err}・τ = {tau}\n')
                L.append('| 条件 | セル | n | $Q_e$ | A 側の誤差が小さい割合 |')
                L.append('|---|---|---|---|---|')
                for cd in CONDS:
                    C = S['cells'].get(f'{series}|{cd}|{err}|pref|1-50|{tau}')
                    if not C:
                        L.append(f'| {NAME[cd]} | 未保存 | | | |')
                        continue
                    for k, v in C.items():
                        L.append(f"| {NAME[cd]} | {k} | {v['n']} | {ci(v['Q'])} | {ci(v['share_A_smaller'], 3)} |")
                L.append('\n| 条件 | 区間 | 好みの側に当たる割合 | 向きの側に当たる割合 |')
                L.append('|---|---|---|---|')
                for cd in CONDS:
                    for w in WINS:
                        sh = S['share'].get(f'{series}|{cd}|{err}|pref|{w}|{tau}')
                        if sh:
                            L.append(f"| {NAME[cd]} | {WN[w]} | {ci(sh['pref_side'], 3)} | {ci(sh['dir_side'], 3)} |")
    # stage B
    L.append('\n## 段階Bのための材料（pref、主の形）。N80＝組の中で |b| を両側 0.05・検出力 0.8 で 0 と区別するエピソード数の目安、'
             'P80＝組の数の目安（正規近似）。探索で選んだ大きさなので楽観的\n')
    L.append('| 系列 | 誤差 | 条件 | 区間 | τ | 項 | 平均 | 組の SD | 組内 SE | N80 | P80 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for series in ('oracle', 'cpm'):
        for err in ('FP', 'L1'):
            for cd in CONDS:
                for w in WINS:
                    for tau in TAUS:
                        x = S['stageB'].get(f'{series}|{cd}|{err}|pref|{w}|{tau}|main')
                        if not x:
                            continue
                        for term, v in x.items():
                            n80 = '—' if v['N_episodes_80pct'] is None else f"{v['N_episodes_80pct']:.0f}"
                            p80 = '—' if v['n_pairs_80pct'] is None else f"{v['n_pairs_80pct']:.1f}"
                            L.append(f"| {SN[series]} | {err} | {NAME[cd]} | {WN[w]} | {tau} | {term} | {f(v['mean'])} | "
                                     f"{f(v['sd_pairs'])} | {f(v['se_episode_mean'])} | {n80} | {p80} |")
    open(os.path.join(D, 'tables.md'), 'w').write('\n'.join(L) + '\n')
    print('wrote', os.path.join(D, 'tables.md'), len(L), 'lines')


if __name__ == '__main__':
    main()
