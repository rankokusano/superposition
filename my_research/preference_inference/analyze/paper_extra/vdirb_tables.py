"""
Paper round 13 rev2, V-dir stage B: markdown tables from Vdir_B/analysis/summary.json (vdirb_analysis.py) only;
formatting, no computation. -> data/result/paper_extra/Vdir_B/analysis/tables.md
"""

import json
import os

import vdirb_analysis as A

S = json.load(open(os.path.join(A.OUT, 'summary.json')))
NAME = {'none': '経験なし', 'p50': '経験50%', 'p80': '経験80%', 'sr80': '符号無作為（80%）', 'sr50': '符号無作為（50%）',
        'sw': '好み切替（組1〜3）', 'p50-none': '経験50% − 経験なし', 'p50-sr50': '経験50% − 符号無作為（50%）',
        'p80-none': '経験80% − 経験なし', 'p80-sr80': '経験80% − 符号無作為（80%）'}
FN = {'main': '主', 's_i': '(i) x 座標', 's_ii': '(ii) 角までの距離', 's_iii': '(iii) 初期位置なし',
      's_iv': '(iv) 3 区分 × 向き', 's_v': '(v) A 側の割合', 'nodir': '向きを調整しない'}
WN = {'1-25': '1〜25', '26-50': '26〜50', '51-100': '51〜100', '1-50': '1〜50', '1-100': '1〜100', 'past': '過去の向き'}
ROWS = list(A.CONDS) + [f'{a}-{b}' for a, b in A.DIFFS]


def f(x, n=4):
    return '—' if x is None else f'{x:+.{n}f}'


def ci(t, n=4):
    if t is None:
        return '—'
    if t.get('mean') is None:
        return '推定不可'
    return f"{f(t['mean'], n)} [{f(t['ci95'][0], n)}, {f(t['ci95'][1], n)}]" if t.get('ci95') else f(t['mean'], n)


def pc(p, n=4):
    return '—' if not p else f'[{f(p[0], n)}, {f(p[1], n)}]'


def blk(key):
    return S['coef'].get(key) or S['diff'].get(key)


def main():
    L = ['# V-dir 段階B 表（`analyze/paper_extra/vdirb_analysis.py`、作業記録 §37.5。事前登録 4216497）\n']
    L.append(f"- 2 回目の一致：{'全配列でビット単位で一致' if S['checks']['rep_all'] else '一致しない配列あり'}"
             f"（{sum(S['checks']['rep_bit_identical'].values())}/{len(S['checks']['rep_bit_identical'])} 配列）。")
    L.append(f"- 欠けた条件：{S['missing'] or 'なし'}。件数：{json.dumps(S['counts'], ensure_ascii=False)}")
    L.append('\n## 判定（$c^{\\pm}$・FP・1〜50・τ = 1.0・主の回帰）\n')
    L.append('| 判定 | 項 | 比較 | 平均 | 6 組の t の 95% CI | 2段の 95% CI | 組ごとの値 | 5 点中 3 点以上で正の組 | 結果 |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for jn in ('P1', 'P2', 'P3', 'D1', 'D2', 'D3'):
        j = S['judgement'][jn]
        cmp_ = NAME[j['cond']] if j['vs'] is None else NAME[f"{j['cond']}-{j['vs']}"]
        L.append(f"| {jn} | $b_{{{j['term']}}}$ | {cmp_} | {f(j['mean'])} | {pc(j['t_ci95'])} | {pc(j['boot2_ci95'])} | "
                 f"{', '.join(f(x) for x in j['per_pair'])} | {j['pairs_ge3of5']}/6 | {j['verdict']} |")
    L.append(f"\n- 族 P：{'成立' if S['judgement']['family_P'] else '成立しない'}。族 D：{'成立' if S['judgement']['family_D'] else '成立しない'}。")
    for series, err in (('cpm', 'FP'), ('cpm', 'L1'), ('oracle', 'FP'), ('oracle', 'L1')):
        L.append(f'\n## {"c±" if series == "cpm" else "oracle"} 系列・{err}・1〜50・τ = 1.0：形ごとの係数\n')
        L.append('| 形 | 条件・差 | n | $b_{pref}$ t CI | $b_{pref}$ 2段 | $b_{dir}$ t CI | $b_{dir}$ 2段 | $b_{dir}-b_{pref}$ t CI | '
                 '$b_{red}$ | 初期位置などの係数（t の平均） | 5点中3点以上（pref / dir） |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for fm in A.FORMS:
            for r in ROWS:
                b = blk(f'{series}|{err}|1-50|1.0|{fm}|{r}')
                if b is None:
                    continue
                t, b2 = b['t'], b.get('boot2', {})
                extra = ', '.join(f"{k} {f(t[k]['mean'])}" for k in b['names'] if k not in ('a', 'pref', 'dir', 'red'))
                pg = b.get('pairs_ge3of5', {})
                L.append(f"| {FN[fm]} | {NAME[r]} | {b['n_used']} | {ci(t.get('pref'))} | {pc(b2.get('pref'))} | "
                         f"{ci(t.get('dir'))} | {pc(b2.get('dir'))} | {ci(t.get('dir_minus_pref'))} | {f(t['red']['mean'])} | "
                         f"{extra} | {pg.get('pref', '—')} / {pg.get('dir', '—')} |")
    L.append('\n## 副次解析（判定しない）：二択の BA と、好みと向きが食い違うエピソードでの好みの正解率（1〜50）\n')
    L.append('| 系列・誤差 | 条件・差 | BA（好み） | 2段 | BA（向き） | 2段 | 食い違う全体 | 2段 | B 側 | 中央 | A 側 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for series, err in (('cpm', 'FP'), ('cpm', 'L1'), ('oracle', 'FP'), ('oracle', 'L1')):
        for r in ROWS:
            x = S['secondary'].get(f'{series}|{err}|{r}')
            if x is None:
                continue
            g = lambda k: ci(x[k]['t'], 3)  # noqa
            b = lambda k: pc(x[k].get('boot2'), 3)  # noqa
            L.append(f"| {series}・{err} | {NAME[r]} | {g('BA_pref')} | {b('BA_pref')} | {g('BA_dir')} | {b('BA_dir')} | "
                     f"{g('acc_disagree')} | {b('acc_disagree')} | {g('acc_disagree_B')} | {g('acc_disagree_mid')} | {g('acc_disagree_A')} |")
    L.append('\n## 区間・τ ごとの主の回帰（6 組の t の CI だけ）\n')
    for series, err in (('cpm', 'FP'), ('cpm', 'L1'), ('oracle', 'FP'), ('oracle', 'L1')):
        L.append(f'\n#### {series}・{err}\n')
        L.append('| 区間 | τ | 条件・差 | n | $b_{pref}$ | $b_{dir}$ | $b_{dir}-b_{pref}$ |')
        L.append('|---|---|---|---|---|---|---|')
        for w in A.V.WINS:
            for tau in A.V.TAUS:
                for r in ROWS:
                    b = blk(f'{series}|{err}|{w}|{tau}|main|{r}')
                    if b is None:
                        continue
                    t = b['t']
                    L.append(f"| {WN[w]} | {tau} | {NAME[r]} | {b['n_used']} | {ci(t['pref'])} | {ci(t['dir'])} | {ci(t['dir_minus_pref'])} |")
    open(os.path.join(A.OUT, 'tables.md'), 'w').write('\n'.join(L) + '\n')
    print('wrote', os.path.join(A.OUT, 'tables.md'), len(L))


if __name__ == '__main__':
    main()
