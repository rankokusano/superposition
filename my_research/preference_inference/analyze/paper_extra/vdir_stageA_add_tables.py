"""
Paper round 12 add-on, V-dir stage A (docs/paper_extra_log.md Sec.36.7; exploratory): markdown tables from
Vdir_A/add_summary.json (vdir_stageA_add.py) and summary.json only; no computation beyond formatting.
-> data/result/paper_extra/Vdir_A/add_tables.md
"""

import json
import os

from vdir_stageA_tables import CONDS, NAME, WINS, WN, TAUS, SETS, SN, D, S, f, ci, pc, nexcl

A = json.load(open(os.path.join(D, 'add_summary.json')))


def g(x, n=4):
    return '—' if x is None else f'{x:.{n}f}'


def nn(x):
    return '—' if x is None else (f'{x:.0f}' if x >= 10 else f'{x:.1f}')


def add1(L):
    L.append('\n## 追加1：$c^{\\pm}$ 系列の主の形の $b_{pref}$ と $b_{dir}-b_{pref}$（summary.json から取り出しただけ）\n')
    L.append('- t CI ＝ 学習の変動、2段 ＝ 両方（参考）、組内ブート ＝ 評価エピソードの変動（0 を上 / 下に外す組の数）。')
    for err in ('FP', 'L1'):
        for st in ('pref', 'green'):
            for tau in TAUS:
                L.append(f'\n#### c± 系列・{err}・{st}・τ = {tau}\n')
                L.append('| 条件 | 区間 | 組 | $b_{pref}$ t CI | $b_{pref}$ 2段 | $b_{pref}$ 組内ブート | $b_{dir}-b_{pref}$ t CI | '
                         '$b_{dir}-b_{pref}$ 2段 | $b_{dir}-b_{pref}$ 組内ブート |')
                L.append('|---|---|---|---|---|---|---|---|---|')
                for cd in CONDS:
                    for w in WINS:
                        b = A['add1'].get(f'cpm|{cd}|{err}|{st}|{w}|{tau}|main')
                        if b is None:
                            L.append(f'| {NAME[cd]} | {WN[w]} | 未保存 | | | | | | |')
                            continue
                        prs = ','.join(str(s + 1) for s in b['pairs'])
                        p, d = b.get('pref'), b.get('dir_minus_pref')
                        if p is None:
                            L.append(f'| {NAME[cd]} | {WN[w]} | {prs} | （緑好みのデータは $s^{{pref}}$ を外す） | | | | | |')
                            continue
                        L.append(f"| {NAME[cd]} | {WN[w]} | {prs} | {ci(p['t'])} | {pc(p['boot2'])} | {nexcl(p['boot_pair'])} | "
                                 f"{ci(d['t'])} | {pc(d['boot2'])} | {nexcl(d['boot_pair'])} |")
    L.append('\n### 参考：同じ 5 組（組1〜3・5・6。c±・FP の経験50% の組）に限った t CI（pref）\n')
    L.append('| 条件 | 区間 | τ | $b_{pref}$ | $b_{dir}$ | $b_{dir}-b_{pref}$ |')
    L.append('|---|---|---|---|---|---|')
    for tau in TAUS:
        for cd in CONDS:
            for w in WINS:
                b = A['add1_5pairs'].get(f'cpm|{cd}|FP|pref|{w}|{tau}|main')
                if b:
                    L.append(f"| {NAME[cd]} | {WN[w]} | {tau} | {ci(b['pref'])} | {ci(b['dir'])} | {ci(b['dir_minus_pref'])} |")


def add2(L):
    L.append('\n## 追加2：初期位置を共変量に入れた形（副 (i)）の係数と3種類の CI\n')
    L.append(f"- 確認：組の値と summary.json の副 (i) の値の差の最大 {A['checks']['cov_per_pair_vs_summary_max_abs']:.3g}"
             f"（{A['checks']['n_cov_blocks']} ブロック）。")
    L.append('- $c_2$・$c_1$ ＝ A-2・A-1 の初期位置（$\\hat n$ への射影）の係数（1 単位あたり）。')
    for series, err in (('cpm', 'FP'), ('cpm', 'L1'), ('oracle', 'L1'), ('oracle', 'FP')):
        for st in SETS:
            for tau in TAUS:
                if not any(k.startswith(f'{series}|') and f'|{err}|{st}|' in k and k.endswith(f'|{tau}|cov') for k in A['cov']):
                    continue
                L.append(f'\n#### {SN[series]} 系列・{err}・{st}・τ = {tau}・初期位置あり\n')
                L.append('| 条件 | 区間 | n | $b_{pref}$ | $b_{dir}$ | $b_{red}$ | $c_2$（A-2） | $c_1$（A-1） | $b_{dir}-b_{pref}$ | '
                         '$b_{dir}$ 2段 | $b_{dir}-b_{pref}$ 2段 | $b_{dir}$ 組内ブート | $b_{dir}-b_{pref}$ 組内ブート |')
                L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
                for cd in CONDS:
                    for w in WINS:
                        b = A['cov'].get(f'{series}|{cd}|{err}|{st}|{w}|{tau}|cov')
                        if b is None:
                            L.append(f'| {NAME[cd]} | {WN[w]} | 未保存 | | | | | | | | | | |')
                            continue
                        t, b2, bp = b['t'], b['boot2'], b['boot_pair']
                        L.append(f"| {NAME[cd]} | {WN[w]} | {b['n_used']} | {ci(t.get('pref'))} | {ci(t['dir'])} | {ci(t['red'])} | "
                                 f"{ci(t['a2y0'])} | {ci(t['a1y0'])} | {ci(t.get('dir_minus_pref'))} | {pc(b2.get('dir'))} | "
                                 f"{pc(b2.get('dir_minus_pref'))} | {nexcl(bp['dir'])} | "
                                 f"{nexcl(bp['dir_minus_pref']) if 'dir_minus_pref' in bp else '—'} |")


def add3(L):
    L.append('\n## 追加3：必要なエピソード数・組の数の目安（初期位置あり、主の形と並べる）\n')
    L.append('- $N_{80} = N_{used}((1.96+0.84)\\,SE/|b|)^2$、$P_{80} = ((1.96+0.84)\\,SD_{組}/|\\bar b|)^2$。SE ＝ 組内ブートの SD の組の平均。')
    L.append('- 楽観的（探索で見た大きさ）。$|\\bar b|$ が 0 に近い行の値は「観察した大きさが本当なら」の値。$P_{80}$ < 3 は意味を持たない。')
    for series, err in (('oracle', 'FP'), ('cpm', 'FP'), ('oracle', 'L1'), ('cpm', 'L1')):
        for st in SETS:
            for tau in TAUS:
                keys = [k for k in A['stageB_cov'] if k.startswith(f'{series}|') and f'|{err}|{st}|' in k and k.endswith(f'|{tau}|cov')]
                if not keys:
                    continue
                L.append(f'\n#### {SN[series]} 系列・{err}・{st}・τ = {tau}\n')
                L.append('| 条件 | 区間 | 項 | 形 | $\\bar b$ [t CI] | SD$_{組}$ | SE | $N_{used}$ | $N_{80}$ | $P_{80}$ |')
                L.append('|---|---|---|---|---|---|---|---|---|---|')
                for cd in CONDS:
                    for w in WINS:
                        kc = f'{series}|{cd}|{err}|{st}|{w}|{tau}|cov'
                        if kc not in A['stageB_cov']:
                            continue
                        km = kc[:-3] + 'main'
                        for nm, lab in (('dir', '$b_{dir}$'), ('dir_minus_pref', '$b_{dir}-b_{pref}$')):
                            for form, sb, cb in (('主', S['stageB'].get(km, {}), S['coef'].get(km)),
                                                 ('初期位置あり', A['stageB_cov'][kc], A['cov'][kc])):
                                r = sb.get(nm)
                                if r is None:
                                    continue
                                L.append(f"| {NAME[cd]} | {WN[w]} | {lab} | {form} | {ci(cb['t'][nm])} | {g(r['sd_pairs'])} | "
                                         f"{g(r['se_episode_mean'])} | {cb['n_used']} | {nn(r['N_episodes_80pct'])} | "
                                         f"{nn(r['n_pairs_80pct'])} |")


def main():
    L = ['# V-dir 段階A 追加の表（探索的（事前登録外）。`analyze/paper_extra/vdir_stageA_add.py`、§36.7）\n']
    add1(L)
    add2(L)
    add3(L)
    open(os.path.join(D, 'add_tables.md'), 'w').write('\n'.join(L) + '\n')
    print('wrote', os.path.join(D, 'add_tables.md'), len(L))


if __name__ == '__main__':
    main()
