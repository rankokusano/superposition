"""
Paper round 32 (paper/check/paper_instructions_round32.md; docs/paper_extra_log.md Sec.56): checksums of the existing
items before and after the round-32 computations. Same items as round 29 (round25_inventory.py GROUPS: stage-I
predictors, stage-II generators, round 20-24 data and results; round-25 data paper_r25_swap50 and result Round25;
results Round26-Round29) plus the round-30 and round-31 outputs (result Round30, Round31) and the items rounds 31-32 read (existing Fig.5 Decoder
v12_fig5_s0, the grid datasets grid / v10_swap_grid at their host locations, the original exp2 Fig.5e output). Python standard library only (host; read only). Earlier
inventory files are not touched.
  before  -> paper/check/round32_existing_inventory_sha256.txt, paper/check/round32_existing_inventory.md
  after   -> paper/check/round32_existing_inventory_after.md
Usage: nice -n 10 python3 analyze/paper_extra/round32_inventory.py before|after
"""

import hashlib
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round25_inventory as R  # noqa

R.GROUPS = R.GROUPS + [('第25弾のデータと結果', ['data/paper_r25_swap50', 'result/paper_extra/Round25']),
                       ('第26弾の結果', ['result/paper_extra/Round26']),
                       ('第27弾の結果', ['result/paper_extra/Round27']),
                       ('第28弾の結果', ['result/paper_extra/Round28']),
                       ('第29弾の結果', ['result/paper_extra/Round29']),
                       ('第30弾の結果', ['result/paper_extra/Round30']),
                       ('第31弾で読むもの（復号器・格子のデータ・元論文の Fig.5e の出力）',
                        ['result/v12_fig5_s0', '../../../../../data/data/grid',
                         '../../../../../my_research/preference_inference/data/data/v10_swap_grid',
                         '../../../../../data/result/exp2/0/test/grid/save/vpt']),
                       ('第31弾の結果', ['result/paper_extra/Round31'])]
R.MANIFEST = os.path.join(R.CHECK, 'round32_existing_inventory_sha256.txt')


def before():
    t0 = time.time()
    assert not os.path.exists(R.MANIFEST), R.MANIFEST
    lines = R.scan()
    open(R.MANIFEST, 'w').write('\n'.join(lines) + '\n')
    pd = R.per_dir(lines)
    ar = R.archive_check()
    L = ['# 第32弾 既存のものの一覧（計算の前）', '',
         f'- 作成：{time.strftime("%Y-%m-%d %H:%M")}（JST）。読むだけ。`analyze/paper_extra/round32_inventory.py before`。',
         '- 対象：第31弾と同じもの（`round31_existing_inventory.md`。復号器 `v12_fig5_s0`・格子のデータ・元のコードの Fig.5e の出力を含む）に、第31弾の結果（`Round31/`）を足した。第30弾の価値の網（`Round30/s1/`）は第30弾の結果の中にある。',
         '- 場所の起点：`my_research/preference_inference/data/`（`.worktrees/v11` の作業木）。',
         f'- ファイルごとの sha256・大きさ・場所の全部：`paper/check/round32_existing_inventory_sha256.txt`（{len(lines)} 行）。',
         '- 下の表の「一覧の sha256」は、その場所の行（上のファイルの行をそのまま、場所の順）をつないだものの sha256。',
         f'- かかった時間：{time.time() - t0:.0f} 秒。', '']
    tot_n = tot_b = 0
    for g, dirs in R.GROUPS:
        L += [f'## {g}', '', '| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |', '|---|---|---|---|']
        for d in dirs:
            r = pd.get(d, {'n': 0, 'bytes': 0, 'lines': []})
            mh = hashlib.sha256('\n'.join(r['lines']).encode()).hexdigest()
            L.append(f"| `{d}` | {r['n']} | {r['bytes']:,} | `{mh}` |")
            tot_n += r['n']
            tot_b += r['bytes']
        L.append('')
    L += [f'- 合計：{tot_n} ファイル、{tot_b:,} バイト。', '',
          '## `/mnt/md0` に移した古いデータ（場所と一覧の確かめだけ）', '',
          f"- 場所：`{R.ARCHIVE}`。一覧：`{ar['manifest']}`（{ar['listed']} 行）。",
          f"- 今あるファイル：{ar['present']}。一覧にあって無いもの：{ar['missing']}。一覧に無くてあるもの：{ar['extra']}。記号リンク {ar['links']} 個。", '']
    open(os.path.join(R.CHECK, 'round32_existing_inventory.md'), 'w').write('\n'.join(L))
    print('\n'.join(L[:9]), f'\nfiles {tot_n} bytes {tot_b}', ar)


def after():
    t0 = time.time()
    old = [ln for ln in open(R.MANIFEST).read().splitlines() if ln]
    new = R.scan()
    so, sn = set(old), set(new)
    po = {ln.split('  ', 2)[2] for ln in old}
    pn = {ln.split('  ', 2)[2] for ln in new}
    changed = sorted({ln.split('  ', 2)[2] for ln in so ^ sn} & po & pn)
    ar = R.archive_check()
    L = ['# 第32弾 既存のものの確かめ（計算の後）', '',
         f'- 作成：{time.strftime("%Y-%m-%d %H:%M")}（JST）。`analyze/paper_extra/round32_inventory.py after`。',
         f'- 計算の前の一覧（`round32_existing_inventory_sha256.txt`、{len(old)} 行）と、今もう一度取った sha256 を比べた。',
         f'- **同じ行：{len(so & sn)}／{len(old)}**。中身が変わったもの：{len(changed)}。無くなったもの：{len(po - pn)}。増えたもの：{len(pn - po)}。',
         f"- `/mnt/md0`：一覧 {ar['listed']} 行、今あるファイル {ar['present']}、無いもの {ar['missing']}、一覧に無いもの {ar['extra']}、記号リンク {ar['links']}。",
         f'- かかった時間：{time.time() - t0:.0f} 秒。', '']
    for p in changed[:50]:
        L.append(f'- 変わった：`{p}`')
    for p in sorted(po - pn)[:50]:
        L.append(f'- 無くなった：`{p}`')
    for p in sorted(pn - po)[:50]:
        L.append(f'- 増えた：`{p}`')
    open(os.path.join(R.CHECK, 'round32_existing_inventory_after.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'before': before, 'after': after}[sys.argv[1]]()
