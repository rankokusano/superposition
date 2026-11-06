"""
Paper round 25 Sec.0 (paper/check/paper_instructions_round25.md; docs/paper_extra_log.md Sec.49): inventory of the
existing predictors, generators and round 20-24 data / results (location, number of files, size, sha256 of every file),
and a check that they are unchanged after the trial. Python standard library only (runs on the host; read only).
  before  -> paper/check/round25_existing_inventory_sha256.txt (one line per file: sha256, size, path) and
             paper/check/round25_existing_inventory.md (per directory: files, bytes, sha256 of its manifest lines);
             /mnt/md0 archive: location and listing only (its own manifest's file list vs the files present; no hashing)
  after   -> recompute and compare with the committed manifest -> paper/check/round25_existing_inventory_after.md
Usage: nice -n 10 python3 analyze/paper_extra/round25_inventory.py before|after
"""

import hashlib
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_PI, 'data')
CHECK = os.path.join(_PI, 'paper', 'check')
MANIFEST = os.path.join(CHECK, 'round25_existing_inventory_sha256.txt')
ARCHIVE = '/mnt/md0/kusano/superposition_archive'

GROUPS = [
    ('段階Iの予測器（L1 期間）', [f'result/{e}_s{s}' for e in ('v12_s1_random_l1', 'v12_s1_mixed_l1', 'v13_s1_mixed_l1') for s in range(6)] +
     [f'result/paper_extra/E1_signrand50/s1_signrand50_l1_s{s}' for s in range(6)] +
     [f'result/paper_extra/C_signrand/s1_signrand_l1_s{s}' for s in range(6)]),
    ('段階Iの予測器（MSE 期間、L1 期間の出発点）', ['result/v12_s1_random_mse', 'result/v12_s1_mixed_mse', 'result/v13_s1_mixed_mse',
                                     'result/paper_extra/E1_signrand50/s1_signrand50_mse',
                                     'result/paper_extra/C_signrand/s1_signrand_mse']),
    ('段階IIの生成器', [f'result/{e}_s{s}' for e in ('v12_s2_pref_mixed', 'v12_s2_pref_random', 'v13_s2_pref_mixed') for s in range(4)]),
    ('第20〜24弾の評価データ', ['data/paper_r20_swap_p80', 'data/paper_r20_swap_p50', 'data/paper_r21_swap']),
    ('第20〜24弾の結果', [f'result/paper_extra/Round{r}' for r in (20, 21, 22, 23, 24)]),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def files(d):
    out = []
    for root, _, fs in os.walk(os.path.join(DATA, d), followlinks=True):
        for f in fs:
            out.append(os.path.relpath(os.path.join(root, f), DATA))
    return sorted(out)


def scan():
    lines = []
    for _, dirs in GROUPS:
        for d in dirs:
            for p in files(d):
                full = os.path.join(DATA, p)
                lines.append(f'{sha256(full)}  {os.path.getsize(full)}  {p}')
    return lines


def per_dir(lines):
    out = {}
    for ln in lines:
        h, size, p = ln.split('  ', 2)
        for _, dirs in GROUPS:
            for d in dirs:
                if p.startswith(d + '/'):
                    r = out.setdefault(d, {'n': 0, 'bytes': 0, 'lines': []})
                    r['n'] += 1
                    r['bytes'] += int(size)
                    r['lines'].append(ln)
    return out


def archive_check():
    man = os.path.join(ARCHIVE, 'archive_md0_2026-11-02.sha256')
    listed = [ln.split('  ', 1)[1].strip() for ln in open(man) if ln.strip()]
    present, links = set(), 0
    for root, _, fs in os.walk(os.path.join(ARCHIVE, 'my_research')):
        for f in fs:
            full = os.path.join(root, f)
            if os.path.islink(full):
                links += 1          # symbolic links (not in the archive manifest) are counted apart
                continue
            present.add(os.path.relpath(full, ARCHIVE))
    tops = sorted(os.listdir(os.path.join(ARCHIVE, 'my_research', 'preference_inference', 'data', 'data'))) + \
        sorted('result/' + x for x in os.listdir(os.path.join(ARCHIVE, 'my_research', 'preference_inference', 'data', 'result')))
    return {'manifest': man, 'listed': len(listed), 'present': len(present), 'links': links,
            'missing': len(set(listed) - present), 'extra': len(present - set(listed)), 'top': tops}


def before():
    t0 = time.time()
    lines = scan()
    open(MANIFEST, 'w').write('\n'.join(lines) + '\n')
    pd = per_dir(lines)
    ar = archive_check()
    L = ['# 第25弾 §0 既存のものの一覧（試しの前）', '',
         f'- 作成：{time.strftime("%Y-%m-%d %H:%M")}（JST）。読むだけ。`analyze/paper_extra/round25_inventory.py before`。',
         f'- 場所の起点：`my_research/preference_inference/data/`（`.worktrees/v11` の作業木）。',
         f'- ファイルごとの sha256・大きさ・場所の全部：`paper/check/round25_existing_inventory_sha256.txt`（{len(lines)} 行）。',
         '- 下の表の「一覧の sha256」は、その場所の行（上のファイルの行をそのまま、場所の順）をつないだものの sha256。',
         f'- かかった時間：{time.time() - t0:.0f} 秒。', '']
    tot_n = tot_b = 0
    for g, dirs in GROUPS:
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
          f"- 場所：`{ARCHIVE}`。移したときの一覧：`{ar['manifest']}`（{ar['listed']} 行）。",
          f"- 今あるファイル（通常のファイル）：{ar['present']}。一覧にあって無いもの：{ar['missing']}。一覧に無くてあるもの：{ar['extra']}。ほかに記号リンク {ar['links']} 個（先の無いもの。一覧には入っていない）。",
          '- 中身（`my_research/preference_inference/data/` の下）：' + '、'.join(f'`{x}`' for x in ar['top']) + '。', '']
    open(os.path.join(CHECK, 'round25_existing_inventory.md'), 'w').write('\n'.join(L))
    print('\n'.join(L[:8]), f'\nfiles {tot_n} bytes {tot_b}', ar['listed'], ar['present'], ar['missing'], ar['extra'])


def after():
    t0 = time.time()
    old = [ln for ln in open(MANIFEST).read().splitlines() if ln]
    new = scan()
    so, sn = set(old), set(new)
    po = {ln.split('  ', 2)[2] for ln in old}
    pn = {ln.split('  ', 2)[2] for ln in new}
    changed = sorted({ln.split('  ', 2)[2] for ln in so ^ sn} & po & pn)
    ar = archive_check()
    L = ['# 第25弾 §0 既存のものの確かめ（試しの後）', '',
         f'- 作成：{time.strftime("%Y-%m-%d %H:%M")}（JST）。`analyze/paper_extra/round25_inventory.py after`。',
         f'- 試しの前の一覧（`round25_existing_inventory_sha256.txt`、{len(old)} 行）と、今もう一度取った sha256 を比べた。',
         f'- **同じ行：{len(so & sn)}／{len(old)}**。中身が変わったもの：{len(changed)}。無くなったもの：{len(po - pn)}。増えたもの：{len(pn - po)}。',
         f"- `/mnt/md0`：一覧 {ar['listed']} 行、今あるファイル {ar['present']}、無いもの {ar['missing']}、一覧に無いもの {ar['extra']}、記号リンク {ar['links']}。",
         f'- かかった時間：{time.time() - t0:.0f} 秒。', '']
    for p in changed[:50]:
        L.append(f'- 変わった：`{p}`')
    for p in sorted(po - pn)[:50]:
        L.append(f'- 無くなった：`{p}`')
    for p in sorted(pn - po)[:50]:
        L.append(f'- 増えた：`{p}`')
    open(os.path.join(CHECK, 'round25_existing_inventory_after.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'before': before, 'after': after}[sys.argv[1]]()
