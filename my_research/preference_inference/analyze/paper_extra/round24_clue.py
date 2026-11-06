"""
Paper round 24 Sec.1.1.3, 1.2, 1.7 (paper/check/paper_instrctions_round24.md; docs/paper_extra_log.md Sec.48;
exploratory, data only, CPU). Input: Round24/clue/p<p_rule>_t<swap_t>_part<i>.h5 (collect_data_round24.py, stage-II
procedure, 600 episodes per setting, every episode with a swap), Round24/check/images_p<p>.h5 (6 episodes with views),
round 20 data paper_r20_swap_p50 (Sec.1.7).
Per episode (liked colour of A-2, pre-swap layout lay0): L_old = liked colour's place in lay0, L_new = in the swapped
layout. Transition t -> t+1 (t = 0..99):
  moved      |p_{t+1} - p_t| > 1e-6
  clue       t >= swap frame, moved, d(p_{t+1}, L_new) < d(p_t, L_new) and d(p_{t+1}, L_old) >= d(p_t, L_old)
             ("toward the liked colour's new place and not toward its old place": wrong for a place habit, right for
             the liked-object rule)
  habit      t >= swap frame, moved, d_old decreases and d_new does not
Counts per episode, fraction of all 100 transitions and of the post-swap transitions; per stage-II epoch (3000 training
episodes) for swap fractions 0.25 / 0.5 / 1.0; steps from the swap until d(p, L_new) < 1.5 (also 1.0 / 2.0), and the
fraction never arriving by frame 100. Rule check (Sec.1.1.3): A-2's rule steps (a2_source 1) after the swap, fraction
toward L_new and fraction 'habit' (toward L_old, not toward L_new).
Sec.1.7: fraction of frames with nonzero pleasure (v11core.pleasure.pleasure_np, Red liked / Green disliked; the same
ranges for Green liked) per range, in A-1's and A-2's views after the swap (round 20 N and O, frames 50-80; round 24
image episodes).
-> Round24/analysis/{clue.json, clue_tables.md, fig_round24_arrival.{pdf,png}, swap_images.png}
"""

import glob
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
from sim_v10 import layout as L  # noqa
from v11core.pleasure import pleasure_np  # noqa

R24 = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round24')
OUT = os.path.join(R24, 'analysis')
SETS = [(p, t) for p in ('0.5', '0.8') for t in ('20', '50', 'rand20-80')]
N_TRAIN = 3000
P_SWAP = (0.25, 0.5, 1.0)
THR = (1.0, 1.5, 2.0)


def load(p, t):
    acc = {}
    for f in sorted(glob.glob(os.path.join(R24, 'clue', f'p{p}_t{t}_part*.h5'))):
        h = h5py.File(f, 'r')
        for k in ('other_position', 'a2_source', 'layout', 'a2_liked', 'swap_frame'):
            acc.setdefault(k, []).append(h[k][()])
    return {k: np.concatenate(v) for k, v in acc.items()}


def targets(lay, lk):
    col = np.where(lk == 1, 'Red', 'Green')
    lo = np.array([L.LANDMARKS[L.CODE_LAYOUT[int(a)]][c] for a, c in zip(lay, col)], np.float64)
    ln = np.array([L.LANDMARKS[L.other_layout(L.CODE_LAYOUT[int(a)])][c] for a, c in zip(lay, col)], np.float64)
    return lo, ln


def analyse(D):
    op = D['other_position'].astype(np.float64)
    N, T = op.shape[:2]
    lo, ln = targets(D['layout'], D['a2_liked'])
    dn = np.linalg.norm(op - ln[:, None], axis=-1)
    do = np.linalg.norm(op - lo[:, None], axis=-1)
    ts = D['swap_frame'].astype(int)
    tt = np.arange(T - 1)[None]
    post = tt >= ts[:, None]
    moved = np.linalg.norm(np.diff(op, axis=1), axis=-1) > 1e-6
    toward_new = np.diff(dn, axis=1) < 0
    not_toward_old = np.diff(do, axis=1) >= 0
    toward_old = np.diff(do, axis=1) < 0
    clue = post & moved & toward_new & not_toward_old
    habit = post & moved & toward_old & ~toward_new
    n_post = post.sum(1)
    r = {'n_episodes': int(N), 'swap_frame_mean': float(ts.mean()), 'swap_frame_min': int(ts.min()), 'swap_frame_max': int(ts.max()),
         'clue_per_ep_mean': float(clue.sum(1).mean()), 'clue_per_ep_median': float(np.median(clue.sum(1))),
         'clue_per_ep_p10': float(np.percentile(clue.sum(1), 10)), 'clue_per_ep_p90': float(np.percentile(clue.sum(1), 90)),
         'clue_zero_frac': float((clue.sum(1) == 0).mean()),
         'clue_frac_of_all_transitions': float(clue.sum() / clue.size),
         'clue_frac_of_post_transitions': float(clue.sum() / n_post.sum()),
         'habit_per_ep_mean': float(habit.sum(1).mean()),
         'moved_frac_post': float((moved & post).sum() / n_post.sum()),
         'post_transitions_per_ep_mean': float(n_post.mean()),
         'dist_new_at_swap_mean': float(dn[np.arange(N), ts].mean()), 'dist_old_at_swap_mean': float(do[np.arange(N), ts].mean())}
    for k in P_SWAP:
        r[f'clue_per_epoch|p_swap{k}'] = float(N_TRAIN * k * r['clue_per_ep_mean'])
    for thr in THR:
        steps = np.full(N, np.nan)
        for n in range(N):
            hit = np.where(dn[n, ts[n]:] < thr)[0]
            if len(hit):
                steps[n] = hit[0]
        ok = np.isfinite(steps)
        r[f'arrive|{thr}'] = {'never_frac': float((~ok).mean()), 'median': float(np.median(steps[ok])) if ok.any() else None,
                              'p25': float(np.percentile(steps[ok], 25)) if ok.any() else None,
                              'p75': float(np.percentile(steps[ok], 75)) if ok.any() else None,
                              'mean': float(steps[ok].mean()) if ok.any() else None,
                              'max_possible_mean': float((T - 1 - ts).mean())}
        if thr == 1.5:
            r['_steps'] = steps
    src = D['a2_source'][:, :-1]
    rule = post & (src == 1)
    r['rule_steps_post'] = int(rule.sum())
    r['rule_toward_new_frac'] = float((rule & toward_new).sum() / max(rule.sum(), 1))
    r['rule_habit_frac'] = float((rule & toward_old & ~toward_new).sum() / max(rule.sum(), 1))
    rnd = post & (src == 0) & moved
    r['random_toward_new_frac'] = float((rnd & toward_new).sum() / max(rnd.sum(), 1))
    r['random_habit_frac'] = float((rnd & toward_old & ~toward_new).sum() / max(rnd.sum(), 1))
    r['source_frac_post'] = {int(s): float(((src == s) & post).sum() / n_post.sum()) for s in (0, 1, 2, 3)}
    return r


def range_pleasure(views):
    """views uint8 (M, 16, 64, 3) -> fraction nonzero per range (8)."""
    P = pleasure_np(views)
    return (np.abs(P) > 0).mean(0).round(4).tolist()


def images():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for p in ('0.8', '0.5'):
        h = h5py.File(os.path.join(R24, 'check', f'images_p{p}.h5'), 'r')
        for i in range(3):
            rows.append((p, i, h))
    fig, axs = plt.subplots(len(rows), 5, figsize=(17, 2.1 * len(rows)), gridspec_kw={'width_ratios': [3, 3, 3, 3, 1.4]})
    for r, (p, i, h) in enumerate(rows):
        ts = int(h['swap_frame'][i])
        lk = int(h['a2_liked'][i])
        lo, ln = targets(h['layout'][i:i + 1], h['a2_liked'][i:i + 1])
        for j, (cam, t) in enumerate((('self_vision', ts - 1), ('self_vision', ts), ('other_vision', ts - 1), ('other_vision', ts))):
            axs[r, j].imshow(h[cam][i, t], interpolation='nearest', aspect='auto')
            axs[r, j].set_title(f'p{p} ep{i} {"A-1" if cam == "self_vision" else "A-2"} view, frame {t}', fontsize=8)
            axs[r, j].axis('off')
        op = h['other_position'][i]
        ax = axs[r, 4]
        ax.plot(op[:ts + 1, 0], op[:ts + 1, 1], color='0.5', lw=1)
        ax.plot(op[ts:, 0], op[ts:, 1], color='C1', lw=1.2)
        ax.plot(*op[ts], 'ko', ms=3)
        ax.plot(*lo[0], 's', color='C3' if lk else 'C2', ms=6)
        ax.plot(*ln[0], '^', color='C3' if lk else 'C2', ms=6)
        ax.set_xlim(-10.5, 10.5); ax.set_ylim(-10.5, 10.5); ax.set_aspect('equal'); ax.tick_params(labelsize=6)
        ax.set_title(f'A-2 likes {"Red" if lk else "Green"}; grey before, orange after swap; sq L_old, tri L_new', fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'swap_images.png'), dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    R, steps = {}, {}
    for p, t in SETS:
        r = analyse(load(p, t))
        steps[(p, t)] = r.pop('_steps')
        R[f'p{p}|t{t}'] = r
    # Sec.1.7
    s17 = {}
    for v in ('N', 'O'):
        g = h5py.File(os.path.join(_PI, 'data', 'data', 'paper_r20_swap_p50', f'data_{v}.h5'), 'r')['test']
        for cam in ('self_vision', 'other_vision'):
            s17[f'r20_p50_{v}|{cam}|frames50-80'] = range_pleasure(g[cam][:, 50:81].reshape(-1, 16, 64, 3))
    for p in ('0.5', '0.8'):
        h = h5py.File(os.path.join(R24, 'check', f'images_p{p}.h5'), 'r')
        for cam in ('self_vision', 'other_vision'):
            v = np.concatenate([h[cam][i, int(h['swap_frame'][i]):] for i in range(h[cam].shape[0])])
            s17[f'r24_images_p{p}|{cam}|after_swap'] = range_pleasure(v)
    R['sec1_7_nonzero_pleasure_per_range'] = s17
    json.dump(R, open(os.path.join(OUT, 'clue.json'), 'w'), indent=1)
    Lt = ['# 第24弾 §1.2 手がかりの数（データだけ）', '', '600 エピソード（すべて入れ替えあり）。遷移 t→t+1、t = 0〜99。',
          '', '| A-2 の規則 | 入れ替えの時点 | 1 エピソードの数（平均・中央値・10〜90%） | 0 のエピソード | 全遷移に占める割合 | 入れ替え後の遷移に占める割合 | 習慣の向きの数（平均） | 1 エポックの数（入れ替え 25%／50%／100%） |',
          '|---|---|---|---|---|---|---|---|']
    for p, t in SETS:
        r = R[f'p{p}|t{t}']
        Lt.append(f"| {p} | {t} | {r['clue_per_ep_mean']:.1f}・{r['clue_per_ep_median']:.0f}・{r['clue_per_ep_p10']:.0f}〜{r['clue_per_ep_p90']:.0f} | "
                  f"{r['clue_zero_frac']:.3f} | {r['clue_frac_of_all_transitions']:.3f} | {r['clue_frac_of_post_transitions']:.3f} | "
                  f"{r['habit_per_ep_mean']:.1f} | " + '／'.join(f"{r[f'clue_per_epoch|p_swap{k}']:.0f}" for k in P_SWAP) + ' |')
    Lt += ['', '## 新しい場所に着くまでの歩数（入れ替えから、$L_{new}$ まで 1.5 未満）', '',
           '| A-2 の規則 | 入れ替えの時点 | 中央値 [25%, 75%] | 平均 | 着かない割合（1.0／1.5／2.0） | 入れ替えの時点での $L_{new}$・$L_{old}$ までの距離 |', '|---|---|---|---|---|---|']
    for p, t in SETS:
        r = R[f'p{p}|t{t}']
        a = r['arrive|1.5']
        Lt.append(f"| {p} | {t} | {a['median']:.0f} [{a['p25']:.0f}, {a['p75']:.0f}] | {a['mean']:.1f} | "
                  + '／'.join(f"{r[f'arrive|{x}']['never_frac']:.3f}" for x in THR) +
                  f" | {r['dist_new_at_swap_mean']:.2f}・{r['dist_old_at_swap_mean']:.2f} |")
    Lt += ['', '## A-2 の規則の確かめ（入れ替えの後の規則の歩）', '',
           '| A-2 の規則 | 入れ替えの時点 | 規則の歩の数 | 新しい場所へ近づく割合 | 古い場所へ近づき新しい場所へ近づかない割合 | 無作為の歩で新しい場所へ／古い場所へ |', '|---|---|---|---|---|---|']
    for p, t in SETS:
        r = R[f'p{p}|t{t}']
        Lt.append(f"| {p} | {t} | {r['rule_steps_post']} | {r['rule_toward_new_frac']:.3f} | {r['rule_habit_frac']:.4f} | "
                  f"{r['random_toward_new_frac']:.3f}／{r['random_habit_frac']:.3f} |")
    Lt += ['', '## §1.7 快・不快が 0 でない割合（範囲 0〜7）', '']
    for k, v in s17.items():
        Lt.append(f'- {k}: {v}')
    open(os.path.join(OUT, 'clue_tables.md'), 'w').write('\n'.join(Lt))
    print('\n'.join(Lt))
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, p in zip(axs, ('0.5', '0.8')):
        for t, c in zip(('20', '50', 'rand20-80'), ('C0', 'C1', 'C2')):
            s = steps[(p, t)]
            ok = np.isfinite(s)
            ax.hist(s[ok], bins=np.arange(0, 82, 2), histtype='step', color=c, label=f'swap t={t} (never {np.mean(~ok):.2f})')
        ax.set_title(f'A-2 p_rule {p}: steps from the swap until d(L_new) < 1.5', fontsize=9)
        ax.set_xlabel('steps')
        ax.legend(fontsize=7)
    axs[0].set_ylabel('episodes')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round24_arrival.{ext}'), dpi=130)
    plt.close(fig)
    images()


if __name__ == '__main__':
    main()
