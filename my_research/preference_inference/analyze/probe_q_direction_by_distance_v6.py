"""
(§27) Does c2's weak direction info come from a range-restriction artifact
in the evaluation data (A-2, following its own Green preference, spends
most of r2_a1random_a2rl clustered near Green, where "direction toward
Green" is barely defined), or is it a genuine critic problem?

Bins frames by distance to the condition's target landmark and reports
dirCorr/within45 per bin, for c1/c2/c3/c4, for each curriculum seed.
Reuses probe_q_direction_info_v6.py's critic loading / probe_q /
condition-loading code directly (imported, not duplicated).

Usage (inside Docker):
    cd /work/my_research/preference_inference
    python3 analyze/probe_q_direction_by_distance_v6.py
"""
import math
import sys

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/my_research/preference_inference/analyze')

import numpy as np

from probe_q_direction_info_v6 import (
    CONDITIONS, load_critic, load_condition, probe_q, PROBES,
)

CANDIDATES = [
    ('curriculum_s0', 'v6', 'v6_rl_critic_seed0_film_norelabel_curriculum.pth', True),
    ('curriculum_s1', 'v6', 'v6_rl_critic_seed1_film_norelabel_curriculum.pth', True),
    ('curriculum_s2', 'v6', 'v6_rl_critic_seed2_film_norelabel_curriculum.pth', True),
]
BIN_EDGES = [0.0, 5.0, 10.0, 15.0, 100.0]
BIN_LABELS = ['<5', '5-10', '10-15', '>=15']


def binned_direction(q, pos, landmark):
    dist = np.linalg.norm(landmark[None, :] - pos, axis=1)
    dn = (landmark[None, :] - pos)
    ok = dist > 1e-6
    dn = dn[ok] / dist[ok, None]
    cosd = dn @ PROBES.T.astype(np.float64)
    qc = q[ok] - q[ok].mean(axis=1, keepdims=True)
    cc = cosd - cosd.mean(axis=1, keepdims=True)
    num = (qc * cc).sum(axis=1)
    den = np.sqrt((qc ** 2).sum(axis=1) * (cc ** 2).sum(axis=1))
    valid = den > 1e-12
    corr_per_frame = np.full(len(dist[ok]), np.nan)
    corr_per_frame[valid] = num[valid] / den[valid]
    within45_per_frame = (cosd[np.arange(cosd.shape[0]), q[ok].argmax(axis=1)] >= math.cos(math.pi / 4))
    d = dist[ok]

    rows = []
    for lo, hi, label in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
        m = (d >= lo) & (d < hi)
        n = int(m.sum())
        if n == 0:
            rows.append((label, n, float('nan'), float('nan')))
            continue
        c = np.nanmean(corr_per_frame[m])
        w = float(within45_per_frame[m].mean())
        rows.append((label, n, c, w))
    return rows


def main():
    cache = {}
    for cname, cond in CONDITIONS.items():
        if cname not in ('c1', 'c2', 'c3', 'c4'):
            continue
        frames, pos, e, t = load_condition(cond)
        cache[cname] = (frames, pos, cond)
        print(f'[{cname}] {cond["dataset"]}/{cond["vision"]}: target={cond["landmark_name"]}, '
              f'mean pos=({pos[:,0].mean():.2f},{pos[:,1].mean():.2f}), '
              f'mean dist to target={np.linalg.norm(cond["landmark"][None,:]-pos,axis=1).mean():.2f}')

    for name, kind, fname, film in CANDIDATES:
        critic = load_critic(kind, fname, film)
        print(f'\n=== {name} ===')
        for cname in ('c1', 'c2', 'c3', 'c4'):
            frames, pos, cond = cache[cname]
            q = probe_q(critic, kind, frames, cond['r'])
            rows = binned_direction(q, pos, cond['landmark'])
            row_str = '  '.join(f'{lbl}(n={n}): dirCorr={c:+.3f} in45={w:.3f}' for lbl, n, c, w in rows)
            print(f'  {cname} (target={cond["landmark_name"]}): {row_str}')


if __name__ == '__main__':
    main()
