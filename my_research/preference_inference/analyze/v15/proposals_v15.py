"""
v15 stage-2 design proposals, data only (docs/v15_experiment_log.md Sec.9; Kusano-san's request 2026-10-24):
  (i)  three positive colours -> two "just ate, a next target exists" moments per episode?
  (ii) lower A-2 rule probability p -> more steps before the ideal observer (d) knows the rank (and fewer eats?)
On the v15p_* datasets (300 episodes each, key v15_p2) and, for reference, the stage-0 v15_s2_c2_p{50,80}
(evaluation episodes 700-999). Functions are those of stage0_v15.py (same definitions); T and the rank values
are taken from each dataset's attributes.

Per dataset:
  eats per episode; informative moments = eat moments with a next target (k = 5 usable: t_e >= 5, t_e + 5 <= T-1),
  their number per episode and their times; next-target accuracy of (a), (b0), (d) at the 1st and the 2nd
  informative moment of an episode ((b0) fitted on the first 2/3 of the episodes, evaluated on the rest);
  steps to identification: awake steps observed before the first eat until the posterior of the true candidate
  reaches 0.95 (main observer and the soft observer with kappa of the stage-0 fit), and the fraction of episodes
  identified before the first eat.

    python analyze/v15/proposals_v15.py    # -> data/result/v15/proposals/proposals.json
"""

import json
import os
import sys

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15 import stage0_v15 as S  # noqa

OUT = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'proposals')
SETS = ['v15p_half_p80', 'v15p_i_grayseq_p80', 'v15p_i_grayhalf_p80'] + [f'v15p_ii_p{p}' for p in (30, 40, 50, 60, 80)] + \
       ['v15p_ii_p50_T151', 'v15_s2_c2_p50', 'v15_s2_c2_p80']
KAPPA = None  # read from the stage-0 summary


def configure(name):
    with h5py.File(os.path.join(S.DATA, name, 'data.h5'), 'r') as f:
        S.T = int(f.attrs.get('T', 101))
        S.VALUES = tuple(float(v) for v in str(f.attrs['rank_values']).split(','))
        n = f['train']['layout'].shape[0]
    return n


def informative(d, k=S.K_MAIN):
    """per episode: list of (t_e, eaten) eat moments with a next target, usable at k."""
    out = {}
    for n, te, c in S.events(d, k):
        if d['tgt'][n, te] >= 0:
            out.setdefault(int(n), []).append((int(te), c))
    for v in out.values():
        v.sort()
    return out


def steps_to_identify(d, eps, kappa):
    res = {'main': [], 'soft': []}
    for n in eps:
        ci = [i for i, c in enumerate(('X', 'Y')) if S.RANKS[c] == tuple(d['rank'][n])][0]
        fe = np.where(d['eat'][n] >= 0)[0]
        stop = min(fe[0] + 1, S.T - 1) if len(fe) else S.T - 1
        for tag, kap in (('main', None), ('soft', kappa)):
            tr = S.posterior_track(d, n, ('X', 'Y'), kappa=kap)
            awake, hit = 0, None
            for t in range(1, stop + 1):
                if np.linalg.norm(d['mot'][n, t - 1]) > 1e-9:
                    awake += 1
                if tr[t, ci] >= 0.95:
                    hit = awake
                    break
            res[tag].append(hit)
    out = {}
    for tag, v in res.items():
        h = [x for x in v if x is not None]
        out[tag] = {'frac_identified_before_first_eat': float(len(h) / len(v)),
                    'median_awake_steps': float(np.median(h)) if h else None,
                    'mean_awake_steps': float(np.mean(h)) if h else None,
                    'p90_awake_steps': float(np.percentile(h, 90)) if h else None}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    kappa = json.load(open(os.path.join(S.OUT, 'stage0_summary.json')))['kappa_fit_c2_p80']
    R = {'kappa': kappa}
    for name in SETS:
        n_ep = configure(name)
        d = S.load(name, 'a2')
        if name.startswith('v15_s2'):
            fit_eps, ev_eps = set(range(S.N_FIT)), list(range(S.N_FIT, n_ep))
        else:
            fit_eps, ev_eps = set(range(2 * n_ep // 3)), list(range(2 * n_ep // 3, n_ep))
        n_eat = (d['eat'] >= 0).sum(1)
        inf = informative(d)
        n_inf = np.array([len(inf.get(n, [])) for n in range(n_ep)])
        r = {'T': S.T, 'values': S.VALUES, 'p': d['p'], 'n_episodes': n_ep,
             'gray_none_frac': float((d['gray0'] < 0).mean()),
             'eat_ge1': float((n_eat >= 1).mean()), 'eat_mean': float(n_eat.mean()),
             'eat_hist': np.bincount(n_eat, minlength=4).tolist(),
             'informative_per_episode_mean': float(n_inf.mean()),
             'informative_hist': np.bincount(n_inf, minlength=3).tolist(),
             'informative_ge2_frac': float((n_inf >= 2).mean())}
        t1 = [v[0][0] for v in inf.values()]
        t2 = [v[1][0] for v in inf.values() if len(v) > 1]
        r['t_inf1_median'] = float(np.median(t1)) if t1 else None
        r['t_inf2_median'] = float(np.median(t2)) if t2 else None
        models = {lv: S.fit_b(d, fit_eps, S.K_MAIN, lv) for lv in (0, 1, 2)}
        for j, lab in ((0, 'inf1'), (1, 'inf2')):
            ev = [(n, v[j][0], v[j][1]) for n, v in inf.items() if n in ev_eps and len(v) > j]
            if not ev:
                r[lab] = None
                continue
            rows = S.predict_events(d, ev, S.K_MAIN, models, kappa=kappa)
            s = S.summarize(rows)
            r[lab] = {'n': s['n_with_target'], **{f'acc_{t}': s[f'acc_{t}'] for t in ('a', 'b0', 'd', 'ds')}}
            if len(rows) >= 5:
                r[lab]['d_minus_a'] = S.boot_diff(rows, 'd', 'a')
                r[lab]['d_minus_b0'] = S.boot_diff(rows, 'd', 'b0')
        r['identify'] = steps_to_identify(d, ev_eps, kappa)
        R[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k not in ('inf1', 'inf2')}), flush=True)
        print('   inf1', r['inf1'], '\n   inf2', r['inf2'], flush=True)
    json.dump(R, open(os.path.join(OUT, 'proposals.json'), 'w'), indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
