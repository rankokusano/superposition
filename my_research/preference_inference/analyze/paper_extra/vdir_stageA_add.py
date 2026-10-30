"""
Paper round 12 add-on, V-dir stage A (docs/paper_extra_log.md Sec.36.7; exploratory, outside the pre-registration;
procedure committed before computing, 5781907). Saved arrays and existing metadata only (CPU only).
  add 1: c+- series main-form b_pref and b_dir - b_pref per condition, read from Vdir_A/summary.json (no new
         regression); reference t CI of the other conditions restricted to the 5 pairs of p50 FP (s0-2, s4, s5).
  add 2: initial-position covariate form (model 'cov') with the within-pair episode bootstrap and the two-stage
         bootstrap (vdir_stageA.boot_block unchanged), all series / conditions / errors / sets / windows / tau.
  add 3: N80 / P80 for that form, same formula as the main form (vdir_stageA.main, stage-B block).
vdir_stageA.py and summary.json are not changed.
-> data/result/paper_extra/Vdir_A/add_summary.json
"""

import json
import os

import numpy as np

import vdir_stageA as V

S = json.load(open(os.path.join(V.OUT, 'summary.json')))
P50_PAIRS = S['coef']['cpm|p50|FP|pref|1-50|1.0|main']['pairs']


def load_q():
    Q = {}
    for series in ('oracle', 'cpm'):
        for cond in V.CONDS:
            for err in V.ERRS:
                for st in V.SETS:
                    pairs, arr = [], {w: [] for w in V.WINS}
                    for s in V.PAIRS[cond]:
                        pts = V.load_pair(series, cond, err, st, s)
                        if pts is None:
                            continue
                        m = M[st]
                        pairs.append(s)
                        for w in V.WINS:
                            qs = []
                            for z in pts:
                                if series == 'oracle':
                                    LA = np.where(m['s_red'][:, None] == 1, z['Lr'], z['Lg'])
                                    LB = np.where(m['s_red'][:, None] == 1, z['Lg'], z['Lr'])
                                    qs.append(V.q_window(LB, w) - V.q_window(LA, w))
                                else:
                                    qs.append(V.q_window(z['B'], w) - V.q_window(z['A'], w))
                            arr[w].append(qs)
                    if pairs:
                        Q[(series, cond, err, st)] = ({w: np.array(arr[w]) for w in V.WINS}, pairs)
    return Q


def stage_b(blk):
    out = {}
    for nm in ('dir', 'pref', 'dir_minus_pref'):
        if nm not in blk['t'] or blk['t'][nm]['mean'] is None:
            continue
        b, se, sd, n = blk['t'][nm]['mean'], blk['boot_se'].get(nm), blk['t'][nm].get('sd'), blk['n_used']
        out[nm] = {'mean': b, 'sd_pairs': sd, 'se_episode_mean': se, 'n_used': n,
                   'N_episodes_80pct': (float(n * (2.8 * se / abs(b)) ** 2) if se and b else None),
                   'n_pairs_80pct': (float((2.8 * sd / abs(b)) ** 2) if sd and b else None)}
    return out


def main():
    R = {'add1': {}, 'add1_5pairs': {}, 'cov': {}, 'stageB_cov': {}, 'checks': {}}
    # ---- add 1 (no new regression)
    for key, blk in S['coef'].items():
        series, cond, err, st, w, tau, model = key.split('|')
        if series != 'cpm' or model != 'main':
            continue
        R['add1'][key] = {nm: {'t': blk['t'][nm], 'boot2': blk['boot2'].get(nm), 'boot_pair': blk['boot_pair'].get(nm),
                               'boot_se': blk['boot_se'].get(nm)} for nm in ('pref', 'dir', 'dir_minus_pref')
                          if nm in blk['t']}
        R['add1'][key]['pairs'] = blk['pairs']
        if err == 'FP' and st == 'pref':
            idx = [blk['pairs'].index(s) for s in P50_PAIRS if s in blk['pairs']]
            if len(idx) == len(P50_PAIRS):
                R['add1_5pairs'][key] = {nm: V.tsum([blk['per_pair'][nm][i] for i in idx])
                                         for nm in ('pref', 'dir', 'dir_minus_pref') if nm in blk['per_pair']}
    # ---- add 2 / 3
    Q = load_q()
    dmax = 0.0
    for (series, cond, err, st), (QW, pairs) in Q.items():
        m = M[st]
        green_only = st.startswith('green')
        for w in V.WINS:
            Y = QW[w]
            for tau in V.TAUS:
                X, names, keep = V.design(m, w, tau, 'cov', green_only)
                Yk = Y[:, :, keep]
                coef = V.solve(X[None], np.swapaxes(Yk, 1, 2))
                pv = coef.mean(2)
                key = f'{series}|{cond}|{err}|{st}|{w}|{tau}|cov'
                blk = {'pairs': pairs, 'n_used': int(keep.sum()), 'names': names, 'per_pair': {}, 't': {}}
                for j, nm in enumerate(names):
                    blk['per_pair'][nm] = pv[:, j].tolist()
                    blk['t'][nm] = V.tsum(pv[:, j])
                if 'pref' in names:
                    d = pv[:, names.index('dir')] - pv[:, names.index('pref')]
                    blk['per_pair']['dir_minus_pref'] = d.tolist()
                    blk['t']['dir_minus_pref'] = V.tsum(d)
                old = S['coef'][key]['per_pair']
                for nm, v in blk['per_pair'].items():
                    a, b = np.array(v, float), np.array(old[nm], float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    assert (np.isfinite(a) == np.isfinite(b)).all(), key
                    if ok.any():
                        dmax = max(dmax, float(np.abs(a[ok] - b[ok]).max()))
                V.boot_block(blk, X, Yk, names)
                R['cov'][key] = blk
                R['stageB_cov'][key] = stage_b(blk)
        print('done', series, cond, err, st, flush=True)
    R['checks']['cov_per_pair_vs_summary_max_abs'] = dmax
    R['checks']['n_cov_blocks'] = len(R['cov'])
    json.dump(R, open(os.path.join(V.OUT, 'add_summary.json'), 'w'), ensure_ascii=False, indent=0)
    print('checks', R['checks'])


M = {st: V.meta(st) for st in V.SETS}

if __name__ == '__main__':
    main()
