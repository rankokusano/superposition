"""
v15 (A) / (B) judgement (pre-registered: docs/v15_experiment_log.md Sec.28.3, 30.3, 30.4, 30.7). CPU only.
Per pair of the generated set (A-2-hidden test, same 930 pairs for every model): blue_X, blue_Y (predicted toward blue on
the X / Y member), m = (blue_X + blue_Y) / 2, d = blue_X - blue_Y, pair accuracy = (acc_X + acc_Y) / 2.
Paired differences over the same pairs, 2000 bootstrap draws (seed 0).
--source cpu (default, used for the judgement, Sec.30.8): every model's CPU evaluation hide_<name>_cpu.json; --source gpu: the
GPU evaluations hide_<name>.json (reference). Writes judge_AB_<enc>_<source>.json.

    python analyze/v15/judge_AB_v15.py --enc v4      # (B) v4 version + (A) v4 seeds
    python analyze/v15/judge_AB_v15.py --enc v15i
Writes data/result/v15/stage2/judge_AB_<enc>.json.
"""

import argparse
import json
import os

import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S2 = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'stage2')


FALLBACK = []  # --source gpu: models whose GPU json has no per-pair arrays (CPU values used)
SOURCE = 'cpu'  # Sec.30.8: the judgement uses CPU evaluations for every model; 'gpu' gives the reference values


def path(exp, seed):
    name = exp if seed == 0 else f'{exp}_seed{seed}'
    if SOURCE == 'cpu':
        return os.path.join(S2, f'hide_{name}_cpu.json')
    p = os.path.join(S2, f'hide_{name}.json')  # GPU evaluation (seed 0 of the 2(b) models was evaluated on GPU first)
    if '_member' not in json.load(open(p)):  # seed-0 GPU runs predate the per-pair arrays: the CPU values stand in
        FALLBACK.append(name)
        return os.path.join(S2, f'hide_{name}_cpu.json')
    return p


def per_pair(exp, seed):
    """-> dict of per-pair arrays ordered by pair id, plus the G 10-step readout."""
    r = json.load(open(path(exp, seed)))
    acc = np.array(r['_acc_ev'], float); pair = np.array(r['_pair']); member = np.array(r['_member'])
    blue = np.array(r['_blue_ev'], float)
    ids = np.unique(pair)
    out = {'ids': ids}
    for key, arr in (('acc', acc), ('blue', blue)):
        for w, s in (('X', 'first'), ('Y', 'second')):
            m = member == s
            o = np.argsort(pair[m])
            out[f'{key}_{w}'] = arr[m][o]
    out['m'] = (out['blue_X'] + out['blue_Y']) / 2
    out['d'] = out['blue_X'] - out['blue_Y']
    out['pairacc'] = (out['acc_X'] + out['acc_Y']) / 2
    out['g10'] = r['gen']['memory_readout']['g']['10']
    return out


def boot(v, seed=0):
    rng = np.random.RandomState(seed)
    d = [v[rng.randint(len(v), size=len(v))].mean() for _ in range(2000)]
    return {'mean': float(v.mean()), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', required=True, choices=['v4', 'v15i'])
    ap.add_argument('--source', default='cpu', choices=['cpu', 'gpu'])
    a = ap.parse_args()
    global SOURCE
    SOURCE = a.source
    e = a.enc
    R = {'enc': e, 'source': SOURCE}
    exp3 = {s: per_pair(f'v15_s2p2h_exp_{e}' + ('' if s == 0 else f'_s{s}'), s) for s in (0, 1, 2)}
    noexp3 = {s: per_pair(f'v15_s2p2h_noexp_{e}' + ('' if s == 0 else f'_s{s}'), s) for s in (0, 1, 2)}
    for s in (0, 1, 2):
        assert np.array_equal(exp3[s]['ids'], exp3[0]['ids']) and np.array_equal(noexp3[s]['ids'], exp3[0]['ids'])
    # (A): sign of exp - noexp per seed, memory kept per seed
    R['A'] = {}
    for s in (0, 1, 2):
        R['A'][s] = {'exp_minus_noexp_pairacc': boot(exp3[s]['pairacc'] - noexp3[s]['pairacc']),
                     'exp_pairacc': boot(exp3[s]['pairacc']), 'noexp_pairacc': boot(noexp3[s]['pairacc']),
                     'exp_g10': exp3[s]['g10'], 'noexp_g10': noexp3[s]['g10'],
                     'exp_m': float(exp3[s]['m'].mean()), 'noexp_m': float(noexp3[s]['m'].mean()),
                     'exp_d': float(exp3[s]['d'].mean()), 'noexp_d': float(noexp3[s]['d'].mean())}
    signs = [np.sign(R['A'][s]['exp_minus_noexp_pairacc']['mean']) for s in (0, 1, 2)]
    R['A']['same_sign_3_seeds'] = bool(len(set(signs)) == 1)
    R['A']['n_seeds_ci_excludes_0'] = int(sum(1 for s in (0, 1, 2) if (R['A'][s]['exp_minus_noexp_pairacc']['ci95'][0] > 0
                                                                      or R['A'][s]['exp_minus_noexp_pairacc']['ci95'][1] < 0)))
    R['A']['memory_kept_all'] = bool(all(R['A'][s][f'{c}_g10']['ci95'][0] > 0.5 for s in (0, 1, 2) for c in ('exp', 'noexp')))
    R['A']['exp_m_range'] = [min(R['A'][s]['exp_m'] for s in (0, 1, 2)), max(R['A'][s]['exp_m'] for s in (0, 1, 2))]
    R['A']['exp_d_range'] = [min(R['A'][s]['exp_d'] for s in (0, 1, 2)), max(R['A'][s]['exp_d'] for s in (0, 1, 2))]
    R['A']['noexp_m_range'] = [min(R['A'][s]['noexp_m'] for s in (0, 1, 2)), max(R['A'][s]['noexp_m'] for s in (0, 1, 2))]
    R['A']['noexp_d_range'] = [min(R['A'][s]['noexp_d'] for s in (0, 1, 2)), max(R['A'][s]['noexp_d'] for s in (0, 1, 2))]
    # (B)
    pB = path(f'v15_s2p2h_expY_{e}', 0)
    if os.path.exists(pB):
        B, X0, N0 = per_pair(f'v15_s2p2h_expY_{e}', 0), exp3[0], noexp3[0]
        assert np.array_equal(B['ids'], X0['ids'])
        b = {'m': float(B['m'].mean()), 'd': float(B['d'].mean()), 'acc_X': float(B['acc_X'].mean()), 'acc_Y': float(B['acc_Y'].mean()),
             'pairacc': boot(B['pairacc']), 'g10': B['g10'],
             'dm': boot(B['m'] - X0['m']), 'dd': boot(B['d'] - X0['d']),
             'E': float(X0['m'].mean() - N0['m'].mean()),
             'm_minus_noexp': boot(B['m'] - N0['m']), 'd_minus_noexp': boot(B['d'] - N0['d']),
             'accX_change': boot(B['acc_X'] - X0['acc_X']), 'accY_change': boot(B['acc_Y'] - X0['acc_Y']),
             'asym_change': boot((B['acc_Y'] - B['acc_X']) - (X0['acc_Y'] - X0['acc_X'])),
             'pairacc_minus_rankX': boot(B['pairacc'] - X0['pairacc']), 'pairacc_minus_noexp': boot(B['pairacc'] - N0['pairacc'])}
        rng_m, rng_d = R['A']['exp_m_range'], R['A']['exp_d_range']
        b['m_below_3seed_range'] = b['m'] < rng_m[0]
        b['m_inside_3seed_range'] = rng_m[0] <= b['m'] <= rng_m[1]
        b['d_inside_3seed_range'] = rng_d[0] <= b['d'] <= rng_d[1]
        b['m_below_noexp_3seeds'] = b['m'] < R['A']['noexp_m_range'][0]
        if e == 'v4':  # Sec.30.3 + 30.7
            dm, E = b['dm'], b['E']
            proj = dm['ci95'][1] < 0 and dm['mean'] <= -E / 2 and b['accX_change']['mean'] < 0 and b['accY_change']['mean'] > 0
            unrel = dm['ci95'][0] <= 0 <= dm['ci95'][1] and abs(dm['mean']) < E / 4 and b['m_minus_noexp']['ci95'][0] > 0
            b['criteria_projection_sec30_3'] = bool(proj)
            b['criteria_unrelated_sec30_3'] = bool(unrel)
            b['verdict'] = ('projection' if proj and b['m_below_3seed_range'] else
                            'undecidable (projection criteria met, m inside the 3-seed range)' if proj else
                            'unrelated to own experience' if unrel else 'neither (describe)')
        else:  # Sec.30.4 + 30.7
            dd, dm = b['dd'], b['dm']
            frozen = (dd['ci95'][0] <= 0 <= dd['ci95'][1] and abs(dd['mean']) < 0.10 and b['d_minus_noexp']['ci95'][1] < 0
                      and abs(dm['mean']) < 0.10)
            interf = ((not (dd['ci95'][0] <= 0 <= dd['ci95'][1]) and abs(dd['mean']) >= 0.10) or
                      (not (dm['ci95'][0] <= 0 <= dm['ci95'][1]) and abs(dm['mean']) >= 0.10))
            b['criteria_frozen_sec30_4'] = bool(frozen)
            b['criteria_interference_sec30_4'] = bool(interf)
            inside = b['d_inside_3seed_range'] and b['m_inside_3seed_range']
            b['verdict'] = ('frozen' if frozen else
                            ('undecidable (interference criteria met, d and m inside the 3-seed ranges)' if inside else 'interference')
                            if interf else 'neither (describe)')
        R['B'] = b
    R['gpu_missing_per_pair_used_cpu'] = sorted(set(FALLBACK))
    json.dump(R, open(os.path.join(S2, f'judge_AB_{e}_{SOURCE}.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


if __name__ == '__main__':
    main()
