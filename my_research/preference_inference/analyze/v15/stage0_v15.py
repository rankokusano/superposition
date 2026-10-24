"""
v15 stage-0 analysis (docs/v15_instructions_stage0.md Sec.5; definitions and
gates pre-registered in docs/v15_experiment_log.md Sec.4). No learning
except the (b) predictors (sklearn HistGradientBoosting, CPU).

    python analyze/v15/stage0_v15.py            # all items -> data/result/v15/stage0/
"""

import json
import os
import sys

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)

from v11core.pleasure import color_direction  # noqa
from v11core.rule_agent import slide_step  # noqa
from v15core.world import COLORS  # noqa

DATA = os.environ.get('V15_DATA', os.path.join(_PI_DIR, 'data', 'data'))  # override only for smoke tests
OUT = os.environ.get('V15_OUT', os.path.join(_PI_DIR, 'data', 'result', 'v15', 'stage0'))
T = 101
KS = (3, 5, 10)
K_MAIN = 5
R_EAT = 1.5
N_FIT = int(os.environ.get('V15_NFIT', 700))
BOOT = 2000
BOUNDS = ((-10, 10), (-10, 10))
# per layout code: colour code -> corner point / cube centre (arena x, y)
_CORNER = {'TL': (-1, 1), 'BL': (-1, -1), 'BR': (1, -1), 'TR': (1, 1)}
_COL_CORNER = {0: ('TL', 'BL', 'BR', 'TR'), 1: ('BL', 'TL', 'BR', 'TR')}  # [layout][colour]
CORNER_PT = {l: np.array([np.array(_CORNER[c]) * 10. for c in _COL_CORNER[l]]) for l in (0, 1)}
CUBE_C = {l: np.array([np.array(_CORNER[c]) * 13. for c in _COL_CORNER[l]]) for l in (0, 1)}
RANKS = {'X': (0, 2, 3, 1), 'Y': (1, 3, 2, 0), 'Xp': (0, 3, 2, 1)}  # colour codes, most liked first
VALUES = (1.0, 0.5, 0.0, -1.0)


def target_of(rank, avail):
    """rank: colour codes most liked first; avail: (4,) -> colour code or -1."""
    for i, c in enumerate(rank):
        if avail[c] and VALUES[i] > 0:
            return c
    return -1


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-12), 0.)


def ang(u, v):
    """angle (deg) between direction vectors (..., 2)."""
    a = np.arctan2(u[..., 1], u[..., 0]) - np.arctan2(v[..., 1], v[..., 0])
    return np.abs(np.rad2deg((a + np.pi) % (2 * np.pi) - np.pi))


# ---------------------------------------------------------------- loading
def load(name, agent):
    """agent: 'a1' (stage 1, self_*) or 'a2' (stage 2, other_*)."""
    pre = 'self' if agent == 'a1' else 'other'
    with h5py.File(os.path.join(DATA, name, 'data.h5'), 'r') as f:
        g = f['train']
        d = {k: g[k][:] for k in ('layout', 'gray0', 'a2_cand', 'available')}
        d['pos'] = g[f'{pre}_position'][:].astype(np.float64)
        d['mot'] = g[f'{pre}_motion'][:].astype(np.float64)
        d['eat'] = g[f'{agent}_eat'][:]
        d['tgt'] = g[f'{agent}_target'][:]
        d['src'] = g[f'{agent}_source'][:]
        d['rank'] = g[f'{agent}_rank'][:]
        d['vision_ds'] = (name, f'{pre}_vision')
        d['a1pos'] = g['self_position'][:]
        d['a1mot'] = g['self_motion'][:]
        d['opos32'] = g[f'{pre}_position'][:]
        d['omot32'] = g[f'{pre}_motion'][:]
        d['p'] = float(f.attrs['p_rule'])
    d['name'] = name
    return d


def vision(d, n):
    name, key = d['vision_ds']
    with h5py.File(os.path.join(DATA, name, 'data.h5'), 'r') as f:
        return f['train'][key][n]


def events(d, k):
    """eat events (n, t_e, colour) with t_e >= k and t_e + k <= 100."""
    ev = []
    for n, t in zip(*np.where(d['eat'] >= 0)):
        te = t + 1
        if te - k >= 0 and te + k <= T - 1:
            ev.append((n, te, int(d['eat'][n, t])))
    return ev


# ---------------------------------------------------------------- 5.1 / 5.2 / 5.4
def eat_stats(d):
    n_eat = (d['eat'] >= 0).sum(1)
    first = np.array([np.where(e >= 0)[0][0] + 1 if (e >= 0).any() else -1 for e in d['eat']])
    after = np.array([(T - f) / T if f >= 0 else 0. for f in first])
    cp = np.concatenate([CORNER_PT[0], CORNER_PT[1]])
    dist = np.min(np.linalg.norm(d['pos'][:, 50:, None, :] - cp[None, None], axis=-1), -1)
    return {
        'frac_ge1': float((n_eat >= 1).mean()),
        'n_eat_hist': np.bincount(n_eat, minlength=4).tolist(),
        'n_eat_mean': float(n_eat.mean()),
        'first_median': float(np.median(first[first >= 0])) if (first >= 0).any() else None,
        'first_times': first.tolist(),
        'frac_after_first_mean': float(after.mean()),
        'corner5_late_mean': float((dist <= 5).mean()),
    }


def turn_stats(d):
    out = {}
    for k in KS:
        a = []
        for n, te, _ in events(d, k):
            pre = d['pos'][n, te] - d['pos'][n, te - k]
            post = d['pos'][n, te + k] - d['pos'][n, te]
            if np.linalg.norm(pre) > 1e-9 and np.linalg.norm(post) > 1e-9:
                a.append(ang(pre, post))
        a = np.array(a)
        out[k] = {'n': len(a), 'frac_gt45': float((a > 45).mean()) if len(a) else None,
                  'median_deg': float(np.median(a)) if len(a) else None, 'angles': a.tolist()}
    return out


def config_counts(d):
    c = np.zeros((2, 4), int)
    for l, g in zip(d['layout'], d['gray0']):
        c[l, g] += 1
    return c.tolist()


# ---------------------------------------------------------------- (d) ideal observer
def posterior_track(d, n, cands, vis=None, kappa=None):
    """log posterior over candidates for frames 0..100 (prior uniform).
    kappa None -> main (exact rule-step match); else soft von Mises observer.
    Returns (101, n_cand) posterior probabilities at the START of each frame
    (evidence from steps 0..t-1, incl. eats after step t-1)."""
    p = d['p']
    if vis is None:
        vis = vision(d, n)
    lay = d['layout'][n]
    pos, mot, eat, av = d['pos'][n], d['mot'][n], d['eat'][n], d['available'][n]
    L = np.zeros(len(cands))
    out = np.zeros((T, len(cands)))
    out[0] = 1. / len(cands)
    q0 = 1e-6
    for t in range(T - 1):
        moved = np.linalg.norm(mot[t]) > 1e-9
        for i, c in enumerate(cands):
            if not np.isfinite(L[i]):
                continue
            tg = target_of(RANKS[c], av[t])
            if moved:
                if tg >= 0:
                    if kappa is None:
                        dr = color_direction(vis[t], COLORS[tg])
                        r = None if dr is None else slide_step(pos[t], dr, BOUNDS, 1.0)
                        if r is not None:
                            L[i] += np.log(p / q0 + (1 - p)) if np.linalg.norm(mot[t] - r) < 1e-4 else np.log(1 - p)
                    else:
                        dirc = CUBE_C[lay][tg] - pos[t]
                        a = np.deg2rad(ang(mot[t], dirc))
                        L[i] += np.log(p * np.exp(kappa * np.cos(a)) / (2 * np.pi * np.i0(kappa)) + (1 - p) / (2 * np.pi))
                else:
                    if kappa is not None:
                        L[i] += np.log(1 / (2 * np.pi))
                # eat evidence (only after awake moves)
                if eat[t] >= 0 and eat[t] != tg:
                    L[i] = -np.inf
                elif eat[t] < 0 and tg >= 0 and np.linalg.norm(pos[t + 1] - CORNER_PT[lay][tg]) <= R_EAT - 1e-3:
                    L[i] = -np.inf
        if np.all(~np.isfinite(L)):
            L[:] = 0.  # inconsistent with every candidate (numerical edge) -> reset to prior
        m = L[np.isfinite(L)].max()
        w = np.where(np.isfinite(L), np.exp(L - m), 0.)
        out[t + 1] = w / w.sum()
    return out


def d_target_dist(post, cands, avail):
    """distribution over next target (colour code or -1) at a frame."""
    dist = {}
    for i, c in enumerate(cands):
        tg = target_of(RANKS[c], avail)
        dist[tg] = dist.get(tg, 0.) + post[i]
    return dist


def fit_kappa(d, eps):
    """MLE of the von Mises concentration from rule steps (source 1) of the fit episodes."""
    from scipy.stats import vonmises
    a = []
    for n in eps:
        lay = d['layout'][n]
        for t in np.where(d['src'][n] == 1)[0]:
            tg = d['tgt'][n, t]
            a.append(np.deg2rad(ang(d['mot'][n, t], CUBE_C[lay][tg] - d['pos'][n, t])))
    a = np.array(a)
    kappa, _, _ = vonmises.fit(a, fscale=1, floc=0)
    return float(kappa)


# ---------------------------------------------------------------- (b) predictors
def features(d, n, te, level):
    f = list(d['pos'][n, te])
    for k in KS:
        f += list(unit(d['pos'][n, te] - d['pos'][n, max(te - k, 0)]))
    if level >= 1:
        f += [d['layout'][n]] + list(np.eye(4)[d['gray0'][n]])
    if level >= 2:
        f += list(d['available'][n, te])
    return f


def fit_b(d_fit, eps, k, level):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    X, y, ycs = [], [], []
    for n, te, _ in events(d_fit, k):
        if n not in eps:
            continue
        post = d_fit['pos'][n, te + k] - d_fit['pos'][n, te]
        tg = d_fit['tgt'][n, te]
        X.append(features(d_fit, n, te, level))
        y.append(tg)
        ycs.append(unit(post))
    X, y, ycs = np.array(X), np.array(y), np.array(ycs)
    m = y >= 0
    clf = HistGradientBoostingClassifier(random_state=0).fit(X[m], y[m])
    mv = np.linalg.norm(ycs, axis=1) > 0
    rc = HistGradientBoostingRegressor(random_state=0).fit(X[mv], ycs[mv, 0])
    rs = HistGradientBoostingRegressor(random_state=0).fit(X[mv], ycs[mv, 1])
    return clf, rc, rs


# ---------------------------------------------------------------- 5.3 prediction table
def predict_events(d, ev_list, k, models, cands=('X', 'Y'), kappa=None, post_cache=None):
    """Per event: dict of predicted target / correctness / angle errors for a, b0-2, c, d, d_soft."""
    rows = []
    post_cache = {} if post_cache is None else post_cache
    for n, te, col in ev_list:
        lay = d['layout'][n]
        av = d['available'][n, te]
        pos = d['pos'][n, te]
        pre = d['pos'][n, te] - d['pos'][n, te - k]
        post = d['pos'][n, te + k] - d['pos'][n, te]
        truth = int(d['tgt'][n, te])
        r = {'n': int(n), 'te': int(te), 'eaten': col, 'truth': truth, 'layout': int(lay)}
        dirs = {}
        # (a)
        cand_cols = [c for c in range(4) if av[c]]
        if np.linalg.norm(pre) > 1e-9 and cand_cols:
            bear = [ang(pre, CUBE_C[lay][c] - pos) for c in cand_cols]
            r['pred_a'] = int(cand_cols[int(np.argmin(bear))])
        else:
            r['pred_a'] = -1
        dirs['a'] = pre
        # (b)
        for lv in (0, 1, 2):
            clf, rc, rs = models[lv]
            x = np.array([features(d, n, te, lv)])
            r[f'pred_b{lv}'] = int(clf.predict(x)[0])
            dirs[f'b{lv}'] = np.array([rc.predict(x)[0], rs.predict(x)[0]])
        # (c)
        r['pred_c'] = truth
        dirs['c'] = CUBE_C[lay][truth] - pos if truth >= 0 else pre
        # (d) main and soft
        for tag, kap in (('d', None), ('ds', kappa)):
            if tag == 'ds' and kap is None:
                continue
            key = (d['name'], int(n), tag)
            if key not in post_cache:
                post_cache[key] = posterior_track(d, n, cands, kappa=kap)
            pt = post_cache[key][te]
            dist = d_target_dist(pt, cands, av)
            best = max(dist.values())
            tied = [c for c, v in dist.items() if abs(v - best) < 1e-12]
            r[f'pred_{tag}'] = int(tied[0])
            r[f'acc_{tag}'] = (1. / len(tied)) if truth in tied else 0.
            r[f'pmass_{tag}'] = float(dist.get(truth, 0.))
            r[f'post_true_{tag}'] = float(pt[cands.index(next(c for c in cands if RANKS[c] == tuple(d['rank'][n])))]) \
                if tuple(d['rank'][n]) in [RANKS[c] for c in cands] else None
            dirs[tag] = CUBE_C[lay][tied[0]] - pos if tied[0] >= 0 else pre
        for tag in ('a', 'b0', 'b1', 'b2', 'c'):
            r[f'acc_{tag}'] = float(r[f'pred_{tag}'] == truth)
        for tag, v in dirs.items():
            r[f'angerr_{tag}'] = float(ang(v, post)) if np.linalg.norm(post) > 1e-9 and np.linalg.norm(v) > 1e-9 else None
        rows.append(r)
    return rows


def boot_diff(rows, a, b, key='acc', unit_key='n', seed=0):
    """difference mean(key_a - key_b) with cluster bootstrap over units."""
    units = sorted(set(r[unit_key] for r in rows))
    idx = {u: [] for u in units}
    for i, r in enumerate(rows):
        idx[r[unit_key]].append(i)
    diff = np.array([r[f'{key}_{a}'] - r[f'{key}_{b}'] for r in rows])
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(BOOT):
        s = rng.randint(len(units), size=len(units))
        ii = np.concatenate([idx[units[j]] for j in s])
        bs.append(diff[ii].mean())
    return {'diff': float(diff.mean()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def summarize(rows, tags=('a', 'b0', 'b1', 'b2', 'c', 'd', 'ds')):
    out = {'n_events': len(rows), 'n_episodes': len(set(r['n'] for r in rows))}
    has = [r for r in rows if r['truth'] >= 0]
    out['n_with_target'] = len(has)
    for t in tags:
        if has and f'acc_{t}' in has[0]:
            out[f'acc_{t}'] = float(np.mean([r[f'acc_{t}'] for r in has]))
            e = [r[f'angerr_{t}'] for r in rows if r.get(f'angerr_{t}') is not None]
            out[f'angerr_median_{t}'] = float(np.median(e)) if e else None
            e = [r[f'angerr_{t}'] for r in has if r.get(f'angerr_{t}') is not None]
            out[f'angerr_median_tgt_{t}'] = float(np.median(e)) if e else None
    return out


# ---------------------------------------------------------------- main analysis per dataset
def analyse_prediction(d, d_fit, label, kappa):
    res = {}
    eval_eps = set(range(N_FIT, len(d['eat'])))
    fit_eps = set(range(N_FIT))
    cache = {}
    for k in KS:
        models = {lv: fit_b(d_fit, fit_eps, k, lv) for lv in (0, 1, 2)}
        ev = [e for e in events(d, k) if e[0] in eval_eps]
        rows = predict_events(d, ev, k, models, kappa=kappa, post_cache=cache)
        has = [r for r in rows if r['truth'] >= 0]
        s = summarize(rows)
        s['d_minus_a'] = boot_diff(has, 'd', 'a')
        s['d_minus_b0'] = boot_diff(has, 'd', 'b0')
        s['d_minus_b1'] = boot_diff(has, 'd', 'b1')
        s['d_minus_b2'] = boot_diff(has, 'd', 'b2')
        s['ds_minus_a'] = boot_diff(has, 'ds', 'a')
        s['ds_minus_b0'] = boot_diff(has, 'ds', 'b0')
        # actually next eaten as target
        nxt = []
        for r in has:
            e = d['eat'][r['n'], r['te']:]
            w = np.where(e >= 0)[0]
            if len(w):
                nxt.append((int(e[w[0]]), r))
        s['acc_next_eaten'] = {t: float(np.mean([float(r[f'pred_{t}'] == c) for c, r in nxt])) if nxt else None
                               for t in ('a', 'b0', 'b1', 'b2', 'c', 'd')}
        s['n_next_eaten'] = len(nxt)
        res[k] = s
        if k == K_MAIN:
            res['rows_k5'] = rows
    # posterior rise (main observer), eval episodes, steps before first eat
    rise = {}
    for n in sorted(eval_eps):
        key = (d['name'], int(n), 'd')
        if key not in cache:
            cache[key] = posterior_track(d, n, ('X', 'Y'))
        tr = cache[key]
        ci = [i for i, c in enumerate(('X', 'Y')) if RANKS[c] == tuple(d['rank'][n])]
        if not ci:
            continue
        fe = np.where(d['eat'][n] >= 0)[0]
        stop = min(fe[0] + 1, T - 1) if len(fe) else T - 1
        awake = 0
        for t in range(1, stop + 1):
            if np.linalg.norm(d['mot'][n, t - 1]) > 1e-9:
                awake += 1
            rise.setdefault(awake, []).append(tr[t, ci[0]])
    res['rise'] = {int(a): [float(np.mean(v)), len(v)] for a, v in sorted(rise.items())}
    res['label'] = label
    return res


# ---------------------------------------------------------------- pairs
def same_upto(dx, dy, n, te):
    return (np.array_equal(dx['opos32'][n, :te + 1], dy['opos32'][n, :te + 1])
            and np.array_equal(dx['omot32'][n, :te], dy['omot32'][n, :te])
            and np.array_equal(dx['a1pos'][n, :te + 1], dy['a1pos'][n, :te + 1])
            and np.array_equal(dx['a1mot'][n, :te], dy['a1mot'][n, :te]))


def first_eat(d, n):
    w = np.where(d['eat'][n] >= 0)[0]
    return (int(w[0]) + 1, int(d['eat'][n, w[0]])) if len(w) else (None, None)


def analyse_generated(dx, dy, d_fit, kappa, cands=('X', 'Y'), corner_check=True, tag='gen'):
    models = {lv: fit_b(d_fit, set(range(N_FIT)), K_MAIN, lv) for lv in (0, 1, 2)}
    out = {'n_keys': len(dx['eat']), 'pairs': []}
    rows = []
    n_both, n_same, n_frame_same, n_usable, diverge = 0, 0, 0, 0, []
    for n in range(len(dx['eat'])):
        tx, cx = first_eat(dx, n)
        ty, cy = first_eat(dy, n)
        if tx is None or ty is None:
            continue
        n_both += 1
        if tx != ty or tx > T - 1:  # eaten after the last step: no frame t_e to compare
            continue
        te = tx
        same = same_upto(dx, dy, n, te)
        if corner_check:
            ok_corner = (CORNER_PT[dx['layout'][n]][cx] == np.array([-10., 10.])).all() and \
                        (CORNER_PT[dy['layout'][n]][cy] == np.array([-10., 10.])).all()
        else:
            ok_corner = cx == cy
        if not (same and ok_corner):
            continue
        n_same += 1
        with h5py.File(os.path.join(DATA, dx['name'], 'data.h5'), 'r') as fx, \
                h5py.File(os.path.join(DATA, dy['name'], 'data.h5'), 'r') as fy:
            fs = all(np.array_equal(fx['train'][k][n, te], fy['train'][k][n, te]) for k in ('self_vision', 'other_vision'))
            pre_diff = [int(np.any(fx['train']['other_vision'][n, t] != fy['train']['other_vision'][n, t])) for t in range(te)]
        n_frame_same += int(fs)
        dv = np.where(np.any(dx['opos32'][n] != dy['opos32'][n], -1))[0]
        diverge.append(int(dv[0]) - te if len(dv) else None)
        if te - K_MAIN < 0 or te + K_MAIN > T - 1:  # Sec.4.2 event definition (t_e >= k, t_e + k <= 100)
            continue
        n_usable += 1
        rx = predict_events(dx, [(n, te, cx)], K_MAIN, models, cands=cands, kappa=kappa)[0]
        ry = predict_events(dy, [(n, te, cy)], K_MAIN, models, cands=cands, kappa=kappa)[0]
        for r in (rx, ry):
            r['pair'] = n
            rows.append(r)
        out['pairs'].append({'n': n, 'te': te, 'frames_after_eat_identical': fs,
                             'pre_eat_frames_with_vision_diff': int(sum(pre_diff)),
                             'truth': [rx['truth'], ry['truth']],
                             'same_a': rx['pred_a'] == ry['pred_a'], 'same_b0': rx['pred_b0'] == ry['pred_b0'],
                             'same_b0_angle': rx['angerr_b0'] is not None and ry['angerr_b0'] is not None})
    out.update({'n_both_eat': n_both, 'n_identical_to_eat': n_same, 'n_frames_identical_after_eat': n_frame_same,
                'n_usable': n_usable, 'diverge_steps_after_eat': diverge})
    has = [r for r in rows if r['truth'] >= 0]
    out['summary'] = summarize(rows)
    out['frac_same_a'] = float(np.mean([p['same_a'] for p in out['pairs']])) if out['pairs'] else None
    out['frac_same_b0'] = float(np.mean([p['same_b0'] for p in out['pairs']])) if out['pairs'] else None
    if has:
        for b in ('a', 'b0', 'b1', 'b2'):
            out[f'd_minus_{b}'] = boot_diff(has, 'd', b, unit_key='pair')
        out['pmass_d_mean'] = float(np.mean([r['pmass_d'] for r in has]))
    out['rows'] = rows
    return out


def corner_state(d, n, t):
    """rendered state of the four corners (TL, BL, BR, TR) at frame t: colour code or 4 (gray)."""
    st = {}
    for c in range(4):
        st[_COL_CORNER[d['layout'][n]][c]] = c if d['available'][n, t, c] else 4
    return tuple(st[k] for k in ('TL', 'BL', 'BR', 'TR'))


def natural_pairs(d, models, kappa, eps=None):
    """Descriptive: eat events at left corners (red/green eaten) of different colour at the same corner.
    Split (requested 2026-10-24, outside the pre-registration): (i) the corner states rendered at the eat
    frame differ (layout readable from the remaining colours), (ii) they are identical (memory needed)."""
    ev = [(n, te, c) for n, te, c in events(d, K_MAIN) if c in (0, 1) and (eps is None or n in eps)]
    P = np.array([d['pos'][n, te] for n, te, _ in ev])
    D = np.array([unit(d['pos'][n, te] - d['pos'][n, te - K_MAIN]) for n, te, _ in ev])
    C = np.array([c for _, _, c in ev])
    TG = np.array([d['tgt'][n, te] for n, te, _ in ev])
    N = np.array([n for n, _, _ in ev])
    CS = [corner_state(d, n, te) for n, te, _ in ev]
    out = {}
    for dd in (1, 2):
        for th in (15, 30):
            pairs = []
            for i in range(len(ev)):
                for j in range(i + 1, len(ev)):
                    if N[i] == N[j] or C[i] == C[j] or TG[i] == TG[j]:
                        continue
                    if np.linalg.norm(P[i] - P[j]) <= dd and ang(D[i], D[j]) <= th:
                        pairs.append((i, j, np.linalg.norm(P[i] - P[j])))
            res = {}
            for cat in ('all', 'i', 'ii'):
                pc = [p for p in pairs if cat == 'all' or (cat == 'ii') == (CS[p[0]] == CS[p[1]])]
                used, match = set(), []
                for i, j, _ in sorted(pc, key=lambda x: x[2]):
                    if i not in used and j not in used:
                        used |= {i, j}
                        match.append((int(i), int(j)))
                eps = set(N[[p[0] for p in pc]]) | set(N[[p[1] for p in pc]]) if pc else set()
                res[cat] = {'n_pairs': len(pc), 'n_episodes': len(eps), 'n_matched': len(match), 'match': match}
            out[f'd{dd}_th{th}'] = res
    # predictions on the matched pairs (d = 2, theta = 30), per category
    for cat in ('all', 'i', 'ii'):
        m = out['d2_th30'][cat]['match']
        if not m:
            out[f'd2_th30_pred_{cat}'] = None
            continue
        rows = predict_events(d, [ev[i] for p in m for i in p], K_MAIN, models, kappa=kappa)
        for r, pid in zip(rows, [k for k, p in enumerate(m) for _ in p]):
            r['pair'] = pid
        has = [r for r in rows if r['truth'] >= 0]
        o = {'summary': summarize(rows),
             'frac_same_a': float(np.mean([rows[2 * i]['pred_a'] == rows[2 * i + 1]['pred_a'] for i in range(len(m))])),
             'frac_same_b0': float(np.mean([rows[2 * i]['pred_b0'] == rows[2 * i + 1]['pred_b0'] for i in range(len(m))]))}
        if len(m) >= 2:
            for b in ('a', 'b0', 'b1', 'b2'):
                o[f'd_minus_{b}'] = boot_diff(has, 'd', b, unit_key='pair')
        out[f'd2_th30_pred_{cat}'] = o
    out['events'] = [(int(n), int(te), int(c)) for n, te, c in ev]
    return out


def avail_only(d_fit, d):
    """How far the availability (+ initial gray) alone identifies the candidate at eat moments."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    def xy(dd, eps):
        X, y = [], []
        for n, te, _ in events(dd, K_MAIN):
            if n in eps:
                X.append(list(dd['available'][n, te]) + list(np.eye(4)[dd['gray0'][n]]))
                y.append(int(dd['a2_cand'][n]))
        return np.array(X), np.array(y)
    X, y = xy(d_fit, set(range(N_FIT)))
    clf = HistGradientBoostingClassifier(random_state=0).fit(X, y)
    Xe, ye = xy(d, set(range(N_FIT, len(d['eat']))))
    return {'acc_candidate': float((clf.predict(Xe) == ye).mean()), 'n': int(len(ye))}


def strip(o):
    if isinstance(o, dict):
        return {str(k): strip(v) for k, v in o.items() if k not in ('rows', 'rows_k5', 'events', 'angles', 'first_times', 'match')}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def main():
    import pickle
    os.makedirs(OUT, exist_ok=True)
    res = {}
    c2 = load('v15_s2_c2_p80', 'a2')
    kappa = fit_kappa(c2, range(N_FIT))
    res['kappa_fit_c2_p80'] = kappa
    # 5.1 / 5.2 / 5.4 / configs for every dataset
    sets = {'v15_s1_p80': 'a1', 'v15_s1_p50': 'a1'}
    for c in ('c1', 'c2'):
        for p in (80, 90, 50):
            sets[f'v15_s2_{c}_p{p}'] = 'a2'
    loaded = {}
    for name, ag in sets.items():
        d = c2 if name == 'v15_s2_c2_p80' else load(name, ag)
        loaded[name] = d
        res[name] = {'eat': eat_stats(d), 'turn': turn_stats(d), 'configs': config_counts(d)}
        print(name, 'eat', res[name]['eat']['frac_ge1'], 'turn k5', res[name]['turn'][5]['frac_gt45'], flush=True)
    # 5.3 predictions: (b) fitted on each dataset's own fit episodes; kappa from c2_p80 fit
    full = {}
    for name in sets:
        d = loaded[name]
        full[name] = analyse_prediction(d, d, name, kappa)
        res[name]['pred'] = full[name]
        print(name, 'k5', {k: v for k, v in full[name][5].items() if k.startswith('acc_')}, flush=True)
    res['avail_only_c2_p80'] = avail_only(c2, c2)
    # 5.5 generated pairs (main) and (3) control, natural pairs
    gx, gy = load('v15_s2_genx_p80', 'a2'), load('v15_s2_geny_p80', 'a2')
    res['generated'] = analyse_generated(gx, gy, c2, kappa)
    print('generated usable', res['generated']['n_usable'], flush=True)
    for p in (80, 50):
        x3, xp3 = load(f'v15_s2_c3x_p{p}', 'a2'), load(f'v15_s2_c3xp_p{p}', 'a2')
        res[f'control3_p{p}'] = analyse_generated(x3, xp3, c2, None, cands=('X', 'Xp'), corner_check=False, tag='c3')
        print('control3', p, res[f'control3_p{p}']['n_usable'], flush=True)
    models = {lv: fit_b(c2, set(range(N_FIT)), K_MAIN, lv) for lv in (0, 1, 2)}
    res['natural_c2_p80'] = natural_pairs(c2, models, kappa)
    # predictions are only valid out of sample: (b) was fitted on episodes 0..N_FIT-1
    res['natural_c2_p80_eval'] = natural_pairs(c2, models, kappa, eps=set(range(N_FIT, len(c2['eat']))))
    with open(os.path.join(OUT, 'stage0_full.pkl'), 'wb') as f:
        pickle.dump(res, f)
    with open(os.path.join(OUT, 'stage0_summary.json'), 'w') as f:
        json.dump(strip(res), f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('saved', OUT)


if __name__ == '__main__':
    main()
