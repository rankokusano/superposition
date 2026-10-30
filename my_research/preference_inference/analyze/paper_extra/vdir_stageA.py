"""
Paper round 12, V-dir stage A (docs/paper_extra_log.md Sec.36; instructions docs/paper_instructions_round12.md).
Exploratory, outside the pre-registration; definitions and procedure committed before computing (fa20545).
Saved arrays and existing dataset metadata only (no model is run, CPU only).
  side A = bins 6, 7 (c+ pleasant), side B = bins 0, 1; axis n = (u_A - u_B) / |u_A - u_B|.
  p_W = (other_position[b] - other_position[a - 1]) . n for the window of frames a..b; s_dir = sign(p_W), |p_W| < tau
  (or p_W = 0) -> no direction (left out of the binary analyses).
  s_red = +1 if red is on side A (layout reference points); s_pref = s_red (red-liked) / -s_red (green-liked).
  Q_e(W) = mean_W L(B pleasant) - mean_W L(A pleasant); oracle series from the E2 true / opp (= -P) arrays (v14-C:
  v14 ref_red / ref_green), c+- series from the v14 matching arrays (cp / cm).
  Regression Q = a + b_pref s_pref + b_dir s_dir + b_red s_red (+ (i) initial positions on n, (ii) p_W instead of
  s_dir, (iii) s_pref s_dir and s_dir s_red); s_pref dropped on the green-preference sets (b_red = b_red - b_pref).
  Per point (ep160..200) -> 5-point mean = pair value; 6-pair t CI (learning variation), within-pair episode
  bootstrap (evaluation-episode variation), two-stage bootstrap (both, reference).
-> data/result/paper_extra/Vdir_A/summary.json, tables.md, points.npz
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_PI, 'analyze', 'v14'), os.path.join(_PI, 'analyze', 'v13'), os.path.join(_PI, 'analyze', 'v12'),
          os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)
import a_cross_eval  # noqa  (registers green_a2p80 / pref_a2p80)
from across_pairs_v13 import test as ttest  # noqa
from eval_s2_v11 import load_set  # noqa
from sim_v10.layout import LANDMARKS, CODE_LAYOUT  # noqa
from v11core.pleasure import BIN_UNIT  # noqa

RES = os.path.join(_PI, 'data', 'result')
RUNS = os.path.join(RES, 'paper_extra', 'E2', 'runs')
INV = os.path.join(RES, 'v14', 'inverse')
CSR = os.path.join(RES, 'paper_extra', 'C_signrand', 'eval', 'inverse')
OUT = os.path.join(RES, 'paper_extra', 'Vdir_A')
SETS = ('pref', 'green', 'pref_a2p80', 'green_a2p80')
CONDS = ('none', 'p50', 'p80', 'sr80', 'sr50', 'sw')
NAME = {'none': '経験なし', 'p50': '経験50%', 'p80': '経験80%', 'sr80': '符号無作為（経験80%）',
        'sr50': '符号無作為（経験50%）', 'sw': '好み切替（v14-C）'}
PAIRS = {c: list(range(3)) if c == 'sw' else list(range(6)) for c in CONDS}
EPS = (160, 170, 180, 190, 200)
ERRS = {'FP': 'mse', 'L1': 'l1'}
# window -> (frames a..b for Q, frames for the direction)
WINS = {'1-25': ((1, 25), (1, 25)), '26-50': ((26, 50), (26, 50)), '51-100': ((51, 100), (51, 100)),
        '1-50': ((1, 50), (1, 50)), '1-100': ((1, 100), (1, 100)), 'past': ((26, 50), (1, 25))}
TAUS = (0.0, 0.5, 1.0, 2.0)
B = 1000
REF_A, REF_B = np.array([-9.0, 9.0]), np.array([-9.0, -9.0])
# c+- arrays: (dir, prefix) per error and condition
CPM = {'FP': {'none': (INV, 'v12mse_random'), 'p50': (INV, 'v12mse_mixed'), 'p80': (INV, 'v13mse_mixed'),
              'sr80': (CSR, 'csrmse_signrand'), 'sw': (INV, 'v14cmse_mixedpref')},
       'L1': {'none': (INV, 'v12_random'), 'p50': (INV, 'v12_mixed'), 'p80': (INV, 'v13_mixed'),
              'sr80': (CSR, 'csr_signrand'), 'sw': (INV, 'v14c_mixedpref')}}
# E2 fam -> v14 / C_signrand files with the same model (for the oracle-equivalence check)
EQUIV = {'FP': {'none': 'v12mse_random', 'p50': 'v12mse_mixed', 'p80': 'v13mse_mixed', 'sr80': 'csrmse_signrand'},
         'L1': {'none': 'v12_random', 'p50': 'v12_mixed', 'p80': 'v13_mixed', 'sr80': 'csr_signrand'}}


def unit(v):
    return v / np.linalg.norm(v)


U_A, U_B = unit(BIN_UNIT[[6, 7]].mean(0)), unit(BIN_UNIT[[0, 1]].mean(0))
NHAT = unit(U_A - U_B)


# ------------------------------------------------------------------ metadata
def meta(name):
    d = load_set(name)
    lt = d['layout_t']
    assert (lt == lt[:, :1]).all() and (lt[:, 0] == d['layout']).all()
    red_dir = {L: float(np.sign(LANDMARKS[CODE_LAYOUT[L]]['Red'] @ NHAT)) for L in (0, 1)}
    s_red = np.array([red_dir[int(L)] for L in d['layout']])
    s_pref = np.where(d['a2_liked'] == 1, s_red, -s_red)
    op, sp = d['other_position'].astype(np.float64), d['self_position'].astype(np.float64)
    m = {'layout': d['layout'], 'liked': d['a2_liked'], 's_red': s_red, 's_pref': s_pref, 'red_dir': red_dir,
         'a2y0': op[:, 0] @ NHAT, 'a1y0': sp[:, 0] @ NHAT,
         'a2dA': np.linalg.norm(op[:, 0] - REF_A, axis=1), 'a2dB': np.linalg.norm(op[:, 0] - REF_B, axis=1),
         'a1dA': np.linalg.norm(sp[:, 0] - REF_A, axis=1), 'a1dB': np.linalg.norm(sp[:, 0] - REF_B, axis=1),
         'p': {}, 'mag': {}}
    for w, (_, (a, b)) in WINS.items():
        dx = op[:, b] - op[:, a - 1]
        m['p'][w], m['mag'][w] = dx @ NHAT, np.linalg.norm(dx, axis=1)
    disp = op[:, 51] - op[:, 1]                       # fp_direction_v14 definition
    m['dir_answer'] = np.where(disp @ BIN_UNIT[[6, 7]].mean(0) / np.linalg.norm(BIN_UNIT[[6, 7]].mean(0)) >
                               disp @ BIN_UNIT[[0, 1]].mean(0) / np.linalg.norm(BIN_UNIT[[0, 1]].mean(0)), 1, -1)
    return m


def sdir(m, w, tau):
    p = m['p'][w]
    keep = (np.abs(p) >= tau) & (p != 0)
    return np.sign(p), keep


# ------------------------------------------------------------------ losses
def path_ok(f):
    return os.path.exists(f)


def load_pair(series, cond, err, st, s):
    """-> list over points of (L_A, L_B, U) per-step arrays (300, 100), or None if not saved."""
    out = []
    for e in EPS:
        if series == 'oracle' and cond != 'sw':
            f = os.path.join(RUNS, f'{cond}_{ERRS[err]}_s{s}_ep{e}_{st}.npz')
            if not path_ok(f):
                return None
            z = np.load(f)
            Lt, Lo, liked = z[f'{err}_true'].astype(np.float64), z[f'{err}_opp'].astype(np.float64), z['a2_liked']
            Lr = np.where(liked[:, None] == 1, Lt, Lo)
            Lg = np.where(liked[:, None] == 1, Lo, Lt)
            U = Lo - Lt
        elif series == 'oracle':
            dr, pre = CPM[err]['sw']
            f = os.path.join(dr, f'{pre}_{s}_ep{e}_{st}.npz')
            if not path_ok(f):
                return None
            z = np.load(f)
            Lr, Lg, liked = z[f'{err}_ref_red'].astype(np.float64), z[f'{err}_ref_green'].astype(np.float64), z['a2_liked']
            U = np.where(liked[:, None] == 1, Lg - Lr, Lr - Lg)
        else:
            if cond not in CPM[err]:
                return None
            dr, pre = CPM[err][cond]
            f = os.path.join(dr, f'{pre}_{s}_ep{e}_{st}.npz')
            if not path_ok(f):
                return None
            z = np.load(f)
            out.append({'A': z[f'{err}_cp'].astype(np.float64), 'B': z[f'{err}_cm'].astype(np.float64), 'U': None,
                        'layout': z['layout'], 'liked': z['a2_liked'], 'cs': z['correct_sign']})
            continue
        out.append({'Lr': Lr, 'Lg': Lg, 'U': U, 'layout': z['layout'], 'liked': liked,
                    'cs': z['correct_sign'] if 'correct_sign' in z.files else None})
    return out


def q_window(L, w):
    a, b = WINS[w][0]
    return L[:, a - 1:b].mean(1)


# ------------------------------------------------------------------ regression
def design(m, w, tau, model, green_only):
    sd, keep = sdir(m, w, tau)
    if model == 'cont':
        keep = np.ones_like(keep)
    cols = {}
    if not green_only:
        cols['pref'] = m['s_pref']
    cols['dir'] = m['p'][w] if model == 'cont' else sd
    cols['red'] = m['s_red']
    if model == 'cov':
        cols['a2y0'], cols['a1y0'] = m['a2y0'], m['a1y0']
    if model == 'inter':
        if not green_only:
            cols['pref_x_dir'] = m['s_pref'] * sd
        cols['dir_x_red'] = sd * m['s_red']
    names = list(cols)
    X = np.column_stack([np.ones(len(sd))] + [cols[k] for k in names])[keep]
    return X, ['a'] + names, keep


def solve(X, Y):
    """X (..., n, k), Y (..., n, r) -> coef (..., k, r); nan where singular."""
    XtX = np.swapaxes(X, -1, -2) @ X
    XtY = np.swapaxes(X, -1, -2) @ Y
    ok = np.linalg.cond(XtX) < 1e10
    XtX = np.where(ok[..., None, None], XtX, np.eye(XtX.shape[-1]))
    c = np.linalg.solve(XtX, XtY)
    return np.where(ok[..., None, None], c, np.nan)


def vif(X):
    Z = X[:, 1:]
    out = []
    for j in range(Z.shape[1]):
        o = np.column_stack([np.ones(len(Z))] + [Z[:, i] for i in range(Z.shape[1]) if i != j])
        y = Z[:, j]
        c, *_ = np.linalg.lstsq(o, y, rcond=None)
        r = y - o @ c
        tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (r ** 2).sum() / tot if tot > 0 else np.nan
        out.append(float(1 / (1 - r2)) if r2 < 1 else float('inf'))
    return out


def tsum(v):
    v = [x for x in v if np.isfinite(x)]
    if len(v) < 2:
        return {'mean': float(np.mean(v)) if v else None, 'ci95': None, 'n': len(v)}
    r = ttest(v)
    return {'mean': r['mean'], 'ci95': r['ci95'], 'n': r['n'], 'n_pos': r['n_pos'], 'sd': float(np.std(v, ddof=1))}


def pct(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else None


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    M = {st: meta(st) for st in SETS}
    R = {'axis': {'u_A': U_A.tolist(), 'u_B': U_B.tolist(), 'n_hat': NHAT.tolist(),
                  'red_dir_by_layout': {st: {str(k): v for k, v in M[st]['red_dir'].items()} for st in SETS}},
         'checks': {}, 'missing': [], 'counts': {}, 'corr': {}, 'vif': {}, 'coef': {}, 'cells': {}, 'share': {},
         'diff': {}, 'stageB': {}}
    # ---- checks on the definitions
    ck = R['checks']
    for st in SETS:
        m = M[st]
        sp = m['s_pref']
        ck[f'{st}|s_pref_eq_red_or_minus_red'] = bool(np.array_equal(sp, np.where(m['liked'] == 1, m['s_red'], -m['s_red'])))
        sd, _ = sdir(m, '1-50', 0.0)
        ck[f'{st}|dir_answer_agreement_1-50_tau0'] = float((sd == m['dir_answer']).mean())
        ck[f'{st}|dir_answer_n_disagree'] = int((sd != m['dir_answer']).sum())
    # ---- load Q arrays
    Q = {}           # (series, cond, err, set) -> {win: array (npairs, 5, 300)}, pairs list
    ident = []
    cs_ok = []
    equiv = {}
    for series in ('oracle', 'cpm'):
        for cond in CONDS:
            for err in ERRS:
                for st in SETS:
                    pairs, arr = [], {w: [] for w in WINS}
                    for s in PAIRS[cond]:
                        pts = load_pair(series, cond, err, st, s)
                        if pts is None:
                            R['missing'].append(f'{series}|{cond}|{err}|{st}|s{s}')
                            continue
                        m = M[st]
                        for z in pts:
                            assert np.array_equal(z['layout'], m['layout']) and np.array_equal(z['liked'], m['liked'])
                            if z['cs'] is not None:
                                cs_ok.append(bool(np.array_equal(z['cs'], m['s_pref'])))
                        pairs.append(s)
                        for w in WINS:
                            qs = []
                            for z in pts:
                                if series == 'oracle':
                                    LA = np.where(m['s_red'][:, None] == 1, z['Lr'], z['Lg'])
                                    LB = np.where(m['s_red'][:, None] == 1, z['Lg'], z['Lr'])
                                    q = q_window(LB, w) - q_window(LA, w)
                                    u = q_window(z['U'], w)
                                    ident.append(float(np.abs(u - m['s_pref'] * q).max()))
                                else:
                                    q = q_window(z['B'], w) - q_window(z['A'], w)
                                qs.append(q)
                            arr[w].append(qs)
                    if pairs:
                        Q[(series, cond, err, st)] = ({w: np.array(arr[w]) for w in WINS}, pairs)
    ck['identity_U_eq_spref_Q_max_abs'] = max(ident)
    ck['correct_sign_eq_s_pref_all'] = all(cs_ok)
    ck['correct_sign_files_checked'] = len(cs_ok)
    # oracle equivalence: E2 true/opp vs v14 ref_red/ref_green (same model)
    for err, mp in EQUIV.items():
        for fam, pre in mp.items():
            for st in ('pref', 'green'):
                dr = CSR if fam == 'sr80' else INV
                n_same = n_tot = 0
                for s in range(6):
                    for e in EPS:
                        f1 = os.path.join(RUNS, f'{fam}_{ERRS[err]}_s{s}_ep{e}_{st}.npz')
                        f2 = os.path.join(dr, f'{pre}_{s}_ep{e}_{st}.npz')
                        if not (os.path.exists(f1) and os.path.exists(f2)):
                            continue
                        a, b = np.load(f1), np.load(f2)
                        liked = a['a2_liked'][:, None] == 1
                        Lr = np.where(liked, a[f'{err}_true'], a[f'{err}_opp'])
                        Lg = np.where(liked, a[f'{err}_opp'], a[f'{err}_true'])
                        n_tot += 1
                        n_same += int(np.array_equal(Lr, b[f'{err}_ref_red']) and np.array_equal(Lg, b[f'{err}_ref_green']))
                equiv[f'{err}|{fam}|{st}'] = f'{n_same}/{n_tot}'
    ck['oracle_E2_eq_v14_ref'] = equiv
    # ---- counts, correlations, VIF (data only)
    for st in SETS:
        m = M[st]
        for w in WINS:
            for tau in TAUS:
                sd, keep = sdir(m, w, tau)
                c = {}
                for lk in (1, 0):
                    for L in (0, 1):
                        sel = (m['liked'] == lk) & (m['layout'] == L)
                        c[f"{'red' if lk else 'green'}|L{L}"] = {'A': int((sel & keep & (sd > 0)).sum()),
                                                                  'B': int((sel & keep & (sd < 0)).sum()),
                                                                  'none': int((sel & ~keep).sum())}
                agree = float((sd[keep] == m['s_pref'][keep]).mean()) if keep.any() else None
                R['counts'][f'{st}|{w}|{tau}'] = {'cells': c, 'n_dir': int(keep.sum()), 'n_none': int((~keep).sum()),
                                                  'pref_dir_agree': agree,
                                                  'red_dir_agree': float((sd[keep] == m['s_red'][keep]).mean()) if keep.any() else None}
                for model in ('main', 'cov', 'inter', 'cont'):
                    if model == 'cont' and tau != 0.0:
                        continue
                    X, names, _ = design(m, w, tau, model, st.startswith('green'))
                    R['vif'][f'{st}|{w}|{tau}|{model}'] = dict(zip(names[1:], vif(X)))
            sd0, _ = sdir(m, w, 0.0)
            V = {'s_pref': m['s_pref'], 's_red': m['s_red'], 's_dir': sd0, 'p': m['p'][w], 'a2y0': m['a2y0'],
                 'a1y0': m['a1y0'], 'a2dA': m['a2dA'], 'a2dB': m['a2dB'], 'a1dA': m['a1dA'], 'a1dB': m['a1dB'],
                 'mag': m['mag'][w]}
            ks = list(V)
            C = np.full((len(ks), len(ks)), np.nan)
            for i, a in enumerate(ks):
                for j, b in enumerate(ks):
                    if V[a].std() > 0 and V[b].std() > 0:
                        C[i, j] = np.corrcoef(V[a], V[b])[0, 1]
            R['corr'][f'{st}|{w}'] = {'vars': ks, 'r': [[None if np.isnan(x) else float(x) for x in row] for row in C]}
    # ---- regressions
    points = {}
    for (series, cond, err, st), (QW, pairs) in Q.items():
        m = M[st]
        green_only = st.startswith('green')
        for w in WINS:
            Y = QW[w]                                           # (npairs, 5, 300)
            for model in ('main', 'cov', 'inter', 'cont'):
                for tau in (TAUS if model != 'cont' else (0.0,)):
                    X, names, keep = design(m, w, tau, model, green_only)
                    Yk = Y[:, :, keep]                          # (np, 5, n)
                    coef = solve(X[None], np.swapaxes(Yk, 1, 2))  # (np, k, 5)
                    key = f'{series}|{cond}|{err}|{st}|{w}|{tau}|{model}'
                    points[key] = coef
                    pv = coef.mean(2)                           # (np, k)
                    blk = {'pairs': pairs, 'n_used': int(keep.sum()), 'names': names, 'per_pair': {}, 't': {}}
                    for j, nm in enumerate(names):
                        blk['per_pair'][nm] = pv[:, j].tolist()
                        blk['t'][nm] = tsum(pv[:, j])
                    if 'pref' in names:
                        d = pv[:, names.index('dir')] - pv[:, names.index('pref')]
                        blk['per_pair']['dir_minus_pref'] = d.tolist()
                        blk['t']['dir_minus_pref'] = tsum(d)
                    if model == 'main':
                        boot_block(blk, X, Yk, names)
                    R['coef'][key] = blk
            # cells and shares (Q itself, no regression)
            for tau in TAUS:
                sd, keep = sdir(m, w, tau)
                cells = {}
                for lk in (1, 0):
                    for L in (0, 1):
                        for dn, dsel in (('A', keep & (sd > 0)), ('B', keep & (sd < 0)), ('none', ~keep)):
                            sel = (m['liked'] == lk) & (m['layout'] == L) & dsel
                            if not sel.any():
                                continue
                            v = Y[:, :, sel].mean(2).mean(1)     # pair values
                            pos = (np.sign(Y[:, :, sel]) * 0.5 + 0.5).mean(2).mean(1)
                            cells[f"{'red' if lk else 'green'}|L{L}|{dn}"] = {'n': int(sel.sum()), 'Q': tsum(v.tolist()),
                                                                               'share_A_smaller': tsum(pos.tolist())}
                R['cells'][f'{series}|{cond}|{err}|{st}|{w}|{tau}'] = cells
                hit_pref = ((np.sign(Y) == m['s_pref']) + 0.5 * (Y == 0)).mean(2).mean(1)
                hit_dir = ((np.sign(Y[:, :, keep]) == sd[keep]) + 0.5 * (Y[:, :, keep] == 0)).mean(2).mean(1)
                R['share'][f'{series}|{cond}|{err}|{st}|{w}|{tau}'] = {'pref_side': tsum(hit_pref.tolist()),
                                                                        'dir_side': tsum(hit_dir.tolist())}
        print('done', series, cond, err, st, flush=True)
    # ---- experience dependence (paired differences)
    comps = (('p50', 'none'), ('p50', 'sr50'), ('p80', 'none'), ('p80', 'sr80'))
    for series in ('oracle', 'cpm'):
        for err in ERRS:
            for st in SETS:
                for w in WINS:
                    for tau in TAUS:
                        for a, b in comps:
                            ka, kb = (series, a, err, st), (series, b, err, st)
                            key = f'{series}|{err}|{st}|{w}|{tau}|{a}-{b}'
                            if ka not in Q or kb not in Q:
                                R['diff'][key] = '未保存'
                                continue
                            R['diff'][key] = diff_block(Q[ka], Q[kb], M[st], w, tau, st.startswith('green'))
    # ---- stage-B material (main model, per pair SE from the episode bootstrap)
    for key, blk in R['coef'].items():
        series, cond, err, st, w, tau, model = key.split('|')
        if model != 'main':
            continue
        out = {}
        for nm in ('dir', 'pref', 'dir_minus_pref'):
            if nm not in blk['t'] or blk['t'][nm]['mean'] is None:
                continue
            b, se = blk['t'][nm]['mean'], blk['boot_se'].get(nm)
            sd = blk['t'][nm].get('sd')
            n = blk['n_used']
            out[nm] = {'mean': b, 'sd_pairs': sd, 'se_episode_mean': se,
                       'N_episodes_80pct': (float(n * (2.8 * se / abs(b)) ** 2) if se and b else None),
                       'n_pairs_80pct': (float((2.8 * sd / abs(b)) ** 2) if sd and b else None)}
        R['stageB'][key] = out
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), ensure_ascii=False, indent=0)
    np.savez_compressed(os.path.join(OUT, 'points.npz'), **{k.replace('|', '__'): v for k, v in points.items()})
    print('checks', json.dumps(ck, ensure_ascii=False))
    print('missing', len(R['missing']))


def boot_block(blk, X, Yk, names):
    """episode bootstrap per pair and two-stage bootstrap; adds 'boot_pair', 'boot_se', 'boot2'."""
    npairs, _, n = Yk.shape
    k = X.shape[1]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, (B, n))
    Xb = X[idx]                                                  # (B, n, k)
    bp = np.zeros((npairs, B, k))
    for i in range(npairs):
        Yb = np.swapaxes(Yk[i][:, idx], 0, 1)                    # (B, 5, n)
        bp[i] = solve(Xb, np.swapaxes(Yb, 1, 2)).mean(2)
    blk['boot_pair'], blk['boot_se'] = {}, {}
    for j, nm in enumerate(names):
        blk['boot_pair'][nm] = [pct(bp[i, :, j]) for i in range(npairs)]
        blk['boot_se'][nm] = float(np.nanmean([np.nanstd(bp[i, :, j], ddof=1) for i in range(npairs)]))
    if 'pref' in names:
        d = bp[:, :, names.index('dir')] - bp[:, :, names.index('pref')]
        blk['boot_pair']['dir_minus_pref'] = [pct(d[i]) for i in range(npairs)]
        blk['boot_se']['dir_minus_pref'] = float(np.nanmean([np.nanstd(d[i], ddof=1) for i in range(npairs)]))
    # two-stage
    rng = np.random.default_rng(0)
    pi = rng.integers(0, npairs, (B, npairs))
    ei = rng.integers(0, n, (B, npairs, n))
    Xb2 = X[ei]                                                  # (B, np, n, k)
    Yb2 = Yk[pi[:, :, None, None], np.arange(5)[None, None, :, None], ei[:, :, None, :]]   # (B, np, 5, n)
    c2 = solve(Xb2, np.swapaxes(Yb2, -1, -2)).mean(-1).mean(1)  # (B, k)
    blk['boot2'] = {nm: pct(c2[:, j]) for j, nm in enumerate(names)}
    if 'pref' in names:
        blk['boot2']['dir_minus_pref'] = pct(c2[:, names.index('dir')] - c2[:, names.index('pref')])


def diff_block(qa, qb, m, w, tau, green_only):
    (QA, pa), (QB, pb) = qa, qb
    common = [s for s in pa if s in pb]
    ia, ib = [pa.index(s) for s in common], [pb.index(s) for s in common]
    X, names, keep = design(m, w, tau, 'main', green_only)
    YA, YB = QA[w][ia][:, :, keep], QB[w][ib][:, :, keep]
    ca = solve(X[None], np.swapaxes(YA, 1, 2)).mean(2)
    cb = solve(X[None], np.swapaxes(YB, 1, 2)).mean(2)
    out = {'pairs': common, 't': {}, 'boot2': {}}
    terms = {nm: (ca[:, j] - cb[:, j]) for j, nm in enumerate(names)}
    if 'pref' in names:
        j1, j2 = names.index('dir'), names.index('pref')
        terms['dir_minus_pref'] = (ca[:, j1] - ca[:, j2]) - (cb[:, j1] - cb[:, j2])
    for nm, v in terms.items():
        out['t'][nm] = tsum(v.tolist())
    rng = np.random.default_rng(0)
    npairs, n = len(common), int(keep.sum())
    pi = rng.integers(0, npairs, (B, npairs))
    ei = rng.integers(0, n, (B, npairs, n))
    Xb = X[ei]
    res = []
    for Y in (YA, YB):
        Yb = Y[pi[:, :, None, None], np.arange(5)[None, None, :, None], ei[:, :, None, :]]
        res.append(solve(Xb, np.swapaxes(Yb, -1, -2)).mean(-1).mean(1))
    d = res[0] - res[1]
    for j, nm in enumerate(names):
        out['boot2'][nm] = pct(d[:, j])
    if 'pref' in names:
        out['boot2']['dir_minus_pref'] = pct(d[:, names.index('dir')] - d[:, names.index('pref')])
    return out


if __name__ == '__main__':
    main()
