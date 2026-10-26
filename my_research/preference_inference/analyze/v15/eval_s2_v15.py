"""
v15 stage-2 evaluation (pre-registered in docs/v15_experiment_log.md Sec.16, 16.7). One trained pair of models
per tag (phase 1 `v15_s2p1_<tag>_mse` ep150, phase 2 `v15_s2p2_<tag>` ep150), tags {exp,noexp}_{v4,v15i}.

Protocols (all with util.seed_all(0); per-episode vision masks / motion-hide flags are passed as tensors):
  P1-a/b  phase-1 model, usual run (training protocol: vision p 0 at t=0, 0.99 after; random hide schedule) on
          v15_s1_p80_half_3000 test: h1 -> A-1 position R^2 on motion-visible frames, h2 -> A-2 position R^2.
  P1-c    same data, A-1 eat moments with a next target: usual run to t_e-1; from t_e for K=10 steps vision masked,
          motion hidden, pleasure held at its value at t_e (the hold rule). d = p^(t_e+K) - p^(t_e) from h1.
  P2-a    phase-1 model on v15_s2_c2_p80_half_6000 test, process-2 = [0,0,P_A2(t),1] (stored a2_pleasure):
          vision visible (p 0) and motion visible to t_e-1; from t_e vision masked, P_A2 held at its t_e value; h2.
  P2-b    phase-1 model, same data: process-2 = candidate X or Y pleasure (A-2's view x candidate values); vision
          visible to t_e, masked for the next 10 steps; choose the candidate with the smaller summed L1 of the
          predicted views over the post-eat steps (main) and over the pre-eat steps 0..t_e-1 (descriptive).
  P2-c    phase-2 model (G) on the generated pairs: training protocol (vision masked for t>0, G sees the views)
          to t_e; for t > t_e the input view (network and G) is frozen at frame t_e; d from h2; blue vs cyan.
          Also on the control (3) pairs and, descriptively, on all A-2 eat moments of the stage-2 test.
  P2-c'   phase-1 model on the generated pairs, v14-A protocol (vision only at t=0), process-2 = candidate X or Y
          pleasure; choose by the summed L1 over steps 0..t_e-1 (pre-eat frames only); predicted next target =
          the chosen candidate's (X -> blue, Y -> cyan).
Readouts: Ridge(1) from h (after input frame t) to the position at frame t+1, fitted on the usual run of one half
of the episodes (or pairs) and applied to the other half (2-fold by episode / pair).

    python analyze/v15/eval_s2_v15.py --tag exp_v4 [--device cuda:0]
Writes data/result/v15/stage2/eval_<tag>.json and .npz.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
import util  # noqa
from model import model_v15  # noqa
from analyze.v15 import stage0_v15 as S0  # noqa
from v15core.pleasure import pleasure_np, rank_values  # noqa
from v15core.world import COLORS  # noqa
from sklearn.linear_model import Ridge  # noqa
from sklearn.metrics import r2_score  # noqa

DATA = os.path.join(_PI_DIR, 'data', 'data')
RES = os.path.join(_PI_DIR, 'data', 'result')
OUT = os.path.join(RES, 'v15', 'stage2')
K = 10
T = 101
BOOT = 2000
CAND_RANK = {'X': ('Red', 'Blue', 'Cyan', 'Green'), 'Y': ('Green', 'Cyan', 'Blue', 'Red')}
CUBE_C = S0.CUBE_C  # [layout][colour] -> cube centre


# ------------------------------------------------------------------ models and data
def load(exp, epoch, dev):
    cfg = util.load_config(os.path.join(_PI_DIR, 'config', 'exp', exp + '.yml'))
    mcfg = util.load_config(os.path.join(_PI_DIR, 'config', 'model', cfg.model.name, cfg.model.config + '.yml'))
    m = getattr(model_v15, cfg.model.name)(mcfg).to(dev)
    m.load_state_dict(torch.load(os.path.join(RES, exp, '0', 'model', f'{epoch:05d}.pth'), map_location='cpu')['model'])
    return m.eval()


def read(name, split, keys, n=None):
    with h5py.File(os.path.join(DATA, name, 'data.h5'), 'r') as f:
        g = f[split]
        return {k: g[k][:n] if n else g[k][()] for k in keys}


def cand_pleasure(ov):
    """ov (N, T, 16, 64, 3) uint8 -> {'X': (N, T, 8), 'Y': ...} pleasure of A-2's view with candidate values."""
    return {c: pleasure_np(ov, rank_values(r)) for c, r in CAND_RANK.items()}


# ------------------------------------------------------------------ the generic runner
@torch.no_grad()
def run(m, sv, sm, dev, pmask, hidden=None, p2=None, frame=None, batch=50, need_err=False):
    """sv (N,T,16,64,3) uint8, sm (N,T,2). pmask(t) -> (N,) float; hidden(t) -> (N,) bool or None;
    p2(t) -> (N,11) float or None; frame(t) -> (N,) int index of the view fed at step t (default t).
    Returns h1, h2 (N,100,128), err (N,100) = L1 of predicted view t+1 (if need_err), q2 (N,100,8) if any."""
    util.seed_all(0)
    N = sv.shape[0]
    H1, H2, ERR, Q2, G = [], [], [], [], []
    for b0 in range(0, N, batch):
        idx = np.arange(b0, min(N, b0 + batch))
        B = len(idx)
        m.init_state(B)
        h1s, h2s, errs, q2s, gs = [], [], [], [], []
        for t in range(T - 1):
            fi = np.full(B, t) if frame is None else frame(t)[idx]
            x = {'self_vision': torch.from_numpy(util.scale_vision(util.transpose_vision(sv[idx, fi], dim=4))).to(dev),
                 'self_motion': torch.from_numpy(sm[idx, t]).to(dev)}
            pm = torch.from_numpy(pmask(t)[idx].astype(np.float32))[:, None].to(dev)
            kw = {}
            if hidden is not None:
                kw['hidden_override'] = torch.from_numpy(hidden(t)[idx])
            if p2 is not None:
                kw['p2_override'] = torch.from_numpy(p2(t)[idx].astype(np.float32)).to(dev)
            out = m(x, pm, pm, **kw)
            st = m.superposition_module.state
            h1s.append(st['self'][0].cpu().numpy()); h2s.append(st['other'][0].cpu().numpy())
            if hasattr(m, 'motion_generator_module'):
                gs.append(m.motion_generator_module.state['motion_generator'][0].cpu().numpy())
            if 'q2_hat' in out:
                q2s.append(out['q2_hat'].cpu().numpy())
            if need_err:
                p01 = ((out['self_vision'] + 1) / 2).clamp(0, 1)
                tr = torch.from_numpy(util.transpose_vision(sv[idx, t + 1], dim=4).astype(np.float32) / 255).to(dev)
                errs.append((p01 - tr).abs().mean((1, 2, 3)).cpu().numpy())
        H1.append(np.stack(h1s, 1)); H2.append(np.stack(h2s, 1))
        if errs:
            ERR.append(np.stack(errs, 1))
        if q2s:
            Q2.append(np.stack(q2s, 1))
        if gs:
            G.append(np.stack(gs, 1))
    out = {'h1': np.concatenate(H1), 'h2': np.concatenate(H2)}
    if ERR:
        out['err'] = np.concatenate(ERR)
    if Q2:
        out['q2'] = np.concatenate(Q2)
    if G:
        out['g'] = np.concatenate(G)
    return out


def schedule(m, N):
    """the model's random hide schedule for N episodes (same seed as run())."""
    util.seed_all(0)
    hs = []
    for b0 in range(0, N, 50):
        B = min(50, N - b0)
        m.init_state(B)
        hs.append(m._hidden.numpy().copy() if m.motion_masking else np.zeros((B, 100), bool))
    return np.concatenate(hs)


# ------------------------------------------------------------------ readouts and metrics
def folds(n):
    a = np.arange(n)
    return [(a % 2 == 0), (a % 2 == 1)]


def readout(h_fit, pos_fit, mask_fit, h_apply, folds_):
    """per fold: Ridge on fit frames of the other fold's episodes -> applied to this fold. h (N,100,128),
    pos (N,101,2) -> positions at frame t+1. Returns (N,100,2)."""
    out = np.zeros(h_apply.shape[:2] + (2,))
    for f in folds_:
        tr = ~f
        X = h_fit[tr][mask_fit[tr]]
        y = pos_fit[tr][:, 1:][mask_fit[tr]]
        r = Ridge(alpha=1.0).fit(X, y)
        out[f] = r.predict(h_apply[f].reshape(-1, h_apply.shape[-1])).reshape(f.sum(), 100, 2)
    return out


def r2_cv(h, pos, mask, folds_):
    pred = readout(h, pos, mask, h, folds_)
    return float(r2_score(pos[:, 1:][mask], pred[mask]))


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-12), 0.)


def boot_mean(v, units, seed=0):
    v, units = np.asarray(v, float), np.asarray(units)
    us = np.unique(units)
    idx = {u: np.where(units == u)[0] for u in us}
    rng = np.random.RandomState(seed)
    bs = [v[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(BOOT)]
    return {'mean': float(v.mean()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], 'n': int(len(v))}


def first_informative(eat, tgt):
    """per episode: (t_e, eaten colour) of the first eat with a next target, usable at K (t_e>=5, t_e+K<=100)."""
    out = {}
    for n in range(eat.shape[0]):
        for t in np.where(eat[n] >= 0)[0]:
            te = t + 1
            if 5 <= te <= T - 1 - K and tgt[n, te] >= 0:
                out[n] = (int(te), int(eat[n, t]))
                break
    return out


def next_metrics(d, pos_te, lay, avail_te, truth):
    """d (2,), true position at t_e, layout, available (4,), true next target colour -> (c_next, correct)."""
    u = unit(CUBE_C[lay][truth] - pos_te)
    c = float(np.dot(unit(d), u))
    cols = [c_ for c_ in range(4) if avail_te[c_]]
    bear = [S0.ang(d, CUBE_C[lay][c_] - pos_te) for c_ in cols]
    return c, float(cols[int(np.argmin(bear))] == truth) if np.linalg.norm(d) > 1e-9 else 0.0


def bc_metrics(d, pos_te, lay, avail_te, truth):
    """Sec.19 items 5-6, only when blue (2) and cyan (3) are both available and the next target is one of them:
    diff = cos(d, next corner) - cos(d, the other of blue/cyan) (cancels a common 'leave the corner' direction);
    blue = 1 if d is closer in angle to blue than to cyan (A-1's own next target after red: projection index)."""
    if not (avail_te[2] and avail_te[3]) or truth not in (2, 3):
        return None, None
    cb = float(np.dot(unit(d), unit(CUBE_C[lay][2] - pos_te)))
    cc = float(np.dot(unit(d), unit(CUBE_C[lay][3] - pos_te)))
    return (cb - cc) if truth == 2 else (cc - cb), float(cb > cc)


def cell_boot(vals, ids, cells, seed=0):
    """Sec.22: per-cell mean (bootstrap over units within the cell), counts, and the equal-weight average of the
    cells (each bootstrap draw resamples units within every cell). None values are dropped."""
    v = np.array([np.nan if x is None else x for x in vals], float)
    ids, cells = np.asarray(ids), np.asarray(cells)
    ok = ~np.isnan(v)
    names = sorted(set(cells[ok]))
    out = {'cells': {}, 'counts': {}}
    rng = np.random.RandomState(seed)
    draws = []
    for c in names:
        m = ok & (cells == c)
        out['cells'][str(c)] = boot_mean(v[m], ids[m])
        out['counts'][str(c)] = int(m.sum())
    for _ in range(BOOT):
        means = []
        for c in names:
            vv = v[ok & (cells == c)]
            means.append(vv[rng.randint(len(vv), size=len(vv))].mean())
        draws.append(np.mean(means))
    out['equal_weight'] = {'mean': float(np.mean([out['cells'][str(c)]['mean'] for c in names])),
                           'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    return out


def eaten_corner(lay, colour):
    return S0._COL_CORNER[int(lay)][int(colour)]


def split_boot(vals, ids, groups):
    """boot_mean overall and per group label (None values dropped)."""
    v = np.array([x for x in vals], dtype=object)
    ok = np.array([x is not None for x in vals])
    out = {'all': boot_mean(v[ok].astype(float), np.asarray(ids)[ok]) if ok.any() else None}
    for gname in sorted(set(groups)):
        m = ok & (np.asarray(groups) == gname)
        out[str(gname)] = boot_mean(v[m].astype(float), np.asarray(ids)[m]) if m.any() else None
    return out


# ------------------------------------------------------------------ phase 1
def phase1_tests(m1, dev):
    d = read('v15_s1_p80_half_3000', 'test', ['self_vision', 'self_motion', 'self_position', 'other_position',
                                               'a1_eat', 'a1_target', 'available', 'layout'])
    N = d['self_vision'].shape[0]
    fo = folds(N)
    hid = schedule(m1, N)
    usual = run(m1, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 0.99),
                hidden=lambda t: hid[:, t])
    vis = ~hid
    r = {'P1a_r2_self_visible': r2_cv(usual['h1'], d['self_position'], vis, fo),
         'P1a_r2_self_all': r2_cv(usual['h1'], d['self_position'], np.ones_like(vis), fo),
         'P1b_r2_other': r2_cv(usual['h2'], d['other_position'], np.ones_like(vis), fo)}
    ev = first_informative(d['a1_eat'], d['a1_target'])
    te = np.array([ev.get(n, (10 ** 6, -1))[0] for n in range(N)])
    p_m = lambda t: np.where(t >= te, 1.0, 0. if t == 0 else 0.99)  # noqa: E731
    h_m = lambda t: np.where(t >= te, True, hid[:, t])  # noqa: E731
    roll = run(m1, d['self_vision'], d['self_motion'], dev, pmask=p_m, hidden=h_m)
    p_hat = readout(usual['h1'], d['self_position'], np.ones_like(vis), roll['h1'], fo)
    cs, acc, acc_a, ids, diffs, corners = [], [], [], [], [], []
    for n, (t_e, col) in ev.items():
        corners.append(eaten_corner(d['layout'][n], col))
        dd = p_hat[n, t_e + K - 1] - p_hat[n, t_e - 1]
        args = (d['self_position'][n, t_e], d['layout'][n], d['available'][n, t_e], d['a1_target'][n, t_e])
        c, a = next_metrics(dd, *args)
        pre = d['self_position'][n, t_e] - d['self_position'][n, t_e - 5]
        _, aa = next_metrics(pre, *args)
        cs.append(c); acc.append(a); acc_a.append(aa); ids.append(n); diffs.append(bc_metrics(dd, *args)[0])
    r['P1c_diff_next_minus_other'] = split_boot(diffs, ids, ['X'] * len(ids))['all']
    r['P1c_diff_by_corner'] = cell_boot(diffs, ids, corners)  # Sec.22: TL / BL and their equal-weight average
    r['P1c_c_next'] = boot_mean(cs, ids)
    r['P1c_acc'] = boot_mean(acc, ids)
    r['P1c_acc_a'] = boot_mean(acc_a, ids)
    return r, {'P1c_c': np.array(cs), 'P1c_ids': np.array(ids), 'P1c_acc_ev': np.array(acc),
               'P1c_diff_ev': np.array([np.nan if x is None else x for x in diffs]), 'P1c_corner_ev': np.array(corners)}


# ------------------------------------------------------------------ phase 2 (on the stage-2 test)
def phase2_tests(m1, dev):
    keys = ['self_vision', 'other_vision', 'self_motion', 'self_position', 'other_position', 'a2_eat', 'a2_target',
            'available', 'layout', 'a2_pleasure', 'a2_cand', 'gray0']
    d = read('v15_s2_c2_p80_half_6000', 'test', keys)
    N = d['self_vision'].shape[0]
    fo = folds(N)
    m1.motion_masking = False
    ev = first_informative(d['a2_eat'], d['a2_target'])
    te = np.array([ev.get(n, (10 ** 6, -1))[0] for n in range(N)])
    ones = np.ones((N, 1)); zeros2 = np.zeros((N, 2))
    pa2 = d['a2_pleasure']
    # P2-a
    usual = run(m1, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.zeros(N),
                p2=lambda t: np.concatenate([zeros2, pa2[:, t], ones], 1))
    held = lambda t: np.where((t >= te)[:, None], pa2[np.arange(N), np.minimum(te, T - 1)], pa2[:, t])  # noqa: E731
    roll = run(m1, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.where(t >= te, 1.0, 0.),
               p2=lambda t: np.concatenate([zeros2, held(t), ones], 1))
    p_hat = readout(usual['h2'], d['other_position'], np.ones((N, 100), bool), roll['h2'], fo)
    r = {'P2a_r2_other_usual': r2_cv(usual['h2'], d['other_position'], np.ones((N, 100), bool), fo)}
    cs, acc, ids, diffs, blue, grp, cellv = [], [], [], [], [], [], []
    for n, (t_e, col) in ev.items():
        cellv.append(eaten_corner(d['layout'][n], col) + '_' + ('X' if d['a2_cand'][n] == 0 else 'Y'))
        dd = p_hat[n, t_e + K - 1] - p_hat[n, t_e - 1]
        args = (d['other_position'][n, t_e], d['layout'][n], d['available'][n, t_e], d['a2_target'][n, t_e])
        c, a = next_metrics(dd, *args)
        df, bl = bc_metrics(dd, *args)
        cs.append(c); acc.append(a); ids.append(n); diffs.append(df); blue.append(bl)
        grp.append('X' if d['a2_cand'][n] == 0 else 'Y')
    r['P2a_c_next'] = boot_mean(cs, ids)
    r['P2a_acc'] = boot_mean(acc, ids)
    r['P2a_c_next_by_cand'] = split_boot(cs, ids, grp)
    r['P2a_acc_by_cand'] = split_boot(acc, ids, grp)
    r['P2a_diff_next_minus_other'] = split_boot(diffs, ids, grp)
    r['P2a_toward_blue'] = split_boot(blue, ids, grp)
    r['P2a_diff_by_corner_cand'] = cell_boot(diffs, ids, cellv)  # Sec.22: 4 cells and their equal-weight average
    r['P2a_c_next_by_corner_cand'] = cell_boot(cs, ids, cellv)
    # P2-b
    pc = cand_pleasure(d['other_vision'])
    errs = {}
    for c in ('X', 'Y'):
        o = run(m1, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.where(t > te, 1.0, 0.),
                p2=lambda t, c=c: np.concatenate([zeros2, pc[c][:, t], ones], 1), need_err=True)
        errs[c] = o['err']
    truth = {n: ('X' if d['a2_cand'][n] == 0 else 'Y') for n in ev}
    post, pre, ids, grp, chx_post, chx_pre = [], [], [], [], [], []
    for n, (t_e, col) in ev.items():
        e_post = {c: errs[c][n, t_e:t_e + K].sum() for c in 'XY'}
        e_pre = {c: errs[c][n, :t_e].sum() for c in 'XY'}
        post.append(float(min(e_post, key=e_post.get) == truth[n]))
        pre.append(float(min(e_pre, key=e_pre.get) == truth[n]))
        chx_post.append(float(min(e_post, key=e_post.get) == 'X')); chx_pre.append(float(min(e_pre, key=e_pre.get) == 'X'))
        ids.append(n); grp.append(truth[n])
    r['P2b_acc_post'] = boot_mean(post, ids)
    r['P2b_acc_pre'] = boot_mean(pre, ids)
    r['P2b_acc_post_by_cand'] = split_boot(post, ids, grp)
    r['P2b_acc_pre_by_cand'] = split_boot(pre, ids, grp)
    r['P2b_chose_X_post'] = split_boot(chx_post, ids, grp)
    r['P2b_chose_X_pre'] = split_boot(chx_pre, ids, grp)
    return r, {'P2a_c': np.array(cs), 'P2a_ids': np.array(list(ev.keys())), 'P2a_acc_ev': np.array(acc),
               'P2a_diff_ev': np.array([np.nan if x is None else x for x in diffs]), 'P2a_cell_ev': np.array(cellv),
               'P2a_blue_ev': np.array([np.nan if x is None else x for x in blue]),
               'P2b_post_ev': np.array(post), 'P2b_pre_ev': np.array(pre)}


def baselines_ab():
    """(a) and (b0) next-target accuracy at the same A-2 moments (stage-0 definitions, k = 5; (b0) = HistGradientBoosting
    on position + past directions, fitted on train episodes 0..699 of v15_s2_c2_p80_half_6000)."""
    tr = S0.load('v15_s2_c2_p80_half_6000', 'a2')
    clf = S0.fit_b(tr, set(range(700)), 5, 0)[0]
    d = read('v15_s2_c2_p80_half_6000', 'test', ['other_position', 'a2_eat', 'a2_target', 'available', 'layout'])
    pos = d['other_position'].astype(np.float64)
    ev = first_informative(d['a2_eat'], d['a2_target'])
    acc_a, acc_b, ids = [], [], []
    for n, (t_e, col) in ev.items():
        truth = d['a2_target'][n, t_e]
        _, aa = next_metrics(pos[n, t_e] - pos[n, t_e - 5], pos[n, t_e], d['layout'][n], d['available'][n, t_e], truth)
        f = list(pos[n, t_e]) + [v for k in S0.KS for v in S0.unit(pos[n, t_e] - pos[n, max(t_e - k, 0)])]
        acc_a.append(aa); acc_b.append(float(clf.predict(np.array([f]))[0] == truth)); ids.append(n)
    return {'a': boot_mean(acc_a, ids), 'b0': boot_mean(acc_b, ids)}


# ------------------------------------------------------------------ pairs (P2-c, P2-c')
def pair_data(nx, ny):
    keys = ['self_vision', 'other_vision', 'self_motion', 'self_position', 'other_position', 'a2_eat', 'a2_target',
            'available', 'layout', 'a2_cand']
    dx, dy = read(nx, 'train', keys), read(ny, 'train', keys)
    keep = []
    for n in range(dx['layout'].shape[0]):
        ex, ey = first_informative(dx['a2_eat'][n:n + 1], dx['a2_target'][n:n + 1]), first_informative(dy['a2_eat'][n:n + 1], dy['a2_target'][n:n + 1])
        if 0 in ex and 0 in ey and ex[0][0] == ey[0][0]:
            t_e = ex[0][0]
            if np.array_equal(dx['other_position'][n, :t_e + 1], dy['other_position'][n, :t_e + 1]) and \
                    np.array_equal(dx['self_position'][n, :t_e + 1], dy['self_position'][n, :t_e + 1]):
                keep.append((n, t_e))
    return dx, dy, keep


def stack_pairs(dx, dy, keep):
    idx = np.array([n for n, _ in keep])
    both = {k: np.concatenate([dx[k][idx], dy[k][idx]]) for k in dx}
    te = np.array([t for _, t in keep] * 2)
    pair = np.concatenate([np.arange(len(keep))] * 2)
    return both, te, pair


def pair_eval_g(m2, dev, both, te, pair):
    """P2-c on stacked pairs: blue (2) vs cyan (3) from the h2 displacement; truth = the episode's next target."""
    N = len(te)
    fo = [(pair % 2 == 0), (pair % 2 == 1)]
    usual = run(m2, both['self_vision'], both['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 1.))
    roll = run(m2, both['self_vision'], both['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 1.),
               frame=lambda t: np.minimum(t, te))
    p_hat = readout(usual['h2'], both['other_position'], np.ones((N, 100), bool), roll['h2'], fo)
    acc, ds, blue = [], [], []
    for i in range(N):
        t_e = te[i]
        dd = p_hat[i, t_e + K - 1] - p_hat[i, t_e - 1]
        lay = both['layout'][i]
        p0 = both['other_position'][i, t_e]
        cb = np.dot(unit(dd), unit(CUBE_C[lay][2] - p0)); cc = np.dot(unit(dd), unit(CUBE_C[lay][3] - p0))
        truth = both['a2_target'][i, t_e]
        acc.append(float((cb > cc) == (truth == 2)) if truth in (2, 3) else None)
        blue.append(float(cb > cc))
        ds.append(dd)
    member = np.array(['first'] * (N // 2) + ['second'] * (N // 2))  # gen: X then Y; ctrl3: X then Xp
    return {'acc': split_boot(acc, pair, ['all'] * N)['all'], 'acc_by_member': split_boot(acc, pair, member),
            'toward_blue': split_boot(blue, pair, member),
            'r2_other_usual': r2_cv(usual['h2'], both['other_position'], np.ones((N, 100), bool), fo)}, \
        {'d': np.array(ds), 'roll': roll, 'acc_ev': np.array([np.nan if x is None else x for x in acc]), 'blue_ev': np.array(blue)}


def pair_eval_choose(m1, dev, both, te, pair):
    """P2-c': v14-A protocol (vision only at t=0), candidate pleasure in process-2, choose by pre-eat error."""
    N = len(te)
    m1.motion_masking = False
    pc = cand_pleasure(both['other_vision'])
    ones = np.ones((N, 1)); zeros2 = np.zeros((N, 2))
    errs = {c: run(m1, both['self_vision'], both['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 1.),
                   p2=lambda t, c=c: np.concatenate([zeros2, pc[c][:, t], ones], 1), need_err=True)['err'] for c in 'XY'}
    acc, chx, grp = [], [], []
    for i in range(N):
        e = {c: errs[c][i, :te[i]].sum() for c in 'XY'}
        truth = 'X' if both['a2_cand'][i] == 0 else 'Y'
        acc.append(float(min(e, key=e.get) == truth)); chx.append(float(min(e, key=e.get) == 'X')); grp.append(truth)
    return {'acc': boot_mean(acc, pair), 'acc_by_cand': split_boot(acc, pair, grp), 'chose_X': split_boot(chx, pair, grp),
            '_acc_ev': np.array(acc)}


def natural_g(m2, dev):
    """descriptive: P2-c protocol on all first informative A-2 moments of the stage-2 test (nearest-bearing accuracy)."""
    keys = ['self_vision', 'self_motion', 'other_position', 'a2_eat', 'a2_target', 'available', 'layout']
    d = read('v15_s2_c2_p80_half_6000', 'test', keys)
    N = d['self_vision'].shape[0]
    ev = first_informative(d['a2_eat'], d['a2_target'])
    te = np.array([ev.get(n, (T - 1, -1))[0] for n in range(N)])
    usual = run(m2, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 1.))
    roll = run(m2, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 1.),
               frame=lambda t: np.minimum(t, te))
    p_hat = readout(usual['h2'], d['other_position'], np.ones((N, 100), bool), roll['h2'], folds(N))
    acc, ids = [], []
    for n, (t_e, col) in ev.items():
        dd = p_hat[n, t_e + K - 1] - p_hat[n, t_e - 1]
        _, a = next_metrics(dd, d['other_position'][n, t_e], d['layout'][n], d['available'][n, t_e], d['a2_target'][n, t_e])
        acc.append(a); ids.append(n)
    return {'acc': boot_mean(acc, ids)}, usual['q2']


def q2_stats(q2, dev):
    d = read('v15_s2_c2_p80_half_6000', 'test', ['a2_pleasure', 'a2_cand'])
    q = q2.reshape(-1, 8)
    p = d['a2_pleasure'][:, :100].reshape(-1, 8)
    var = q.var(0)
    corr = [float(np.corrcoef(q[:, k], p[:, k])[0, 1]) if q[:, k].std() > 1e-9 and p[:, k].std() > 1e-9 else None for k in range(8)]
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    X = q2[:, 20:100:5].reshape(-1, 8)
    y = np.repeat(d['a2_cand'], 16)
    g = np.repeat(np.arange(len(d['a2_cand'])), 16)
    acc = cross_val_score(LogisticRegression(max_iter=2000), X, y, groups=g, cv=GroupKFold(5)).mean()
    return {'var_frac_per_bin': (var / var.sum()).tolist(), 'corr_q2_pA2_per_bin': corr, 'cand_from_q2_acc': float(acc)}


def control_check(m2, dev, both, te, pair):
    """Sec.22: the control (3) pairs as a check of the evaluation pipeline (not a learning result).
    (i) the inputs of the two episodes up to t_e must be identical (max |diff| of the views fed and of A-1's motion);
        under the P2-c protocol (inputs frozen at t_e afterwards) the predictions must then be (almost) identical
        (max |diff| of h2 over t <= t_e + K and of the read displacement);
    (ii) descriptive, usual protocol (true views throughout): first step where the fed views differ (t_in), where A-2's
        positions differ (t_traj) and where h2 differs (> 1e-4, t_pred); a prediction that diverges before the inputs
        (t_pred < t_in) is reported as a leak."""
    N = len(te); H = N // 2
    a, b = np.arange(H), np.arange(H) + H
    in_diff = max(int(np.abs(both['self_vision'][a[i], :te[a[i]] + 1].astype(int) - both['self_vision'][b[i], :te[b[i]] + 1]).max())
                  for i in range(H))
    mot_diff = max(float(np.abs(both['self_motion'][a[i], :te[a[i]] + 1] - both['self_motion'][b[i], :te[b[i]] + 1]).max())
                   for i in range(H))
    pm = lambda t: np.full(N, 0. if t == 0 else 1.)  # noqa: E731
    roll = run(m2, both['self_vision'], both['self_motion'], dev, pmask=pm, frame=lambda t: np.minimum(t, te))
    h2d = max(float(np.abs(roll['h2'][a[i], :te[a[i]] + K] - roll['h2'][b[i], :te[b[i]] + K]).max()) for i in range(H))
    usual = run(m2, both['self_vision'], both['self_motion'], dev, pmask=pm)
    p_hat = readout(usual['h2'], both['other_position'], np.ones((N, 100), bool), roll['h2'], [(pair % 2 == 0), (pair % 2 == 1)])
    dd = max(float(np.abs((p_hat[a[i], te[a[i]] + K - 1] - p_hat[a[i], te[a[i]] - 1]) -
                          (p_hat[b[i], te[b[i]] + K - 1] - p_hat[b[i], te[b[i]] - 1])).max()) for i in range(H))
    t_in, t_traj, t_pred, leak = [], [], [], 0
    for i in range(H):
        dv = np.where(np.any(both['self_vision'][a[i]] != both['self_vision'][b[i]], axis=(1, 2, 3)))[0]
        dp = np.where(np.any(both['other_position'][a[i]] != both['other_position'][b[i]], axis=1))[0]
        dh = np.where(np.abs(usual['h2'][a[i]] - usual['h2'][b[i]]).max(1) > 1e-4)[0]
        ti = int(dv[0]) if len(dv) else None
        tp = int(dh[0]) if len(dh) else None
        t_in.append(ti); t_traj.append(int(dp[0]) if len(dp) else None); t_pred.append(tp)
        if tp is not None and (ti is None or tp < ti):
            leak += 1
    def med(x):
        x = [v for v in x if v is not None]
        return float(np.median(x)) if x else None
    return {'max_input_diff_upto_te': in_diff, 'max_motion_diff_upto_te': mot_diff,
            'max_h2_diff_frozen_protocol': h2d, 'max_displacement_diff_frozen_protocol': dd,
            'usual_first_step_inputs_differ_median_minus_te': med([None if x is None else x - te[a[i]] for i, x in enumerate(t_in)]),
            'usual_first_step_A2_positions_differ_median_minus_te': med([None if x is None else x - te[a[i]] for i, x in enumerate(t_traj)]),
            'usual_first_step_h2_differs_median_minus_te': med([None if x is None else x - te[a[i]] for i, x in enumerate(t_pred)]),
            'leaks_prediction_before_input': leak, 'n_pairs': H}


OFFSETS = (0, 1, 3, 5, 10)


def memory_readout(m2, dev, roll_pairs, both, te, pair):
    """Sec.19 item 7 (descriptive): A-2's candidate (X / Y) read linearly from h2 and from G's LSTM state at t_e + o
    (o in OFFSETS; state after input frame t). Readout = logistic regression fitted on the training data
    (v15_s2_c2_p80_half_1000t train, usual phase-2 protocol, first informative A-2 moment of each episode, same
    offsets pooled), applied to the generated pairs under the P2-c protocol (input frozen at frame t_e)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = read('v15_s2_c2_p80_half_1000t', 'train', ['self_vision', 'self_motion', 'a2_eat', 'a2_target', 'a2_cand'])
    N = d['self_vision'].shape[0]
    usual = run(m2, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 1.))
    ev = first_informative(d['a2_eat'], d['a2_target'])
    out = {}
    for key in ('h2', 'g'):
        X = np.concatenate([usual[key][n, min(t + o, T - 2)][None] for n, (t, _) in ev.items() for o in OFFSETS])
        y = np.array([d['a2_cand'][n] for n, _ in ev.items() for o in OFFSETS])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X, y)
        truth = (both['a2_cand'] == 0).astype(int)  # 1 = X
        res = {}
        for o in OFFSETS:
            Z = roll_pairs[key][np.arange(len(te)), np.minimum(te + o, T - 2)]
            pred = clf.predict(Z)
            res[str(o)] = boot_mean((pred == both['a2_cand']).astype(float), pair)
        out[key] = res
    return out


def load_ref0(tag, dev):
    """'before training' references (Sec.19 item 1): phase-1 model at epoch 0 (encoders loaded and frozen, the rest
    at its initial values), and the phase-2 class with those weights and G at its initial values (seed 0)."""
    m1 = load(f'v15_s2p1_{tag}_mse', 0, dev)
    torch.manual_seed(0)
    mcfg = util.load_config(os.path.join(_PI_DIR, 'config', 'model', 'SuperpositionNetworkMotionPleasureHoldOtherV15', 'default.yml'))
    m2 = model_v15.SuperpositionNetworkMotionPleasureHoldOtherV15(mcfg)
    m2.load_state_dict(torch.load(os.path.join(RES, f'v15_s2p1_{tag}_mse', '0', 'model', '00000.pth'), map_location='cpu')['model'], strict=False)
    return m1, m2.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--epoch', type=int, default=150)
    ap.add_argument('--ref0', action='store_true', help='the before-training reference of this tag (Sec.19)')
    a = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(a.device)
    get1 = (lambda: load_ref0(a.tag, dev)[0]) if a.ref0 else (lambda: load(f'v15_s2p1_{a.tag}_mse', a.epoch, dev))
    m2 = load_ref0(a.tag, dev)[1] if a.ref0 else load(f'v15_s2p2_{a.tag}', a.epoch, dev)
    name = f'{a.tag}_ref0' if a.ref0 else a.tag
    R, NPZ = {'tag': name, 'epoch': 0 if a.ref0 else a.epoch, 'K': K}, {}
    r, z = phase1_tests(get1(), dev); R.update(r); NPZ.update(z); print('phase 1', r, flush=True)
    r, z = phase2_tests(get1(), dev); R.update(r); NPZ.update(z); print('phase 2', r, flush=True)
    R['baselines_ab_at_A2_moments'] = baselines_ab()
    for key, (nx, ny) in (('gen', ('v15_s2_genx_p80', 'v15_s2_geny_p80')), ('ctrl3', ('v15_s2_c3x_p80', 'v15_s2_c3xp_p80'))):
        dx, dy, keep = pair_data(nx, ny)
        both, te, pair = stack_pairs(dx, dy, keep)
        R[f'{key}_n_pairs'] = len(keep)
        if key == 'ctrl3':
            R['ctrl3_pipeline_check'] = control_check(m2, dev, both, te, pair)
            print(key, R['ctrl3_pipeline_check'], flush=True)
            continue
        r, z = pair_eval_g(m2, dev, both, te, pair); R[f'P2c_{key}'] = r
        NPZ[f'{key}_d'] = z['d']; NPZ[f'{key}_te'] = te; NPZ[f'{key}_keys'] = np.array([n for n, _ in keep])
        NPZ[f'{key}_acc_ev'] = z['acc_ev']; NPZ[f'{key}_blue_ev'] = z['blue_ev']; NPZ[f'{key}_pair'] = pair
        if key == 'gen':
            R["P2c'_gen"] = pair_eval_choose(get1(), dev, both, te, pair)
            NPZ['gen_choose_acc_ev'] = R["P2c'_gen"].pop('_acc_ev')
            R['memory_readout_gen'] = memory_readout(m2, dev, z['roll'], both, te, pair)
        print(key, R[f'P2c_{key}'], flush=True)
    r, q2 = natural_g(m2, dev); R['P2c_natural_all_moments'] = r
    R['q2_stats'] = q2_stats(q2, dev)
    json.dump(R, open(os.path.join(OUT, f'eval_{name}.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(OUT, f'eval_{name}.npz'), **NPZ)
    print(json.dumps(R), flush=True)


if __name__ == '__main__':
    main()
