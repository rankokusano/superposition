"""
v11 stage-2 evaluation, primary + priority secondary (docs/v11_experiment_log.md Sec.7.42).
For one stage-2 run (setting green/pref, condition, seed) and one G checkpoint:
  * run the model like the exp3 runner (vision masked for t > 0 in both processes,
    model.eval(), util.seed_all(0)) on v11_s2t_green, v11_s2t_green_swap, v11_s2t_red
    (test, 300 ep) and on the stage-2 test split of v11_s2_pref; keep Q2 (N, 100, 8);
  * T2: s_T2 at t = 50 (and mean over t = 50..52), per episode (swap vs no-swap pair);
  * T5: d_T5 on the data-only matched pairs (t >= 20, same layout, A-2 within r2, A-1
    within 3.0, greedy nearest one-to-one; r2 = 1.0, or 1.5 if < 500 pairs per layout);
  * corr(Q2, P_A2) pooled over frames x bins, and per layout;
  * Q2 -> A-1's own step direction (cos, sin), and -> A-2's step direction, Ridge(1)
    episode GroupKFold(5) R^2, raw and after removing a quadratic fit on (A-1, A-2) positions;
  * the T2 "position read-out": per layout a 10-NN regressor Q2 ~ (A-2 pos, A-1 pos) fit on
    v11_s2t_green t >= 20 frames, applied at t = 50 of the swap episodes with the NEW layout.
Per-episode / per-pair values are saved (.npz) so paired differences can be bootstrapped.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import DATA, RES, load_model  # noqa
import util  # noqa
from v11core.pleasure import pleasure_np  # noqa

GB = {0: [0, 1], 1: [6, 7]}   # green bins by layout code
RB = {0: [6, 7], 1: [0, 1]}   # red bins
SETS = {'green': ('v11_s2t_green', 'test'), 'swap': ('v11_s2t_green_swap', 'test'),
        'red': ('v11_s2t_red', 'test'), 'pref': ('v11_s2_pref', 'test')}
OUT = os.path.join(RES, 'v11', 'stage2', 'eval')


@torch.no_grad()
def run_q2(model, path, split, dev, p2=None, batch=50):
    g = h5py.File(path, 'r')[split]
    N = g['self_vision'].shape[0]
    Q, LS, LO = [], [], []
    util.seed_all(0)
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        B = x.shape[0]
        model.init_state(B)
        q, ls = [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            ov = None if p2 is None else torch.from_numpy(p2[b0:b0 + batch, t]).float().to(dev)
            out = model({'self_vision': x[:, t]}, pm, pm, p2_override=ov)
            if 'q2_hat' in out:
                q.append(out['q2_hat'].cpu().numpy())
            ls.append(torch.abs(out['self_vision'] - x[:, t + 1]).mean((1, 2, 3)).cpu().numpy())
        if q:
            Q.append(np.stack(q, 1))
        LS.append(np.stack(ls, 1))
    return (np.concatenate(Q) if Q else None), np.concatenate(LS)


def load_set(name):
    path, split = os.path.join(DATA, SETS[name][0], 'data.h5'), SETS[name][1]
    g = h5py.File(path, 'r')[split]
    d = {k: g[k][()] for k in ('self_position', 'other_position', 'self_motion', 'other_motion', 'layout', 'layout_t',
                              'a1_source', 'a2_source')}
    # a2_liked (0 Green, 1 Red) is stored only in v11_s2_pref; the T2/T5 sets have a fixed preference
    d['a2_liked'] = g['a2_liked'][()] if 'a2_liked' in g else np.full(len(d['layout']), int(name == 'red'))
    d['path'], d['split'] = path, split
    return d


def true_pleasure(name, d):
    g = h5py.File(d['path'], 'r')[d['split']]
    ov = g['other_vision'][:, :100]
    liked = d['a2_liked']
    P = np.zeros(ov.shape[:2] + (8,), np.float32)
    for c, (lk, dl) in ((0, ('Green', 'Red')), (1, ('Red', 'Green'))):
        m = liked == c
        if m.any():
            P[m] = pleasure_np(ov[m], lk, dl)
    return P


def t5_pairs(dg, dr, r2=1.0, r1=3.0, tmin=20):
    from scipy.spatial import cKDTree
    pairs = []
    for L in (0, 1):
        eg, er = np.where(dg['layout'] == L)[0], np.where(dr['layout'] == L)[0]
        def frames(d, eps):  # noqa
            e, t = np.meshgrid(eps, np.arange(tmin, 100), indexing='ij')  # t <= 99 (Q2 has 100 steps)
            e, t = e.ravel(), t.ravel()
            return e, t, d['other_position'][e, t], d['self_position'][e, t]
        ge, gt, g2, g1 = frames(dg, eg)
        re_, rt, r2p, r1p = frames(dr, er)
        tree = cKDTree(r2p)
        cand = []
        for i, nb in enumerate(tree.query_ball_point(g2, r2)):
            for j in nb:
                d1 = np.linalg.norm(g1[i] - r1p[j])
                if d1 <= r1:
                    cand.append((np.linalg.norm(g2[i] - r2p[j]) + d1 / r1, i, j))
        cand.sort()
        ug, ur = set(), set()
        for _, i, j in cand:
            if i in ug or j in ur:
                continue
            ug.add(i); ur.add(j)
            pairs.append((L, ge[i], gt[i], re_[j], rt[j]))
    return np.array(pairs, dtype=int)


def r2_cv(X, Y, g):
    Yh = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y, g):
        Yh[te] = Ridge(alpha=1.0).fit(X[tr], Y[tr]).predict(X[te])
    return float(1 - ((Y - Yh) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def motion_readout(Q, d):
    N = Q.shape[0]
    ts = np.arange(1, 100)
    X = Q[:, ts].reshape(-1, 8)
    g = np.repeat(np.arange(N)[:, None], len(ts), 1).reshape(-1)
    out = {}
    pos = np.concatenate([d['self_position'][:, ts].reshape(-1, 2), d['other_position'][:, ts].reshape(-1, 2)], 1)
    P2 = PolynomialFeatures(2).fit_transform(pos)
    resid = lambda Z: Z - Ridge(alpha=1e-3).fit(P2, Z).predict(P2)  # noqa
    for who, mk, sk in (('A1', 'self_motion', 'a1_source'), ('A2', 'other_motion', 'a2_source')):
        m = d[mk][:, ts].reshape(-1, 2)
        awake = (d[sk][:, ts].reshape(-1) != 3) & (np.linalg.norm(m, axis=1) > 1e-6)
        ang = np.arctan2(m[:, 1], m[:, 0])
        Y = np.stack([np.cos(ang), np.sin(ang)], 1)
        out[f'Q2_to_{who}_step_dir'] = r2_cv(X[awake], Y[awake], g[awake])
        out[f'Q2_to_{who}_step_dir_pos_controlled'] = r2_cv(resid(X)[awake], resid(Y)[awake], g[awake])
    return out


def corr_q2(Q, P, lay):
    out = {'all': float(np.corrcoef(Q.reshape(-1), P.reshape(-1))[0, 1])}
    for L, nm in ((0, 'original'), (1, 'swap_rg')):
        m = lay == L
        out[nm] = float(np.corrcoef(Q[m].reshape(-1), P[m].reshape(-1))[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='setting:cond:seed (e.g. green:mixed:0) or d6:cond:seed')
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in SETS}
    PA2 = {k: true_pleasure(k, D[k]) for k in SETS}
    pp = os.path.join(OUT, 't5_pairs.npz')
    if not os.path.exists(pp):
        pr = t5_pairs(D['green'], D['red'], 1.0)
        n_by = {L: int((pr[:, 0] == L).sum()) for L in (0, 1)}
        r_used = 1.0
        if min(n_by.values()) < 500:
            pr = t5_pairs(D['green'], D['red'], 1.5)
            r_used = 1.5
        np.savez(pp, pairs=pr, radius=r_used)
    z = np.load(pp)
    pairs, radius = z['pairs'], float(z['radius'])
    for spec in args.runs:
        st, c, s = spec.split(':')
        s = int(s)
        # d6: P_A2 in place of Q2. The T2/T5/correlation values then depend on the data only
        # (no model); the model-dependent D6 parts (T3, h2, M1(d)-other) are computed later.
        eps = args.epochs if st != 'd6' else [0]
        for e in eps:
            tag = f'{st}_{c}_s{s}_ep{e}' if st != 'd6' else 'd6_PA2'
            Q, LS = {}, {}
            if st == 'd6':
                Q = PA2
            else:
                model, _ = load_model(f'v11_s2_{st}_{c}_s{s}', s, e, dev)
                for k in SETS:
                    Q[k], LS[k] = run_q2(model, D[k]['path'], D[k]['split'], dev)
            r = {'tag': tag, 't5_radius': radius, 't5_pairs_per_layout': {str(L): int((pairs[:, 0] == L).sum()) for L in (0, 1)}}
            # T2
            L0 = D['swap']['layout']
            newG = np.array([RB[int(l)] for l in L0]); oldG = np.array([GB[int(l)] for l in L0])
            def gsel(q, idx):  # noqa  q (N, 8), idx (N, 2)
                return np.take_along_axis(q, idx, 1).mean(1)
            sT2 = {}
            for name, ts in (('t50', [50]), ('t50_52', [50, 51, 52])):
                v = np.mean([(gsel(Q['swap'][:, t], newG) - gsel(Q['swap'][:, t], oldG))
                             - (gsel(Q['green'][:, t], newG) - gsel(Q['green'][:, t], oldG)) for t in ts], 0)
                sT2[name] = v
            # T2 position read-out (10-NN per layout on the no-swap set, t >= 20)
            dg = D['green']
            pred_new = np.zeros((len(L0), 8))
            for L in (0, 1):
                m = dg['layout'] == L
                Xf = np.concatenate([dg['other_position'][m][:, 20:100].reshape(-1, 2), dg['self_position'][m][:, 20:100].reshape(-1, 2)], 1)
                knn = KNeighborsRegressor(10).fit(Xf, Q['green'][m][:, 20:100].reshape(-1, 8))
                tgt = np.where(1 - L0 == L)[0]   # swap episodes whose NEW layout is L
                if len(tgt):
                    xq = np.concatenate([D['swap']['other_position'][tgt, 50], D['swap']['self_position'][tgt, 50]], 1)
                    pred_new[tgt] = knn.predict(xq)
            sT2_pos = (gsel(pred_new, newG) - gsel(pred_new, oldG)) - (gsel(Q['green'][:, 50], newG) - gsel(Q['green'][:, 50], oldG))
            # T5
            Lp = pairs[:, 0]
            def gr(q, L):  # noqa
                return gsel(q, np.array([GB[int(l)] for l in L])) - gsel(q, np.array([RB[int(l)] for l in L]))
            qg = Q['green'][pairs[:, 1], pairs[:, 2]]; qr = Q['red'][pairs[:, 3], pairs[:, 4]]
            dT5 = gr(qg, Lp) - gr(qr, Lp)
            r['T2_mean'] = {k: float(v.mean()) for k, v in sT2.items()}
            r['T2_pos_readout_mean'] = float(sT2_pos.mean())
            r['T5_mean'] = float(dT5.mean())
            r['T5_by_t_bins'] = {f'{a}-{b}': float(dT5[(pairs[:, 2] >= a) & (pairs[:, 2] < b)].mean()) for a, b in ((20, 40), (40, 60), (60, 80), (80, 100))}
            # secondary (priority)
            r['corr_Q2_PA2'] = {'green_set': corr_q2(Q['green'], PA2['green'], D['green']['layout']),
                                'pref_test': corr_q2(Q['pref'], PA2['pref'], D['pref']['layout'])}
            for liked in (0, 1):
                m = D['pref']['a2_liked'] == liked
                r['corr_Q2_PA2'][f'pref_test_liked{liked}'] = corr_q2(Q['pref'][m], PA2['pref'][m], D['pref']['layout'][m])
            if st != 'd6':
                r['motion_readout'] = {'green_set': motion_readout(Q['green'], D['green']),
                                       'pref_test': motion_readout(Q['pref'], D['pref'])}
            r['self_vision_L1_per_pixel'] = {k: float(v.mean()) for k, v in LS.items()}
            with open(os.path.join(OUT, f'{tag}.json'), 'w') as f:
                json.dump(r, f, indent=1)
            np.savez_compressed(os.path.join(OUT, f'{tag}_per.npz'), sT2_t50=sT2['t50'], sT2_t50_52=sT2['t50_52'],
                                sT2_pos=sT2_pos, dT5=dT5, t5_green_ep=pairs[:, 1])
            print(tag, json.dumps({k: r[k] for k in ('T2_mean', 'T2_pos_readout_mean', 'T5_mean')}), flush=True)


if __name__ == '__main__':
    main()
