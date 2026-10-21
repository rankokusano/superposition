"""
v12 stage 2c primary evaluation on the HIDDEN test (docs/v12_experiment_log.md Sec.46). Same definitions as
analyze/v11/eval_s2_v11.py (T2, T5, corr(Q2, P_A2), Q2 -> step direction, T2 position read-out; v11 code unchanged,
its functions are reused), but
  * the evaluation sets are the A-2-hidden copies (v12_hide_v11_s2t_green / _swap / _red, v12_hide_v11_s2_pref test):
    the network INPUT is self_vision_in (A-2 removed on hidden steps), the target self_vision (prediction error);
  * main (written to data/result/v12/<out>/eval):  T2 on the episodes where A-2 is HIDDEN at t = 50 (the swap
    moment), T5 on matched pairs whose two frames are both HIDDEN (pairs built as v11, same radius rule);
  * reference (data/result/v12/<out>_hidvis/eval): the same on the episodes VISIBLE at t = 50 / both-visible pairs.
  * corr(Q2, P_A2) is also given on hidden and on visible frames.
The per-episode / per-pair files keep the v11 keys (sT2_t50, sT2_t50_52, sT2_pos, dT5, t5_green_ep), so
analyze/v12/aggregate_s2_v12.py applies the registered pair / pair-count rules unchanged (--out <out> / <out>_hidvis).
Usage: python analyze/v12/eval_s2c_v12.py --runs green:mixed:0 ... [--prefix v12_s2c] [--out stage2c] [--epochs ...]
       (--runs d6:data:0 gives the data-only D6 reference on the same subsets.)
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
sys.path.insert(0, _HERE)
import eval_s2_v11 as E  # noqa
from eval_trial_v12 import RES, load_v12  # noqa
from s2run import S2Feed  # noqa
import util  # noqa

SETS_H = {'green': ('v12_hide_v11_s2t_green', 'test'), 'swap': ('v12_hide_v11_s2t_green_swap', 'test'),
          'red': ('v12_hide_v11_s2t_red', 'test'), 'pref': ('v12_hide_v11_s2_pref', 'test')}
E.SETS = SETS_H     # E.load_set / E.true_pleasure read from these paths (P_A2 uses A-2's own camera, unchanged)


@torch.no_grad()
def run_q2_in(model, path, split, dev, p2=None, batch=50, key='self_vision_in'):
    """E.run_q2 with the input from self_vision_in; prediction error against the original self_vision."""
    g = h5py.File(path, 'r')[split]
    model.reset(g['self_motion'][()])
    N = g['self_vision'].shape[0]
    Q, LS = [], []
    util.seed_all(0)
    for b0 in range(0, N, batch):
        xi = torch.from_numpy(util.scale_vision(util.transpose_vision(g[key][b0:b0 + batch]))).to(dev)
        xt = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        model.init_state(xi.shape[0])
        q, ls = [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            ov = None if p2 is None else torch.from_numpy(p2[b0:b0 + batch, t]).float().to(dev)
            out = model({'self_vision': xi[:, t]}, pm, pm, p2_override=ov)
            q.append(out['q2_hat'].cpu().numpy())
            ls.append(torch.abs(out['self_vision'] - xt[:, t + 1]).mean((1, 2, 3)).cpu().numpy())
        Q.append(np.stack(q, 1)); LS.append(np.stack(ls, 1))
    return np.concatenate(Q), np.concatenate(LS)


def t5_pairs_masked(dg, dr, Mg, Mr, r2, r1=3.0, tmin=20):
    """E.t5_pairs restricted to frames with Mg (green set) / Mr (red set) true."""
    pairs = []
    from scipy.spatial import cKDTree
    for L in (0, 1):
        eg, er = np.where(dg['layout'] == L)[0], np.where(dr['layout'] == L)[0]

        def frames(d, eps, M):
            e, t = np.meshgrid(eps, np.arange(tmin, 100), indexing='ij')
            e, t = e.ravel(), t.ravel()
            k = M[e, t]
            e, t = e[k], t[k]
            return e, t, d['other_position'][e, t], d['self_position'][e, t]
        ge, gt, g2, g1 = frames(dg, eg, Mg)
        re_, rt, r2p, r1p = frames(dr, er, Mr)
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
    return np.array(pairs, dtype=int).reshape(-1, 5)


def get_pairs(out_dir, D, Mg, Mr):
    pp = os.path.join(out_dir, 't5_pairs.npz')
    if not os.path.exists(pp):
        pr, r_used = t5_pairs_masked(D['green'], D['red'], Mg, Mr, 1.0), 1.0
        if min(int((pr[:, 0] == L).sum()) for L in (0, 1)) < 500:
            pr, r_used = t5_pairs_masked(D['green'], D['red'], Mg, Mr, 1.5), 1.5
        np.savez(pp, pairs=pr, radius=r_used)
    z = np.load(pp)
    return z['pairs'], float(z['radius'])


def corr_mask(Q, P, M):
    m = np.repeat(M[:, :, None], 8, 2)
    return float(np.corrcoef(Q[m], P[m])[0, 1])


def sign_agree(Q, P, lay, M):
    """Sec.35 sign agreement (bins 0,1,6,7, frames x bins with P_A2 != 0) on frames M, overall and per layout."""
    b = [0, 1, 6, 7]
    out = {}
    for nm, lm in (('all', np.ones(len(lay), bool)), ('original', lay == 0), ('swap_rg', lay == 1)):
        q, p = Q[lm][:, :, b], P[lm][:, :, b]
        m = M[lm][:, :, None].repeat(4, 2) & (p != 0)
        out[nm] = float((np.sign(q[m]) == np.sign(p[m])).mean()) if m.any() else float('nan')
    return out


def evaluate(Q, LS, D, PA2, H, eps2, pairs, radius, tag, st):
    gsel = lambda q, idx: np.take_along_axis(q, idx, 1).mean(1)  # noqa
    L0 = D['swap']['layout'][eps2]
    newG = np.array([E.RB[int(l)] for l in L0]); oldG = np.array([E.GB[int(l)] for l in L0])
    sT2 = {}
    for name, ts in (('t50', [50]), ('t50_52', [50, 51, 52])):
        sT2[name] = np.mean([(gsel(Q['swap'][eps2, t], newG) - gsel(Q['swap'][eps2, t], oldG))
                             - (gsel(Q['green'][eps2, t], newG) - gsel(Q['green'][eps2, t], oldG)) for t in ts], 0)
    dg = D['green']
    pred_new = np.zeros((len(eps2), 8))
    for L in (0, 1):
        m = dg['layout'] == L
        Xf = np.concatenate([dg['other_position'][m][:, 20:100].reshape(-1, 2), dg['self_position'][m][:, 20:100].reshape(-1, 2)], 1)
        knn = KNeighborsRegressor(10).fit(Xf, Q['green'][m][:, 20:100].reshape(-1, 8))
        tgt = np.where(1 - L0 == L)[0]
        if len(tgt):
            e2 = eps2[tgt]
            pred_new[tgt] = knn.predict(np.concatenate([D['swap']['other_position'][e2, 50], D['swap']['self_position'][e2, 50]], 1))
    sT2_pos = (gsel(pred_new, newG) - gsel(pred_new, oldG)) - (gsel(Q['green'][eps2, 50], newG) - gsel(Q['green'][eps2, 50], oldG))
    Lp = pairs[:, 0]
    gr = lambda q, L: gsel(q, np.array([E.GB[int(l)] for l in L])) - gsel(q, np.array([E.RB[int(l)] for l in L]))  # noqa
    dT5 = gr(Q['green'][pairs[:, 1], pairs[:, 2]], Lp) - gr(Q['red'][pairs[:, 3], pairs[:, 4]], Lp)
    r = {'tag': tag, 'n_T2_episodes': int(len(eps2)), 't5_radius': radius,
         't5_pairs_per_layout': {str(L): int((Lp == L).sum()) for L in (0, 1)},
         'T2_mean': {k: float(v.mean()) for k, v in sT2.items()}, 'T2_pos_readout_mean': float(sT2_pos.mean()),
         'T5_mean': float(dT5.mean()),
         'corr_Q2_PA2': {'green_set': E.corr_q2(Q['green'], PA2['green'], dg['layout']),
                         'pref_test': E.corr_q2(Q['pref'], PA2['pref'], D['pref']['layout']),
                         'by_frames': {f'{k}_{nm}': corr_mask(Q[k], PA2[k], H[k] == v)
                                       for k in ('green', 'pref') for nm, v in (('hidden', 1), ('visible', 0))}},
         'T5_by_t_bins': {f'{a}-{b}': float(dT5[(pairs[:, 2] >= a) & (pairs[:, 2] < b)].mean()) if
                          ((pairs[:, 2] >= a) & (pairs[:, 2] < b)).any() else float('nan')
                          for a, b in ((20, 40), (40, 60), (60, 80), (80, 100))}}
    r['sign_agreement'] = {f'{k}_{nm}': sign_agree(Q[k], PA2[k], D[k]['layout'], H[k] == v)
                           for k in ('green', 'pref') for nm, v in (('hidden', 1), ('visible', 0))}
    if st != 'd6':
        r['motion_readout'] = {'green_set': E.motion_readout(Q['green'], dg), 'pref_test': E.motion_readout(Q['pref'], D['pref'])}
        r['self_vision_L1_per_pixel'] = {k: float(v.mean()) for k, v in LS.items()}
    per = dict(sT2_t50=sT2['t50'], sT2_t50_52=sT2['t50_52'], sT2_pos=sT2_pos, dT5=dT5, t5_green_ep=pairs[:, 1],
               t2_episodes=eps2)
    return r, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--prefix', default='v12_s2c')
    ap.add_argument('--out', default='stage2c')
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--input_key', default='self_vision_in', help='self_vision only for the reproduction check')
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cudnn.deterministic = True
    D = {k: E.load_set(k) for k in SETS_H}
    PA2 = {k: E.true_pleasure(k, D[k]) for k in SETS_H}
    H = {k: h5py.File(D[k]['path'], 'r')[D[k]['split']]['a2_hidden'][:, :100] for k in SETS_H}
    assert (H['green'] == H['swap']).all() and (H['green'] == H['red']).all()   # same test schedule seed
    outs = {'hidden': os.path.join(RES, 'v12', a.out, 'eval'), 'visible': os.path.join(RES, 'v12', a.out + '_hidvis', 'eval')}
    sub = {}
    for kind, val in (('hidden', 1), ('visible', 0)):
        os.makedirs(outs[kind], exist_ok=True)
        eps2 = np.where(H['swap'][:, 50] == val)[0]
        pairs, radius = get_pairs(outs[kind], D, H['green'] == val, H['red'] == val)
        sub[kind] = (eps2, pairs, radius)
    for spec in a.runs:
        st, c, s = spec.split(':')
        s = int(s)
        for e in (a.epochs if st != 'd6' else [0]):
            tag = f'{st}_{c}_s{s}_ep{e}' if st != 'd6' else 'd6_PA2'
            if st == 'd6':
                Q, LS = PA2, {}
            else:
                m, _ = load_v12(f'{a.prefix}_{st}_{c}_s{s}', s, e, dev)
                model = S2Feed(m)
                Q, LS = {}, {}
                for k in SETS_H:
                    Q[k], LS[k] = run_q2_in(model, D[k]['path'], D[k]['split'], dev, key=a.input_key)
                # T3 on the hidden test (Sec.46 D): self-vision error (vs the true next view) with process-2 =
                # generated Q2 / zero / D6 (P_A2), green set, split by A-2 hidden / visible frames (t >= 1)
                t3 = {'gen': LS['green']}
                for nm, p2 in (('zero', np.zeros_like(PA2['green'])), ('d6', PA2['green'])):
                    _, t3[nm] = run_q2_in(model, D['green']['path'], D['green']['split'], dev, p2=p2, key=a.input_key)
                Hg = H['green'][:, 1:] == 1
                t3s = {f'{nm}_{fr}': float(v[:, 1:][Hg if fr == 'hidden' else ~Hg].mean())
                       for nm, v in t3.items() for fr in ('hidden', 'visible')}
            for kind in ('hidden', 'visible'):
                r, per = evaluate(Q, LS, D, PA2, H, *sub[kind], tag, st)
                r['subset'] = kind
                if st != 'd6':
                    r['T3_self_L1_green_by_frames'] = t3s
                json.dump(r, open(os.path.join(outs[kind], f'{tag}.json'), 'w'), indent=1)
                np.savez_compressed(os.path.join(outs[kind], f'{tag}_per.npz'), **per)
                print(kind, tag, json.dumps({k: r[k] for k in ('n_T2_episodes', 't5_pairs_per_layout', 'T2_mean', 'T5_mean')}), flush=True)


if __name__ == '__main__':
    main()
