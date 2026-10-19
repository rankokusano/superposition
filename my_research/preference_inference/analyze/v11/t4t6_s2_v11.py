"""
v11 stage-2 secondary: T4, T6 and A-2 tracking in the predicted images
(docs/v11_experiment_log.md Sec.7.53, method recorded before computing).
Q2 = generated / zero / P_A2 (D6). A-2's position is read from the predicted image as in
Sec.7.33 (p2test_v11.read_a2), distance calibrated on v11_s2t_green true views (every 5th step).
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import load_model  # noqa
from eval_s2_v11 import GB, OUT, RB, load_set, true_pleasure  # noqa
from eval_m1_v11 import refs  # noqa
from p2test_v11 import calibrate, read_a2  # noqa
import util  # noqa

OUT4 = os.path.join(os.path.dirname(OUT), 't4t6')
SETS4 = ('green', 'swap', 'pref')


@torch.no_grad()
def run(model, d, dev, alpha, beta, p2=None, batch=50):
    """-> Q2 (N, 100, 8), est A-2 position of v_{t+1} (N, 100, 2)."""
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    Q, E = [], []
    util.seed_all(0)
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        sp = d['self_position'][b0:b0 + batch]
        model.init_state(x.shape[0])
        q, e = [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            ov = None if p2 is None else torch.from_numpy(p2[b0:b0 + batch, t]).float().to(dev)
            out = model({'self_vision': x[:, t]}, pm, pm, p2_override=ov)
            q.append(out['q2_hat'].cpu().numpy())
            p01 = ((out['self_vision'] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            e.append(read_a2(p01, sp[:, t + 1], alpha, beta))
        Q.append(np.stack(q, 1)); E.append(np.stack(e, 1))
    return np.concatenate(Q), np.concatenate(E)


def tracking(est, d):
    err = np.linalg.norm(est[:, 1:100] - d['other_position'][:, 2:101], axis=-1)
    ok = np.isfinite(err)
    return {'median': float(np.median(err[ok])), 'mean': float(err[ok].mean()), 'frac_read': float(ok.mean())}


def t4(est_s, est_n, ds):
    L0 = ds['layout']
    p50 = ds['other_position'][:, 50]
    newG = np.concatenate([refs(L0[i:i + 1], 'Red') for i in range(len(L0))])[:, 0]   # old red place
    oldG = np.concatenate([refs(L0[i:i + 1], 'Green') for i in range(len(L0))])[:, 0]

    def score(D):
        def cos(a, b):
            return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)
        return cos(D, newG - p50) - cos(D, oldG - p50)
    Ds, Dn = est_s[:, 59] - est_s[:, 50], est_n[:, 59] - est_n[:, 50]
    ok = np.isfinite(Ds).all(1) & np.isfinite(Dn).all(1)
    ss, sn = score(Ds), score(Dn)
    true = score(ds['other_position'][:, 60] - ds['other_position'][:, 51])
    return {'swap': float(ss[ok].mean()), 'noswap': float(sn[ok].mean()), 'swap_minus_noswap': float((ss - sn)[ok].mean()),
            'n_ep': int(ok.sum()), 'true_A2_swap': float(true.mean())}, np.where(ok, ss - sn, np.nan)


def gsel(q, L, B):
    idx = np.where(L[:, None, None] == 0, np.array(B[0])[None, None], np.array(B[1])[None, None])
    return np.take_along_axis(q, idx, -1).mean(-1)


def t6(Q, P, d):
    L = d['layout']
    dq = gsel(Q, L, GB) - gsel(Q, L, RB)
    dp = gsel(P, L, GB) - gsel(P, L, RB)
    p = d['other_position'][:, :100]
    m = d['other_position'][:, 1:101] - p
    u = refs(L, 'Green') - p
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
    comp = (m * u).sum(-1)
    aw = (d['a2_source'][:, :100] != 3) & (np.linalg.norm(m, axis=-1) > 1e-6)
    out = {}
    for Lc, nm in ((0, 'original'), (1, 'swap_rg')):
        e = L == Lc
        a = dq[e][:, 1:].ravel(), dp[e][:, 1:].ravel()
        ma = aw[e][:, 1:].ravel()
        out[nm] = {'corr_dQ_dP': float(np.corrcoef(*a)[0, 1]),
                   'corr_dQ_A2step_toward_green': float(np.corrcoef(dq[e][:, 1:].ravel()[ma], comp[e][:, 1:].ravel()[ma])[0, 1]),
                   'corr_dP_A2step_toward_green': float(np.corrcoef(dp[e][:, 1:].ravel()[ma], comp[e][:, 1:].ravel()[ma])[0, 1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    args = ap.parse_args()
    os.makedirs(OUT4, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in SETS4}
    PA2 = {k: true_pleasure(k, D[k]) for k in SETS4}
    g = h5py.File(D['green']['path'], 'r')['test']
    alpha, beta = calibrate(g['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    for spec in args.runs:
        st, c, s = spec.split(':')
        for e in args.epochs:
            tag = f'{st}_{c}_s{s}_ep{e}'
            model, _ = load_model(f'v11_s2_{st}_{c}_s{s}', int(s), e, dev)
            r, per = {'tag': tag, 'calib': [alpha, beta]}, {}
            for cond in ('gen', 'zero', 'd6'):
                Q, EST = {}, {}
                for k in SETS4:
                    p2 = None if cond == 'gen' else (np.zeros_like(PA2[k]) if cond == 'zero' else PA2[k])
                    Q[k], EST[k] = run(model, D[k], dev, alpha, beta, p2)
                rc = {'tracking': {k: tracking(EST[k], D[k]) for k in ('green', 'pref')}}
                rc['T4'], per[f'{cond}_T4'] = t4(EST['swap'], EST['green'], D['swap'])
                if cond == 'gen':
                    rc['T6'] = {'green_set': t6(Q['green'], PA2['green'], D['green']),
                                'pref_test': t6(Q['pref'], PA2['pref'], D['pref'])}
                    for lk in (0, 1):
                        m = D['pref']['a2_liked'] == lk
                        sub = {kk: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v) for kk, v in D['pref'].items()}
                        rc['T6'][f'pref_test_liked{lk}'] = t6(Q['pref'][m], PA2['pref'][m], sub)
                r[cond] = rc
            json.dump(r, open(os.path.join(OUT4, f'{tag}.json'), 'w'), indent=1)
            np.savez_compressed(os.path.join(OUT4, f'{tag}_per.npz'), **per)
            print(tag, json.dumps({cc: {'trk': r[cc]['tracking']['green']['median'], 'T4': r[cc]['T4']['swap_minus_noswap']} for cc in ('gen', 'zero', 'd6')}), flush=True)


if __name__ == '__main__':
    main()
