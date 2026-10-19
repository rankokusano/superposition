"""
v11 stage-2 secondary: T1, T3 and the spread of Q2 (docs/v11_experiment_log.md Sec.7.51,
method recorded before computing). For one stage-2 run and one G checkpoint, on v11_s2t_green
and on the v11_s2_pref test split (masking/seeding as eval_s2_v11.py):
  * T3: Q2 replaced by zero / generated / P_A2 (D6) / position proxy -> self-vision L1 per pixel
    (vs x_{t+1}) and other-side feature-prediction MSE (predict_feature(h2) vs the unmasked
    other encoder of x_{t+1}), per episode; and h2 -> A-2 position R^2 per condition;
  * T1: generated Q2 -> A-2 position / A-1 position R^2 (all, per layout); share of variance in
    bins 2..5;
  * spread of Q2: per-bin mean, overall std, between-episode / within-episode (time) /
    across-bin std, 1-99 percentiles.
P_A2 values for T1 / spread (data only) are written once as tag 'd6_PA2'.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import load_model  # noqa
from eval_s2_v11 import OUT, SETS, load_set, true_pleasure, r2_cv  # noqa
import util  # noqa

SETS2 = ('green', 'pref')
OUT3 = os.path.join(os.path.dirname(OUT), 't1t3')


def proxy(d, PA2):
    rel = d['other_position'][:, :100] - d['self_position'][:, :100]
    ang = np.arctan2(rel[..., 1], rel[..., 0])
    k = np.clip(np.floor((ang + np.pi) / (np.pi / 4)).astype(int), 0, 7)
    dist = np.maximum(np.linalg.norm(rel, axis=-1), 1e-3)
    P = np.zeros(rel.shape[:2] + (8,), np.float32)
    np.put_along_axis(P, k[..., None], (1 / dist)[..., None].astype(np.float32), -1)
    return P * (np.abs(PA2).mean() / np.abs(P).mean())


@torch.no_grad()
def run(model, d, dev, p2=None, batch=50):
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    Q, LS, LF, H2 = [], [], [], []
    util.seed_all(0)
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        model.init_state(x.shape[0])
        q, ls, lf, h2 = [], [], [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            ov = None if p2 is None else torch.from_numpy(p2[b0:b0 + batch, t]).float().to(dev)
            out = model({'self_vision': x[:, t]}, pm, pm, p2_override=ov)
            q.append(out['q2_hat'].cpu().numpy())
            ls.append(torch.abs(out['self_vision'] - x[:, t + 1]).mean((1, 2, 3)).cpu().numpy())
            os_ = model.get_state()['other'][0]
            tgt = model.other_vision_encoder_module(x[:, t + 1])
            lf.append(F.mse_loss(model.predict_feature(os_), tgt, reduction='none').mean(1).cpu().numpy())
            h2.append(os_.cpu().numpy())
        Q.append(np.stack(q, 1)); LS.append(np.stack(ls, 1)); LF.append(np.stack(lf, 1)); H2.append(np.stack(h2, 1))
    return np.concatenate(Q), np.concatenate(LS), np.concatenate(LF), np.concatenate(H2)


def pos_r2(X, d, who):
    ts = np.arange(1, 100)
    N = X.shape[0]
    Xf = X[:, ts].reshape(-1, X.shape[-1])
    Y = d[who][:, ts].reshape(-1, 2)
    g = np.repeat(np.arange(N)[:, None], len(ts), 1).reshape(-1)
    lay = np.repeat(d['layout'][:, None], len(ts), 1).reshape(-1)
    out = {'all': r2_cv(Xf, Y, g)}
    for L, nm in ((0, 'original'), (1, 'swap_rg')):
        m = lay == L
        out[nm] = r2_cv(Xf[m], Y[m], g[m])
    return out


def spread(Q):
    Q = Q[:, 1:100]
    v = Q.reshape(-1, 8).var(0)
    return {'bin_mean': Q.reshape(-1, 8).mean(0).tolist(), 'std_all': float(Q.std()),
            'std_between_episode': float(Q.mean(1).std(0).mean()),
            'std_within_episode_time': float(Q.std(1).mean()),
            'std_across_bins': float(Q.std(2).mean()),
            'p1_p99': [float(np.percentile(Q, 1)), float(np.percentile(Q, 99))],
            'var_share_bins2_5': float(v[2:6].sum() / v.sum())}


def boot(a, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='setting:cond:seed, or d6:data:0 for the data-only values')
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    args = ap.parse_args()
    os.makedirs(OUT3, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in SETS2}
    PA2 = {k: true_pleasure(k, D[k]) for k in SETS2}
    PX = {k: proxy(D[k], PA2[k]) for k in SETS2}
    for spec in args.runs:
        st, c, s = spec.split(':')
        if st == 'd6':
            r = {k: {'T1_A2pos': pos_r2(PA2[k], D[k], 'other_position'), 'T1_A1pos': pos_r2(PA2[k], D[k], 'self_position'),
                     'spread': spread(PA2[k]), 'proxy_spread': spread(PX[k]),
                     'proxy_T1_A2pos': pos_r2(PX[k], D[k], 'other_position')} for k in SETS2}
            json.dump(r, open(os.path.join(OUT3, 'd6_PA2.json'), 'w'), indent=1)
            print('d6 done', flush=True)
            continue
        for e in args.epochs:
            tag = f'{st}_{c}_s{s}_ep{e}'
            model, _ = load_model(f'v11_s2_{st}_{c}_s{s}', int(s), e, dev)
            r = {'tag': tag}
            per = {}
            for k in SETS2:
                rk = {}
                Qg = None
                for cond, p2 in (('gen', None), ('zero', np.zeros_like(PA2[k])), ('d6', PA2[k]), ('proxy', PX[k])):
                    Q, LS, LF, H2 = run(model, D[k], dev, p2)
                    if cond == 'gen':
                        Qg = Q
                    rk[cond] = {'self_L1': float(LS.mean()), 'other_fp_mse': float(LF.mean()),
                                'h2_A2pos': pos_r2(H2, D[k], 'other_position')['all']}
                    per[f'{k}_{cond}_self'] = LS.mean(1); per[f'{k}_{cond}_fp'] = LF.mean(1)
                for lk in ('self', 'fp'):
                    dd = per[f'{k}_gen_{lk}'] - per[f'{k}_zero_{lk}']
                    rk[f'gen_minus_zero_{lk}'] = {'mean': float(dd.mean()), 'ci': boot(dd)}
                rk['T1_A2pos'] = pos_r2(Qg, D[k], 'other_position')
                rk['T1_A1pos'] = pos_r2(Qg, D[k], 'self_position')
                rk['spread'] = spread(Qg)
                r[k] = rk
            json.dump(r, open(os.path.join(OUT3, f'{tag}.json'), 'w'), indent=1)
            np.savez_compressed(os.path.join(OUT3, f'{tag}_per.npz'), **per)
            print(tag, json.dumps({k: dict(gen=r[k]['gen'], zero=r[k]['zero'], T1=r[k]['T1_A2pos']['all']) for k in SETS2}), flush=True)


if __name__ == '__main__':
    main()
