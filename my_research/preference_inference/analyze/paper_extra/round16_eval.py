"""
Paper round 16 Sec.2 (docs/paper_instructions_round16.md; docs/paper_extra_log.md Sec.40; exploratory, outside the
pre-registration; procedure committed before computing). No training: the stage-I predictor and G are fixed; forward
passes only. Protocol and loss path = stageC_eval (stage-B set vdirb, 1200 episodes; t = 0 vision shown, t > 0 hidden;
A-1's true motion fed; eval mode; util.seed_all(0); cudnn deterministic; batch 50). Q2 is replaced at EVERY step
t = 0..99 through p2_override = [0, 0, Q2_t, 1] (G keeps running inside; only its output is replaced). Losses per
episode and step (step t predicts frame t + 1): self-vision L1, feature prediction from h1 (MSE), from h2 (MSE);
kept as episode means over steps 0..49 (target frames 1-50, main) and 0..99 (reference).
Q2 inputs (sigma_e = s_pref of the stage-B labels, +1 if the liked colour is on side A; c+ the stage-B constant;
s in {1, 2, 4, 7}; every input is clipped to [-1, 1] and the clipped fraction is saved):
  a      G's own output (no override; Q2 and the pre-tanh fc output z are recorded)
  z      0
  p+-s   +-sigma s c+ (constant)                  a+-s   Q2a +- sigma s c+
  t+-s   tanh(z +- sigma s c+) (free G only; added, descriptive)
  pos    cross-fitted fit of Q2a on a degree-2 polynomial (with cross terms) of A-2 and A-1 positions (x, y) at frame
         t plus the layout; 5 folds of episodes (stage-C folds, layout x preference), each fold predicted from the
         other four; OLS over all steps 0..99 of the training episodes; R^2 saved (steps 0..99 and 0..49)
  res    Q2a - pos            ax  (Q2a . c^) c^ (c^ = c+/|c+|)            orth  Q2a - ax
One-bit G (Q2a = u c+): a, z, a+-s only.
Also saved: for a+-s (and t+-s) the per-episode mean over steps 0..49 of sigma (Q2' - Q2a) . c^ and of |Q2' - Q2a|.
Modes:
  --check   GPU bit identity with the saved stage-C arrays on v12_s2_pref_mixed_s0 ep200: Q2 of the a-pass vs
            StageC/eval Q2, and stageC_eval.gradients at G's point vs the saved grad_tot_G (1200 episodes).
  --models <exp>:<seed>:<epoch>[,<epoch>...] ...   (every run also compares Q2a with the saved stage-C Q2)
-> data/result/paper_extra/Round16/eval/<exp>_ep<e>.npz (+ .json) ; check -> Round16/check/summary.json
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stageC_eval as C  # noqa  (protocol, loaders, gradients; registers 'vdirb' and the one-bit class)
import stageC_analysis as SC  # noqa  (stage-B labels, folds)
from sklearn.linear_model import LinearRegression  # noqa
from sklearn.preprocessing import PolynomialFeatures  # noqa

OUT = os.path.join(C.RES, 'paper_extra', 'Round16')
SVALS = (1, 2, 4, 7)
CP = C.CP.astype(np.float32)
CH = C.CPU_.astype(np.float32)
SIG = SC.LAB['pref'].astype(np.float32)                      # (1200,) +1 liked colour on side A
WIN = {'0_49': slice(0, 50), '0_99': slice(0, 100)}


def load_pos():
    d = C.E.load_set('vdirb')
    g = h5py.File(d['path'], 'r')[d['split']]
    return g['other_position'][()][:, :100].astype(np.float64), g['self_position'][()][:, :100].astype(np.float64)


@torch.no_grad()
def run(model, g, N, dev, Q=None, fc=None, B=50):
    """-> L (N, 100, 3) [L1, FP h1, FP h2], and (Q2, z) of G's own pass when Q is None."""
    C.util.seed_all(0)
    Ls, Qs, Zs = [], [], []
    for b0, b1, x, m in C.batches(g, N, B):
        x, m = x.to(dev), m.to(dev)
        model.init_state(b1 - b0)
        ls, qs, zs = [], [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            p2 = None if Q is None else C.p2_input(torch.from_numpy(Q[b0:b1, t]).to(dev))
            o = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=p2)
            tgt = x[:, t + 1]
            st = model.get_state()
            l1 = F.l1_loss(o['self_vision'], tgt, reduction='none').mean((1, 2, 3))
            f1 = F.mse_loss(model.predict_feature(st['self'][0]), model.self_vision_encoder_module(tgt),
                            reduction='none').mean(1)
            f2 = F.mse_loss(model.predict_feature(st['other'][0]), model.other_vision_encoder_module(tgt),
                            reduction='none').mean(1)
            ls.append(torch.stack([l1, f1, f2], 1).cpu().numpy())
            if Q is None:
                qs.append(o['q2_hat'].cpu().numpy())
                zs.append(fc.out.cpu().numpy())
        Ls.append(np.stack(ls, 1))
        if Q is None:
            Qs.append(np.stack(qs, 1)); Zs.append(np.stack(zs, 1))
    L = np.concatenate(Ls)
    return (L, np.concatenate(Qs), np.concatenate(Zs)) if Q is None else L


def pos_fit(Qa, P2, P1):
    """cross-fitted degree-2 polynomial of (A-2 x, y, A-1 x, y) + layout -> (N, 100, 8), R^2."""
    N = Qa.shape[0]
    lay = np.repeat(SC.LAB['lay'][:, None], 100, 1)[..., None]
    X = PolynomialFeatures(2, include_bias=False).fit_transform(np.concatenate([P2, P1], -1).reshape(-1, 4))
    X = np.concatenate([X, lay.reshape(-1, 1)], 1).reshape(N, 100, -1)
    out = np.zeros_like(Qa, dtype=np.float64)
    for f in range(SC.NF):
        tr, va = SC.FOLD != f, SC.FOLD == f
        reg = LinearRegression().fit(X[tr].reshape(-1, X.shape[-1]), Qa[tr].reshape(-1, 8))
        out[va] = reg.predict(X[va].reshape(-1, X.shape[-1])).reshape(-1, 100, 8)
    r2 = {}
    for w, sl in WIN.items():
        y, p = Qa[:, sl].reshape(-1, 8).astype(np.float64), out[:, sl].reshape(-1, 8)
        ss, st = ((y - p) ** 2).sum(0), ((y - y.mean(0)) ** 2).sum(0)
        r2[w] = {'pooled': float(1 - ss.sum() / st.sum()), 'per_dim': (1 - ss / st).tolist()}
    return out.astype(np.float32), r2


def clip(Q):
    return np.clip(Q, -1.0, 1.0).astype(np.float32), float((np.abs(Q) > 1.0).mean())


def evaluate(exp, s, e, dev, P2, P1):
    model, _ = C.load_v12(exp, s, e, dev)
    model.eval()
    fc = C.Hook(model.motion_generator_module.fc)
    d = C.E.load_set('vdirb')
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    t0 = time.time()
    La, Qa, Za = run(model, g, N, dev, None, fc)
    onebit = type(model).__name__.endswith('OneBit')
    sv = np.load(os.path.join(C.OUT, 'eval', f'{exp}_ep{e}.npz'))
    info = {'exp': exp, 'seed': s, 'epoch': e, 'onebit': bool(onebit), 'N': N,
            'Q2_vs_stageC_identical': bool(np.array_equal(Qa, sv['Q2'])),
            'Q2_vs_stageC_maxabs': float(np.abs(Qa - sv['Q2']).max()), 'clip': {}, 'r2_pos': None}
    sig = SIG[:, None, None]
    Q = {'z': np.zeros_like(Qa)}
    for sv_ in SVALS:
        d_ = sig * sv_ * CP
        Q[f'p+{sv_}'] = np.broadcast_to(d_, Qa.shape).copy()
        Q[f'p-{sv_}'] = np.broadcast_to(-d_, Qa.shape).copy()
        Q[f'a+{sv_}'] = Qa + d_
        Q[f'a-{sv_}'] = Qa - d_
        if not onebit:
            Q[f't+{sv_}'] = np.tanh(Za + d_)
            Q[f't-{sv_}'] = np.tanh(Za - d_)
    if not onebit:
        Qp, info['r2_pos'] = pos_fit(Qa, P2, P1)
        Q['pos'], Q['res'] = Qp, Qa - Qp
        ax = (Qa @ CH)[..., None] * CH
        Q['ax'], Q['orth'] = ax, Qa - ax
    sav = {}
    for w, sl in WIN.items():
        for i, nm in enumerate(('L1', 'FP1', 'FP2')):
            sav[f'a|{w}|{nm}'] = La[:, sl, i].mean(1).astype(np.float32)
    for k in Q:
        q, info['clip'][k] = clip(Q[k])
        if k[0] in 'at' and k[1] in '+-':
            dlt = q - Qa
            sav[f'dax|{k}'] = (SIG[:, None] * (dlt[:, :50] @ CH)).mean(1).astype(np.float32)
            sav[f'dnorm|{k}'] = np.linalg.norm(dlt[:, :50], axis=-1).mean(1).astype(np.float32)
        L = run(model, g, N, dev, q)
        for w, sl in WIN.items():
            for i, nm in enumerate(('L1', 'FP1', 'FP2')):
                sav[f'{k}|{w}|{nm}'] = L[:, sl, i].mean(1).astype(np.float32)
    info['t_s'] = time.time() - t0
    return sav, info


def check(dev):
    exp, s, e = 'v12_s2_pref_mixed_s0', 0, 200
    model, _ = C.load_v12(exp, s, e, dev)
    model.eval()
    fc = C.Hook(model.motion_generator_module.fc)
    d = C.E.load_set('vdirb')
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    sv = np.load(os.path.join(C.OUT, 'eval', f'{exp}_ep{e}.npz'))
    t0 = time.time()
    La, Qa, Za = run(model, g, N, dev, None, fc)
    t_run = time.time() - t0
    gt, gv = C.gradients(model, g, N, dev, sv['Q2'])
    L2 = run(model, g, N, dev, Qa)
    R = {'model': f'{exp} ep{e}', 'N': N, 'Q2_identical': bool(np.array_equal(Qa, sv['Q2'])),
         'Q2_maxabs': float(np.abs(Qa - sv['Q2']).max()),
         'grad_tot_G_identical': bool(np.array_equal(gt, sv['grad_tot_G'])),
         'grad_tot_G_maxabs': float(np.abs(gt - sv['grad_tot_G']).max()),
         'grad_vis_G_identical': bool(np.array_equal(gv, sv['grad_vis_G'])),
         'tanh_z_equals_Q2': bool(np.array_equal(np.tanh(Za), Qa)), 'tanh_z_maxabs': float(np.abs(np.tanh(Za) - Qa).max()),
         'override_own_Q2_identical': bool(np.array_equal(La, L2)), 't_one_pass_s': t_run}
    os.makedirs(os.path.join(OUT, 'check'), exist_ok=True)
    json.dump(R, open(os.path.join(OUT, 'check', 'summary.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--models', nargs='*', default=[])
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if a.check:
        return check(dev)
    P2, P1 = load_pos()
    os.makedirs(os.path.join(OUT, 'eval'), exist_ok=True)
    for spec in a.models:
        exp, s, eps = spec.split(':')
        for e in map(int, eps.split(',')):
            sav, info = evaluate(exp, int(s), e, dev, P2, P1)
            tag = f'{exp}_ep{e}'
            np.savez_compressed(os.path.join(OUT, 'eval', f'{tag}.npz'), **sav)
            json.dump(info, open(os.path.join(OUT, 'eval', f'{tag}.json'), 'w'), indent=1)
            print(tag, 'Q2 identical', info['Q2_vs_stageC_identical'], 't', round(info['t_s'], 1), flush=True)


if __name__ == '__main__':
    main()
