"""
Paper round 22 (paper/check/paper_instructions_round22.md; docs/paper_extra_log.md Sec.46; exploratory, outside the
pre-registration; the main mode is fixed before computing). GPU (GPU1-3), no training. Data and flow of round 21
(round21_eval.py, data/data/paper_r21_swap, static A-2, swap at t = 20).

Mode `check` (Sec.1 check 2 and smoke; first 50 episodes, pair 0, ep200): the sign-random 80% predictor
(paper_extra/C_signrand/s1_signrand_l1_s<s>) through round21_eval.evaluate (keys must equal round 21's); the other
runs below for p50 and p80, with the original inputs (hold / flip / zero, V1) compared with the stored round-21 values
of the same episodes (must be bit-identical: same batch of 50 and seeds); shapes and times only.

Mode `main` (Sec.2), checkpoints ep160-200 (l1 stage), pairs 0-5:
  sr80  round21_eval.evaluate unchanged -> Round22/eval21/sr80_s<s>_ep<e>.npz (the round-21 keys: self and other,
        V1 / V0, hold / flip / zero), plus the other runs 'orig' below -> Round22/eval/sr80_s<s>_ep<e>.npz;
  p50, p80   other runs 'orig' + 'X' below -> Round22/eval/<c>_s<s>_ep<e>.npz;
  none, sr50 other runs 'orig'.
Other run (as round21_eval.freeze_flow 'other', identical code path): t < T0 vision shown, then K = 10 steps with
vision masked, motion hidden, p1_override = 0, p2_override = [0, 0, P2, 1]. Inputs P2 (8 ranges):
  orig  hold = compute_pleasure(A-2's view, frame T0-1), flip = -hold, zero          (V1, swap and noswap)
  X1    x2_hold / x2_flip / x4_hold / x4_flip = clip(+-k * hold, -1, 1), k = 2, 4    (V1, swap and noswap)
  X2    m_hold / m_flip = clip(+-f * hold, -1, 1), f = mean|P_self| / mean|P_other| from Round22/check/check.json
        (V1 factor 0.8525: the other's input is already larger than the self's, so X2 is a reduction)
  X3    b_old / b_new = +0.28 (G5b's A) in the one range nearest the direction from A-2 to L_old / L_new
        (argmax cos(angle - BIN_ANGLES)), other ranges 0                              (V1, swap and noswap)
  V0    (freeze from T0 = 20, no post-swap frame seen; swap = noswap input) b_old, b_new, zero on noswap.
Read-outs per run:
  img   round 21's: A-2 position est_k from the predicted image (p2test_v11.read_a2, common-test calibration, A-1 at
        frame T0+k-1), s_L = OLS slope over k of (est_k - p2) . unit(L - p2) (>= 5 finite points),
        Delta = s_Lold - s_Lnew;
  feat  (Sec.2.4, V1 only) the predicted other-viewpoint feature f_k = predict_feature(os) at step T0+k-1 (os =
        get_state()['other'][0], as the FP monitor; a prediction of other_vision_encoder(frame T0+k)) compared with
        F_L(d) = other_vision_encoder(view of frame T0+k with A-2 moved d = -5..5 steps toward L) (round22_views.py;
        both layouts, colour-agnostic minimum as round 21's main self read-out): d*_L(k) = argmin_d mean (f_k - F)^2
        over valid d; (episode, L) used only if d = 0..5 are valid (the self rule); s_L = OLS slope of d*_L over
        k = 1..10; Delta_feat = s_Lold - s_Lnew.
Keys '<d>_<V>_<input>_img' (N,), '_est' (N, 10, 2), '<d>_V1_<input>_feat' (N,), '_fdstar' (N, 2, 10) float16;
'repro_*' = max |Delta_img - stored round-21 Delta| for the orig inputs (p50 / none / sr50 / p80).
Usage: python analyze/paper_extra/round22_eval.py check | main --worker w --n_workers n
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round21_eval as R  # noqa  (also registers the signed-pleasure classes)
from p2test_v11 import read_a2  # noqa
from v11core.pleasure import BIN_ANGLES  # noqa
import util  # noqa

T = R.T
OUT = os.path.join(R.RES, 'paper_extra', 'Round22')
EV21 = os.path.join(R.RES, 'paper_extra', 'Round21', 'eval')
R.EXP['sr80'] = 'paper_extra/C_signrand/s1_signrand_l1_s{s}'
CONDS = ('p50', 'p80', 'sr80', 'none', 'sr50')
X_CONDS = ('p50', 'p80')
PAIRS = tuple(range(6))
EPS = (160, 170, 180, 190, 200)
K = 10
A3 = 0.28
ORIG = ('hold', 'flip', 'zero')
XIN = ('x2_hold', 'x2_flip', 'x4_hold', 'x4_flip', 'm_hold', 'm_flip', 'b_old', 'b_new')
DREF = np.arange(-5, 6)
D0 = int(np.where(DREF == 0)[0][0])


def bins(lo, ln, p2):
    out = []
    for tg in (lo, ln):
        v = tg - p2
        ang = np.arctan2(v[:, 1], v[:, 0])
        out.append(np.argmax(np.cos(ang[:, None] - BIN_ANGLES[None]), 1))
    return out


def targets_bins():
    """-> L_old, L_new, A-2's position p2 (N, 2) and the ranges toward L_old / L_new (N,), from data_noswap."""
    g = h5py.File(os.path.join(R.DATA, 'data_noswap.h5'), 'r')['test']
    lo, ln = R.targets(g['layout'][()])
    p2 = g['other_position'][:, 0].astype(np.float64)
    bo, bn = bins(lo, ln, p2)
    return lo, ln, p2, bo, bn


def load_fviews(n=0):
    d = os.path.join(OUT, 'views')
    parts = sorted(f for f in os.listdir(d) if f.startswith('views_part'))
    acc = {k: [] for k in ('idx', 'views', 'valid')}
    for f in parts:
        h = h5py.File(os.path.join(d, f), 'r')
        assert int(h.attrs['rerender_mismatch']) == 0, f
        for k in acc:
            acc[k].append(h[k][:n] if n else h[k][()])
        if n:
            break
    V = {k: np.concatenate(v) for k, v in acc.items()}
    assert np.array_equal(V['idx'], np.arange(len(V['idx'])))
    return V


@torch.no_grad()
def ref_feats(model, views, dev, chunk=4096):
    """views (N, K, 2, 2, nd, H, W, 3) uint8 -> other_vision_encoder features (N, K, 2, 2, nd, F) torch on dev."""
    sh = views.shape[:5]
    flat = views.reshape((-1,) + views.shape[5:])
    out = []
    for c0 in range(0, flat.shape[0], chunk):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(flat[c0:c0 + chunk], dim=4))).to(dev)
        out.append(model.other_vision_encoder_module(x))
    f = torch.cat(out)
    return f.reshape(sh + (f.shape[-1],))


def p2_input(inp, model, ov, B, b_old, b_new, fm, dev):
    Q = torch.zeros(B, 11, device=dev)
    Q[:, 10] = 1.0
    if inp == 'zero':
        return Q
    if inp in ('b_old', 'b_new'):
        b = torch.from_numpy(b_old if inp == 'b_old' else b_new).long().to(dev)
        Q[torch.arange(B, device=dev), 2 + b] = A3
        return Q
    base, pv = (inp.split('_') + [None])[:2] if '_' in inp else ('x1', inp)
    sgn = 1.0 if pv == 'hold' else -1.0
    P = sgn * model.compute_pleasure(ov)
    if base != 'x1':
        k = {'x2': 2.0, 'x4': 4.0, 'm': fm}[base]
        P = (k * P).clamp(-1.0, 1.0)
    Q[:, 2:10] = P
    return Q


@torch.no_grad()
def other_flow(model, D, T0, inp, dev, bo, bn, fm, want_feat, batch=50):
    """As round21_eval.freeze_flow(path='other') with input `inp` -> preds (N, K, 16, 64, 3), feats (N, K, F) or None."""
    N = D['self_vision'].shape[0]
    out, fo = [], []
    for b0 in range(0, N, batch):
        x = R.scaled(D['self_vision'][b0:b0 + batch], dev)
        m = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(dev)
        B = x.shape[0]
        util.seed_all(0)
        model.init_state(B)
        for t in range(T0):
            model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0)
        zero = torch.zeros(B, 8, device=dev)
        ov = R.scaled(D['other_vision'][b0:b0 + batch, T0 - 1:T0], dev)[:, 0]
        P1 = zero
        Q = p2_input(inp, model, ov, B, bo[b0:b0 + B], bn[b0:b0 + B], fm, dev)
        hid = torch.ones(B, dtype=torch.bool, device=dev)
        preds, fs = [], []
        for k in range(K):
            pr = model({'self_vision': x[:, T0 + k], 'self_motion': m[:, T0 + k]}, 1.0, 1.0, p1_override=P1,
                       p2_override=Q, hidden_override=hid)['self_vision']
            preds.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
            if want_feat:
                fs.append(model.predict_feature(model.get_state()['other'][0]))
        out.append(np.stack(preds, 1))
        if want_feat:
            fo.append(torch.stack(fs, 1))
    return np.concatenate(out), (torch.cat(fo) if want_feat else None)


@torch.no_grad()
def feat_dstar(f, Fr, valid):
    """f (N, K, F), Fr (N, K, 2, 2, nd, F) torch; valid (N, 2, nd) -> d* (N, 2, K) float (NaN if unused)."""
    dist = ((f[:, :, None, None, None] - Fr) ** 2).mean(-1).amin(3)            # (N, K, 2 targets, nd)
    ok = torch.from_numpy(valid).to(f.device)                                   # (N, 2, nd)
    dist = torch.where(ok[:, None], dist, torch.full_like(dist, float('inf')))
    d = torch.from_numpy(DREF).to(f.device)[dist.argmin(-1)].double().permute(0, 2, 1).cpu().numpy()
    use = valid[:, :, D0:].all(-1)                                              # d = 0..5 valid
    d[~use] = np.nan
    return d


def evaluate_other(cond, model, DD, FV, ab, dev, fm, inputs_v1, v0_inputs):
    sav = {}
    lo, ln = R.targets(DD['noswap']['layout'])
    p2 = DD['noswap']['other_position'][:, 0].astype(np.float64)
    bo, bn = bins(lo, ln, p2)
    Fr = ref_feats(model, FV['views'], dev) if FV is not None else None
    runs = [(d, 'V1', 23, inp) for d in ('swap', 'noswap') for inp in inputs_v1] + \
           [('noswap', 'V0', 20, inp) for inp in v0_inputs]
    for dname, vname, T0, inp in runs:
        D = DD[dname]
        want = vname == 'V1' and Fr is not None
        pr, f = other_flow(model, D, T0, inp, dev, bo, bn, fm, want)
        sp = D['self_position'][:, T0:T0 + K].astype(np.float64)
        est = np.stack([read_a2(pr[:, i], sp[:, i], *ab) for i in range(K)], 1)
        S = R.other_slopes(est, p2, lo, ln)
        sav[f'{dname}_{vname}_{inp}_img'] = (S[:, 0] - S[:, 1]).astype(np.float32)
        sav[f'{dname}_{vname}_{inp}_est'] = est.astype(np.float32)
        if want:
            ds = feat_dstar(f, Fr, FV['valid'])
            sl = R.slopes_t(ds, K)
            sav[f'{dname}_{vname}_{inp}_feat'] = (sl[:, 0] - sl[:, 1]).astype(np.float32)
            sav[f'{dname}_{vname}_{inp}_fdstar'] = ds.astype(np.float16)
    return sav


def repro(sav, cond, s, e, n=None):
    if cond == 'sr80':
        return {}
    z = np.load(os.path.join(EV21, f'{cond}_s{s}_ep{e}.npz'))
    r = {}
    for d in ('swap', 'noswap'):
        for pv in ORIG:
            a, b = sav[f'{d}_V1_{pv}_img'], z[f'{d}_V1_other_{pv}'][:n]
            r[f'repro_{d}_{pv}'] = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
            r[f'repro_nan_same_{d}_{pv}'] = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    return r


def x2_factor():
    return float(json.load(open(os.path.join(OUT, 'check', 'check.json')))['inputs']['V1|X2_factor'])


def jobs():
    return [(c, s, e) for c in CONDS for s in PAIRS for e in EPS]


def main_eval(worker, n_workers):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    DD = {v: R.load_data(v) for v in ('swap', 'noswap')}
    V21 = R.load_views()
    FV = load_fviews()
    ab = R.calib_common()
    fm = x2_factor()
    os.makedirs(os.path.join(OUT, 'eval'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'eval21'), exist_ok=True)
    for i, (c, s, e) in enumerate(jobs()):
        if i % n_workers != worker:
            continue
        f = os.path.join(OUT, 'eval', f'{c}_s{s}_ep{e}.npz')
        if os.path.exists(f):
            continue
        t0 = time.time()
        if c == 'sr80':
            f21 = os.path.join(OUT, 'eval21', f'{c}_s{s}_ep{e}.npz')
            if not os.path.exists(f21):
                sav21 = R.evaluate(c, s, e, dev, DD, V21, ab)
                np.savez_compressed(f21 + '.tmp.npz', **sav21)
                os.replace(f21 + '.tmp.npz', f21)
        model, _ = T.load_v12(R.EXP[c].format(s=s), s, e, dev)
        model.eval()
        sav = evaluate_other(c, model, DD, FV, ab, dev, fm, ORIG + (XIN if c in X_CONDS else ()),
                             ('b_old', 'b_new', 'zero') if c in X_CONDS else ())
        sav.update({k: np.array(v) for k, v in repro(sav, c, s, e).items()})
        np.savez_compressed(f + '.tmp.npz', **sav)
        os.replace(f + '.tmp.npz', f)
        print(f'{c}:{s}:{e} {time.time() - t0:.0f}s {len(sav)} {time.strftime("%m-%d %H:%M")}', flush=True)


def check():
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    n = 50
    DD = {v: R.load_data(v, n) for v in ('swap', 'noswap')}
    V21 = R.load_views(n)
    FV = load_fviews(n)
    ab = R.calib_common()
    fm = x2_factor()
    out = {'n_episodes': n, 'x2_factor': fm}
    t0 = time.time()
    sav21 = R.evaluate('sr80', 0, 200, dev, DD, V21, ab)
    keys21 = set(np.load(os.path.join(EV21, 'p80_s0_ep200.npz')).files)
    out['sr80_round21_flow'] = {'time_s': time.time() - t0, 'keys_equal_round21': set(sav21) == keys21,
                                'n_keys': len(sav21),
                                'finite_frac_V1_self_hold': float(np.isfinite(sav21['swap_V1_self_hold']).mean()),
                                'finite_frac_V1_other_hold': float(np.isfinite(sav21['swap_V1_other_hold']).mean())}
    for c in ('p50', 'p80', 'sr80'):
        model, _ = T.load_v12(R.EXP[c].format(s=0), 0, 200, dev)
        model.eval()
        t0 = time.time()
        sav = evaluate_other(c, model, DD, FV, ab, dev, fm, ORIG + (XIN if c in X_CONDS else ()),
                             ('b_old', 'b_new', 'zero') if c in X_CONDS else ())
        r = {'time_s': time.time() - t0, 'n_keys': len(sav),
             'shapes': sorted({str(v.shape) for v in sav.values()}),
             'finite_frac_feat': {k: float(np.isfinite(v).mean()) for k, v in sav.items() if k.endswith('_feat')},
             'finite_frac_img': {k: float(np.isfinite(v).mean()) for k, v in sav.items() if k.endswith('_img')}}
        r.update(repro(sav, c, 0, 200, n))
        if c == 'sr80':
            for d in ('swap', 'noswap'):
                for pv in ORIG:
                    a, b = sav[f'{d}_V1_{pv}_img'], sav21[f'{d}_V1_other_{pv}']
                    r[f'same_as_round21_flow_{d}_{pv}'] = bool(np.array_equal(a, b, equal_nan=True))
        out[c] = r
    os.makedirs(os.path.join(OUT, 'check'), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'check', 'eval_check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('check', 'main'))
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--n_workers', type=int, default=1)
    a = ap.parse_args()
    if a.mode == 'check':
        check()
    else:
        main_eval(a.worker, a.n_workers)
