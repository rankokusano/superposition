"""
Paper round 21 (paper/check/paper_instructions_round21.md; docs/paper_extra_log.md Sec.45; exploratory, outside the
pre-registration; the main mode is fixed before computing). GPU (GPU1-3), no training.
Data: data/data/paper_r21_swap/data_{swap,noswap}.h5 (600 ep, 41 frames, static A-2, swap at T_S = 20);
reference views: Round21/views/views_part*.h5 (round21_views.py).

Mode `check` (Sec.1.2 check 3, existing code unchanged; first 100 episodes of data_noswap copied to
Round21/check/noswap_first100.h5): pair 0, ep200
  G3  eval_trial_v12.freeze (K = 15, hold) + localise_v11.dstar / slopes / g_of with views4 (= cf_dist_views made
      for the new data), mse stage (v12_s1_{mixed,random}_mse, as the paper's G3) and l1 stage
      (v12_s1_{mixed,random}_l1_s0), on the new data and (l1 stage) on the common test;
  G5b eval_g5_v12.g5b, l1 stage, on the new data.
  Compared with the stored per-episode values of the same models (v12/main/s1eval/s0/per_ep200.npz,
  v13/{v12ref,main}/g5/s0/g5_per_ep200.npz).

Mode `main` (Sec.2): conditions p50 (v12_s1_mixed_l1_s<s>), none (v12_s1_random_l1_s<s>), sr50
(paper_extra/E1_signrand50/s1_signrand50_l1_s<s>), p80 (v13_s1_mixed_l1_s<s>), pairs 0-5, ep160-200 (l1 stage),
loaded with eval_trial_v12.load_v12 (canonical motion-masking schedule kept, as experiment 1's freeze / G5b).
Flow (per batch of 50: util.seed_all(0), init_state), for data d in {swap, noswap} and T0 in {23 (V1), 20 (V0)}:
  t = 0..T0-1  vision shown (mask probability 0 for both processes), motion by the model's schedule;
  t = T0..T0+K-1  vision masked (1.0), motion hidden (hidden_override), and
    self runs  (K = 15): p1_override = P1 in {hold = model.compute_pleasure(A-1 view at frame T0-1), flip = -hold,
                         zero}; process-2 not overridden (zeros, as experiment 1's freeze);
    other runs (K = 10): p1_override = 0; p2_override = [0, 0, P2, 1] with P2 in {hold = model.compute_pleasure(A-2's
                         view at frame T0-1) (the model's liked colour, Red: "A-2 likes red"), flip = -hold, zero}.
Read-outs (per episode, step k = 1..K = prediction made at step T0+k-1):
  self  d*_L(k) = argmin_d mean|pred_k - view(p_T0 + d u_L)| over valid d (dstar rule: (episode, L) used only if
        d = 0..5 are valid), L in {L_old, L_new}; main = colour-agnostic (minimum over the views rendered in the
        pre-swap and the swapped layout), supplement = the data's current layout (as the existing d*);
        s_L = OLS slope of d*_L over k (localise_v11.slopes); Delta_self = s_Lold - s_Lnew.
  other A-2 position est_k from the predicted image (p2test_v11.read_a2, calibration of the common test as G5b,
        A-1 position of frame T0+k-1 as G5b); s_L = OLS slope over k of (est_k - p2) . unit(L - p2) with >= 5 finite
        points (as eval_g5_v12.g5a_metric); Delta_other = s_Lold - s_Lnew.
Writes Round21/eval/<cond>_s<s>_ep<e>.npz: per run key '<d>_<V>_<path>_<p>' -> Delta (N,) [self: main and _cur],
d* curves (N, 2, K) float16 (self, main), est (N, 10, 2) float32 (other).
Usage: python analyze/paper_extra/round21_eval.py check | main --worker w --n_workers n
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
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _PI)
import vdir_c_run  # noqa  (registers the signed-pleasure class used by the sign-randomised models)
import eval_trial_v12 as T  # noqa
import eval_g5_v12 as G  # noqa
from eval_trial_v11 import boot_ci  # noqa
from localise_v11 import dstar, slopes, g_of, DMIN_VALID  # noqa
from p2test_v11 import calibrate, read_a2  # noqa
from sim_v10 import layout as L  # noqa
import util  # noqa

RES = T.RES
DATA = os.path.join(_PI, 'data', 'data', 'paper_r21_swap')
OUT = os.path.join(RES, 'paper_extra', 'Round21')
COMMON0 = os.path.join(_PI, 'data', 'data', 'v11_s1_random', 'data.h5')
EXP = {'p50': 'v12_s1_mixed_l1_s{s}', 'none': 'v12_s1_random_l1_s{s}',
       'sr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}', 'p80': 'v13_s1_mixed_l1_s{s}'}
CONDS = ('p50', 'none', 'sr50', 'p80')
PAIRS = tuple(range(6))
EPS = (160, 170, 180, 190, 200)
FLOWS = (('V1', 23), ('V0', 20))
K_SELF, K_OTHER = 15, 10
PV = ('hold', 'flip', 'zero')


def load_data(variant, n=0):
    g = h5py.File(os.path.join(DATA, f'data_{variant}.h5'), 'r')['test']
    n = n or g['self_vision'].shape[0]
    return {k: g[k][:n] for k in ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_position',
                                  'layout')}


def load_views(n=0):
    parts = sorted(f for f in os.listdir(os.path.join(OUT, 'views')) if f.startswith('views_part'))
    acc = {k: [] for k in ('idx', 'views_lt', 'valid_lt', 'views4', 'valid4')}
    for f in parts:
        h = h5py.File(os.path.join(OUT, 'views', f), 'r')
        assert int(h.attrs['rerender_mismatch']) == 0, f
        for k in acc:
            acc[k].append(h[k][()])
    V = {k: np.concatenate(v) for k, v in acc.items()}
    assert np.array_equal(V['idx'], np.arange(len(V['idx'])))
    if n:
        V = {k: v[:n] for k, v in V.items()}
    return V


def targets(lay):
    lo = np.array([L.LANDMARKS[L.CODE_LAYOUT[int(c)]]['Red'] for c in lay], np.float64)
    ln = np.array([L.LANDMARKS[L.other_layout(L.CODE_LAYOUT[int(c)])]['Red'] for c in lay], np.float64)
    return lo, ln


def scaled(v, dev):
    return torch.from_numpy(util.scale_vision(util.transpose_vision(v))).to(dev)


@torch.no_grad()
def freeze_flow(model, D, T0, K, path, pv, dev, batch=50):
    """-> predictions (N, K, 16, 64, 3) float32 in [0, 1]."""
    N = D['self_vision'].shape[0]
    out = []
    for b0 in range(0, N, batch):
        x = scaled(D['self_vision'][b0:b0 + batch], dev)
        m = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(dev)
        B = x.shape[0]
        util.seed_all(0)
        model.init_state(B)
        for t in range(T0):
            model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0)
        zero = torch.zeros(B, 8, device=dev)
        sgn = {'hold': 1.0, 'flip': -1.0, 'zero': 0.0}[pv]
        if path == 'self':
            P1, Q = sgn * model.compute_pleasure(x[:, T0 - 1]) if pv != 'zero' else zero, None
        else:
            ov = scaled(D['other_vision'][b0:b0 + batch, T0 - 1:T0], dev)[:, 0]
            P1 = zero
            Q = torch.zeros(B, 11, device=dev)
            Q[:, 10] = 1.0
            if pv != 'zero':
                Q[:, 2:10] = sgn * model.compute_pleasure(ov)
        hid = torch.ones(B, dtype=torch.bool, device=dev)
        preds = []
        for k in range(K):
            pr = model({'self_vision': x[:, T0 + k], 'self_motion': m[:, T0 + k]}, 1.0, 1.0, p1_override=P1,
                       p2_override=Q, hidden_override=hid)['self_vision']
            preds.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
        out.append(np.stack(preds, 1))
    return np.concatenate(out)


def dstar_lt(preds, views, valid, layout_idx, chunk=25):
    """preds (N,K,H,W,C); views (N,2 targets,2 layouts,D,H,W,C) uint8; valid (N,2,2,D);
    layout_idx None = colour-agnostic (min over layouts), else (N,) layout index -> d* (N, 2, K)."""
    N, K = preds.shape[:2]
    out = np.full((N, 2, K), np.nan)
    for c0 in range(0, N, chunk):
        sl = slice(c0, c0 + chunk)
        p = preds[sl][:, :, None, None]                                      # (n,K,1,1,H,W,C)
        for tg in range(2):
            v = views[sl, tg].astype(np.float32) / 255.0                     # (n,2,D,H,W,C)
            Lv = np.abs(p - v[:, None]).mean((4, 5, 6))                      # (n,K,2,D)
            ok = valid[sl, tg]                                               # (n,2,D)
            Lv = np.where(ok[:, None], Lv, np.inf)
            if layout_idx is None:
                Lm = Lv.min(2)
            else:
                Lm = Lv[np.arange(Lv.shape[0]), :, layout_idx[sl]]           # (n,K,D)
            use = ok[:, 0, :DMIN_VALID + 1].all(1)
            d = Lm.argmin(2).astype(float)
            d[~use] = np.nan
            out[sl, tg] = d
    return out


def other_slopes(est, p2, lo, ln):
    ks = np.arange(1, est.shape[1] + 1)
    S = np.full((est.shape[0], 2), np.nan)
    for i, tg in enumerate((lo, ln)):
        u = tg - p2
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        proj = ((est - p2[:, None]) * u[:, None]).sum(-1)
        for n in range(est.shape[0]):
            ok = np.isfinite(proj[n])
            if ok.sum() >= 5:
                S[n, i] = np.polyfit(ks[ok], proj[n][ok], 1)[0]
    return S


def calib_common():
    g = h5py.File(COMMON0, 'r')['test']
    return calibrate(g['self_vision'][:, :100:5], g['self_position'][:, :100:5], g['other_position'][:, :100:5])


def evaluate(cond, s, e, dev, DD, V, ab):
    model, _ = T.load_v12(EXP[cond].format(s=s), s, e, dev)
    model.eval()
    sav = {}
    lo, ln = targets(DD['noswap']['layout'])
    p2 = DD['noswap']['other_position'][:, 0].astype(np.float64)
    for dname in ('swap', 'noswap'):
        D = DD[dname]
        cur = np.full(len(lo), 1 if dname == 'swap' else 0)
        for vname, T0 in FLOWS:
            ti = 0 if T0 == 20 else 1
            for pv in PV:
                pr = freeze_flow(model, D, T0, K_SELF, 'self', pv, dev)
                for tag, li in (('', None), ('_cur', cur)):
                    ds = dstar_lt(pr, V['views_lt'][:, ti], V['valid_lt'][:, ti], li)
                    sl = slopes(ds, K_SELF)
                    sav[f'{dname}_{vname}_self_{pv}{tag}'] = (sl[:, 0] - sl[:, 1]).astype(np.float32)
                    if tag == '':
                        sav[f'{dname}_{vname}_self_{pv}_dstar'] = ds.astype(np.float16)
                pr = freeze_flow(model, D, T0, K_OTHER, 'other', pv, dev)
                sp = D['self_position'][:, T0:T0 + K_OTHER].astype(np.float64)
                est = np.stack([read_a2(pr[:, i], sp[:, i], *ab) for i in range(K_OTHER)], 1)
                S = other_slopes(est, p2, lo, ln)
                sav[f'{dname}_{vname}_other_{pv}'] = (S[:, 0] - S[:, 1]).astype(np.float32)
                sav[f'{dname}_{vname}_other_{pv}_est'] = est.astype(np.float32)
    return sav


def jobs():
    return [(c, s, e) for c in CONDS for s in PAIRS for e in EPS]


def main_eval(worker, n_workers):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    DD = {v: load_data(v) for v in ('swap', 'noswap')}
    V = load_views()
    ab = calib_common()
    os.makedirs(os.path.join(OUT, 'eval'), exist_ok=True)
    for i, (c, s, e) in enumerate(jobs()):
        if i % n_workers != worker:
            continue
        f = os.path.join(OUT, 'eval', f'{c}_s{s}_ep{e}.npz')
        if os.path.exists(f):
            continue
        t0 = time.time()
        sav = evaluate(c, s, e, dev, DD, V, ab)
        np.savez_compressed(f + '.tmp.npz', **sav)
        os.replace(f + '.tmp.npz', f)
        print(f'{c}:{s}:{e} {time.time() - t0:.0f}s {len(sav)} {time.strftime("%m-%d %H:%M")}', flush=True)


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {'mean': float(v.mean()), 'ci95': [float(x) for x in boot_ci(v)], 'n': int(len(v))}


def check():
    """Sec.1.2 check 3."""
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    NCK = 100
    sub = os.path.join(OUT, 'check', 'noswap_first100.h5')
    if not os.path.exists(sub):
        src = h5py.File(os.path.join(DATA, 'data_noswap.h5'), 'r')['test']
        with h5py.File(sub, 'w') as f:
            for k in src:
                f.create_dataset(f'test/{k}', data=src[k][:NCK])
    V = load_views(NCK)
    R = {'n_episodes': NCK, 'time': {}}
    st = {'s1': np.load(os.path.join(RES, 'v12', 'main', 's1eval', 's0', 'per_ep200.npz')),
          'g5_50': np.load(os.path.join(RES, 'v13', 'v12ref', 'g5', 's0', 'g5_per_ep200.npz')),
          'g5_none': np.load(os.path.join(RES, 'v13', 'main', 'g5', 's0', 'g5_per_ep200.npz'))}
    for c, cn in (('mixed', 'p50'), ('random', 'none')):
        r = {}
        # G3, mse stage (the paper's G3) and l1 stage, existing freeze on the new data
        for stage, exp in (('mse', f'v12_s1_{c}_mse'), ('l1', f'v12_s1_{c}_l1_s0')):
            model, pm = T.load_v12(exp, 0, 200, dev)
            t0 = time.time()
            for nm, path, views, valid in (('new', sub, V['views4'], V['valid4']), ('common', COMMON0, None, None)):
                if nm == 'common' and stage == 'mse':
                    continue
                if nm == 'common':
                    fv = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_dist_views.h5'), 'r')
                    views, valid = fv['views'][()], fv['valid'][()]
                T.COMMON = path
                pr = T.freeze(model, pm, dev, 15, True)
                gg = g_of(slopes(dstar(pr, views, valid), 15))
                r[f'G3_{stage}_{nm}'] = summ(gg)
                r[f'G3_{stage}_{nm}_frac_read'] = float(np.isfinite(gg).mean())
            R['time'][f'G3_{c}_{stage}'] = time.time() - t0
        r['G3_mse_stored_common'] = summ(st['s1'][f'{c}__freeze_K15_hold'])
        # G5b, l1 stage, existing g5b on the new data
        model, pm = T.load_v12(f'v12_s1_{c}_l1_s0', 0, 200, dev)
        G.COMMON = sub
        t0 = time.time()
        res, main_b, _ = G.g5b(model, pm, dev)
        R['time'][f'G5b_{c}'] = time.time() - t0
        r['G5b_l1_new'] = summ(main_b)
        r['G5b_l1_new_frac_read_ctrl'] = res['frac_read_ctrl']
        r['G5b_l1_stored_common'] = summ(st['g5_50' if c == 'mixed' else 'g5_none'][f'{c}__G5b'])
        R[cn] = r
    T.COMMON = COMMON0
    G.COMMON = COMMON0

    def overlap(a, b):
        return bool(a['ci95'][0] <= b['ci95'][1] and b['ci95'][0] <= a['ci95'][1])
    crit = {}
    for cn in ('p50', 'none'):
        crit[f'G3_{cn}_overlap'] = overlap(R[cn]['G3_mse_new'], R[cn]['G3_mse_stored_common'])
        crit[f'G5b_{cn}_overlap'] = overlap(R[cn]['G5b_l1_new'], R[cn]['G5b_l1_stored_common'])
    for q, a, b in (('G3', 'G3_mse_new', 'G3_mse_stored_common'), ('G5b', 'G5b_l1_new', 'G5b_l1_stored_common')):
        crit[f'{q}_p50_minus_none_same_sign'] = bool(np.sign(R['p50'][a]['mean'] - R['none'][a]['mean']) ==
                                                     np.sign(R['p50'][b]['mean'] - R['none'][b]['mean']))
    R['criteria'] = crit
    R['pass'] = bool(all(crit.values()))
    json.dump(R, open(os.path.join(OUT, 'check', 'eval_check.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1))


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
