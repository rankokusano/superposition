"""
Paper round 23 (paper/check/paper_instructions_round23.md; docs/paper_extra_log.md Sec.47; exploratory, outside the
pre-registration; the main modes are fixed before computing). GPU (GPU1-3), no training.
The other-vision encoder (Encoder-2) is the same frozen reference encoder in every predictor used (checked in Sec.47.1),
so reference features and true features are computed once.

Mode `encode` (GPU1, data preparation): reference features F = other_vision_encoder(view) of Round23/views_r20
  (600, 27 frames, 2 targets, 2 layouts, 31 d) and views_dir (600, 10 frames, 3 classes, 11 d)
  -> Round23/ref/{feats,valid}_{r20,dir}.npy.
Mode `true` (Sec.2.1 true read-outs, model independent): true features = other_vision_encoder(frame f) and true image
  positions = read_a2(frame f, A-1 at frame f) for frames f = 54..80 of N / O / S / R -> Round23/ref/true_r20.npz.
Mode `main` --part r20 | dir:
  r20 (Sec.2.1)  round 20 data p_rule 0.5, flow V2 of round 20 (round20_eval.run: model.eval, seed_all(0) per run,
        batch 50, A-1's true motion, all steps shown, eval_inverse_v14.load = motion masking off; step t takes frame t
        and predicts frame t+1), steps 53..79 (round 20's evaluation interval). Inputs Z ([0,0,0x8,1], main) and T
        (A-2's true pleasure, round20_eval.a2_pleasure) for p50 (v12:mixed), p80 (v13:mixed), none (v12:random),
        pairs 0-5; G (G's own output) for v12_s2_pref_{mixed,random}_s0-3 (descriptive). ep160-200.
        Read-outs per run: feat = predict_feature(os) at step t vs F(frame t+1) (colour-agnostic minimum over the
        two layouts, argmin over valid d, (episode, L) used only if d = 0..5 are valid); img = read_a2(pred image of
        step t, A-1 at frame t) (the round-21 convention), s_L = OLS slope over the 27 steps of (est - p50) .
        unit(L - p50) (>= 5 finite). Delta = s_X - s_Y (> 0 = toward X).
  dir (Sec.2.2)  round 21 data_noswap, flow of rounds 21/22 (round22_eval.other_flow: eval_trial_v12.load_v12 with
        the canonical motion schedule, V1 = frames 0..22 shown, freeze from T0 = 23, K = 10, p1 = 0,
        p2 = [0, 0, P2, 1]); P2 = zero, or +0.28 in the range of class E / B / R (Round23/check/ranges.npz; zero for
        episodes without that class, marked NaN). p50, p80, none, sr50, pairs 0-5, ep160-200.
        Read-outs: along u = BIN_UNIT[b] of the class: feat d*(k) = argmin over valid d = -5..5 of mean (f_k - F)^2,
        used only if d = 0..5 are valid; img (est_k - p2) . u (A-1 at frame T0+k-1); slope over k = 1..10.
Keys: r20  '<inp>_<v>_feat' / '_img' (N,) Delta, '_fd' (N, 2, 27) float16 d*;  dir  '<inp>_<c>_feat' / '_img' (N,)
slope along class c's direction ('zero_<c>_*' = zero input read along c).
Mode `check` (first 50 episodes, pair 0, ep200): r20 L1 of the V2 / Z / T runs vs the stored round-20 values
(Round20/eval/l1_p50_s0_ep200.npz, must be bit-identical); dir zero-input img Delta along L_old / L_new vs the
stored round-22 values (Round22/eval/p50_s0_ep200.npz 'noswap_V1_zero_img'); shapes and times only.
Usage: python analyze/paper_extra/round23_eval.py encode | true | check | main --part r20|dir --worker w --n_workers n
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
import round22_eval as R22  # noqa  (round21_eval as R22.R; signed classes registered)
import round20_eval as R20  # noqa
from p2test_v11 import read_a2  # noqa
from v11core.pleasure import BIN_UNIT  # noqa
from sim_v10 import layout as L  # noqa
import util  # noqa

R21 = R22.R
T = R21.T
EI = R20.EI
RES = R21.RES
OUT = os.path.join(RES, 'paper_extra', 'Round23')
STEPS = np.arange(53, 80)            # steps t; predicted / reference frames t + 1 = 54..80
R20_D = np.arange(-10, 21)
DIR_D = np.arange(-5, 6)
CLASSES = ('E', 'B', 'R')
A3 = 0.28
PAIRS = tuple(range(6))
EPS = (160, 170, 180, 190, 200)
R20_SPEC = {'p50': 'v12:mixed', 'p80': 'v13:mixed', 'none': 'v12:random'}
G_EXP = {'Gp50': 'v12_s2_pref_mixed', 'Gnone': 'v12_s2_pref_random'}
DIR_EXP = {'p50': 'v12_s1_mixed_l1_s{s}', 'p80': 'v13_s1_mixed_l1_s{s}', 'none': 'v12_s1_random_l1_s{s}',
           'sr50': 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}'}


def r20_geom(n=0):
    g = h5py.File(R20.dpath(50, 'N'), 'r')['test']
    N = n or g['layout'].shape[0]
    lay, lk = g['layout'][:N], g['a2_liked'][:N]
    p50 = g['other_position'][:N, 50].astype(np.float64)
    col = np.where(lk == 1, 'Red', 'Green')
    X = np.array([L.LANDMARKS[L.CODE_LAYOUT[int(a)]][c] for a, c in zip(lay, col)], np.float64)
    Y = np.array([L.LANDMARKS[L.other_layout(L.CODE_LAYOUT[int(a)])][c] for a, c in zip(lay, col)], np.float64)
    U = np.stack([(t - p50) / np.linalg.norm(t - p50, axis=1, keepdims=True) for t in (X, Y)], 1)   # (N, 2, 2)
    return p50, U


def load_views(mode, n=0):
    d = os.path.join(OUT, f'views_{mode}')
    parts = sorted((f for f in os.listdir(d) if f.startswith('views_part')),
                   key=lambda s: int(s[len('views_part'):].split('of')[0]))
    acc = {k: [] for k in ('idx', 'views', 'valid')}
    for f in parts:
        h = h5py.File(os.path.join(d, f), 'r')
        assert int(h.attrs['rerender_mismatch']) == 0, f
        for k in acc:
            acc[k].append(h[k][()])
    V = {k: np.concatenate(v) for k, v in acc.items()}
    assert np.array_equal(V['idx'], np.arange(len(V['idx'])))
    return {k: v[:n] for k, v in V.items()} if n else V


@torch.no_grad()
def encode_imgs(model, imgs, dev, chunk=8192):
    """imgs uint8 (..., H, W, 3) -> (..., 64) float32 numpy."""
    sh = imgs.shape[:-3]
    flat = imgs.reshape((-1,) + imgs.shape[-3:])
    out = []
    for c0 in range(0, flat.shape[0], chunk):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(flat[c0:c0 + chunk], dim=4))).to(dev)
        out.append(model.other_vision_encoder_module(x).cpu().numpy())
    return np.concatenate(out).reshape(sh + (-1,))


def ref_model(dev):
    m, _ = T.load_v12('v12_s1_mixed_l1_s0', 0, 200, dev)
    return m.eval()


# ---------------------------------------------------------------- read-outs
def feat_dstar(f, F, valid, D, chunk=100):
    """f (N, K, 64); F (N, K, T targets, [layouts,] nd, 64); valid (N, T, nd) -> d* (N, T, K) (NaN if unused).
    Layout axis, when present, is minimised over (colour-agnostic)."""
    N, K = f.shape[:2]
    nt = F.shape[2]
    out = np.full((N, nt, K), np.nan)
    d0 = int(np.where(D == 0)[0][0])
    use = valid[:, :, d0:d0 + 6].all(-1)                                       # d = 0..5 valid
    for c0 in range(0, N, chunk):
        sl = slice(c0, c0 + chunk)
        ft = torch.from_numpy(f[sl]).cuda()
        Ft = torch.from_numpy(F[sl]).cuda()
        if Ft.dim() == 6:
            dist = ((ft[:, :, None, None, None] - Ft) ** 2).mean(-1).amin(3)    # (n, K, T, nd)
        else:
            dist = ((ft[:, :, None, None] - Ft) ** 2).mean(-1)                  # (n, K, T, nd)
        ok = torch.from_numpy(valid[sl]).cuda()[:, None]
        dist = torch.where(ok, dist, torch.full_like(dist, float('inf')))
        d = torch.from_numpy(D).cuda()[dist.argmin(-1)].double().permute(0, 2, 1).cpu().numpy()
        out[sl] = d
    out[~use] = np.nan
    return out


def slope_rows(Y, min_finite=None):
    """Y (N, K) -> OLS slope over k = 1..K per row; NaN if any NaN (min_finite None) or < min_finite finite."""
    ks = np.arange(1, Y.shape[1] + 1)
    s = np.full(Y.shape[0], np.nan)
    for n in range(Y.shape[0]):
        ok = np.isfinite(Y[n])
        if (min_finite is None and ok.all()) or (min_finite is not None and ok.sum() >= min_finite):
            s[n] = np.polyfit(ks[ok], Y[n][ok], 1)[0]
    return s


def img_delta(est, p50, U):
    """est (N, K, 2) -> Delta = s_X - s_Y from the projections on unit(L - p50) (>= 5 finite, as round 21)."""
    s = [slope_rows(((est - p50[:, None]) * U[:, None, i]).sum(-1), 5) for i in range(2)]
    return s[0] - s[1]


def feat_delta(ds):
    s = [slope_rows(ds[:, i]) for i in range(2)]
    return s[0] - s[1]


# ---------------------------------------------------------------- encode
def encode():
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    os.makedirs(os.path.join(OUT, 'ref'), exist_ok=True)
    m = ref_model(dev)
    t0 = time.time()
    for mode in ('r20', 'dir'):
        V = load_views(mode)
        np.save(os.path.join(OUT, 'ref', f'feats_{mode}.npy'), encode_imgs(m, V['views'], dev))
        np.save(os.path.join(OUT, 'ref', f'valid_{mode}.npy'), V['valid'])
        del V
    print(f'encode done {time.time() - t0:.0f}s', flush=True)


def true_r20():
    """Sec.2.1 true read-outs (model independent): true features / true image positions of frames 54..80."""
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    m = ref_model(dev)
    t0 = time.time()
    p50, U = r20_geom()
    F = np.load(os.path.join(OUT, 'ref', 'feats_r20.npy'), mmap_mode='r')
    valid = np.load(os.path.join(OUT, 'ref', 'valid_r20.npy'))
    ab = R21.calib_common()
    sav = {}
    for v in R20.VARS:
        g = h5py.File(R20.dpath(50, v), 'r')['test']
        fr = g['self_vision'][:, STEPS + 1]
        sp = g['self_position'][:, STEPS + 1].astype(np.float64)
        ft = encode_imgs(m, fr, dev)
        ds = feat_dstar(ft, np.asarray(F), valid, R20_D)
        sav[f'true_{v}_feat'] = feat_delta(ds).astype(np.float32)
        sav[f'true_{v}_fd'] = ds.astype(np.float16)
        est = np.stack([read_a2(fr[:, i].astype(np.float32) / 255.0, sp[:, i], *ab) for i in range(len(STEPS))], 1)
        sav[f'true_{v}_img'] = img_delta(est, p50, U).astype(np.float32)
        sav[f'true_{v}_est'] = est.astype(np.float32)
    np.savez_compressed(os.path.join(OUT, 'ref', 'true_r20.npz'), **sav)
    print(f'true done {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- Sec.2.1 flow
@torch.no_grad()
def run_v2(model, sv, sm, P8, dev, batch=50):
    """round20_eval.run with all steps shown; -> preds (N, 27, H, W, 3) at STEPS, feats (N, 27, 64), L1 (N, 80)."""
    util.seed_all(0)
    N = sv.shape[0]
    PR, FS, L1 = [], [], []
    for b0 in range(0, N, batch):
        b1 = min(b0 + batch, N)
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv[b0:b1]))).to(dev)
        m = torch.from_numpy(sm[b0:b1]).float().to(dev)
        B = x.shape[0]
        model.init_state(B)
        pr, fs, l1 = [], [], []
        for t in range(R20.NSTEP):
            if P8 is None:
                p2 = None
            else:
                c = torch.from_numpy(P8[b0:b1, t]).float().to(dev)
                z = torch.zeros(B, 1, device=dev)
                p2 = torch.cat([z, z, c, torch.ones_like(z)], 1)
            o = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0, p2_override=p2)
            l1.append(torch.abs(o['self_vision'] - x[:, t + 1]).mean((1, 2, 3)).cpu().numpy())
            if t >= STEPS[0]:
                pr.append(((o['self_vision'] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
                fs.append(model.predict_feature(model.get_state()['other'][0]).cpu().numpy())
        PR.append(np.stack(pr, 1)); FS.append(np.stack(fs, 1)); L1.append(np.stack(l1, 1))
    return np.concatenate(PR), np.concatenate(FS), np.concatenate(L1)


def load_r20(n=0):
    return R20.load_data(50, n)      # 'pos' = A-1's position (N, 81, 2)


def eval_r20(cond, s, e, dev, D, F, valid, ab, p50, U):
    if cond in G_EXP:
        model, _ = T.load_v12(f'{G_EXP[cond]}_s{s}', s, e, dev)
        model.eval()
        inputs = ('G',)
    else:
        model, width = EI.load(f'{R20_SPEC[cond]}:{s}', e, dev)
        assert width == 11
        inputs = ('Z', 'T')
    sav = {}
    for inp in inputs:
        for v in R20.VARS:
            P8 = None if inp == 'G' else (D[v]['PT'] if inp == 'T' else np.zeros_like(D[v]['PT']))
            pr, fs, l1 = run_v2(model, D[v]['sv'], D[v]['sm'], P8, dev)
            ds = feat_dstar(fs, F, valid, R20_D)
            sav[f'{inp}_{v}_feat'] = feat_delta(ds).astype(np.float32)
            sav[f'{inp}_{v}_fd'] = ds.astype(np.float16)
            sp = D[v]['pos'][:, STEPS]
            est = np.stack([read_a2(pr[:, i], sp[:, i], *ab) for i in range(len(STEPS))], 1)
            sav[f'{inp}_{v}_img'] = img_delta(est, p50, U).astype(np.float32)
            sav[f'{inp}_{v}_L1'] = l1.astype(np.float32)
    return sav


# ---------------------------------------------------------------- Sec.2.2 flow
def dir_q(model, ov, B, b, dev):
    Q = torch.zeros(B, 11, device=dev)
    Q[:, 10] = 1.0
    if b is not None:
        bb = torch.from_numpy(b).long().to(dev)
        has = bb >= 0
        Q[torch.arange(B, device=dev)[has], 2 + bb[has]] = A3
    return Q


@torch.no_grad()
def other_flow_dir(model, D, b_all, dev, batch=50, T0=23, K=10):
    """round22_eval.other_flow with P2 = +0.28 in range b (or zero if b_all is None)."""
    N = D['self_vision'].shape[0]
    out, fo = [], []
    for b0 in range(0, N, batch):
        x = R21.scaled(D['self_vision'][b0:b0 + batch], dev)
        m = torch.from_numpy(D['self_motion'][b0:b0 + batch]).float().to(dev)
        B = x.shape[0]
        util.seed_all(0)
        model.init_state(B)
        for t in range(T0):
            model({'self_vision': x[:, t], 'self_motion': m[:, t]}, 0, 0)
        ov = R21.scaled(D['other_vision'][b0:b0 + batch, T0 - 1:T0], dev)[:, 0]
        P1 = torch.zeros(B, 8, device=dev)
        Q = dir_q(model, ov, B, None if b_all is None else b_all[b0:b0 + B], dev)
        hid = torch.ones(B, dtype=torch.bool, device=dev)
        preds, fs = [], []
        for k in range(K):
            pr = model({'self_vision': x[:, T0 + k], 'self_motion': m[:, T0 + k]}, 1.0, 1.0, p1_override=P1,
                       p2_override=Q, hidden_override=hid)['self_vision']
            preds.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
            fs.append(model.predict_feature(model.get_state()['other'][0]).cpu().numpy())
        out.append(np.stack(preds, 1))
        fo.append(np.stack(fs, 1))
    return np.concatenate(out), np.concatenate(fo)


def eval_dir(cond, s, e, dev, D, F, valid, rg, ab, n=None):
    model, _ = T.load_v12(DIR_EXP[cond].format(s=s), s, e, dev)
    model.eval()
    p2 = D['other_position'][:, 0].astype(np.float64)
    sp = D['self_position'][:, 23:33].astype(np.float64)
    sav = {}
    runs = {'zero': None, **{c: rg[c] for c in CLASSES}}
    for inp, b in runs.items():
        pr, fs = other_flow_dir(model, D, b, dev)
        est = np.stack([read_a2(pr[:, i], sp[:, i], *ab) for i in range(10)], 1)
        ds = feat_dstar(fs, F, valid, DIR_D)                                    # (N, 3 classes, 10)
        sav[f'{inp}_est'] = est.astype(np.float32)
        for ci, c in enumerate(CLASSES):
            has = rg[c] >= 0
            u = BIN_UNIT[np.maximum(rg[c], 0)]
            si = slope_rows(((est - p2[:, None]) * u[:, None]).sum(-1), 5)
            sf = slope_rows(ds[:, ci])
            si[~has], sf[~has] = np.nan, np.nan
            sav[f'{inp}_{c}_img'] = si.astype(np.float32)
            sav[f'{inp}_{c}_feat'] = sf.astype(np.float32)
            sav[f'{inp}_{c}_fd'] = ds[:, ci].astype(np.float16)
    return sav


def load_dir(n=0):
    D = R21.load_data('noswap', n)
    rg = np.load(os.path.join(OUT, 'check', 'ranges.npz'))
    rg = {c: rg[c][:n] if n else rg[c] for c in CLASSES}
    return D, rg


def jobs(part):
    if part == 'r20':
        return [(c, s, e) for c in R20_SPEC for s in PAIRS for e in EPS] + \
               [(c, s, e) for c in G_EXP for s in range(4) for e in EPS]
    return [(c, s, e) for c in DIR_EXP for s in PAIRS for e in EPS]


def main_eval(part, worker, n_workers):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    ab = R21.calib_common()
    F = np.load(os.path.join(OUT, 'ref', f'feats_{part}.npy'))
    valid = np.load(os.path.join(OUT, 'ref', f'valid_{part}.npy'))
    if part == 'r20':
        D = load_r20()
        p50, U = r20_geom()
    else:
        D, rg = load_dir()
    od = os.path.join(OUT, f'eval_{part}')
    os.makedirs(od, exist_ok=True)
    for i, (c, s, e) in enumerate(jobs(part)):
        if i % n_workers != worker:
            continue
        f = os.path.join(od, f'{c}_s{s}_ep{e}.npz')
        if os.path.exists(f):
            continue
        t0 = time.time()
        sav = eval_r20(c, s, e, dev, D, F, valid, ab, p50, U) if part == 'r20' else eval_dir(c, s, e, dev, D, F, valid, rg, ab)
        np.savez_compressed(f + '.tmp.npz', **sav)
        os.replace(f + '.tmp.npz', f)
        torch.cuda.empty_cache()
        print(f'{part}:{c}:{s}:{e} {time.time() - t0:.0f}s {len(sav)} {time.strftime("%m-%d %H:%M")}', flush=True)


def check():
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    n = 50
    ab = R21.calib_common()
    out = {'n_episodes': n}
    # r20: L1 reproduction and shapes
    D = load_r20(n)
    p50, U = r20_geom(n)
    F = np.load(os.path.join(OUT, 'ref', 'feats_r20.npy'), mmap_mode='r')[:n]
    valid = np.load(os.path.join(OUT, 'ref', 'valid_r20.npy'))[:n]
    t0 = time.time()
    sav = eval_r20('p50', 0, 200, dev, D, np.asarray(F), valid, ab, p50, U)
    z = np.load(os.path.join(RES, 'paper_extra', 'Round20', 'eval', 'l1_p50_s0_ep200.npz'))
    out['r20'] = {'time_s': time.time() - t0, 'n_keys': len(sav),
                  'L1_max_abs_diff_vs_round20': {f'{inp}_{v}': float(np.abs(sav[f'{inp}_{v}_L1'] - z[f'L1_p50_V2_{inp}_{v}'][:n]).max())
                                                for inp in ('Z', 'T') for v in R20.VARS},
                  'finite_frac': {k: float(np.isfinite(v).mean()) for k, v in sav.items() if k.endswith(('_feat', '_img'))}}
    t0 = time.time()
    sg = eval_r20('Gp50', 0, 200, dev, D, np.asarray(F), valid, ab, p50, U)
    zg = np.load(os.path.join(RES, 'paper_extra', 'Round20', 'eval', 'G_p50_s0_ep200.npz'))
    out['r20_G'] = {'time_s': time.time() - t0,
                    'L1_max_abs_diff_vs_round20': {v: float(np.abs(sg[f'G_{v}_L1'] - zg[f'L1_p50_V2_G_{v}'][:n]).max()) for v in R20.VARS}}
    # dir: zero-input reproduction of round 22 (Delta along L_old / L_new from the same est)
    Dd, rg = load_dir(n)
    Fd = np.load(os.path.join(OUT, 'ref', 'feats_dir.npy'), mmap_mode='r')[:n]
    vd = np.load(os.path.join(OUT, 'ref', 'valid_dir.npy'))[:n]
    t0 = time.time()
    sd = eval_dir('p50', 0, 200, dev, Dd, np.asarray(Fd), vd, rg, ab)
    lo, ln = R21.targets(Dd['layout'])
    S = R21.other_slopes(sd['zero_est'].astype(np.float64), Dd['other_position'][:, 0].astype(np.float64), lo, ln)
    z22 = np.load(os.path.join(RES, 'paper_extra', 'Round22', 'eval', 'p50_s0_ep200.npz'))['noswap_V1_zero_img'][:n]
    out['dir'] = {'time_s': time.time() - t0, 'n_keys': len(sd),
                  'zero_img_delta_max_abs_diff_vs_round22': float(np.nanmax(np.abs((S[:, 0] - S[:, 1]).astype(np.float32) - z22))),
                  'finite_frac': {k: float(np.isfinite(v).mean()) for k, v in sd.items() if k.endswith(('_feat', '_img'))}}
    json.dump(out, open(os.path.join(OUT, 'check', 'eval_check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('encode', 'true', 'check', 'main'))
    ap.add_argument('--part', choices=('r20', 'dir'), default='r20')
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--n_workers', type=int, default=1)
    a = ap.parse_args()
    if a.mode == 'encode':
        encode()
    elif a.mode == 'true':
        true_r20()
    elif a.mode == 'check':
        check()
    else:
        main_eval(a.part, a.worker, a.n_workers)
