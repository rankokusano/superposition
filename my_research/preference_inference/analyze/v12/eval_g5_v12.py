"""
v12 gate G5 (docs/v12_experiment_log.md Sec.13, registered before computing).
G5a (other-side accumulation): stage-2 test sets (v11_s2t_green; v11_s2_pref test), stage-2 protocol
  (vision shown at t = 0 only; process-1 motion visible, flag 0); process-2 = [0, 0, P_A2(t), 1] for
  t < T0, then for K steps [0, 0, P_A2(T0), 1] ('freeze', main) / P_A2(t) ('continue') / 0 ('zero').
  A-2 position read from each predicted image; per episode the OLS slope over k of
  (est_k - p2(T0)) . u_X, u_X = unit vector from p2(T0) to colour X's reference point;
  g_other = s_liked - mean(s_others). T0 = 20, K = 15 (main) and 20.
G5b (process-2 test): common test, canonical; t = 20..29 process-2 = [0, 0, +-a e_k, 1] vs control
  [0, 0, 0, 1]; per episode mean_k [(D_k(+a, 10) - D_k(-a, 10)) . u_k].
Writes data/result/v12/<sub>/g5_<cond>_ep<e>.json (+ paired).
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
from eval_trial_v12 import COMMON, RES, batch_x, load_v12, summ  # noqa
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_s2_v11 import load_set, true_pleasure  # noqa
from eval_m1_v11 import refs  # noqa
from p2test_v11 import calibrate, read_a2  # noqa
from v11core.pleasure import BIN_UNIT  # noqa
import util  # noqa

COLORS = ('Red', 'Green', 'Blue', 'Cyan')
T0, A = 20, 0.28
BG_OFF, BG_MIN_MASS = 0.05, 0.5


def bg_weights(img):
    """Sec.22 (B1): darkness below the row median of the same image. img (..., H, W, 3) in [0, 1]."""
    mx = img.max(-1)
    return np.clip(np.median(mx, axis=-1, keepdims=True) - mx - BG_OFF, 0, None)


def calibrate_bg(sv, sp, op):
    """True views (uint8): log(distance) = alpha + beta log(M_bg), frames where A-2 is visible."""
    v = sv.astype(np.float32) / 255.0
    M = bg_weights(v).sum((-1, -2)).reshape(-1)
    vis = ((v.max(-1) < 0.1).sum((-1, -2)) >= 3).reshape(-1)
    dist = np.linalg.norm(op - sp, axis=-1).reshape(-1)
    m = vis & (M >= BG_MIN_MASS)
    beta, alpha = np.polyfit(np.log(M[m]), np.log(dist[m]), 1)
    return float(alpha), float(beta)


def read_a2_bg(pred, sp, alpha, beta):
    from v11core.pleasure import COL_ANGLES
    w = bg_weights(pred)
    col = w.sum(1)
    M = col.sum(1)
    phi = np.angle((col * np.exp(1j * COL_ANGLES)[None]).sum(1))
    dist = np.exp(alpha + beta * np.log(np.maximum(M, 1e-6)))
    est = sp + dist[:, None] * np.stack([np.cos(phi), np.sin(phi)], 1)
    est[M < BG_MIN_MASS] = np.nan
    return est


@torch.no_grad()
def g5a_est(model, d, PA2, dev, alpha, beta, K, variant, batch=50, bg=None):
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    util.seed_all(0)
    out, out_bg = [], []
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b0 + batch]).float().to(dev)
        sp = d['self_position'][b0:b0 + batch]
        B = x.shape[0]
        model.init_state(B)
        est, est_bg = [], []
        for t in range(T0 + K):
            q = torch.zeros(B, 11, device=dev)
            q[:, 10] = 1.0
            if t < T0 or variant == 'continue':
                q[:, 2:10] = torch.from_numpy(PA2[b0:b0 + batch, t]).to(dev)
            elif variant == 'freeze':
                q[:, 2:10] = torch.from_numpy(PA2[b0:b0 + batch, T0]).to(dev)
            pm = 0 if t == 0 else 1
            pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=q)['self_vision']
            if t >= T0:
                p01 = ((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                est.append(read_a2(p01, sp[:, t + 1], alpha, beta))
                if bg is not None:
                    est_bg.append(read_a2_bg(p01, sp[:, t + 1], *bg))
        out.append(np.stack(est, 1))
        if bg is not None:
            out_bg.append(np.stack(est_bg, 1))
    if bg is not None:
        return np.concatenate(out), np.concatenate(out_bg)
    return np.concatenate(out)   # (N, K, 2)


def g5a_metric(est, d):
    N, K = est.shape[:2]
    p0 = d['other_position'][:, T0]
    liked = np.where(d['a2_liked'] == 1, 'Red', 'Green')
    ks = np.arange(1, K + 1)
    S = np.full((N, 4), np.nan)
    for i, X in enumerate(COLORS):
        u = refs(d['layout'], X)[:, 0] - p0
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        proj = ((est - p0[:, None]) * u[:, None]).sum(-1)    # (N, K)
        for n in range(N):
            ok = np.isfinite(proj[n])
            if ok.sum() >= 5:
                S[n, i] = np.polyfit(ks[ok], proj[n][ok], 1)[0]
    li = np.array([COLORS.index(l) for l in liked])
    s_l = S[np.arange(N), li]
    others = np.stack([np.delete(S[n], li[n]) for n in range(N)])
    # per-step curves for the figure (Sec.5.2 of the instructions): mean projection on the liked
    # direction and mean over the three other directions, relative to p2(T0)
    P = np.stack([((est - p0[:, None]) * (lambda u: u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9))(
        refs(d['layout'], X)[:, 0] - p0)[:, None]).sum(-1) for X in COLORS], 1)   # (N, 4, K)
    curve_l = np.nanmean(P[np.arange(N), li], 0)
    curve_o = np.nanmean(np.stack([np.nanmean(np.delete(P[n], li[n], axis=0), 0) for n in range(N)]), 0)
    g5a_metric.curves = (curve_l.round(4).tolist(), curve_o.round(4).tolist())
    return s_l - np.nanmean(others, 1), s_l, S


@torch.no_grad()
def g5b(model, p_mask, dev, K=10, batch=50):
    g = h5py.File(COMMON, 'r')['test']
    sp_all, op_all = g['self_position'][()], g['other_position'][()]
    alpha, beta = calibrate(g['self_vision'][:, :100:5], sp_all[:, :100:5], op_all[:, :100:5])
    bg = calibrate_bg(g['self_vision'][:, :100:5], sp_all[:, :100:5], op_all[:, :100:5])
    N = sp_all.shape[0]
    est, est_bg = {}, {}
    for b0 in range(0, N, batch):
        x, m = batch_x(g, b0, batch, dev)
        B = x.shape[0]
        sp = sp_all[b0:b0 + batch, T0:T0 + K]
        keys = [('ctrl', None)] + [(f'{sg}{k}', (sgn, k)) for sg, sgn in (('p', 1.0), ('m', -1.0)) for k in range(8)]
        for key, spec in keys:
            Q = torch.zeros(B, 11, device=dev)
            Q[:, 10] = 1.0
            if spec is not None:
                Q[:, 2 + spec[1]] = spec[0] * A
            util.seed_all(0)
            model.init_state(B)
            prs = []
            for t in range(T0 + K):
                pm = 0 if t == 0 else p_mask
                pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm,
                           p2_override=Q if t >= T0 else None)['self_vision']
                if t >= T0:
                    prs.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
            est.setdefault(key, []).append(np.stack([read_a2(prs[i], sp[:, i], alpha, beta) for i in range(K)], 1))
            est_bg.setdefault(key, []).append(np.stack([read_a2_bg(prs[i], sp[:, i], *bg) for i in range(K)], 1))
    est = {k: np.concatenate(v) for k, v in est.items()}
    est_bg = {k: np.concatenate(v) for k, v in est_bg.items()}
    res_bg, main_bg = g5b_summary(est_bg)
    res_main, main = g5b_summary(est)
    res_main['bg'] = res_bg
    res_main['bg']['calib'] = list(bg)
    return res_main, main, main_bg


def g5b_summary(est):
    Dp = np.stack([est[f'p{k}'][:, -1] - est['ctrl'][:, -1] for k in range(8)], 1)   # (N, 8, 2)
    Dm = np.stack([est[f'm{k}'][:, -1] - est['ctrl'][:, -1] for k in range(8)], 1)
    own_p = (Dp * BIN_UNIT[None]).sum(-1)
    own_m = (Dm * BIN_UNIT[None]).sum(-1)
    main = np.nanmean(own_p - own_m, 1)
    return {'main_pm_own': summ(main), 'own_p': summ(np.nanmean(own_p, 1)), 'own_m': summ(np.nanmean(own_m, 1)),
            'pm_own_bins0167': float(np.nanmean((own_p - own_m)[:, [0, 1, 6, 7]])),
            'pm_own_bins2to5': float(np.nanmean((own_p - own_m)[:, 2:6])),
            'pm_own_per_bin': np.nanmean(own_p - own_m, 0).round(4).tolist(),
            'own_p_per_bin': np.nanmean(own_p, 0).round(4).tolist(), 'own_m_per_bin': np.nanmean(own_m, 0).round(4).tolist(),
            'mean_vec_p': np.nanmean(Dp, 0).round(4).tolist(), 'mean_vec_m': np.nanmean(Dm, 0).round(4).tolist(),
            'frac_read_ctrl': float(np.isfinite(est['ctrl'][..., 0]).mean())}, main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', default='v12_trial_pc_{}_mse')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epoch', type=int, default=150)
    ap.add_argument('--sub', default='trial/g5_pc')
    args = ap.parse_args()
    out = os.path.join(RES, 'v12', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in ('green', 'pref')}
    PA2 = {k: true_pleasure(k, D[k]) for k in D}
    g = h5py.File(D['green']['path'], 'r')['test']
    alpha, beta = calibrate(g['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    bgc = calibrate_bg(g['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    per = {}
    for c in ('random', 'mixed'):
        model, p_mask = load_v12(args.exp_fmt.format(c), args.seed, args.epoch, dev)
        r = {'exp': args.exp_fmt.format(c), 'epoch': args.epoch, 'G5a': {}}
        model.motion_masking = False     # stage-2 protocol for G5a
        for k in D:
            for K in (15, 20):
                for variant in ('freeze', 'continue', 'zero'):
                    if K == 20 and variant != 'freeze':
                        continue
                    est, est_bg = g5a_est(model, D[k], PA2[k], dev, alpha, beta, K, variant, bg=bgc)
                    gb, slb, Sb = g5a_metric(est_bg, D[k])
                    cb = g5a_metric.curves
                    r.setdefault('G5a_bg', {})[f'{k}_K{K}_{variant}'] = {'g_other': summ(gb), 's_liked': summ(slb),
                                                                       'curve_liked': cb[0], 'curve_others': cb[1],
                                                                       'S_mean_RGBC': np.nanmean(Sb, 0).round(4).tolist(),
                                                                       'frac_read': float(np.isfinite(est_bg[..., 0]).mean())}
                    per[(c, f'G5a_bg_{k}_K{K}_{variant}')] = gb
                    gg, sl, S = g5a_metric(est, D[k])
                    r['G5a'][f'{k}_K{K}_{variant}'] = {'g_other': summ(gg), 's_liked': summ(sl),
                                                       'curve_liked': g5a_metric.curves[0], 'curve_others': g5a_metric.curves[1],
                                                       'S_mean_RGBC': np.nanmean(S, 0).round(4).tolist(),
                                                       'frac_read': float(np.isfinite(est[..., 0]).mean())}
                    per[(c, f'G5a_{k}_K{K}_{variant}')] = gg
        model.motion_masking = True      # canonical for G5b (as the stage-1 process-2 test)
        r['G5b'], main_b, main_b_bg = g5b(model, p_mask, dev)
        per[(c, 'G5b')] = main_b
        per[(c, 'G5b_bg')] = main_b_bg
        r['calib_bg_G5a'] = list(bgc)
        with open(os.path.join(out, f'g5_{c}_ep{args.epoch}.json'), 'w') as f:
            json.dump(r, f, indent=1)
        print(c, json.dumps({'G5a_freeze_K15': {k: r['G5a'][f'{k}_K15_freeze']['g_other']['mean'] for k in D},
                             'G5b': r['G5b']['main_pm_own']['mean']}), flush=True)
    keys = sorted({k for (_, k) in per})
    paired = {k: summ(per[('mixed', k)] - per[('random', k)]) for k in keys}
    with open(os.path.join(out, f'g5_paired_ep{args.epoch}.json'), 'w') as f:
        json.dump(paired, f, indent=1)
    print('paired', json.dumps({k: round(v['mean'], 4) for k, v in paired.items()}), flush=True)


if __name__ == '__main__':
    main()
